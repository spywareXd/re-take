//! The seam between the editor and whatever finds retakes: the retake service.
//!
//! The editor never knows how detection works. It makes three kinds of request and gets documents
//! back as text in the retake contract (`docs/retake-contract.md`), which it reads and checks
//! exactly as it would a file from disk. A real pipeline (a local service), a file made by hand and
//! the [`FakeService`] below all go through the same checks.
//!
//! 1. **analyze**: the file to look at -> candidates (sentences and pairs to ask the user about);
//! 2. **resolve**: the user's answers -> groups of sentences (after the trained cross-encoder);
//! 3. **adjudicate** (optional): groups -> which sentences an LLM would keep. Only the Ask AI button
//!    uses it, and only when the service says it can ([`Capabilities`]).
//!
//! The pipeline never says what to keep and never calls an LLM by itself.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use filmcraft_project::TrackKind;
use filmcraft_time::Tick;

use crate::editor::{Editor, ItemId};
use crate::error::{EditorError, Result};
use crate::refine::CutMode;
use crate::review::{self, Capabilities, DocV2, KeeperInfo, LabelsRequest, PairClass, ReviewGroup, Verdict};

/// One file to analyse.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisRequest {
    pub item: ItemId,
    /// The project's name for the file.
    pub name: String,
    /// Where the file is on disk.
    pub path: String,
    /// The file's length.
    pub duration: Tick,
    /// Hash of the file (see `review::fingerprint_file`), when it could be read.
    pub fingerprint: Option<String>,
    /// The parts of the file that are on the timeline, in file time (merged, sorted). A service may
    /// skip the rest.
    pub used: Vec<(Tick, Tick)>,
    /// The transcription model and silence gap chosen for this analysis (None = the project's / the service's default).
    pub settings: crate::options::AnalysisSettings,
}

/// The user's answers, to be turned into groups.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolveRequest {
    pub item: ItemId,
    pub labels: LabelsRequest,
    /// The candidates document the answers are about, as received. A service that kept its own copy
    /// ignores it; one that restarted can use it.
    pub candidates: Option<String>,
}

/// Groups to ask an LLM about.
#[derive(Debug, Clone, PartialEq)]
pub struct AdjudicateRequest {
    pub item: ItemId,
    pub job: String,
    pub groups: Vec<ReviewGroup>,
}

/// Exact cut ranges for sentences the user removes.
#[derive(Debug, Clone, PartialEq)]
pub struct RefineRequest {
    pub item: ItemId,
    pub job: String,
    pub mode: CutMode,
    /// The removed sentences to refine (see [`crate::ids_to_refine`]).
    pub remove: Vec<String>,
    /// Where those sentences are now (id, start, end), for a service that has no transcript of its own.
    pub spans: Vec<(String, Tick, Tick)>,
}

/// How a request is going.
#[derive(Debug, Clone, PartialEq)]
pub enum JobState {
    Running {
        fraction: f32,
        status: String,
    },
    /// Finished: the document (or reply), as text.
    Done(String),
    Failed(String),
    Cancelled,
}

/// A running request. Polled from the interface once per frame, so it must not block.
pub trait AnalysisJob: Send {
    fn poll(&mut self) -> JobState;
    fn cancel(&mut self);

    /// The stage events that arrived since the last call (see [`crate::progress`]). A service that reports none gives an empty list.
    fn events(&mut self) -> Vec<crate::progress::StageEvent> {
        Vec::new()
    }
}

/// Something that can find retakes.
pub trait RetakeService: Send + Sync {
    /// A short name for messages.
    fn name(&self) -> String;

    /// What this service can do beyond finding groups. The Ask AI button appears only when
    /// `adjudicate` is true.
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    /// The settings the service offers (transcription models, silence-gap range). Empty for a service that offers none.
    fn options(&self) -> crate::options::ServiceOptions {
        crate::options::ServiceOptions::default()
    }

    /// The project folder the editor has open (None: no folder yet, or an old project). The service keeps what it computes for the project in it.
    fn set_project(&self, _folder: Option<std::path::PathBuf>) {}

    /// Export the cut video / audio / subtitles / cut list / transcript into the project folder. The job's events carry the `export_render` stage (with the ffmpeg
    /// percent), its result is `{"path", "relative", "kind", "plan", "notes", "verify"}`. None when the service cannot.
    fn export(&self, _request: crate::options::ExportRequest) -> Option<Box<dyn AnalysisJob>> {
        None
    }

    /// A request about the user's export presets or the export options (see [`crate::UserCall`]); the result is `{"status", "body"}`. None when the service cannot.
    fn user_call(&self, _call: crate::options::UserCall) -> Option<Box<dyn AnalysisJob>> {
        None
    }

    /// A request about the project's stored settings and results (see [`crate::ProjectCall`]). None when the service cannot.
    fn project_call(&self, _call: crate::options::ProjectCall) -> Option<Box<dyn AnalysisJob>> {
        None
    }

    /// Look at a file. Returns at once; the work happens in the background. The result is a
    /// candidates document (or, when there is nothing to ask, a groups document).
    fn analyze(&self, request: AnalysisRequest) -> Box<dyn AnalysisJob>;

    /// Turn the user's answers into groups.
    fn resolve(&self, request: ResolveRequest) -> Box<dyn AnalysisJob>;

    /// Ask an LLM which sentences to keep. None when the service cannot.
    fn adjudicate(&self, _request: AdjudicateRequest) -> Option<Box<dyn AnalysisJob>> {
        None
    }

    /// The keepers this service offers and the one it uses now. None when it does not say (an
    /// older service, or one that cannot be reached).
    fn keeper(&self) -> Option<KeeperInfo> {
        None
    }

    /// Use keeper `id` for the next AI suggestions. The job's result is the new keeper state
    /// (`GET /v1/keeper` form); a refusal fails the job with the service's sentence and changes
    /// nothing. None when the service cannot switch.
    fn set_keeper(&self, _id: &str) -> Option<Box<dyn AnalysisJob>> {
        None
    }

    /// Turn removed sentences into exact cut ranges in the chosen mode. The job's result is
    /// `{"ranges": [{"start", "end"}]}` in seconds of file time; check it with [`crate::plan_cuts`].
    /// None when the service cannot ([`Capabilities::refine`]).
    fn refine(&self, _request: RefineRequest) -> Option<Box<dyn AnalysisJob>> {
        None
    }
}

impl Editor {
    /// What to analyse for the video on the timeline: each file used on the main video track once,
    /// with the parts of it that are used. Offline files are an error (there is nothing to read).
    pub fn analysis_requests(&self) -> Result<Vec<AnalysisRequest>> {
        let q = self.session().active_sequence().ok_or(EditorError::NoSequence)?;
        let track = q.video_tracks.first().ok_or(EditorError::NoSequence)?;
        debug_assert_eq!(track.kind, TrackKind::Video);
        let mut requests: Vec<AnalysisRequest> = Vec::new();
        for clip in &track.items {
            let used = (clip.source_in, clip.source_in + clip.duration);
            match requests.iter_mut().find(|r| r.item == clip.item) {
                Some(r) => r.used.push(used),
                None => {
                    let m = self.media_entry(clip.item).ok_or(EditorError::NoSuchMedia)?;
                    if m.offline {
                        return Err(EditorError::FileMissing(m.offline_reason.unwrap_or(m.name)));
                    }
                    let Some(path) = m.path else { continue };
                    let fingerprint = crate::review::fingerprint_file(std::path::Path::new(&path)).ok();
                    requests.push(AnalysisRequest {
                        item: clip.item,
                        name: m.name,
                        path,
                        duration: m.duration,
                        fingerprint,
                        used: vec![used],
                        settings: crate::options::AnalysisSettings::default(),
                    });
                }
            }
        }
        for r in &mut requests {
            r.used.sort();
            let mut merged: Vec<(Tick, Tick)> = Vec::new();
            for (a, b) in r.used.drain(..) {
                match merged.last_mut() {
                    Some(last) if a <= last.1 => last.1 = last.1.max(b),
                    _ => merged.push((a, b)),
                }
            }
            r.used = merged;
        }
        if requests.is_empty() {
            return Err(EditorError::NotAvailable("There is no video on the timeline to look at yet.".into()));
        }
        Ok(requests)
    }
}

// ── a stand-in service ─────────────────────────────────────────────────────

/// A service that behaves like the real pipeline, on invented sentences, to build and test
/// everything around it. Every request takes `steps` steps of `step_delay`.
///
/// `analyze` returns candidates: 14 sentences spread over the file with 9 pairs, four of them
/// ambiguous. `resolve` clusters the pairs the way the pipeline's Step 05 does: a pair the user
/// answered "retake" joins its sentences, "different" does not; a pair the pipeline was sure about
/// joins them; an unanswered ambiguous pair is decided by a pretend model (score + 0.26: a group
/// whose pairs reach 0.8 is `likely`, one that only reaches 0.45 is returned as not likely, below
/// that the pair is dropped). It never says what to keep.
#[derive(Debug, Clone)]
pub struct FakeService {
    pub steps: u32,
    pub step_delay: Duration,
    /// `analyze` fails with this.
    pub fail_analyze: Option<String>,
    /// `resolve` fails with this.
    pub fail_resolve: Option<String>,
    /// Report a video length that does not match the file (to test the "different file" refusal).
    pub wrong_length: bool,
    /// Offer the optional LLM extension.
    pub adjudicate: bool,
    /// `analyze` answers with groups straight away (nothing to ask the user).
    pub skip_questions: bool,
    /// Offer exact cut ranges (`refine`).
    pub refine: bool,
    /// `refine` answers with ranges that break the rules, to test the fallbacks.
    pub bad_refine: bool,
    /// `refine` fails with this message, to test error handling.
    pub fail_refine: Option<String>,
}

impl Default for FakeService {
    fn default() -> Self {
        Self {
            steps: 4,
            step_delay: Duration::from_millis(30),
            fail_analyze: None,
            fail_resolve: None,
            wrong_length: false,
            adjudicate: true,
            skip_questions: false,
            refine: true,
            bad_refine: false,
            fail_refine: None,
        }
    }
}

/// (id, start, end, text) at 60 s; scaled to the file.
const SENTENCES: [(&str, f64, f64, &str); 14] = [
    ("3:0", 4.10, 6.30, "So now we're going to learn about"),
    ("4:0", 7.00, 9.40, "So [redacted sentence] editing"),
    ("5:0", 10.20, 14.60, "So [redacted sentence] video editing basics."),
    ("9:0", 21.50, 24.00, "And that is the key idea."),
    ("11:0", 27.30, 31.80, "And that is the key idea of the whole video."),
    ("13:0", 33.00, 34.80, "Let's get started."),
    ("14:0", 35.00, 37.40, "Let's get started with the first step."),
    ("15:0", 38.00, 40.20, "Click export and wait."),
    ("16:0", 41.00, 43.10, "Click export and then wait."),
    ("17:0", 44.00, 45.50, "Now open the menu."),
    ("18:0", 46.00, 48.00, "Now open the file menu."),
    ("20:0", 49.00, 51.40, "Thanks for watching."),
    ("21:0", 52.10, 54.00, "Thanks so much for watching."),
    ("22:0", 55.00, 58.20, "Thanks for watching and see you next time."),
];

/// (a, b, likely?, score)
const PAIRS: [(&str, &str, bool, f64); 9] = [
    ("3:0", "4:0", true, 0.93),
    ("3:0", "5:0", true, 0.90),
    ("4:0", "5:0", true, 0.96),
    ("9:0", "11:0", false, 0.58),
    ("13:0", "14:0", false, 0.55),
    ("15:0", "16:0", true, 0.89),
    ("17:0", "18:0", false, 0.52),
    ("20:0", "21:0", true, 0.91),
    ("21:0", "22:0", false, 0.62),
];

impl FakeService {
    fn job_id(item: ItemId) -> String {
        format!("fake-{}", item.0)
    }

    fn candidates_for(&self, request: &AnalysisRequest) -> String {
        let total = request.duration.0 as f64 / 254_016_000_000.0;
        let k = total / 60.0;
        let claimed = if self.wrong_length { total + 30.0 } else { total };
        let sentences: Vec<_> = SENTENCES
            .iter()
            .map(|(id, a, b, text)| serde_json::json!({"id": id, "start": round2(a * k), "end": round2(b * k), "text": text, "confidence": 0.97}))
            .collect();
        let pairs: Vec<_> = PAIRS
            .iter()
            .map(|(a, b, likely, score)| {
                serde_json::json!({"id": format!("{a}|{b}"), "a": a, "b": b, "class": if *likely { "likely" } else { "ambiguous" }, "score": score, "lean": if *score >= 0.5 { "retake" } else { "different" }})
            })
            .collect();
        serde_json::json!({
            "schema": "simplecut.retakes/2", "stage": "candidates", "job": Self::job_id(request.item),
            "media": {"path": request.path, "duration_seconds": claimed, "fingerprint": request.fingerprint},
            "generator": {"name": "fake", "version": "0"},
            "sentences": sentences, "pairs": pairs
        })
        .to_string()
    }

    /// Cluster `candidates` with the user's `labels` into a groups document.
    fn groups_from(candidates: &DocV2, labels: &LabelsRequest) -> String {
        // edge: (pair index, confident?)
        let mut edges: Vec<(usize, usize, bool)> = Vec::new();
        for p in &candidates.pairs {
            let (Some(a), Some(b)) = (candidates.sentences.iter().position(|s| s.id == p.a), candidates.sentences.iter().position(|s| s.id == p.b)) else {
                continue;
            };
            match labels.labels.iter().find(|l| l.pair == p.id).map(|l| l.label) {
                Some(Verdict::Retake) => edges.push((a, b, true)),
                Some(Verdict::Different) => {}
                None => match p.class {
                    PairClass::Likely => edges.push((a, b, true)),
                    PairClass::Ambiguous => {
                        let ce = (p.score + 0.26).min(1.0);
                        if ce >= 0.8 {
                            edges.push((a, b, true));
                        } else if ce >= 0.45 {
                            edges.push((a, b, false));
                        }
                    }
                },
            }
        }
        let n = candidates.sentences.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], x: usize) -> usize {
            let mut r = x;
            while parent.get(r).copied().unwrap_or(r) != r {
                r = parent.get(r).copied().unwrap_or(r);
            }
            r
        }
        for (a, b, _) in &edges {
            let (ra, rb) = (find(&mut parent, *a), find(&mut parent, *b));
            if ra != rb
                && let Some(slot) = parent.get_mut(rb)
            {
                *slot = ra;
            }
        }
        let mut comps: Vec<(usize, Vec<usize>)> = Vec::new();
        for i in 0..n {
            let r = find(&mut parent, i);
            match comps.iter_mut().find(|(root, _)| *root == r) {
                Some((_, v)) => v.push(i),
                None => comps.push((r, vec![i])),
            }
        }
        let mut groups: Vec<serde_json::Value> = Vec::new();
        let mut sorted: Vec<Vec<usize>> = comps.into_iter().map(|(_, v)| v).filter(|v| v.len() >= 2).collect();
        sorted.sort_by(|a, b| {
            let first = |v: &Vec<usize>| v.iter().filter_map(|i| candidates.sentences.get(*i)).map(|s| (s.start * 1000.0) as i64).min();
            first(a).cmp(&first(b))
        });
        for (k, members) in sorted.iter().enumerate() {
            let sentences: Vec<&crate::review::DocSentence> = members.iter().filter_map(|i| candidates.sentences.get(*i)).collect();
            let start = sentences.iter().map(|s| s.start).fold(f64::INFINITY, f64::min);
            let end = sentences.iter().map(|s| s.end).fold(f64::NEG_INFINITY, f64::max);
            let likely = edges.iter().filter(|(a, _, _)| members.contains(a)).all(|(_, _, confident)| *confident);
            groups.push(serde_json::json!({
                "id": format!("span_{k}"), "start": start, "end": end, "likely": likely,
                "sentences": sentences.iter().map(|s| serde_json::json!({"id": s.id, "start": s.start, "end": s.end, "text": s.text})).collect::<Vec<_>>()
            }));
        }
        serde_json::json!({
            "schema": "simplecut.retakes/2", "stage": "groups", "job": candidates.job,
            "media": {"path": candidates.media.path, "duration_seconds": candidates.media.duration_seconds},
            "generator": {"name": "fake", "version": "0", "model": "fake-cross-encoder"},
            "groups": groups
        })
        .to_string()
    }
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

impl RetakeService for FakeService {
    fn name(&self) -> String {
        "Demo service".into()
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { adjudicate: self.adjudicate, refine: self.refine }
    }

    fn analyze(&self, request: AnalysisRequest) -> Box<dyn AnalysisJob> {
        let me = self.clone();
        let label = format!("Listening to “{}”", request.name);
        spawn_job(self.steps, self.step_delay, label, move || {
            if let Some(why) = &me.fail_analyze {
                return Err(why.clone());
            }
            let candidates = me.candidates_for(&request);
            if me.skip_questions {
                let doc = review::parse_v2(&candidates).map_err(|e| e.to_string())?;
                return Ok(Self::groups_from(&doc, &LabelsRequest { job: doc.job.clone(), labels: Vec::new(), skipped: Vec::new() }));
            }
            Ok(candidates)
        })
    }

    fn resolve(&self, request: ResolveRequest) -> Box<dyn AnalysisJob> {
        let me = self.clone();
        spawn_job(self.steps, self.step_delay, "Checking with the model".into(), move || {
            if let Some(why) = &me.fail_resolve {
                return Err(why.clone());
            }
            let text = request.candidates.as_deref().ok_or("The demo service needs the candidates it sent.")?;
            let doc = review::parse_v2(text).map_err(|e| e.to_string())?;
            Ok(Self::groups_from(&doc, &request.labels))
        })
    }

    fn adjudicate(&self, request: AdjudicateRequest) -> Option<Box<dyn AnalysisJob>> {
        if !self.adjudicate {
            return None;
        }
        Some(spawn_job(self.steps, self.step_delay, "Asking the AI".into(), move || {
            let reply: Vec<_> = request
                .groups
                .iter()
                .filter_map(|g| g.sentences.last().map(|s| serde_json::json!({"group": g.id, "keep": [s.id], "reason": "Kept the last take (demo)."})))
                .collect();
            Ok(serde_json::Value::Array(reply).to_string())
        }))
    }

    fn refine(&self, request: RefineRequest) -> Option<Box<dyn AnalysisJob>> {
        if !self.refine {
            return None;
        }
        let bad = self.bad_refine;
        let me = self.clone();
        Some(spawn_job(self.steps, self.step_delay, "Refining the cuts".into(), move || {
            if let Some(why) = &me.fail_refine {
                return Err(why.clone());
            }
            let secs = |t: Tick| t.0 as f64 / 254_016_000_000.0;
            // how far past the sentence each mode reaches, standing in for the real audio analysis
            let (before, after) = match request.mode {
                CutMode::Sentences => (0.3, 0.3),
                CutMode::Anchored => (0.2, 0.2),
                CutMode::RetakeOnly => (-0.05, -0.05),
            };
            let mut spans: Vec<(f64, f64)> = request.spans.iter().map(|(_, a, b)| (secs(*a), secs(*b))).collect();
            spans.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut merged: Vec<(f64, f64)> = Vec::new();
            for (a, b) in spans {
                let (start, end) = (round2((a - before).max(0.0)), round2(b + after));
                match merged.last_mut() {
                    Some(last) if start <= last.1 => last.1 = last.1.max(end),
                    _ => merged.push((start, end)),
                }
            }
            let mut ranges: Vec<serde_json::Value> = merged.iter().map(|(s, e)| serde_json::json!({"start": s, "end": e})).collect();
            if bad {
                ranges.push(serde_json::json!({"start": 5.0, "end": 4.0}));
            }
            Ok(serde_json::json!({"ranges": ranges}).to_string())
        }))
    }
}

// ── jobs on a thread ───────────────────────────────────────────────────────

struct ThreadJob {
    state: Arc<Mutex<JobState>>,
    cancel: Arc<AtomicBool>,
}

impl AnalysisJob for ThreadJob {
    fn poll(&mut self) -> JobState {
        self.state.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn cancel(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Run `work` on a thread after `steps` pauses, reporting progress and honouring cancel.
fn spawn_job(steps: u32, delay: Duration, label: String, work: impl FnOnce() -> std::result::Result<String, String> + Send + 'static) -> Box<dyn AnalysisJob> {
    let state = Arc::new(Mutex::new(JobState::Running { fraction: 0.0, status: format!("{label}…") }));
    let cancel = Arc::new(AtomicBool::new(false));
    let (shared, stop) = (state.clone(), cancel.clone());
    let steps = steps.max(1);
    let worker = std::thread::Builder::new().name("simplecut-fake-service".into()).spawn(move || {
        let set = |s: JobState| *shared.lock().unwrap_or_else(PoisonError::into_inner) = s;
        for i in 0..steps {
            std::thread::sleep(delay);
            if stop.load(Ordering::Relaxed) {
                set(JobState::Cancelled);
                return;
            }
            set(JobState::Running { fraction: (i + 1) as f32 / steps as f32, status: format!("{label} ({} of {steps})", i + 1) });
        }
        set(match work() {
            Ok(text) => JobState::Done(text),
            Err(why) => JobState::Failed(why),
        });
    });
    if worker.is_err() {
        *state.lock().unwrap_or_else(PoisonError::into_inner) = JobState::Failed("Could not start the request.".into());
    }
    Box::new(ThreadJob { state, cancel })
}
