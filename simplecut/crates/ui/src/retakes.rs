//! Remove Retakes, step by step.
//!
//! 1. **Analyzing**: the service looks at the file and returns candidates (sentences and pairs).
//! 2. **Questions**: the user answers the pairs the pipeline was unsure about ("same line" /
//!    "different lines"), then presses Next. The answers are saved to a local log first, then sent.
//! 3. **Resolving**: the service's trained cross-encoder turns the answers into groups of sentences.
//! 4. **Review**: the user chooses which sentences of each group to **keep**. Every sentence that is
//!    not kept is cut, in one undo step. Nothing is cut until the user presses Remove, the pipeline
//!    never says what to keep, and the optional AI button only fills in ticks the user can change.
//!
//! Which groups can still be acted on is read from the live timeline every frame (a group with
//! fewer than two sentences on the timeline is "already handled"), so Undo brings groups back.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use egui::{Color32, CornerRadius, RichText, WidgetInfo, WidgetType, vec2};
use simplecut_core::retake::{CutProposal, CutScope, MAX_DOCUMENT_BYTES, Presence};
use simplecut_core::review::parse_advice;
use simplecut_core::{
    AdjudicateRequest, AnalysisJob, Candidates, Checked, ItemId, JobState, Label, LabelsRequest, PairClass, ResolveRequest, ReviewGroup, Sentence, Tick,
    Verdict,
};

use crate::app::{Action, SimpleCutApp, Source};
use crate::{format, theme};

const TICKS_PER_SECOND: f64 = 254_016_000_000.0;

/// Who set a sentence's keep tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    User,
    Ai,
}

pub(crate) struct Running {
    job: Box<dyn AnalysisJob>,
    pub item: ItemId,
    pub status: String,
    pub fraction: f32,
}

impl Running {
    fn new(job: Box<dyn AnalysisJob>, item: ItemId, status: &str) -> Self {
        Self { job, item, status: status.into(), fraction: 0.0 }
    }

    /// Follow the job; the finished result is handed back.
    fn poll(&mut self) -> Option<JobState> {
        match self.job.poll() {
            JobState::Running { fraction, status } => {
                self.fraction = fraction;
                self.status = status;
                None
            }
            done => Some(done),
        }
    }
}

/// The pairs the user is asked about.
pub(crate) struct Questions {
    pub item: ItemId,
    /// The candidates document as received (a service that restarted can use it again).
    pub raw: String,
    pub candidates: Candidates,
    /// Ambiguous pair ids, in video order.
    pub order: Vec<String>,
    pub answers: BTreeMap<String, Verdict>,
    pub skipped: BTreeSet<String>,
    /// The card shown.
    pub index: usize,
    /// Next was pressed with unanswered questions: waiting for the user to confirm.
    pub confirm: bool,
}

impl Questions {
    fn new(item: ItemId, raw: String, candidates: Candidates) -> Self {
        let order = candidates.ambiguous().into_iter().map(|p| p.id.clone()).collect();
        Self { item, raw, candidates, order, answers: BTreeMap::new(), skipped: BTreeSet::new(), index: 0, confirm: false }
    }

    fn unanswered(&self) -> usize {
        self.order.iter().filter(|id| !self.answers.contains_key(*id)).count()
    }
}

/// The groups, and what the user keeps.
pub(crate) struct Review {
    pub item: ItemId,
    pub job: String,
    pub groups: Vec<ReviewGroup>,
    /// Sentence id -> (keep, who set it). A sentence not in the map is kept: nothing is cut until
    /// the user (or the AI, on request) chooses.
    pub keep: HashMap<String, (bool, Origin)>,
    /// Sentence id -> original (start, end) on review open.
    pub original: HashMap<String, (Tick, Tick)>,
    /// Sentences that have been on the timeline at some point.
    pub seen_on: HashSet<String>,
    pub rejected: Vec<(String, String)>,
    /// The questions this came from, so they can be asked again.
    pub from: Option<Questions>,
    pub advice: Option<Running>,
    pub notes: Vec<String>,
    /// AI reasons by group id.
    pub reasons: HashMap<String, String>,
    pub ai_consent_pending: bool,
    pub skip_unticked: bool,
}

impl Review {
    fn is_kept(&self, id: &str) -> bool {
        self.keep.get(id).is_none_or(|(k, _)| *k)
    }

    fn origin(&self, id: &str) -> Option<Origin> {
        self.keep.get(id).map(|(_, o)| *o)
    }
}

// only one exists at a time
#[allow(clippy::large_enum_variant)]
pub(crate) enum Stage {
    Analyzing(Running),
    Questions(Questions),
    Resolving { run: Running, questions: Questions },
    Review(Review),
}

pub(crate) struct RetakeUi {
    pub stage: Option<Stage>,
    /// The user agreed that sentences may be sent to the AI.
    pub ai_consented: bool,
    /// Where answers are logged before they are sent.
    pub label_log: Option<PathBuf>,
    /// Where sentence span edits are logged before they are applied.
    pub span_log: Option<PathBuf>,
    /// Highlight strength of cut sentences on the timeline (0.1..=0.8).
    pub highlight_strength: f32,
}

impl Default for RetakeUi {
    fn default() -> Self {
        Self { stage: None, ai_consented: false, label_log: None, span_log: None, highlight_strength: 0.45 }
    }
}

impl RetakeUi {
    pub fn visible(&self) -> bool {
        self.stage.is_some()
    }

    pub fn busy(&self) -> bool {
        match &self.stage {
            Some(Stage::Analyzing(_) | Stage::Resolving { .. }) => true,
            Some(Stage::Review(r)) => r.advice.is_some(),
            _ => false,
        }
    }
}

/// What the timeline draws for the review.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum MarkKind {
    /// The span of a group (stronger for a likely one).
    Band { likely: bool },
    /// A sentence that will be cut.
    Cut,
    /// A sentence that stays.
    Kept,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Mark {
    pub start: Tick,
    pub end: Tick,
    pub color: usize,
    pub kind: MarkKind,
    pub sentence_id: Option<String>,
}

/// A group's place in the list: its label.
fn label_of(index: usize) -> String {
    format!("Retake {}", index + 1)
}

fn colour_of(index: usize) -> Color32 {
    theme::GROUP_COLORS[index % theme::GROUP_COLORS.len()]
}

impl SimpleCutApp {
    /// Connect the service that finds retakes. Until then the "Remove Retakes" button is hidden.
    pub fn set_service(&mut self, service: Option<Box<dyn simplecut_core::RetakeService>>) {
        self.service = service;
    }

    pub fn has_service(&self) -> bool {
        self.service.is_some()
    }

    /// Whether the service offers the optional AI extension.
    pub fn ai_available(&self) -> bool {
        self.service.as_ref().is_some_and(|s| s.capabilities().adjudicate)
    }

    /// True while a request to the service is running.
    pub fn is_analyzing(&self) -> bool {
        self.retakes.busy()
    }

    /// Where the user's answers are logged before they are sent.
    pub fn set_label_log(&mut self, path: Option<PathBuf>) {
        self.retakes.label_log = path;
    }

    /// Where sentence span adjustments are logged before they are applied.
    pub fn set_span_log(&mut self, path: Option<PathBuf>) {
        self.retakes.span_log = path;
    }

    /// Highlight strength of cut sentences on the timeline (0.1..=0.8).
    pub fn highlight_strength(&self) -> f32 {
        self.retakes.highlight_strength
    }

    pub fn set_highlight_strength(&mut self, val: f32) {
        self.retakes.highlight_strength = val.clamp(0.1, 0.8);
    }

    /// Which step the Retakes panel is on: "none", "analyzing", "questions", "resolving" or "review".
    pub fn review_stage(&self) -> &'static str {
        match &self.retakes.stage {
            None => "none",
            Some(Stage::Analyzing(_)) => "analyzing",
            Some(Stage::Questions(_)) => "questions",
            Some(Stage::Resolving { .. }) => "resolving",
            Some(Stage::Review(_)) => "review",
        }
    }

    /// (questions, answered, skipped, the card shown) while the questions are open.
    pub fn question_summary(&self) -> Option<(usize, usize, usize, usize)> {
        match &self.retakes.stage {
            Some(Stage::Questions(q)) => Some((q.order.len(), q.answers.len(), q.skipped.len(), q.index)),
            _ => None,
        }
    }

    /// All question pair ids, in order.
    pub fn question_ids(&self) -> Vec<String> {
        match &self.retakes.stage {
            Some(Stage::Questions(q)) => q.order.clone(),
            _ => Vec::new(),
        }
    }

    /// The groups that can be acted on now: (label, colour index, likely, sentence ids).
    pub fn review_groups(&self) -> Vec<(String, usize, bool, Vec<String>)> {
        let Some(Stage::Review(r)) = &self.retakes.stage else { return Vec::new() };
        r.groups
            .iter()
            .enumerate()
            .filter(|(_, g)| self.actionable(r.item, g))
            .map(|(i, g)| (label_of(i), i % theme::GROUP_COLORS.len(), g.likely, g.sentences.iter().map(|s| s.id.clone()).collect()))
            .collect()
    }

    /// Ids of the groups that are already handled or outside the used part of the video.
    pub fn review_handled(&self) -> Vec<String> {
        let Some(Stage::Review(r)) = &self.retakes.stage else { return Vec::new() };
        r.groups.iter().filter(|g| !self.actionable(r.item, g)).map(|g| g.id.clone()).collect()
    }

    /// Whether a sentence is ticked as kept (None when there is no such sentence).
    pub fn review_kept(&self, id: &str) -> Option<bool> {
        let Some(Stage::Review(r)) = &self.retakes.stage else { return None };
        r.groups.iter().any(|g| g.sentences.iter().any(|s| s.id == id)).then(|| r.is_kept(id))
    }

    /// Who set a sentence's tick: "user", "ai", or None when nobody has (the default).
    pub fn review_origin(&self, id: &str) -> Option<&'static str> {
        let Some(Stage::Review(r)) = &self.retakes.stage else { return None };
        r.origin(id).map(|o| if o == Origin::Ai { "ai" } else { "user" })
    }

    /// How many sentences Remove would cut now.
    pub fn review_cut_count(&self) -> usize {
        match &self.retakes.stage {
            Some(Stage::Review(r)) => self.cuts(r).len(),
            _ => 0,
        }
    }

    /// Notes about things in the last AI reply that were ignored.
    pub fn review_notes(&self) -> Vec<String> {
        match &self.retakes.stage {
            Some(Stage::Review(r)) => r.notes.clone(),
            _ => Vec::new(),
        }
    }

    /// How many unusable entries were dropped when the document was read.
    pub fn review_ignored(&self) -> usize {
        match &self.retakes.stage {
            Some(Stage::Review(r)) => r.rejected.len(),
            Some(Stage::Questions(q)) => q.candidates.rejected.len(),
            _ => 0,
        }
    }

    /// Whether unticked sentences are skipped during playback.
    pub fn review_skips_unticked(&self) -> bool {
        match &self.retakes.stage {
            Some(Stage::Review(r)) => r.skip_unticked,
            _ => false,
        }
    }

    /// The media item of the review currently open.
    pub fn review_item(&self) -> Option<ItemId> {
        match &self.retakes.stage {
            Some(Stage::Review(r)) => Some(r.item),
            _ => None,
        }
    }

    /// Group (start, end) spans in media time ticks for the review currently open.
    pub fn review_group_spans(&self) -> Vec<(Tick, Tick)> {
        match &self.retakes.stage {
            Some(Stage::Review(r)) => r.groups.iter().map(|g| (g.start, g.end)).collect(),
            _ => Vec::new(),
        }
    }

    /// Sentence (start, end) span in media time ticks.
    pub fn review_sentence_span(&self, sentence_id: &str) -> Option<(Tick, Tick)> {
        match &self.retakes.stage {
            Some(Stage::Review(r)) => {
                for g in &r.groups {
                    if let Some(s) = g.sentences.iter().find(|s| s.id == sentence_id) {
                        return Some((s.start, s.end));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Sentence (start, end) span in seconds.
    pub fn sentence_span_f64(&self, sentence_id: &str) -> Option<(f64, f64)> {
        self.review_sentence_span(sentence_id).map(|(s, e)| (s.0 as f64 / 254_016_000_000.0, e.0 as f64 / 254_016_000_000.0))
    }

    /// Whether a sentence's span has been changed from its original time.
    pub fn is_sentence_span_edited(&self, sentence_id: &str) -> bool {
        match &self.retakes.stage {
            Some(Stage::Review(r)) => {
                if let Some(orig) = r.original.get(sentence_id)
                    && let Some(cur) = self.review_sentence_span(sentence_id)
                {
                    return (cur.0.0 - orig.0.0).abs() > 1000 || (cur.1.0 - orig.1.0).abs() > 1000;
                }
                false
            }
            _ => false,
        }
    }

    fn on(&self, item: ItemId, s: &Sentence) -> bool {
        self.editor.presence_of(item, s.start, s.end) == Presence::On
    }

    /// A group can be acted on while at least two of its sentences are on the timeline.
    fn actionable(&self, item: ItemId, g: &ReviewGroup) -> bool {
        g.sentences.iter().filter(|s| self.on(item, s)).count() >= 2
    }

    /// The sentences Remove would cut: not kept, on the timeline, in a group that can be acted on.
    fn cuts<'a>(&self, r: &'a Review) -> Vec<&'a Sentence> {
        r.groups.iter().filter(|g| self.actionable(r.item, g)).flat_map(|g| g.sentences.iter()).filter(|s| !r.is_kept(&s.id) && self.on(r.item, s)).collect()
    }

    /// What the timeline draws for the open review.
    pub(crate) fn review_marks(&self) -> Vec<Mark> {
        let Some(Stage::Review(r)) = &self.retakes.stage else { return Vec::new() };
        let mut out = Vec::new();
        for (i, g) in r.groups.iter().enumerate() {
            if !self.actionable(r.item, g) {
                continue;
            }
            let color = i % theme::GROUP_COLORS.len();
            for (s, e) in self.editor.timeline_ranges_of(r.item, g.start, g.end) {
                out.push(Mark { start: s, end: e, color, kind: MarkKind::Band { likely: g.likely }, sentence_id: None });
            }
            for sent in g.sentences.iter().filter(|s| self.on(r.item, s)) {
                let kind = if r.is_kept(&sent.id) { MarkKind::Kept } else { MarkKind::Cut };
                for (s, e) in self.editor.timeline_ranges_of(r.item, sent.start, sent.end) {
                    out.push(Mark { start: s, end: e, color, kind, sentence_id: Some(sent.id.clone()) });
                }
            }
        }
        out
    }

    fn mark_rects(&self, want: impl Fn(&MarkKind) -> bool) -> Vec<(egui::Rect, usize, MarkKind)> {
        let Some(g) = self.tl.last.as_ref() else { return Vec::new() };
        let Some(lane) = g.lanes.first() else { return Vec::new() };
        self.review_marks()
            .into_iter()
            .filter(|m| want(&m.kind))
            .map(|m| (egui::Rect::from_min_max(egui::pos2(g.axis.x_of(m.start), lane.top()), egui::pos2(g.axis.x_of(m.end), lane.bottom())), m.color, m.kind))
            .collect()
    }

    /// Screen rectangles of the sentences that will be cut, on the video lane, as last drawn.
    pub fn review_marker_rects(&self) -> Vec<egui::Rect> {
        self.mark_rects(|k| *k == MarkKind::Cut).into_iter().map(|(r, ..)| r).collect()
    }

    /// Colour indexes of those, in the same order.
    pub fn review_marker_colors(&self) -> Vec<usize> {
        self.mark_rects(|k| *k == MarkKind::Cut).into_iter().map(|(_, c, _)| c).collect()
    }

    /// Screen rectangles of kept sentences (outlined on the timeline).
    pub fn review_kept_rects(&self) -> Vec<egui::Rect> {
        self.mark_rects(|k| *k == MarkKind::Kept).into_iter().map(|(r, ..)| r).collect()
    }

    /// Screen rectangles of the group spans with their colour index and whether the group is likely.
    pub fn review_band_rects(&self) -> Vec<(egui::Rect, usize, bool)> {
        self.mark_rects(|k| matches!(k, MarkKind::Band { .. })).into_iter().map(|(r, c, k)| (r, c, matches!(k, MarkKind::Band { likely: true }))).collect()
    }

    // ── actions ────────────────────────────────────────────────────────────

    pub(crate) fn perform_retake(&mut self, action: Action) {
        match action {
            Action::StartRetakeAnalysis => self.start_analysis(),
            Action::CancelAnalysis => self.cancel_request(),
            Action::OpenAnalysisFile => {
                if let Some(path) = self.dialogs.pick_analysis_file() {
                    self.load_analysis(path);
                }
            }
            Action::LoadAnalysis(path) => self.load_analysis(path),
            Action::AnswerPair { pair, verdict } => self.answer_pair(&pair, verdict),
            Action::SetQuestion(i) => {
                if let Some(Stage::Questions(q)) = &mut self.retakes.stage {
                    q.index = i.min(q.order.len().saturating_sub(1));
                    q.confirm = false;
                }
            }
            Action::QuestionsNext => self.questions_next(false),
            Action::QuestionsSendAnyway => self.questions_next(true),
            Action::SetKeep { sentence, keep } => self.set_keep(&sentence, keep),
            Action::KeepLastOnly(group) => self.keep_last(&group),
            Action::KeepAllOf(group) => self.keep_all(&group),
            Action::RemoveAllOf(group) => self.remove_all(&group),
            Action::KeepLastInLikely => {
                let ids: Vec<String> = match &self.retakes.stage {
                    Some(Stage::Review(r)) => r.groups.iter().filter(|g| g.likely && self.actionable(r.item, g)).map(|g| g.id.clone()).collect(),
                    _ => Vec::new(),
                };
                for id in ids {
                    self.keep_last(&id);
                }
            }
            Action::AskAi => self.ask_ai(false),
            Action::ConfirmAi => self.ask_ai(true),
            Action::DeclineAi => {
                if let Some(Stage::Review(r)) = &mut self.retakes.stage {
                    r.ai_consent_pending = false;
                }
            }
            Action::RedoQuestions => self.redo_questions(),
            Action::GoToSentence(item, start) => {
                self.go_to(item, start);
            }
            Action::PlaySentence(item, start) => {
                if self.go_to(item, start) && !self.player.is_playing() {
                    self.queue(Action::TogglePlay);
                }
            }
            Action::ApplyRetakes => self.apply_retakes(),
            Action::CloseReview => self.close_review(),
            Action::SetSkipUnticked(skip) => {
                if let Some(Stage::Review(r)) = &mut self.retakes.stage {
                    r.skip_unticked = skip;
                }
            }
            Action::SetSentenceSpan { sentence, start, end } => self.set_sentence_span(&sentence, start, end),
            Action::ResetSentenceSpan(sentence) => self.reset_sentence_span(&sentence),
            Action::SaveGroups => self.save_groups(),
            _ => {}
        }
    }

    fn start_analysis(&mut self) {
        if self.retakes.busy() || matches!(self.retakes.stage, Some(Stage::Questions(_))) {
            return;
        }
        let Some(service) = &self.service else {
            self.fail_text("No retake service is connected.");
            return;
        };
        let requests = match self.editor.analysis_requests() {
            Ok(r) => r,
            Err(e) => {
                self.fail_text(&e.to_string());
                return;
            }
        };
        let Some(first) = requests.first().cloned() else { return };
        let job = service.analyze(first.clone());
        self.player.pause();
        self.retakes.stage = Some(Stage::Analyzing(Running::new(job, first.item, "Starting…")));
        if requests.len() > 1 {
            self.info_text(format!("The timeline uses {} video files. Looking at “{}” first; run it again for the others.", requests.len(), first.name));
        }
    }

    fn cancel_request(&mut self) {
        match self.retakes.stage.take() {
            Some(Stage::Analyzing(mut run)) => {
                run.job.cancel();
                self.info_text("Analysis cancelled.");
            }
            Some(Stage::Resolving { mut run, questions }) => {
                run.job.cancel();
                self.retakes.stage = Some(Stage::Questions(questions));
                self.info_text("Cancelled. Your answers are kept.");
            }
            Some(Stage::Review(mut r)) => {
                if let Some(mut a) = r.advice.take() {
                    a.job.cancel();
                    self.info_text("Cancelled the AI request.");
                }
                self.retakes.stage = Some(Stage::Review(r));
            }
            other => self.retakes.stage = other,
        }
    }

    fn close_review(&mut self) {
        self.player.set_skips(Vec::new());
        match self.retakes.stage.take() {
            Some(Stage::Analyzing(mut run)) | Some(Stage::Resolving { mut run, .. }) => run.job.cancel(),
            Some(Stage::Review(mut r)) => {
                if let Some(a) = &mut r.advice {
                    a.job.cancel();
                }
            }
            _ => {}
        }
    }

    /// Called once per frame: follow the running request and keep the live-view bookkeeping.
    pub(crate) fn pump_retakes(&mut self) {
        let Some(stage) = self.retakes.stage.take() else { return };
        self.retakes.stage = match stage {
            Stage::Analyzing(mut run) => match run.poll() {
                None => Some(Stage::Analyzing(run)),
                Some(JobState::Done(text)) => self.analysis_done(run.item, &text),
                Some(JobState::Failed(why)) => {
                    self.fail_text(&format!("The retake service could not finish: {why}"));
                    None
                }
                Some(_) => {
                    self.info_text("Analysis cancelled.");
                    None
                }
            },
            Stage::Resolving { mut run, questions } => match run.poll() {
                None => Some(Stage::Resolving { run, questions }),
                Some(JobState::Done(text)) => self.resolve_done(&text, questions),
                Some(JobState::Failed(why)) => {
                    self.fail_text(&format!("The model could not finish: {why}. Your answers are kept."));
                    Some(Stage::Questions(questions))
                }
                Some(_) => Some(Stage::Questions(questions)),
            },
            Stage::Review(mut r) => {
                self.review_tick(&mut r);
                Some(Stage::Review(r))
            }
            other => Some(other),
        };
    }

    fn analysis_done(&mut self, item: ItemId, text: &str) -> Option<Stage> {
        match self.editor.read_review(text, item) {
            Err(e) => {
                self.fail_text(&e.to_string());
                None
            }
            Ok(Checked::Candidates(c)) => Some(self.open_questions(Questions::new(item, text.to_string(), c))),
            Ok(Checked::Groups(g)) => Some(self.open_review(g.item, g.job, g.groups, g.rejected, None)),
        }
    }

    /// Show the questions, or, when there are none, go straight to the model.
    fn open_questions(&mut self, q: Questions) -> Stage {
        if !q.order.is_empty() {
            return Stage::Questions(q);
        }
        let Some(service) = &self.service else { return Stage::Questions(q) };
        let labels = LabelsRequest { job: q.candidates.job.clone(), labels: Vec::new(), skipped: Vec::new() };
        let job = service.resolve(ResolveRequest { item: q.item, labels, candidates: Some(q.raw.clone()) });
        let item = q.item;
        Stage::Resolving { run: Running::new(job, item, "Checking with the model…"), questions: q }
    }

    fn open_review(
        &mut self,
        item: ItemId,
        job: String,
        groups: Vec<ReviewGroup>,
        rejected: Vec<simplecut_core::retake::Rejected>,
        from: Option<Questions>,
    ) -> Stage {
        let found = groups.len();
        let likely = groups.iter().filter(|g| g.likely).count();
        let ignored = rejected.len();
        let mut original = HashMap::new();
        for g in &groups {
            for s in &g.sentences {
                original.insert(s.id.clone(), (s.start, s.end));
            }
        }
        let mut r = Review {
            item,
            job,
            groups,
            keep: HashMap::new(),
            original,
            seen_on: HashSet::new(),
            rejected: rejected.into_iter().map(|x| (x.id, x.reason)).collect(),
            from,
            advice: None,
            notes: Vec::new(),
            reasons: HashMap::new(),
            ai_consent_pending: false,
            skip_unticked: true,
        };
        self.review_tick(&mut r);
        let handled = r.groups.iter().filter(|g| !self.actionable(item, g)).count();
        let mut text = if found == 0 {
            "No retakes found.".to_string()
        } else {
            format!("Found {found} group{} ({likely} likely). Choose which sentences to keep, then remove the rest.", if found == 1 { "" } else { "s" })
        };
        if handled > 0 {
            text.push_str(&format!(" {handled} already handled."));
        }
        if ignored > 0 {
            text.push_str(&format!(" {ignored} unusable entr{} ignored.", if ignored == 1 { "y was" } else { "ies were" }));
        }
        self.info_text(text);
        Stage::Review(r)
    }

    fn review_tick(&mut self, r: &mut Review) {
        for g in &r.groups {
            for s in &g.sentences {
                if self.editor.presence_of(r.item, s.start, s.end) == Presence::On {
                    r.seen_on.insert(s.id.clone());
                }
            }
        }
        let Some(run) = &mut r.advice else { return };
        let Some(state) = run.poll() else { return };
        r.advice = None;
        match state {
            JobState::Done(text) => match parse_advice(&text, &r.groups) {
                Err(e) => self.fail_text(&e.to_string()),
                Ok((advice, notes)) => {
                    let n = advice.len();
                    for a in advice {
                        let ids: Vec<String> =
                            r.groups.iter().find(|g| g.id == a.group).map(|g| g.sentences.iter().map(|s| s.id.clone()).collect()).unwrap_or_default();
                        for id in ids {
                            let keep = a.keep.contains(&id);
                            r.keep.insert(id, (keep, Origin::Ai));
                        }
                        if !a.reason.trim().is_empty() {
                            r.reasons.insert(a.group, a.reason);
                        }
                    }
                    r.notes = notes;
                    self.info_text(format!("The AI chose what to keep in {n} group{}. Nothing is cut until you press Remove.", if n == 1 { "" } else { "s" }));
                }
            },
            JobState::Failed(why) => self.fail_text(&format!("The AI could not finish: {why}")),
            _ => self.info_text("Cancelled the AI request."),
        }
    }

    // ── questions ──────────────────────────────────────────────────────────

    fn answer_pair(&mut self, pair: &str, verdict: Option<Verdict>) {
        let Some(Stage::Questions(q)) = &mut self.retakes.stage else { return };
        let Some(pos) = q.order.iter().position(|id| id == pair) else { return };
        q.confirm = false;
        match verdict {
            Some(v) => {
                q.answers.insert(pair.to_string(), v);
                q.skipped.remove(pair);
            }
            None => {
                q.answers.remove(pair);
                q.skipped.insert(pair.to_string());
            }
        }
        q.index = (pos + 1).min(q.order.len().saturating_sub(1));
    }

    fn questions_next(&mut self, confirmed: bool) {
        let Some(Stage::Questions(q)) = &mut self.retakes.stage else { return };
        if !confirmed && q.unanswered() > 0 {
            q.confirm = true;
            return;
        }
        let Some(Stage::Questions(mut q)) = self.retakes.stage.take() else { return };
        q.confirm = false;
        let labels = LabelsRequest {
            job: q.candidates.job.clone(),
            labels: q.answers.iter().map(|(pair, v)| Label { pair: pair.clone(), label: *v, source: "user".into() }).collect(),
            skipped: q.order.iter().filter(|id| !q.answers.contains_key(*id)).cloned().collect(),
        };
        if let Some(path) = self.retakes.label_log.clone()
            && let Err(e) = append_labels(&path, &q)
        {
            self.fail_text(&format!("Could not save your answers to {}: {e}. Sending them anyway.", path.display()));
        }
        let Some(service) = &self.service else {
            self.retakes.stage = Some(Stage::Questions(q));
            return;
        };
        let job = service.resolve(ResolveRequest { item: q.item, labels, candidates: Some(q.raw.clone()) });
        let item = q.item;
        self.retakes.stage = Some(Stage::Resolving { run: Running::new(job, item, "Checking with the model…"), questions: q });
    }

    fn resolve_done(&mut self, text: &str, q: Questions) -> Option<Stage> {
        match self.editor.read_review(text, q.item) {
            Err(e) => {
                self.fail_text(&format!("{e} Your answers are kept."));
                Some(Stage::Questions(q))
            }
            Ok(Checked::Candidates(_)) => {
                self.fail_text("The service sent more questions instead of groups. Your answers are kept.");
                Some(Stage::Questions(q))
            }
            Ok(Checked::Groups(g)) => {
                if !g.job.is_empty() && g.job != q.candidates.job {
                    self.fail_text("The groups are for a different analysis than your answers. Your answers are kept.");
                    return Some(Stage::Questions(q));
                }
                let job = q.candidates.job.clone();
                Some(self.open_review(g.item, job, g.groups, g.rejected, Some(q)))
            }
        }
    }

    fn redo_questions(&mut self) {
        match self.retakes.stage.take() {
            Some(Stage::Review(mut r)) => match r.from.take() {
                Some(q) => self.retakes.stage = Some(Stage::Questions(q)),
                None => {
                    self.info_text("There are no questions to ask again for this analysis.");
                    self.retakes.stage = Some(Stage::Review(r));
                }
            },
            other => self.retakes.stage = other,
        }
    }

    fn load_analysis(&mut self, path: PathBuf) {
        let text = match std::fs::metadata(&path) {
            Ok(m) if m.len() as usize > MAX_DOCUMENT_BYTES => Err("That file is too large to be a retake analysis.".to_string()),
            Ok(_) => std::fs::read_to_string(&path).map_err(|e| format!("Could not read {}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(format!("The file could not be found: {}", path.display())),
            Err(e) => Err(format!("Could not read {}: {e}", path.display())),
        };
        let result = text.and_then(|t| {
            let item = self.editor.find_media_for_review(&t).map_err(|e| e.to_string())?;
            self.editor.read_review(&t, item).map(|c| (t, item, c)).map_err(|e| e.to_string())
        });
        match result {
            Err(msg) => self.fail_text(&msg),
            Ok((text, item, checked)) => {
                self.player.pause();
                self.close_review();
                self.retakes.stage = Some(match checked {
                    Checked::Candidates(c) => self.open_questions(Questions::new(item, text, c)),
                    Checked::Groups(g) => self.open_review(g.item, g.job, g.groups, g.rejected, None),
                });
            }
        }
    }

    // ── review ─────────────────────────────────────────────────────────────

    fn set_keep(&mut self, sentence: &str, keep: bool) {
        let Some(Stage::Review(mut r)) = self.retakes.stage.take() else { return };
        r.keep.insert(sentence.to_string(), (keep, Origin::User));
        self.retakes.stage = Some(Stage::Review(r));
    }

    fn remove_all(&mut self, group: &str) {
        let Some(Stage::Review(mut r)) = self.retakes.stage.take() else { return };
        if let Some(g) = r.groups.iter().find(|g| g.id == group) {
            for s in &g.sentences {
                if self.on(r.item, s) {
                    r.keep.insert(s.id.clone(), (false, Origin::User));
                }
            }
        }
        self.retakes.stage = Some(Stage::Review(r));
    }

    fn keep_last(&mut self, group: &str) {
        let Some(Stage::Review(mut r)) = self.retakes.stage.take() else { return };
        if let Some(g) = r.groups.iter().find(|g| g.id == group) {
            let last = g.sentences.iter().rfind(|s| self.on(r.item, s)).map(|s| s.id.clone());
            if let Some(last) = last {
                let updates: Vec<(String, bool)> = g.sentences.iter().map(|s| (s.id.clone(), s.id == last || !self.on(r.item, s))).collect();
                for (id, keep) in updates {
                    r.keep.insert(id, (keep, Origin::User));
                }
            }
        }
        self.retakes.stage = Some(Stage::Review(r));
    }

    fn keep_all(&mut self, group: &str) {
        let Some(Stage::Review(mut r)) = self.retakes.stage.take() else { return };
        if let Some(g) = r.groups.iter().find(|g| g.id == group) {
            for s in &g.sentences {
                r.keep.insert(s.id.clone(), (true, Origin::User));
            }
        }
        self.retakes.stage = Some(Stage::Review(r));
    }

    fn ask_ai(&mut self, consented: bool) {
        if !self.ai_available() {
            self.fail_text("The AI extension is not available from this service.");
            return;
        }
        if consented {
            self.retakes.ai_consented = true;
        }
        let consent = self.retakes.ai_consented;
        let Some(Stage::Review(mut r)) = self.retakes.stage.take() else { return };
        if r.advice.is_some() {
            self.retakes.stage = Some(Stage::Review(r));
            return;
        }
        if !consent {
            r.ai_consent_pending = true;
            self.retakes.stage = Some(Stage::Review(r));
            return;
        }
        r.ai_consent_pending = false;
        let live: Vec<ReviewGroup> = r.groups.iter().filter(|g| self.actionable(r.item, g)).cloned().collect();
        let unsure: Vec<ReviewGroup> = live.iter().filter(|g| !g.likely).cloned().collect();
        let chosen = if unsure.is_empty() { live } else { unsure };
        if chosen.is_empty() {
            self.info_text("There is nothing to ask the AI about.");
            self.retakes.stage = Some(Stage::Review(r));
            return;
        }
        let request = AdjudicateRequest { item: r.item, job: r.job.clone(), groups: chosen };
        match self.service.as_ref().and_then(|s| s.adjudicate(request)) {
            Some(job) => r.advice = Some(Running::new(job, r.item, "Asking the AI…")),
            None => self.fail_text("The AI extension is not available from this service."),
        }
        self.retakes.stage = Some(Stage::Review(r));
    }

    fn go_to(&mut self, item: ItemId, start: Tick) -> bool {
        match self.editor.timeline_ranges_of(item, start, Tick(start.0.saturating_add(1))).first() {
            Some((at, _)) => {
                self.queue(Action::SeekTimeline(*at));
                true
            }
            None => {
                self.info_text("That part of the video is not on the timeline.");
                false
            }
        }
    }

    fn apply_retakes(&mut self) {
        let Some(Stage::Review(r)) = &self.retakes.stage else { return };
        let cuts: Vec<CutProposal> = self.cuts(r).into_iter().map(Sentence::as_cut).collect();
        let item = r.item;
        match self.editor.apply_cut_sets(&[(item, cuts)], CutScope::MainTracks) {
            Ok(report) => {
                self.source = Source::Timeline;
                let secs = report.removed.0 as f64 / 254_016_000_000.0;
                let mut text = format!("Removed {} piece{} ({secs:.1} s). Ctrl+Z puts them back.", report.cuts, if report.cuts == 1 { "" } else { "s" });
                if let Some(n) = report.notes.first() {
                    text.push(' ');
                    text.push_str(n);
                }
                self.info_text(text);
                self.player.set_skips(Vec::new());
            }
            Err(e) => self.fail_text(&e.to_string()),
        }
    }

    fn set_sentence_span(&mut self, sentence: &str, start: f64, end: f64) {
        let Some(Stage::Review(r)) = &mut self.retakes.stage else { return };
        let Some(entry) = self.editor.media_entry(r.item) else { return };
        let media_duration = entry.duration.0 as f64 / TICKS_PER_SECOND;
        let Some(group) = r.groups.iter_mut().find(|g| g.sentences.iter().any(|s| s.id == sentence)) else { return };
        let Some(sent) = group.sentences.iter().find(|s| s.id == sentence) else { return };
        let old = (sent.start.0 as f64 / TICKS_PER_SECOND, sent.end.0 as f64 / TICKS_PER_SECOND);
        if (old.0 - start).abs() < 1e-4 && (old.1 - end).abs() < 1e-4 {
            return;
        }
        if let Err(e) = group.set_span(sentence, start, end, media_duration) {
            let msg = e.to_string();
            self.fail_text(&msg);
            return;
        }
        // log what was stored (it may have been clamped to a neighbour), and only edits that happened
        let Some(sent) = group.sentences.iter().find(|s| s.id == sentence) else { return };
        let new = (sent.start.0 as f64 / TICKS_PER_SECOND, sent.end.0 as f64 / TICKS_PER_SECOND);
        if new != old
            && let Some(path) = self.retakes.span_log.clone()
            && let Err(e) = append_span_edit(&path, &r.job, sentence, old, new)
        {
            log::warn!("could not write {}: {e}", path.display());
        }
    }

    fn reset_sentence_span(&mut self, sentence: &str) {
        let Some(Stage::Review(r)) = &mut self.retakes.stage else { return };
        let Some(orig) = r.original.get(sentence).copied() else { return };
        let Some(entry) = self.editor.media_entry(r.item) else { return };
        let media_duration = entry.duration.0 as f64 / TICKS_PER_SECOND;
        let Some(group) = r.groups.iter_mut().find(|g| g.sentences.iter().any(|s| s.id == sentence)) else { return };
        let Some(sent) = group.sentences.iter().find(|s| s.id == sentence) else { return };
        let old = (sent.start.0 as f64 / TICKS_PER_SECOND, sent.end.0 as f64 / TICKS_PER_SECOND);
        let orig_sec = (orig.0.0 as f64 / TICKS_PER_SECOND, orig.1.0 as f64 / TICKS_PER_SECOND);
        if let Err(e) = group.set_span(sentence, orig_sec.0, orig_sec.1, media_duration) {
            let msg = e.to_string();
            self.fail_text(&msg);
            return;
        }
        if let Some(sent) = group.sentences.iter_mut().find(|s| s.id == sentence) {
            sent.start = orig.0;
            sent.end = orig.1;
        }
        if let (Some(min_s), Some(max_e)) = (group.sentences.iter().map(|s| s.start).min(), group.sentences.iter().map(|s| s.end).max()) {
            group.start = min_s;
            group.end = max_e;
        }
        if old != orig_sec
            && let Some(path) = self.retakes.span_log.clone()
            && let Err(e) = append_span_edit(&path, &r.job, sentence, old, orig_sec)
        {
            log::warn!("could not write {}: {e}", path.display());
        }
    }

    fn save_groups(&mut self) {
        let Some(Stage::Review(r)) = &self.retakes.stage else { return };
        let Some(entry) = self.editor.media_entry(r.item) else { return };
        let suggested = if r.job.is_empty() { "groups.json".to_string() } else { format!("{}-groups.json", r.job) };
        let Some(path) = self.dialogs.pick_groups_save_path(&suggested) else { return };
        let duration_seconds = entry.duration.0 as f64 / 254_016_000_000.0;
        let media_path = entry.path.as_deref().unwrap_or(&entry.name);
        let json = simplecut_core::review::groups_to_json(&r.job, media_path, duration_seconds, &r.groups);
        if let Err(e) = std::fs::write(&path, json) {
            self.fail_text(&format!("Could not save groups to {}: {e}", path.display()));
        } else {
            self.info_text(format!("Saved groups to {}.", path.display()));
        }
    }
}

/// Add the user's answers to the local log, one JSON object per line, before they are sent.
fn append_labels(path: &Path, q: &Questions) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let secs = |t: Tick| t.0 as f64 / 254_016_000_000.0;
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    for (pair, verdict) in &q.answers {
        let Some(p) = q.candidates.pairs.iter().find(|p| &p.id == pair) else { continue };
        let side = |id: &str| {
            q.candidates
                .sentences
                .iter()
                .find(|s| s.id == id)
                .map(|s| serde_json::json!({"id": s.id, "text": s.text, "start": secs(s.start), "end": secs(s.end)}))
        };
        let line = serde_json::json!({"time": now, "job": q.candidates.job, "pair": pair, "label": verdict, "source": "user", "a": side(&p.a), "b": side(&p.b), "score": p.score, "lean": p.lean});
        writeln!(file, "{line}")?;
    }
    Ok(())
}

/// Add a sentence span adjustment to the local log, one JSON object per line, before it is applied.
fn append_span_edit(path: &Path, job: &str, sentence: &str, old: (f64, f64), new: (f64, f64)) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let line = serde_json::json!({
        "time": now,
        "job": job,
        "sentence": sentence,
        "old": { "start": old.0, "end": old.1 },
        "new": { "start": new.0, "end": new.1 },
        "source": "user",
    });
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{line}")?;
    Ok(())
}

// ── the panel ──────────────────────────────────────────────────────────────

pub(crate) fn panel(app: &mut SimpleCutApp, ui: &mut egui::Ui) {
    let (title, closable) = match app.retakes.stage {
        Some(Stage::Questions(_)) => ("Retakes · Step 1: check uncertain pairs", true),
        Some(Stage::Analyzing(_)) => ("Retakes · looking…", false),
        Some(Stage::Resolving { .. }) => ("Retakes · Step 1: check uncertain pairs", false),
        Some(Stage::Review(_)) => ("Retakes · Step 2: choose what to keep", true),
        None => return,
    };
    ui.horizontal(|ui| {
        ui.heading(RichText::new(title).size(15.0));
        if closable {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("Close").on_hover_text("Close the Retakes panel without cutting anything").clicked() {
                    app.queue(Action::CloseReview);
                }
            });
        }
    });
    ui.separator();
    let fps = app.editor.sequence().map_or(30.0, |s| s.fps);
    match &app.retakes.stage {
        Some(Stage::Analyzing(run)) => {
            let (s, f) = (run.status.clone(), run.fraction);
            progress(app, ui, &s, f, "Cancel analysis");
        }
        Some(Stage::Resolving { run, .. }) => {
            let (s, f) = (run.status.clone(), run.fraction);
            progress(app, ui, &s, f, "Cancel checking");
        }
        Some(Stage::Questions(_)) => questions_panel(app, ui, fps),
        Some(Stage::Review(_)) => review_panel(app, ui, fps),
        None => {}
    }
}

fn progress(app: &mut SimpleCutApp, ui: &mut egui::Ui, status: &str, fraction: f32, cancel: &str) {
    let t = theme::tokens();
    ui.label(RichText::new(status).color(t.text));
    ui.add(egui::ProgressBar::new(fraction.clamp(0.0, 1.0)).desired_width(ui.available_width()));
    ui.add_space(6.0);
    if ui.button(cancel).clicked() {
        app.queue(Action::CancelAnalysis);
    }
}

struct QuestionView {
    count: usize,
    index: usize,
    answered: usize,
    skipped: usize,
    pair_id: String,
    lean: Option<Verdict>,
    a: Sentence,
    b: Sentence,
    verdict: Option<Verdict>,
    was_skipped: bool,
    item: ItemId,
    confirm: bool,
    unanswered: usize,
}

fn question_view(app: &SimpleCutApp) -> Option<QuestionView> {
    let Some(Stage::Questions(q)) = &app.retakes.stage else { return None };
    let pair_id = q.order.get(q.index)?.clone();
    let pair = q.candidates.pairs.iter().find(|p| p.id == pair_id && p.class == PairClass::Ambiguous)?;
    let find = |id: &str| q.candidates.sentences.iter().find(|s| s.id == id).cloned();
    Some(QuestionView {
        count: q.order.len(),
        index: q.index,
        answered: q.answers.len(),
        skipped: q.skipped.len(),
        lean: pair.lean,
        a: find(&pair.a)?,
        b: find(&pair.b)?,
        verdict: q.answers.get(&pair_id).copied(),
        was_skipped: q.skipped.contains(&pair_id),
        pair_id,
        item: q.item,
        confirm: q.confirm,
        unanswered: q.unanswered(),
    })
}

fn sentence_block(app: &mut SimpleCutApp, ui: &mut egui::Ui, item: ItemId, s: &Sentence, fps: f64, label: &str) {
    let t = theme::tokens();
    egui::Frame::new().fill(t.panel_raised).corner_radius(CornerRadius::same(6)).inner_margin(8).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            ui.label(RichText::new(label).color(t.text_dim).small());
            ui.label(RichText::new(format!("{} – {}", format::timecode(s.start, fps), format::timecode(s.end, fps))).monospace().color(t.text_dim));
        });
        ui.label(RichText::new(format!("“{}”", s.text)).color(t.text).size(15.0));
        ui.horizontal(|ui| {
            let play = ui.small_button("Play").on_hover_text("Play this sentence from the timeline");
            play.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, format!("Play {}", s.id)));
            if play.clicked() {
                app.queue(Action::PlaySentence(item, s.start));
            }
            let go = ui.small_button("Go to").on_hover_text("Move the playhead here");
            go.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, format!("Go to {}", s.id)));
            if go.clicked() {
                app.queue(Action::GoToSentence(item, s.start));
            }
        });
    });
}

fn questions_panel(app: &mut SimpleCutApp, ui: &mut egui::Ui, fps: f64) {
    let t = theme::tokens();
    let Some(v) = question_view(app) else {
        ui.label("There is nothing to ask.");
        return;
    };
    ui.label(RichText::new(format!("Question {} of {} · {} answered · {} skipped", v.index + 1, v.count, v.answered, v.skipped)).color(t.text_dim));
    ui.label(RichText::new("Are these two the same line, said again?").color(t.text).strong());
    ui.add_space(4.0);
    sentence_block(app, ui, v.item, &v.a, fps, "Earlier");
    ui.add_space(4.0);
    sentence_block(app, ui, v.item, &v.b, fps, "Later");
    ui.add_space(2.0);
    match v.lean {
        Some(Verdict::Retake) => ui.label(RichText::new("The model leans: same line.").color(t.text_dim).small()),
        Some(Verdict::Different) => ui.label(RichText::new("The model leans: different lines.").color(t.text_dim).small()),
        None => ui.label(RichText::new("The model has no lean on this one.").color(t.text_dim).small()),
    };
    ui.add_space(4.0);
    ui.horizontal_wrapped(|ui| {
        let same = ui.add(egui::Button::new("Same line").selected(v.verdict == Some(Verdict::Retake))).on_hover_text("These are attempts at the same line");
        if same.clicked() {
            app.queue(Action::AnswerPair { pair: v.pair_id.clone(), verdict: Some(Verdict::Retake) });
        }
        let diff = ui.add(egui::Button::new("Different lines").selected(v.verdict == Some(Verdict::Different))).on_hover_text("These are two different lines");
        if diff.clicked() {
            app.queue(Action::AnswerPair { pair: v.pair_id.clone(), verdict: Some(Verdict::Different) });
        }
        let skip = ui.add(egui::Button::new("Skip").selected(v.was_skipped)).on_hover_text("Let the model decide");
        if skip.clicked() {
            app.queue(Action::AnswerPair { pair: v.pair_id.clone(), verdict: None });
        }
    });
    ui.horizontal(|ui| {
        if ui.add_enabled(v.index > 0, egui::Button::new("Previous question")).clicked() {
            app.queue(Action::SetQuestion(v.index.saturating_sub(1)));
        }
        if ui.add_enabled(v.index + 1 < v.count, egui::Button::new("Next question")).clicked() {
            app.queue(Action::SetQuestion(v.index + 1));
        }
    });
    ui.add_space(8.0);
    ui.separator();
    if v.confirm {
        let (is, them) = if v.unanswered == 1 { (" is", "it") } else { ("s are", "them") };
        ui.label(RichText::new(format!("{} question{is} unanswered. The model will decide {them}.", v.unanswered)).color(t.text));
        ui.horizontal(|ui| {
            if ui.button("Send anyway").clicked() {
                app.queue(Action::QuestionsSendAnyway);
            }
            if ui.button("Keep answering").clicked() {
                app.queue(Action::SetQuestion(v.index));
            }
        });
    } else if ui
        .add(egui::Button::new(RichText::new("Next").strong()).min_size(egui::vec2(ui.available_width(), 30.0)))
        .on_hover_text("Send your answers to the model and make the groups")
        .clicked()
    {
        app.queue(Action::QuestionsNext);
    }
}

struct SentenceRow {
    sentence: Sentence,
    kept: bool,
    on: bool,
    origin: Option<Origin>,
    edited: bool,
}

struct GroupView {
    id: String,
    label: String,
    color_index: usize,
    likely: bool,
    start: Tick,
    end: Tick,
    rows: Vec<SentenceRow>,
    reason: Option<String>,
    seen_off: bool,
}

fn review_panel(app: &mut SimpleCutApp, ui: &mut egui::Ui, fps: f64) {
    let t = theme::tokens();
    let ai = app.ai_available();
    let (item, groups, pending, advice_running, notes, can_redo, ignored) = {
        let Some(Stage::Review(r)) = &app.retakes.stage else { return };
        let groups: Vec<GroupView> = r
            .groups
            .iter()
            .enumerate()
            .map(|(i, g)| GroupView {
                id: g.id.clone(),
                label: label_of(i),
                color_index: i,
                likely: g.likely,
                start: g.start,
                end: g.end,
                rows: g
                    .sentences
                    .iter()
                    .map(|s| {
                        let edited = match r.original.get(&s.id) {
                            Some(orig) => (s.start.0 - orig.0.0).abs() > 1000 || (s.end.0 - orig.1.0).abs() > 1000,
                            None => false,
                        };
                        SentenceRow { sentence: s.clone(), kept: r.is_kept(&s.id), on: app.on(r.item, s), origin: r.origin(&s.id), edited }
                    })
                    .collect(),
                reason: r.reasons.get(&g.id).cloned(),
                seen_off: g.sentences.iter().any(|s| r.seen_on.contains(&s.id) && !app.on(r.item, s)),
            })
            .collect();
        (r.item, groups, r.ai_consent_pending, r.advice.as_ref().map(|a| (a.status.clone(), a.fraction)), r.notes.clone(), r.from.is_some(), r.rejected.len())
    };
    let live = |g: &GroupView| g.rows.iter().filter(|r| r.on).count() >= 2;
    let cut_count: usize = groups.iter().filter(|g| live(g)).map(|g| g.rows.iter().filter(|r| r.on && !r.kept).count()).sum();
    let group_count = groups.iter().filter(|g| live(g) && g.rows.iter().any(|r| r.on && !r.kept)).count();

    ui.horizontal_wrapped(|ui| {
        if ai
            && ui
                .add_enabled(advice_running.is_none(), egui::Button::new("Ask AI to choose"))
                .on_hover_text("Ask the AI which sentences to keep. It only fills in the ticks; nothing is cut until you press Remove.")
                .clicked()
        {
            app.queue(Action::AskAi);
        }
        if ui.small_button("Keep last in all likely groups").on_hover_text("Tick only the last sentence of every likely group").clicked() {
            app.queue(Action::KeepLastInLikely);
        }
        if can_redo && ui.small_button("Redo the questions").on_hover_text("Go back and change your answers").clicked() {
            app.queue(Action::RedoQuestions);
        }
        if ui.small_button("Save groups...").on_hover_text("Save the groups document with edited spans").clicked() {
            app.queue(Action::SaveGroups);
        }
    });

    let mut skip_unticked = app.review_skips_unticked();
    if ui
        .checkbox(&mut skip_unticked, "Skip unticked sentences when playing")
        .on_hover_text(
            "When checked, playback skips over sentences marked for removal. Music is not cut by Remove, so the preview is approximate for the music bed.",
        )
        .changed()
    {
        app.queue(Action::SetSkipUnticked(skip_unticked));
    }
    let mut hs = app.highlight_strength();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Highlight strength").small().color(t.text_dim));
        let r = ui.add(egui::Slider::new(&mut hs, 0.1..=0.8));
        if r.changed() {
            app.set_highlight_strength(hs);
        }
    });
    if pending {
        egui::Frame::new().fill(t.panel_raised).corner_radius(CornerRadius::same(6)).inner_margin(8).show(ui, |ui| {
            ui.label("The sentences of the groups you ask about will be sent to the AI service. Continue?");
            ui.horizontal(|ui| {
                if ui.button("Continue").clicked() {
                    app.queue(Action::ConfirmAi);
                }
                if ui.button("Not now").clicked() {
                    app.queue(Action::DeclineAi);
                }
            });
        });
    }
    if let Some((status, fraction)) = &advice_running {
        ui.label(RichText::new(status).color(t.text));
        ui.add(egui::ProgressBar::new(fraction.clamp(0.0, 1.0)).desired_width(ui.available_width()));
        if ui.button("Cancel AI request").clicked() {
            app.queue(Action::CancelAnalysis);
        }
    }
    for n in &notes {
        ui.label(RichText::new(n).color(t.danger).small());
    }
    ui.label(
        RichText::new(format!(
            "{cut_count} sentence{} will be removed from {group_count} group{}",
            if cut_count == 1 { "" } else { "s" },
            if group_count == 1 { "" } else { "s" }
        ))
        .color(t.text_dim),
    );
    ui.add_space(2.0);

    let list_height = (ui.available_height() - 52.0).max(80.0);
    egui::ScrollArea::vertical().max_height(list_height).auto_shrink([false, false]).show(ui, |ui| {
        let likely: Vec<&GroupView> = groups.iter().filter(|g| live(g) && g.likely).collect();
        let possible: Vec<&GroupView> = groups.iter().filter(|g| live(g) && !g.likely).collect();
        let handled: Vec<&GroupView> = groups.iter().filter(|g| !live(g)).collect();
        if likely.is_empty() && possible.is_empty() {
            ui.label(RichText::new("Nothing to choose here. Everything is already handled; Ctrl+Z brings removed sentences back.").color(t.text_dim));
        }
        if !likely.is_empty() {
            ui.label(RichText::new(format!("Likely retakes ({})", likely.len())).strong().color(t.text));
            for g in likely {
                group_card(app, ui, item, g, fps, true);
            }
        }
        if !possible.is_empty() {
            ui.add_space(6.0);
            egui::CollapsingHeader::new(RichText::new(format!("Possible retakes ({})", possible.len())).strong().color(t.text))
                .id_salt("possible-retakes")
                .default_open(false)
                .show(ui, |ui| {
                    for g in possible {
                        group_card(app, ui, item, g, fps, false);
                    }
                });
        }
        if !handled.is_empty() {
            ui.add_space(6.0);
            egui::CollapsingHeader::new(RichText::new(format!("Already handled or not on the timeline ({})", handled.len())).color(t.text_dim))
                .id_salt("handled-retakes")
                .default_open(false)
                .show(ui, |ui| {
                    for g in handled {
                        let why = if g.seen_off { "already removed" } else { "outside the part of the video you used" };
                        ui.label(RichText::new(format!("{} · {} sentences · {}", g.label, g.rows.len(), why)).color(t.text_dim).small());
                    }
                });
        }
        if ignored > 0 {
            ui.add_space(4.0);
            ui.label(RichText::new(format!("{ignored} unusable entr{} ignored.", if ignored == 1 { "y was" } else { "ies were" })).color(t.text_dim).small());
        }
    });

    ui.add_space(6.0);
    let label = format!("Remove {cut_count} sentence{}", if cut_count == 1 { "" } else { "s" });
    if ui
        .add_enabled(cut_count > 0, egui::Button::new(RichText::new(label).strong()).min_size(egui::vec2(ui.available_width(), 30.0)))
        .on_hover_text("Cut every sentence that is not ticked as kept (one undo step)")
        .clicked()
    {
        app.queue(Action::ApplyRetakes);
    }
}

fn group_card(app: &mut SimpleCutApp, ui: &mut egui::Ui, item: ItemId, g: &GroupView, fps: f64, open: bool) {
    let t = theme::tokens();
    let color = colour_of(g.color_index);
    let on_rows = g.rows.iter().filter(|r| r.on).count();
    let any_kept = g.rows.iter().any(|r| r.on && r.kept);
    let ai = g.rows.iter().any(|r| r.origin == Some(Origin::Ai));

    let id = ui.make_persistent_id(("retake-group", &g.id));
    let state = egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, open);
    let res = state.show_header(ui, |ui| {
        ui.horizontal(|ui| {
            let title = format!(
                "●  {}  ·  {} sentences  ·  {} – {}{}",
                g.label,
                on_rows,
                format::timecode(g.start, fps),
                format::timecode(g.end, fps),
                if ai { "  ·  AI" } else { "" }
            );
            ui.label(RichText::new(title).color(color).strong());
            if !any_kept && on_rows > 0 {
                ui.label(RichText::new("all removed").color(t.danger).small().strong());
            }
        });
    });
    let (toggle_btn, _, _) = res.body(|ui| {
        if let Some(reason) = &g.reason {
            ui.label(RichText::new(format!("AI: {reason}")).color(t.text_dim).small());
        }
        for row in &g.rows {
            let s = &row.sentence;
            egui::Frame::new().fill(t.panel_raised).corner_radius(CornerRadius::same(6)).inner_margin(8).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    let mut kept = row.kept;
                    let text = format!("{} – {}", format::timecode(s.start, fps), format::timecode(s.end, fps));
                    let cb = ui.add_enabled(row.on, egui::Checkbox::new(&mut kept, text));
                    cb.widget_info(|| WidgetInfo::selected(WidgetType::Checkbox, row.on, row.kept, format!("Keep {}", s.id)));
                    if cb.clicked() {
                        app.queue(Action::SetKeep { sentence: s.id.clone(), keep: !row.kept });
                    }
                    if !row.on {
                        ui.label(RichText::new("not on the timeline").color(t.text_dim).small());
                    } else if !row.kept {
                        ui.label(RichText::new("will be removed").color(t.danger).small());
                    }
                    if row.origin == Some(Origin::Ai) {
                        ui.label(RichText::new("AI").color(color).small());
                    }
                });
                ui.label(RichText::new(format!("“{}”", s.text)).color(if row.on { t.text } else { t.text_dim }));
                ui.horizontal(|ui| {
                    let play = ui.small_button("Play");
                    play.widget_info(|| WidgetInfo::labeled(WidgetType::Button, row.on, format!("Play {}", s.id)));
                    if play.clicked() {
                        app.queue(Action::PlaySentence(item, s.start));
                    }
                    let go = ui.small_button("Go to");
                    go.widget_info(|| WidgetInfo::labeled(WidgetType::Button, row.on, format!("Go to {}", s.id)));
                    if go.clicked() {
                        app.queue(Action::GoToSentence(item, s.start));
                    }
                });
                ui.horizontal(|ui| {
                    let mut start_sec = s.start.0 as f64 / 254_016_000_000.0;
                    let mut end_sec = s.end.0 as f64 / 254_016_000_000.0;

                    let b_sm = ui.add(egui::Button::new("−").min_size(vec2(18.0, 0.0)));
                    b_sm.widget_info(|| WidgetInfo::labeled(WidgetType::Button, row.on, format!("Nudge start of {} earlier", s.id)));
                    if b_sm.clicked() {
                        app.queue(Action::SetSentenceSpan { sentence: s.id.clone(), start: start_sec - 0.05, end: end_sec });
                    }
                    let r_start = ui.add(egui::DragValue::new(&mut start_sec).speed(0.01).max_decimals(2).min_decimals(2).suffix(" s"));
                    r_start.widget_info(|| WidgetInfo::labeled(WidgetType::DragValue, true, format!("Start of {}", s.id)));
                    if r_start.changed() {
                        app.queue(Action::SetSentenceSpan { sentence: s.id.clone(), start: start_sec, end: end_sec });
                    }
                    let b_sp = ui.add(egui::Button::new("+").min_size(vec2(18.0, 0.0)));
                    b_sp.widget_info(|| WidgetInfo::labeled(WidgetType::Button, row.on, format!("Nudge start of {} later", s.id)));
                    if b_sp.clicked() {
                        app.queue(Action::SetSentenceSpan { sentence: s.id.clone(), start: start_sec + 0.05, end: end_sec });
                    }

                    ui.label(RichText::new("..").color(t.text_dim));

                    let b_em = ui.add(egui::Button::new("−").min_size(vec2(18.0, 0.0)));
                    b_em.widget_info(|| WidgetInfo::labeled(WidgetType::Button, row.on, format!("Nudge end of {} earlier", s.id)));
                    if b_em.clicked() {
                        app.queue(Action::SetSentenceSpan { sentence: s.id.clone(), start: start_sec, end: end_sec - 0.05 });
                    }
                    let r_end = ui.add(egui::DragValue::new(&mut end_sec).speed(0.01).max_decimals(2).min_decimals(2).suffix(" s"));
                    r_end.widget_info(|| WidgetInfo::labeled(WidgetType::DragValue, true, format!("End of {}", s.id)));
                    if r_end.changed() {
                        app.queue(Action::SetSentenceSpan { sentence: s.id.clone(), start: start_sec, end: end_sec });
                    }
                    let b_ep = ui.add(egui::Button::new("+").min_size(vec2(18.0, 0.0)));
                    b_ep.widget_info(|| WidgetInfo::labeled(WidgetType::Button, row.on, format!("Nudge end of {} later", s.id)));
                    if b_ep.clicked() {
                        app.queue(Action::SetSentenceSpan { sentence: s.id.clone(), start: start_sec, end: end_sec + 0.05 });
                    }

                    if row.edited {
                        ui.label(RichText::new("edited").color(t.accent).small().strong());
                        let reset_btn = ui.small_button("Reset span");
                        reset_btn.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, format!("Reset span of {}", s.id)));
                        if reset_btn.clicked() {
                            app.queue(Action::ResetSentenceSpan(s.id.clone()));
                        }
                    }
                });
            });
            ui.add_space(3.0);
        }
        ui.horizontal(|ui| {
            let last = ui.small_button("Keep last only");
            last.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, format!("Keep last only in {}", g.label)));
            if last.clicked() {
                app.queue(Action::KeepLastOnly(g.id.clone()));
            }
            let all = ui.small_button("Keep all");
            all.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, format!("Keep all in {}", g.label)));
            if all.clicked() {
                app.queue(Action::KeepAllOf(g.id.clone()));
            }
            let remove_all = ui.small_button("Remove all");
            remove_all.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, format!("Remove all in {}", g.label)));
            if remove_all.clicked() {
                app.queue(Action::RemoveAllOf(g.id.clone()));
            }
        });
    });
    toggle_btn.widget_info(|| WidgetInfo::labeled(WidgetType::CollapsingHeader, true, format!("Group {}", g.label)));
    ui.add_space(4.0);
}
