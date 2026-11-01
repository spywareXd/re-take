//! The progress stepper and log of the Retakes panel: the current stage and elapsed time while a request runs, "cached" stages, the exact message of a failed stage,
//! and the log kept after the run. A scripted service wraps `FakeService` and injects stage events.
//!
//! Skipped (with a message) when ffmpeg / ffprobe are not installed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use common::harness::{Script, Window, click, item, skip, step_until};
use common::raster::{Picture, SoftRenderer};
use common::{make_clip, scratch};
use egui::vec2;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use retake_editor_core::{AnalysisJob, AnalysisRequest, FakeService, JobState, Lane, ResolveRequest, RetakeService, StageEvent, progress::STAGES};
use retake_editor_ui::{Action, RetakeEditorApp};
use serde_json::json;

fn ev(seq: u64, stage: &str, status: &str, pct: Option<f64>, message: &str, cached: bool) -> StageEvent {
    let mut v = json!({"seq": seq, "stage": stage, "status": status, "message": message, "timestamp": "2026-10-29T22:35:55.591+00:00", "cached": cached});
    if let Some(p) = pct {
        v["pct"] = json!(p);
    }
    StageEvent::from_json(&v).unwrap()
}

#[derive(Clone)]
enum Ending {
    /// let the wrapped FakeService finish normally
    Inner,
    /// stay running after the scripted events
    Hold,
    /// fail with the service's sentence once the events are delivered
    Fail(String),
}

struct ScriptedJob {
    inner: Box<dyn AnalysisJob>,
    chunks: Vec<Vec<StageEvent>>,
    next: usize,
    ending: Ending,
}

impl AnalysisJob for ScriptedJob {
    fn poll(&mut self) -> JobState {
        if self.next < self.chunks.len() {
            return JobState::Running { fraction: 0.3, status: "Working".into() };
        }
        match &self.ending {
            Ending::Inner => self.inner.poll(),
            Ending::Hold => JobState::Running { fraction: 0.5, status: "Transcribing".into() },
            Ending::Fail(why) => JobState::Failed(why.clone()),
        }
    }

    fn cancel(&mut self) {
        self.inner.cancel();
    }

    fn events(&mut self) -> Vec<StageEvent> {
        let out = self.chunks.get(self.next).cloned().unwrap_or_default();
        self.next += 1;
        out
    }
}

struct ScriptedService {
    fake: FakeService,
    chunks: Vec<Vec<StageEvent>>,
    ending: Ending,
    analyses: Arc<Mutex<u32>>,
}

impl RetakeService for ScriptedService {
    fn name(&self) -> String {
        "scripted".into()
    }

    fn analyze(&self, request: AnalysisRequest) -> Box<dyn AnalysisJob> {
        *self.analyses.lock().unwrap() += 1;
        Box::new(ScriptedJob { inner: self.fake.analyze(request), chunks: self.chunks.clone(), next: 0, ending: self.ending.clone() })
    }

    fn resolve(&self, request: ResolveRequest) -> Box<dyn AnalysisJob> {
        self.fake.resolve(request)
    }
}

fn window(renderer: SoftRenderer, service: ScriptedService) -> (Window, PathBuf) {
    let dir = scratch("progress");
    let clip = make_clip(&dir, "talk.mp4", 4);
    let script = Script::default();
    script.media.lock().unwrap().push_back(vec![clip]);
    let mut h = Harness::builder().with_size(vec2(1360.0, 860.0)).with_max_steps(10_000).renderer(renderer).build_eframe(move |cc| {
        let mut app = RetakeEditorApp::new(&cc.egui_ctx, Box::new(script), None);
        app.set_service(Some(Box::new(service)));
        app
    });
    h.step();
    click(&mut h, "Import");
    step_until(&mut h, "the import", |a| a.editor().media().len() == 1);
    let video = item(&h, "talk.mp4");
    h.state_mut().perform(Action::AddToTimeline(video, Lane::Main));
    h.step();
    h.step();
    (h, dir)
}

fn running_chunks() -> Vec<Vec<StageEvent>> {
    vec![
        vec![ev(1, "audio_extraction", "started", None, "16 kHz mono WAV", false), ev(2, "audio_extraction", "done", Some(100.0), "", false)],
        vec![ev(3, "model_download", "done", Some(100.0), "cached", true), ev(4, "model_load", "started", None, "large-v2 on cuda", false)],
        vec![ev(5, "model_load", "done", Some(100.0), "compute type int8_float16", false), ev(6, "vad", "done", Some(100.0), "", false)],
        vec![ev(7, "transcription", "started", None, "Model large-v2", false), ev(8, "transcription", "progress", Some(42.0), "", false)],
    ]
}

fn start(h: &mut Window, steps: usize) {
    click(h, "Remove Retakes");
    for _ in 0..steps {
        h.step();
        h.step();
    }
}

fn svc(chunks: Vec<Vec<StageEvent>>, ending: Ending) -> (ScriptedService, Arc<Mutex<u32>>) {
    let n = Arc::new(Mutex::new(0));
    (ScriptedService { fake: FakeService { skip_questions: false, ..Default::default() }, chunks, ending, analyses: Arc::clone(&n) }, n)
}

#[test]
fn while_running_the_stepper_shows_every_stage_the_current_one_cached_stages_and_the_log() {
    if skip() {
        return;
    }
    let (service, _) = svc(running_chunks(), Ending::Hold);
    let (mut h, _d) = window(SoftRenderer::default(), service);
    start(&mut h, 6);
    assert_eq!(h.state().review_stage(), "analyzing");
    h.get_by_label("Transcribe · 42%"); // the current stage and its percent
    for (_, label) in STAGES {
        assert!(h.query_by_label(label).is_some(), "the stepper lists '{label}' (every stage, in order, even before it runs)");
    }
    assert!(h.query_by_label("cached").is_some(), "a stage that was skipped reads 'cached'");
    assert!(h.query_by_label_contains("Load model: done — compute type int8_float16").is_some(), "the log has the stage messages");
    assert!(h.query_by_label_contains("Transcribe: 42%").is_some());
    let labels: Vec<f32> = STAGES.iter().map(|(_, l)| h.get_by_label(l).rect().top()).collect();
    assert!(labels.windows(2).all(|w| w[0] < w[1]), "stages are listed top to bottom in pipeline order: {labels:?}");
}

#[test]
fn a_failed_stage_shows_its_exact_message_and_the_log_stays_after_the_run() {
    if skip() {
        return;
    }
    let mut chunks = running_chunks();
    chunks.push(vec![ev(9, "transcription", "error", None, "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB", false)]);
    let (service, _) = svc(chunks, Ending::Fail("Something went wrong while processing the file (job j_1). See the service log.".into()));
    let (mut h, _d) = window(SoftRenderer::default(), service);
    start(&mut h, 8);
    assert_eq!(h.state().review_stage(), "none", "the request ended");
    // the panel stays open on the last run
    h.get_by_label("Retakes · last run");
    h.get_by_label("Failed at: Transcribe");
    h.get_by_label("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"); // exactly as the service sent it
    assert!(h.query_by_label_contains("Transcribe: ERROR").is_some(), "and it is in the log");
    let banner = h.state().banner().map(|b| b.text.clone()).expect("the error banner");
    assert!(banner.contains("Something went wrong while processing the file"), "{banner}");
    assert!(
        banner.contains("(Transcribe: RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB)"),
        "the banner names the stage and its exact message: {banner}"
    );
    click(&mut h, "Close");
    assert!(h.query_by_label("Retakes · last run").is_none(), "Close hides the log");
}

#[test]
fn the_log_is_kept_after_a_successful_run() {
    if skip() {
        return;
    }
    let (service, analyses) = svc(running_chunks(), Ending::Inner);
    let (mut h, _d) = window(SoftRenderer::default(), service);
    start(&mut h, 4);
    step_until(&mut h, "the questions or the review", |a| matches!(a.review_stage(), "questions" | "review"));
    h.step();
    assert!(h.query_by_label_contains("Pipeline log").is_some(), "the finished run's log is folded under the panel");
    assert_eq!(*analyses.lock().unwrap(), 1);
}

#[test]
fn a_service_without_events_changes_nothing() {
    if skip() {
        return;
    }
    let (service, _) = svc(Vec::new(), Ending::Inner);
    let (mut h, _d) = window(SoftRenderer::default(), service);
    start(&mut h, 3);
    step_until(&mut h, "the questions or the review", |a| matches!(a.review_stage(), "questions" | "review"));
    assert!(h.query_by_label_contains("Pipeline log").is_none(), "no events, no log section");
}

fn shot(h: &Window, r: &SoftRenderer) -> Picture {
    r.draw(&h.ctx, h.output(), 1360.0, 860.0, egui::Color32::from_rgb(16, 16, 20))
}

/// Not a check: writes the panel in its three states (running, failed, finished) to <dir>/<tag>_<state>.png, cropped to the right-hand panel. Runs only when
/// RETAKE_EDITOR_SHOT_DIR and RETAKE_EDITOR_SHOT_TAG are set.
#[test]
fn write_progress_screenshots() {
    let (Some(dir), Ok(tag)) = (std::env::var_os("RETAKE_EDITOR_SHOT_DIR"), std::env::var("RETAKE_EDITOR_SHOT_TAG")) else {
        eprintln!("SKIPPED: set RETAKE_EDITOR_SHOT_DIR and RETAKE_EDITOR_SHOT_TAG to write screenshots");
        return;
    };
    if skip() {
        return;
    }
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let crop = |p: &Picture| p.crop_scaled(1020, 40, 340, 520, 2);

    let (service, _) = svc(running_chunks(), Ending::Hold);
    let r = SoftRenderer::default();
    let (mut h, _d) = window(r.clone(), service);
    start(&mut h, 6);
    crop(&shot(&h, &r)).save_png(&dir.join(format!("{tag}_1_running.png")));

    let mut chunks = running_chunks();
    chunks.push(vec![ev(9, "transcription", "error", None, "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB", false)]);
    let (service, _) = svc(chunks, Ending::Fail("Something went wrong while processing the file (job j_1). See the service log.".into()));
    let r = SoftRenderer::default();
    let (mut h, _d) = window(r.clone(), service);
    start(&mut h, 8);
    crop(&shot(&h, &r)).save_png(&dir.join(format!("{tag}_2_failed.png")));
}
