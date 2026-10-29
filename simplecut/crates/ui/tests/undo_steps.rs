//! Ctrl+Z / Ctrl+Y for the pipeline's own steps, and the Reset buttons: a step that created files is undone like an edit (set aside, never deleted, answers untouched);
//! the editor's own edits keep their turn; a step that found everything in the cache made nothing, so there is nothing to undo.
//!
//! Skipped (with a message) when ffmpeg / ffprobe are not installed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::{Arc, Mutex};

use common::harness::{Script, Window, click, item, skip, step_until, window_with};
use egui_kittest::kittest::Queryable;
use serde_json::json;
use simplecut_core::{
    AdjudicateRequest, AnalysisJob, AnalysisRequest, FakeService, JobState, KeeperInfo, KeeperProvider, Lane, ResolveRequest, RetakeService, StageEvent, Tick,
    Verdict, WorkCall, WorkKind,
};
use simplecut_ui::{Action, OptionsAction};

fn ev(seq: u64, stage: &str, status: &str, cached: bool) -> StageEvent {
    StageEvent::from_json(&json!({"seq": seq, "stage": stage, "status": status, "message": if cached { "cached" } else { "" }, "cached": cached, "timestamp": "2026-10-30T10:00:00.000+00:00"}))
        .unwrap()
}

struct EventedJob {
    inner: Box<dyn AnalysisJob>,
    events: Vec<StageEvent>,
}

impl AnalysisJob for EventedJob {
    fn poll(&mut self) -> JobState {
        self.inner.poll()
    }

    fn cancel(&mut self) {
        self.inner.cancel();
    }

    fn events(&mut self) -> Vec<StageEvent> {
        std::mem::take(&mut self.events)
    }
}

struct Done(Option<JobState>);

impl AnalysisJob for Done {
    fn poll(&mut self) -> JobState {
        self.0.take().unwrap_or(JobState::Cancelled)
    }

    fn cancel(&mut self) {}
}

struct StepService {
    fake: FakeService,
    /// Did the analysis find everything in the cache?
    cached: bool,
    work: Arc<Mutex<Vec<WorkCall>>>,
}

impl RetakeService for StepService {
    fn name(&self) -> String {
        "step test service".into()
    }

    fn capabilities(&self) -> simplecut_core::Capabilities {
        self.fake.capabilities()
    }

    fn analyze(&self, request: AnalysisRequest) -> Box<dyn AnalysisJob> {
        let events = ["transcription", "unit_splitting", "pair_scoring"]
            .iter()
            .enumerate()
            .flat_map(|(i, s)| {
                let n = 10 * i as u64;
                if self.cached { vec![ev(n + 1, s, "done", true)] } else { vec![ev(n + 1, s, "started", false), ev(n + 2, s, "done", false)] }
            })
            .collect();
        Box::new(EventedJob { inner: self.fake.analyze(request), events })
    }

    fn resolve(&self, request: ResolveRequest) -> Box<dyn AnalysisJob> {
        self.fake.resolve(request)
    }

    fn keeper(&self) -> Option<KeeperInfo> {
        Some(KeeperInfo {
            current: "finetuned".into(),
            default: "finetuned".into(),
            providers: vec![KeeperProvider { id: "finetuned".into(), available: true, reason: String::new() }],
        })
    }

    fn adjudicate(&self, request: AdjudicateRequest) -> Option<Box<dyn AnalysisJob>> {
        let reply: Vec<_> = request
            .groups
            .iter()
            .filter_map(|g| {
                g.sentences
                    .last()
                    .map(|last| json!({"group": g.id, "keep": [last.id], "reason": "r", "provider": "finetuned", "decided_by": "model", "unclear": false}))
            })
            .collect();
        Some(Box::new(Done(Some(JobState::Done(json!(reply).to_string())))))
    }

    fn work_call(&self, call: WorkCall) -> Option<Box<dyn AnalysisJob>> {
        self.work.lock().unwrap().push(call);
        Some(Box::new(Done(Some(JobState::Done(json!({"status": 200, "body": {"reset": ["transcript"]}}).to_string())))))
    }
}

fn start(cached: bool) -> (Window, Arc<Mutex<Vec<WorkCall>>>) {
    let dir = common::scratch("undo-steps");
    let script = Script::default();
    script.media.lock().unwrap().push_back(vec![common::make_clip(&dir, "talk.mp4", 4), common::make_music(&dir, "bed.wav", 6)]);
    let work = Arc::new(Mutex::new(Vec::new()));
    let service = StepService { fake: FakeService { adjudicate: true, ..Default::default() }, cached, work: work.clone() };
    let mut h = window_with(script, Some(Box::new(service)));
    click(&mut h, "Import");
    step_until(&mut h, "the import", |a| a.editor().media().len() == 2);
    let video = item(&h, "talk.mp4");
    h.state_mut().perform(Action::AddToTimeline(video, Lane::Main));
    h.state_mut().perform(Action::SelectClips { clips: vec![], add: false });
    h.step();
    h.step();
    (h, work)
}

/// Start, answer the way the documentation does, press Next and wait for the groups.
fn to_review(h: &mut Window) {
    click(h, "Remove Retakes");
    step_until(h, "the questions", |a| a.review_stage() == "questions");
    let ids = h.state().question_ids();
    h.state_mut().perform(Action::AnswerPair { pair: ids[0].clone(), verdict: Some(Verdict::Retake) });
    h.state_mut().perform(Action::AnswerPair { pair: ids[1].clone(), verdict: None });
    h.state_mut().perform(Action::AnswerPair { pair: ids[2].clone(), verdict: None });
    h.state_mut().perform(Action::AnswerPair { pair: ids[3].clone(), verdict: Some(Verdict::Different) });
    h.step();
    h.state_mut().perform(Action::QuestionsNext);
    h.step();
    h.state_mut().perform(Action::QuestionsSendAnyway);
    step_until(h, "the review", |a| a.review_stage() == "review");
}

fn depth(h: &Window) -> usize {
    h.state().editor().session().history.undo.len()
}

fn work(w: &Arc<Mutex<Vec<WorkCall>>>) -> Vec<(bool, WorkKind, bool)> {
    w.lock()
        .unwrap()
        .iter()
        .map(|c| match c {
            WorkCall::Reset { what, path } => (true, *what, path.ends_with("talk.mp4")),
            WorkCall::Restore { what, path } => (false, *what, path.ends_with("talk.mp4")),
        })
        .collect()
}

/// The Options window is centred and changes height as questions and notes appear: let it settle before aiming at a button.
fn settle_then_click(h: &mut Window, label: &str) {
    for _ in 0..3 {
        h.step();
    }
    click(h, label);
    for _ in 0..2 {
        h.step();
    }
}

fn undo(h: &mut Window) {
    h.state_mut().perform(Action::Undo);
    h.step();
    h.step();
}

fn redo(h: &mut Window) {
    h.state_mut().perform(Action::Redo);
    h.step();
    h.step();
}

#[test]
fn ctrl_z_after_a_transcription_sets_it_aside_and_ctrl_y_brings_it_back() {
    if skip() {
        return;
    }
    let (mut h, w) = start(false);
    to_review(&mut h);
    assert!(work(&w).is_empty(), "nothing is set aside by running the process");
    let editor_depth = depth(&h);
    undo(&mut h);
    assert_eq!(work(&w), vec![(true, WorkKind::Transcription, true)], "the transcription was set aside, naming the video");
    assert_eq!(h.state().review_stage(), "none", "the review of what no longer exists is closed");
    assert_eq!(depth(&h), editor_depth, "the editor's own undo stack was not touched");
    assert!(
        h.state().banner().is_some_and(|b| b.text.contains("Undid the transcription") && b.text.contains("nothing is deleted")),
        "{:?}",
        h.state().banner()
    );
    redo(&mut h);
    assert_eq!(work(&w).last(), Some(&(false, WorkKind::Transcription, true)), "Ctrl+Y asks the service to bring it back");
    assert!(h.state().banner().is_some_and(|b| b.text.contains("The transcription is back")));
}

#[test]
fn a_process_that_found_everything_in_the_cache_made_no_files_so_there_is_nothing_to_undo() {
    if skip() {
        return;
    }
    let (mut h, w) = start(true);
    to_review(&mut h);
    undo(&mut h);
    assert!(work(&w).is_empty(), "no files were made, so Ctrl+Z asks the service for nothing");
    assert_eq!(h.state().review_stage(), "review", "the review stays");
}

#[test]
fn an_edit_made_after_the_step_is_undone_first_then_the_step() {
    if skip() {
        return;
    }
    let (mut h, w) = start(false);
    to_review(&mut h);
    h.state_mut().perform(Action::SeekTimeline(Tick::from_seconds_f64(2.0)));
    click(&mut h, "Split");
    let after_edit = depth(&h);
    undo(&mut h);
    assert_eq!(depth(&h), after_edit - 1, "the first Ctrl+Z undid the split");
    assert!(work(&w).is_empty(), "and left the transcription alone");
    undo(&mut h);
    assert_eq!(work(&w), vec![(true, WorkKind::Transcription, true)], "the second one undid the transcription");
}

#[test]
fn the_ais_choices_are_undone_and_redone_like_a_step() {
    if skip() {
        return;
    }
    let (mut h, w) = start(false);
    to_review(&mut h);
    click(&mut h, "Ask AI to choose");
    step_until(&mut h, "the AI", |a| !a.is_analyzing() && a.banner().is_some_and(|b| b.text.contains("AI chose")));
    let tagged = |h: &Window| h.state().review_groups().iter().flat_map(|g| g.3.clone()).filter(|id| h.state().review_origin(id) == Some("ai")).count();
    let with_ai = tagged(&h);
    assert!(with_ai >= 2, "the AI filled in ticks");
    undo(&mut h);
    assert_eq!(tagged(&h), 0, "the ticks are back as they were");
    assert_eq!(h.state().review_stage(), "review", "the transcription is still there: only the AI's step was undone");
    assert_eq!(work(&w), vec![(true, WorkKind::KeeperGroups, true)]);
    redo(&mut h);
    assert_eq!(tagged(&h), with_ai, "Ctrl+Y puts the same choices back");
    assert_eq!(work(&w).last(), Some(&(false, WorkKind::KeeperGroups, true)));
}

#[test]
fn doing_something_new_after_an_undo_ends_the_redo() {
    if skip() {
        return;
    }
    let (mut h, w) = start(false);
    to_review(&mut h);
    click(&mut h, "Ask AI to choose");
    step_until(&mut h, "the AI", |a| !a.is_analyzing() && a.banner().is_some_and(|b| b.text.contains("AI chose")));
    undo(&mut h);
    click(&mut h, "Ask AI to choose");
    step_until(&mut h, "the AI again", |a| !a.is_analyzing() && a.banner().is_some_and(|b| b.text.contains("AI chose")));
    let before = work(&w).len();
    redo(&mut h);
    assert_eq!(work(&w).len(), before, "the new AI answer ended the redo chain: Ctrl+Y has nothing to redo");
}

#[test]
fn the_reset_buttons_ask_first_then_set_the_work_aside() {
    if skip() {
        return;
    }
    let (mut h, w) = start(false);
    to_review(&mut h);
    click(&mut h, "Ask AI to choose");
    step_until(&mut h, "the AI", |a| !a.is_analyzing() && a.banner().is_some_and(|b| b.text.contains("AI chose")));
    h.state_mut().perform(Action::Options(OptionsAction::Open));
    for _ in 0..4 {
        h.step();
    }
    h.get_by_label_contains("Nothing is deleted: the old results stay as numbered copies");
    // keeper groups: ask, cancel, ask, confirm
    settle_then_click(&mut h, "Reset keeper groups…");
    h.get_by_label_contains("Set aside the keeper groups (the AI's choices)?");
    assert!(work(&w).is_empty(), "asked, not done");
    settle_then_click(&mut h, "Cancel");
    assert!(work(&w).is_empty() && h.query_by_label_contains("Set aside the keeper groups").is_none());
    settle_then_click(&mut h, "Reset keeper groups…");
    settle_then_click(&mut h, "Reset");
    assert_eq!(work(&w), vec![(true, WorkKind::KeeperGroups, true)]);
    let ai_ticks = h.state().review_groups().iter().flat_map(|g| g.3.clone()).filter(|id| h.state().review_origin(id) == Some("ai")).count();
    assert_eq!(ai_ticks, 0, "the AI's ticks are cleared");
    assert_eq!(h.state().review_stage(), "review", "the transcription is untouched");
    // transcription
    settle_then_click(&mut h, "Reset transcription…");
    h.get_by_label_contains("Set aside the transcription (transcript, sentences and pair scores)?");
    settle_then_click(&mut h, "Reset");
    assert_eq!(work(&w).last(), Some(&(true, WorkKind::Transcription, true)));
    assert_eq!(h.state().review_stage(), "none", "the review of the old transcription is closed");
    h.get_by_label_contains("The transcription (transcript, sentences, pair scores) is set aside. The next Remove Retakes starts from scratch");
    undo(&mut h);
    assert_eq!(work(&w).len(), 2, "after a reset there is no step left to undo");
}
