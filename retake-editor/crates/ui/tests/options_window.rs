//! The Retake options window: the model dropdown (details, warnings, nothing downloaded to look), the silence slider with its sentence count, what happens to an
//! existing transcript when a setting changes (stale marks, the question, the confirmation), the project folder, the source-video warning and the legacy import.
//!
//! Skipped (with a message) when ffmpeg / ffprobe are not installed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use common::harness::{Script, Window, click, item, skip, step_until};
use common::raster::{Picture, SoftRenderer};
use common::{make_clip, scratch};
use egui::accesskit::Role;
use egui::vec2;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use retake_editor_core::{
    AnalysisJob, AnalysisRequest, AnalysisSettings, FakeService, JobState, Lane, ProjectCall, ResolveRequest, RetakeService, ServiceOptions, SilenceInfo,
    TranscriptionInfo,
};
use retake_editor_ui::{Action, OptionsAction, RetakeEditorApp};
use serde_json::{Value, json};

struct Done(Option<JobState>);

impl AnalysisJob for Done {
    fn poll(&mut self) -> JobState {
        self.0.take().unwrap_or(JobState::Cancelled)
    }

    fn cancel(&mut self) {}
}

type Replier = dyn Fn(&ProjectCall) -> (u16, Value) + Send + Sync;

#[derive(Clone, Default)]
struct Seen {
    calls: Arc<Mutex<Vec<ProjectCall>>>,
    projects: Arc<Mutex<Vec<Option<PathBuf>>>>,
    analyses: Arc<Mutex<Vec<AnalysisSettings>>>,
}

struct OptionsService {
    fake: FakeService,
    options: ServiceOptions,
    reply: Arc<Replier>,
    seen: Seen,
}

impl RetakeService for OptionsService {
    fn name(&self) -> String {
        "options test service".into()
    }

    fn options(&self) -> ServiceOptions {
        self.options.clone()
    }

    fn set_project(&self, folder: Option<PathBuf>) {
        self.seen.projects.lock().unwrap().push(folder);
    }

    fn project_call(&self, call: ProjectCall) -> Option<Box<dyn AnalysisJob>> {
        self.seen.calls.lock().unwrap().push(call.clone());
        let (status, body) = (self.reply)(&call);
        Some(Box::new(Done(Some(JobState::Done(json!({"status": status, "body": body}).to_string())))))
    }

    fn analyze(&self, request: AnalysisRequest) -> Box<dyn AnalysisJob> {
        self.seen.analyses.lock().unwrap().push(request.settings.clone());
        self.fake.analyze(request)
    }

    fn resolve(&self, request: ResolveRequest) -> Box<dyn AnalysisJob> {
        self.fake.resolve(request)
    }
}

fn options() -> ServiceOptions {
    ServiceOptions {
        transcription: TranscriptionInfo::from_json(&json!({
        "default": "large-v2",
        "gpu": {"name": "NVIDIA RTX 3070 Ti", "total_gb": 8.0, "free_gb": 5.2},
        "compute_type": "auto: tries int8_float16, float16, int8 in order",
        "models": [
            {"name": "tiny", "downloaded": false, "download_gb": 0.08, "vram_gb": 1.0, "relative_speed": 12.0},
            {"name": "base", "downloaded": true, "size_on_disk_gb": 0.15, "download_gb": 0.15, "vram_gb": 1.2, "relative_speed": 8.0},
            {"name": "large-v2", "default": true, "downloaded": true, "size_on_disk_gb": 3.09, "download_gb": 3.1, "vram_gb": 5.5, "relative_speed": 1.0,
             "warning": "Needs about 5.5 GB of GPU memory (approximate); only 5.2 GB is free right now."},
            {"name": "large-v3", "downloaded": false, "download_gb": 3.1, "vram_gb": 5.5, "relative_speed": 1.0}
        ]})),
        silence_gap: SilenceInfo::from_json(&json!({"default": 0.7, "min": 0.1, "max": 3.0, "step": 0.05})),
    }
}

fn status(model: Option<&str>, has_units: bool, source: (&str, &str), legacy: &[&str]) -> Value {
    json!({"settings": {}, "stale": [], "transcript_model": model, "has": {"transcript": model.is_some(), "units": has_units},
           "source": {"status": source.0, "message": source.1}, "legacy": legacy})
}

fn start(renderer: SoftRenderer, reply: Arc<Replier>) -> (Window, PathBuf, Seen) {
    let dir = scratch("options");
    let clip = make_clip(&dir, "talk.mp4", 4);
    let script = Script::default();
    script.media.lock().unwrap().push_back(vec![clip]);
    script.save.lock().unwrap().push_back(dir.join("Proj.fcproj"));
    let seen = Seen::default();
    let service = OptionsService { fake: FakeService::default(), options: options(), reply, seen: seen.clone() };
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
    (h, dir, seen)
}

fn saved(h: &mut Window) {
    click(h, "Save");
    h.step();
    h.step();
}

fn open_options(h: &mut Window) {
    click(h, "Options");
    // a window that just opened lays itself out a frame late, and the project's status arrives a frame after that: let it settle before aiming at it
    for _ in 0..3 {
        h.step();
    }
}

fn calls(seen: &Seen) -> Vec<ProjectCall> {
    seen.calls.lock().unwrap().clone()
}

fn quiet() -> Arc<Replier> {
    Arc::new(|call| match call {
        ProjectCall::Status => (200, status(None, false, ("ok", "The source video matches."), &[])),
        ProjectCall::SetSettings { .. } => (200, json!({"settings": {}, "stale": [], "newly_stale": [], "ask": null})),
        _ => (200, json!({})),
    })
}

#[test]
fn the_dropdown_lists_every_model_with_its_details_and_warnings_without_downloading_anything() {
    if skip() {
        return;
    }
    let (mut h, _dir, seen) = start(SoftRenderer::default(), quiet());
    open_options(&mut h);
    h.get_by_label("Retake options");
    h.get_by_value("large-v2 (default)  ·  downloaded  ·  ≈5.5 GB GPU  ·  1.0× speed"); // the combo box shows the chosen model and what it takes
    // the details of the chosen model
    h.get_by_label("already downloaded (3.09 GB on disk)");
    h.get_by_label("about 5.5 GB (approximate)");
    h.get_by_label_contains("Warning: Needs about 5.5 GB of GPU memory");
    h.get_by_label_contains("GPU: NVIDIA RTX 3070 Ti · 8.0 GB, 5.2 GB free now");
    // open the list: every model is there, with what it takes
    h.get_by_role(Role::ComboBox).click();
    h.step();
    h.get_by_label_contains("tiny  ·  download ≈0.1 GB  ·  ≈1.0 GB GPU  ·  12.0× speed");
    h.get_by_label_contains("base  ·  downloaded  ·  ≈1.2 GB GPU  ·  8.0× speed");
    h.get_by_label_contains("large-v3  ·  download ≈3.1 GB");
    assert!(calls(&seen).is_empty(), "looking at the list asks the service nothing (it was told what it has with the capabilities)");
}

#[test]
fn an_unsaved_project_keeps_the_choice_for_the_next_analysis_and_says_so() {
    if skip() {
        return;
    }
    let (mut h, _dir, seen) = start(SoftRenderer::default(), quiet());
    open_options(&mut h);
    h.get_by_role(Role::ComboBox).click();
    h.step();
    h.get_by_label_contains("base  ·  downloaded").click();
    h.step();
    h.state_mut().perform(Action::Options(OptionsAction::SetSilenceGap(0.3)));
    h.step();
    assert!(calls(&seen).is_empty(), "no project folder: nothing to ask the service");
    click(&mut h, "Close");
    click(&mut h, "Remove Retakes");
    let sent = seen.analyses.lock().unwrap().last().cloned().expect("an analysis was started");
    assert_eq!((sent.model.as_deref(), sent.silence_gap_s, sent.replace_existing), (Some("base"), Some(0.3), false));
}

#[test]
fn a_saved_project_tells_the_service_its_folder_and_stores_the_choice_there() {
    if skip() {
        return;
    }
    let (mut h, dir, seen) = start(SoftRenderer::default(), quiet());
    saved(&mut h);
    assert_eq!(seen.projects.lock().unwrap().last().cloned().flatten(), Some(dir.join("Proj")), "the service learns the project folder X/");
    assert!(calls(&seen).contains(&ProjectCall::Status), "and the project's status is asked for when it is saved/opened");
    open_options(&mut h);
    h.get_by_role(Role::ComboBox).click();
    h.step();

    h.get_by_label_contains("base  ·  downloaded").click();
    h.step();
    h.step();
    assert!(calls(&seen).contains(&ProjectCall::SetSettings { model: Some("base".into()), silence_gap_s: None }));
    click(&mut h, "New");
    assert_eq!(seen.projects.lock().unwrap().last().cloned(), Some(None), "a new project has no folder");
}

#[test]
fn changing_the_model_after_a_transcript_exists_asks_before_re_transcribing() {
    if skip() {
        return;
    }
    let reply: Arc<Replier> = Arc::new(|call| match call {
        ProjectCall::Status => (200, status(Some("large-v2"), true, ("ok", ""), &[])),
        ProjectCall::SetSettings { model: Some(m), .. } if m != "large-v2" => (
            200,
            json!({"settings": {"transcription_model": m}, "stale": ["transcript", "units"], "newly_stale": ["transcript", "units"],
                   "ask": "The transcript was made with a different model. Re-transcribe to bring it up to date?"}),
        ),
        ProjectCall::SetSettings { .. } => (200, json!({"settings": {}, "stale": [], "newly_stale": [], "ask": null})),
        _ => (200, json!({})),
    });
    let (mut h, _dir, seen) = start(SoftRenderer::default(), reply);
    saved(&mut h);
    open_options(&mut h);
    h.get_by_role(Role::ComboBox).click();
    h.step();
    h.get_by_label_contains("base  ·  downloaded").click();
    h.step();
    h.step();
    h.step();
    h.get_by_label_contains("The transcript was made with “large-v2”. Re-transcribe with “base”?");
    assert_eq!(h.state().review_stage(), "none", "nothing was re-transcribed by choosing");
    // keep the old one: the choice goes back and the question is gone
    click(&mut h, "Keep the old transcript");
    h.step();
    assert!(h.query_by_label("Re-transcribe now").is_none());
    assert!(
        calls(&seen).contains(&ProjectCall::SetSettings { model: Some("large-v2".into()), silence_gap_s: None }),
        "the service is told to go back, so the stale marks are lifted"
    );
    assert!(seen.analyses.lock().unwrap().is_empty());

    // choose again and confirm: the analysis starts and says the replacement was agreed
    h.get_by_role(Role::ComboBox).click();
    h.step();
    h.get_by_label_contains("base  ·  downloaded").click();
    h.step();
    h.step();
    h.step();
    click(&mut h, "Re-transcribe now");
    let sent = seen.analyses.lock().unwrap().last().cloned().expect("the analysis started");
    assert_eq!((sent.model.as_deref(), sent.replace_existing), (Some("base"), true));
    step_until(&mut h, "the analysis to end", |a| !a.is_analyzing());
    h.state_mut().perform(Action::CloseReview); // the questions of the first analysis are still open
    h.step();
    click(&mut h, "Remove Retakes");
    assert!(!seen.analyses.lock().unwrap().last().unwrap().replace_existing, "the agreement was for one analysis only");
}

#[test]
fn the_slider_shows_the_sentence_count_and_resplitting_asks_what_will_be_replaced() {
    if skip() {
        return;
    }
    let stale: Arc<Mutex<Vec<&'static str>>> = Arc::default(); // what the service has marked out of date so far
    let reply: Arc<Replier> = Arc::new(move |call| match call {
        ProjectCall::Status => {
            let mut st = status(Some("large-v2"), true, ("ok", ""), &[]);
            st["stale"] = json!(*stale.lock().unwrap());
            (200, st)
        }
        ProjectCall::UnitsPreview { silence_gap_s } => (200, json!({"silence_gap_s": silence_gap_s, "units": 12, "current_units": 8})),
        ProjectCall::SetSettings { silence_gap_s: Some(_), .. } => {
            *stale.lock().unwrap() = vec!["units", "keepers"];
            (
                200,
                json!({"settings": {}, "stale": ["units", "keepers"], "newly_stale": ["units", "keepers"],
                   "ask": "The sentences were split at a different silence gap. Re-split them now (no re-transcription)?"}),
            )
        }
        ProjectCall::Resplit { confirm: false, silence_gap_s } => (
            409,
            json!({"error": "Re-splitting replaces ...", "code": "resplit_needs_confirmation", "will_replace": ["units", "keepers", "cutlist"], "silence_gap_s": silence_gap_s}),
        ),
        ProjectCall::Resplit { confirm: true, .. } => {
            (200, json!({"units": 12, "silence_gap_s": 0.3, "stale": ["keepers"], "previous_units_copy": "units.1.json"}))
        }
        _ => (200, json!({})),
    });
    let (mut h, _dir, seen) = start(SoftRenderer::default(), reply);
    saved(&mut h);
    open_options(&mut h);
    // the real slider (it comes with a numeric box): focus it and press the arrow key
    h.get_all_by_role(Role::Slider).next().unwrap().focus();
    h.step();
    h.key_press(egui::Key::ArrowLeft);
    h.step();
    h.step();
    h.step();
    assert!(calls(&seen).iter().any(|c| matches!(c, ProjectCall::UnitsPreview { .. })), "moving the slider asks how many sentences it would give");
    h.state_mut().perform(Action::Options(OptionsAction::SetSilenceGap(0.3)));
    h.step();
    h.step();
    h.step();
    h.get_by_label("At 0.30 s the transcript gives 12 sentences (now 8).");
    // let go: stored in the project, and the question about the stale results appears
    h.state_mut().perform(Action::Options(OptionsAction::CommitSilenceGap));
    h.step();
    h.step();
    h.step();
    h.get_by_label_contains("Out of date: sentences, keeper groups");
    click(&mut h, "Re-split now");
    h.step();
    h.step();
    h.get_by_label_contains("Re-split at 0.30 s? This replaces: sentences, keeper groups, cut list.");
    assert!(!calls(&seen).iter().any(|c| matches!(c, ProjectCall::Resplit { confirm: true, .. })), "nothing is replaced before the confirmation");
    click(&mut h, "Re-split");
    h.step();
    h.step();
    h.get_by_label_contains("Re-split into 12 sentences.");
    assert!(calls(&seen).iter().any(|c| matches!(c, ProjectCall::Resplit { confirm: true, silence_gap_s } if (*silence_gap_s - 0.3).abs() < 1e-9)));
}

#[test]
fn a_missing_or_changed_source_video_is_warned_about_once_when_the_project_is_opened() {
    if skip() {
        return;
    }
    let reply: Arc<Replier> = Arc::new(|call| match call {
        ProjectCall::Status => (
            200,
            status(Some("large-v2"), true, ("changed", "The source video has changed since this project was saved; the transcript may not match it."), &[]),
        ),
        _ => (200, json!({})),
    });
    let (mut h, _dir, _seen) = start(SoftRenderer::default(), reply);
    saved(&mut h);
    let b = h.state().banner().cloned().expect("a warning");
    assert!(b.error && b.text.contains("The source video has changed"), "{b:?}");
    h.state_mut().perform(Action::DismissBanner);
    open_options(&mut h);
    h.get_by_label_contains("The source video has changed since this project was saved");
    assert!(h.state().banner().is_none(), "not repeated every time the window asks again");
}

#[test]
fn old_data_is_offered_for_import_and_copied_only_when_asked() {
    if skip() {
        return;
    }
    let reply: Arc<Replier> = Arc::new(|call| match call {
        ProjectCall::Status => (200, status(None, false, ("ok", ""), &["transcript", "units", "scores__final-ce_stacker-v1"])),
        ProjectCall::ImportLegacy { .. } => (200, json!({"imported": ["transcript", "units"], "originals_untouched": true})),
        _ => (200, json!({})),
    });
    let (mut h, _dir, seen) = start(SoftRenderer::default(), reply);
    saved(&mut h);
    open_options(&mut h);
    h.get_by_label_contains("Earlier work on this video was found (transcript, sentences, pair scores). Copy it into this project?");
    let imports = |seen: &Seen| calls(seen).iter().filter(|c| matches!(c, ProjectCall::ImportLegacy { .. })).count();
    assert_eq!(imports(&seen), 1, "saving already brought the earlier work along once (automatically); this is the button for a project opened later");
    click(&mut h, "Import into this project");
    h.step();
    h.step();
    h.get_by_label_contains("Copied into the project: transcript, sentences.");
    assert_eq!(imports(&seen), 2, "the button sends it again");
    assert!(calls(&seen).iter().all(|c| !matches!(c, ProjectCall::ImportLegacy { path } if !path.ends_with("talk.mp4"))), "every call names the video");
}

fn shot(h: &Window, r: &SoftRenderer) -> Picture {
    r.draw(&h.ctx, h.output(), 1360.0, 860.0, egui::Color32::from_rgb(16, 16, 20))
}

/// Not a check: writes the window to <dir>/<tag>_<state>.png. Runs only when RETAKE_EDITOR_SHOT_DIR and RETAKE_EDITOR_SHOT_TAG are set.
#[test]
fn write_options_screenshots() {
    let (Some(dir), Ok(tag)) = (std::env::var_os("RETAKE_EDITOR_SHOT_DIR"), std::env::var("RETAKE_EDITOR_SHOT_TAG")) else {
        eprintln!("SKIPPED: set RETAKE_EDITOR_SHOT_DIR and RETAKE_EDITOR_SHOT_TAG to write screenshots");
        return;
    };
    if skip() {
        return;
    }
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let crop = |p: &Picture, x: usize, y: usize, w: usize, hgt: usize| p.crop_scaled(x, y, w, hgt, 2);

    let reply: Arc<Replier> = Arc::new(|call| match call {
        ProjectCall::Status => (200, status(Some("large-v2"), true, ("ok", ""), &[])),
        ProjectCall::UnitsPreview { silence_gap_s } => (200, json!({"silence_gap_s": silence_gap_s, "units": 12, "current_units": 8})),
        ProjectCall::SetSettings { model: Some(m), .. } if m != "large-v2" => (
            200,
            json!({"settings": {}, "stale": ["transcript", "units"], "newly_stale": ["transcript", "units"], "ask": "The transcript was made with a different model."}),
        ),
        _ => (200, json!({"settings": {}, "stale": [], "newly_stale": [], "ask": null})),
    });
    let r = SoftRenderer::default();
    let (mut h, _d, _seen) = start(r.clone(), reply);
    saved(&mut h);
    open_options(&mut h);
    h.state_mut().perform(Action::Options(OptionsAction::SetSilenceGap(0.3)));
    for _ in 0..3 {
        h.step();
    }
    crop(&shot(&h, &r), 400, 160, 560, 540).save_png(&dir.join(format!("{tag}_1_options.png")));
    h.get_by_role(Role::ComboBox).click();
    h.step();
    h.step();
    crop(&shot(&h, &r), 400, 160, 560, 540).save_png(&dir.join(format!("{tag}_2_dropdown.png")));
    h.get_by_label_contains("base  ·  downloaded").click();
    for _ in 0..4 {
        h.step();
    }
    crop(&shot(&h, &r), 400, 120, 560, 600).save_png(&dir.join(format!("{tag}_3_ask.png")));
}

/// Every row of the window (labels, buttons, boxes) with where it is.
fn layout(h: &Window) -> Vec<(String, egui::Rect)> {
    use egui_kittest::kittest::NodeT;
    let mut v = Vec::new();
    for role in [Role::Label, Role::Button, Role::ComboBox, Role::Slider, Role::SpinButton] {
        for n in h.query_all_by_role(role) {
            let name = n.accesskit_node().label().or_else(|| n.accesskit_node().value()).unwrap_or_default();
            v.push((format!("{role:?}:{name}"), n.rect()));
        }
    }
    v
}

#[test]
fn hovering_over_the_model_dropdown_and_its_items_moves_no_row() {
    if skip() {
        return;
    }
    let (mut h, _dir, _seen) = start(SoftRenderer::default(), quiet());
    saved(&mut h);
    open_options(&mut h);
    let closed = layout(&h);
    let combo = h.get_by_role(Role::ComboBox).rect();
    h.hover_at(combo.center());
    for _ in 0..4 {
        h.step();
    }
    assert_eq!(layout(&h), closed, "hovering the closed drop-down moves nothing");
    h.get_by_role(Role::ComboBox).click();
    for _ in 0..4 {
        h.step();
    }
    let open = layout(&h);
    let items: Vec<egui::Pos2> = h.query_all_by_label_contains("·").map(|n| n.rect().center()).collect();
    assert!(items.len() >= 4, "the list is open: {}", items.len());
    for at in items {
        h.hover_at(at);
        for _ in 0..3 {
            h.step();
        }
        let now = layout(&h);
        // the hovered row may show its highlight; no OTHER row may move
        let moved: Vec<_> =
            open.iter().zip(&now).filter(|(a, b)| a.1 != b.1 && !a.1.contains(at)).map(|(a, b)| format!("{}: {:?} -> {:?}", a.0, a.1, b.1)).collect();
        assert!(moved.is_empty(), "hovering the item at {at:?} moved other rows: {moved:?}");
    }
}

#[test]
fn saving_a_project_brings_the_earlier_transcription_along_and_says_so() {
    if skip() {
        return;
    }
    let reply: Arc<Replier> = Arc::new(|call| match call {
        ProjectCall::Status => (200, status(Some("large-v2"), true, ("ok", ""), &[])),
        ProjectCall::ImportLegacy { .. } => {
            (200, json!({"imported": ["transcript", "units", "units_meta", "pairs__final-ce_stacker-v1"], "originals_untouched": true}))
        }
        _ => (200, json!({})),
    });
    let (mut h, _dir, seen) = start(SoftRenderer::default(), reply);
    assert!(calls(&seen).is_empty(), "nothing is asked before the project is saved");
    saved(&mut h);
    let adopted: Vec<_> = calls(&seen).into_iter().filter(|c| matches!(c, ProjectCall::ImportLegacy { path } if path.ends_with("talk.mp4"))).collect();
    assert_eq!(adopted.len(), 1, "right after the first save, the work done before it is copied into the project");
    open_options(&mut h);
    h.get_by_label_contains("Copied into the project: transcript, sentences, pair scores.");
}

#[test]
fn work_the_user_reset_is_not_called_out_of_date() {
    if skip() {
        return;
    }
    let reset = Arc::new(Mutex::new(true));
    let flag = reset.clone();
    let reply: Arc<Replier> = Arc::new(move |call| match call {
        ProjectCall::Status => {
            let mut st = status(Some("large-v2"), true, ("ok", ""), &[]);
            if *flag.lock().unwrap() {
                st["stale"] = json!(["transcript", "units", "pairs__final-ce_stacker-v1", "words"]);
                st["reset"] = json!(["transcript", "units", "pairs__final-ce_stacker-v1", "words"]);
            } else {
                st["stale"] = json!(["units", "pairs__final-ce_stacker-v1", "keepers"]);
            }
            (200, st)
        }
        _ => (200, json!({})),
    });
    let (mut h, _dir, _seen) = start(SoftRenderer::default(), reply);
    saved(&mut h);
    open_options(&mut h);
    assert!(h.query_by_label_contains("Out of date").is_none(), "set aside on purpose: nothing to report");
    *reset.lock().unwrap() = false;
    h.state_mut().perform(Action::Options(OptionsAction::Open));
    for _ in 0..4 {
        h.step();
    }
    h.get_by_label("Out of date: sentences, pair scores, keeper groups"); // a real stale result is named in plain words, each once
}
