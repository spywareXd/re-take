//! The Exports window (the retake service's exports): the form is built from the service's real options reply (a fixture captured from the Python service), the
//! request carries exactly what was chosen, progress and the finished output (size, fps, constant or variable) are shown, ffmpeg's own error text is shown, and
//! hardware encoding is offered only when the service says it works.
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
use egui_kittest::kittest::{NodeT, Queryable};
use serde_json::{Value, json};
use simplecut_core::{
    AnalysisJob, AnalysisRequest, ExportOptionsInfo, ExportRequest, FakeService, JobState, Lane, ResolveRequest, RetakeService, StageEvent, UserCall,
};
use simplecut_ui::{Action, SimpleCutApp};

const OPTIONS: &str = include_str!("fixtures/export_options.json");

fn options_json(hardware: bool) -> Value {
    let mut v: Value = serde_json::from_str(OPTIONS).unwrap();
    v["video"]["hardware"] = if hardware { json!({"h264": true, "h265": true}) } else { Value::Null };
    v
}

fn ev(seq: u64, stage: &str, status: &str, pct: Option<f64>, message: &str) -> StageEvent {
    let mut v = json!({"seq": seq, "stage": stage, "status": status, "message": message, "timestamp": "2026-10-30T10:00:00.000+00:00"});
    if let Some(p) = pct {
        v["pct"] = json!(p);
    }
    StageEvent::from_json(&v).unwrap()
}

struct ExportJob {
    chunks: Vec<Vec<StageEvent>>,
    next: usize,
    end: JobState,
    hold: bool,
}

impl AnalysisJob for ExportJob {
    fn poll(&mut self) -> JobState {
        if self.hold || self.next < self.chunks.len() {
            return JobState::Running { fraction: 0.5, status: "Exporting".into() };
        }
        self.end.clone()
    }

    fn cancel(&mut self) {}

    fn events(&mut self) -> Vec<StageEvent> {
        let out = self.chunks.get(self.next).cloned().unwrap_or_default();
        self.next += 1;
        out
    }
}

struct Immediate(Option<JobState>);

impl AnalysisJob for Immediate {
    fn poll(&mut self) -> JobState {
        self.0.take().unwrap_or(JobState::Cancelled)
    }

    fn cancel(&mut self) {}
}

#[derive(Clone, Default)]
struct Seen {
    exports: Arc<Mutex<Vec<ExportRequest>>>,
    user_calls: Arc<Mutex<Vec<UserCall>>>,
    /// The scripted dialogs of the window (what the file and folder pickers answer).
    script: Script,
}

struct Svc {
    fake: FakeService,
    options: Value,
    seen: Seen,
    chunks: Vec<Vec<StageEvent>>,
    end: JobState,
    hold: bool,
}

impl RetakeService for Svc {
    fn name(&self) -> String {
        "export test service".into()
    }

    fn analyze(&self, request: AnalysisRequest) -> Box<dyn AnalysisJob> {
        self.fake.analyze(request)
    }

    fn resolve(&self, request: ResolveRequest) -> Box<dyn AnalysisJob> {
        self.fake.resolve(request)
    }

    fn export(&self, request: ExportRequest) -> Option<Box<dyn AnalysisJob>> {
        self.seen.exports.lock().unwrap().push(request);
        Some(Box::new(ExportJob { chunks: self.chunks.clone(), next: 0, end: self.end.clone(), hold: self.hold }))
    }

    fn user_call(&self, call: UserCall) -> Option<Box<dyn AnalysisJob>> {
        self.seen.user_calls.lock().unwrap().push(call.clone());
        let body = match call {
            UserCall::ExportOptions => self.options.clone(),
            _ => json!({}),
        };
        Some(Box::new(Immediate(Some(JobState::Done(json!({"status": 200, "body": body}).to_string())))))
    }
}

fn ok_result() -> JobState {
    JobState::Done(
        json!({"path": "C:/x/Proj/data/exports/talk_YouTube 1080p.mp4", "relative": "data/exports/talk_YouTube 1080p.mp4", "kind": "video",
               "verify": {"width": 1920, "height": 1080, "fps": 30.0, "frame_rate_mode": "CFR", "duration_s": 4.5, "bytes": 1_596_231, "has_audio": true},
               "notes": ["Upscaling from 360p to 1080p: the file is bigger, not sharper.", "Output frame rate: CFR"]})
        .to_string(),
    )
}

fn start(renderer: SoftRenderer, svc: impl FnOnce(Seen) -> Svc, save: bool) -> (Window, PathBuf, Seen) {
    let dir = scratch("svc-export");
    let clip = make_clip(&dir, "talk.mp4", 4);
    let script = Script::default();
    script.media.lock().unwrap().push_back(vec![clip]);
    script.save.lock().unwrap().push_back(dir.join("Proj.fcproj"));
    let seen = Seen { script: script.clone(), ..Default::default() };
    let service = svc(seen.clone());
    let mut h = Harness::builder().with_size(vec2(1360.0, 860.0)).with_max_steps(10_000).renderer(renderer).build_eframe(move |cc| {
        let mut app = SimpleCutApp::new(&cc.egui_ctx, Box::new(script), None);
        app.set_service(Some(Box::new(service)));
        app
    });
    h.step();
    click(&mut h, "Import");
    step_until(&mut h, "the import", |a| a.editor().media().len() == 1);
    let video = item(&h, "talk.mp4");
    h.state_mut().perform(Action::AddToTimeline(video, Lane::Main));
    h.step();
    if save {
        click(&mut h, "Save");
    }
    (h, dir, seen)
}

fn svc(options: Value, chunks: Vec<Vec<StageEvent>>, end: JobState, hold: bool) -> impl FnOnce(Seen) -> Svc {
    move |seen| Svc { fake: FakeService::default(), options, seen, chunks, end, hold }
}

fn open(h: &mut Window) {
    click(h, "Export");
    for _ in 0..3 {
        h.step();
    }
}

fn pick(h: &mut Window, current_value: &str, item_label_contains: &str) {
    // the window is centred and changes height as results appear: let it settle before aiming at the combo box
    h.step();
    h.step();
    h.get_by_value(current_value).click();
    // a popup that just opened lays itself out a frame late: let it settle before aiming at an item
    for _ in 0..3 {
        h.step();
    }
    h.get_by_label_contains(item_label_contains).click();
    h.step();
    h.step();
}

// ---- the form, from the real options reply ----

#[test]
fn the_options_reply_of_the_real_service_is_understood() {
    let v: Value = serde_json::from_str(OPTIONS).unwrap();
    let info = ExportOptionsInfo::from_json(&v).expect("the real reply parses");
    assert_eq!(
        info.presets.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
        vec!["Match Source", "YouTube 1080p", "YouTube 4K", "Shorts / Reels / TikTok", "Draft / Preview"]
    );
    assert!(info.presets.iter().all(|p| p.builtin));
    assert_eq!(info.heights, vec!["2160p", "1440p", "1080p", "720p", "480p"]);
    assert_eq!(info.canvases.len(), 2);
    assert_eq!(info.fits, vec!["letterbox", "crop", "stretch"]);
    assert_eq!((info.dim_min, info.dim_max), (16, 7680));
    assert_eq!(info.fps_choices, vec![24, 25, 30, 50, 60]);
    assert_eq!(info.codecs, vec!["h264", "h265"]);
    assert_eq!((info.crf_range, info.bitrate_range), ((0, 51), (100, 200_000)));
    assert!(info.speeds.contains(&"ultrafast".to_string()) && info.speeds.contains(&"veryslow".to_string()));
    assert_eq!(info.audio_bitrates, vec![64, 96, 128, 160, 192, 256, 320, 384, 512]);
    assert_eq!(info.audio_formats, vec!["mp3", "wav"]);
    assert_eq!(info.text_formats.iter().find(|(k, _)| k == "subtitles").map(|(_, f)| f.clone()), Some(vec!["srt".to_string(), "vtt".to_string()]));
    assert_eq!((info.fade_default, info.fade_max), (15.0, 100.0));
    let no_hw = ExportOptionsInfo::from_json(&options_json(false)).unwrap();
    assert!(no_hw.hardware.is_empty(), "no hardware listed when the service did not list it");
    assert!(!ExportOptionsInfo::from_json(&options_json(true)).unwrap().hardware.is_empty());
    assert!(ExportOptionsInfo::from_json(&json!({"nope": 1})).is_none());
}

// ---- the window ----

#[test]
fn the_window_offers_the_presets_and_every_option_and_hides_hardware_unless_it_works() {
    if skip() {
        return;
    }
    let (mut h, _d, _s) = start(SoftRenderer::default(), svc(options_json(false), vec![], ok_result(), false), true);
    open(&mut h);
    h.get_by_label("Export options");
    h.get_by_value("Video (MP4)");
    h.get_by_value("YouTube 1080p"); // the default preset
    h.get_by_value("1080p"); // its size
    h.get_by_value("Match source (variable rate kept)");
    h.get_by_value("H.264");
    h.get_by_label_contains("Normalise loudness (off by default)");
    assert!(h.query_by_label_contains("NVENC").is_none(), "the service listed no hardware encoder: the option is not offered");
    h.get_by_label_contains("Names are never reused: an earlier export is never overwritten");
    // another service that did list it
    let (mut h2, _d2, _s2) = start(SoftRenderer::default(), svc(options_json(true), vec![], ok_result(), false), true);
    open(&mut h2);
    h2.get_by_label_contains("Use the NVIDIA GPU encoder (NVENC)");
}

#[test]
fn choosing_a_preset_fills_the_form_and_the_request_carries_exactly_what_is_shown() {
    if skip() {
        return;
    }
    let chunks = vec![vec![ev(1, "export_render", "started", None, "video -> x.mp4")], vec![ev(2, "export_render", "progress", Some(40.0), "1.8 of 4.5 s")]];
    let (mut h, _d, seen) = start(SoftRenderer::default(), svc(options_json(false), chunks, ok_result(), false), true);
    open(&mut h);
    pick(&mut h, "YouTube 1080p", "Shorts / Reels / TikTok");
    h.get_by_value("Shorts / Reels / TikTok");
    h.get_by_value("Vertical 1080×1920");
    h.get_by_value("30 fps");
    h.get_by_value("crop"); // how the picture fills the frame
    click(&mut h, "Start export");
    let req = seen.exports.lock().unwrap().last().cloned().expect("the export was requested");
    assert!(req.path.ends_with("talk.mp4"));
    let b = &req.body;
    assert_eq!(b["kind"], "video");
    assert_eq!(b["preset"], "Shorts / Reels / TikTok");
    assert_eq!(b["resolution"], json!({"mode": "preset", "preset": "vertical_1080x1920", "fit": "crop"}));
    assert_eq!(b["fps"], 30);
    assert_eq!(b["video"], json!({"codec": "h264", "quality": {"crf": 20}, "speed": "medium", "hardware": false}));
    assert_eq!(
        b["audio"],
        json!({"bitrate_kbps": 128, "normalize_loudness": false, "fade_ms": 15.0}),
        "loudness normalisation stays off, the crossfade is the pipeline's 15 ms"
    );
}

#[test]
fn progress_and_the_finished_output_are_shown_with_constant_or_variable_frame_rate() {
    if skip() {
        return;
    }
    let chunks = vec![vec![ev(1, "export_render", "started", None, "")], vec![ev(2, "export_render", "progress", Some(40.0), "")]];
    let (mut h, _d, _s) = start(SoftRenderer::default(), svc(options_json(false), chunks, ok_result(), false), true);
    open(&mut h);
    click(&mut h, "Start export");
    h.step();
    h.get_by_label_contains("Saved: data/exports/talk_YouTube 1080p.mp4");
    h.get_by_label_contains("Output: 1920×1080 · 30.000 fps · CFR (constant frame rate) · 4.5 s · 1.6 MB");
    h.get_by_label_contains("Upscaling from 360p to 1080p");
}

#[test]
fn the_progress_bar_follows_the_ffmpeg_percent_while_rendering() {
    if skip() {
        return;
    }
    let chunks = vec![vec![ev(1, "export_render", "started", None, "")], vec![ev(2, "export_render", "progress", Some(40.0), "")]];
    let (mut h, _d, _s) = start(SoftRenderer::default(), svc(options_json(false), chunks, ok_result(), true), true);
    open(&mut h);
    click(&mut h, "Start export");
    for _ in 0..4 {
        h.step();
    }
    h.get_by_label_contains("Rendering 40%");
    h.get_by_label("Cancel export");
}

#[test]
fn an_ffmpeg_error_is_shown_exactly_as_ffmpeg_wrote_it() {
    if skip() {
        return;
    }
    let ffmpeg = "Unrecognized option 'bogus_option'.\nError splitting the argument list: Option not found";
    let chunks = vec![vec![ev(1, "export_render", "started", None, ""), ev(2, "export_render", "error", None, ffmpeg)]];
    let end = JobState::Failed("Something went wrong while processing the file (job r_1). See the service log.".into());
    let (mut h, _d, _s) = start(SoftRenderer::default(), svc(options_json(false), chunks, end, false), true);
    open(&mut h);
    click(&mut h, "Start export");
    h.step();
    h.get_by_label_contains("Something went wrong while processing the file");
    h.get_by_label_contains("Unrecognized option 'bogus_option'.");
    h.get_by_label_contains("Error splitting the argument list: Option not found");
}

#[test]
fn an_unsaved_project_cannot_export_and_says_why() {
    if skip() {
        return;
    }
    let (mut h, _d, seen) = start(SoftRenderer::default(), svc(options_json(false), vec![], ok_result(), false), false);
    open(&mut h);
    h.get_by_value("Video — whole timeline (MP4)"); // no folder yet: the editor's own export is the default
    pick(&mut h, "Video — whole timeline (MP4)", "Video (MP4)");
    h.get_by_label("Save the project first: these exports use its folder.");
    assert!(h.get_by_label("Start export").accesskit_node().is_disabled());
    assert!(seen.exports.lock().unwrap().is_empty());
}

#[test]
fn the_kind_switches_the_formats_and_a_preset_of_your_own_is_saved_through_the_service() {
    if skip() {
        return;
    }
    let (mut h, _d, seen) = start(SoftRenderer::default(), svc(options_json(false), vec![], ok_result(), false), true);
    open(&mut h);
    pick(&mut h, "Video (MP4)", "Subtitles of the kept text");
    h.get_by_value("SRT");
    pick(&mut h, "SRT", "VTT");
    h.get_by_value("VTT"); // the combo shows the new choice
    click(&mut h, "Start export");
    let b = seen.exports.lock().unwrap().last().unwrap().body.clone();
    assert_eq!((b["kind"].as_str(), b["format"].as_str()), (Some("subtitles"), Some("vtt")));
    assert!(b.get("resolution").is_none(), "a text export carries no video options");
    pick(&mut h, "Subtitles of the kept text", "Video (MP4)");
    h.get_by_role(Role::TextInput).focus();
    h.step();
    h.get_by_role(Role::TextInput).type_text("My draft");
    h.step();
    click(&mut h, "Save as preset");
    let saved = seen
        .user_calls
        .lock()
        .unwrap()
        .iter()
        .find_map(|c| if let UserCall::SavePreset { name, preset } = c { Some((name.clone(), preset.clone())) } else { None });
    let (name, preset) = saved.expect("the preset was sent to the service (it is kept in the user's config, not the project)");
    assert_eq!(name, "My draft");
    assert!(preset["video"].is_object() && preset["resolution"].is_object() && preset["audio"].is_object());
}

fn shot(h: &Window, r: &SoftRenderer) -> Picture {
    r.draw(&h.ctx, h.output(), 1360.0, 860.0, egui::Color32::from_rgb(16, 16, 20))
}

/// Not a check: writes the window to <dir>/<tag>_<state>.png. Runs only when SIMPLECUT_SHOT_DIR and SIMPLECUT_SHOT_TAG are set.
#[test]
fn write_export_screenshots() {
    let (Some(dir), Ok(tag)) = (std::env::var_os("SIMPLECUT_SHOT_DIR"), std::env::var("SIMPLECUT_SHOT_TAG")) else {
        eprintln!("SKIPPED: set SIMPLECUT_SHOT_DIR and SIMPLECUT_SHOT_TAG to write screenshots");
        return;
    };
    if skip() {
        return;
    }
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let chunks = vec![vec![ev(1, "export_render", "started", None, "")], vec![ev(2, "export_render", "progress", Some(40.0), "")]];
    let r = SoftRenderer::default();
    let (mut h, _d, _s) = start(r.clone(), svc(options_json(true), chunks, ok_result(), false), true);
    open(&mut h);
    shot(&h, &r).crop_scaled(380, 60, 600, 740, 2).save_png(&dir.join(format!("{tag}_1_form.png")));
    click(&mut h, "Start export");
    h.step();
    h.step();
    shot(&h, &r).crop_scaled(380, 60, 600, 740, 2).save_png(&dir.join(format!("{tag}_2_done.png")));
}

// ---- one Export menu ----

#[test]
fn there_is_one_export_button_and_a_saved_project_opens_on_the_service_video_with_options() {
    if skip() {
        return;
    }
    let (mut h, _d, _s) = start(SoftRenderer::default(), svc(options_json(false), vec![], ok_result(), false), true);
    assert!(h.query_by_label("Exports").is_none(), "the second Export button is gone");
    open(&mut h);
    h.get_by_label("Export options");
    h.get_by_value("Video (MP4)"); // saved project + service: the video with options is the default
    h.get_by_value("YouTube 1080p");
    h.get_by_label("Destination folder:");
    h.get_by_label("Browse folder…");
    h.get_by_label("Start export"); // the Export button is inside the dialog
}

#[test]
fn the_whole_timeline_export_is_in_the_same_dialog_with_its_own_destination() {
    if skip() {
        return;
    }
    let (mut h, dir, seen) = start(SoftRenderer::default(), svc(options_json(false), vec![], ok_result(), false), true);
    open(&mut h);
    pick(&mut h, "Video (MP4)", "Video — whole timeline (MP4)");
    h.get_by_label("Preset: YouTube 1080p Full HD (1920×1080, H.264 + AAC)");
    h.get_by_label("Use graphics card if available");
    seen.script.export.lock().unwrap().push_back(dir.join("my cut.mp4"));
    click(&mut h, "Browse...");
    h.get_by_label("my cut.mp4");
    click(&mut h, "Start export");
    step_until(&mut h, "the timeline export to start", |a| a.is_exporting());
    assert!(seen.exports.lock().unwrap().is_empty(), "the editor rendered it itself: the service was not asked");
}

#[test]
fn the_service_exports_go_to_the_folder_the_user_picked_until_they_go_back_to_the_project() {
    if skip() {
        return;
    }
    let (mut h, dir, seen) = start(SoftRenderer::default(), svc(options_json(false), vec![], ok_result(), false), true);
    open(&mut h);
    let folder = dir.join("deliver");
    seen.script.export_folder.lock().unwrap().push_back(folder.clone());
    click(&mut h, "Browse folder…");
    h.get_by_label_contains(&folder.display().to_string());
    for _ in 0..3 {
        h.step(); // the long folder name widens the window a frame later and it is re-centred: settle before aiming at the button
    }
    click(&mut h, "Start export");
    let body = seen.exports.lock().unwrap().last().unwrap().body.clone();
    assert_eq!(body["destination"].as_str(), Some(folder.to_string_lossy().as_ref()), "the request carries the chosen folder");
    open(&mut h);
    for _ in 0..3 {
        h.step();
    }
    click(&mut h, "Use the project folder");
    for _ in 0..3 {
        h.step();
    }
    click(&mut h, "Start export");
    assert!(seen.exports.lock().unwrap().last().unwrap().body.get("destination").is_none(), "back to the project's own exports folder");
}
