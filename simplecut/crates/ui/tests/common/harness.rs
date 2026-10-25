//! Helpers shared by the window tests: scripted dialogs, a headless window, stepping, clicking.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::{have, make_clip, make_music, scratch};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use simplecut_core::{ItemId, Tick};
use simplecut_ui::{Dialogs, SimpleCutApp};

/// Dialogs that answer from a script.
#[derive(Clone, Default)]
pub struct Script {
    pub media: Arc<Mutex<VecDeque<Vec<PathBuf>>>>,
    pub open: Arc<Mutex<VecDeque<PathBuf>>>,
    pub save: Arc<Mutex<VecDeque<PathBuf>>>,
    pub export: Arc<Mutex<VecDeque<PathBuf>>>,
    pub analysis: Arc<Mutex<VecDeque<PathBuf>>>,
    pub relink: Arc<Mutex<VecDeque<PathBuf>>>,
    pub groups_save: Arc<Mutex<VecDeque<PathBuf>>>,
    pub opened_folders: Arc<Mutex<Vec<PathBuf>>>,
}

impl Dialogs for Script {
    fn pick_media(&self) -> Vec<PathBuf> {
        self.media.lock().unwrap().pop_front().unwrap_or_default()
    }

    fn pick_project_to_open(&self) -> Option<PathBuf> {
        self.open.lock().unwrap().pop_front()
    }

    fn pick_project_save_path(&self, _suggested_name: &str) -> Option<PathBuf> {
        self.save.lock().unwrap().pop_front()
    }

    fn pick_analysis_file(&self) -> Option<PathBuf> {
        self.analysis.lock().unwrap().pop_front()
    }

    fn pick_groups_save_path(&self, _suggested_name: &str) -> Option<PathBuf> {
        self.groups_save.lock().unwrap().pop_front()
    }

    fn pick_export_path(&self, _suggested_name: &str) -> Option<PathBuf> {
        self.export.lock().unwrap().pop_front()
    }

    fn pick_relink_path(&self, _suggested_name: &str) -> Option<PathBuf> {
        self.relink.lock().unwrap().pop_front()
    }

    fn open_folder(&self, path: &std::path::Path) {
        self.opened_folders.lock().unwrap().push(path.to_path_buf());
    }
}

pub type Window = Harness<'static, SimpleCutApp>;

pub fn window(script: Script) -> Window {
    window_with(script, None)
}

/// A window with a retake service connected (or none).
pub fn window_with(script: Script, service: Option<Box<dyn simplecut_core::RetakeService>>) -> Window {
    let mut h = Harness::builder().with_size(egui::vec2(1300.0, 820.0)).with_max_steps(10_000).build_eframe(move |cc| {
        let mut app = SimpleCutApp::new(&cc.egui_ctx, Box::new(script), None);
        app.set_service(service);
        app
    });
    h.step();
    h
}

/// Keep drawing frames until `cond` holds.
pub fn step_until(h: &mut Window, what: &str, mut cond: impl FnMut(&SimpleCutApp) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        h.step();
        if cond(h.state()) {
            return;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

pub fn click(h: &mut Window, label: &str) {
    // a window that just opened lays itself out one frame late: let it settle before aiming at it
    h.step();
    h.get_by_label(label).click();
    h.step();
    h.step();
}

pub fn secs(s: f64) -> Tick {
    Tick::from_seconds_f64(s)
}

pub fn skip() -> bool {
    let missing = !have("ffmpeg") || !have("ffprobe");
    if missing {
        eprintln!("SKIPPED: ffmpeg / ffprobe are not installed");
    }
    missing
}

/// A window with a 4 s clip and a 6 s music file imported through the Import button.
pub fn with_media() -> (Window, Script, PathBuf) {
    let dir = scratch("window");
    let script = Script::default();
    script.media.lock().unwrap().push_back(vec![make_clip(&dir, "talk.mp4", 4), make_music(&dir, "bed.wav", 6)]);
    let mut h = window(script.clone());
    click(&mut h, "Import");
    step_until(&mut h, "the import", |a| a.editor().media().len() == 2);
    (h, script, dir)
}

/// Like [`with_media`], with a retake service connected.
pub fn with_media_and_service(service: simplecut_core::FakeService) -> (Window, Script, PathBuf) {
    let dir = scratch("window-retakes");
    let script = Script::default();
    script.media.lock().unwrap().push_back(vec![make_clip(&dir, "talk.mp4", 4), make_music(&dir, "bed.wav", 6)]);
    let mut h = window_with(script.clone(), Some(Box::new(service)));
    click(&mut h, "Import");
    step_until(&mut h, "the import", |a| a.editor().media().len() == 2);
    (h, script, dir)
}

pub fn item(h: &Window, name: &str) -> ItemId {
    h.state().editor().media().iter().find(|m| m.name == name).unwrap_or_else(|| panic!("no media {name}")).id
}
