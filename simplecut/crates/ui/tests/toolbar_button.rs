//! The Remove Retakes button: the one primary action of the top bar. Its states (disabled, normal, hovered, keyboard-focused, loading with progress) and a screenshot
//! generator for before/after review.
//!
//! Skipped (with a message) when ffmpeg / ffprobe are not installed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::PathBuf;
use std::time::Duration;

use common::harness::{Script, Window, click, item, skip, step_until};
use common::raster::{Picture, SoftRenderer};
use common::{make_clip, scratch};
use egui::vec2;
use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT, Queryable};
use simplecut_core::{FakeService, Lane};
use simplecut_ui::{Action, SimpleCutApp};

const LABEL: &str = "Remove Retakes";

/// A window with a service, drawn by `renderer`; `with_clip` puts a 4 s clip on the timeline.
fn window(renderer: SoftRenderer, service: FakeService, with_clip: bool) -> (Window, PathBuf) {
    let dir = scratch("button");
    let clip = make_clip(&dir, "talk.mp4", 4);
    let script = Script::default();
    script.media.lock().unwrap().push_back(vec![clip]);
    let mut h = Harness::builder().with_size(vec2(1360.0, 860.0)).with_max_steps(10_000).renderer(renderer).build_eframe(move |cc| {
        let mut app = SimpleCutApp::new(&cc.egui_ctx, Box::new(script), None);
        app.set_service(Some(Box::new(service)));
        app
    });
    h.step();
    click(&mut h, "Import");
    step_until(&mut h, "the import", |a| a.editor().media().len() == 1);
    if with_clip {
        let video = item(&h, "talk.mp4");
        h.state_mut().perform(Action::AddToTimeline(video, Lane::Main));
        h.step();
        h.step();
    }
    (h, dir)
}

fn shot(h: &Window, r: &SoftRenderer) -> Picture {
    r.draw(&h.ctx, h.output(), 1360.0, 860.0, egui::Color32::from_rgb(16, 16, 20))
}

fn crop_button(h: &Window, pic: &Picture) -> Picture {
    let b = h.get_by_label(LABEL).rect();
    pic.crop_scaled((b.left() - 70.0).max(0.0) as usize, (b.top() - 14.0).max(0.0) as usize, (b.width() + 140.0) as usize, (b.height() + 28.0) as usize, 3)
}

/// Not a check: writes <dir>/<tag>_<state>.png for every state of the button (3x crop), and <tag>_window.png (the whole window, normal state). Runs only when
/// SIMPLECUT_SHOT_DIR and SIMPLECUT_SHOT_TAG are set. Drawn by a software renderer: layout-faithful, not pixel-identical to the GPU output.
#[test]
fn write_button_screenshots() {
    let (Some(dir), Ok(tag)) = (std::env::var_os("SIMPLECUT_SHOT_DIR"), std::env::var("SIMPLECUT_SHOT_TAG")) else {
        eprintln!("SKIPPED: set SIMPLECUT_SHOT_DIR and SIMPLECUT_SHOT_TAG to write screenshots");
        return;
    };
    if skip() {
        return;
    }
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let save = |name: &str, p: &Picture| p.save_png(&dir.join(format!("{tag}_{name}.png")));

    // disabled: a service, but nothing on the timeline
    let r = SoftRenderer::default();
    let (mut h, _d) = window(r.clone(), FakeService::default(), false);
    h.step();
    save("1_disabled", &crop_button(&h, &shot(&h, &r)));

    // normal, hovered, keyboard-focused: a clip on the timeline
    let r = SoftRenderer::default();
    let (mut h, _d) = window(r.clone(), FakeService::default(), true);
    h.step();
    h.step();
    let normal = shot(&h, &r);
    save("2_normal", &crop_button(&h, &normal));
    save("window", &normal);
    let centre = h.get_by_label(LABEL).rect().center();
    h.hover_at(centre);
    h.step();
    h.step();
    save("3_hover", &crop_button(&h, &shot(&h, &r)));
    h.hover_at(egui::pos2(5.0, 400.0));
    h.step();
    h.get_by_label(LABEL).focus();
    h.step();
    h.step();
    save("4_focus", &crop_button(&h, &shot(&h, &r)));

    // loading: the request is running, slowly
    let r = SoftRenderer::default();
    let slow = FakeService { steps: 10, step_delay: Duration::from_millis(400), ..Default::default() };
    let (mut h, _d) = window(r.clone(), slow, true);
    h.get_by_label(LABEL).click();
    for _ in 0..3 {
        h.step();
        std::thread::sleep(Duration::from_millis(900));
    }
    h.step();
    h.step();
    save("5_loading", &crop_button(&h, &shot(&h, &r)));
}

// ---- behaviour ----

fn disabled(h: &Window) -> bool {
    h.get_by_label(LABEL).accesskit_node().is_disabled()
}

#[test]
fn the_button_keeps_its_name_and_is_disabled_until_there_is_something_to_look_at() {
    if skip() {
        return;
    }
    let r = SoftRenderer::default();
    let (mut h, _d) = window(r.clone(), FakeService::default(), false);
    h.step();
    assert!(disabled(&h), "no clip on the timeline");
    let video = item(&h, "talk.mp4");
    h.state_mut().perform(Action::AddToTimeline(video, Lane::Main));
    h.step();
    h.step();
    assert!(!disabled(&h), "a clip is on the timeline");
}

#[test]
fn while_the_service_works_the_button_shows_progress_cannot_be_pressed_again_and_keeps_its_name() {
    if skip() {
        return;
    }
    let slow = FakeService { steps: 10, step_delay: Duration::from_millis(300), ..Default::default() };
    let (mut h, _d) = window(SoftRenderer::default(), slow, true);
    assert!(h.state().analysis_progress().is_none(), "idle: no progress");
    click(&mut h, LABEL);
    step_until(&mut h, "the request to be running", |a| a.analysis_progress().is_some());
    let (_, fraction) = h.state().analysis_progress().unwrap();
    assert!((0.0..=1.0).contains(&fraction));
    h.step();
    assert!(disabled(&h), "busy: not pressable");
    assert_eq!(h.state().review_stage(), "analyzing");
    h.get_by_label(LABEL).click(); // a second press while busy does nothing
    h.step();
    assert_eq!(h.state().review_stage(), "analyzing");
}

#[test]
fn the_button_works_from_the_keyboard() {
    if skip() {
        return;
    }
    let (mut h, _d) = window(SoftRenderer::default(), FakeService::default(), true);
    h.step();
    h.get_by_label(LABEL).focus();
    h.step();
    h.key_press(egui::Key::Enter);
    h.step();
    h.step();
    assert_ne!(h.state().review_stage(), "none", "Enter on the focused button started the analysis");
}

#[test]
fn the_blue_button_has_the_same_height_and_top_as_the_other_buttons() {
    if skip() {
        return;
    }
    let r = SoftRenderer::default();
    let (mut h, _d) = window(r, FakeService::default(), true);
    h.step();
    h.step();
    let rect = |h: &Window, l: &str| h.get_by_label(l).rect();
    let blue = rect(&h, LABEL);
    for other in ["Options", "Export", "Shortcuts"] {
        let o = rect(&h, other);
        assert_eq!((blue.top(), blue.height()), (o.top(), o.height()), "{other}: {o:?} vs the blue button {blue:?}");
    }
}
