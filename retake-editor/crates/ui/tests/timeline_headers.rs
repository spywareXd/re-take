//! The track headers (name, mute button, volume in dB) must never draw outside their cell, whatever the volume text or the window size.
//!
//! Needs no ffmpeg: a new project already has the Video, Dialogue and Music tracks.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::harness::Script;
use common::raster::SoftRenderer;
use egui::{Color32, Rect, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use retake_editor_ui::{Action, RetakeEditorApp};

type Win = Harness<'static, RetakeEditorApp>;

fn window(width: f32, height: f32) -> Win {
    window_drawn(width, height, SoftRenderer::default())
}

/// A window whose frames `renderer` can turn into a picture afterwards.
fn window_drawn(width: f32, height: f32, renderer: SoftRenderer) -> Win {
    let mut h = Harness::builder()
        .with_size(vec2(width, height))
        .with_max_steps(10_000)
        .renderer(renderer)
        .build_eframe(move |cc| RetakeEditorApp::new(&cc.egui_ctx, Box::new(Script::default()), None));
    h.step();
    h.step();
    h
}

/// The header cell of track `i`: it ends 6 px before the lanes begin (see `track_headers` in timeline.rs).
fn cell_right(h: &Win) -> f32 {
    h.state().timeline_lane(0).expect("the timeline is drawn").left() - 6.0
}

fn set_volume(h: &mut Win, track: &str, db: f64) {
    let id = h.state().editor().sequence().unwrap().tracks.iter().find(|t| t.name == track).unwrap().id;
    h.state_mut().perform(Action::SetTrackVolume(id, db));
    h.step();
    h.step();
}

fn check_track(h: &Win, name: &str) {
    let right = cell_right(h);
    let mute: Rect = h.get_by_label(&format!("Mute {name}")).rect();
    let volume: Rect = h.get_by_label(&format!("{name} volume")).rect();
    assert!(volume.right() <= right + 0.5, "{name}: the dB box ends at {} but its cell ends at {right}", volume.right());
    assert!(mute.right() <= right + 0.5, "{name}: the mute button ends at {} but its cell ends at {right}", mute.right());
    assert!(mute.right() <= volume.left() + 0.5, "{name}: the mute button and the dB box overlap");
    let label = h.query_by_label(name).map(|n| n.rect());
    if let Some(label) = label {
        assert!(label.right() <= mute.left() + 0.5, "{name}: the name runs into the mute button ({} > {})", label.right(), mute.left());
    }
}

#[test]
fn header_controls_stay_inside_their_cell_for_every_volume_text() {
    let mut h = window(1300.0, 820.0);
    for db in [0.0, -14.0, 3.5, -9.5, -60.0, 12.0] {
        set_volume(&mut h, "Dialogue", db);
        set_volume(&mut h, "Music", db);
        check_track(&h, "Dialogue");
        check_track(&h, "Music");
    }
}

#[test]
fn header_controls_stay_inside_their_cell_at_the_smallest_window() {
    for (w, hgt) in [(960.0, 600.0), (1100.0, 700.0), (1920.0, 1080.0)] {
        let mut h = window(w, hgt);
        set_volume(&mut h, "Dialogue", -60.0);
        set_volume(&mut h, "Music", -60.0);
        check_track(&h, "Dialogue");
        check_track(&h, "Music");
    }
}

/// Not a check: writes screenshots of the window (Dialogue at -14 dB) at three window sizes, for before/after review. Runs only when RETAKE_EDITOR_SHOT_DIR and RETAKE_EDITOR_SHOT_TAG
/// are set; writes <dir>/<tag>_<W>x<H>.png (the whole window) and <tag>_<W>x<H>_headers.png (the track-header corner, 3x). Drawn by a software renderer: layout-faithful, not
/// pixel-identical to the GPU output.
#[test]
fn write_screenshots() {
    let (Some(dir), Some(tag)) = (std::env::var_os("RETAKE_EDITOR_SHOT_DIR"), std::env::var("RETAKE_EDITOR_SHOT_TAG").ok()) else {
        eprintln!("SKIPPED: set RETAKE_EDITOR_SHOT_DIR and RETAKE_EDITOR_SHOT_TAG to write screenshots");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (w, hgt) in [(1360.0_f32, 860.0_f32), (1100.0, 700.0), (960.0, 600.0)] {
        let r = SoftRenderer::default();
        let mut h = window_drawn(w, hgt, r.clone());
        set_volume(&mut h, "Dialogue", -14.0);
        h.step();
        let pic = r.draw(&h.ctx, h.output(), w, hgt, Color32::from_rgb(16, 16, 20));
        let name = format!("{tag}_{}x{}", w as u32, hgt as u32);
        pic.save_png(&dir.join(format!("{name}.png")));
        let lane = h.state().timeline_lane(0).unwrap();
        let (x, y) = ((lane.left() - 196.0 - 6.0).max(0.0) as usize, (lane.top() - 36.0).max(0.0) as usize);
        pic.crop_scaled(x, y, 250, 175, 3).save_png(&dir.join(format!("{name}_headers.png")));
    }
}
