//! UI tests of Robustness: offline media relinking, crash recovery dialog, keyboard shortcuts modal, and crash logging.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use common::harness::{Script, click, item, window, with_media};
use common::{have, make_clip, scratch};
use egui::Key;
use egui_kittest::kittest::Queryable;
use simplecut_core::{Editor, Lane, PlaceAt};
use simplecut_ui::Action;

fn skip() -> bool {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("SKIPPED: ffmpeg / ffprobe are not installed");
        return true;
    }
    false
}

#[test]
fn offline_media_relink_flow_via_preview() {
    if skip() {
        return;
    }
    let (mut h, script, dir) = with_media();
    let file = dir.join("proj.fcproj");
    script.save.lock().unwrap().push_back(file.clone());
    click(&mut h, "Save");

    // Rename file to make it offline
    let orig = dir.join("talk.mp4");
    let moved = dir.join("talk_relocated.mp4");
    std::fs::rename(&orig, &moved).unwrap();

    // Reopen project
    script.open.lock().unwrap().push_back(file);
    click(&mut h, "Open");

    let video = item(&h, "talk.mp4");
    assert!(h.state().editor().media_entry(video).unwrap().offline);

    // Preview the offline clip
    h.state_mut().perform(Action::PreviewMedia(video));
    h.step();

    // "Relink media…" button is present in preview
    script.relink.lock().unwrap().push_back(moved.clone());
    click(&mut h, "Relink media…");

    // Media is relinked and "Add to timeline" is now enabled
    assert!(!h.state().editor().media_entry(video).unwrap().offline);
    let banner = h.state().banner().unwrap();
    assert!(!banner.error);
    assert!(banner.text.contains("Relinked talk.mp4"));

    click(&mut h, "Add to timeline");
    assert_eq!(h.state().editor().sequence().unwrap().clip_count(), 2);
}

#[test]
fn keyboard_shortcuts_modal_toggle_and_f1() {
    if skip() {
        return;
    }
    let mut h = window(Script::default());

    // 1. Click "Shortcuts" button in the toolbar
    click(&mut h, "Shortcuts");
    assert!(h.state().showing_shortcuts());
    h.get_by_label("Keyboard Shortcuts");
    h.get_by_label("Play / Pause");
    h.get_by_label("Split clip at playhead");

    // 2. Close modal
    click(&mut h, "Close shortcuts");
    assert!(!h.state().showing_shortcuts());

    // 3. Open with F1 key
    h.key_press(Key::F1);
    h.step();
    assert!(h.state().showing_shortcuts());
    h.get_by_label("Keyboard Shortcuts");
    click(&mut h, "Close shortcuts");
    assert!(!h.state().showing_shortcuts());
}

#[test]
fn crash_recovery_modal_flow() {
    if skip() {
        return;
    }
    let data_dir = scratch("ui_recovery_data");
    let clip_dir = scratch("ui_recovery_clips");
    let clip = make_clip(&clip_dir, "clip.mp4", 3);

    // Create a crashed session with unsaved changes
    let mut e1 = Editor::new();
    e1.enable_autosave(data_dir.clone()).unwrap();
    e1.new_project("RecoverMe").unwrap();
    let report = e1.import(&[clip]).unwrap();
    e1.place(report.imported[0], Lane::Main, PlaceAt::End).unwrap();

    e1.session_mut().sync_persistence();
    std::thread::sleep(Duration::from_millis(1100));
    e1.session_mut().sync_persistence();

    let persistence = e1.session_mut().persistence.take().expect("persistence was active");
    persistence.simulate_crash();

    // Launch UI window and enable autosave on same data dir
    let mut h = window(Script::default());
    h.state_mut().enable_autosave(data_dir);
    h.step();

    // Recovery modal should appear
    assert!(h.state().asking_about_recovery());
    h.get_by_label("Recover unsaved work");

    // Click "Recover work"
    click(&mut h, "Recover work");
    assert!(!h.state().asking_about_recovery());
    assert_eq!(h.state().editor().project_name(), "RecoverMe");
    assert_eq!(h.state().editor().sequence().unwrap().clip_count(), 2);
}

#[test]
fn write_crash_log_creates_log_file() {
    let dir = scratch("crash_log_test");
    let proj = dir.join("test_proj.fcproj");
    std::fs::write(&proj, "{}").unwrap();

    simplecut_ui::write_crash_log("Simulated test error", Some(proj.to_str().unwrap()));

    let crash_file = dir.join("simplecut-crash.log");
    assert!(crash_file.exists());
    let content = std::fs::read_to_string(&crash_file).unwrap();
    assert!(content.contains("Simulated test error"));
    assert!(content.contains("test_proj.fcproj"));
}
