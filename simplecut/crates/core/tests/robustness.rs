//! Headless tests of Robustness: autosave persistence, crash recovery, and offline media relinking.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use common::{have, make_clip, scratch};
use simplecut_core::{Editor, EditorError, Lane, PlaceAt};

fn skip() -> bool {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("SKIPPED: ffmpeg / ffprobe are not installed");
        return true;
    }
    false
}

#[test]
fn autosave_and_crash_recovery_journal_restores_unsaved_work() {
    if skip() {
        return;
    }
    let data_dir = scratch("autosave_data");
    let clip_dir = scratch("autosave_clips");
    let clip = make_clip(&clip_dir, "clip.mp4", 3, true);

    // Session 1: Create project, import clip, place on timeline
    let mut e1 = Editor::new();
    e1.enable_autosave(data_dir.clone()).unwrap();
    e1.new_project("RobustProject").unwrap();
    let report = e1.import(&[clip]).unwrap();
    let item_id = report.imported[0];
    e1.place(item_id, Lane::Main, PlaceAt::End).unwrap();
    assert!(e1.is_dirty(), "new placement makes session dirty");
    assert_eq!(e1.sequence().unwrap().clip_count(), 2); // video + dialogue

    // Wait briefly for debounce, sync persistence
    e1.session_mut().sync_persistence();
    std::thread::sleep(Duration::from_millis(1100));
    e1.session_mut().sync_persistence();

    // Simulate an abrupt crash (releases lock without removing journal)
    let persistence = e1.session_mut().persistence.take().expect("persistence was active");
    persistence.simulate_crash();

    // Session 2: Launch next session using the same data directory
    let mut e2 = Editor::new();
    e2.enable_autosave(data_dir).unwrap();
    let candidates = e2.recovery_candidates();
    assert_eq!(candidates.len(), 1, "found 1 recovery candidate");
    assert_eq!(candidates[0].meta.project_name, "RobustProject");

    // Recover unsaved project
    e2.recover(None).unwrap();
    assert_eq!(e2.project_name(), "RobustProject");
    assert_eq!(e2.sequence().unwrap().clip_count(), 2, "recovered 2 clips on timeline");
    assert!(e2.is_dirty(), "recovered project starts dirty");
    assert!(e2.recovery_candidates().is_empty(), "recovery candidate consumed");

    // Clean shutdown removes session journal
    e2.shutdown();
}

#[test]
fn discard_recovery_clears_candidate() {
    if skip() {
        return;
    }
    let data_dir = scratch("discard_data");
    let clip_dir = scratch("discard_clips");
    let clip = make_clip(&clip_dir, "clip.mp4", 2, true);

    let mut e1 = Editor::new();
    e1.enable_autosave(data_dir.clone()).unwrap();
    e1.new_project("DiscardProject").unwrap();
    let report = e1.import(&[clip]).unwrap();
    e1.place(report.imported[0], Lane::Main, PlaceAt::End).unwrap();

    e1.session_mut().sync_persistence();
    std::thread::sleep(Duration::from_millis(1100));
    e1.session_mut().sync_persistence();

    let persistence = e1.session_mut().persistence.take().expect("persistence was active");
    persistence.simulate_crash();

    let mut e2 = Editor::new();
    e2.enable_autosave(data_dir).unwrap();
    assert_eq!(e2.recovery_candidates().len(), 1);

    e2.discard_recovery(None).unwrap();
    assert!(e2.recovery_candidates().is_empty(), "recovery candidates cleared after discard");
    e2.shutdown();
}

#[test]
fn offline_media_relink_restores_usable_status() {
    if skip() {
        return;
    }
    let dir = scratch("relink_test");
    let clip = make_clip(&dir, "original.mp4", 3, true);

    let mut e = Editor::new();
    let report = e.import(std::slice::from_ref(&clip)).unwrap();
    let id = report.imported[0];
    let initial_entry = e.media_entry(id).unwrap();
    assert!(!initial_entry.offline);

    // Rename file to simulate offline / moved media
    let moved = dir.join("moved_target.mp4");
    std::fs::rename(&clip, &moved).unwrap();

    // Verify media is recognized as offline
    let offline_entry = e.media_entry(id).unwrap();
    assert!(offline_entry.offline);
    assert!(offline_entry.offline_reason.is_some());

    // Relink to a nonexistent path should cleanly fail
    let nonexistent = dir.join("does_not_exist.mp4");
    let err = e.relink(id, &nonexistent).unwrap_err();
    assert!(matches!(err, EditorError::FileMissing(_)));

    // Relink to moved path succeeds and restores online status
    e.relink(id, &moved).unwrap();
    let relinked_entry = e.media_entry(id).unwrap();
    assert!(!relinked_entry.offline);
    assert_eq!(relinked_entry.duration, initial_entry.duration);
    assert_eq!(relinked_entry.video, initial_entry.video);
}

#[test]
fn auto_relink_searches_folder_tree() {
    if skip() {
        return;
    }
    let dir = scratch("autorelink_test");
    let sub = dir.join("nested").join("subfolder");
    std::fs::create_dir_all(&sub).unwrap();
    let clip = make_clip(&dir, "source.mp4", 2, true);

    let mut e = Editor::new();
    let report = e.import(std::slice::from_ref(&clip)).unwrap();
    let id = report.imported[0];

    // Move to nested folder
    let moved = sub.join("source.mp4");
    std::fs::rename(&clip, &moved).unwrap();
    assert!(e.media_entry(id).unwrap().offline);

    // Auto-relink with folder tree search
    let count = e.auto_relink(&dir).unwrap();
    assert_eq!(count, 1);
    assert!(!e.media_entry(id).unwrap().offline);
}
