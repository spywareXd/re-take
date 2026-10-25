//! The window, driven headless: real buttons clicked by their labels, real files made with ffmpeg,
//! the editor checked underneath. File dialogs are scripted, so nothing opens on the desktop.
//!
//! Skipped (with a message) when ffmpeg / ffprobe are not installed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::PathBuf;

use common::harness::{Script, click, item, secs, skip, step_until, window, with_media};
use egui_kittest::kittest::{NodeT, Queryable};
use simplecut_core::{Lane, Tick};
use simplecut_ui::{Action, Source};

#[test]
fn a_new_window_starts_on_an_empty_project() {
    if skip() {
        return;
    }
    let mut h = window(Script::default());
    h.step();
    let app = h.state();
    assert_eq!(app.editor().project_name(), "Untitled");
    assert_eq!(app.editor().sequence().unwrap().clip_count(), 0);
    assert!(app.editor().media().is_empty());
    assert_eq!(app.source(), Source::Timeline);
    assert!(app.banner().is_none());
    // the buttons that need something to act on are disabled
    assert!(h.get_by_label("Undo").accesskit_node().is_disabled());
    assert!(h.get_by_label("Split").accesskit_node().is_disabled());
    assert!(!h.get_by_label("Import").accesskit_node().is_disabled());
    assert!(h.get_by_label("Export").accesskit_node().is_disabled(), "export arrives in a later step");
}

#[test]
fn importing_fills_the_library_with_thumbnails_and_previews_the_first_file() {
    if skip() {
        return;
    }
    let (mut h, _, _) = with_media();
    let (video, music) = (item(&h, "talk.mp4"), item(&h, "bed.wav"));
    assert_eq!(h.state().source(), Source::Media(video), "the first import is shown at once");
    assert_eq!(h.state().banner().map(|b| (b.text.as_str(), b.error)), Some(("Imported 2 files.", false)));
    // the rows are there to click
    h.get_by_label("talk.mp4");
    h.get_by_label("bed.wav");

    step_until(&mut h, "thumbnails", |a| a.thumbnail_state(video) == Some("ready") && a.thumbnail_state(music) == Some("none"));
    step_until(&mut h, "the preview picture", |a| a.preview_size().is_some_and(|[w, hgt]| w * 3 == hgt * 4));
    let [w, hgt] = h.state().preview_size().unwrap();
    assert_eq!(w * 3, hgt * 4, "the 4:3 clip keeps its shape: {w}x{hgt}");
}

#[test]
fn a_previewed_file_is_added_to_the_timeline_and_music_goes_on_its_own_track() {
    if skip() {
        return;
    }
    let (mut h, _, _) = with_media();
    let (video, music) = (item(&h, "talk.mp4"), item(&h, "bed.wav"));

    click(&mut h, "Add to timeline");
    let seq = h.state().editor().sequence().unwrap();
    assert_eq!(seq.tracks.iter().map(|t| t.clips.len()).collect::<Vec<_>>(), [1, 1, 0], "picture and its sound");
    assert_eq!(h.state().source(), Source::Timeline);
    assert_eq!(h.state().banner().map(|b| b.text.as_str()), Some("Added to the timeline."));
    step_until(&mut h, "the timeline picture", |a| a.preview_size().is_some());

    h.state_mut().perform(Action::PreviewMedia(music));
    h.step();
    click(&mut h, "Add as music");
    let seq = h.state().editor().sequence().unwrap();
    assert_eq!(seq.tracks.iter().map(|t| t.clips.len()).collect::<Vec<_>>(), [1, 1, 1]);
    assert_eq!(seq.tracks[2].name, "Music");

    // the same through the library: double-clicking a row
    h.state_mut().perform(Action::AddToTimeline(video, Lane::Main));
    assert_eq!(h.state().editor().sequence().unwrap().clip_count(), 5);
}

#[test]
fn split_undo_and_redo_buttons_edit_the_project() {
    if skip() {
        return;
    }
    let (mut h, _, _) = with_media();
    click(&mut h, "Add to timeline");
    h.state_mut().perform(Action::SeekTimeline(secs(2.0)));
    h.step();
    assert!(h.state().editor().is_dirty());

    click(&mut h, "Split");
    assert_eq!(h.state().editor().sequence().unwrap().clip_count(), 4);
    click(&mut h, "Undo");
    assert_eq!(h.state().editor().sequence().unwrap().clip_count(), 2);
    click(&mut h, "Redo");
    assert_eq!(h.state().editor().sequence().unwrap().clip_count(), 4);
}

#[test]
fn the_space_bar_plays_and_pauses_even_without_a_sound_card() {
    if skip() {
        return;
    }
    let (mut h, _, _) = with_media();
    click(&mut h, "Add to timeline");
    assert!(!h.state().player().is_playing());

    h.key_press(egui::Key::Space);
    h.step();
    assert!(h.state().player().is_playing());
    step_until(&mut h, "the playhead to move", |a| a.editor().playhead() > secs(0.15));
    h.key_press(egui::Key::Space);
    h.step();
    assert!(!h.state().player().is_playing());
    let stopped_at = h.state().editor().playhead();
    for _ in 0..5 {
        h.step();
    }
    assert_eq!(h.state().editor().playhead(), stopped_at, "paused means paused");
}

#[test]
fn clicking_the_timeline_moves_the_playhead() {
    if skip() {
        return;
    }
    let (mut h, _, _) = with_media();
    click(&mut h, "Add to timeline");
    assert_eq!(h.state().editor().playhead(), Tick::ZERO);
    click(&mut h, "Timeline lanes"); // the centre of the timeline, well to the right of its start
    assert!(h.state().editor().playhead() > secs(1.0), "{:?}", h.state().editor().playhead());
    assert_eq!(h.state().source(), Source::Timeline);
}

#[test]
fn saving_new_and_opening_ask_before_throwing_work_away() {
    if skip() {
        return;
    }
    let (mut h, script, dir) = with_media();
    click(&mut h, "Add to timeline");
    let file = dir.join("talk.fcproj");

    // Save on a project with no file asks where
    script.save.lock().unwrap().push_back(file.clone());
    click(&mut h, "Save");
    assert!(file.exists());
    assert!(!h.state().editor().is_dirty());
    assert_eq!(h.state().banner().map(|b| b.text.as_str()), Some("Saved talk.fcproj."));

    // an unsaved edit, then New: the window asks first
    h.state_mut().perform(Action::SeekTimeline(secs(2.0)));
    click(&mut h, "Split");
    assert!(h.state().editor().is_dirty());
    click(&mut h, "New");
    assert!(h.state().asking_about_unsaved());
    click(&mut h, "Cancel");
    assert!(!h.state().asking_about_unsaved());
    assert_eq!(h.state().editor().sequence().unwrap().clip_count(), 4, "cancel keeps the project");

    click(&mut h, "New");
    click(&mut h, "Discard changes");
    assert_eq!(h.state().editor().sequence().unwrap().clip_count(), 0);
    assert!(h.state().editor().media().is_empty());
    assert_eq!(h.state().source(), Source::Timeline);

    // open the saved file again: it holds the state as saved (before the split)
    script.open.lock().unwrap().push_back(file);
    click(&mut h, "Open");
    assert_eq!(h.state().editor().sequence().unwrap().clip_count(), 2);
    assert_eq!(h.state().editor().media().len(), 2);
    step_until(&mut h, "the thumbnails of the reopened project", |a| {
        a.editor().media().iter().all(|m| a.thumbnail_state(m.id).is_some_and(|s| s != "pending"))
    });
}

#[test]
fn saving_from_the_unsaved_question_saves_then_continues() {
    if skip() {
        return;
    }
    let (mut h, script, dir) = with_media();
    click(&mut h, "Add to timeline");
    let file = dir.join("keep.fcproj");
    script.save.lock().unwrap().push_back(file.clone());

    click(&mut h, "New");
    assert!(h.state().asking_about_unsaved());
    click(&mut h, "Save changes");
    assert!(file.exists(), "saved before moving on");
    assert!(!h.state().asking_about_unsaved());
    assert_eq!(h.state().editor().sequence().unwrap().clip_count(), 0, "and then it started the new project");
}

#[test]
fn problems_are_reported_in_the_window_not_thrown() {
    if skip() {
        return;
    }
    let mut h = window(Script::default());
    h.state_mut().perform(Action::ImportPaths(vec![PathBuf::from("Z:/definitely/not/here.mp4")]));
    h.step();
    let b = h.state().banner().unwrap().clone();
    assert!(b.error && b.text.contains("could not be found"), "{b:?}");
    click(&mut h, "Dismiss");
    assert!(h.state().banner().is_none());

    h.state_mut().perform(Action::Split);
    h.step();
    assert_eq!(h.state().banner().map(|b| b.text.as_str()), Some("There is nothing to split at the playhead."));
    h.state_mut().perform(Action::Delete);
    h.step();
    assert_eq!(h.state().banner().map(|b| b.text.as_str()), Some("Nothing is selected."));
    h.state_mut().perform(Action::TogglePlay);
    h.step();
    assert_eq!(h.state().banner().map(|b| b.error), Some(true), "playing an empty timeline says why not");
    h.state_mut().perform(Action::Undo);
    h.step();
    assert_eq!(h.state().banner().map(|b| b.text.as_str()), Some("Nothing to undo."));
}

#[test]
fn a_file_that_went_missing_is_flagged_and_cannot_be_added() {
    if skip() {
        return;
    }
    let (mut h, script, dir) = with_media();
    let file = dir.join("p.fcproj");
    script.save.lock().unwrap().push_back(file.clone());
    click(&mut h, "Save");
    std::fs::rename(dir.join("talk.mp4"), dir.join("moved.mp4")).unwrap();

    script.open.lock().unwrap().push_back(file);
    click(&mut h, "Open");
    let video = item(&h, "talk.mp4");
    let entry = h.state().editor().media_entry(video).unwrap();
    assert!(entry.offline && entry.offline_reason.unwrap().contains("could not be found"));
    h.state_mut().perform(Action::PreviewMedia(video));
    h.step();
    assert!(h.query_by_label("Add to timeline").is_none(), "an offline file offers no way to add it");
    h.get_by_label("Back to timeline");
}

#[test]
fn playing_a_previewed_file_leaves_the_timeline_alone() {
    if skip() {
        return;
    }
    let (mut h, _, _) = with_media();
    click(&mut h, "Add to timeline");
    h.state_mut().perform(Action::SeekTimeline(secs(1.0)));
    let music = item(&h, "bed.wav");
    h.state_mut().perform(Action::PreviewMedia(music));
    h.step();
    h.key_press(egui::Key::Space);
    h.step();
    assert!(h.state().player().is_playing());
    step_until(&mut h, "the file position to move", |a| a.player().position() > secs(0.15));
    assert_eq!(h.state().editor().playhead(), secs(1.0), "the timeline playhead stayed put");
    click(&mut h, "Back to timeline");
    assert!(!h.state().player().is_playing(), "leaving the file stops it");
    assert_eq!(h.state().source(), Source::Timeline);
}
