//! Headless tests of the `Editor` facade against real media made with ffmpeg: every operation the
//! requirements list, plus the invariants that matter (no gaps, no overlaps, picture and sound stay
//! together, one undo step per user action).
//!
//! Skipped (with a message) when ffmpeg / ffprobe are not installed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::PathBuf;

use common::{have, make_clip, make_music, scratch};
use simplecut_core::{Edge, Editor, EditorError, ItemId, Lane, MediaKind, PlaceAt, SequenceView, Tick, TrackView, TrimMode};

const TPS: f64 = 254_016_000_000.0;

fn secs(s: f64) -> Tick {
    Tick::from_seconds_f64(s)
}

/// A clip's (start, end) in whole milliseconds.
fn spans(t: &TrackView) -> Vec<(i64, i64)> {
    let ms = |x: Tick| (x.0 as f64 / TPS * 1000.0).round() as i64;
    t.clips.iter().map(|c| (ms(c.start), ms(c.end()))).collect()
}

fn track<'a>(v: &'a SequenceView, name: &str) -> &'a TrackView {
    v.tracks.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no track {name}"))
}

/// Every linked pair (picture and its sound) has the same position, length and source start.
fn assert_in_sync(v: &SequenceView) {
    let video = track(v, "Video");
    let dialogue = track(v, "Dialogue");
    for c in video.clips.iter().filter(|c| c.link.is_some()) {
        let partner = dialogue.clips.iter().find(|d| d.link == c.link).unwrap_or_else(|| panic!("clip {:?} lost its sound", c.id));
        assert_eq!((c.start, c.duration, c.source_in), (partner.start, partner.duration, partner.source_in), "picture and sound drifted apart");
    }
}

fn assert_valid(e: &Editor) {
    e.session().active_sequence().expect("a sequence").check().expect("sequence invariants (no overlaps) hold");
    assert_in_sync(&e.sequence().expect("a sequence"));
}

struct Fx {
    dir: PathBuf,
}

fn fx(name: &str) -> Option<Fx> {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("SKIPPED: ffmpeg / ffprobe are not installed");
        return None;
    }
    Some(Fx { dir: scratch(name) })
}

/// A new project with a 4 s talking-head clip on the main tracks.
fn project_with_clip(fx: &Fx, secs_long: u32) -> (Editor, ItemId) {
    let mut e = Editor::new();
    e.new_project("Test").unwrap();
    let r = e.import(&[make_clip(&fx.dir, "a.mp4", secs_long, true)]).unwrap();
    let item = r.imported[0];
    e.place(item, Lane::Main, PlaceAt::End).unwrap();
    (e, item)
}

#[test]
fn a_new_project_has_video_dialogue_and_music_tracks() {
    let mut e = Editor::new();
    assert!(e.sequence().is_none());
    e.new_project("My video").unwrap();
    let v = e.sequence().unwrap();
    let names: Vec<&str> = v.tracks.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["Video", "Dialogue", "Music"]);
    assert_eq!((v.width, v.height), (1920, 1080));
    assert_eq!(v.clip_count(), 0);
    assert_eq!(e.project_name(), "My video");
    assert!(!e.can_undo(), "the template is not something to undo");
    assert!(!e.is_dirty(), "an untouched new project has nothing to save");
    e.import(&[]).unwrap_err();
    assert!(!e.is_dirty(), "a failed action changes nothing");
}

#[test]
fn import_lists_media_with_its_metadata() {
    let Some(fx) = fx("import-meta") else { return };
    let mut e = Editor::new();
    e.new_project("T").unwrap();
    let r = e.import(&[make_clip(&fx.dir, "talk.mp4", 4, true), make_music(&fx.dir, "bed.wav", 6).unwrap()]).unwrap();
    assert_eq!(r.imported.len(), 2);
    assert!(r.failed.is_empty());

    let video = e.media_entry(r.imported[0]).unwrap();
    assert_eq!(video.kind, MediaKind::Movie);
    assert!((video.duration.0 as f64 / TPS - 4.0).abs() < 0.1, "{:?}", video.duration);
    assert_eq!(video.video.map(|(w, h, f)| (w, h, f.round() as u32)), Some((320, 240, 25)));
    assert_eq!(video.audio.map(|(_, rate)| rate), Some(48_000));
    assert!(!video.offline && video.offline_reason.is_none());
    assert!(video.path.as_deref().is_some_and(|p| p.ends_with("talk.mp4")));

    let music = e.media_entry(r.imported[1]).unwrap();
    assert_eq!(music.kind, MediaKind::AudioOnly);
    assert!(music.video.is_none() && music.audio.is_some());
    assert_eq!(e.media().len(), 2);
}

#[test]
fn import_keeps_the_good_files_and_explains_the_bad_ones() {
    let Some(fx) = fx("import-errors") else { return };
    let mut e = Editor::new();
    e.new_project("T").unwrap();
    let good = make_clip(&fx.dir, "ok.mp4", 2, true);
    let missing = fx.dir.join("nope.mp4");
    let not_media = fx.dir.join("notes.txt");
    std::fs::write(&not_media, "hello").unwrap();

    let r = e.import(&[good, missing.clone()]).unwrap();
    assert_eq!(r.imported.len(), 1);
    assert!(r.failed.iter().any(|f| f.contains("nope.mp4") && f.contains("could not be found")), "{:?}", r.failed);

    let err = e.import(std::slice::from_ref(&missing)).unwrap_err();
    assert!(matches!(&err, EditorError::ImportFailed(m) if m.contains("could not be found")), "{err}");
    assert!(e.import(&[not_media]).is_err(), "a text file is not media");
    assert!(matches!(e.import(&[]), Err(EditorError::Invalid(_))));
    assert_eq!(e.media().len(), 1, "failed imports add nothing");
}

#[test]
fn placing_video_puts_picture_and_linked_sound_end_to_end() {
    let Some(fx) = fx("place") else { return };
    let (mut e, item) = project_with_clip(&fx, 4);
    let v = e.sequence().unwrap();
    assert_eq!(spans(track(&v, "Video")), [(0, 4000)]);
    assert_eq!(spans(track(&v, "Dialogue")), [(0, 4000)]);
    assert!(track(&v, "Video").clips[0].link.is_some());
    assert_in_sync(&v);

    // a second placement lands right after the first: no gap
    e.place(item, Lane::Main, PlaceAt::End).unwrap();
    let v = e.sequence().unwrap();
    assert_eq!(spans(track(&v, "Video")), [(0, 4000), (4000, 8000)]);
    assert_eq!(spans(track(&v, "Dialogue")), [(0, 4000), (4000, 8000)]);
    assert_valid(&e);
}

#[test]
fn music_goes_on_its_own_track_and_needs_sound() {
    let Some(fx) = fx("music") else { return };
    let (mut e, _) = project_with_clip(&fx, 4);
    let bed = e.import(&[make_music(&fx.dir, "bed.wav", 10).unwrap()]).unwrap().imported[0];
    e.place(bed, Lane::Music, PlaceAt::At(Tick::ZERO)).unwrap();
    let v = e.sequence().unwrap();
    assert_eq!(spans(track(&v, "Music")), [(0, 10_000)]);
    assert_eq!(spans(track(&v, "Video")), [(0, 4000)], "the main tracks are untouched");

    let silent = e.import(&[make_clip(&fx.dir, "mute.mp4", 2, false)]).unwrap().imported[0];
    assert_eq!(e.place(silent, Lane::Music, PlaceAt::End).unwrap_err(), EditorError::NoAudio);
    assert_eq!(e.place(ItemId(987_654), Lane::Main, PlaceAt::End).unwrap_err(), EditorError::NoSuchMedia);
}

#[test]
fn splitting_cuts_picture_and_sound_but_not_the_music() {
    let Some(fx) = fx("split") else { return };
    let (mut e, _) = project_with_clip(&fx, 4);
    let bed = e.import(&[make_music(&fx.dir, "bed.wav", 10).unwrap()]).unwrap().imported[0];
    e.place(bed, Lane::Music, PlaceAt::At(Tick::ZERO)).unwrap();
    e.select(&[]).unwrap();
    let before = e.sequence().unwrap();

    e.set_playhead(secs(2.0));
    assert_eq!(e.split_at_playhead().unwrap(), 1, "one linked pair");
    let v = e.sequence().unwrap();
    assert_eq!(spans(track(&v, "Video")), [(0, 2000), (2000, 4000)]);
    assert_eq!(spans(track(&v, "Dialogue")), [(0, 2000), (2000, 4000)]);
    assert_eq!(track(&v, "Music"), track(&before, "Music"), "music is not split unless selected");
    assert_valid(&e);

    // the second piece shows the second half of the source
    let second = &track(&v, "Video").clips[1];
    assert!((second.source_in.0 as f64 / TPS - 2.0).abs() < 0.05, "{:?}", second.source_in);

    // one user action, one undo step
    e.undo().unwrap();
    assert_eq!(e.sequence().unwrap().clip_count(), before.clip_count());
    e.redo().unwrap();
    assert_eq!(e.sequence().unwrap().clip_count(), before.clip_count() + 2);
}

#[test]
fn a_selected_music_clip_is_split_when_it_is_the_selection() {
    let Some(fx) = fx("split-music") else { return };
    let (mut e, _) = project_with_clip(&fx, 4);
    let bed = e.import(&[make_music(&fx.dir, "bed.wav", 10).unwrap()]).unwrap().imported[0];
    let music = e.place(bed, Lane::Music, PlaceAt::At(Tick::ZERO)).unwrap();
    e.select(&music).unwrap();
    e.set_playhead(secs(3.0));
    assert_eq!(e.split_at_playhead().unwrap(), 1);
    let v = e.sequence().unwrap();
    assert_eq!(spans(track(&v, "Music")), [(0, 3000), (3000, 10_000)]);
    assert_eq!(spans(track(&v, "Video")), [(0, 4000)], "only the selection was cut");
}

#[test]
fn splitting_where_there_is_nothing_says_so() {
    let Some(fx) = fx("split-none") else { return };
    let (mut e, _) = project_with_clip(&fx, 4);
    e.select(&[]).unwrap();
    for t in [0.0, 4.0, 9.0] {
        e.set_playhead(secs(t));
        assert_eq!(e.split_at_playhead().unwrap_err(), EditorError::NothingToSplit, "at {t} s");
    }
    assert_eq!(e.sequence().unwrap().clip_count(), 2, "nothing changed");
}

#[test]
fn trimming_shortens_picture_and_sound_together_with_or_without_closing_the_gap() {
    let Some(fx) = fx("trim") else { return };
    let (mut e, item) = project_with_clip(&fx, 4);
    e.place(item, Lane::Main, PlaceAt::End).unwrap();
    let first = track(&e.sequence().unwrap(), "Video").clips[0].id;

    // regular: the clip gets shorter and leaves a gap
    e.trim(first, Edge::End, secs(-1.0), TrimMode::Regular).unwrap();
    let v = e.sequence().unwrap();
    assert_eq!(spans(track(&v, "Video")), [(0, 3000), (4000, 8000)]);
    assert_eq!(spans(track(&v, "Dialogue")), [(0, 3000), (4000, 8000)]);
    assert_valid(&e);
    e.undo().unwrap();

    // ripple: the next clip follows the edge
    e.trim(first, Edge::End, secs(-1.0), TrimMode::Ripple).unwrap();
    let v = e.sequence().unwrap();
    assert_eq!(spans(track(&v, "Video")), [(0, 3000), (3000, 7000)]);
    assert_eq!(spans(track(&v, "Dialogue")), [(0, 3000), (3000, 7000)]);
    assert_valid(&e);
    e.undo().unwrap();

    // the start edge: the clip begins later in its source
    e.trim(first, Edge::Start, secs(1.0), TrimMode::Ripple).unwrap();
    let v = e.sequence().unwrap();
    let c = &track(&v, "Video").clips[0];
    assert!((c.source_in.0 as f64 / TPS - 1.0).abs() < 0.05, "{:?}", c.source_in);
    assert_valid(&e);
}

#[test]
fn deleting_a_piece_closes_the_gap_and_leaves_the_music_alone() {
    let Some(fx) = fx("delete") else { return };
    let (mut e, _) = project_with_clip(&fx, 4);
    let bed = e.import(&[make_music(&fx.dir, "bed.wav", 10).unwrap()]).unwrap().imported[0];
    e.place(bed, Lane::Music, PlaceAt::At(Tick::ZERO)).unwrap();
    e.select(&[]).unwrap();
    for t in [1.0, 3.0] {
        e.set_playhead(secs(t));
        e.split_at_playhead().unwrap();
    }
    let v = e.sequence().unwrap();
    assert_eq!(spans(track(&v, "Video")), [(0, 1000), (1000, 3000), (3000, 4000)]);
    let middle = track(&v, "Video").clips[1].id;

    e.select(&[middle]).unwrap();
    e.delete_selected().unwrap();
    let v = e.sequence().unwrap();
    assert_eq!(spans(track(&v, "Video")), [(0, 1000), (1000, 2000)], "no gap where the piece was");
    assert_eq!(spans(track(&v, "Dialogue")), [(0, 1000), (1000, 2000)], "its sound went with it");
    assert_eq!(spans(track(&v, "Music")), [(0, 10_000)], "music does not follow the ripple");
    assert_valid(&e);

    assert_eq!(e.delete(&[]).unwrap_err(), EditorError::NothingSelected);
    e.undo().unwrap();
    assert_eq!(spans(track(&e.sequence().unwrap(), "Video")).len(), 3);
}

#[test]
fn moving_a_clip_takes_its_sound_along_and_never_overlaps() {
    let Some(fx) = fx("move") else { return };
    let (mut e, item) = project_with_clip(&fx, 4);
    e.place(item, Lane::Main, PlaceAt::End).unwrap();
    let second = track(&e.sequence().unwrap(), "Video").clips[1].id;

    e.move_clip(second, secs(10.0)).unwrap();
    let v = e.sequence().unwrap();
    assert_eq!(spans(track(&v, "Video")), [(0, 4000), (10_000, 14_000)]);
    assert_eq!(spans(track(&v, "Dialogue")), [(0, 4000), (10_000, 14_000)]);
    assert_valid(&e);

    // dropping it onto the first clip: whatever the engine decides, the timeline stays valid
    e.move_clip(second, secs(2.0)).unwrap();
    assert_valid(&e);
    assert!(e.move_clip(second, secs(1.0)).is_ok());
    assert_eq!(e.move_clip(simplecut_core::ClipId(987_654), secs(1.0)).unwrap_err(), EditorError::NoSuchClip);
}

#[test]
fn volume_and_mute_change_and_undo() {
    let Some(fx) = fx("audio") else { return };
    let (mut e, _) = project_with_clip(&fx, 4);
    let bed = e.import(&[make_music(&fx.dir, "bed.wav", 10).unwrap()]).unwrap().imported[0];
    let music_clip = e.place(bed, Lane::Music, PlaceAt::At(Tick::ZERO)).unwrap()[0];
    let v = e.sequence().unwrap();
    let music_track = track(&v, "Music").id;
    let dialogue_clip = track(&v, "Dialogue").clips[0].id;

    e.set_clip_gain_db(music_clip, -12.0).unwrap();
    e.set_clip_gain_db(dialogue_clip, 3.0).unwrap();
    let v = e.sequence().unwrap();
    assert_eq!(v.clip(music_clip).unwrap().gain_db, -12.0);
    assert_eq!(v.clip(dialogue_clip).unwrap().gain_db, 3.0);

    e.set_track_muted(music_track, true).unwrap();
    assert!(track(&e.sequence().unwrap(), "Music").muted);
    e.set_track_muted(music_track, false).unwrap();
    assert!(!track(&e.sequence().unwrap(), "Music").muted);
    e.set_track_volume_db(music_track, -6.0).unwrap();
    assert_eq!(track(&e.sequence().unwrap(), "Music").volume_db, -6.0);

    e.undo().unwrap();
    assert_eq!(track(&e.sequence().unwrap(), "Music").volume_db, 0.0);

    assert!(matches!(e.set_clip_gain_db(music_clip, f64::NAN), Err(EditorError::Invalid(_))));
    assert!(matches!(e.set_track_volume_db(music_track, f64::INFINITY), Err(EditorError::Invalid(_))));
    assert!(matches!(e.set_track_muted(simplecut_core::TrackId(987_654), true), Err(EditorError::Invalid(_))));
}

#[test]
fn undo_and_redo_walk_back_and_forward() {
    let Some(fx) = fx("undo") else { return };
    let mut e = Editor::new();
    e.new_project("T").unwrap();
    assert!(!e.can_undo() && !e.can_redo());
    let item = e.import(&[make_clip(&fx.dir, "a.mp4", 4, true)]).unwrap().imported[0];
    e.place(item, Lane::Main, PlaceAt::End).unwrap();
    e.select(&[]).unwrap();
    e.set_playhead(secs(2.0));
    e.split_at_playhead().unwrap();
    assert_eq!(e.sequence().unwrap().clip_count(), 4);

    assert!(e.can_undo());
    e.undo().unwrap();
    assert_eq!(e.sequence().unwrap().clip_count(), 2, "the split is undone");
    e.undo().unwrap();
    assert_eq!(e.sequence().unwrap().clip_count(), 0, "so is the placement");
    assert!(e.can_redo());
    e.redo().unwrap();
    e.redo().unwrap();
    assert_eq!(e.sequence().unwrap().clip_count(), 4);
    assert!(!e.can_redo());
    assert_valid(&e);
}

#[test]
fn projects_save_and_reopen_with_their_edits_and_media() {
    let Some(fx) = fx("save") else { return };
    let (mut e, _) = project_with_clip(&fx, 4);
    let bed = e.import(&[make_music(&fx.dir, "bed.wav", 10).unwrap()]).unwrap().imported[0];
    e.place(bed, Lane::Music, PlaceAt::At(Tick::ZERO)).unwrap();
    e.select(&[]).unwrap();
    e.set_playhead(secs(2.0));
    e.split_at_playhead().unwrap();

    assert_eq!(e.save().unwrap_err(), EditorError::NeedsPath, "a new project has nowhere to save to yet");
    let file = fx.dir.join("talk.fcproj");
    e.save_as(&file).unwrap();
    assert!(!e.is_dirty());
    assert_eq!(e.project_path().map(|p| p.ends_with("talk.fcproj")), Some(true));
    e.trim(track(&e.sequence().unwrap(), "Video").clips[1].id, Edge::End, secs(-1.0), TrimMode::Ripple).unwrap();
    assert!(e.is_dirty());
    e.save().unwrap();
    assert!(!e.is_dirty());

    let mut again = Editor::new();
    again.open(&file).unwrap();
    let (a, b) = (e.sequence().unwrap(), again.sequence().unwrap());
    for name in ["Video", "Dialogue", "Music"] {
        let strip = |t: &TrackView| t.clips.iter().map(|c| (c.item, c.start, c.duration, c.source_in, c.link.is_some(), c.gain_db)).collect::<Vec<_>>();
        assert_eq!(strip(track(&a, name)), strip(track(&b, name)), "{name} survives a save and reopen");
    }
    assert_eq!(e.media().len(), again.media().len());
    assert!(again.media().iter().all(|m| !m.offline));
    assert_valid(&again);
}

#[test]
fn opening_a_missing_or_broken_project_fails_cleanly() {
    let Some(fx) = fx("open-errors") else { return };
    let mut e = Editor::new();
    assert!(matches!(e.open(&fx.dir.join("none.fcproj")), Err(EditorError::FileMissing(_))));
    let junk = fx.dir.join("junk.fcproj");
    std::fs::write(&junk, "this is not a project").unwrap();
    assert!(e.open(&junk).is_err());
    assert!(matches!(e.open(&fx.dir), Err(EditorError::FileUnreadable(_))), "a folder is not a project");
    // the editor is still usable
    e.new_project("After").unwrap();
    assert!(e.sequence().is_some());
}

#[test]
fn media_that_goes_missing_is_flagged_offline_but_the_project_still_opens() {
    let Some(fx) = fx("offline") else { return };
    let (mut e, _) = project_with_clip(&fx, 4);
    let file = fx.dir.join("p.fcproj");
    e.save_as(&file).unwrap();
    std::fs::rename(fx.dir.join("a.mp4"), fx.dir.join("moved.mp4")).unwrap();

    let mut again = Editor::new();
    again.open(&file).unwrap();
    let m = again.media();
    assert_eq!(m.len(), 1);
    assert!(m[0].offline, "{:?}", m[0]);
    assert!(m[0].offline_reason.as_deref().is_some_and(|r| r.contains("could not be found")), "{:?}", m[0].offline_reason);
    assert_eq!(again.sequence().unwrap().clip_count(), 2, "the edit is intact");
}

#[test]
fn mp3_music_imports_and_places() {
    let Some(fx) = fx("mp3") else { return };
    let Some(mp3) = make_music(&fx.dir, "song.mp3", 6) else {
        eprintln!("SKIPPED: this ffmpeg cannot encode MP3");
        return;
    };
    let (mut e, _) = project_with_clip(&fx, 4);
    let song = e.import(&[mp3]).unwrap().imported[0];
    assert_eq!(e.media_entry(song).unwrap().kind, MediaKind::AudioOnly);
    e.place(song, Lane::Music, PlaceAt::At(Tick::ZERO)).unwrap();
    assert_eq!(track(&e.sequence().unwrap(), "Music").clips.len(), 1);
}

#[test]
fn selecting_a_clip_selects_what_moves_with_it_and_volume_applies_to_the_group() {
    let Some(fx) = fx("select") else { return };
    let (mut e, _) = project_with_clip(&fx, 4);
    let v = e.sequence().unwrap();
    let picture = track(&v, "Video").clips[0].id;
    let sound = track(&v, "Dialogue").clips[0].id;

    e.select(&[picture]).unwrap();
    let mut sel = e.selection().to_vec();
    sel.sort_by_key(|c| c.0);
    let mut both = vec![picture, sound];
    both.sort_by_key(|c| c.0);
    assert_eq!(sel, both, "the picture's own sound is selected with it");
    let v = e.sequence().unwrap();
    assert!(v.clip(picture).unwrap().selected && v.clip(sound).unwrap().selected);

    // volume: the picture is skipped, its sound changes, in one undo step
    let selection = e.selection().to_vec();
    e.set_clips_gain_db(&selection, -9.0).unwrap();
    let v = e.sequence().unwrap();
    assert_eq!(v.clip(sound).unwrap().gain_db, -9.0);
    e.undo().unwrap();
    assert_eq!(e.sequence().unwrap().clip(sound).unwrap().gain_db, 0.0);

    e.select(&[]).unwrap();
    assert!(e.selection().is_empty());
    assert!(matches!(e.set_clips_gain_db(&[picture], -3.0), Err(EditorError::NotAvailable(_))), "a picture alone has no volume");
    assert!(matches!(e.set_clips_gain_db(&[sound], f64::NAN), Err(EditorError::Invalid(_))));
}
