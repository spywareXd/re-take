//! Cutting retakes out of a real timeline: the contract's times (seconds on the media file's own
//! clock) must land on the right frames however the clip was trimmed, moved or repeated; picture and
//! sound stay together, music is untouched, and the whole thing is one undo step.
//!
//! Skipped (with a message) when ffmpeg / ffprobe are not installed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::{have, make_clip, make_music, scratch};
use simplecut_core::retake::{Analysis, CutProposal, CutScope, Decision, PALETTE_SIZE, Presence, groups, groups_with_offset};
use simplecut_core::{Edge, Editor, EditorError, ItemId, Lane, PlaceAt, SequenceView, Tick, TrackView, TrimMode};

const TPS: f64 = 254_016_000_000.0;

fn secs(s: f64) -> Tick {
    Tick::from_seconds_f64(s)
}

fn s_of(t: Tick) -> f64 {
    t.0 as f64 / TPS
}

fn track<'a>(v: &'a SequenceView, name: &str) -> &'a TrackView {
    v.tracks.iter().find(|t| t.name == name).unwrap()
}

/// Each clip as (timeline start, duration, source start) in seconds.
fn pieces(t: &TrackView) -> Vec<(f64, f64, f64)> {
    t.clips.iter().map(|c| (s_of(c.start), s_of(c.duration), s_of(c.source_in))).collect()
}

fn same(a: &[(f64, f64, f64)], b: &[(f64, f64, f64)]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x.0 - y.0).abs() < 0.04 && (x.1 - y.1).abs() < 0.04 && (x.2 - y.2).abs() < 0.04)
}

fn proposal(id: &str, start: f64, end: f64) -> CutProposal {
    CutProposal {
        id: id.into(),
        start: secs(start),
        end: secs(end),
        confidence: 0.9,
        decision: Decision::Remove,
        reason: String::new(),
        removed_text: String::new(),
        kept_text: String::new(),
        kept: None,
        source: None,
        group: None,
    }
}

fn assert_valid(e: &Editor) {
    e.session().active_sequence().unwrap().check().unwrap();
    let v = e.sequence().unwrap();
    let (video, dialogue) = (track(&v, "Video"), track(&v, "Dialogue"));
    for c in video.clips.iter().filter(|c| c.link.is_some()) {
        let d = dialogue.clips.iter().find(|d| d.link == c.link).expect("picture kept its sound");
        assert_eq!((c.start, c.duration, c.source_in), (d.start, d.duration, d.source_in), "picture and sound stay together");
    }
    // no gaps on the main tracks
    for t in [video, dialogue] {
        for w in t.clips.windows(2) {
            assert_eq!(w[0].end(), w[1].start, "no gap left behind");
        }
    }
}

/// A project with a 10 s clip placed, and 10 s of music under it.
fn project() -> Option<(Editor, ItemId)> {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("SKIPPED: ffmpeg / ffprobe are not installed");
        return None;
    }
    let dir = scratch("retake");
    let mut e = Editor::new();
    e.new_project("T").unwrap();
    let item = e.import(&[make_clip(&dir, "a.mp4", 10, true)]).unwrap().imported[0];
    let music = e.import(&[make_music(&dir, "bed.wav", 10).unwrap()]).unwrap().imported[0];
    e.place(item, Lane::Main, PlaceAt::End).unwrap();
    e.place(music, Lane::Music, PlaceAt::At(Tick::ZERO)).unwrap();
    e.select(&[]).unwrap();
    Some((e, item))
}

#[test]
fn cuts_in_file_time_land_on_the_right_places_in_one_undo_step() {
    let Some((mut e, item)) = project() else { return };
    let before = e.sequence().unwrap();
    let steps = e.session().history.undo.len();

    let r = e.apply_cuts(item, &[proposal("a", 2.0, 3.0), proposal("b", 6.0, 7.0)], CutScope::MainTracks).unwrap();
    assert_eq!(r.cuts, 2);
    assert!((s_of(r.removed) - 2.0).abs() < 0.04, "{:?}", r.removed);
    assert!(r.notes.is_empty());

    let v = e.sequence().unwrap();
    // the file's 0-2, 3-6 and 7-10 remain, back to back
    let want = [(0.0, 2.0, 0.0), (2.0, 3.0, 3.0), (5.0, 3.0, 7.0)];
    assert!(same(&pieces(track(&v, "Video")), &want), "{:?}", pieces(track(&v, "Video")));
    assert!(same(&pieces(track(&v, "Dialogue")), &want), "its sound is cut identically");
    assert_eq!(track(&v, "Music"), track(&before, "Music"), "music is untouched");
    assert_valid(&e);
    assert_eq!(e.session().history.undo.len(), steps + 1, "one undo step");

    e.undo().unwrap();
    assert_eq!(e.sequence().unwrap().tracks, before.tracks, "undo restores exactly");
    e.redo().unwrap();
    assert!(same(&pieces(track(&e.sequence().unwrap(), "Video")), &want));
}

#[test]
fn a_trimmed_clip_maps_file_time_through_where_it_starts() {
    let Some((mut e, item)) = project() else { return };
    let clip = track(&e.sequence().unwrap(), "Video").clips[0].id;
    // show only the file's 1 s to 10 s
    e.trim(clip, Edge::Start, secs(1.0), TrimMode::Ripple).unwrap();
    let start = s_of(track(&e.sequence().unwrap(), "Video").clips[0].start);

    // 2 s to 3 s of the FILE is the second second of what is on the timeline
    e.apply_cuts(item, &[proposal("a", 2.0, 3.0)], CutScope::MainTracks).unwrap();
    let v = e.sequence().unwrap();
    let want = [(start, 1.0, 1.0), (start + 1.0, 7.0, 3.0)];
    assert!(same(&pieces(track(&v, "Video")), &want), "{:?}", pieces(track(&v, "Video")));
    assert_valid(&e);
}

#[test]
fn a_cut_partly_outside_the_used_part_is_clipped_and_one_wholly_outside_is_ignored() {
    let Some((mut e, item)) = project() else { return };
    let clip = track(&e.sequence().unwrap(), "Video").clips[0].id;
    e.trim(clip, Edge::Start, secs(2.0), TrimMode::Ripple).unwrap(); // file 2 s .. 10 s is on the timeline
    // 1 s to 3 s: only file 2-3 is on the timeline
    e.apply_cuts(item, &[proposal("edge", 1.0, 3.0)], CutScope::MainTracks).unwrap();
    let v = e.sequence().unwrap();
    let c = &track(&v, "Video").clips;
    assert_eq!(c.len(), 1);
    assert!((s_of(c[0].source_in) - 3.0).abs() < 0.04 && (s_of(c[0].duration) - 7.0).abs() < 0.04, "{:?}", pieces(track(&v, "Video")));

    // entirely before what is on the timeline: nothing to do, and it says so
    let steps = e.session().history.undo.len();
    let err = e.apply_cuts(item, &[proposal("gone", 0.2, 0.8)], CutScope::MainTracks).unwrap_err();
    assert!(matches!(err, EditorError::NotAvailable(m) if m.contains("None of those cuts")));
    assert_eq!(e.session().history.undo.len(), steps, "nothing changed, no undo step");
}

#[test]
fn a_file_used_twice_is_cut_in_both_places() {
    let Some((mut e, item)) = project() else { return };
    e.place(item, Lane::Main, PlaceAt::End).unwrap(); // the file again, after the first: 10-20 s
    e.apply_cuts(item, &[proposal("a", 4.0, 5.0)], CutScope::MainTracks).unwrap();
    let v = e.sequence().unwrap();
    let want = [(0.0, 4.0, 0.0), (4.0, 5.0, 5.0), (9.0, 4.0, 0.0), (13.0, 5.0, 5.0)];
    assert!(same(&pieces(track(&v, "Video")), &want), "{:?}", pieces(track(&v, "Video")));
    assert_valid(&e);
}

#[test]
fn overlapping_and_touching_cuts_become_one_and_a_cut_does_not_leave_a_sliver() {
    let Some((mut e, item)) = project() else { return };
    let r = e.apply_cuts(item, &[proposal("a", 2.0, 3.0), proposal("b", 2.5, 4.0), proposal("c", 4.0, 4.5)], CutScope::MainTracks).unwrap();
    assert_eq!(r.cuts, 1, "2-3, 2.5-4 and 4-4.5 are one piece, 2 to 4.5");
    assert!((s_of(r.removed) - 2.5).abs() < 0.04);
    let v = e.sequence().unwrap();
    assert!(same(&pieces(track(&v, "Video")), &[(0.0, 2.0, 0.0), (2.0, 5.5, 4.5)]), "{:?}", pieces(track(&v, "Video")));
    assert_valid(&e);
}

#[test]
fn the_scope_chooses_whether_music_is_cut_at_the_same_moments() {
    let Some((mut e, item)) = project() else { return };
    // main tracks only (the default): the picture and its sound are cut, the music plays on whole
    e.apply_cuts(item, &[proposal("a", 2.0, 3.0)], CutScope::MainTracks).unwrap();
    assert!(same(&pieces(track(&e.sequence().unwrap(), "Music")), &[(0.0, 10.0, 0.0)]));
    e.undo().unwrap();

    // all tracks: the music loses the same moments, so it stays lined up with the picture
    e.apply_cuts(item, &[proposal("a", 2.0, 3.0)], CutScope::AllTracks).unwrap();
    let v = e.sequence().unwrap();
    assert!(same(&pieces(track(&v, "Music")), &[(0.0, 2.0, 0.0), (2.0, 7.0, 3.0)]), "{:?}", pieces(track(&v, "Music")));
    assert!(same(&pieces(track(&v, "Video")), &[(0.0, 2.0, 0.0), (2.0, 7.0, 3.0)]));
}

#[test]
fn nothing_to_cut_and_hostile_values_are_refused_without_a_panic() {
    let Some((mut e, item)) = project() else { return };
    assert_eq!(e.apply_cuts(item, &[], CutScope::MainTracks).unwrap_err(), EditorError::NothingSelected);
    for (a, b) in [(Tick(i64::MAX), Tick(i64::MAX)), (Tick(-5), Tick(-1)), (secs(5.0), secs(5.0)), (secs(5.0), secs(4.0)), (Tick(i64::MIN), Tick(i64::MAX))] {
        let p = CutProposal { start: a, end: b, ..proposal("x", 1.0, 2.0) };
        let _ = e.apply_cuts(item, &[p], CutScope::MainTracks); // an error or a sensible cut, never a panic
    }
    e.session().active_sequence().unwrap().check().unwrap();
    let unknown = e.apply_cuts(ItemId(987_654), &[proposal("a", 1.0, 2.0)], CutScope::MainTracks).unwrap_err();
    assert!(matches!(unknown, EditorError::NotAvailable(m) if m.contains("None of those cuts")), "a file that is not on the timeline has nothing to cut");
}

#[test]
fn a_document_is_read_checked_against_the_file_and_applied() {
    let Some((mut e, item)) = project() else { return };
    let json = r#"{"schema":"simplecut.retakes/1","media":{"path":"x.mp4","duration_seconds":10.0},
        "proposals":[{"id":"s:0","start":2.0,"end":3.0,"confidence":0.9,"reason":"restart"},
                     {"id":"bad","start":3,"end":2,"confidence":0.5}]}"#;
    let a = e.read_analysis(json, item).unwrap();
    assert_eq!(a.proposals.len(), 1);
    assert_eq!(a.rejected.len(), 1);
    assert_eq!(a.proposals[0].reason, "restart");
    e.apply_cuts(item, &a.proposals, CutScope::MainTracks).unwrap();
    let v = e.sequence().unwrap();
    assert!(same(&pieces(track(&v, "Video")), &[(0.0, 2.0, 0.0), (2.0, 7.0, 3.0)]), "the checked proposal was applied: {:?}", pieces(track(&v, "Video")));

    // the same results for a 60 s video are not for this 10 s file
    let wrong = json.replace("10.0", "60.0");
    assert!(matches!(e.read_analysis(&wrong, item), Err(EditorError::Invalid(m)) if m.contains("different file")));
    assert_eq!(e.read_analysis(json, ItemId(987_654)).unwrap_err(), EditorError::NoSuchMedia);
}

#[test]
fn presence_reports_on_off_and_handles_speed_changed_and_frame_hold_clips() {
    let Some((mut e, item)) = project() else { return };

    let p = proposal("a", 2.0, 3.0);
    assert_eq!(e.presence(item, &p), Presence::On);
    assert!(!e.is_outside_used(item, &p));

    // Wholly outside what is on the timeline (e.g. beyond 10 s)
    let p_late = proposal("late", 12.0, 13.0);
    assert_eq!(e.presence(item, &p_late), Presence::Off);
    assert!(e.is_outside_used(item, &p_late));

    // Sliver case: 1.0 s proposal, but only 0.05 s (< 10%) covered -> Off
    let clip = track(&e.sequence().unwrap(), "Video").clips[0].id;
    // Trim start so only 2.95s to 10s is on the timeline: proposal 2.0..3.0 has only 2.95..3.0 (0.05s) on timeline
    e.trim(clip, Edge::Start, secs(2.95), TrimMode::Ripple).unwrap();
    assert_eq!(e.presence(item, &p), Presence::Off);

    // Speed-changed clip: no mapped ranges, so Off, and is_outside_used is true so not reported as already removed
    let Some((mut e2, item2)) = project() else { return };
    e2.session_mut()
        .edit_sequence("speed", |seq, _, _| {
            seq.video_tracks[0].items[0].speed = 1.5;
            Ok(())
        })
        .unwrap();
    assert_eq!(e2.presence(item2, &p), Presence::Off);
    assert!(e2.is_outside_used(item2, &p));

    // Frame-hold clip: also Off and is_outside_used is true
    let Some((mut e3, item3)) = project() else { return };
    e3.session_mut()
        .edit_sequence("hold", |seq, _, _| {
            seq.video_tracks[0].items[0].frame_hold = Some(Tick(0));
            Ok(())
        })
        .unwrap();
    assert_eq!(e3.presence(item3, &p), Presence::Off);
    assert!(e3.is_outside_used(item3, &p));
}

#[test]
fn groups_groups_proposals_by_key_ordered_by_start_with_color_modulo_and_offset() {
    let mut proposals = Vec::new();
    // Group "g_b" starts at 1.0s and 1.5s
    let mut p0 = proposal("b1", 1.0, 1.2);
    p0.group = Some("g_b".into());
    let mut p1 = proposal("b2", 1.5, 1.8);
    p1.group = Some("g_b".into());

    // Group "g_a" starts at 0.5s
    let mut p2 = proposal("a1", 0.5, 0.8);
    p2.group = Some("g_a".into());

    // Singleton without group starts at 0.2s
    let p3 = proposal("single", 0.2, 0.4);

    // Add in arbitrary order
    proposals.push(p0); // idx 0
    proposals.push(p1); // idx 1
    proposals.push(p2); // idx 2
    proposals.push(p3); // idx 3

    let analysis = Analysis { item: ItemId(1), generator: None, proposals, rejected: Vec::new() };

    let grps = groups(&analysis);
    assert_eq!(grps.len(), 3);

    // 1. Singleton without group starts earliest (0.2s) -> Retake 1, color 0, key is proposal id
    assert_eq!(grps[0].key, "single");
    assert_eq!(grps[0].label, "Retake 1");
    assert_eq!(grps[0].color, 0);
    assert_eq!(grps[0].members, vec![3]);

    // 2. Group "g_a" starts at 0.5s -> Retake 2, color 1
    assert_eq!(grps[1].key, "g_a");
    assert_eq!(grps[1].label, "Retake 2");
    assert_eq!(grps[1].color, 1);
    assert_eq!(grps[1].members, vec![2]);

    // 3. Group "g_b" starts at 1.0s -> Retake 3, color 2, 2 members ordered by start
    assert_eq!(grps[2].key, "g_b");
    assert_eq!(grps[2].label, "Retake 3");
    assert_eq!(grps[2].color, 2);
    assert_eq!(grps[2].members, vec![0, 1]);

    // Multi-file continuation with offset
    let grps2 = groups_with_offset(&analysis, 3);
    assert_eq!(grps2[0].label, "Retake 4");
    assert_eq!(grps2[0].color, 3);
    assert_eq!(grps2[1].label, "Retake 5");
    assert_eq!(grps2[1].color, 4);
    assert_eq!(grps2[2].label, "Retake 6");
    assert_eq!(grps2[2].color, 5);

    // Color modulo wraps at PALETTE_SIZE (8)
    assert_eq!(PALETTE_SIZE, 8);
    let grps_wrapped = groups_with_offset(&analysis, 7);
    assert_eq!(grps_wrapped[0].color, 7);
    assert_eq!(grps_wrapped[1].color, 0); // 8 % 8 = 0
    assert_eq!(grps_wrapped[2].color, 1); // 9 % 8 = 1

    // Determinism: calling groups multiple times yields identical results
    for _ in 0..10 {
        assert_eq!(groups(&analysis), grps);
    }
}

#[test]
fn a_group_removed_completely_and_two_adjacent_groups_removed_completely_are_one_undo_step() {
    let Some((mut e, item)) = project() else { return };

    // 1. A group removed completely:
    // Group "g1" spans 2.0 to 4.0 with two sentences: 2.0..3.0 and 3.0..4.0
    let mut p1 = proposal("p1", 2.0, 3.0);
    p1.group = Some("g1".into());
    let mut p2 = proposal("p2", 3.0, 4.0);
    p2.group = Some("g1".into());

    let steps = e.session().history.undo.len();
    let r = e.apply_cuts(item, &[p1, p2], CutScope::MainTracks).unwrap();
    assert_eq!(r.cuts, 1, "contiguous sentences of one group merge into one cut");
    assert!((s_of(r.removed) - 2.0).abs() < 0.04);
    assert_valid(&e);

    // Remaining clips on Video track: 0.0..2.0 (source 0.0) and 2.0..8.0 (source 4.0)
    let v = e.sequence().unwrap();
    let video = track(&v, "Video");
    assert_eq!(video.clips.len(), 2);
    assert!((s_of(video.clips[0].start) - 0.0).abs() < 0.04);
    assert!((s_of(video.clips[0].duration) - 2.0).abs() < 0.04);
    assert!((s_of(video.clips[0].source_in) - 0.0).abs() < 0.04);

    assert!((s_of(video.clips[1].start) - 2.0).abs() < 0.04);
    assert!((s_of(video.clips[1].duration) - 6.0).abs() < 0.04);
    assert!((s_of(video.clips[1].source_in) - 4.0).abs() < 0.04);

    // One undo step brings it back
    assert_eq!(e.session().history.undo.len(), steps + 1);
    e.undo().unwrap();
    assert_eq!(e.sequence().unwrap().tracks[0].clips.len(), 1);
    assert!((s_of(e.sequence().unwrap().duration) - 10.0).abs() < 0.05);

    // 2. Two adjacent groups removed completely:
    // Group "g1" (2.0..4.0) and Group "g2" (4.0..6.0) touch at 4.0
    let mut g1_a = proposal("g1_a", 2.0, 3.0);
    g1_a.group = Some("g1".into());
    let mut g1_b = proposal("g1_b", 3.0, 4.0);
    g1_b.group = Some("g1".into());

    let mut g2_a = proposal("g2_a", 4.0, 5.0);
    g2_a.group = Some("g2".into());
    let mut g2_b = proposal("g2_b", 5.0, 6.0);
    g2_b.group = Some("g2".into());

    let steps2 = e.session().history.undo.len();
    let r2 = e.apply_cuts(item, &[g1_a, g1_b, g2_a, g2_b], CutScope::MainTracks).unwrap();
    // They merge into one contiguous cut of 4 seconds (2.0 to 6.0)
    assert_eq!(r2.cuts, 1);
    assert!((s_of(r2.removed) - 4.0).abs() < 0.04);
    assert_valid(&e);

    let v2 = e.sequence().unwrap();
    let video2 = track(&v2, "Video");
    assert_eq!(video2.clips.len(), 2);
    // Clip 1: 0.0..2.0 (source_in 0.0)
    assert!((s_of(video2.clips[0].start) - 0.0).abs() < 0.04);
    assert!((s_of(video2.clips[0].duration) - 2.0).abs() < 0.04);
    assert!((s_of(video2.clips[0].source_in) - 0.0).abs() < 0.04);

    // Clip 2: 2.0..6.0 (duration 4.0, source_in 6.0)
    assert!((s_of(video2.clips[1].start) - 2.0).abs() < 0.04);
    assert!((s_of(video2.clips[1].duration) - 4.0).abs() < 0.04);
    assert!((s_of(video2.clips[1].source_in) - 6.0).abs() < 0.04);

    // One undo step brings it back
    assert_eq!(e.session().history.undo.len(), steps2 + 1);
    e.undo().unwrap();
    assert_eq!(e.sequence().unwrap().tracks[0].clips.len(), 1);
    assert!((s_of(e.sequence().unwrap().duration) - 10.0).abs() < 0.05);
}

#[test]
fn file_time_at_honours_in_point_speed_and_reports_gaps() {
    let Some((mut e, item)) = project() else { return };

    // 1. Plain clip: placed at 0.0s, duration 10.0s, source_in 0.0s, speed 1.0
    assert_eq!(e.file_time_at(item, secs(0.0)), Some(secs(0.0)));
    assert_eq!(e.file_time_at(item, secs(3.5)), Some(secs(3.5)));
    assert_eq!(e.file_time_at(item, secs(10.0)), Some(secs(10.0)));

    // 2. A gap / outside bounds / wrong item
    assert_eq!(e.file_time_at(item, secs(-1.0)), None, "negative timeline tick is outside");
    assert_eq!(e.file_time_at(item, secs(10.5)), None, "past clip end is a gap");
    assert_eq!(e.file_time_at(ItemId(999_999), secs(3.0)), None, "wrong item");

    // 3. After a trim: trim start of clip by 2.0s
    let clip_id = track(&e.sequence().unwrap(), "Video").clips[0].id;
    e.trim(clip_id, Edge::Start, secs(2.0), TrimMode::Ripple).unwrap();
    // Clip now starts at 0.0s, duration 8.0s, source_in 2.0s
    assert_eq!(e.file_time_at(item, secs(0.0)), Some(secs(2.0)));
    assert_eq!(e.file_time_at(item, secs(3.0)), Some(secs(5.0)));
    assert_eq!(e.file_time_at(item, secs(8.0)), Some(secs(10.0)));
    assert_eq!(e.file_time_at(item, secs(8.5)), None, "past trimmed end");

    // 4. After a razor split: split at playhead = 3.0s (which is source time 5.0s)
    e.set_playhead(secs(3.0));
    e.split_at_playhead().unwrap();
    // Clip 1: timeline 0.0..3.0, source 2.0..5.0
    // Clip 2: timeline 3.0..8.0, source 5.0..10.0
    assert_eq!(e.file_time_at(item, secs(1.5)), Some(secs(3.5)));
    assert_eq!(e.file_time_at(item, secs(3.0)), Some(secs(5.0)), "at split point matches clip 2 start");
    assert_eq!(e.file_time_at(item, secs(5.0)), Some(secs(7.0)));

    // 5. Gap between clips:
    // Move second clip to start at 4.0s (leaving a 1.0s gap from 3.0 to 4.0)
    let clip2_id = track(&e.sequence().unwrap(), "Video").clips[1].id;
    e.move_clip(clip2_id, secs(4.0)).unwrap();
    assert_eq!(e.file_time_at(item, secs(3.5)), None, "inside gap between clips");

    // 6. Speed-changed clip:
    let Some((mut e_speed, item_speed)) = project() else { return };
    e_speed
        .session_mut()
        .edit_sequence("speed", |seq, _, _| {
            seq.video_tracks[0].items[0].speed = 1.5;
            seq.video_tracks[0].items[0].source_in = secs(1.0);
            seq.video_tracks[0].items[0].duration = secs(4.0);
            Ok(())
        })
        .unwrap();
    // Clip starts at 0.0, source_in 1.0s, speed 1.5, duration 4.0s
    // At timeline 2.0s: offset = 2.0s * 1.5 = 3.0s -> file time = 1.0 + 3.0 = 4.0s
    assert_eq!(e_speed.file_time_at(item_speed, secs(2.0)), Some(secs(4.0)));
    assert_eq!(e_speed.file_time_at(item_speed, secs(0.0)), Some(secs(1.0)));
    assert_eq!(e_speed.file_time_at(item_speed, secs(4.0)), Some(secs(7.0)));
    assert_eq!(e_speed.file_time_at(item_speed, secs(4.5)), None);
}
