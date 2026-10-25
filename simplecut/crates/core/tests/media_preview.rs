//! Previewing one imported file on its own (the library's preview) and the library thumbnails.
//!
//! Skipped (with a message) when ffmpeg / ffprobe are not installed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use common::{Card, have, make_clip, make_music, peak, pump_until, rms, scratch};
use simplecut_core::{Editor, EditorError, ItemId, Lane, PlaceAt, PlayState, Player, Snapshot, Target, Thumbnails, Tick};

fn secs(s: f64) -> Tick {
    Tick::from_seconds_f64(s)
}

fn near(a: Tick, b: Tick) -> bool {
    (a.0 - b.0).abs() <= secs(1.0 / 25.0).0
}

/// A project holding a 4 s clip (placed) and a 6 s music file (imported only).
fn project() -> Option<(Editor, ItemId, ItemId)> {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("SKIPPED: ffmpeg / ffprobe are not installed");
        return None;
    }
    let dir = scratch("media-preview");
    let mut e = Editor::new();
    e.new_project("T").unwrap();
    let video = e.import(&[make_clip(&dir, "a.mp4", 4, true)]).unwrap().imported[0];
    let music = e.import(&[make_music(&dir, "bed.wav", 6).unwrap()]).unwrap().imported[0];
    e.place(video, Lane::Main, PlaceAt::End).unwrap();
    e.select(&[]).unwrap();
    Some((e, video, music))
}

#[test]
fn a_media_snapshot_answers_for_the_file_alone() {
    let Some((e, video, music)) = project() else { return };
    let snap = Snapshot::of_media(&e, video).unwrap();
    assert_eq!(snap.target(), Target::Media(video));
    assert!(near(snap.duration(), secs(4.0)), "{:?}", snap.duration());
    assert_eq!(snap.sample_rate(), 48_000);
    let f = snap.render_frame(secs(1.0), 0.5);
    assert_eq!((f.width, f.height), (160, 120));
    assert!(peak(&snap.mix_audio(0, 48_000).unwrap().channels[0]) > 0.05);

    // an audio file: no picture (black stand-in), sound yes
    let m = Snapshot::of_media(&e, music).unwrap();
    assert!(near(m.duration(), secs(6.0)));
    assert!(m.render_frame(Tick::ZERO, 0.5).rgba.as_chunks::<4>().0.iter().all(|p| p[..3] == [0, 0, 0]));
    assert!(peak(&m.mix_audio(0, 48_000).unwrap().channels[0]) > 0.05);

    // the sequence is not a media file
    let seq = Snapshot::of(&e).unwrap();
    let Target::Sequence(id) = seq.target() else { panic!("not a sequence") };
    assert!(Snapshot::of_media(&e, id).is_none());
    assert!(Snapshot::of_media(&e, ItemId(987_654)).is_none());
}

#[test]
fn thumbnails_arrive_once_per_file_and_audio_has_none() {
    let Some((e, video, music)) = project() else { return };
    let thumbs = Thumbnails::new();
    let snap = Snapshot::of(&e).unwrap();
    thumbs.request(snap.clone(), video);
    thumbs.request(snap.clone(), music);
    thumbs.request(snap.clone(), video); // asking again changes nothing
    assert!(thumbs.wait_idle(Duration::from_secs(30)));

    let mut ready = thumbs.take_ready();
    ready.sort_by_key(|(id, _)| id.0);
    assert_eq!(ready.len(), 2, "one each");
    let picture = ready.iter().find(|(id, _)| *id == video).unwrap().1.as_ref().unwrap();
    assert_eq!(picture.width * 3, picture.height * 4, "the 4:3 picture keeps its shape");
    assert!(picture.width <= 160, "a thumbnail is small: {} wide", picture.width);
    assert!(picture.rgba.as_chunks::<4>().0.iter().any(|p| p[0] > 60 || p[1] > 60 || p[2] > 60), "not black");
    assert!(ready.iter().find(|(id, _)| *id == music).unwrap().1.is_none(), "audio has no picture");
    assert!(thumbs.take_ready().is_empty(), "handed over once");

    // forgetting allows a second try (after a relink)
    thumbs.forget(video);
    thumbs.request(snap, video);
    assert!(thumbs.wait_idle(Duration::from_secs(30)));
    assert_eq!(thumbs.take_ready().len(), 1);
}

#[test]
fn previewing_a_file_plays_it_without_touching_the_edit() {
    let Some((mut e, video, _)) = project() else { return };
    e.set_playhead(secs(1.0));
    let card = Card::new(48_000);
    let mut p = Player::new(Some(Box::new(card.clone())));
    p.play_media(&e, video, Tick::ZERO).unwrap();
    assert_eq!(p.target(), Some(Target::Media(video)));

    let heard = pump_until(&card, &p, 24_000);
    assert_eq!(p.tick(&mut e), PlayState::Playing);
    assert!(near(p.position(), secs(0.5)), "{:?}", p.position());
    assert_eq!(e.playhead(), secs(1.0), "the timeline playhead did not move");
    assert!(rms(&heard) > 0.03);

    // jump around inside the file
    p.seek_media(&e, video, secs(2.0));
    assert!(near(p.position(), secs(2.0)));
    pump_until(&card, &p, 12_000);
    p.tick(&mut e);
    assert!(near(p.position(), secs(2.25)), "{:?}", p.position());
    assert_eq!(e.playhead(), secs(1.0));
}

#[test]
fn a_previewed_file_stops_at_its_end_and_audio_only_files_play() {
    let Some((mut e, video, music)) = project() else { return };
    let card = Card::new(48_000);
    let mut p = Player::new(Some(Box::new(card.clone())));
    p.play_media(&e, video, secs(3.8)).unwrap();
    pump_until(&card, &p, 9_600);
    assert_eq!(p.tick(&mut e), PlayState::Ended);
    assert!(!p.is_playing());
    assert!(near(p.position(), secs(4.0)), "{:?}", p.position());

    p.play_media(&e, music, Tick::ZERO).unwrap();
    let heard = pump_until(&card, &p, 24_000);
    assert!(rms(&heard) > 0.03, "the music is heard: {}", rms(&heard));
    assert_eq!(p.tick(&mut e), PlayState::Playing);
}

#[test]
fn moving_the_timeline_playhead_stops_a_file_preview() {
    let Some((mut e, video, _)) = project() else { return };
    let mut p = Player::new(None);
    p.play_media(&e, video, Tick::ZERO).unwrap();
    assert!(p.is_playing());
    p.seek(&mut e, secs(2.0));
    assert!(!p.is_playing(), "scrubbing the timeline leaves the file preview");
    assert_eq!(e.playhead(), secs(2.0));
    assert!(near(p.position(), secs(2.0)));

    // and the other way round
    p.play(&mut e).unwrap();
    p.seek_media(&e, video, secs(1.0));
    assert!(!p.is_playing());
}

#[test]
fn previewing_something_that_is_not_media_is_refused() {
    let Some((e, _, _)) = project() else { return };
    let mut p = Player::new(None);
    assert_eq!(p.play_media(&e, ItemId(987_654), Tick::ZERO).unwrap_err(), EditorError::NoSuchMedia);
    let Target::Sequence(seq) = Snapshot::of(&e).unwrap().target() else { panic!("not a sequence") };
    assert_eq!(p.play_media(&e, seq, Tick::ZERO).unwrap_err(), EditorError::NoSuchMedia);
    assert!(!p.is_playing());
}
