//! Preview and playback, headless: pictures from a snapshot, sound from the mixer, a latest-wins
//! renderer, and the player with a fake sound card that the test drives by hand.
//!
//! Skipped (with a message) when ffmpeg / ffprobe are not installed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::{Duration, Instant};

use common::{Card, have, make_clip, make_music, peak, pump_until, rms, scratch};
use simplecut_core::{Editor, EditorError, Lane, PlaceAt, PlayState, Player, PreviewRenderer, Snapshot, Tick};

fn secs(s: f64) -> Tick {
    Tick::from_seconds_f64(s)
}

fn within_a_frame(a: Tick, b: Tick) -> bool {
    (a.0 - b.0).abs() <= secs(1.0 / 30.0).0
}

/// A project with a 4 s talking-head clip (picture and a 440 Hz tone).
fn project() -> Option<(Editor, simplecut_core::ItemId, std::path::PathBuf)> {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("SKIPPED: ffmpeg / ffprobe are not installed");
        return None;
    }
    let dir = scratch("playback");
    let mut e = Editor::new();
    e.new_project("T").unwrap();
    let item = e.import(&[make_clip(&dir, "a.mp4", 4, true)]).unwrap().imported[0];
    e.place(item, Lane::Main, PlaceAt::End).unwrap();
    e.select(&[]).unwrap();
    Some((e, item, dir))
}

#[test]
fn a_snapshot_draws_the_picture_and_black_past_the_end() {
    let Some((e, item, dir)) = project() else { return };
    let snap = Snapshot::of(&e).unwrap();
    assert_eq!(snap.duration(), secs(4.0));

    let f = snap.render_frame(secs(1.0), 0.25);
    assert_eq!((f.width, f.height), (480, 270));
    assert_eq!(f.rgba.len(), 480 * 270 * 4);
    assert!(f.rgba.as_chunks::<4>().0.iter().all(|p| p[3] == 255), "opaque");
    assert!(f.rgba.as_chunks::<4>().0.iter().any(|p| p[0] > 60 || p[1] > 60 || p[2] > 60), "the colour bars are visible");

    let past = snap.render_frame(secs(10.0), 0.25);
    assert!(past.rgba.as_chunks::<4>().0.iter().all(|p| p[..3] == [0, 0, 0]), "nothing to show after the last clip");
    assert_eq!(snap.render_frame(Tick(-5), 0.25).width, 480, "a negative time is treated as the start");
    assert_eq!(snap.render_frame(secs(1.0), f32::NAN).width, 960, "a bad scale falls back to half size");

    // the library preview shows one file by itself
    let m = snap.render_media_frame(item, secs(1.0), 0.5).unwrap();
    assert_eq!((m.width, m.height), (160, 120));
    let music = make_music(&dir, "bed.wav", 3).unwrap();
    let mut e2 = Editor::new();
    e2.new_project("M").unwrap();
    let bed = e2.import(&[music]).unwrap().imported[0];
    assert!(Snapshot::of(&e2).unwrap().render_media_frame(bed, Tick::ZERO, 0.5).is_none(), "an audio file has no picture");
}

#[test]
fn a_snapshot_mixes_the_sequence_sound_including_volume_and_mute() {
    let Some((mut e, _, _)) = project() else { return };
    let snap = Snapshot::of(&e).unwrap();
    let one_second = snap.mix_audio(0, 48_000).unwrap();
    assert_eq!(one_second.channels.len(), 2);
    assert_eq!(one_second.channels[0].len(), 48_000);
    let loud = peak(&one_second.channels[0]);
    assert!(loud > 0.05, "the tone is there: {loud}");

    let after = snap.mix_audio(4 * 48_000 + 100, 1000).unwrap();
    assert!(peak(&after.channels[0]) < 1e-6, "silent after the end");

    // quieter clip
    let dialogue = e.sequence().unwrap().tracks.iter().find(|t| t.name == "Dialogue").unwrap().clone();
    e.set_clip_gain_db(dialogue.clips[0].id, -20.0).unwrap();
    let quiet = peak(&Snapshot::of(&e).unwrap().mix_audio(0, 48_000).unwrap().channels[0]);
    assert!((quiet / loud - 0.1).abs() < 0.03, "-20 dB is a tenth of the level: {quiet} vs {loud}");

    // muted track
    e.set_track_muted(dialogue.id, true).unwrap();
    let muted = peak(&Snapshot::of(&e).unwrap().mix_audio(0, 48_000).unwrap().channels[0]);
    assert!(muted < 1e-4, "{muted}");

    // the earlier snapshot is unchanged by later edits
    assert!((peak(&snap.mix_audio(0, 48_000).unwrap().channels[0]) - loud).abs() < 1e-6);
}

#[test]
fn the_renderer_delivers_the_newest_request() {
    let Some((e, _, _)) = project() else { return };
    let snap = Snapshot::of(&e).unwrap();
    let renderer = PreviewRenderer::new();
    let mut last = 0;
    for i in 0..12 {
        last = renderer.request(snap.clone(), secs(f64::from(i) * 0.25), 0.25);
    }
    let deadline = Instant::now() + Duration::from_secs(20);
    while !renderer.is_idle() {
        assert!(Instant::now() < deadline, "the renderer is stuck");
        std::thread::sleep(Duration::from_millis(5));
    }
    let r = renderer.take_result().unwrap();
    assert_eq!(r.id, last, "the newest picture wins");
    assert_eq!((r.frame.width, r.frame.height), (480, 270));
    assert!(renderer.take_result().is_none(), "a result is handed out once");
    assert!(renderer.wait_result(Duration::from_millis(30)).is_none());
}

#[test]
fn the_sound_card_drives_the_clock() {
    let Some((mut e, _, _)) = project() else { return };
    let card = Card::new(48_000);
    let mut p = Player::new(Some(Box::new(card.clone())));
    p.play(&mut e).unwrap();
    assert!(p.is_playing() && p.has_audio_clock());

    let heard = pump_until(&card, &p, 24_000);
    assert_eq!(p.tick(&mut e), PlayState::Playing);
    assert!(within_a_frame(e.playhead(), secs(0.5)), "{:?}", e.playhead());
    assert!(rms(&heard) > 0.03, "the card heard the tone: {}", rms(&heard));
    p.pause();
    assert!(!p.is_playing());
    assert_eq!(p.tick(&mut e), PlayState::Idle);
}

#[test]
fn a_late_mixer_stalls_the_picture_instead_of_drifting() {
    let Some((mut e, _, _)) = project() else { return };
    let card = Card::new(48_000);
    let mut p = Player::new(Some(Box::new(card)));
    p.play(&mut e).unwrap();
    // the card asks for nothing, so no sound is delivered and no time has passed on the clock
    std::thread::sleep(Duration::from_millis(150));
    assert_eq!(p.tick(&mut e), PlayState::Playing);
    assert_eq!(e.playhead(), Tick::ZERO, "the wall clock does not move the picture while sound is late");
}

#[test]
fn a_44100_hz_card_keeps_time_and_hears_the_tone() {
    let Some((mut e, _, _)) = project() else { return };
    let card = Card::new(44_100);
    let mut p = Player::new(Some(Box::new(card.clone())));
    p.play(&mut e).unwrap();
    let heard = pump_until(&card, &p, 44_100);
    p.tick(&mut e);
    assert!(within_a_frame(e.playhead(), secs(1.0)), "{:?}", e.playhead());
    assert!(rms(&heard) > 0.03);
}

#[test]
fn seeking_while_playing_continues_from_the_new_place() {
    let Some((mut e, _, _)) = project() else { return };
    let card = Card::new(48_000);
    let mut p = Player::new(Some(Box::new(card.clone())));
    p.play(&mut e).unwrap();
    pump_until(&card, &p, 9_600);
    p.seek(&mut e, secs(2.0));
    assert!(within_a_frame(e.playhead(), secs(2.0)));
    pump_until(&card, &p, 24_000);
    p.tick(&mut e);
    assert!(within_a_frame(e.playhead(), secs(2.5)), "{:?}", e.playhead());
}

#[test]
fn playback_stops_at_the_end_and_play_again_starts_over() {
    let Some((mut e, _, _)) = project() else { return };
    let card = Card::new(48_000);
    let mut p = Player::new(Some(Box::new(card.clone())));
    e.set_playhead(secs(3.5));
    p.play(&mut e).unwrap();
    pump_until(&card, &p, 24_000);
    assert_eq!(p.tick(&mut e), PlayState::Ended);
    assert!(!p.is_playing());
    assert_eq!(e.playhead(), secs(4.0), "the playhead rests on the end");
    p.play(&mut e).unwrap();
    assert_eq!(e.playhead(), Tick::ZERO, "play at the end begins again");
}

#[test]
fn edits_during_playback_are_heard_at_once() {
    let Some((mut e, _, _)) = project() else { return };
    let card = Card::new(48_000);
    let mut p = Player::new(Some(Box::new(card.clone())));
    p.play(&mut e).unwrap();
    let before = pump_until(&card, &p, 9_600);
    assert!(peak(&before) > 0.05);

    let dialogue = e.sequence().unwrap().tracks.iter().find(|t| t.name == "Dialogue").unwrap().id;
    e.set_track_muted(dialogue, true).unwrap();
    assert_eq!(p.tick(&mut e), PlayState::Playing);
    assert_eq!(p.restarts(), 1, "playback restarted on the edited project");
    let after = pump_until(&card, &p, 9_600);
    assert!(peak(&after) < 1e-4, "muted: {}", peak(&after));
    assert_eq!(p.tick(&mut e), PlayState::Playing);
    assert_eq!(p.restarts(), 1, "moving the playhead is not an edit");
}

#[test]
fn without_a_sound_card_the_wall_clock_plays() {
    let Some((mut e, _, _)) = project() else { return };
    for mut p in [Player::new(None), Player::new(Some(Box::new(Card::broken())))] {
        e.set_playhead(Tick::ZERO);
        p.play(&mut e).unwrap();
        assert!(!p.has_audio_clock());
        std::thread::sleep(Duration::from_millis(250));
        p.tick(&mut e);
        let t = e.playhead();
        assert!(t >= secs(0.15) && t <= secs(1.0), "{t:?}");
        p.pause();
    }
}

#[test]
fn a_card_that_never_asks_for_sound_is_given_up_on() {
    let Some((mut e, _, _)) = project() else { return };
    let mut p = Player::new(Some(Box::new(Card::new(48_000))));
    p.play(&mut e).unwrap();
    std::thread::sleep(Duration::from_millis(2300));
    p.tick(&mut e);
    assert!(!p.has_audio_clock(), "fell back to the wall clock");
    std::thread::sleep(Duration::from_millis(250));
    p.tick(&mut e);
    assert!(e.playhead() >= secs(0.15), "and time moves again: {:?}", e.playhead());
}

#[test]
fn playing_needs_something_on_the_timeline() {
    let mut p = Player::new(None);
    let mut e = Editor::new();
    assert_eq!(p.play(&mut e).unwrap_err(), EditorError::NoSequence);
    e.new_project("Empty").unwrap();
    assert!(matches!(p.play(&mut e), Err(EditorError::NotAvailable(_))));
    assert!(!p.is_playing());
}

#[test]
fn playback_skips_unticked_sentences_and_handles_edits() {
    let Some((mut e, _item, _dir)) = project() else { return };
    let card = Card::new(48_000);
    let mut p = Player::new(Some(Box::new(card.clone())));

    // Timeline duration is 4.0s.
    // Three sections: a (0.0..1.0), b (1.0..2.0), c (2.0..4.0).
    // Only b is kept, so a and c are skipped.
    p.set_skips(vec![(secs(0.0), secs(1.0)), (secs(2.0), secs(4.0))]);

    // 1. Play from start of a -> jumps over a, plays b, jumps over c, ends.
    e.set_playhead(secs(0.0));
    p.play(&mut e).unwrap();
    // Starting inside skip [0.0, 1.0) jumps playhead to 1.0
    assert!(within_a_frame(e.playhead(), secs(1.0)), "jumped over a at start: {:?}", e.playhead());
    assert_eq!(p.skips(), 1);

    // Pump audio for b (1.0s = 48_000 frames)
    let heard = pump_until(&card, &p, 48_000);
    assert!(rms(&heard) > 0.03, "b played sound");
    let state = p.tick(&mut e);
    // At 2.0s, now enters skip [2.0, 4.0) -> e=4.0 >= end -> ends playback!
    assert_eq!(state, PlayState::Ended);
    assert_eq!(p.skips(), 2, "jumped over c and ended");
    assert_eq!(e.playhead(), secs(4.0));

    // 2. Checkbox off -> no skips
    p.set_skips(vec![]);
    e.set_playhead(secs(0.0));
    p.play(&mut e).unwrap();
    assert_eq!(e.playhead(), secs(0.0), "starts at 0 when no skips");
    pump_until(&card, &p, 24_000);
    p.tick(&mut e);
    assert!(within_a_frame(e.playhead(), secs(0.5)));
    assert_eq!(p.skips(), 0, "no skips triggered when skips list is empty");
    p.pause();

    // 3. Seeking into a range lands at its end
    p.set_skips(vec![(secs(0.0), secs(1.0)), (secs(2.0), secs(4.0))]);
    p.seek(&mut e, secs(0.5));
    assert_eq!(e.playhead(), secs(1.0), "seeking into a lands at 1.0");
    p.seek(&mut e, secs(2.5));
    assert_eq!(e.playhead(), secs(4.0), "seeking into c lands at 4.0");
    p.seek(&mut e, secs(1.5));
    assert_eq!(e.playhead(), secs(1.5), "seeking into b stays at 1.5");

    // 4. Edits made during playback do not break skipping
    e.set_playhead(secs(1.0));
    p.play(&mut e).unwrap();
    pump_until(&card, &p, 9_600);
    let dialogue = e.sequence().unwrap().tracks.iter().find(|t| t.name == "Dialogue").unwrap().id;
    e.set_track_muted(dialogue, true).unwrap();
    assert_eq!(p.tick(&mut e), PlayState::Playing);
    assert_eq!(p.restarts(), 1, "restarted on edit");
    // Pump rest of b up to 2.0s (the run restarted at about 1.2 s, so 0.8 s of b is left)
    pump_until(&card, &p, 38_400);
    assert_eq!(p.tick(&mut e), PlayState::Ended);
    assert_eq!(p.skips(), 1, "c was still skipped after project edit");
}

#[test]
fn skipping_does_not_restart_the_sound_card_and_leaves_no_gap() {
    let Some((mut e, _item, _dir)) = project() else { return };
    let card = Card::new(48_000);
    let mut p = Player::new(Some(Box::new(card.clone())));
    // keep 0.0-1.0 and 2.0-4.0 of the 4 s tone; skip 1.0-2.0
    p.set_skips(vec![(secs(1.0), secs(2.0))]);
    e.set_playhead(secs(0.5));
    p.play(&mut e).unwrap();
    // 0.5 s before the skip and 0.5 s after it, in one unbroken run of the card
    let heard = pump_until(&card, &p, 48_000);
    assert_eq!(card.starts.load(std::sync::atomic::Ordering::Relaxed), 1, "the card was started once and never restarted for the skip");
    assert_eq!(p.restarts(), 0);
    assert!(heard.len() >= 96_000, "at least 1 s of stereo frames");
    // every 10 ms window has the tone: no gap at the join (0.5 s in)
    for w in heard.chunks(960) {
        assert!(rms(w) > 0.01, "a silent hole in the sound");
    }
    // the playhead is already past the skip: 0.5 s + 0.5 s of sound + the skipped second
    assert_eq!(p.tick(&mut e), PlayState::Playing);
    assert!(within_a_frame(e.playhead(), secs(2.5)), "{:?}", e.playhead());
    assert_eq!(p.skips(), 1);
    assert_eq!(card.starts.load(std::sync::atomic::Ordering::Relaxed), 1, "ticking over the skip did not restart the card either");
}
