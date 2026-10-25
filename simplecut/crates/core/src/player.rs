//! Playback: sound drives the clock and the picture follows.
//!
//! While playing, a mixer thread renders the sequence's sound a little ahead of time into a ring
//! buffer, the sound card drains it, and the **number of sample frames actually delivered to the
//! card is the clock** (the same design FilmCraft's own player uses). If the mixer ever falls
//! behind, no samples are delivered and the clock waits with them, so sound and picture cannot
//! drift apart. Without a sound card, playback falls back to the wall clock.
//!
//! The sound card itself sits behind [`AudioOutput`]; the app supplies the real device and tests
//! supply a fake, so all of this runs headless.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use filmcraft_project::ItemId;
use filmcraft_time::Tick;

use crate::editor::Editor;
use crate::error::{EditorError, Result};
use crate::snapshot::{Snapshot, Target};

/// What the sound card calls to get its next buffer: interleaved `f32` samples to fill, and the
/// number of channels the card wants.
pub type FillFn = Box<dyn FnMut(&mut [f32], usize) + Send>;

/// A sound card (or a stand-in for one).
pub trait AudioOutput {
    /// Start output. `fill(buffer, channels)` is called on the audio thread with an interleaved
    /// `f32` buffer to fill. Returns the device's sample rate.
    fn start(&mut self, fill: FillFn) -> std::result::Result<u32, String>;
    /// Stop output. After this returns `fill` is not called again.
    fn stop(&mut self);
}

/// How far ahead of the card the mixer works.
const AHEAD: Duration = Duration::from_millis(500);
/// Sequence sample frames mixed at a time.
const BLOCK: usize = 2048;
/// If the card has delivered nothing this long after starting, give up on it and use the wall clock.
const DEVICE_PATIENCE: Duration = Duration::from_secs(2);

/// Interleaved stereo samples waiting for the card.
#[derive(Default)]
struct Ring {
    samples: Mutex<VecDeque<f32>>,
}

impl Ring {
    fn lock(&self) -> MutexGuard<'_, VecDeque<f32>> {
        self.samples.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Linear sample-rate conversion that carries its position from one block to the next.
struct Resampler {
    /// Input frames per output frame.
    step: f64,
    /// Position of the next output frame, counted from the carried-over last input frame.
    pos: f64,
    prev: [f32; 2],
    primed: bool,
}

impl Resampler {
    fn new(from: u32, to: u32) -> Self {
        let step = if to == 0 { 1.0 } else { f64::from(from) / f64::from(to) };
        Self { step, pos: 1.0, prev: [0.0; 2], primed: false }
    }

    /// Convert one block of stereo frames; returns interleaved output.
    fn process(&mut self, block: &[[f32; 2]], out: &mut Vec<f32>) {
        let Some(first) = block.first() else { return };
        if (self.step - 1.0).abs() < f64::EPSILON {
            out.extend(block.iter().flatten());
            return;
        }
        if !self.primed {
            self.prev = *first;
            self.primed = true;
        }
        let n = block.len();
        // combined input: [prev] ++ block, so index i is c(i)
        let at = |i: usize| -> [f32; 2] { if i == 0 { self.prev } else { block.get(i - 1).copied().unwrap_or(self.prev) } };
        while (self.pos.floor() as usize) < n {
            let i = self.pos.floor() as usize;
            let f = (self.pos - i as f64) as f32;
            let (a, b) = (at(i), at(i + 1));
            out.push(a[0] + (b[0] - a[0]) * f);
            out.push(a[1] + (b[1] - a[1]) * f);
            self.pos += self.step;
        }
        self.pos -= n as f64;
        self.prev = block.get(n - 1).copied().unwrap_or(self.prev);
    }
}

/// The mixer thread and the buffer it feeds.
struct Feeder {
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    /// Sample frames the card has actually taken, at the card's rate.
    delivered: Arc<AtomicU64>,
    device_rate: u32,
}

impl Drop for Feeder {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

fn make_fill(ring: Arc<Ring>, delivered: Arc<AtomicU64>) -> FillFn {
    Box::new(move |out: &mut [f32], channels: usize| {
        let ch = channels.max(1);
        let frames = out.len() / ch;
        let mut q = ring.lock();
        let n = frames.min(q.len() / 2);
        for f in 0..n {
            let l = q.pop_front().unwrap_or(0.0);
            let r = q.pop_front().unwrap_or(0.0);
            let frame = &mut out[f * ch..(f + 1) * ch];
            if ch == 1 {
                frame[0] = (l + r) * 0.5;
            } else {
                frame[0] = l;
                frame[1] = r;
                frame[2..].fill(0.0);
            }
        }
        out[n * ch..].fill(0.0);
        delivered.fetch_add(n as u64, Ordering::Relaxed);
    })
}

/// Mix ahead of the card until told to stop.
/// `skips` are sample ranges (at the sequence rate, sorted, disjoint) that are never mixed: the mixer jumps
/// over them inside the same stream, so the card never stops and there is no gap in the sound.
fn mix_loop(snapshot: Snapshot, start_sample: i64, device_rate: u32, ring: Arc<Ring>, stop: Arc<AtomicBool>, skips: Vec<(i64, i64)>) {
    let seq_rate = snapshot.sample_rate();
    let end_sample = snapshot.duration().to_units_floor(i64::from(seq_rate));
    let target = (AHEAD.as_secs_f64() * f64::from(device_rate)) as usize * 2;
    let mut resampler = Resampler::new(seq_rate, device_rate);
    let mut pos = start_sample.max(0);
    let mut out: Vec<f32> = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        if ring.lock().len() >= target || pos > end_sample + BLOCK as i64 {
            std::thread::sleep(Duration::from_millis(4));
            continue;
        }
        if let Some(&(_, e)) = skips.iter().find(|&&(s, e)| s <= pos && pos < e) {
            pos = e;
            continue;
        }
        // stop a block exactly where the next skip begins
        let n = skips.iter().find(|&&(s, _)| s > pos).map_or(BLOCK, |&(s, _)| BLOCK.min(usize::try_from(s - pos).unwrap_or(BLOCK)).max(1));
        let mixed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| snapshot.mix_audio(pos, n)));
        let block: Vec<[f32; 2]> = match mixed {
            Ok(Some(b)) => {
                let (l, r) = (b.channels.first(), b.channels.get(1).or(b.channels.first()));
                (0..n).map(|i| [l.and_then(|c| c.get(i)).copied().unwrap_or(0.0), r.and_then(|c| c.get(i)).copied().unwrap_or(0.0)]).collect()
            }
            Ok(None) => vec![[0.0; 2]; n],
            Err(_) => {
                log::error!("mixing the sound failed at sample {pos}; playing silence there");
                vec![[0.0; 2]; n]
            }
        };
        out.clear();
        resampler.process(&block, &mut out);
        ring.lock().extend(out.iter().copied());
        pos += n as i64;
    }
}

/// What a call to [`Player::tick`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayState {
    /// Not playing.
    Idle,
    /// Playing; the playhead was moved to the clock.
    Playing,
    /// Playback reached the end of the sequence and stopped.
    Ended,
}

pub struct Player {
    output: Option<Box<dyn AudioOutput>>,
    playing: bool,
    /// Playhead when the current run (or restart) began.
    origin: Tick,
    started: Option<Instant>,
    feeder: Option<Feeder>,
    snapshot: Option<Snapshot>,
    restarts: u32,
    /// Where playback is (or stopped): the sequence playhead, or the position in the media file.
    position: Tick,
    skips: Vec<(Tick, Tick)>,
    /// The skips the current run was started with (the mixer and the clock must agree on them).
    active_skips: Vec<(Tick, Tick)>,
    skips_counted: usize,
    /// `skips_counted` when the current run began.
    count_base: usize,
}

impl Player {
    /// A player on `output`, or on the wall clock alone when there is no sound card.
    pub fn new(output: Option<Box<dyn AudioOutput>>) -> Self {
        Self {
            output,
            playing: false,
            origin: Tick::ZERO,
            started: None,
            feeder: None,
            snapshot: None,
            restarts: 0,
            position: Tick::ZERO,
            skips: Vec::new(),
            active_skips: Vec::new(),
            skips_counted: 0,
            count_base: 0,
        }
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// How many skips were triggered during playback.
    pub fn skips(&self) -> usize {
        self.skips_counted
    }

    /// Current timeline skip ranges.
    pub fn current_skips(&self) -> &[(Tick, Tick)] {
        &self.skips
    }

    /// Set intervals on the timeline that playback should skip over.
    ///
    /// Ranges are filtered to non-empty within valid timeline bounds, sorted,
    /// merged when touching or overlapping, and capped in quantity.
    pub fn set_skips(&mut self, skips: Vec<(Tick, Tick)>) {
        const MAX_SKIPS: usize = 10_000;
        let mut valid: Vec<(Tick, Tick)> = skips.into_iter().filter(|&(s, e)| s.0 >= 0 && e.0 > s.0).collect();
        valid.sort_by_key(|&(s, _)| s);
        let mut merged: Vec<(Tick, Tick)> = Vec::new();
        for (s, e) in valid {
            if let Some(last) = merged.last_mut()
                && s <= last.1
            {
                last.1 = last.1.max(e);
                continue;
            }
            merged.push((s, e));
        }
        merged.truncate(MAX_SKIPS);
        self.skips = merged;
    }

    fn find_skip(&self, t: Tick) -> Option<(Tick, Tick)> {
        self.skips.iter().find(|&&(s, e)| s <= t && t < e).copied()
    }

    /// How many times playback restarted itself because the project was edited while playing.
    pub fn restarts(&self) -> u32 {
        self.restarts
    }

    /// Sample frames the sound card has taken since playback (re)started.
    pub fn delivered_frames(&self) -> u64 {
        self.feeder.as_ref().map(|f| f.delivered.load(Ordering::Relaxed)).unwrap_or(0)
    }

    /// Whether the sound card is driving the clock (false: the wall clock is).
    pub fn has_audio_clock(&self) -> bool {
        self.feeder.is_some()
    }

    /// Where playback is: the sequence playhead, or the position in the media file being previewed.
    pub fn position(&self) -> Tick {
        self.position
    }

    /// What is being played (or was last played).
    pub fn target(&self) -> Option<Target> {
        self.snapshot.as_ref().map(Snapshot::target)
    }

    /// The snapshot playback is running on.
    pub fn snapshot(&self) -> Option<&Snapshot> {
        self.snapshot.as_ref()
    }

    /// Preview one imported file from `from` (from its start when `from` is at or past its end).
    /// Moves no playhead: the edit is untouched.
    pub fn play_media(&mut self, editor: &Editor, item: ItemId, from: Tick) -> Result<()> {
        let snapshot = Snapshot::of_media(editor, item).ok_or(EditorError::NoSuchMedia)?;
        let end = snapshot.duration();
        if end <= Tick::ZERO {
            return Err(EditorError::NotAvailable("That file has nothing to play.".into()));
        }
        let from = if from < Tick::ZERO || from >= end { Tick::ZERO } else { from };
        self.halt();
        self.begin(snapshot, from);
        self.playing = true;
        Ok(())
    }

    /// Start playing from the playhead (from the start when the playhead is at the end).
    pub fn play(&mut self, editor: &mut Editor) -> Result<()> {
        let snapshot = Snapshot::of(editor).ok_or(EditorError::NoSequence)?;
        let end = snapshot.duration();
        if end <= Tick::ZERO {
            return Err(EditorError::NotAvailable("The timeline is empty.".into()));
        }
        self.skips_counted = 0;
        let mut from = editor.playhead();
        if from >= end {
            from = Tick::ZERO;
            editor.set_playhead(from);
        }
        if let Some((_s, e)) = self.find_skip(from) {
            from = e;
            self.skips_counted += 1;
            if from >= end {
                self.halt();
                self.playing = false;
                self.position = end;
                editor.set_playhead(end);
                return Ok(());
            }
            editor.set_playhead(from);
        }
        self.halt();
        self.begin(snapshot, from);
        self.playing = true;
        Ok(())
    }

    pub fn pause(&mut self) {
        self.halt();
        self.playing = false;
    }

    pub fn toggle(&mut self, editor: &mut Editor) -> Result<()> {
        if self.playing {
            self.pause();
            Ok(())
        } else {
            self.play(editor)
        }
    }

    /// Move the position in a media file; playback of that file continues from there when it was
    /// playing it. Anything else that was playing is paused.
    pub fn seek_media(&mut self, editor: &Editor, item: ItemId, to: Tick) {
        let duration = Snapshot::of_media(editor, item).map(|s| s.duration()).unwrap_or_default();
        let to = Tick(to.0.clamp(0, duration.0.max(0)));
        self.position = to;
        if self.playing && self.target() == Some(Target::Media(item)) {
            if let Some(snapshot) = Snapshot::of_media(editor, item) {
                self.halt();
                self.begin(snapshot, to);
            }
        } else {
            self.pause();
        }
    }

    /// Move the playhead; playback continues from there when it was playing the sequence.
    pub fn seek(&mut self, editor: &mut Editor, mut to: Tick) {
        if (matches!(self.target(), Some(Target::Sequence(_))) || self.snapshot.is_none())
            && let Some((_s, e)) = self.find_skip(to)
        {
            to = e;
        }
        editor.set_playhead(to);
        self.position = editor.playhead();
        if self.playing && !matches!(self.target(), Some(Target::Sequence(_))) {
            self.pause();
        } else if self.playing {
            let at = editor.playhead();
            if let Some(snapshot) = Snapshot::of(editor) {
                self.halt();
                self.begin(snapshot, at);
            }
        }
    }

    /// Call once per UI frame: moves the playhead to the clock, picks up edits made during
    /// playback, and stops at the end of the sequence.
    pub fn tick(&mut self, editor: &mut Editor) -> PlayState {
        if !self.playing {
            return PlayState::Idle;
        }
        // the card never started delivering: carry on without it
        if let (Some(f), Some(started)) = (&self.feeder, self.started)
            && f.delivered.load(Ordering::Relaxed) == 0
            && started.elapsed() > DEVICE_PATIENCE
        {
            log::warn!("the sound card delivered nothing for {DEVICE_PATIENCE:?}; playing on the wall clock");
            self.halt();
            self.started = Some(Instant::now());
        }
        let Some(snapshot) = self.snapshot.clone() else {
            self.playing = false;
            return PlayState::Idle;
        };
        let end = snapshot.duration();
        let now = self.clock();
        self.skips_counted = self.count_base + self.crossed();
        let media = matches!(snapshot.target(), Target::Media(_));
        if now >= end {
            self.halt();
            self.playing = false;
            self.position = end;
            if !media {
                editor.set_playhead(end);
            }
            return PlayState::Ended;
        }
        // the skips were changed while playing (the user ticked something): restart once with the new ones
        if !media && self.skips != self.active_skips {
            let mut from = now;
            if let Some((_s, e)) = self.find_skip(from) {
                from = e;
                self.skips_counted += 1;
            }
            if from >= end {
                self.halt();
                self.playing = false;
                self.position = end;
                editor.set_playhead(end);
                return PlayState::Ended;
            }
            self.halt();
            self.begin(snapshot, from);
            self.position = from;
            editor.set_playhead(from);
            return PlayState::Playing;
        }
        self.position = now;
        if media {
            return PlayState::Playing;
        }
        editor.set_playhead(now);
        if editor.session().revision != snapshot.revision
            && let Some(fresh) = Snapshot::of(editor)
        {
            self.halt();
            self.restarts += 1;
            self.begin(fresh, now);
        }
        PlayState::Playing
    }

    /// The playback clock: where the playhead should be now. Time that the mixer skipped is added back,
    /// so the playhead jumps over a skip at the moment the sound does.
    pub fn clock(&self) -> Tick {
        let raw = match &self.feeder {
            Some(f) => self.origin + Tick::from_units(i64::try_from(f.delivered.load(Ordering::Relaxed)).unwrap_or(i64::MAX), i64::from(f.device_rate.max(1))),
            None => self.origin + Tick::from_seconds_f64(self.started.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0)),
        };
        let mut t = raw;
        for &(s, e) in &self.active_skips {
            if e <= self.origin {
                continue;
            }
            if t < s {
                break;
            }
            t = Tick(t.0.saturating_add(e.0.saturating_sub(s.0.max(self.origin.0))));
        }
        t
    }

    /// How many skips the clock has gone over in this run.
    fn crossed(&self) -> usize {
        let raw = self.clock();
        self.active_skips.iter().filter(|&&(_, e)| e > self.origin && e <= raw).count()
    }

    fn begin(&mut self, snapshot: Snapshot, from: Tick) {
        self.origin = from;
        self.position = from;
        self.started = Some(Instant::now());
        self.active_skips = if matches!(snapshot.target(), Target::Sequence(_)) { self.skips.clone() } else { Vec::new() };
        self.count_base = self.skips_counted;
        self.feeder = self.start_feeder(&snapshot, from);
        self.snapshot = Some(snapshot);
    }

    fn start_feeder(&mut self, snapshot: &Snapshot, from: Tick) -> Option<Feeder> {
        let output = self.output.as_mut()?;
        let ring = Arc::new(Ring::default());
        let delivered = Arc::new(AtomicU64::new(0));
        let device_rate = match output.start(make_fill(ring.clone(), delivered.clone())) {
            Ok(rate) if rate > 0 => rate,
            Ok(_) => {
                output.stop();
                log::warn!("the sound card reported a sample rate of 0; playing without sound");
                return None;
            }
            Err(e) => {
                log::warn!("could not start the sound card ({e}); playing without sound");
                return None;
            }
        };
        let stop = Arc::new(AtomicBool::new(false));
        let start_sample = from.to_units_floor(i64::from(snapshot.sample_rate()));
        let (snap, r, s) = (snapshot.clone(), ring, stop.clone());
        let rate = i64::from(snapshot.sample_rate());
        let skips: Vec<(i64, i64)> = self.active_skips.iter().map(|&(a, b)| (a.to_units_floor(rate), b.to_units_floor(rate))).collect();
        let worker = std::thread::Builder::new().name("simplecut-mixer".into()).spawn(move || mix_loop(snap, start_sample, device_rate, r, s, skips));
        match worker {
            Ok(w) => Some(Feeder { stop, worker: Some(w), delivered, device_rate }),
            Err(e) => {
                log::error!("could not start the mixer thread: {e}");
                output.stop();
                None
            }
        }
    }

    /// Stop the card and the mixer; the playing flag is left to the caller.
    fn halt(&mut self) {
        if let Some(out) = self.output.as_mut() {
            out.stop();
        }
        self.feeder = None; // joins the mixer thread
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.halt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<[f32; 2]> {
        (0..n).map(|i| [i as f32, -(i as f32)]).collect()
    }

    #[test]
    fn equal_rates_pass_through_untouched() {
        let mut r = Resampler::new(48_000, 48_000);
        let mut out = Vec::new();
        r.process(&ramp(5), &mut out);
        assert_eq!(out, [0.0, 0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]);
    }

    #[test]
    fn conversion_keeps_the_duration_and_is_continuous_across_blocks() {
        // 48 kHz to 44.1 kHz over many small blocks: the right number of frames, and a ramp stays a ramp
        let mut r = Resampler::new(48_000, 44_100);
        let mut out = Vec::new();
        let data = ramp(48_000);
        for block in data.chunks(1000) {
            r.process(block, &mut out);
        }
        let frames = out.len() / 2;
        assert!((frames as i64 - 44_100).abs() <= 2, "{frames} frames out for one second in");
        let left: Vec<f32> = out.as_chunks::<2>().0.iter().map(|f| f[0]).collect();
        let max_jump = left.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        assert!(max_jump < 1.2 && left.windows(2).all(|w| w[1] > w[0]), "a ramp stays smooth and rising across block edges (max step {max_jump})");
    }

    #[test]
    fn upsampling_works_too() {
        let mut r = Resampler::new(22_050, 44_100);
        let mut out = Vec::new();
        for block in ramp(2205).chunks(300) {
            r.process(block, &mut out);
        }
        let frames = out.len() / 2;
        assert!((frames as i64 - 4410).abs() <= 2, "{frames}");
    }
}
