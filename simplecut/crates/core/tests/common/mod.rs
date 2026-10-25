//! Shared test fixtures: small media files made with ffmpeg (a test-only oracle, never linked).
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use simplecut_core::{AudioOutput, Player};

pub fn have(tool: &str) -> bool {
    Command::new(tool).arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// A scratch directory under cargo's target tmp, unique per call (tests run in parallel).
pub fn scratch(name: &str) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("scratch dir");
    d
}

/// Whole seconds of 320x240 colour bars at 25 fps, with a 440 Hz tone when `audio`. H.264 + AAC.
pub fn make_clip(dir: &Path, file: &str, secs: u32, audio: bool) -> PathBuf {
    let out = dir.join(file);
    let mut c = Command::new("ffmpeg");
    c.args(["-y", "-v", "error", "-f", "lavfi", "-i"]).arg(format!("testsrc=size=320x240:rate=25:duration={secs}"));
    if audio {
        c.args(["-f", "lavfi", "-i"]).arg(format!("sine=frequency=440:sample_rate=48000:duration={secs}"));
    }
    c.args(["-c:v", "libx264", "-pix_fmt", "yuv420p"]);
    if audio {
        c.args(["-c:a", "aac", "-shortest"]);
    }
    let status = c.arg(&out).status().expect("run ffmpeg");
    assert!(status.success(), "ffmpeg could not make {file}");
    out
}

/// Stereo 48 kHz tone as audio-only `file` (the extension picks the format). None when this
/// ffmpeg cannot encode that format.
pub fn make_music(dir: &Path, file: &str, secs: u32) -> Option<PathBuf> {
    let out = dir.join(file);
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-f", "lavfi", "-i"])
        .arg(format!("sine=frequency=220:sample_rate=48000:duration={secs}"))
        .args(["-ac", "2"])
        .arg(&out)
        .status()
        .ok()?;
    status.success().then_some(out)
}

pub type Fill = simplecut_core::FillFn;

/// A sound card the test drives: `pump` is the card asking for samples.
#[derive(Clone)]
pub struct Card {
    pub rate: u32,
    pub broken: bool,
    pub fill: Arc<Mutex<Option<Fill>>>,
    /// How many times the card was started (a restart of playback starts it again).
    pub starts: Arc<std::sync::atomic::AtomicUsize>,
}

impl Card {
    pub fn new(rate: u32) -> Self {
        Card { rate, broken: false, fill: Arc::default(), starts: Arc::default() }
    }

    pub fn broken() -> Self {
        Card { rate: 48_000, broken: true, fill: Arc::default(), starts: Arc::default() }
    }

    /// The card asks for `frames` stereo frames.
    pub fn pump(&self, frames: usize) -> Vec<f32> {
        let mut buf = vec![0.0f32; frames * 2];
        if let Some(f) = self.fill.lock().unwrap().as_mut() {
            f(&mut buf, 2);
        }
        buf
    }
}

impl AudioOutput for Card {
    fn start(&mut self, fill: Fill) -> Result<u32, String> {
        if self.broken {
            return Err("no such device".into());
        }
        self.starts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        *self.fill.lock().unwrap() = Some(fill);
        Ok(self.rate)
    }

    fn stop(&mut self) {
        *self.fill.lock().unwrap() = None;
    }
}

/// Keep asking the card for sound until `total` frames have been delivered; returns the real
/// (not underrun) samples.
pub fn pump_until(card: &Card, player: &Player, total: u64) -> Vec<f32> {
    let mut heard = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(10);
    while player.delivered_frames() < total {
        assert!(Instant::now() < deadline, "the mixer never caught up: {} of {total}", player.delivered_frames());
        let before = player.delivered_frames();
        let buf = card.pump(480);
        let got = (player.delivered_frames() - before) as usize;
        heard.extend_from_slice(&buf[..got * 2]);
        if got == 0 {
            std::thread::sleep(Duration::from_millis(3));
        }
    }
    heard
}

pub fn peak(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |m, v| m.max(v.abs()))
}

pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() { 0.0 } else { (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt() }
}
