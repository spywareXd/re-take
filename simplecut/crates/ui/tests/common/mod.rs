//! Fixtures for the window tests: small media made with ffmpeg (a test-only oracle).
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod harness;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

pub fn have(tool: &str) -> bool {
    Command::new(tool).arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

pub fn scratch(name: &str) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("scratch dir");
    d
}

/// Whole seconds of 320x240 colour bars at 25 fps with a 440 Hz tone. H.264 + AAC.
pub fn make_clip(dir: &Path, file: &str, secs: u32) -> PathBuf {
    let out = dir.join(file);
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-f", "lavfi", "-i"])
        .arg(format!("testsrc=size=320x240:rate=25:duration={secs}"))
        .args(["-f", "lavfi", "-i"])
        .arg(format!("sine=frequency=440:sample_rate=48000:duration={secs}"))
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"])
        .arg(&out)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg could not make {file}");
    out
}

pub fn make_music(dir: &Path, file: &str, secs: u32) -> PathBuf {
    let out = dir.join(file);
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-f", "lavfi", "-i"])
        .arg(format!("sine=frequency=220:sample_rate=48000:duration={secs}"))
        .args(["-ac", "2"])
        .arg(&out)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg could not make {file}");
    out
}
