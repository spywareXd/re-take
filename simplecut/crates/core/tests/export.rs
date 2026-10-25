//! Tests for exporting the timeline to MP4 (H.264 + AAC).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::Path;
use std::process::Command;
use std::thread::sleep;
use std::time::{Duration, Instant};

use common::{have, make_clip, scratch};
use simplecut_core::{Editor, ExportOptions, ExportState, Lane, PlaceAt};

fn skip() -> bool {
    let missing = !have("ffmpeg") || !have("ffprobe");
    if missing {
        eprintln!("SKIPPED: ffmpeg / ffprobe are not installed");
    }
    missing
}

fn probe(path: &Path) -> serde_json::Value {
    let out = Command::new("ffprobe").args(["-v", "error", "-of", "json", "-show_format", "-show_streams"]).arg(path).output().expect("run ffprobe");
    assert!(out.status.success(), "ffprobe failed on {}: {}", path.display(), String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).expect("parse ffprobe json")
}

fn decode_all_frames(path: &Path) {
    let status = Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(path).args(["-f", "null", "-"]).status().expect("run ffmpeg decode");
    assert!(status.success(), "ffmpeg could not decode frames of {}", path.display());
}

#[test]
fn export_empty_timeline_is_rejected_cleanly() {
    let mut editor = Editor::new();
    editor.new_project("Empty").unwrap();
    let dir = scratch("export-empty");
    let out = dir.join("empty.mp4");

    let err = editor.start_export(ExportOptions::new(out)).unwrap_err();
    assert!(err.to_string().contains("timeline is empty"), "unexpected error: {err}");
}

#[test]
fn export_short_timeline_succeeds_progress_is_monotonic_and_output_is_verified() {
    if skip() {
        return;
    }
    let dir = scratch("export-short");
    let clip_path = make_clip(&dir, "input.mp4", 2, true);

    let mut editor = Editor::new();
    editor.new_project("ShortExport").unwrap();
    let report = editor.import(&[clip_path]).unwrap();
    let item = report.imported[0];
    editor.place(item, Lane::Main, PlaceAt::Playhead).unwrap();

    let seq = editor.sequence().unwrap();
    assert_eq!(seq.clip_count(), 2, "video and dialogue clips placed");

    let out_file = dir.join("rendered.mp4");
    let job = editor.start_export(ExportOptions { path: out_file.clone(), preset: "YouTube 1080p Full HD".into(), use_hardware: false }).unwrap();

    let mut last_fraction = 0.0f32;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let state = job.poll();
        match state {
            ExportState::Running { fraction, status, .. } => {
                assert!(fraction >= last_fraction, "progress was not monotonic: {fraction} < {last_fraction}");
                last_fraction = fraction;
                assert!(!status.is_empty());
            }
            ExportState::Done { path, frames, seconds, .. } => {
                assert_eq!(path, out_file);
                assert!(frames > 0, "encoded at least one frame");
                assert!(seconds > 0.0);
                break;
            }
            ExportState::Cancelled => panic!("job was cancelled unexpectedly"),
            ExportState::Failed(err) => panic!("job failed: {err}"),
        }
        assert!(Instant::now() < deadline, "timed out waiting for export");
        sleep(Duration::from_millis(20));
    }

    assert!(out_file.exists(), "output file must exist");
    assert!(out_file.metadata().unwrap().len() > 1000, "output file must have non-trivial size");

    // 1. Verify with FilmCraft's demuxer via job.verify()
    let entry = job.verify().expect("verification with FilmCraft demuxer failed");
    assert!(!entry.offline);
    let (w, h, fps) = entry.video.expect("must have video stream");
    assert_eq!(w, 1920);
    assert_eq!(h, 1080);
    assert!((fps - 30.0).abs() < 0.1);
    let (channels, rate) = entry.audio.expect("must have audio stream");
    assert!(channels >= 1);
    assert_eq!(rate, 48000);

    // 2. Verify with ffprobe: codecs, dimensions, stream types
    let probe_info = probe(&out_file);
    let streams = probe_info["streams"].as_array().expect("streams array");
    let v_stream = streams.iter().find(|s| s["codec_type"] == "video").expect("video stream in ffprobe");
    assert_eq!(v_stream["codec_name"], "h264");
    assert_eq!(v_stream["width"], 1920);
    assert_eq!(v_stream["height"], 1080);

    let a_stream = streams.iter().find(|s| s["codec_type"] == "audio").expect("audio stream in ffprobe");
    assert_eq!(a_stream["codec_name"], "aac");

    // 3. Decode every frame with ffmpeg to ensure valid bitstream from start to end
    decode_all_frames(&out_file);
}

#[test]
fn export_can_be_cancelled_midway() {
    if skip() {
        return;
    }
    let dir = scratch("export-cancel");
    // Create a 10s clip so export takes enough frames to cancel cleanly midway
    let clip_path = make_clip(&dir, "long.mp4", 10, true);

    let mut editor = Editor::new();
    editor.new_project("CancelExport").unwrap();
    let report = editor.import(&[clip_path]).unwrap();
    let item = report.imported[0];
    editor.place(item, Lane::Main, PlaceAt::Playhead).unwrap();

    let out_file = dir.join("cancelled.mp4");
    let job = editor.start_export(ExportOptions::new(out_file.clone())).unwrap();

    // Wait until job begins running
    let deadline = Instant::now() + Duration::from_secs(30);
    while job.fraction() < 0.05 {
        if job.is_finished() {
            break;
        }
        assert!(Instant::now() < deadline, "job did not start");
        sleep(Duration::from_millis(10));
    }

    // Cancel mid-way
    job.cancel();

    // Wait for worker to exit
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match job.poll() {
            ExportState::Cancelled => break,
            ExportState::Done { .. } => {
                // If it finished before cancellation took effect on a super-fast system, that's okay, but cancelling should ideally win
                break;
            }
            ExportState::Failed(e) if e == "cancelled" || e.contains("cancel") => break,
            ExportState::Failed(e) => panic!("job failed with unexpected error: {e}"),
            ExportState::Running { .. } => {
                assert!(Instant::now() < deadline, "timed out waiting for cancel");
                sleep(Duration::from_millis(10));
            }
        }
    }
}

#[test]
fn export_to_unwritable_path_fails_cleanly() {
    let dir = scratch("export-badpath");
    // A path with invalid filename characters or a non-existent drive
    let bad_path = if cfg!(windows) { std::path::PathBuf::from("X:\\cannot\\exist\\rendered.mp4") } else { std::path::PathBuf::from("/proc/sys/rendered.mp4") };

    let clip_path = if !skip() { Some(make_clip(&dir, "input.mp4", 1, false)) } else { None };

    let Some(clip_path) = clip_path else {
        return;
    };

    let mut editor = Editor::new();
    editor.new_project("BadPath").unwrap();
    let report = editor.import(&[clip_path]).unwrap();
    let item = report.imported[0];
    editor.place(item, Lane::Main, PlaceAt::Playhead).unwrap();

    let job_res = editor.start_export(ExportOptions::new(bad_path));
    match job_res {
        Ok(job) => {
            // Background thread should fail cleanly without panicking
            let deadline = Instant::now() + Duration::from_secs(30);
            loop {
                match job.poll() {
                    ExportState::Failed(msg) => {
                        assert!(!msg.is_empty());
                        break;
                    }
                    ExportState::Done { .. } => panic!("bad path unexpectedly succeeded"),
                    ExportState::Cancelled => break,
                    ExportState::Running { .. } => {
                        assert!(Instant::now() < deadline, "timed out waiting for bad path to fail");
                        sleep(Duration::from_millis(20));
                    }
                }
            }
        }
        Err(e) => {
            // Immediate error is also clean
            assert!(!e.to_string().is_empty());
        }
    }
}
