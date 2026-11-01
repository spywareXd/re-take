//! Step 0 spike: FilmCraft's engine, used as a path dependency from outside its workspace, can
//! import a clip, build a sequence, split it and export an H.264 + AAC MP4 that an independent tool
//! (ffprobe, test-only oracle, never linked) agrees with.
//!
//! Skipped (with a message) when ffmpeg / ffprobe are not installed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::process::Command;

use filmcraft_engine::Session;
use serde_json::{Value, json};

fn have(tool: &str) -> bool {
    Command::new(tool).arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("smoke-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("scratch dir");
    d
}

/// 2 s of 320x240 colour bars at 25 fps with a 440 Hz tone, H.264 + AAC.
fn make_clip(dir: &Path) -> PathBuf {
    let out = dir.join("input.mp4");
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2"])
        .args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"])
        .arg(&out)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg could not make the fixture");
    out
}

fn probe(path: &Path) -> Value {
    let out = Command::new("ffprobe").args(["-v", "error", "-print_format", "json", "-show_streams", "-show_format"]).arg(path).output().expect("run ffprobe");
    assert!(out.status.success(), "ffprobe failed: {}", String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).expect("ffprobe json")
}

fn stream<'a>(probe: &'a Value, kind: &str) -> &'a Value {
    probe["streams"].as_array().and_then(|s| s.iter().find(|x| x["codec_type"] == kind)).unwrap_or(&Value::Null)
}

#[test]
fn import_split_and_export_h264_aac() {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("SKIPPED: ffmpeg / ffprobe are not installed");
        return;
    }
    let dir = scratch("export");
    let clip = make_clip(&dir);

    let mut s = Session::default();
    s.execute("file.newProject", json!({"name": "Smoke"})).expect("new project");
    let imported = s.execute("file.import", json!({"paths": [clip.to_string_lossy()]})).expect("import");
    let item = imported["items"][0].as_u64().unwrap_or_else(|| panic!("no item imported: {imported}"));

    // a sequence made from the clip places it (picture and sound) at 0
    s.execute("file.newSequence", json!({"name": "Main", "fromItem": item, "video": 1, "audio": 2})).expect("new sequence");
    let clips = |s: &Session| {
        let q = s.active_sequence().expect("active sequence");
        q.video_tracks.iter().chain(q.audio_tracks.iter()).map(|t| t.items.len()).sum::<usize>()
    };
    assert_eq!(clips(&s), 2, "one video clip and its linked audio");

    // split at 1 s: video and audio both cut, nothing lost
    s.execute("sequence.addEditAllTracks", json!({"seconds": 1.0})).expect("split");
    assert_eq!(clips(&s), 4, "two pieces on each of the two tracks");
    s.active_sequence().expect("sequence").check().expect("sequence invariants hold (no overlaps)");

    // export with the default YouTube preset
    let out = dir.join("out.mp4");
    let r = s.execute("file.exportMedia", json!({"path": out.to_string_lossy(), "preset": "YouTube 1080p Full HD", "wait": true})).expect("export");
    assert!(r["result"].is_object(), "a waited export answers its result: {r}");

    let p = probe(&out);
    let (v, a) = (stream(&p, "video"), stream(&p, "audio"));
    assert_eq!(v["codec_name"], "h264", "{p}");
    assert_eq!(a["codec_name"], "aac", "{p}");
    assert_eq!((v["width"].as_u64(), v["height"].as_u64()), (Some(1920), Some(1080)), "{p}");
    let secs: f64 = p["format"]["duration"].as_str().and_then(|d| d.parse().ok()).unwrap_or(0.0);
    assert!((secs - 2.0).abs() < 0.15, "the split must not change the length: {secs} s");
}
