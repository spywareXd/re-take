//! Words and numbers for the screen. Display only: nothing here feeds back into edits.

use simplecut_core::{MediaEntry, MediaKind, Tick};

const TICKS_PER_SECOND: f64 = 254_016_000_000.0;

fn seconds(t: Tick) -> f64 {
    t.0.max(0) as f64 / TICKS_PER_SECOND
}

/// `0:04.12` or `1:02:03.12`: minutes, seconds and frames at `fps`.
pub fn timecode(t: Tick, fps: f64) -> String {
    let secs = seconds(t);
    let whole = secs.floor() as u64;
    let fps = if fps.is_finite() && fps >= 1.0 { fps } else { 30.0 };
    let frame = (((secs - whole as f64) * fps).floor() as u64).min(fps.ceil() as u64 - 1);
    let (h, m, s) = (whole / 3600, (whole / 60) % 60, whole % 60);
    if h > 0 { format!("{h}:{m:02}:{s:02}.{frame:02}") } else { format!("{m}:{s:02}.{frame:02}") }
}

/// `0:04` or `1:02:03`: a length, to the second.
pub fn length(t: Tick) -> String {
    let whole = seconds(t).round() as u64;
    let (h, m, s) = (whole / 3600, (whole / 60) % 60, whole % 60);
    if h > 0 { format!("{h}:{m:02}:{s:02}") } else { format!("{m}:{s:02}") }
}

/// `0:04 · 1920×1080 · 30 fps · stereo`: what the library shows under a file's name.
pub fn media_summary(m: &MediaEntry) -> String {
    let mut parts = vec![length(m.duration)];
    if let Some((w, h, fps)) = m.video {
        parts.push(format!("{w}×{h}"));
        parts.push(format!("{} fps", trim_zeros(fps)));
    }
    if let Some((channels, rate)) = m.audio {
        let layout = match channels {
            1 => "mono".to_string(),
            2 => "stereo".to_string(),
            n => format!("{n} ch"),
        };
        parts.push(format!("{layout} {} kHz", trim_zeros(f64::from(rate) / 1000.0)));
    } else if m.kind != MediaKind::Still {
        parts.push("no sound".into());
    }
    parts.join(" · ")
}

fn trim_zeros(x: f64) -> String {
    let s = format!("{x:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(s: f64) -> Tick {
        Tick::from_seconds_f64(s)
    }

    #[test]
    fn timecodes_show_minutes_seconds_and_frames() {
        assert_eq!(timecode(Tick::ZERO, 30.0), "0:00.00");
        assert_eq!(timecode(secs(4.5), 30.0), "0:04.15");
        assert_eq!(timecode(secs(62.0), 25.0), "1:02.00");
        assert_eq!(timecode(secs(3723.0), 30.0), "1:02:03.00");
        assert_eq!(timecode(Tick(-5), 30.0), "0:00.00", "negative time reads as the start");
        assert_eq!(timecode(secs(1.0), f64::NAN), "0:01.00", "a bad frame rate falls back");
    }

    #[test]
    fn lengths_round_to_the_second() {
        assert_eq!(length(secs(4.4)), "0:04");
        assert_eq!(length(secs(59.6)), "1:00");
        assert_eq!(length(secs(3723.0)), "1:02:03");
    }

    #[test]
    fn numbers_lose_trailing_zeros() {
        assert_eq!(trim_zeros(30.0), "30");
        assert_eq!(trim_zeros(29.97), "29.97");
        assert_eq!(trim_zeros(48.0), "48");
    }
}
