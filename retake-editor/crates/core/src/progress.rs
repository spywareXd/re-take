//! Pipeline progress as the service reports it: structured stage events (`{stage, status, pct, message, timestamp}`), and a log that turns them into the stepper
//! the panel shows (one row per stage, in order) plus the scrolling text log. The log is kept after the job ends.
//!
//! Everything from the service is treated as hostile: unknown stages are ignored, percents are clamped, texts are capped.

use std::time::{Duration, Instant};

/// The stages of the pipeline, in order: (id the service uses, label shown).
pub const STAGES: [(&str, &str); 11] = [
    ("model_download", "Download model"),
    ("model_load", "Load model"),
    ("audio_extraction", "Extract audio"),
    ("vad", "Detect speech (VAD)"),
    ("transcription", "Transcribe"),
    ("alignment", "Align words"),
    ("unit_splitting", "Split sentences"),
    ("pair_scoring", "Score sentence pairs"),
    ("keeper_selection", "Choose keepers"),
    ("cutlist_build", "Build cut list"),
    ("export_render", "Render export"),
];

/// Most events a log keeps (a long transcription reports one per percent at most, so this is generous).
pub const MAX_EVENTS: usize = 2_000;
/// Longest message kept: an error message from ffmpeg or Python can be long, and it is shown exactly, but not without a limit.
pub const MAX_MESSAGE_CHARS: usize = 4_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageStatus {
    Started,
    Progress,
    Done,
    Error,
}

impl StageStatus {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "started" => Some(Self::Started),
            "progress" => Some(Self::Progress),
            "done" => Some(Self::Done),
            "error" => Some(Self::Error),
            _ => None,
        }
    }
}

/// One event, as sent: the stage id, what happened, an optional percent (0..=100), a message and the service's timestamp (ISO 8601, shown as is).
#[derive(Debug, Clone, PartialEq)]
pub struct StageEvent {
    pub seq: u64,
    pub stage: String,
    pub status: StageStatus,
    pub pct: Option<f32>,
    pub message: String,
    pub timestamp: String,
    /// A stage that was skipped because its result already existed ("cached").
    pub cached: bool,
}

impl StageEvent {
    /// Build one from a JSON object of the service; None when it is not a usable event.
    pub fn from_json(v: &serde_json::Value) -> Option<Self> {
        let stage = v.get("stage")?.as_str()?;
        if !STAGES.iter().any(|(id, _)| *id == stage) {
            return None;
        }
        let status = StageStatus::parse(v.get("status")?.as_str()?)?;
        let pct = v.get("pct").and_then(serde_json::Value::as_f64).filter(|p| p.is_finite()).map(|p| p.clamp(0.0, 100.0) as f32);
        Some(Self {
            seq: v.get("seq").and_then(serde_json::Value::as_u64).unwrap_or(0),
            stage: stage.to_string(),
            status,
            pct,
            message: v.get("message").and_then(serde_json::Value::as_str).unwrap_or("").chars().take(MAX_MESSAGE_CHARS).collect(),
            timestamp: v.get("timestamp").and_then(serde_json::Value::as_str).unwrap_or("").chars().take(40).collect(),
            cached: v.get("cached").and_then(serde_json::Value::as_bool).unwrap_or(false),
        })
    }
}

/// What the stepper shows for one stage.
#[derive(Debug, Clone, PartialEq)]
pub enum StageState {
    /// Nothing reported for it (yet, or it does not apply to this run).
    Pending,
    Running {
        pct: Option<f32>,
        message: String,
    },
    Done {
        message: String,
    },
    /// Skipped because its result was already there: shown as "cached", never hidden.
    Cached,
    /// The exact message of the failure.
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, Default)]
struct StageTimes {
    first: Option<Instant>,
    last: Option<Instant>,
}

/// The events of one run, and what they add up to.
#[derive(Debug, Clone)]
pub struct StageLog {
    events: Vec<StageEvent>,
    times: [StageTimes; STAGES.len()],
    started: Instant,
    ended: Option<Instant>,
}

impl Default for StageLog {
    fn default() -> Self {
        Self::new()
    }
}

impl StageLog {
    pub fn new() -> Self {
        Self { events: Vec::new(), times: Default::default(), started: Instant::now(), ended: None }
    }

    pub fn push(&mut self, event: StageEvent) {
        if self.events.len() >= MAX_EVENTS {
            self.events.remove(0);
        }
        if let Some(i) = STAGES.iter().position(|(id, _)| *id == event.stage) {
            let now = Instant::now();
            let t = &mut self.times[i];
            t.first.get_or_insert(now);
            t.last = Some(now);
        }
        self.events.push(event);
    }

    /// The run is over: the clock stops.
    pub fn finish(&mut self) {
        self.ended.get_or_insert_with(Instant::now);
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn events(&self) -> &[StageEvent] {
        &self.events
    }

    /// The stepper row for a stage id.
    pub fn state(&self, stage: &str) -> StageState {
        let mut state = StageState::Pending;
        for e in self.events.iter().filter(|e| e.stage == stage) {
            state = match e.status {
                StageStatus::Started => StageState::Running { pct: None, message: e.message.clone() },
                StageStatus::Progress => StageState::Running { pct: e.pct, message: e.message.clone() },
                StageStatus::Done if e.cached => StageState::Cached,
                StageStatus::Done => StageState::Done { message: e.message.clone() },
                StageStatus::Error => StageState::Error { message: e.message.clone() },
            };
        }
        state
    }

    /// The stage that is running now: (label, percent). The latest stage with a Running state.
    pub fn current(&self) -> Option<(&'static str, Option<f32>)> {
        STAGES.iter().rev().find_map(|(id, label)| match self.state(id) {
            StageState::Running { pct, .. } => Some((*label, pct)),
            _ => None,
        })
    }

    /// Time since the run began (stops when [`finish`](Self::finish) is called).
    pub fn elapsed(&self) -> Duration {
        self.ended.unwrap_or_else(Instant::now).duration_since(self.started)
    }

    /// How long a stage has taken, as seen by this editor (first event to last event, or to now while it runs).
    pub fn stage_elapsed(&self, stage: &str) -> Option<Duration> {
        let i = STAGES.iter().position(|(id, _)| *id == stage)?;
        let t = &self.times[i];
        let first = t.first?;
        let end = match self.state(stage) {
            StageState::Running { .. } => self.ended.unwrap_or_else(Instant::now),
            _ => t.last?,
        };
        Some(end.saturating_duration_since(first))
    }

    /// The message of the first failed stage, exactly as the service sent it: (stage label, message).
    pub fn error(&self) -> Option<(&'static str, String)> {
        STAGES.iter().find_map(|(id, label)| match self.state(id) {
            StageState::Error { message } => Some((*label, message)),
            _ => None,
        })
    }

    /// One text line per event, for the scrolling log: `<time> <stage>: <message>`.
    pub fn lines(&self) -> Vec<String> {
        self.events
            .iter()
            .map(|e| {
                let label = STAGES.iter().find(|(id, _)| *id == e.stage).map_or(e.stage.as_str(), |(_, l)| *l);
                let time = e.timestamp.get(11..19).unwrap_or("");
                let what = match (e.status, e.cached) {
                    (StageStatus::Started, _) => "started".to_string(),
                    (StageStatus::Progress, _) => e.pct.map_or("progress".to_string(), |p| format!("{p:.0}%")),
                    (StageStatus::Done, true) => "cached".to_string(),
                    (StageStatus::Done, false) => "done".to_string(),
                    (StageStatus::Error, _) => "ERROR".to_string(),
                };
                if e.message.is_empty() || (e.cached && e.message == "cached") {
                    format!("{time} {label}: {what}")
                } else {
                    format!("{time} {label}: {what} — {}", e.message)
                }
            })
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use serde_json::json;

    use super::*;

    fn ev(stage: &str, status: &str, pct: Option<f64>, message: &str, cached: bool) -> StageEvent {
        let mut v = json!({"seq": 1, "stage": stage, "status": status, "message": message, "timestamp": "2026-10-29T22:35:55.591+00:00", "cached": cached});
        if let Some(p) = pct {
            v["pct"] = json!(p);
        }
        StageEvent::from_json(&v).unwrap()
    }

    #[test]
    fn events_are_read_defensively() {
        assert!(StageEvent::from_json(&json!({"stage": "nonsense", "status": "done"})).is_none());
        assert!(StageEvent::from_json(&json!({"stage": "vad", "status": "weird"})).is_none());
        assert!(StageEvent::from_json(&json!({"status": "done"})).is_none());
        assert!(StageEvent::from_json(&json!("text")).is_none());
        let e = StageEvent::from_json(&json!({"stage": "vad", "status": "progress", "pct": 1e9, "message": "x".repeat(10_000)})).unwrap();
        assert_eq!(e.pct, Some(100.0));
        assert_eq!(e.message.chars().count(), MAX_MESSAGE_CHARS);
        assert_eq!(StageEvent::from_json(&json!({"stage": "vad", "status": "progress", "pct": -5})).unwrap().pct, Some(0.0));
        assert_eq!(StageEvent::from_json(&json!({"stage": "vad", "status": "progress", "pct": "high"})).unwrap().pct, None);
    }

    #[test]
    fn the_stepper_follows_the_events_and_shows_cached_and_errors() {
        let mut log = StageLog::new();
        assert!(STAGES.iter().all(|(id, _)| log.state(id) == StageState::Pending));
        log.push(ev("audio_extraction", "started", None, "16 kHz mono WAV", false));
        log.push(ev("audio_extraction", "done", Some(100.0), "", false));
        log.push(ev("model_load", "done", Some(100.0), "cached", true));
        log.push(ev("transcription", "started", None, "", false));
        log.push(ev("transcription", "progress", Some(42.0), "", false));
        assert_eq!(log.state("audio_extraction"), StageState::Done { message: String::new() });
        assert_eq!(log.state("model_load"), StageState::Cached);
        assert_eq!(log.state("transcription"), StageState::Running { pct: Some(42.0), message: String::new() });
        assert_eq!(log.state("vad"), StageState::Pending);
        assert_eq!(log.current(), Some(("Transcribe", Some(42.0))));
        log.push(ev("transcription", "error", None, "RuntimeError: CUDA out of memory", false));
        assert_eq!(log.error(), Some(("Transcribe", "RuntimeError: CUDA out of memory".to_string())));
        assert!(log.current().is_none());
        let lines = log.lines();
        assert_eq!(lines.len(), 6);
        assert!(lines[0].starts_with("22:35:55 Extract audio: started — 16 kHz"), "{}", lines[0]);
        assert_eq!(lines[2], "22:35:55 Load model: cached");
        assert_eq!(lines[4], "22:35:55 Transcribe: 42%");
        assert_eq!(lines[5], "22:35:55 Transcribe: ERROR — RuntimeError: CUDA out of memory");
    }

    #[test]
    fn the_clock_stops_when_the_run_ends_and_the_log_is_bounded() {
        let mut log = StageLog::new();
        log.push(ev("vad", "started", None, "", false));
        assert!(log.stage_elapsed("vad").is_some() && log.stage_elapsed("alignment").is_none());
        log.finish();
        let frozen = log.elapsed();
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(log.elapsed(), frozen, "no longer ticking");
        for _ in 0..MAX_EVENTS + 50 {
            log.push(ev("vad", "progress", Some(1.0), "", false));
        }
        assert_eq!(log.events().len(), MAX_EVENTS);
    }
}
