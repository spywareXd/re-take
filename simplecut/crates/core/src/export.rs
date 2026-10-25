//! Exporting the timeline to video files.
//!
//! Wraps FilmCraft's export engine (`file.exportMedia`) into a background [`ExportJob`] that can
//! be polled for progress, cancelled, and verified.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use filmcraft_time::Tick;

use crate::error::{EditorError, Result};
use crate::view::MediaEntry;

/// Settings for an export job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportOptions {
    /// Where to write the exported video.
    pub path: PathBuf,
    /// The FilmCraft export preset name (e.g. "YouTube 1080p Full HD").
    pub preset: String,
    /// Whether to attempt hardware-accelerated video encoding if available.
    pub use_hardware: bool,
}

impl ExportOptions {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into(), preset: "YouTube 1080p Full HD".into(), use_hardware: false }
    }
}

/// Current state of an active or finished export job.
#[derive(Debug, Clone, PartialEq)]
pub enum ExportState {
    /// Encoding is in progress.
    Running {
        /// Progress from 0.0 to 1.0.
        fraction: f32,
        /// Estimated duration remaining, if known.
        eta: Option<Duration>,
        /// Status text from the encoder.
        status: String,
    },
    /// Export finished successfully.
    Done { path: PathBuf, frames: u64, seconds: f64, bytes: u64 },
    /// Cancelled by user.
    Cancelled,
    /// Failed with an error message.
    Failed(String),
}

/// A background export job with progress and cancellation.
#[derive(Clone)]
pub struct ExportJob {
    pub(crate) id: u64,
    pub(crate) path: PathBuf,
    pub(crate) progress: Arc<filmcraft_export::Progress>,
    pub(crate) result: Arc<Mutex<Option<std::result::Result<filmcraft_export::Report, String>>>>,
    pub(crate) expected_duration: Tick,
    pub(crate) expected_fps: f64,
    pub(crate) expected_width: u32,
    pub(crate) expected_height: u32,
}

impl std::fmt::Debug for ExportJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportJob").field("id", &self.id).field("path", &self.path).field("fraction", &self.fraction()).field("status", &self.status()).finish()
    }
}

impl ExportJob {
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn fraction(&self) -> f32 {
        self.progress.fraction()
    }

    pub fn eta(&self) -> Option<Duration> {
        self.progress.eta()
    }

    pub fn status(&self) -> String {
        let s = self.progress.status.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if s.is_empty() { "Exporting...".into() } else { s }
    }

    /// Request cancellation. Sets the cancel flag on the shared progress.
    pub fn cancel(&self) {
        self.progress.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_finished(&self) -> bool {
        self.result.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    /// Poll the current state of the export.
    pub fn poll(&self) -> ExportState {
        let res_guard = self.result.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(res) = res_guard.as_ref() {
            if self.progress.cancel.load(Ordering::Relaxed) {
                return ExportState::Cancelled;
            }
            match res {
                Ok(r) => ExportState::Done { path: self.path.clone(), frames: r.frames, seconds: r.seconds, bytes: r.bytes },
                Err(e) => {
                    if e == "cancelled" || e.contains("cancel") {
                        ExportState::Cancelled
                    } else {
                        ExportState::Failed(e.clone())
                    }
                }
            }
        } else if self.progress.cancel.load(Ordering::Relaxed) {
            ExportState::Cancelled
        } else {
            ExportState::Running { fraction: self.fraction(), eta: self.eta(), status: self.status() }
        }
    }

    /// Verify the exported media file with FilmCraft's demuxer.
    pub fn verify(&self) -> Result<MediaEntry> {
        verify_export(&self.path, self.expected_duration, self.expected_fps, self.expected_width, self.expected_height)
    }
}

/// Verify an exported media file by importing it into a scratch engine session.
/// Checks that the file has a video stream, audio stream, matching dimensions, and duration within one frame.
pub fn verify_export(path: &Path, expected_duration: Tick, expected_fps: f64, expected_width: u32, expected_height: u32) -> Result<MediaEntry> {
    let path_buf = path.to_path_buf();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        verify_export_inner(&path_buf, expected_duration, expected_fps, expected_width, expected_height)
    })) {
        Ok(res) => res,
        Err(_) => Err(EditorError::Engine("Verification panicked unexpectedly".into())),
    }
}

fn verify_export_inner(path: &Path, expected_duration: Tick, _expected_fps: f64, expected_width: u32, expected_height: u32) -> Result<MediaEntry> {
    if !path.exists() {
        return Err(EditorError::FileMissing(format!("Exported file does not exist: {}", path.to_string_lossy())));
    }
    let mut session = filmcraft_engine::Session::default();
    let r = session.execute("file.import", serde_json::json!({"paths": [path.to_string_lossy()]}))?;
    let items = r["items"].as_array().ok_or_else(|| EditorError::Engine("import did not return items".into()))?;
    let first = items
        .first()
        .and_then(|v| v.as_u64())
        .map(filmcraft_project::ItemId)
        .ok_or_else(|| EditorError::FileUnreadable(format!("Could not read exported media: {}", path.to_string_lossy())))?;
    let project_item = session.project.item(first).ok_or(EditorError::NoSuchMedia)?;
    let filmcraft_project::ItemKind::Media(mc) = &project_item.kind else {
        return Err(EditorError::Engine("imported item is not media".into()));
    };
    if mc.offline {
        return Err(EditorError::FileUnreadable(format!("Exported media is offline: {}", path.to_string_lossy())));
    }
    let video_info = mc.info.video.as_ref().ok_or_else(|| EditorError::Invalid("Exported file has no video stream".into()))?;
    if video_info.width != expected_width || video_info.height != expected_height {
        return Err(EditorError::Invalid(format!(
            "Exported video dimensions {}x{} do not match expected {}x{}",
            video_info.width, video_info.height, expected_width, expected_height
        )));
    }
    if mc.info.audio_streams.is_empty() {
        return Err(EditorError::Invalid("Exported file has no audio stream".into()));
    }

    // Duration check within one frame (approx 33.3ms at 30 fps) plus audio priming tolerance
    let frame_ticks = Tick::from_seconds_f64(1.0 / 30.0);
    let tolerance = frame_ticks + Tick::from_seconds_f64(0.06);
    let diff = if mc.info.duration > expected_duration { mc.info.duration - expected_duration } else { expected_duration - mc.info.duration };
    if diff > tolerance {
        return Err(EditorError::Invalid(format!(
            "Exported duration {:?} differs from timeline duration {:?} by more than one frame",
            mc.info.duration, expected_duration
        )));
    }

    Ok(MediaEntry {
        id: first,
        name: project_item.name.clone(),
        kind: mc.info.kind,
        duration: mc.info.duration,
        video: Some((video_info.width, video_info.height, video_info.frame_rate.as_f64())),
        audio: mc.info.audio_streams.first().map(|a| (a.channels, a.sample_rate)),
        path: Some(path.to_string_lossy().into_owned()),
        file_size: mc.info.file_size,
        offline: false,
        offline_reason: None,
    })
}
