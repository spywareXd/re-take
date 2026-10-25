//! The editor: a typed front over FilmCraft's engine `Session`.
//!
//! The UI mutates the project only through these methods. Each one runs engine commands, so a
//! change is one undo step, the project stays FilmCraft's native format, and the engine's checks
//! (no overlapping clips, linked clips together) apply. Reading goes through the snapshot types in
//! [`crate::view`].

use std::path::{Path, PathBuf};

use filmcraft_engine::Session;
use filmcraft_media::MediaKind;
use filmcraft_project::{ClipId, ItemKind, MediaRef, ProjectItem, Track, TrackId};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::error::{EditorError, Result};
use crate::export::{ExportJob, ExportOptions};
use crate::view::{ClipView, ImportReport, MediaEntry, SequenceView, TrackView};

pub use filmcraft_project::ItemId;

/// The fixed tracks of a SimpleCut sequence.
const VIDEO_TRACK: &str = "V1";
const DIALOGUE_TRACK: &str = "A1";
const MUSIC_TRACK: &str = "A2";

/// Which part of the timeline media goes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    /// Picture on V1 with its own sound on A1 (or sound only on A1 for an audio file).
    Main,
    /// Background music on A2.
    Music,
}

/// Where on the timeline to place media.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceAt {
    /// At the playhead.
    Playhead,
    /// After the last clip of the lane.
    End,
    /// At an exact time.
    At(Tick),
}

/// Which end of a clip to trim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    Start,
    End,
}

/// What a trim does to the clips after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimMode {
    /// Leaves a gap where the clip got shorter.
    Regular,
    /// Closes the gap: later clips move with the edge.
    Ripple,
}

pub struct Editor {
    session: Session,
}

impl Default for Editor {
    fn default() -> Self {
        Self::new()
    }
}

impl Editor {
    pub fn new() -> Self {
        Self { session: Session::default() }
    }

    pub fn with_session(session: Session) -> Self {
        Self { session }
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
    }

    pub fn into_session(self) -> Session {
        self.session
    }

    /// Run any engine command by id. For tests and plug-ins; the UI uses the typed methods.
    pub fn execute_raw(&mut self, id: &str, params: Value) -> Result<Value> {
        Ok(self.session.execute(id, params)?)
    }

    // ── project ────────────────────────────────────────────────────────────

    /// A new project with one sequence: V1 (Video), A1 (Dialogue, linked to V1) and A2 (Music).
    /// 1920x1080 at 30 fps, 48 kHz. Music does not follow ripple edits of the main tracks.
    pub fn new_project(&mut self, name: &str) -> Result<()> {
        let name = if name.trim().is_empty() { "Untitled" } else { name.trim() };
        self.session.execute("file.newProject", json!({"name": name}))?;
        self.session
            .execute("file.newSequence", json!({"name": name, "width": 1920, "height": 1080, "fps": 30.0, "sampleRate": 48000, "video": 1, "audio": 2}))?;
        for (track, label, sync) in [(VIDEO_TRACK, "Video", true), (DIALOGUE_TRACK, "Dialogue", true), (MUSIC_TRACK, "Music", false)] {
            self.session.execute("timeline.setTrack", json!({"track": track, "name": label, "syncLock": sync}))?;
        }
        // the template is the starting point, not something to undo
        self.session.history.undo.clear();
        self.session.history.redo.clear();
        // and a project nobody has touched yet has nothing to save
        self.session.saved_revision = self.session.revision;
        Ok(())
    }

    pub fn open(&mut self, path: &Path) -> Result<()> {
        let shown = path.to_string_lossy().into_owned();
        match std::fs::metadata(path) {
            Ok(m) if m.is_file() => {}
            Ok(_) => return Err(EditorError::FileUnreadable(format!("{shown} is not a file"))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(EditorError::FileMissing(shown)),
            Err(e) => return Err(EditorError::FileUnreadable(format!("{shown}: {e}"))),
        }
        self.session.execute("file.open", json!({"path": shown}))?;
        Ok(())
    }

    /// Save to the project's own file. Fails with [`EditorError::NeedsPath`] for a new project.
    pub fn save(&mut self) -> Result<()> {
        if self.session.path.is_none() {
            return Err(EditorError::NeedsPath);
        }
        self.session.execute("file.save", json!({}))?;
        Ok(())
    }

    pub fn save_as(&mut self, path: &Path) -> Result<()> {
        self.session.execute("file.saveAs", json!({"path": path.to_string_lossy()}))?;
        Ok(())
    }

    pub fn is_dirty(&self) -> bool {
        self.session.is_dirty()
    }

    pub fn project_name(&self) -> &str {
        &self.session.project.name
    }

    pub fn project_path(&self) -> Option<&str> {
        self.session.path.as_deref()
    }

    // ── persistence & recovery ─────────────────────────────────────────────

    /// Start auto-save and the crash-recovery journal using `data_dir`.
    pub fn enable_autosave(&mut self, data_dir: PathBuf) -> Result<()> {
        let cfg = filmcraft_engine::autosave::AutosaveConfig::new(data_dir);
        self.session.start_autosave(cfg).map_err(|e| EditorError::Engine(e.to_string()))?;
        Ok(())
    }

    /// Clean shutdown: flush and stop the persistence worker.
    pub fn shutdown(&mut self) {
        self.session.shutdown();
    }

    /// List any recovery candidates left by dead sessions.
    pub fn recovery_candidates(&self) -> &[filmcraft_engine::autosave::RecoveryCandidate] {
        self.session.recovery_candidates()
    }

    /// Recover unsaved changes from a candidate session (or the newest if `id` is None).
    pub fn recover(&mut self, id: Option<&str>) -> Result<()> {
        self.session.execute(
            "file.recover",
            match id {
                Some(id) => json!({ "id": id }),
                None => json!({}),
            },
        )?;
        Ok(())
    }

    /// Discard unsaved changes from a candidate session (or all if `id` is None).
    pub fn discard_recovery(&mut self, id: Option<&str>) -> Result<()> {
        self.session.execute(
            "file.discardRecovery",
            match id {
                Some(id) => json!({ "id": id }),
                None => json!({ "all": true }),
            },
        )?;
        Ok(())
    }

    // ── media ──────────────────────────────────────────────────────────────

    /// Import video and audio files. Succeeds when at least one file was imported; the others are
    /// listed in the report with the reason.
    pub fn import(&mut self, paths: &[PathBuf]) -> Result<ImportReport> {
        if paths.is_empty() {
            return Err(EditorError::Invalid("Choose at least one file to import.".into()));
        }
        let mut failed = Vec::new();
        let mut usable = Vec::new();
        for p in paths {
            let shown = p.to_string_lossy().into_owned();
            match std::fs::metadata(p) {
                Ok(m) if m.is_file() => usable.push(shown),
                Ok(_) => failed.push(format!("{shown}: not a file")),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => failed.push(format!("{shown}: the file could not be found")),
                Err(e) => failed.push(format!("{shown}: {e}")),
            }
        }
        if usable.is_empty() {
            return Err(EditorError::ImportFailed(failed.join("; ")));
        }
        let r = match self.session.execute("file.import", json!({"paths": usable})) {
            Ok(r) => r,
            Err(e) => {
                failed.push(e.to_string());
                return Err(EditorError::ImportFailed(failed.join("; ")));
            }
        };
        let imported: Vec<ItemId> = r["items"].as_array().into_iter().flatten().filter_map(Value::as_u64).map(ItemId).collect();
        failed.extend(r["errors"].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_string));
        if imported.is_empty() {
            let why = if failed.is_empty() { "None of the files could be used as video or audio.".to_string() } else { failed.join("; ") };
            return Err(EditorError::ImportFailed(why));
        }
        Ok(ImportReport { imported, failed })
    }

    /// Every video and audio file in the project.
    pub fn media(&self) -> Vec<MediaEntry> {
        self.session.project.items.values().filter_map(|it| self.media_entry_of(it)).collect()
    }

    pub fn media_entry(&self, id: ItemId) -> Option<MediaEntry> {
        self.session.project.item(id).and_then(|it| self.media_entry_of(it))
    }

    /// Relink an offline media item to a new file path.
    pub fn relink(&mut self, item: ItemId, path: &Path) -> Result<()> {
        if !path.exists() {
            return Err(EditorError::FileMissing(format!("Relink destination does not exist: {}", path.to_string_lossy())));
        }
        self.session.execute(
            "media.relink",
            json!({
                "item": item.0,
                "path": path.to_string_lossy(),
                "match": { "fileName": false },
                "force": false,
            }),
        )?;
        Ok(())
    }

    /// Automatically relink missing media by searching a directory tree.
    pub fn auto_relink(&mut self, folder: &Path) -> Result<usize> {
        if !folder.exists() {
            return Err(EditorError::FileMissing(format!("Folder does not exist: {}", folder.to_string_lossy())));
        }
        let res = self.session.execute(
            "media.autoRelink",
            json!({
                "folder": folder.to_string_lossy(),
            }),
        )?;
        let count = res.get("relinked").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
        Ok(count)
    }

    fn media_entry_of(&self, it: &ProjectItem) -> Option<MediaEntry> {
        let ItemKind::Media(mc) = &it.kind else { return None };
        let path = match &mc.media {
            MediaRef::File { path } => Some(path.clone()),
            MediaRef::Generator(_) => None,
        };
        let status = self.session.media.offline_status(it.id);
        let missing = path.as_deref().is_some_and(|p| !Path::new(p).exists());
        let offline = mc.offline || status.is_some() || missing;
        let offline_reason = if mc.offline {
            Some("This media was taken offline.".to_string())
        } else if missing {
            Some(format!("The file could not be found: {}", path.clone().unwrap_or_default()))
        } else {
            status.map(|s| s.error)
        };
        Some(MediaEntry {
            id: it.id,
            name: it.name.clone(),
            kind: mc.info.kind,
            duration: mc.info.duration,
            video: mc.info.video.as_ref().map(|v| (v.width, v.height, v.frame_rate.as_f64())),
            audio: mc.info.audio_streams.first().map(|a| (a.channels, a.sample_rate)),
            path,
            file_size: mc.info.file_size,
            offline,
            offline_reason,
        })
    }

    // ── timeline: reading ──────────────────────────────────────────────────

    pub fn sequence(&self) -> Option<SequenceView> {
        let q = self.session.active_sequence()?;
        let selected = &self.session.state.selection;
        let view_of = |t: &Track| TrackView {
            id: t.id,
            kind: t.kind,
            name: t.name.clone(),
            muted: t.muted,
            volume_db: t.volume_db,
            locked: t.locked,
            clips: t
                .items
                .iter()
                .map(|c| ClipView {
                    id: c.id,
                    item: c.item,
                    name: c.name.clone(),
                    start: c.start,
                    duration: c.duration,
                    source_in: c.source_in,
                    link: c.link,
                    enabled: c.enabled,
                    gain_db: c.gain_db,
                    selected: selected.contains(&c.id),
                })
                .collect(),
        };
        Some(SequenceView {
            width: q.settings.width,
            height: q.settings.height,
            fps: q.settings.frame_rate.as_f64(),
            duration: q.duration(),
            playhead: self.session.playhead(),
            tracks: q.video_tracks.iter().chain(q.audio_tracks.iter()).map(view_of).collect(),
        })
    }

    pub fn playhead(&self) -> Tick {
        self.session.playhead()
    }

    pub fn set_playhead(&mut self, t: Tick) {
        self.session.set_playhead(Tick(t.0.max(0)));
    }

    pub fn selection(&self) -> &[ClipId] {
        &self.session.state.selection
    }

    /// Select clips, and with them whatever moves with them (a picture's own sound), so what is
    /// highlighted is what an edit will act on.
    pub fn select(&mut self, clips: &[ClipId]) -> Result<()> {
        let mut ids: Vec<u64> = clips.iter().map(|c| c.0).collect();
        if let Some(q) = self.session.active_sequence() {
            let links: Vec<u64> = clips.iter().filter_map(|c| q.find_item(*c).and_then(|(_, it)| it.link)).collect();
            for t in q.video_tracks.iter().chain(q.audio_tracks.iter()) {
                for c in &t.items {
                    if c.link.is_some_and(|l| links.contains(&l)) && !ids.contains(&c.id.0) {
                        ids.push(c.id.0);
                    }
                }
            }
        }
        self.session.execute("timeline.select", json!({"clips": ids}))?;
        Ok(())
    }

    // ── timeline: editing ──────────────────────────────────────────────────

    /// Put media on the timeline. A video file brings its sound onto A1, linked to the picture.
    pub fn place(&mut self, item: ItemId, lane: Lane, at: PlaceAt) -> Result<Vec<ClipId>> {
        if self.session.project.item(item).is_none() {
            return Err(EditorError::NoSuchMedia);
        }
        if self.session.active_sequence().is_none() {
            return Err(EditorError::NoSequence);
        }
        let (has_video, has_audio) = self.session.project.source_streams(item);
        let time = self.resolve(at, lane);
        let mut p = json!({"item": item.0, "time": time.0});
        match lane {
            Lane::Main if has_video => {
                p["track"] = json!(VIDEO_TRACK);
                p["audioTrack"] = json!(DIALOGUE_TRACK);
                p["audio"] = json!(has_audio);
            }
            Lane::Main if has_audio => p["track"] = json!(DIALOGUE_TRACK),
            Lane::Main => return Err(EditorError::NothingToPlace),
            Lane::Music => {
                if !has_audio {
                    return Err(EditorError::NoAudio);
                }
                p["track"] = json!(MUSIC_TRACK);
                p["video"] = json!(false);
            }
        }
        let r = self.session.execute("timeline.place", p)?;
        Ok(r["clips"].as_array().into_iter().flatten().filter_map(Value::as_u64).map(ClipId).collect())
    }

    fn resolve(&self, at: PlaceAt, lane: Lane) -> Tick {
        match at {
            PlaceAt::At(t) => Tick(t.0.max(0)),
            PlaceAt::Playhead => self.session.playhead(),
            PlaceAt::End => {
                let Some(q) = self.session.active_sequence() else { return Tick::ZERO };
                let tracks: Vec<&Track> = match lane {
                    Lane::Main => q.video_tracks.first().into_iter().chain(q.audio_tracks.first()).collect(),
                    Lane::Music => q.audio_tracks.get(1).into_iter().collect(),
                };
                tracks.iter().flat_map(|t| t.items.iter()).map(|c| c.start + c.duration).max().unwrap_or(Tick::ZERO)
            }
        }
    }

    /// Move a clip (and the clips linked to it) so it starts at `to`.
    pub fn move_clip(&mut self, clip: ClipId, to: Tick) -> Result<()> {
        // the engine wants the destination track spelled out; here it is the clip's own
        let track = {
            let q = self.session.active_sequence().ok_or(EditorError::NoSequence)?;
            q.find_item(clip).map(|(track, _)| track).ok_or(EditorError::NoSuchClip)?
        };
        self.session.execute("timeline.move", json!({"moves": [{"clip": clip.0, "track": track.0, "time": to.0.max(0)}], "linked": true}))?;
        Ok(())
    }

    /// Move one edge of a clip by `delta` (positive moves the edge later in time).
    pub fn trim(&mut self, clip: ClipId, edge: Edge, delta: Tick, mode: TrimMode) -> Result<()> {
        self.require_clip(clip)?;
        let edge = match edge {
            Edge::Start => "in",
            Edge::End => "out",
        };
        let mode = match mode {
            TrimMode::Regular => "regular",
            TrimMode::Ripple => "ripple",
        };
        self.session.execute("timeline.trim", json!({"clip": clip.0, "edge": edge, "mode": mode, "delta": delta.0}))?;
        Ok(())
    }

    /// Split at the playhead: the selected clips under it, else the main-track clips under it.
    /// Linked sound is split with its picture; music is only split when it is selected.
    /// One undo step. Returns how many clips were split.
    pub fn split_at_playhead(&mut self) -> Result<usize> {
        let t = self.session.playhead();
        let targets = {
            let q = self.session.active_sequence().ok_or(EditorError::NoSequence)?;
            let under = |c: &filmcraft_project::TrackItem| c.start < t && t < c.start + c.duration;
            let selected: Vec<(ClipId, Option<u64>)> = q
                .video_tracks
                .iter()
                .chain(q.audio_tracks.iter())
                .flat_map(|tr| tr.items.iter())
                .filter(|c| under(c) && self.session.state.selection.contains(&c.id))
                .map(|c| (c.id, c.link))
                .collect();
            if selected.is_empty() {
                q.video_tracks.first().into_iter().flat_map(|tr| tr.items.iter()).filter(|c| under(c)).map(|c| (c.id, c.link)).collect()
            } else {
                selected
            }
        };
        if targets.is_empty() {
            return Err(EditorError::NothingToSplit);
        }
        // a linked pair is cut by cutting either member
        let mut seen: Vec<u64> = Vec::new();
        let mut clips = Vec::new();
        for (id, link) in targets {
            if let Some(l) = link {
                if seen.contains(&l) {
                    continue;
                }
                seen.push(l);
            }
            clips.push(id);
        }
        let n = clips.len();
        self.session.edit_group("Split", |s| {
            for c in &clips {
                s.execute("timeline.razor", json!({"time": t.0, "clip": c.0}))?;
            }
            Ok(())
        })?;
        Ok(n)
    }

    /// Delete clips and close the gap they leave (linked sound goes with its picture).
    pub fn delete(&mut self, clips: &[ClipId]) -> Result<()> {
        if clips.is_empty() {
            return Err(EditorError::NothingSelected);
        }
        for c in clips {
            self.require_clip(*c)?;
        }
        let ids: Vec<u64> = clips.iter().map(|c| c.0).collect();
        self.session.execute("edit.rippleDelete", json!({"clips": ids}))?;
        Ok(())
    }

    pub fn delete_selected(&mut self) -> Result<()> {
        let sel = self.session.state.selection.clone();
        self.delete(&sel)
    }

    fn require_clip(&self, clip: ClipId) -> Result<()> {
        let q = self.session.active_sequence().ok_or(EditorError::NoSequence)?;
        q.find_item(clip).map(|_| ()).ok_or(EditorError::NoSuchClip)
    }

    // ── audio ──────────────────────────────────────────────────────────────

    /// Set an audio clip's volume (0 dB = as recorded).
    pub fn set_clip_gain_db(&mut self, clip: ClipId, db: f64) -> Result<()> {
        if !db.is_finite() {
            return Err(EditorError::Invalid("Volume must be a number.".into()));
        }
        self.require_clip(clip)?;
        self.session.execute("clip.audioGain", json!({"clips": [clip.0], "mode": "set", "db": db.clamp(-60.0, 24.0)}))?;
        Ok(())
    }

    /// Set the volume of every audio clip among `clips` in one step. Clips on the video track are
    /// skipped (their sound is the linked clip on the dialogue track).
    pub fn set_clips_gain_db(&mut self, clips: &[ClipId], db: f64) -> Result<()> {
        if !db.is_finite() {
            return Err(EditorError::Invalid("Volume must be a number.".into()));
        }
        let audio: Vec<ClipId> = {
            let q = self.session.active_sequence().ok_or(EditorError::NoSequence)?;
            q.audio_tracks.iter().flat_map(|t| t.items.iter()).filter(|c| clips.contains(&c.id)).map(|c| c.id).collect()
        };
        if audio.is_empty() {
            return Err(EditorError::NotAvailable("Select a clip with sound first.".into()));
        }
        let ids: Vec<u64> = audio.iter().map(|c| c.0).collect();
        self.session.execute("clip.audioGain", json!({"clips": ids, "mode": "set", "db": db.clamp(-60.0, 24.0)}))?;
        Ok(())
    }

    pub fn set_track_muted(&mut self, track: TrackId, muted: bool) -> Result<()> {
        self.require_track(track)?;
        self.session.execute("timeline.setTrack", json!({"track": track.0, "muted": muted}))?;
        Ok(())
    }

    pub fn set_track_volume_db(&mut self, track: TrackId, db: f64) -> Result<()> {
        if !db.is_finite() {
            return Err(EditorError::Invalid("Volume must be a number.".into()));
        }
        self.require_track(track)?;
        self.session.execute("timeline.setTrack", json!({"track": track.0, "volumeDb": db.clamp(-60.0, 12.0)}))?;
        Ok(())
    }

    fn require_track(&self, track: TrackId) -> Result<()> {
        let q = self.session.active_sequence().ok_or(EditorError::NoSequence)?;
        if q.video_tracks.iter().chain(q.audio_tracks.iter()).any(|t| t.id == track) {
            Ok(())
        } else {
            Err(EditorError::Invalid("That track does not exist.".into()))
        }
    }

    // ── history ────────────────────────────────────────────────────────────

    pub fn can_undo(&self) -> bool {
        self.session.history.can_undo()
    }

    pub fn can_redo(&self) -> bool {
        self.session.history.can_redo()
    }

    /// Undo the last change; returns what was undone.
    pub fn undo(&mut self) -> Option<String> {
        self.session.undo()
    }

    pub fn redo(&mut self) -> Option<String> {
        self.session.redo()
    }

    /// Whether the media has a picture (for the library's icon).
    pub fn has_picture(kind: MediaKind) -> bool {
        !matches!(kind, MediaKind::AudioOnly)
    }

    // ── export ─────────────────────────────────────────────────────────────

    /// Start a background export job for the active sequence.
    pub fn start_export(&mut self, options: ExportOptions) -> Result<ExportJob> {
        let seq = self.sequence().ok_or(EditorError::NoSequence)?;
        if seq.clip_count() == 0 || seq.duration == Tick::ZERO {
            return Err(EditorError::Invalid("The timeline is empty. Add media to the timeline before exporting.".into()));
        }
        if options.path.as_os_str().is_empty() {
            return Err(EditorError::Invalid("Choose where to save the exported video.".into()));
        }
        let params = json!({
            "path": options.path.to_string_lossy(),
            "preset": options.preset,
            "hardwareEncoding": if options.use_hardware { "auto" } else { "off" },
            "wait": false,
        });
        let r = self.session.execute("file.exportMedia", params)?;
        let job_id = r["job"].as_u64().ok_or_else(|| EditorError::Engine("exportMedia did not return a job id".into()))?;
        let job = self.session.jobs.iter().find(|j| j.id == job_id).ok_or_else(|| EditorError::Engine("job not found in session".into()))?;
        Ok(ExportJob {
            id: job_id,
            path: options.path,
            progress: job.progress.clone(),
            result: job.result.clone(),
            expected_duration: seq.duration,
            expected_fps: seq.fps,
            expected_width: seq.width,
            expected_height: seq.height,
        })
    }
}
