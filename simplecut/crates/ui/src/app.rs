//! The application: state, actions, and the per-frame loop.
//!
//! Everything the user can do is an [`Action`]. Buttons, shortcuts and tests all go through
//! [`SimpleCutApp::perform`], so there is one path for each thing and one place that reports its
//! errors. The panels only draw and queue actions.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use egui::{Align2, Key, Modifiers, RichText};
use simplecut_core::{
    AudioOutput, ClipId, Edge, Editor, EditorError, ExportJob, ExportOptions, ExportState, FrameRate, ItemId, Lane, MediaEntry, PlaceAt, PlayState, Player,
    PreviewFrame, PreviewRenderer, Snapshot, Target, Thumbnails, Tick, TrackId, TrimMode,
};

use crate::dialogs::Dialogs;
use crate::retakes::RetakeUi;
use crate::timeline::TimelineUi;
use crate::{library, preview, retakes, theme, timeline, toolbar};

/// What the preview area shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The edit, at the playhead.
    Timeline,
    /// One imported file, before it is placed.
    Media(ItemId),
}

/// Everything the user can ask for.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    NewProject,
    OpenProject,
    Save,
    SaveAs,
    /// Choose files with the file dialog, then import them.
    Import,
    ImportPaths(Vec<PathBuf>),
    Undo,
    Redo,
    TogglePlay,
    SeekStart,
    SeekEnd,
    /// Move by this many frames (negative: back).
    Step(i64),
    /// Go to a time in whatever the preview shows.
    Seek(Tick),
    /// Go to a time on the timeline and show the timeline.
    SeekTimeline(Tick),
    PreviewMedia(ItemId),
    ShowTimeline,
    AddToTimeline(ItemId, Lane),
    Split,
    Delete,
    /// Select clips (and what moves with them); `add` toggles them in or out of the selection.
    SelectClips {
        clips: Vec<ClipId>,
        add: bool,
    },
    MoveClip {
        clip: ClipId,
        to: Tick,
    },
    /// Move one end of a clip by `delta` (positive: later). `ripple` closes the gap it leaves.
    TrimClip {
        clip: ClipId,
        edge: Edge,
        delta: Tick,
        ripple: bool,
    },
    /// Set the volume of the selected clips, in dB.
    SetClipVolume(f64),
    SetTrackMuted(TrackId, bool),
    SetTrackVolume(TrackId, f64),
    ZoomIn,
    ZoomOut,
    ZoomFit,
    DismissBanner,
    OpenExportDialog,
    StartExport {
        path: PathBuf,
        use_hardware: bool,
    },
    CancelExport,
    OpenFolder(PathBuf),
    RelinkMedia(ItemId),
    OpenShortcuts,
    Recover(Option<String>),
    DiscardRecovery(Option<String>),
    /// Look for retakes in the video on the timeline (needs a detector).
    StartRetakeAnalysis,
    CancelAnalysis,
    /// Choose an analysis file (see docs/retake-contract.md) and review it.
    OpenAnalysisFile,
    LoadAnalysis(PathBuf),
    /// Mark a question: Some(verdict), or None to skip it.
    AnswerPair {
        pair: String,
        verdict: Option<simplecut_core::Verdict>,
    },
    SetQuestion(usize),
    /// Send the answers to the model and make the groups.
    QuestionsNext,
    QuestionsSendAnyway,
    /// Tick or untick a sentence as kept.
    SetKeep {
        sentence: String,
        keep: bool,
    },
    KeepLastOnly(String),
    KeepAllOf(String),
    RemoveAllOf(String),
    KeepLastInLikely,
    AskAi,
    ConfirmAi,
    DeclineAi,
    RedoQuestions,
    GoToSentence(ItemId, Tick),
    PlaySentence(ItemId, Tick),
    /// Cut every sentence that is not kept.
    ApplyRetakes,
    CloseReview,
    SetSkipUnticked(bool),
    /// Set the media-clock span of a sentence in seconds.
    SetSentenceSpan {
        sentence: String,
        start: f64,
        end: f64,
    },
    /// Reset the span of a sentence to its original time.
    ResetSentenceSpan(String),
    /// Save the current groups to a JSON file.
    SaveGroups,
}

/// A line of feedback under the work area.
#[derive(Debug, Clone, PartialEq)]
pub struct Banner {
    pub text: String,
    pub error: bool,
    pub action: Option<(String, Action)>,
}

#[derive(Debug, Clone, PartialEq)]
enum Then {
    New,
    Open(PathBuf),
    Close,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExportModalState {
    pub path: PathBuf,
    pub use_hardware: bool,
    pub overwrite_confirm: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecoveryModalItem {
    pub id: String,
    pub name: String,
    pub saved_at: String,
}

#[derive(Debug, Clone, PartialEq)]
enum Modal {
    Unsaved(Then),
    Export(ExportModalState),
    Recovery { candidates: Vec<RecoveryModalItem>, selected: usize },
    Shortcuts,
}

/// A library thumbnail.
pub(crate) enum Thumb {
    Pending,
    /// The file has no picture, or it could not be read.
    None,
    Ready(egui::TextureHandle),
}

/// What the preview picture was last drawn for; a different one means ask again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Want {
    source: Source,
    revision: u64,
    frame: i64,
    scale_step: u8,
}

pub struct SimpleCutApp {
    pub(crate) editor: Editor,
    pub(crate) player: Player,
    renderer: PreviewRenderer,
    thumbs: Thumbnails,
    pub(crate) dialogs: Box<dyn Dialogs>,
    pub(crate) service: Option<Box<dyn simplecut_core::RetakeService>>,
    pub(crate) retakes: RetakeUi,

    pub(crate) source: Source,
    /// Position in the previewed file (the timeline's is the editor's playhead).
    pub(crate) media_time: Tick,
    pub(crate) thumb_state: HashMap<ItemId, Thumb>,
    media_cache: (u64, Option<std::time::Instant>, Vec<MediaEntry>),
    pub(crate) preview_texture: Option<egui::TextureHandle>,
    last_asked: Option<Want>,
    /// Width in points of the preview area as last drawn, to pick a picture size.
    pub(crate) preview_width: f32,

    pub(crate) tl: TimelineUi,
    pub(crate) banner: Option<Banner>,
    modal: Option<Modal>,
    pub(crate) active_export: Option<ExportJob>,
    close_now: bool,
    title: String,
    queue: Vec<Action>,
}

impl SimpleCutApp {
    /// A window on a new empty project. `output` is the sound card (None: no sound).
    pub fn new(ctx: &egui::Context, dialogs: Box<dyn Dialogs>, output: Option<Box<dyn AudioOutput>>) -> Self {
        theme::apply(ctx);
        let mut editor = Editor::new();
        let mut banner = None;
        if let Err(e) = editor.new_project("Untitled") {
            banner = Some(Banner { text: format!("Could not start a new project: {e}"), error: true, action: None });
        }
        Self {
            editor,
            player: Player::new(output),
            renderer: PreviewRenderer::new(),
            thumbs: Thumbnails::new(),
            dialogs,
            service: None,
            retakes: RetakeUi::default(),
            source: Source::Timeline,
            media_time: Tick::ZERO,
            thumb_state: HashMap::new(),
            media_cache: (u64::MAX, None, Vec::new()),
            preview_texture: None,
            last_asked: None,
            preview_width: 640.0,
            tl: TimelineUi::default(),
            banner,
            modal: None,
            active_export: None,
            close_now: false,
            title: String::new(),
            queue: Vec::new(),
        }
    }

    // ── for tests and the app shell ────────────────────────────────────────

    pub fn editor(&self) -> &Editor {
        &self.editor
    }

    pub fn editor_mut(&mut self) -> &mut Editor {
        &mut self.editor
    }

    pub fn player(&self) -> &Player {
        &self.player
    }

    pub fn player_mut(&mut self) -> &mut Player {
        &mut self.player
    }

    pub fn source(&self) -> Source {
        self.source
    }

    pub fn banner(&self) -> Option<&Banner> {
        self.banner.as_ref()
    }

    /// Size in pixels of the picture currently on screen in the preview.
    pub fn preview_size(&self) -> Option<[usize; 2]> {
        self.preview_texture.as_ref().map(|t| t.size())
    }

    pub fn asking_about_unsaved(&self) -> bool {
        matches!(self.modal, Some(Modal::Unsaved(_)))
    }

    /// Whether an export is currently being encoded.
    pub fn is_exporting(&self) -> bool {
        self.active_export.is_some()
    }

    /// The thumbnail state of a library item: 'pending', 'none' or 'ready'.
    pub fn thumbnail_state(&self, item: ItemId) -> Option<&'static str> {
        self.thumb_state.get(&item).map(|t| match t {
            Thumb::Pending => "pending",
            Thumb::None => "none",
            Thumb::Ready(_) => "ready",
        })
    }

    /// Open a project file given at launch.
    pub fn open_project_path(&mut self, path: PathBuf) {
        self.open_now(path);
    }

    /// Start autosave persistence in `data_dir` and check for recovery candidates from dead sessions.
    pub fn enable_autosave(&mut self, data_dir: PathBuf) {
        if let Err(e) = self.editor.enable_autosave(data_dir) {
            log::warn!("could not start autosave: {e}");
            return;
        }
        self.check_recovery();
    }

    /// Check if previous crashed sessions left unsaved projects to recover.
    pub fn check_recovery(&mut self) {
        let candidates = self.editor.recovery_candidates();
        if !candidates.is_empty() && self.modal.is_none() {
            let list = candidates
                .iter()
                .map(|c| {
                    let name = if c.meta.project_name.is_empty() { "Untitled".to_string() } else { c.meta.project_name.clone() };
                    let saved_at = c.meta.saved_unix;
                    RecoveryModalItem { id: c.id.clone(), name, saved_at: format!("{saved_at}") }
                })
                .collect();
            self.modal = Some(Modal::Recovery { candidates: list, selected: 0 });
        }
    }

    pub fn asking_about_recovery(&self) -> bool {
        matches!(self.modal, Some(Modal::Recovery { .. }))
    }

    pub fn showing_shortcuts(&self) -> bool {
        matches!(self.modal, Some(Modal::Shortcuts))
    }

    pub(crate) fn queue(&mut self, a: Action) {
        self.queue.push(a);
    }

    pub(crate) fn media(&mut self) -> Vec<MediaEntry> {
        let revision = self.editor.session().revision;
        let stale = self.media_cache.1.is_none_or(|t| t.elapsed() > Duration::from_secs(2));
        if self.media_cache.0 != revision || stale {
            self.media_cache = (revision, Some(std::time::Instant::now()), self.editor.media());
        }
        self.media_cache.2.clone()
    }

    // ── actions ────────────────────────────────────────────────────────────

    fn info(&mut self, text: impl Into<String>) {
        self.banner = Some(Banner { text: text.into(), error: false, action: None });
    }

    pub(crate) fn info_text(&mut self, text: impl Into<String>) {
        self.info(text);
    }

    pub(crate) fn fail_text(&mut self, text: &str) {
        self.banner = Some(Banner { text: text.to_string(), error: true, action: None });
    }

    fn fail(&mut self, e: &EditorError) {
        self.banner = Some(Banner { text: e.to_string(), error: true, action: None });
    }

    /// Show an error if there is one; otherwise hand the value back.
    fn check<T>(&mut self, r: Result<T, EditorError>) -> Option<T> {
        match r {
            Ok(v) => Some(v),
            Err(e) => {
                self.fail(&e);
                None
            }
        }
    }

    pub fn perform(&mut self, action: Action) {
        if self.is_exporting() {
            match action {
                Action::CancelExport | Action::DismissBanner | Action::OpenFolder(_) | Action::OpenShortcuts => {}
                Action::Seek(_) | Action::SeekTimeline(_) | Action::SeekStart | Action::SeekEnd | Action::Step(_) => {}
                _ => {
                    self.fail(&EditorError::NotAvailable("Cannot make changes while an export is in progress.".into()));
                    return;
                }
            }
        }
        match action {
            Action::StartRetakeAnalysis
            | Action::CancelAnalysis
            | Action::OpenAnalysisFile
            | Action::LoadAnalysis(_)
            | Action::AnswerPair { .. }
            | Action::SetQuestion(_)
            | Action::QuestionsNext
            | Action::QuestionsSendAnyway
            | Action::SetKeep { .. }
            | Action::KeepLastOnly(_)
            | Action::KeepAllOf(_)
            | Action::RemoveAllOf(_)
            | Action::KeepLastInLikely
            | Action::AskAi
            | Action::ConfirmAi
            | Action::DeclineAi
            | Action::RedoQuestions
            | Action::GoToSentence(..)
            | Action::PlaySentence(..)
            | Action::ApplyRetakes
            | Action::CloseReview
            | Action::SetSkipUnticked(_)
            | Action::SetSentenceSpan { .. }
            | Action::ResetSentenceSpan(_)
            | Action::SaveGroups => self.perform_retake(action),
            Action::NewProject => self.guard_unsaved(Then::New),
            Action::OpenProject => {
                if let Some(path) = self.dialogs.pick_project_to_open() {
                    self.guard_unsaved(Then::Open(path));
                }
            }
            Action::Save => {
                self.save_project();
            }
            Action::SaveAs => {
                self.save_project_as();
            }
            Action::Import => {
                let paths = self.dialogs.pick_media();
                if !paths.is_empty() {
                    self.import(paths);
                }
            }
            Action::ImportPaths(paths) => self.import(paths),
            Action::Undo => match self.editor.undo() {
                Some(what) => self.info(format!("Undid {}", what.to_lowercase())),
                None => self.info("Nothing to undo."),
            },
            Action::Redo => match self.editor.redo() {
                Some(what) => self.info(format!("Redid {}", what.to_lowercase())),
                None => self.info("Nothing to redo."),
            },
            Action::TogglePlay => self.toggle_play(),
            Action::SeekStart => self.seek_in_source(Tick::ZERO),
            Action::SeekEnd => {
                let end = self.source_duration();
                self.seek_in_source(end);
            }
            Action::Step(n) => {
                let fd = self.source_frame_duration();
                let now = self.source_time();
                self.seek_in_source(Tick(now.0.saturating_add(fd.0.saturating_mul(n))));
            }
            Action::Seek(t) => self.seek_in_source(t),
            Action::SeekTimeline(t) => {
                self.source = Source::Timeline;
                self.seek_in_source(t);
            }
            Action::PreviewMedia(item) => {
                if self.player.is_playing() {
                    self.player.pause();
                }
                self.source = Source::Media(item);
                self.media_time = Tick::ZERO;
            }
            Action::ShowTimeline => {
                if matches!(self.source, Source::Media(_)) {
                    self.player.pause();
                }
                self.source = Source::Timeline;
            }
            Action::AddToTimeline(item, lane) => self.add_to_timeline(item, lane),
            Action::Split => {
                let r = self.editor.split_at_playhead();
                if let Some(n) = self.check(r) {
                    self.info(if n == 1 { "Split the clip at the playhead.".to_string() } else { format!("Split {n} clips at the playhead.") });
                }
            }
            Action::Delete => {
                let r = self.editor.delete_selected();
                if self.check(r).is_some() {
                    self.info("Deleted the selected clips.");
                }
            }
            Action::SelectClips { clips, add } => {
                let new = if add {
                    let current = self.editor.selection().to_vec();
                    let group: Vec<ClipId> = clips.iter().flat_map(|c| self.linked_group(*c)).collect();
                    if !group.is_empty() && group.iter().all(|g| current.contains(g)) {
                        current.into_iter().filter(|c| !group.contains(c)).collect()
                    } else {
                        let mut all = current;
                        all.extend(clips);
                        all
                    }
                } else {
                    clips
                };
                let r = self.editor.select(&new);
                self.check(r);
            }
            Action::MoveClip { clip, to } => {
                let r = self.editor.move_clip(clip, to);
                self.check(r);
            }
            Action::TrimClip { clip, edge, delta, ripple } => {
                let r = self.editor.trim(clip, edge, delta, if ripple { TrimMode::Ripple } else { TrimMode::Regular });
                self.check(r);
            }
            Action::SetClipVolume(db) => {
                let selection = self.editor.selection().to_vec();
                let r = self.editor.set_clips_gain_db(&selection, db);
                self.check(r);
            }
            Action::SetTrackMuted(track, muted) => {
                let r = self.editor.set_track_muted(track, muted);
                self.check(r);
            }
            Action::SetTrackVolume(track, db) => {
                let r = self.editor.set_track_volume_db(track, db);
                self.check(r);
            }
            Action::ZoomIn => timeline::zoom_by(self, 1.5),
            Action::ZoomOut => timeline::zoom_by(self, 1.0 / 1.5),
            Action::ZoomFit => timeline::zoom_fit(self),
            Action::DismissBanner => self.banner = None,
            Action::OpenExportDialog => self.open_export_dialog(),
            Action::StartExport { path, use_hardware } => self.start_export(path, use_hardware),
            Action::CancelExport => self.cancel_export(),
            Action::OpenFolder(path) => self.dialogs.open_folder(&path),
            Action::RelinkMedia(id) => self.relink_media(id),
            Action::OpenShortcuts => self.modal = Some(Modal::Shortcuts),
            Action::Recover(id) => self.recover_project(id),
            Action::DiscardRecovery(id) => {
                let r = self.editor.discard_recovery(id.as_deref());
                self.check(r);
                self.modal = None;
            }
        }
    }

    fn relink_media(&mut self, id: ItemId) {
        let name = self.editor.media_entry(id).map(|m| m.name).unwrap_or_else(|| "media".into());
        let Some(path) = self.dialogs.pick_relink_path(&name) else { return };
        match self.editor.relink(id, &path) {
            Ok(()) => {
                self.thumb_state.remove(&id);
                self.media_cache = (0, None, Vec::new());
                let filename = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                self.info(format!("Relinked {name} to {filename}."));
            }
            Err(e) => self.fail(&e),
        }
    }

    fn recover_project(&mut self, id: Option<String>) {
        match self.editor.recover(id.as_deref()) {
            Ok(()) => {
                self.modal = None;
                self.thumb_state.clear();
                self.media_cache = (0, None, Vec::new());
                self.source = Source::Timeline;
                self.info(format!("Recovered unsaved changes to {}.", self.editor.project_name()));
            }
            Err(e) => self.fail(&e),
        }
    }

    fn open_export_dialog(&mut self) {
        if self.active_export.is_some() {
            return;
        }
        let _seq = match self.editor.sequence() {
            Some(s) if s.clip_count() > 0 && s.duration > Tick::ZERO => s,
            _ => {
                self.fail(&EditorError::Invalid("The timeline is empty. Add media to the timeline before exporting.".into()));
                return;
            }
        };
        let default_name = format!("{}.mp4", self.editor.project_name());
        let default_path = if let Some(p) = self.editor.project_path() { PathBuf::from(p).with_file_name(default_name) } else { PathBuf::from(default_name) };
        self.modal = Some(Modal::Export(ExportModalState { path: default_path, use_hardware: false, overwrite_confirm: false }));
    }

    fn start_export(&mut self, path: PathBuf, use_hardware: bool) {
        if self.player.is_playing() {
            self.player.pause();
        }
        let r = self.editor.start_export(ExportOptions { path, preset: "YouTube 1080p Full HD".into(), use_hardware });
        if let Some(job) = self.check(r) {
            self.active_export = Some(job);
        }
    }

    fn cancel_export(&mut self) {
        if let Some(job) = &self.active_export {
            job.cancel();
            self.info("Export cancelled.");
            self.active_export = None;
        }
    }

    /// `clip` and everything that moves with it.
    fn linked_group(&self, clip: ClipId) -> Vec<ClipId> {
        let Some(seq) = self.editor.sequence() else { return Vec::new() };
        let link = seq.clip(clip).and_then(|c| c.link);
        seq.tracks.iter().flat_map(|t| t.clips.iter()).filter(|c| c.id == clip || (link.is_some() && c.link == link)).map(|c| c.id).collect()
    }

    // ── where the timeline drew things (for tests) ─────────────────────────

    /// The screen rectangle of a clip as last drawn.
    pub fn timeline_clip_rect(&self, clip: ClipId) -> Option<egui::Rect> {
        self.tl.last.as_ref()?.clips.iter().find(|g| g.id == clip).map(|g| g.rect)
    }

    /// The screen x of a time on the timeline as last drawn.
    pub fn timeline_x(&self, t: Tick) -> Option<f32> {
        self.tl.last.as_ref().map(|g| g.axis.x_of(t))
    }

    /// The screen rectangle of a track's lane (0 = the top one) as last drawn.
    pub fn timeline_lane(&self, index: usize) -> Option<egui::Rect> {
        self.tl.last.as_ref()?.lanes.get(index).copied()
    }

    /// The screen rectangle of the ruler as last drawn.
    pub fn timeline_ruler(&self) -> Option<egui::Rect> {
        self.tl.last.as_ref().map(|g| g.ruler)
    }

    /// Pixels per second of the timeline as last drawn.
    pub fn timeline_scale(&self) -> Option<f64> {
        self.tl.last.as_ref().map(|g| g.axis.pps)
    }

    /// Whether the timeline is zoomed in from "fit".
    pub fn timeline_zoomed(&self) -> bool {
        self.tl.zoom.is_some()
    }

    /// The screen rectangle of the overview seek slider as last drawn.
    pub fn seek_slider_rect(&self) -> Option<egui::Rect> {
        self.tl.seek_slider_rect
    }

    fn import(&mut self, paths: Vec<PathBuf>) {
        let first_project_media = self.editor.media().is_empty();
        match self.editor.import(&paths) {
            Ok(report) => {
                let n = report.imported.len();
                if report.failed.is_empty() {
                    self.info(if n == 1 { "Imported 1 file.".to_string() } else { format!("Imported {n} files.") });
                } else {
                    self.banner =
                        Some(Banner { text: format!("Imported {n}, but some could not be used: {}", report.failed.join("; ")), error: true, action: None });
                }
                if first_project_media && let Some(first) = report.imported.first() {
                    self.source = Source::Media(*first);
                    self.media_time = Tick::ZERO;
                }
            }
            Err(e) => self.fail(&e),
        }
    }

    fn add_to_timeline(&mut self, item: ItemId, lane: Lane) {
        let at = match lane {
            Lane::Main => PlaceAt::End,
            Lane::Music => PlaceAt::At(Tick::ZERO),
        };
        self.player.pause();
        let r = self.editor.place(item, lane, at);
        if let Some(clips) = self.check(r) {
            let _ = self.editor.select(&clips);
            self.source = Source::Timeline;
            self.info("Added to the timeline.");
        }
    }

    fn toggle_play(&mut self) {
        match self.source {
            Source::Timeline => {
                let r = self.player.toggle(&mut self.editor);
                if let Err(e) = r {
                    self.fail(&e);
                }
            }
            Source::Media(item) => {
                if self.player.is_playing() {
                    self.player.pause();
                } else {
                    let r = self.player.play_media(&self.editor, item, self.media_time);
                    if let Err(e) = r {
                        self.fail(&e);
                    }
                }
            }
        }
    }

    fn seek_in_source(&mut self, to: Tick) {
        match self.source {
            Source::Timeline => self.player.seek(&mut self.editor, to),
            Source::Media(item) => {
                self.player.seek_media(&self.editor, item, to);
                self.media_time = self.player.position();
            }
        }
    }

    /// Where the preview is now.
    pub(crate) fn source_time(&self) -> Tick {
        match self.source {
            Source::Timeline => self.editor.playhead(),
            Source::Media(_) => self.media_time,
        }
    }

    pub(crate) fn source_snapshot(&self) -> Option<Snapshot> {
        match self.source {
            Source::Timeline => Snapshot::of(&self.editor),
            Source::Media(item) => Snapshot::of_media(&self.editor, item),
        }
    }

    pub(crate) fn source_duration(&self) -> Tick {
        self.source_snapshot().map(|s| s.duration()).unwrap_or_default()
    }

    pub(crate) fn source_frame_rate(&self) -> FrameRate {
        self.source_snapshot().map(|s| s.frame_rate()).unwrap_or_default()
    }

    fn source_frame_duration(&self) -> Tick {
        self.source_frame_rate().frame_duration()
    }

    /// True while the player is playing what the preview shows.
    pub(crate) fn playing_source(&self) -> bool {
        self.player.is_playing()
            && match (self.source, self.player.target()) {
                (Source::Timeline, Some(Target::Sequence(_))) => true,
                (Source::Media(a), Some(Target::Media(b))) => a == b,
                _ => false,
            }
    }

    // ── project: new / open / save ─────────────────────────────────────────

    fn guard_unsaved(&mut self, then: Then) {
        if self.editor.is_dirty() {
            self.modal = Some(Modal::Unsaved(then));
        } else {
            self.run_then(then);
        }
    }

    fn run_then(&mut self, then: Then) {
        match then {
            Then::New => self.new_now(),
            Then::Open(path) => self.open_now(path),
            Then::Close => self.close_now = true,
        }
    }

    /// Forget everything tied to the old project's item ids.
    fn reset_views(&mut self) {
        if let Some(job) = &self.active_export {
            job.cancel();
        }
        self.active_export = None;
        self.player.pause();
        self.player.set_skips(Vec::new());
        self.retakes = RetakeUi::default();
        self.source = Source::Timeline;
        self.media_time = Tick::ZERO;
        self.thumb_state.clear();
        self.thumbs = Thumbnails::new();
        self.preview_texture = None;
        self.last_asked = None;
        self.media_cache = (u64::MAX, None, Vec::new());
        self.tl = TimelineUi::default();
    }

    fn new_now(&mut self) {
        self.reset_views();
        let r = self.editor.new_project("Untitled");
        if self.check(r).is_some() {
            self.info("Started a new project.");
        }
    }

    fn open_now(&mut self, path: PathBuf) {
        self.reset_views();
        match self.editor.open(&path) {
            Ok(()) => {
                let name = self.editor.project_name().to_string();
                self.info(format!("Opened {name}."));
            }
            Err(e) => {
                self.fail(&e);
                // the old project is still there; start it afresh if opening replaced it with nothing
                if self.editor.sequence().is_none() {
                    let _ = self.editor.new_project("Untitled");
                }
            }
        }
    }

    /// Save, asking where first if the project has no file yet. True when it was saved.
    fn save_project(&mut self) -> bool {
        if self.editor.project_path().is_none() {
            return self.save_project_as();
        }
        let r = self.editor.save();
        self.check(r).is_some()
    }

    fn save_project_as(&mut self) -> bool {
        let suggested = format!("{}.fcproj", self.editor.project_name());
        let Some(path) = self.dialogs.pick_project_save_path(&suggested) else { return false };
        let r = self.editor.save_as(&path);
        let ok = self.check(r).is_some();
        if ok {
            self.info(format!("Saved {}.", path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()));
        }
        ok
    }

    // ── the frame ──────────────────────────────────────────────────────────

    /// Everything that happens between drawing frames: the clock, thumbnails, preview pictures, and background jobs.
    fn pump(&mut self, ctx: &egui::Context) {
        self.editor.session_mut().poll_persistence();

        if let Some(crate::retakes::Stage::Review(r)) = &self.retakes.stage
            && r.skip_unticked
        {
            let skips: Vec<(Tick, Tick)> =
                self.review_marks().into_iter().filter(|m| m.kind == crate::retakes::MarkKind::Cut).map(|m| (m.start, m.end)).collect();
            self.player.set_skips(skips);
        } else if !self.player.current_skips().is_empty() {
            self.player.set_skips(Vec::new());
        }

        let state = self.player.tick(&mut self.editor);
        self.pump_retakes();
        if matches!(self.player.target(), Some(Target::Media(_))) && matches!(self.source, Source::Media(_)) {
            self.media_time = self.player.position();
        }
        if state == PlayState::Ended {
            ctx.request_repaint();
        }

        // thumbnails
        for entry in self.media() {
            if !self.thumb_state.contains_key(&entry.id)
                && let Some(snap) = Snapshot::of(&self.editor)
            {
                self.thumbs.request(snap, entry.id);
                self.thumb_state.insert(entry.id, Thumb::Pending);
            }
        }
        for (id, frame) in self.thumbs.take_ready() {
            let state = match frame {
                Some(f) => Thumb::Ready(ctx.load_texture(format!("thumb-{}", id.0), to_image(&f), egui::TextureOptions::LINEAR)),
                None => Thumb::None,
            };
            self.thumb_state.insert(id, state);
        }

        // the preview picture
        self.ask_for_picture();
        if let Some(done) = self.renderer.take_result() {
            let image = to_image(&done.frame);
            match &mut self.preview_texture {
                Some(t) => t.set(image, egui::TextureOptions::LINEAR),
                None => self.preview_texture = Some(ctx.load_texture("preview", image, egui::TextureOptions::LINEAR)),
            }
        }

        // active export job
        if let Some(job) = &self.active_export {
            ctx.request_repaint_after(Duration::from_millis(16));
            match job.poll() {
                ExportState::Running { .. } => {}
                ExportState::Done { path, .. } => {
                    match job.verify() {
                        Ok(_entry) => {
                            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "video".into());
                            self.banner = Some(Banner {
                                text: format!("Export finished: {name}"),
                                error: false,
                                action: Some(("Open folder".into(), Action::OpenFolder(path.clone()))),
                            });
                        }
                        Err(e) => {
                            self.banner = Some(Banner { text: format!("Export verification failed: {e}"), error: true, action: None });
                        }
                    }
                    self.active_export = None;
                }
                ExportState::Cancelled => {
                    self.info("Export cancelled.");
                    self.active_export = None;
                }
                ExportState::Failed(err) => {
                    self.banner = Some(Banner { text: format!("Export failed: {err}"), error: true, action: None });
                    self.active_export = None;
                }
            }
        }

        let busy = self.player.is_playing() || !self.renderer.is_idle() || !self.thumbs.is_idle() || self.active_export.is_some() || self.retakes.busy();
        if busy {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }

    fn ask_for_picture(&mut self) {
        let Some(snapshot) = (if self.playing_source() { self.player.snapshot().cloned() } else { self.source_snapshot() }) else {
            if self.preview_texture.take().is_some() {
                self.last_asked = None;
            }
            return;
        };
        let time = self.source_time();
        let fd = snapshot.frame_rate().frame_duration().0.max(1);
        let reference = match self.source {
            Source::Timeline => self.editor.sequence().map(|s| s.width).unwrap_or(1920),
            Source::Media(item) => self.editor.media_entry(item).and_then(|m| m.video).map(|v| v.0).unwrap_or(1280),
        };
        let wanted = self.preview_width.max(64.0) / reference.max(1) as f32;
        let step: u8 = match wanted {
            w if w <= 0.25 => 1,
            w if w <= 0.5 => 2,
            w if w <= 0.75 => 3,
            _ => 4,
        };
        let want = Want { source: self.source, revision: snapshot.revision, frame: time.0.div_euclid(fd), scale_step: step };
        if self.last_asked != Some(want) {
            self.last_asked = Some(want);
            self.renderer.request(snapshot, time, f32::from(step) * 0.25);
        }
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let mut actions = Vec::new();
        ctx.input_mut(|i| {
            let mut key = |m: Modifiers, k: Key, a: Action| {
                if i.consume_key(m, k) {
                    actions.push(a);
                }
            };
            key(Modifiers::NONE, Key::F1, Action::OpenShortcuts);
            key(Modifiers::NONE, Key::Space, Action::TogglePlay);
            key(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z, Action::Redo);
            key(Modifiers::COMMAND, Key::Y, Action::Redo);
            key(Modifiers::COMMAND, Key::Z, Action::Undo);
            key(Modifiers::COMMAND | Modifiers::SHIFT, Key::S, Action::SaveAs);
            key(Modifiers::COMMAND, Key::S, Action::Save);
            key(Modifiers::COMMAND, Key::O, Action::OpenProject);
            key(Modifiers::COMMAND, Key::N, Action::NewProject);
            key(Modifiers::COMMAND, Key::I, Action::Import);
            key(Modifiers::COMMAND | Modifiers::SHIFT, Key::R, Action::OpenAnalysisFile);
            key(Modifiers::NONE, Key::Home, Action::SeekStart);
            key(Modifiers::NONE, Key::End, Action::SeekEnd);
            key(Modifiers::NONE, Key::ArrowLeft, Action::Step(-1));
            key(Modifiers::NONE, Key::ArrowRight, Action::Step(1));
            key(Modifiers::NONE, Key::S, Action::Split);
            key(Modifiers::NONE, Key::Delete, Action::Delete);
            key(Modifiers::NONE, Key::Backspace, Action::Delete);
            key(Modifiers::COMMAND, Key::E, Action::OpenExportDialog);
        });
        self.queue.extend(actions);
    }

    fn handle_drops(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        if !dropped.is_empty() {
            self.queue.push(Action::ImportPaths(dropped));
        }
    }

    fn handle_close(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.viewport().close_requested()) && self.editor.is_dirty() && !self.close_now {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.modal = Some(Modal::Unsaved(Then::Close));
        }
        if self.close_now {
            self.editor.shutdown();
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }

    fn update_title(&mut self, ctx: &egui::Context) {
        let title = format!("{}{} — SimpleCut", self.editor.project_name(), if self.editor.is_dirty() { " *" } else { "" });
        if title != self.title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.title = title;
        }
    }

    fn modal_ui(&mut self, ctx: &egui::Context) {
        let Some(modal) = self.modal.clone() else { return };
        match modal {
            Modal::Unsaved(then) => {
                let mut answer: Option<&str> = None;
                egui::Window::new("Unsaved changes").collapsible(false).resizable(false).anchor(Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
                    ui.label(format!("Save changes to “{}” first?", self.editor.project_name()));
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Save changes").clicked() {
                            answer = Some("save");
                        }
                        if ui.button("Discard changes").clicked() {
                            answer = Some("discard");
                        }
                        if ui.button("Cancel").clicked() {
                            answer = Some("cancel");
                        }
                    });
                });
                match answer {
                    Some("save") => {
                        self.modal = None;
                        if self.save_project() {
                            self.run_then(then);
                        }
                    }
                    Some("discard") => {
                        self.modal = None;
                        self.run_then(then);
                    }
                    Some(_) => self.modal = None,
                    None => {}
                }
            }
            Modal::Export(mut state) => {
                let t = theme::tokens();
                let mut close = false;
                let mut start: Option<(PathBuf, bool)> = None;
                egui::Window::new("Export video").collapsible(false).resizable(false).anchor(Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
                    ui.label(RichText::new("Preset: YouTube 1080p Full HD (1920×1080, H.264 + AAC)").color(t.text_dim));
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label("Destination:");
                        let shown =
                            state.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| state.path.to_string_lossy().into_owned());
                        ui.label(RichText::new(shown).color(t.text));
                        if ui.button("Browse...").clicked() {
                            let suggested = state.path.file_name().and_then(|n| n.to_str()).unwrap_or("output.mp4");
                            if let Some(picked) = self.dialogs.pick_export_path(suggested) {
                                state.path = picked;
                                state.overwrite_confirm = false;
                            }
                        }
                    });
                    ui.add_space(4.0);
                    ui.checkbox(&mut state.use_hardware, "Use graphics card if available");
                    ui.add_space(8.0);

                    if state.overwrite_confirm {
                        ui.label(RichText::new("A file with this name already exists. Overwrite?").color(t.danger));
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            if ui.button("Overwrite").clicked() {
                                start = Some((state.path.clone(), state.use_hardware));
                                close = true;
                            }
                            if ui.button("Cancel").clicked() {
                                state.overwrite_confirm = false;
                            }
                        });
                    } else {
                        ui.horizontal(|ui| {
                            if ui.button("Start export").clicked() {
                                if state.path.exists() {
                                    state.overwrite_confirm = true;
                                } else {
                                    start = Some((state.path.clone(), state.use_hardware));
                                    close = true;
                                }
                            }
                            if ui.button("Cancel").clicked() {
                                close = true;
                            }
                        });
                    }
                });
                if close {
                    self.modal = None;
                } else {
                    self.modal = Some(Modal::Export(state));
                }
                if let Some((path, use_hardware)) = start {
                    self.perform(Action::StartExport { path, use_hardware });
                }
            }
            Modal::Recovery { candidates, mut selected } => {
                let mut close = false;
                let mut recover_id = None;
                let mut discard_id = None;
                egui::Window::new("Recover unsaved work").collapsible(false).resizable(false).anchor(Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
                    ui.label("SimpleCut found unsaved changes from a previous session:");
                    ui.add_space(8.0);
                    for (i, item) in candidates.iter().enumerate() {
                        let label = format!("{} ({})", item.name, item.saved_at);
                        if ui.selectable_label(selected == i, label).clicked() {
                            selected = i;
                        }
                    }
                    ui.add_space(12.0);
                    ui.horizontal(|ui| {
                        if ui.button("Recover work").clicked() {
                            recover_id = candidates.get(selected).map(|c| c.id.clone());
                            close = true;
                        }
                        if ui.button("Discard recovery").clicked() {
                            discard_id = candidates.get(selected).map(|c| c.id.clone());
                            close = true;
                        }
                        if ui.button("Not now").clicked() {
                            close = true;
                        }
                    });
                });
                if close {
                    self.modal = None;
                } else {
                    self.modal = Some(Modal::Recovery { candidates, selected });
                }
                if let Some(id) = recover_id {
                    self.perform(Action::Recover(Some(id)));
                } else if let Some(id) = discard_id {
                    self.perform(Action::DiscardRecovery(Some(id)));
                }
            }
            Modal::Shortcuts => {
                let mut close = false;
                egui::Window::new("Keyboard Shortcuts").collapsible(false).resizable(false).anchor(Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
                    ui.set_max_width(440.0);
                    let t = theme::tokens();
                    egui::Grid::new("shortcuts_grid").num_columns(2).spacing([24.0, 6.0]).show(ui, |ui| {
                        let row = |ui: &mut egui::Ui, key: &str, desc: &str| {
                            ui.label(RichText::new(key).strong().monospace().color(t.accent));
                            ui.label(desc);
                            ui.end_row();
                        };
                        row(ui, "Space", "Play / Pause");
                        row(ui, "S", "Split clip at playhead");
                        row(ui, "Delete / Backspace", "Delete selected clip(s)");
                        row(ui, "Ctrl+Z / Cmd+Z", "Undo");
                        row(ui, "Ctrl+Y / Ctrl+Shift+Z", "Redo");
                        row(ui, "Ctrl+S", "Save project");
                        row(ui, "Ctrl+Shift+S", "Save project as");
                        row(ui, "Ctrl+O", "Open project");
                        row(ui, "Ctrl+N", "New project");
                        row(ui, "Ctrl+I", "Import media");
                        row(ui, "Ctrl+E", "Export video");
                        row(ui, "Left / Right Arrow", "Step 1 frame backward / forward");
                        row(ui, "Home / End", "Seek to start / end of timeline");
                        row(ui, "Alt + Drag Trim", "Ripple trim (moves later clips)");
                        row(ui, "Ctrl + Mouse Wheel", "Zoom timeline in / out");
                        row(ui, "F1", "Open keyboard shortcuts guide");
                    });
                    ui.add_space(14.0);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Close shortcuts").clicked() {
                            close = true;
                        }
                    });
                });
                if close {
                    self.modal = None;
                }
            }
        }
    }

    fn frame(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.handle_close(&ctx);
        if !self.is_exporting() {
            self.handle_drops(&ctx);
        }
        self.pump(&ctx);
        if self.modal.is_none() && self.active_export.is_none() {
            self.handle_keys(&ctx);
        }

        egui::Panel::top("toolbar").show(ui, |ui| toolbar::show(self, ui));
        egui::Panel::bottom("status").show_separator_line(false).exact_size(28.0).show(ui, |ui| crate::status::show(self, ui));
        if self.retakes.visible() {
            egui::Panel::right("retakes").resizable(true).default_size(340.0).min_size(260.0).show(ui, |ui| {
                ui.add_enabled_ui(!self.is_exporting(), |ui| retakes::panel(self, ui));
            });
        }
        egui::Panel::bottom("timeline").resizable(true).default_size(250.0).min_size(170.0).show(ui, |ui| {
            ui.add_enabled_ui(!self.is_exporting(), |ui| timeline::show(self, ui));
        });
        egui::Panel::left("library").resizable(true).default_size(330.0).min_size(240.0).show(ui, |ui| {
            ui.add_enabled_ui(!self.is_exporting(), |ui| library::show(self, ui));
        });
        egui::CentralPanel::default().show(ui, |ui| {
            ui.add_enabled_ui(!self.is_exporting(), |ui| preview::show(self, ui));
        });

        self.modal_ui(&ctx);

        if let Some(job) = &self.active_export {
            let t = theme::tokens();
            let mut cancel_clicked = false;
            egui::Window::new("Exporting video").collapsible(false).resizable(false).anchor(Align2::CENTER_CENTER, [0.0, 0.0]).show(&ctx, |ui| {
                let frac = job.fraction();
                ui.add(egui::ProgressBar::new(frac).text(format!("{:.0}%", frac * 100.0)).desired_width(280.0));
                ui.add_space(6.0);
                ui.label(RichText::new(job.status()).color(t.text));
                if let Some(eta) = job.eta() {
                    let s = eta.as_secs();
                    ui.label(RichText::new(format!("{:02}:{:02} remaining", s / 60, s % 60)).color(t.text_dim));
                }
                ui.add_space(8.0);
                if ui.button("Cancel export").clicked() {
                    cancel_clicked = true;
                }
            });
            if cancel_clicked {
                self.queue(Action::CancelExport);
            }
        }

        self.update_title(&ctx);

        let actions: Vec<Action> = std::mem::take(&mut self.queue);
        for a in actions {
            self.perform(a);
        }
    }
}

impl Drop for SimpleCutApp {
    fn drop(&mut self) {
        self.editor.shutdown();
    }
}

impl eframe::App for SimpleCutApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // a bug in one panel must not close the window and lose the user's work
        if let Err(err) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.frame(ui))) {
            let msg = if let Some(s) = err.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = err.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic in UI frame".to_string()
            };
            log::error!("the interface hit an internal error and carried on: {msg}");
            write_crash_log(&msg, self.editor.project_path());
            self.banner = Some(Banner { text: "Something went wrong in the interface. Your project is safe; try again.".into(), error: true, action: None });
            self.queue.clear();
        }
    }
}

/// Write a crash log entry when an internal UI error or panic occurs.
pub fn write_crash_log(msg: &str, project_path: Option<&str>) {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let log_line =
        format!("[{now}] SimpleCut Crash / Internal Error\nMessage: {msg}\nProject: {}\nOS: {}\n\n", project_path.unwrap_or("none"), std::env::consts::OS);

    if let Some(p) = project_path {
        let path = std::path::Path::new(p);
        if let Some(parent) = path.parent() {
            let crash_file = parent.join("simplecut-crash.log");
            let _ = std::fs::OpenOptions::new().create(true).append(true).open(crash_file).and_then(|mut f| {
                use std::io::Write;
                f.write_all(log_line.as_bytes())
            });
        }
    }

    let _ = std::fs::OpenOptions::new().create(true).append(true).open("simplecut-crash.log").and_then(|mut f| {
        use std::io::Write;
        f.write_all(log_line.as_bytes())
    });
}

pub(crate) fn to_image(f: &PreviewFrame) -> egui::ColorImage {
    // egui panics when the bytes do not match the size, so a damaged frame becomes black
    if f.width > 0 && f.height > 0 && f.rgba.len() == f.width * f.height * 4 {
        egui::ColorImage::from_rgba_unmultiplied([f.width, f.height], &f.rgba)
    } else {
        egui::ColorImage::from_rgba_unmultiplied([2, 2], &[0, 0, 0, 255].repeat(4))
    }
}
