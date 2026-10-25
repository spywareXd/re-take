//! Read-only snapshots of the project for the UI to draw. Plain data: nothing here can edit.

use filmcraft_media::MediaKind;
use filmcraft_project::{ClipId, ItemId, TrackId, TrackKind};
use filmcraft_time::Tick;

/// One imported file, as the media library lists it.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaEntry {
    pub id: ItemId,
    pub name: String,
    pub kind: MediaKind,
    pub duration: Tick,
    /// Width, height and frames per second of the picture, if there is one.
    pub video: Option<(u32, u32, f64)>,
    /// Channels and sample rate of the sound, if there is some.
    pub audio: Option<(u32, u32)>,
    pub path: Option<String>,
    pub file_size: Option<u64>,
    /// The file is missing, unreadable or was made offline on purpose.
    pub offline: bool,
    /// Why it is offline, when known.
    pub offline_reason: Option<String>,
}

/// One clip on the timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipView {
    pub id: ClipId,
    pub item: ItemId,
    pub name: String,
    pub start: Tick,
    pub duration: Tick,
    /// Where in the source file the clip begins.
    pub source_in: Tick,
    /// Clips with the same link move, trim and split together (a video and its sound).
    pub link: Option<u64>,
    pub enabled: bool,
    pub gain_db: f64,
    pub selected: bool,
}

impl ClipView {
    pub fn end(&self) -> Tick {
        self.start + self.duration
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrackView {
    pub id: TrackId,
    pub kind: TrackKind,
    pub name: String,
    pub muted: bool,
    pub volume_db: f64,
    pub locked: bool,
    pub clips: Vec<ClipView>,
}

/// The open sequence: tracks top to bottom (video first, then audio).
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceView {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration: Tick,
    pub playhead: Tick,
    pub tracks: Vec<TrackView>,
}

impl SequenceView {
    pub fn clip(&self, id: ClipId) -> Option<&ClipView> {
        self.tracks.iter().flat_map(|t| t.clips.iter()).find(|c| c.id == id)
    }

    pub fn track_of(&self, id: ClipId) -> Option<&TrackView> {
        self.tracks.iter().find(|t| t.clips.iter().any(|c| c.id == id))
    }

    pub fn clip_count(&self) -> usize {
        self.tracks.iter().map(|t| t.clips.len()).sum()
    }
}

/// What an import did: the new media, and the files that could not be read (with the reason).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ImportReport {
    pub imported: Vec<ItemId>,
    pub failed: Vec<String>,
}
