//! SimpleCut's headless editor layer.
//!
//! No UI crates here (the layering rule FilmCraft enforces): everything in this crate runs and is
//! tested without a window. [`Editor`] is a thin, typed front over FilmCraft's engine `Session`,
//! so projects are FilmCraft's native `.fcproj` and undo/redo is the engine's.

mod analyzer;
mod editor;
mod error;
mod export;
mod player;
mod renderer;
pub mod retake;
pub mod review;
mod snapshot;
mod thumbnails;
mod view;

pub use analyzer::{AdjudicateRequest, AnalysisJob, AnalysisRequest, FakeService, JobState, ResolveRequest, RetakeService};
pub use editor::{Edge, Editor, ItemId, Lane, PlaceAt, TrimMode};
pub use error::{EditorError, Result};
pub use export::{ExportJob, ExportOptions, ExportState, verify_export};
pub use player::{AudioOutput, FillFn, PlayState, Player};
pub use renderer::{PreviewRenderer, RenderedFrame};
pub use retake::{Analysis, ApplyReport, CutProposal, CutScope, Decision, Group, Presence, Rejected, RetakeDocument, groups, groups_with_offset};
pub use review::{Advice, Candidates, Capabilities, Checked, Groups, Label, LabelsRequest, Pair, PairClass, ReviewGroup, Sentence, Stage, Verdict};
pub use snapshot::{PreviewFrame, Snapshot, Target};
pub use thumbnails::Thumbnails;
pub use view::{ClipView, ImportReport, MediaEntry, SequenceView, TrackView};

pub use filmcraft_engine::autosave::RecoveryCandidate;
pub use filmcraft_media::MediaKind;
pub use filmcraft_project::{ClipId, TrackId, TrackKind};
pub use filmcraft_time::{FrameRate, Tick};

/// The FilmCraft commit this crate is built and tested against.
pub const FILMCRAFT_BASE_COMMIT: &str = "a504f3c5";
