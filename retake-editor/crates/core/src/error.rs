//! Errors the UI can show as they are: plain sentences, no command ids.

use filmcraft_engine::EngineError;
use filmcraft_media::MediaError;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EditorError {
    #[error("There is no sequence open.")]
    NoSequence,
    #[error("Nothing is selected.")]
    NothingSelected,
    #[error("There is nothing to split at the playhead.")]
    NothingToSplit,
    #[error("That clip is no longer on the timeline.")]
    NoSuchClip,
    #[error("That media is not in the project.")]
    NoSuchMedia,
    #[error("This project has not been saved yet. Use Save As to choose where.")]
    NeedsPath,
    #[error("The file could not be found: {0}")]
    FileMissing(String),
    #[error("The file could not be opened: {0}")]
    FileUnreadable(String),
    #[error("Nothing was imported. {0}")]
    ImportFailed(String),
    #[error("That media has no sound, so it cannot be used as music.")]
    NoAudio,
    #[error("That media has no picture or sound to place.")]
    NothingToPlace,
    #[error("This can't be done right now: {0}")]
    NotAvailable(String),
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Engine(String),
}

impl From<EngineError> for EditorError {
    fn from(e: EngineError) -> Self {
        match e {
            EngineError::NoSequence => EditorError::NoSequence,
            EngineError::Disabled(_, reason) => EditorError::NotAvailable(reason),
            EngineError::BadParams { msg, .. } => EditorError::Invalid(msg),
            EngineError::Media(MediaError::Offline(m)) => EditorError::FileMissing(m),
            EngineError::Media(MediaError::Io(m)) => EditorError::FileUnreadable(m),
            other => EditorError::Engine(other.to_string()),
        }
    }
}

pub type Result<T> = std::result::Result<T, EditorError>;
