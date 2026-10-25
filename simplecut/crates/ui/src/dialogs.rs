//! File dialogs, behind a trait so the window can be driven by tests with no operating-system
//! dialogs. The app supplies the real ones.

use std::path::PathBuf;

pub trait Dialogs {
    /// Choose video / audio files to import. Empty: cancelled.
    fn pick_media(&self) -> Vec<PathBuf>;
    /// Choose a project to open.
    fn pick_project_to_open(&self) -> Option<PathBuf>;
    /// Choose where to save the project; `suggested_name` is a file name to offer.
    fn pick_project_save_path(&self, suggested_name: &str) -> Option<PathBuf>;
    /// Choose where to export video; `suggested_name` is a file name to offer.
    fn pick_export_path(&self, suggested_name: &str) -> Option<PathBuf>;
    /// Choose a replacement file to relink offline media; `suggested_name` is the original file name.
    fn pick_relink_path(&self, suggested_name: &str) -> Option<PathBuf>;
    /// Choose a retake analysis file (JSON, see docs/retake-contract.md).
    fn pick_analysis_file(&self) -> Option<PathBuf>;
    /// Choose where to save a retake groups document; `suggested_name` is a file name to offer.
    fn pick_groups_save_path(&self, _suggested_name: &str) -> Option<PathBuf> {
        None
    }
    /// Open the system file browser showing the given file/folder.
    fn open_folder(&self, path: &std::path::Path);
}

/// Dialogs that are always cancelled.
pub struct NoDialogs;

impl Dialogs for NoDialogs {
    fn pick_media(&self) -> Vec<PathBuf> {
        Vec::new()
    }

    fn pick_project_to_open(&self) -> Option<PathBuf> {
        None
    }

    fn pick_project_save_path(&self, _suggested_name: &str) -> Option<PathBuf> {
        None
    }

    fn pick_export_path(&self, _suggested_name: &str) -> Option<PathBuf> {
        None
    }

    fn pick_relink_path(&self, _suggested_name: &str) -> Option<PathBuf> {
        None
    }

    fn pick_analysis_file(&self) -> Option<PathBuf> {
        None
    }

    fn open_folder(&self, _path: &std::path::Path) {}
}
