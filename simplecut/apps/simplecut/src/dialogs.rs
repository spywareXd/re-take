//! The operating system's file dialogs.

use std::path::PathBuf;

use simplecut_ui::Dialogs;

const MEDIA: &[&str] =
    &["mp4", "mov", "m4v", "mkv", "webm", "avi", "ts", "mts", "m2ts", "mxf", "mp3", "wav", "aac", "m4a", "flac", "ogg", "opus", "aif", "aiff"];

pub struct NativeDialogs;

impl Dialogs for NativeDialogs {
    fn pick_media(&self) -> Vec<PathBuf> {
        rfd::FileDialog::new()
            .set_title("Import video or audio")
            .add_filter("Video and audio", MEDIA)
            .add_filter("All files", &["*"])
            .pick_files()
            .unwrap_or_default()
    }

    fn pick_project_to_open(&self) -> Option<PathBuf> {
        rfd::FileDialog::new().set_title("Open project").add_filter("Projects", &["fcproj"]).pick_file()
    }

    fn pick_project_save_path(&self, suggested_name: &str) -> Option<PathBuf> {
        let mut path = rfd::FileDialog::new().set_title("Save project").add_filter("Projects", &["fcproj"]).set_file_name(suggested_name).save_file()?;
        if path.extension().is_none() {
            path.set_extension("fcproj");
        }
        Some(path)
    }

    fn pick_export_path(&self, suggested_name: &str) -> Option<PathBuf> {
        let mut path = rfd::FileDialog::new().set_title("Export video").add_filter("MP4 video", &["mp4"]).set_file_name(suggested_name).save_file()?;
        if path.extension().is_none() {
            path.set_extension("mp4");
        }
        Some(path)
    }

    fn pick_relink_path(&self, suggested_name: &str) -> Option<PathBuf> {
        rfd::FileDialog::new()
            .set_title(format!("Locate media for {suggested_name}"))
            .add_filter("Video and audio", MEDIA)
            .add_filter("All files", &["*"])
            .set_file_name(suggested_name)
            .pick_file()
    }

    fn pick_analysis_file(&self) -> Option<PathBuf> {
        rfd::FileDialog::new().set_title("Open retake analysis").add_filter("Retake analysis", &["json"]).pick_file()
    }

    fn pick_groups_save_path(&self, suggested_name: &str) -> Option<PathBuf> {
        let mut path =
            rfd::FileDialog::new().set_title("Save retake groups").add_filter("Retake groups", &["json"]).set_file_name(suggested_name).save_file()?;
        if path.extension().is_none() {
            path.set_extension("json");
        }
        Some(path)
    }

    fn open_folder(&self, path: &std::path::Path) {
        #[cfg(target_os = "windows")]
        {
            let _ = std::process::Command::new("explorer").arg(format!("/select,{}", path.to_string_lossy())).spawn();
        }
        #[cfg(target_os = "macos")]
        {
            let _ = std::process::Command::new("open").arg("-R").arg(path).spawn();
        }
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            if let Some(parent) = path.parent() {
                let _ = std::process::Command::new("xdg-open").arg(parent).spawn();
            }
        }
    }
}
