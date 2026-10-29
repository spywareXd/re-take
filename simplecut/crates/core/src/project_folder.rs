//! The project folder: saving a project named X makes `X/` with the project file `X/X.<ext>` and a `data/` folder next to it. The backend (the retake service) keeps
//! everything it computes for the project in `X/data/`; this module only decides where the folder is and creates it. Nothing here deletes or moves anything.

use std::path::{Path, PathBuf};

/// The folders of `data/` the backend uses (created up front so the layout is the same for every project).
pub const DATA_SUBDIRS: [&str; 4] = ["audio", "cache", "stale", "exports"];

/// The project folder a project file belongs to: `X/X.<ext>` gives `X/`. A file that does not sit in a folder of its own name belongs to no project folder
/// (an old project, or one saved by hand elsewhere), and the backend then falls back to its old locations.
pub fn folder_of(project_file: &Path) -> Option<PathBuf> {
    let stem = project_file.file_stem()?;
    let dir = project_file.parent()?;
    (dir.file_name()? == stem).then(|| dir.to_path_buf())
}

/// Where a project should be saved when the user picked `chosen` in the Save dialog: `<chosen's folder>/X/X.<ext>`, unless it already is `X/X.<ext>`.
pub fn layout_for(chosen: &Path) -> PathBuf {
    if folder_of(chosen).is_some() {
        return chosen.to_path_buf();
    }
    let (Some(stem), Some(parent), Some(name)) = (chosen.file_stem(), chosen.parent(), chosen.file_name()) else { return chosen.to_path_buf() };
    parent.join(stem).join(name)
}

/// Create the project folder and its `data/` layout for a project file path (existing folders and files are left alone).
pub fn prepare(project_file: &Path) -> std::io::Result<PathBuf> {
    let root = folder_of(project_file)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "the project file is not inside a folder of its own name"))?;
    std::fs::create_dir_all(&root)?;
    for sub in DATA_SUBDIRS {
        std::fs::create_dir_all(root.join("data").join(sub))?;
    }
    Ok(root)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn a_project_file_in_a_folder_of_its_own_name_has_a_project_folder() {
        assert_eq!(folder_of(Path::new("C:/work/Film/Film.fcproj")), Some(PathBuf::from("C:/work/Film")));
        assert_eq!(folder_of(Path::new("C:/work/Film/Other.fcproj")), None);
        assert_eq!(folder_of(Path::new("Film.fcproj")), None);
        assert_eq!(folder_of(Path::new("")), None);
    }

    #[test]
    fn saving_makes_the_folder_layout_without_nesting_twice() {
        assert_eq!(layout_for(Path::new("C:/work/Film.fcproj")), PathBuf::from("C:/work/Film/Film.fcproj"));
        assert_eq!(layout_for(Path::new("C:/work/Film/Film.fcproj")), PathBuf::from("C:/work/Film/Film.fcproj"), "already in its folder: unchanged");
        assert_eq!(layout_for(Path::new("C:/work/Film/Draft.fcproj")), PathBuf::from("C:/work/Film/Draft/Draft.fcproj"));
    }

    #[test]
    fn prepare_creates_the_data_layout_and_touches_nothing_else() {
        let dir = std::env::temp_dir().join(format!("simplecut-folder-test-{}", std::process::id()));
        let file = dir.join("Film").join("Film.fcproj");
        let root = prepare(&file).unwrap();
        for sub in DATA_SUBDIRS {
            assert!(root.join("data").join(sub).is_dir());
        }
        std::fs::write(root.join("data").join("keep.txt"), "mine").unwrap();
        prepare(&file).unwrap();
        assert_eq!(std::fs::read_to_string(root.join("data").join("keep.txt")).unwrap(), "mine", "a second call leaves existing files alone");
        assert!(prepare(&dir.join("Film").join("Other.fcproj")).is_err());
        let _ = std::fs::remove_dir_all(&dir); // this test's own scratch folder
    }
}
