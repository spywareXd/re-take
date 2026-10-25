//! SimpleCut's window: media library on the left, preview in the middle, timeline at the bottom,
//! essential controls in the toolbar. Everything the user can do is an [`Action`]; the panels only
//! draw and queue actions, and [`SimpleCutApp::perform`] is the one place they are carried out, so
//! tests and the buttons share a path.

mod app;
mod dialogs;
mod format;
mod library;
mod preview;
mod retakes;
mod status;
mod status_text;
mod theme;
mod timeline;
mod tl_math;
mod toolbar;

pub use app::{Action, Banner, SimpleCutApp, Source, write_crash_log};
pub use dialogs::{Dialogs, NoDialogs};
