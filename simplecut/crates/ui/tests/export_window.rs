//! Tests for the Export dialog, overwrite confirmation, active export progress, cancellation,
//! and opening the containing folder from the status bar.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::harness::{click, skip, step_until, with_media};
use egui_kittest::kittest::{NodeT, Queryable};
use simplecut_ui::Action;

#[test]
fn export_button_disabled_when_timeline_is_empty_and_enabled_with_media() {
    if skip() {
        return;
    }
    let (mut h, _, _) = with_media();
    // Before adding to timeline, timeline is empty
    assert!(h.get_by_label("Export").accesskit_node().is_disabled());

    // Add to timeline
    click(&mut h, "Add to timeline");
    assert!(!h.get_by_label("Export").accesskit_node().is_disabled());
}

#[test]
fn export_dialog_can_browse_and_cancel() {
    if skip() {
        return;
    }
    let (mut h, script, dir) = with_media();
    click(&mut h, "Add to timeline");

    // Click Export in toolbar
    click(&mut h, "Export");

    // "Export video" window is open
    h.get_by_label("Export video");
    h.get_by_label("Preset: YouTube 1080p Full HD (1920×1080, H.264 + AAC)");

    // Script an export path
    let chosen = dir.join("my_youtube_cut.mp4");
    script.export.lock().unwrap().push_back(chosen.clone());

    click(&mut h, "Browse...");
    h.get_by_label("my_youtube_cut.mp4");

    // Cancel closes dialog
    click(&mut h, "Cancel");
    assert!(h.query_by_label("Export video").is_none());
}

#[test]
fn export_full_flow_verifies_and_open_folder_works() {
    if skip() {
        return;
    }
    let (mut h, script, dir) = with_media();
    click(&mut h, "Add to timeline");

    let out_file = dir.join("exported_final.mp4");
    script.export.lock().unwrap().push_back(out_file.clone());

    click(&mut h, "Export");
    click(&mut h, "Browse...");
    click(&mut h, "Start export");

    // Now export is actively running
    assert!(h.state().is_exporting());

    // Wait for export to finish
    step_until(&mut h, "export to finish", |a| !a.is_exporting());

    // Exported file exists on disk
    assert!(out_file.exists());

    // Banner indicates success
    let banner = h.state().banner().unwrap().clone();
    assert!(!banner.error);
    assert!(banner.text.contains("Export finished: exported_final.mp4"));

    // Click "Open folder" button in the status bar
    click(&mut h, "Open folder");
    let opened = script.opened_folders.lock().unwrap().clone();
    assert_eq!(opened, vec![out_file]);
}

#[test]
fn export_over_existing_file_requires_confirmation() {
    if skip() {
        return;
    }
    let (mut h, script, dir) = with_media();
    click(&mut h, "Add to timeline");

    let existing = dir.join("existing.mp4");
    std::fs::write(&existing, b"pre-existing content").unwrap();
    script.export.lock().unwrap().push_back(existing.clone());

    click(&mut h, "Export");
    click(&mut h, "Browse...");

    // Clicking Start export on existing file shows overwrite warning
    click(&mut h, "Start export");
    h.get_by_label("A file with this name already exists. Overwrite?");

    // Clicking Cancel in overwrite prompt keeps the dialog without starting export
    click(&mut h, "Cancel");
    assert!(!h.state().is_exporting());
    h.get_by_label("Export video");

    // Clicking Start export then Overwrite starts the job
    click(&mut h, "Start export");
    click(&mut h, "Overwrite");
    assert!(h.state().is_exporting());

    // Cancel active export
    click(&mut h, "Cancel export");
    assert!(!h.state().is_exporting());
}

#[test]
fn ctrl_e_opens_export_dialog() {
    if skip() {
        return;
    }
    let (mut h, _, _) = with_media();
    click(&mut h, "Add to timeline");

    // Trigger OpenExportDialog action directly
    h.state_mut().perform(Action::OpenExportDialog);
    h.step();
    h.get_by_label("Export video");
    click(&mut h, "Cancel");
}

#[test]
fn export_blocks_editing_interactions_while_running() {
    if skip() {
        return;
    }
    let (mut h, script, dir) = with_media();
    click(&mut h, "Add to timeline");
    let initial_clip_count = h.state().editor().sequence().unwrap().clip_count();

    let out_file = dir.join("locked_export.mp4");
    script.export.lock().unwrap().push_back(out_file);

    click(&mut h, "Export");
    click(&mut h, "Browse...");
    click(&mut h, "Start export");
    assert!(h.state().is_exporting());

    // 1. Toolbar editing buttons are disabled
    assert!(h.get_by_label("Split").accesskit_node().is_disabled());
    assert!(h.get_by_label("Undo").accesskit_node().is_disabled());
    assert!(h.get_by_label("Import").accesskit_node().is_disabled());

    // 2. Timeline lanes widget is disabled by panel-level disabled state
    assert!(h.get_by_label("Timeline lanes").accesskit_node().is_disabled());

    // 3. Attempting mutating actions directly is rejected with an error banner
    let music = h.state().editor().media().iter().find(|m| m.name == "bed.wav").unwrap().id;
    h.state_mut().perform(Action::AddToTimeline(music, simplecut_core::Lane::Music));
    h.step();
    assert_eq!(
        h.state().banner().map(|b| (b.error, b.text.as_str())),
        Some((true, "This can't be done right now: Cannot make changes while an export is in progress."))
    );
    assert_eq!(h.state().editor().sequence().unwrap().clip_count(), initial_clip_count, "clip count unchanged");

    // 4. Cancel export restores normal interaction
    click(&mut h, "Cancel export");
    assert!(!h.state().is_exporting());
    assert!(!h.get_by_label("Split").accesskit_node().is_disabled());
    assert!(!h.get_by_label("Timeline lanes").accesskit_node().is_disabled());
}
