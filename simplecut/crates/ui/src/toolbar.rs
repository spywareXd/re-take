//! The top bar: project, import, undo / redo and the editing buttons.

use egui::{Align, Layout, RichText};

use crate::app::{Action, SimpleCutApp};
use crate::theme;

pub(crate) fn show(app: &mut SimpleCutApp, ui: &mut egui::Ui) {
    let t = theme::tokens();
    let is_exporting = app.is_exporting();
    let can_edit = !is_exporting;
    let can_undo = app.editor.can_undo() && can_edit;
    let can_redo = app.editor.can_redo() && can_edit;
    let has_selection = !app.editor.selection().is_empty() && can_edit;
    let has_timeline = app.editor.sequence().is_some_and(|s| s.clip_count() > 0);
    let can_export = has_timeline && !is_exporting;
    let export_tip = if is_exporting {
        "Exporting in progress..."
    } else if !has_timeline {
        "Add media to the timeline before exporting."
    } else {
        "Export video (Ctrl+E)"
    };
    ui.horizontal(|ui| {
        ui.add_space(2.0);
        button(app, ui, "New", "Start a new project (Ctrl+N)", Action::NewProject, can_edit);
        button(app, ui, "Open", "Open a project (Ctrl+O)", Action::OpenProject, can_edit);
        button(app, ui, "Save", "Save the project (Ctrl+S)", Action::Save, can_edit);
        ui.separator();
        button(app, ui, "Import", "Add video or audio files to the project (Ctrl+I)", Action::Import, can_edit);
        ui.separator();
        button(app, ui, "Undo", "Undo the last change (Ctrl+Z)", Action::Undo, can_undo);
        button(app, ui, "Redo", "Redo (Ctrl+Shift+Z)", Action::Redo, can_redo);
        ui.separator();
        button(app, ui, "Split", "Cut the clip at the playhead (S)", Action::Split, has_timeline && can_edit);
        button(app, ui, "Delete", "Delete the selected clips and close the gap (Delete)", Action::Delete, has_selection);

        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            button(app, ui, "Shortcuts", "View keyboard shortcuts (F1)", Action::OpenShortcuts, true);
            ui.separator();
            button(app, ui, "Export", export_tip, Action::OpenExportDialog, can_export);
            if app.has_service() {
                let busy = app.is_analyzing();
                let tip = if is_exporting {
                    "Wait for the export to finish."
                } else if busy {
                    "Looking for retakes…"
                } else if !has_timeline {
                    "Add a video to the timeline first."
                } else {
                    "Find failed takes in the video and review them before cutting."
                };
                button(app, ui, "Remove Retakes", tip, Action::StartRetakeAnalysis, has_timeline && !busy && !is_exporting);
            }
            ui.add_space(8.0);
            let dirty = app.editor.is_dirty();
            let name = app.editor.project_name().to_string();
            let label = if dirty { format!("{name} •") } else { name };
            ui.label(RichText::new(label).color(if dirty { t.text } else { t.text_dim }));
        });
    });
}

fn button(app: &mut SimpleCutApp, ui: &mut egui::Ui, label: &str, tip: &str, action: Action, enabled: bool) {
    let r = ui.add_enabled(enabled, egui::Button::new(label)).on_hover_text(tip);
    if r.clicked() {
        app.queue(action);
    }
}
