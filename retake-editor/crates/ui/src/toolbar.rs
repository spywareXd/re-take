//! The top bar: project, import, undo / redo and the editing buttons.

use egui::{Align, Layout, RichText};

use crate::app::{Action, RetakeEditorApp};
use crate::theme;

pub(crate) fn show(app: &mut RetakeEditorApp, ui: &mut egui::Ui) {
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
        "Export the video, audio, subtitles, cut list or transcript (Ctrl+E)"
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
                primary_button(app, ui, "Remove Retakes", tip, Action::StartRetakeAnalysis, has_timeline && !busy && !is_exporting);
                button(
                    app,
                    ui,
                    "Options",
                    "Transcription model and sentence splitting for Remove Retakes",
                    Action::Options(crate::retake_options::OptionsAction::Open),
                    !is_exporting,
                );
            }
            ui.add_space(8.0);
            let dirty = app.editor.is_dirty();
            let name = app.editor.project_name().to_string();
            let label = if dirty { format!("{name} •") } else { name };
            ui.label(RichText::new(label).color(if dirty { t.text } else { t.text_dim }));
        });
    });
}

/// The primary action of the bar (Remove Retakes): an ordinary button, so its height, corners, border and font are the same as the others, filled with the accent colour.
/// Hover is lighter, pressed slightly darker, keyboard focus draws a bright outline, and while the service works a thin bar along the bottom shows the progress (the stage is
/// in the tooltip). The dark label on the accent fill has a contrast ratio of about 6:1 (WCAG AA needs 4.5:1).
fn primary_button(app: &mut RetakeEditorApp, ui: &mut egui::Ui, label: &str, tip: &str, action: Action, enabled: bool) {
    let t = theme::tokens();
    let progress = app.analysis_progress();
    let live = enabled && progress.is_none();
    let tip = match &progress {
        Some((status, fraction)) if !status.is_empty() => format!("{tip} ({status}, {}%)", (fraction * 100.0).round() as u32),
        _ => tip.to_string(),
    };
    let response = ui
        .scope(|ui| {
            if live {
                let v = ui.visuals_mut();
                v.widgets.inactive.weak_bg_fill = t.accent;
                v.widgets.inactive.fg_stroke.color = t.bg;
                v.widgets.hovered.weak_bg_fill = t.accent.lerp_to_gamma(egui::Color32::WHITE, HOVER_LIGHTEN);
                v.widgets.hovered.fg_stroke.color = t.bg;
                v.widgets.active.weak_bg_fill = t.accent.lerp_to_gamma(egui::Color32::BLACK, PRESSED_DARKEN);
                v.widgets.active.fg_stroke.color = t.bg;
                v.selection.stroke = egui::Stroke::new(2.5, t.text);
            }
            ui.add_enabled(live, egui::Button::new(label))
        })
        .inner
        .on_hover_text(tip);
    if let Some((_, fraction)) = &progress {
        let r = response.rect;
        let bar = egui::Rect::from_min_size(egui::pos2(r.left() + 3.0, r.bottom() - 5.0), egui::vec2((r.width() - 6.0) * fraction, 3.0));
        ui.painter().rect_filled(bar, egui::CornerRadius::same(1), t.accent);
    }
    if live && response.clicked() {
        app.queue(action);
    }
}

const HOVER_LIGHTEN: f32 = 0.18; // how far the fill moves towards white on hover
const PRESSED_DARKEN: f32 = 0.08; // ... and towards black while pressed (small: the dark label must stay at 4.5:1 or better)

fn button(app: &mut RetakeEditorApp, ui: &mut egui::Ui, label: &str, tip: &str, action: Action, enabled: bool) {
    let r = ui.add_enabled(enabled, egui::Button::new(label)).on_hover_text(tip);
    if r.clicked() {
        app.queue(action);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use egui::Color32;

    use crate::theme;

    fn luminance(c: Color32) -> f64 {
        let f = |v: u8| {
            let v = v as f64 / 255.0;
            if v <= 0.03928 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
        };
        0.2126 * f(c.r()) + 0.7152 * f(c.g()) + 0.0722 * f(c.b())
    }

    fn contrast(a: Color32, b: Color32) -> f64 {
        let (x, y) = (luminance(a), luminance(b));
        (x.max(y) + 0.05) / (x.min(y) + 0.05)
    }

    /// WCAG AA for normal text is 4.5:1. The label colour on the normal, hovered and pressed fills of the Remove Retakes button.
    #[test]
    fn the_primary_button_label_meets_wcag_aa_in_every_state() {
        let t = theme::tokens();
        for (name, fill) in [
            ("normal", t.accent),
            ("hovered", t.accent.lerp_to_gamma(Color32::WHITE, super::HOVER_LIGHTEN)),
            ("pressed", t.accent.lerp_to_gamma(Color32::BLACK, super::PRESSED_DARKEN)),
        ] {
            let ratio = contrast(t.bg, fill);
            assert!(ratio >= 4.5, "{name}: contrast {ratio:.2}:1 is below 4.5:1");
        }
        assert!(contrast(t.text, t.accent) < 4.5, "sanity: light text on the accent would NOT pass, which is why the label is dark");
    }

    /// The focus ring must stand out from the button fill and from the bar behind it.
    #[test]
    fn the_focus_ring_is_visible_against_the_fill_and_the_bar() {
        let t = theme::tokens();
        assert!(contrast(t.text, t.accent) >= 2.5, "ring vs fill");
        assert!(contrast(t.text, t.panel) >= 7.0, "ring vs bar");
    }
}
