//! The thin bar at the bottom: the latest message, or a hint.

use egui::{Align, Layout, RichText};

use crate::app::{Action, SimpleCutApp};
use crate::theme;

pub(crate) fn show(app: &mut SimpleCutApp, ui: &mut egui::Ui) {
    let t = theme::tokens();
    ui.horizontal_centered(|ui| match app.banner.clone() {
        Some(b) => {
            ui.label(RichText::new(&b.text).color(if b.error { t.danger } else { t.text }));
            if let Some((label, action)) = b.action {
                ui.add_space(8.0);
                if ui.small_button(label).clicked() {
                    app.queue(action);
                }
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.small_button("Dismiss").clicked() {
                    app.queue(Action::DismissBanner);
                }
            });
        }
        None => {
            let hint = if app.editor.media().is_empty() {
                "Import video or audio to get started, or drop files here."
            } else {
                "Space plays and pauses. S splits at the playhead."
            };
            ui.label(RichText::new(hint).color(t.text_dim));
        }
    });
}
