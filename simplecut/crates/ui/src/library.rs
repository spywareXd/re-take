//! The media library: imported files with a thumbnail, what they are, and whether they are usable.

use egui::{Align2, Color32, CornerRadius, FontId, Rect, Sense, Stroke, StrokeKind, WidgetInfo, WidgetType, pos2, vec2};
use simplecut_core::{Lane, MediaEntry, MediaKind};

use crate::app::{Action, SimpleCutApp, Source, Thumb};
use crate::format;
use crate::theme;

const ROW_HEIGHT: f32 = 68.0;
const THUMB: egui::Vec2 = vec2(96.0, 54.0);

pub(crate) fn show(app: &mut SimpleCutApp, ui: &mut egui::Ui) {
    let t = theme::tokens();
    ui.horizontal(|ui| {
        ui.heading("Media");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Add media").on_hover_text("Add video or audio files (Ctrl+I)").clicked() {
                app.queue(Action::Import);
            }
        });
    });
    ui.separator();

    let media = app.media();
    if media.is_empty() {
        ui.add_space(24.0);
        ui.vertical_centered(|ui| {
            ui.label(egui::RichText::new("No media yet").color(t.text).size(16.0).strong());
            ui.add_space(6.0);
            ui.label(egui::RichText::new("Drag and drop video or audio files here,\nor click Import media to get started.").color(t.text_dim));
            ui.add_space(14.0);
            if ui.button("Import media…").clicked() {
                app.queue(Action::Import);
            }
        });
        return;
    }
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 4.0;
        for m in &media {
            row(app, ui, m);
        }
    });
}

fn row(app: &mut SimpleCutApp, ui: &mut egui::Ui, m: &MediaEntry) {
    let t = theme::tokens();
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), ROW_HEIGHT), Sense::click());
    resp.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, &m.name));
    let selected = app.source == Source::Media(m.id);
    let fill = if selected {
        t.accent.gamma_multiply(0.28)
    } else if resp.hovered() {
        t.panel_raised
    } else {
        t.panel
    };
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(6), fill);
    if selected {
        painter.rect_stroke(rect, CornerRadius::same(6), Stroke::new(1.0, t.accent), StrokeKind::Inside);
    }

    // thumbnail
    let thumb = Rect::from_min_size(pos2(rect.left() + 8.0, rect.center().y - THUMB.y / 2.0), THUMB);
    painter.rect_filled(thumb, CornerRadius::same(4), t.lane);
    match app.thumb_state.get(&m.id) {
        Some(Thumb::Ready(tex)) => {
            let fit = fit_inside(thumb, tex.size_vec2());
            painter.image(tex.id(), fit, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
        }
        Some(Thumb::None) | None if m.kind == MediaKind::AudioOnly || m.video.is_none() => {
            painter.text(thumb.center(), Align2::CENTER_CENTER, "AUDIO", FontId::proportional(12.0), t.music_clip);
        }
        _ => {}
    }

    // text
    let text_left = thumb.right() + 10.0;
    let text_width = (rect.right() - text_left - 8.0).max(40.0);
    let name = ui.painter().layout(m.name.clone(), FontId::proportional(14.0), t.text, text_width);
    ui.painter().galley(pos2(text_left, rect.top() + 12.0), name, t.text);
    let summary = ui.painter().layout(format::media_summary(m), FontId::proportional(12.0), t.text_dim, text_width);
    ui.painter().galley(pos2(text_left, rect.top() + 36.0), summary, t.text_dim);

    if m.offline {
        let badge = Rect::from_min_size(pos2(rect.right() - 66.0, rect.top() + 6.0), vec2(58.0, 16.0));
        ui.painter().rect_filled(badge, CornerRadius::same(8), t.danger.gamma_multiply(0.85));
        ui.painter().text(badge.center(), Align2::CENTER_CENTER, "OFFLINE", FontId::proportional(10.0), Color32::BLACK);
        if resp.hovered() {
            let why = m.offline_reason.clone().unwrap_or_else(|| "This file cannot be used right now.".into());
            resp.clone().on_hover_text(why);
        }
    }

    let lane = if m.video.is_some() { Lane::Main } else { Lane::Music };
    if resp.clicked() {
        app.queue(Action::PreviewMedia(m.id));
    }
    if resp.double_clicked() {
        if m.offline {
            app.queue(Action::RelinkMedia(m.id));
        } else {
            app.queue(Action::AddToTimeline(m.id, lane));
        }
    }
    resp.context_menu(|ui| {
        if m.offline && ui.button("Relink media…").clicked() {
            app.queue(Action::RelinkMedia(m.id));
            ui.close();
        }
        if !m.offline && m.video.is_some() && ui.button("Add to timeline").clicked() {
            app.queue(Action::AddToTimeline(m.id, Lane::Main));
            ui.close();
        }
        if !m.offline && m.audio.is_some() && ui.button("Add as music").clicked() {
            app.queue(Action::AddToTimeline(m.id, Lane::Music));
            ui.close();
        }
        if ui.button("Preview").clicked() {
            app.queue(Action::PreviewMedia(m.id));
            ui.close();
        }
    });
}

/// The largest rectangle of `size`'s shape that fits inside `outer`, centred.
pub(crate) fn fit_inside(outer: Rect, size: egui::Vec2) -> Rect {
    if size.x <= 0.0 || size.y <= 0.0 {
        return outer;
    }
    let k = (outer.width() / size.x).min(outer.height() / size.y);
    Rect::from_center_size(outer.center(), size * k)
}
