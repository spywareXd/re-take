//! The preview: the picture, a seek bar, and the play controls. Shows either the timeline or one
//! library file (with buttons to put it on the timeline).

use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Shape, Stroke, WidgetInfo, WidgetType, pos2, vec2};
use simplecut_core::{Lane, MediaKind, Tick};

use crate::app::{Action, SimpleCutApp, Source};
use crate::library::fit_inside;
use crate::theme;
use crate::{format, status_text};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Icon {
    Start,
    Play,
    Pause,
    End,
}

pub(crate) fn show(app: &mut SimpleCutApp, ui: &mut egui::Ui) {
    let t = theme::tokens();
    let media = match app.source {
        Source::Media(item) => app.editor.media_entry(item),
        Source::Timeline => None,
    };
    let bottom = 22.0 + 40.0 + if media.is_some() { 40.0 } else { 0.0 } + 16.0;
    let size = vec2(ui.available_width(), (ui.available_height() - bottom).max(120.0));
    let (area, _) = ui.allocate_exact_size(size, Sense::hover());
    app.preview_width = area.width();

    // the picture
    let painter = ui.painter();
    painter.rect_filled(area, CornerRadius::same(6), Color32::BLACK);
    if let Some(tex) = &app.preview_texture {
        let fit = fit_inside(area.shrink(2.0), tex.size_vec2());
        painter.image(tex.id(), fit, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
    }
    let (headline, detail) = placeholder(app, media.as_ref());
    if app.preview_texture.is_none() || media.as_ref().is_some_and(|m| m.kind == MediaKind::AudioOnly) {
        painter.text(area.center() - vec2(0.0, 10.0), Align2::CENTER_CENTER, headline, FontId::proportional(18.0), t.text);
        painter.text(area.center() + vec2(0.0, 14.0), Align2::CENTER_CENTER, detail, FontId::proportional(13.0), t.text_dim);
    }
    let label = match &media {
        Some(m) => format!("Preview: {}", m.name),
        None => "Timeline".to_string(),
    };
    let pill = painter.layout_no_wrap(label, FontId::proportional(12.0), t.text);
    let pill_rect = Rect::from_min_size(area.min + vec2(10.0, 10.0), pill.size() + vec2(16.0, 8.0));
    painter.rect_filled(pill_rect, CornerRadius::same(10), Color32::from_black_alpha(160));
    painter.galley(pill_rect.min + vec2(8.0, 4.0), pill, t.text);

    // the seek bar
    let (position, duration, fps) = (app.source_time(), app.source_duration(), app.source_frame_rate().as_f64());
    if let Some(to) = seek_bar(ui, position, duration) {
        app.queue(Action::Seek(to));
    }

    // play controls
    let playing = app.playing_source();
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(format!("{} / {}", format::timecode(position, fps), format::timecode(duration, fps))).monospace().color(t.text));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(ui.available_width() / 2.0 - 70.0);
            if icon_button(ui, Icon::End, "Go to end", true).clicked() {
                app.queue(Action::SeekEnd);
            }
            let (icon, label) = if playing { (Icon::Pause, "Pause") } else { (Icon::Play, "Play") };
            if icon_button(ui, icon, label, duration > Tick::ZERO).clicked() {
                app.queue(Action::TogglePlay);
            }
            if icon_button(ui, Icon::Start, "Go to start", true).clicked() {
                app.queue(Action::SeekStart);
            }
        });
    });

    // putting a previewed file on the timeline
    if let Some(m) = media {
        ui.horizontal(|ui| {
            if m.offline {
                ui.label(egui::RichText::new(m.offline_reason.clone().unwrap_or_else(|| "This file is offline.".into())).color(t.danger));
                if ui.button("Relink media…").clicked() {
                    app.queue(Action::RelinkMedia(m.id));
                }
            }
            if !m.offline && m.video.is_some() && ui.button("Add to timeline").on_hover_text("Put this at the end of the video track").clicked() {
                app.queue(Action::AddToTimeline(m.id, Lane::Main));
            }
            if !m.offline && m.audio.is_some() {
                if ui.button("Add as music").on_hover_text("Put this on the music track, from the start").clicked() {
                    app.queue(Action::AddToTimeline(m.id, Lane::Music));
                }
                if m.video.is_none() && ui.button("Add to timeline").on_hover_text("Put this on the dialogue track").clicked() {
                    app.queue(Action::AddToTimeline(m.id, Lane::Main));
                }
            }
            if ui.button("Back to timeline").clicked() {
                app.queue(Action::ShowTimeline);
            }
        });
    }
}

/// What to say when there is no picture to show.
fn placeholder(app: &SimpleCutApp, media: Option<&simplecut_core::MediaEntry>) -> (String, String) {
    match media {
        Some(m) if m.kind == MediaKind::AudioOnly || m.video.is_none() => (m.name.clone(), "Audio file".into()),
        Some(_) => ("Loading…".into(), String::new()),
        None => {
            let empty = app.editor.sequence().is_none_or(|s| s.clip_count() == 0);
            if empty { status_text::empty_timeline() } else { ("Loading…".into(), String::new()) }
        }
    }
}

/// A slim bar showing `position` of `duration`; returns the time clicked or dragged to.
fn seek_bar(ui: &mut egui::Ui, position: Tick, duration: Tick) -> Option<Tick> {
    let t = theme::tokens();
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 22.0), Sense::click_and_drag());
    resp.widget_info(|| WidgetInfo::labeled(WidgetType::Slider, true, "Seek"));
    let bar = Rect::from_center_size(rect.center(), vec2(rect.width() - 8.0, 6.0));
    let frac = if duration > Tick::ZERO { (position.0 as f64 / duration.0 as f64).clamp(0.0, 1.0) as f32 } else { 0.0 };
    let painter = ui.painter();
    painter.rect_filled(bar, CornerRadius::same(3), t.lane);
    painter.rect_filled(Rect::from_min_max(bar.min, pos2(bar.left() + bar.width() * frac, bar.bottom())), CornerRadius::same(3), t.accent);
    painter.circle_filled(pos2(bar.left() + bar.width() * frac, bar.center().y), 6.0, Color32::WHITE);
    if (resp.clicked() || resp.dragged() || resp.drag_started()) && duration > Tick::ZERO {
        let x = resp.interact_pointer_pos()?.x;
        let f = ((x - bar.left()) / bar.width().max(1.0)).clamp(0.0, 1.0);
        let to = Tick((duration.0 as f64 * f64::from(f)) as i64);
        if to != position {
            return Some(to);
        }
    }
    None
}

fn icon_button(ui: &mut egui::Ui, icon: Icon, label: &str, enabled: bool) -> egui::Response {
    let t = theme::tokens();
    let (rect, resp) = ui.allocate_exact_size(vec2(40.0, 32.0), if enabled { Sense::click() } else { Sense::hover() });
    resp.widget_info(|| WidgetInfo::labeled(WidgetType::Button, enabled, label));
    let painter = ui.painter();
    if enabled && resp.hovered() {
        painter.rect_filled(rect, CornerRadius::same(6), t.panel_raised);
    }
    let color = if enabled { t.text } else { t.text_dim.gamma_multiply(0.5) };
    let c = rect.center();
    let tri = |dir: f32, at: Pos2, s: f32| {
        Shape::convex_polygon(vec![pos2(at.x + dir * s, at.y), pos2(at.x - dir * s, at.y - s), pos2(at.x - dir * s, at.y + s)], color, Stroke::NONE)
    };
    match icon {
        Icon::Play => {
            painter.add(tri(1.0, c, 7.0));
        }
        Icon::Pause => {
            painter.rect_filled(Rect::from_center_size(c - vec2(4.5, 0.0), vec2(4.0, 14.0)), CornerRadius::same(1), color);
            painter.rect_filled(Rect::from_center_size(c + vec2(4.5, 0.0), vec2(4.0, 14.0)), CornerRadius::same(1), color);
        }
        Icon::Start => {
            painter.rect_filled(Rect::from_center_size(c - vec2(6.0, 0.0), vec2(2.5, 14.0)), CornerRadius::same(1), color);
            painter.add(tri(-1.0, c + vec2(2.0, 0.0), 6.0));
        }
        Icon::End => {
            painter.rect_filled(Rect::from_center_size(c + vec2(6.0, 0.0), vec2(2.5, 14.0)), CornerRadius::same(1), color);
            painter.add(tri(1.0, c - vec2(2.0, 0.0), 6.0));
        }
    }
    resp
}
