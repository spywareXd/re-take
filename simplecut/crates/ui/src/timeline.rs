//! The timeline: ruler, one lane per track, clips, playhead.
//!
//! Click a clip to select it (Ctrl or Shift adds). Drag a clip to move it, drag either end of a
//! clip to trim it, drag the ruler or empty space to move the playhead. Ctrl + wheel zooms, the
//! wheel with Shift (or sideways) scrolls. Tracks have a mute button and a volume.
//!
//! A dragged clip snaps to the playhead and to other clips and can never land on top of another
//! clip. Nothing is changed until the mouse is released, so one drag is one undo step.

use egui::{Align2, Color32, CornerRadius, CursorIcon, FontId, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, UiBuilder, WidgetInfo, WidgetType, pos2, vec2};
use simplecut_core::{ClipId, ClipView, Edge, SequenceView, Tick, TrackId, TrackKind};

use crate::app::{Action, SimpleCutApp};
use crate::format;
use crate::retakes::{Mark, MarkKind};
use crate::theme;
use crate::tl_math::{self, Axis, ClipGeom, Hit};

const HEADER_W: f32 = 168.0;
const RULER_H: f32 = 26.0;
const LANE_H: f32 = 48.0;
/// How close (in pixels) the pointer must be to a clip's end to grab it.
const EDGE_PX: f32 = 7.0;
/// How close (in pixels) a dragged edge snaps to something.
const SNAP_PX: f32 = 9.0;

/// What a mouse drag on the timeline is doing right now.
#[derive(Clone, Debug)]
enum Drag {
    /// Moving the playhead.
    Scrub,
    Move {
        clip: ClipId,
        orig: Tick,
        grab: Tick,
        to: Tick,
    },
    Trim {
        clip: ClipId,
        edge: Edge,
        orig: Tick,
        to: Tick,
        regular: bool,
    },
    SentenceEdge {
        sentence: String,
        edge: Edge,
    },
}

/// Where things were drawn last frame (for zoom buttons and tests).
#[derive(Clone, Debug)]
pub(crate) struct Geometry {
    pub axis: Axis,
    pub lane_w: f32,
    pub ruler: Rect,
    pub lanes: Vec<Rect>,
    pub clips: Vec<ClipGeom>,
}

#[derive(Default)]
pub(crate) struct TimelineUi {
    /// Pixels per second; None: the whole timeline fits.
    pub zoom: Option<f64>,
    /// Seconds at the left edge of the lanes.
    pub scroll: f64,
    drag: Option<Drag>,
    pub last: Option<Geometry>,
    /// A track volume being dragged: only applied when the mouse is released.
    track_volume: Option<(TrackId, f64)>,
    clip_volume: Option<f64>,
    /// Where the overview seek slider was drawn last frame (for tests).
    pub seek_slider_rect: Option<egui::Rect>,
    /// Whether playback was active before dragging the seek slider.
    pub slider_was_playing: bool,
    /// Last tick queued by the seek slider (to avoid duplicate seek actions).
    pub slider_last_seek: Option<Tick>,
}

pub(crate) fn show(app: &mut SimpleCutApp, ui: &mut egui::Ui) {
    let t = theme::tokens();
    let Some(seq) = app.editor.sequence() else {
        ui.label("No project open.");
        return;
    };
    seek_slider_row(app, ui, &seq);
    header_row(app, ui, &seq);

    let height = RULER_H + LANE_H * seq.tracks.len() as f32;
    let (outer, _) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::hover());
    let lanes_left = outer.left() + HEADER_W;
    let lanes_rect = Rect::from_min_max(pos2(lanes_left, outer.top()), pos2(outer.right() - 6.0, outer.bottom()));
    let lane_w = lanes_rect.width().max(50.0);
    let resp = ui.interact(lanes_rect, ui.id().with("timeline-lanes"), Sense::click_and_drag());
    resp.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, "Timeline lanes"));

    // the view: scale and scroll
    let fit = tl_math::fit_pps(lane_w, seq.duration);
    let mut pps = app.tl.zoom.map(|z| z.clamp(tl_math::MIN_PPS.min(fit), tl_math::MAX_PPS)).unwrap_or(fit);
    let mut scroll = if app.tl.zoom.is_some() { tl_math::clamp_scroll(app.tl.scroll, pps, lane_w, seq.duration) } else { 0.0 };
    if resp.hovered() {
        // egui turns Ctrl + wheel (and a pinch) into a zoom factor rather than a scroll
        let (delta, zoom) = ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta()));
        if (zoom - 1.0).abs() > f32::EPSILON {
            let anchor = resp.hover_pos().map_or(lanes_left, |p| p.x);
            let z = tl_math::zoom_about(Axis { left: lanes_left, pps, scroll }, f64::from(zoom), anchor, lane_w, seq.duration);
            (pps, scroll) = (z.pps, z.scroll);
            app.tl.zoom = Some(pps);
        } else if delta != egui::Vec2::ZERO && app.tl.zoom.is_some() {
            let d = if delta.x != 0.0 { delta.x } else { delta.y };
            scroll = tl_math::clamp_scroll(scroll - f64::from(d) / pps, pps, lane_w, seq.duration);
        }
    }
    // while playing a zoomed-in timeline, keep the playhead in view
    if app.tl.zoom.is_some() && app.playing_source() {
        let head = tl_math::secs(seq.playhead);
        if head < scroll || head > scroll + f64::from(lane_w) / pps * 0.95 {
            scroll = tl_math::clamp_scroll(head - f64::from(lane_w) / pps * 0.1, pps, lane_w, seq.duration);
        }
    }
    app.tl.scroll = scroll;
    let axis = Axis { left: lanes_left, pps, scroll };

    // where everything is
    let lanes: Vec<Rect> = (0..seq.tracks.len())
        .map(|i| {
            let top = outer.top() + RULER_H + i as f32 * LANE_H;
            Rect::from_min_max(pos2(lanes_left, top + 2.0), pos2(lanes_rect.right(), top + LANE_H - 2.0))
        })
        .collect();
    let mut clips = Vec::new();
    for (i, track) in seq.tracks.iter().enumerate() {
        for c in &track.clips {
            let l = lanes[i];
            let x0 = axis.x_of(c.start);
            let r = Rect::from_min_max(pos2(x0, l.top() + 2.0), pos2(axis.x_of(c.end()).max(x0 + 2.0), l.bottom() - 2.0));
            clips.push(ClipGeom { id: c.id, rect: r });
        }
    }
    let ruler = Rect::from_min_max(lanes_rect.left_top(), pos2(lanes_rect.right(), outer.top() + RULER_H));
    let geom = Geometry { axis, lane_w, ruler, lanes: lanes.clone(), clips: clips.clone() };
    app.tl.last = Some(geom.clone());

    interact(app, ui, &resp, &seq, &geom);

    // ── paint ──
    let painter = ui.painter_at(lanes_rect.expand2(vec2(0.0, 2.0)));
    painter.rect_filled(ruler, CornerRadius::ZERO, t.panel);
    let step = tl_math::ruler_step(pps);
    let mut s = (scroll / step).floor() * step;
    while s <= scroll + f64::from(lane_w) / pps {
        let x = axis.x_of(tl_math::ticks(s));
        painter.line_segment([pos2(x, ruler.bottom() - 7.0), pos2(x, ruler.bottom())], Stroke::new(1.0, t.text_dim));
        painter.text(pos2(x + 3.0, ruler.top() + 5.0), Align2::LEFT_TOP, format::length(tl_math::ticks(s)), FontId::proportional(11.0), t.text_dim);
        s += step;
    }
    for (i, track) in seq.tracks.iter().enumerate() {
        painter.rect_filled(lanes[i], CornerRadius::same(4), t.lane);
        let color = lane_color(track.kind, &track.name, &t);
        for c in &track.clips {
            let Some(g) = clips.iter().find(|g| g.id == c.id) else { continue };
            if g.rect.right() < lanes_rect.left() || g.rect.left() > lanes_rect.right() {
                continue;
            }
            let fill = if track.muted || !c.enabled { color.gamma_multiply(0.35) } else { color };
            painter.rect_filled(g.rect, CornerRadius::same(4), fill);
            if c.selected {
                painter.rect_stroke(g.rect, CornerRadius::same(4), Stroke::new(2.0, Color32::WHITE), StrokeKind::Inside);
            }
            let label_clip = g.rect.shrink2(vec2(4.0, 0.0)).intersect(lanes_rect);
            let text_painter = painter.with_clip_rect(label_clip);
            text_painter.text(
                pos2(g.rect.left().max(lanes_rect.left()) + 6.0, g.rect.center().y),
                Align2::LEFT_CENTER,
                &c.name,
                FontId::proportional(12.0),
                Color32::WHITE,
            );
            if c.gain_db.abs() >= 0.05 {
                text_painter.text(
                    pos2(g.rect.right() - 6.0, g.rect.center().y),
                    Align2::RIGHT_CENTER,
                    format!("{:+.0} dB", c.gain_db),
                    FontId::proportional(11.0),
                    Color32::WHITE.gamma_multiply(0.8),
                );
            }
        }
    }
    if seq.clip_count() == 0 {
        painter.text(
            pos2(lanes_rect.center().x, outer.top() + RULER_H + LANE_H * 1.5),
            Align2::CENTER_CENTER,
            "Double-click a file in the library to add it here",
            FontId::proportional(14.0),
            t.text_dim,
        );
    }
    paint_review(app, &painter, &axis, &lanes);
    paint_drag(app, &painter, &seq, &axis, &lanes, &clips, lanes_rect);

    // playhead
    let x = axis.x_of(seq.playhead);
    if x >= lanes_rect.left() - 1.0 && x <= lanes_rect.right() + 1.0 {
        painter.line_segment([pos2(x, outer.top()), pos2(x, outer.bottom())], Stroke::new(1.5, t.playhead));
        painter.add(Shape::convex_polygon(vec![pos2(x - 6.0, outer.top()), pos2(x + 6.0, outer.top()), pos2(x, outer.top() + 9.0)], t.playhead, Stroke::NONE));
    }

    track_headers(app, ui, &seq, outer);
}

fn lane_color(kind: TrackKind, name: &str, t: &theme::Tokens) -> Color32 {
    match (kind, name) {
        (TrackKind::Video, _) => t.video_clip,
        (_, "Music") => t.music_clip,
        _ => t.dialogue_clip,
    }
}

// ── the overview seek slider ──────────────────────────────────────────────

fn seek_slider_row(app: &mut SimpleCutApp, ui: &mut egui::Ui, seq: &SequenceView) {
    let t = theme::tokens();
    let is_empty = seq.tracks.iter().all(|t| t.clips.is_empty()) || seq.duration == Tick::ZERO;
    let fps = seq.fps.max(1.0);
    let cur_str = format::timecode(seq.playhead, fps);
    let tot_str = format::timecode(seq.duration, fps);
    let time_text = format!("{cur_str} / {tot_str}");

    ui.horizontal(|ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(time_text).monospace().color(t.text_dim));
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                let slider_w = ui.available_width().max(50.0);
                ui.spacing_mut().slider_width = slider_w;

                let max_ticks = seq.duration.0.max(1);
                let mut current_ticks = seq.playhead.0.clamp(0, max_ticks);
                let frame_ticks = tl_math::ticks(1.0 / fps).0.max(1);

                let slider = egui::Slider::new(&mut current_ticks, 0..=max_ticks).step_by(frame_ticks as f64).show_value(false);

                let r = ui.add_enabled(!is_empty, slider);
                r.widget_info(|| WidgetInfo::labeled(WidgetType::Slider, true, "Timeline seek"));
                app.tl.seek_slider_rect = Some(r.rect);

                if r.drag_started() {
                    app.tl.slider_was_playing = app.player.is_playing();
                    if app.tl.slider_was_playing {
                        app.player.pause();
                    }
                    app.tl.slider_last_seek = None;
                }

                let mut stepped = false;
                if r.changed() {
                    let new_tick = Tick(current_ticks);
                    if app.tl.slider_last_seek != Some(new_tick) && new_tick != seq.playhead {
                        app.queue(Action::SeekTimeline(new_tick));
                        app.tl.slider_last_seek = Some(new_tick);
                        stepped = true;
                    }
                }

                if !stepped && r.has_focus() {
                    if ui.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
                        let to = Tick((seq.playhead.0 - frame_ticks).max(0));
                        if to != seq.playhead {
                            app.queue(Action::SeekTimeline(to));
                            app.tl.slider_last_seek = Some(to);
                        }
                    } else if ui.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
                        let to = Tick((seq.playhead.0 + frame_ticks).min(seq.duration.0));
                        if to != seq.playhead {
                            app.queue(Action::SeekTimeline(to));
                            app.tl.slider_last_seek = Some(to);
                        }
                    }
                }

                if r.drag_stopped() {
                    if app.tl.slider_was_playing {
                        let _ = app.player.play(&mut app.editor);
                        app.tl.slider_was_playing = false;
                    }
                    app.tl.slider_last_seek = None;
                }
            });
        });
    });
}

// ── the title row: clip volume and zoom ────────────────────────────────────

fn header_row(app: &mut SimpleCutApp, ui: &mut egui::Ui, seq: &SequenceView) {
    let t = theme::tokens();
    ui.horizontal(|ui| {
        ui.heading("Timeline");
        ui.label(egui::RichText::new(format::timecode(seq.playhead, seq.fps)).monospace().color(t.text_dim));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if small(ui, "Fit", "Fit", "Show the whole timeline").clicked() {
                app.queue(Action::ZoomFit);
            }
            if small(ui, "+", "Zoom in", "Zoom in (Ctrl + wheel)").clicked() {
                app.queue(Action::ZoomIn);
            }
            if small(ui, "−", "Zoom out", "Zoom out (Ctrl + wheel)").clicked() {
                app.queue(Action::ZoomOut);
            }
            ui.separator();
            clip_volume(app, ui, seq);
        });
    });
}

fn small(ui: &mut egui::Ui, text: &str, label: &str, tip: &str) -> egui::Response {
    let r = ui.add(egui::Button::new(text).min_size(vec2(28.0, 0.0))).on_hover_text(tip);
    r.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, label));
    r
}

/// The selection's sound level: shown when something with sound is selected, applied on release.
fn clip_volume(app: &mut SimpleCutApp, ui: &mut egui::Ui, seq: &SequenceView) {
    let selected: Vec<&ClipView> = seq.tracks.iter().filter(|t| t.kind == TrackKind::Audio).flat_map(|t| t.clips.iter()).filter(|c| c.selected).collect();
    let Some(first) = selected.first() else {
        ui.add_enabled(false, egui::Label::new("Select a clip to set its volume"));
        return;
    };
    let mut db = app.tl.clip_volume.unwrap_or(first.gain_db);
    let r = ui.add(egui::DragValue::new(&mut db).range(-60.0..=24.0).speed(0.25).suffix(" dB").max_decimals(1));
    r.widget_info(|| WidgetInfo::labeled(WidgetType::DragValue, true, "Clip volume"));
    ui.label("Volume");
    if r.changed() {
        app.tl.clip_volume = Some(db);
    }
    if (r.drag_stopped() || r.lost_focus())
        && let Some(v) = app.tl.clip_volume.take()
        && (v - first.gain_db).abs() > f64::EPSILON
    {
        app.queue(Action::SetClipVolume(v));
    }
}

// ── track names, mute and volume ───────────────────────────────────────────

fn track_headers(app: &mut SimpleCutApp, ui: &mut egui::Ui, seq: &SequenceView, outer: Rect) {
    let t = theme::tokens();
    for (i, track) in seq.tracks.iter().enumerate() {
        let top = outer.top() + RULER_H + i as f32 * LANE_H;
        let cell = Rect::from_min_max(pos2(outer.left(), top + 2.0), pos2(outer.left() + HEADER_W - 6.0, top + LANE_H - 2.0));
        ui.painter().rect_filled(cell, CornerRadius::same(4), t.panel_raised);
        let mut cui = ui.new_child(UiBuilder::new().max_rect(cell.shrink2(vec2(8.0, 4.0))).layout(egui::Layout::left_to_right(egui::Align::Center)));
        cui.label(egui::RichText::new(&track.name).color(t.text));
        cui.add_space(2.0);
        if track.kind == TrackKind::Audio {
            let m = cui.add(egui::Button::new("M").selected(track.muted).min_size(vec2(24.0, 0.0))).on_hover_text(if track.muted { "Unmute" } else { "Mute" });
            m.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, format!("Mute {}", track.name)));
            if m.clicked() {
                app.queue(Action::SetTrackMuted(track.id, !track.muted));
            }
            let mut db = match app.tl.track_volume {
                Some((id, v)) if id == track.id => v,
                _ => track.volume_db,
            };
            let r = cui.add(egui::DragValue::new(&mut db).range(-60.0..=12.0).speed(0.25).suffix(" dB").max_decimals(1));
            r.widget_info(|| WidgetInfo::labeled(WidgetType::DragValue, true, format!("{} volume", track.name)));
            if r.changed() {
                app.tl.track_volume = Some((track.id, db));
            }
            if (r.drag_stopped() || r.lost_focus())
                && let Some((id, v)) = app.tl.track_volume.take()
                && id == track.id
                && (v - track.volume_db).abs() > f64::EPSILON
            {
                app.queue(Action::SetTrackVolume(id, v));
            }
        }
    }
}

// ── mouse ──────────────────────────────────────────────────────────────────

/// What a clip's neighbours tell us: edges worth snapping to, and spans (lane, start, end) to stay
/// clear of.
fn others(seq: &SequenceView, group: &[ClipId]) -> (Vec<Tick>, Vec<(usize, Tick, Tick)>) {
    let mut points = vec![Tick::ZERO, seq.playhead];
    let mut spans = Vec::new();
    for (i, t) in seq.tracks.iter().enumerate() {
        for c in t.clips.iter().filter(|c| !group.contains(&c.id)) {
            points.push(c.start);
            points.push(c.end());
            spans.push((i, c.start, c.end()));
        }
    }
    (points, spans)
}

/// The clips that move and trim together with `clip` (itself included), with their lane.
fn group_of(seq: &SequenceView, clip: ClipId) -> Vec<(usize, ClipView)> {
    let Some(me) = seq.clip(clip) else { return Vec::new() };
    let mut out = Vec::new();
    for (i, t) in seq.tracks.iter().enumerate() {
        for c in &t.clips {
            if c.id == clip || (me.link.is_some() && c.link == me.link) {
                out.push((i, c.clone()));
            }
        }
    }
    out
}

fn hit_sentence_edge(marks: &[Mark], top_lane: Rect, axis: &Axis, p: Pos2, edge_px: f32) -> Option<(String, Edge)> {
    if p.y < top_lane.top() - 2.0 || p.y > top_lane.bottom() + 2.0 {
        return None;
    }
    let mut best: Option<(f32, String, Edge)> = None;
    for m in marks {
        let Some(ref sid) = m.sentence_id else { continue };
        if m.kind != MarkKind::Cut && m.kind != MarkKind::Kept {
            continue;
        }
        let x_start = axis.x_of(m.start);
        let x_end = axis.x_of(m.end);
        let d_start = (p.x - x_start).abs();
        let d_end = (p.x - x_end).abs();
        if d_start <= edge_px && best.as_ref().is_none_or(|(d, ..)| d_start < *d) {
            best = Some((d_start, sid.clone(), Edge::Start));
        }
        if d_end <= edge_px && best.as_ref().is_none_or(|(d, ..)| d_end < *d) {
            best = Some((d_end, sid.clone(), Edge::End));
        }
    }
    best.map(|(_, sid, edge)| (sid, edge))
}

fn interact(app: &mut SimpleCutApp, ui: &egui::Ui, resp: &egui::Response, seq: &SequenceView, geom: &Geometry) {
    let (axis, clips, ruler, lanes) = (&geom.axis, &geom.clips, geom.ruler, &geom.lanes);
    let mods = ui.input(|i| i.modifiers);
    if app.tl.drag.is_none() {
        let sentence_hover = lanes
            .first()
            .and_then(|tl| resp.hover_pos().filter(|p| !ruler.contains(*p)).and_then(|p| hit_sentence_edge(&app.review_marks(), *tl, axis, p, 5.0)));
        if sentence_hover.is_some() {
            ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
        } else {
            let hover = resp.hover_pos().filter(|p| !ruler.contains(*p)).map_or(Hit::Empty, |p| tl_math::hit(clips, p, EDGE_PX));
            match hover {
                Hit::Edge(..) => ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal),
                Hit::Body(_) => ui.ctx().set_cursor_icon(CursorIcon::Grab),
                Hit::Empty => {}
            }
        }
    }

    if resp.drag_started() {
        let origin = ui.input(|i| i.pointer.press_origin()).or(resp.interact_pointer_pos());
        if let Some(p) = origin {
            let sentence_hit = if ruler.contains(p) { None } else { lanes.first().and_then(|tl| hit_sentence_edge(&app.review_marks(), *tl, axis, p, 5.0)) };
            if let Some((sentence, edge)) = sentence_hit {
                app.tl.drag = Some(Drag::SentenceEdge { sentence, edge });
            } else {
                let h = if ruler.contains(p) { Hit::Empty } else { tl_math::hit(clips, p, EDGE_PX) };
                app.tl.drag = Some(match h {
                    Hit::Edge(id, edge) => match seq.clip(id) {
                        Some(c) => {
                            let orig = if edge == Edge::Start { c.start } else { c.end() };
                            Drag::Trim { clip: id, edge, orig, to: orig, regular: mods.alt }
                        }
                        None => Drag::Scrub,
                    },
                    Hit::Body(id) => match seq.clip(id) {
                        Some(c) => {
                            if !c.selected {
                                app.queue(Action::SelectClips { clips: vec![id], add: false });
                            }
                            Drag::Move { clip: id, orig: c.start, grab: Tick(axis.time_at(p.x).0 - c.start.0), to: c.start }
                        }
                        None => Drag::Scrub,
                    },
                    Hit::Empty => Drag::Scrub,
                });
            }
        }
    }

    if resp.dragged()
        && let Some(p) = resp.interact_pointer_pos()
        && let Some(drag) = app.tl.drag.clone()
    {
        let reach = axis.span_of(SNAP_PX);
        let next = match drag {
            Drag::Scrub => {
                let to = axis.time_at(p.x);
                if to != seq.playhead {
                    app.queue(Action::SeekTimeline(to));
                }
                Drag::Scrub
            }
            Drag::Move { clip, orig, grab, .. } => match seq.clip(clip) {
                Some(me) => {
                    let group = group_of(seq, clip);
                    let ids: Vec<ClipId> = group.iter().map(|(_, c)| c.id).collect();
                    let (points, spans) = others(seq, &ids);
                    let lanes: Vec<usize> = group.iter().map(|(i, _)| *i).collect();
                    let blocked: Vec<(Tick, Tick)> = spans.iter().filter(|(i, ..)| lanes.contains(i)).map(|(_, a, b)| (*a, *b)).collect();
                    let wanted = Tick((axis.time_at(p.x).0 - grab.0).max(0));
                    let snapped = tl_math::snap_span(wanted, me.duration, &points, reach);
                    Drag::Move { clip, orig, grab, to: tl_math::place_without_overlap(snapped, me.duration, &blocked) }
                }
                None => Drag::Scrub,
            },
            Drag::Trim { clip, edge, orig, regular, .. } => match seq.clip(clip) {
                Some(me) => {
                    let ids: Vec<ClipId> = group_of(seq, clip).iter().map(|(_, c)| c.id).collect();
                    let (points, spans) = others(seq, &ids);
                    let frame = tl_math::ticks(1.0 / seq.fps.max(1.0));
                    let source_len = app.media().iter().find(|m| m.id == me.item).map(|m| m.duration);
                    let (mut lo, mut hi) = match edge {
                        Edge::Start => (Tick(me.start.0 - me.source_in.0), Tick(me.end().0 - frame.0)),
                        Edge::End => (Tick(me.start.0 + frame.0), source_len.map_or(Tick(i64::MAX / 4), |l| Tick(me.start.0 + (l.0 - me.source_in.0)))),
                    };
                    if regular {
                        // a plain trim may not run into the next clip
                        let lane = seq.tracks.iter().position(|t| t.clips.iter().any(|c| c.id == clip)).unwrap_or(0);
                        for (i, a, b) in spans.iter().filter(|(i, ..)| *i == lane) {
                            let _ = i;
                            match edge {
                                Edge::Start if *b <= me.start => lo = Tick(lo.0.max(b.0)),
                                Edge::End if *a >= me.end() => hi = Tick(hi.0.min(a.0)),
                                _ => {}
                            }
                        }
                    }
                    let snapped = tl_math::snap(axis.time_at(p.x), &points, reach);
                    let (lo, hi) = (lo.0.min(hi.0), hi.0.max(lo.0));
                    Drag::Trim { clip, edge, orig, to: Tick(snapped.0.clamp(lo, hi)), regular }
                }
                None => Drag::Scrub,
            },
            Drag::SentenceEdge { sentence, edge } => {
                if let Some(item) = app.review_item() {
                    let t = axis.time_at(p.x);
                    if let Some(file_t) = app.editor.file_time_at(item, t) {
                        let new_sec = file_t.0 as f64 / 254_016_000_000.0;
                        if let Some((cur_start, cur_end)) = app.sentence_span_f64(&sentence) {
                            let (s, e) = match edge {
                                Edge::Start => (new_sec, cur_end),
                                Edge::End => (cur_start, new_sec),
                            };
                            let changed = match edge {
                                Edge::Start => (new_sec - cur_start).abs() > 1e-4,
                                Edge::End => (new_sec - cur_end).abs() > 1e-4,
                            };
                            if changed {
                                app.queue(Action::SetSentenceSpan { sentence: sentence.clone(), start: s, end: e });
                            }
                        }
                        resp.show_tooltip_text(format!("{:.2} s", new_sec));
                    }
                }
                Drag::SentenceEdge { sentence, edge }
            }
        };
        app.tl.drag = Some(next);
    }

    if resp.drag_stopped() {
        match app.tl.drag.take() {
            Some(Drag::Move { clip, orig, to, .. }) if to != orig => app.queue(Action::MoveClip { clip, to }),
            Some(Drag::Trim { clip, edge, orig, to, regular }) if to != orig => {
                app.queue(Action::TrimClip { clip, edge, delta: Tick(to.0 - orig.0), ripple: !regular })
            }
            _ => {}
        }
    }

    if resp.clicked()
        && let Some(p) = resp.interact_pointer_pos()
    {
        if ruler.contains(p) {
            app.queue(Action::SeekTimeline(axis.time_at(p.x)));
            return;
        }
        match tl_math::hit(clips, p, EDGE_PX) {
            Hit::Edge(id, _) | Hit::Body(id) => app.queue(Action::SelectClips { clips: vec![id], add: mods.command || mods.shift }),
            Hit::Empty => {
                if !(mods.command || mods.shift) {
                    app.queue(Action::SelectClips { clips: Vec::new(), add: false });
                }
                app.queue(Action::SeekTimeline(axis.time_at(p.x)));
            }
        }
    }
}

/// Where the retake groups are: a band over each group's span, filled where a sentence will be
/// cut, outlined where it is kept.
fn paint_review(app: &SimpleCutApp, painter: &egui::Painter, axis: &Axis, lanes: &[Rect]) {
    let marks = app.review_marks();
    if marks.is_empty() {
        return;
    }
    let (Some(top), Some(bottom)) = (lanes.first(), lanes.get(1)) else { return };
    for m in marks {
        let color = theme::GROUP_COLORS[m.color % theme::GROUP_COLORS.len()];
        let r = Rect::from_min_max(pos2(axis.x_of(m.start), top.top()), pos2(axis.x_of(m.end).max(axis.x_of(m.start) + 2.0), bottom.bottom()));
        let radius = CornerRadius::same(3);
        match m.kind {
            MarkKind::Band { likely } => {
                painter.rect_filled(r, radius, color.gamma_multiply(if likely { 0.22 } else { 0.1 }));
            }
            MarkKind::Cut => {
                painter.rect_filled(r, radius, color.gamma_multiply(app.highlight_strength()));
                painter.rect_stroke(r, radius, Stroke::new(2.0, color), StrokeKind::Inside);
            }
            MarkKind::Kept => {
                painter.rect_stroke(r, radius, Stroke::new(1.5, color), StrokeKind::Inside);
            }
        }
    }
}

/// The ghost of a drag in progress, and the snap guide.
fn paint_drag(app: &SimpleCutApp, painter: &egui::Painter, seq: &SequenceView, axis: &Axis, lanes: &[Rect], clips: &[ClipGeom], lanes_rect: Rect) {
    let t = theme::tokens();
    match &app.tl.drag {
        Some(Drag::Move { clip, orig, to, .. }) if to != orig => {
            let delta = Tick(to.0 - orig.0);
            for (i, c) in group_of(seq, *clip) {
                let Some(lane) = lanes.get(i) else { continue };
                let r = Rect::from_min_max(pos2(axis.x_of(c.start + delta), lane.top() + 2.0), pos2(axis.x_of(c.end() + delta), lane.bottom() - 2.0));
                painter.rect_filled(r, CornerRadius::same(4), Color32::WHITE.gamma_multiply(0.18));
                painter.rect_stroke(r, CornerRadius::same(4), Stroke::new(2.0, t.accent), StrokeKind::Inside);
            }
            let x = axis.x_of(*to);
            painter.line_segment([pos2(x, lanes_rect.top()), pos2(x, lanes_rect.bottom())], Stroke::new(1.0, t.accent.gamma_multiply(0.7)));
        }
        Some(Drag::Trim { clip, orig, to, .. }) if to != orig => {
            if let Some(g) = clips.iter().find(|g| g.id == *clip) {
                let (a, b) = (axis.x_of(*orig), axis.x_of(*to));
                let r = Rect::from_min_max(pos2(a.min(b), g.rect.top()), pos2(a.max(b), g.rect.bottom()));
                painter.rect_filled(r, CornerRadius::ZERO, t.accent.gamma_multiply(0.35));
                painter.line_segment([pos2(b, lanes_rect.top()), pos2(b, lanes_rect.bottom())], Stroke::new(1.5, t.accent));
            }
        }
        _ => {}
    }
}

// ── zoom buttons ───────────────────────────────────────────────────────────

/// Zoom by `factor` about the playhead if it is in view, else the middle of the view.
pub(crate) fn zoom_by(app: &mut SimpleCutApp, factor: f64) {
    let Some(g) = app.tl.last.clone() else { return };
    let seq = app.editor.sequence();
    let head_x = seq.as_ref().map(|s| g.axis.x_of(s.playhead)).filter(|x| *x >= g.axis.left && *x <= g.axis.left + g.lane_w);
    let anchor = head_x.unwrap_or(g.axis.left + g.lane_w / 2.0);
    let duration = seq.map(|s| s.duration).unwrap_or_default();
    let z = tl_math::zoom_about(g.axis, factor, anchor, g.lane_w, duration);
    app.tl.zoom = Some(z.pps);
    app.tl.scroll = z.scroll;
}

pub(crate) fn zoom_fit(app: &mut SimpleCutApp) {
    app.tl.zoom = None;
    app.tl.scroll = 0.0;
}
