//! Timeline arithmetic with no drawing in it: time to pixel and back, zoom, snapping, and keeping a
//! dragged clip from landing on top of another. Pure functions, so they are tested on their own.

use egui::{Pos2, Rect};
use simplecut_core::{ClipId, Edge, Tick};

const TICKS_PER_SECOND: f64 = 254_016_000_000.0;

/// The timeline always has room for at least this many seconds.
const MIN_SPAN_SECS: f64 = 10.0;
/// Extra room past the end, as a fraction of the length.
const ROOM: f64 = 1.04;
pub const MIN_PPS: f64 = 2.0;
pub const MAX_PPS: f64 = 600.0;

pub fn secs(t: Tick) -> f64 {
    t.0 as f64 / TICKS_PER_SECOND
}

pub fn ticks(s: f64) -> Tick {
    Tick::from_seconds_f64(if s.is_finite() { s } else { 0.0 })
}

/// How time maps onto the screen: `left` is the x of `scroll` seconds, `pps` pixels per second.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axis {
    pub left: f32,
    pub pps: f64,
    pub scroll: f64,
}

impl Axis {
    pub fn x_of(&self, t: Tick) -> f32 {
        self.left + ((secs(t) - self.scroll) * self.pps) as f32
    }

    /// The time under pixel `x`, never before the start.
    pub fn time_at(&self, x: f32) -> Tick {
        ticks(((f64::from(x - self.left)) / self.pps + self.scroll).max(0.0))
    }

    /// How long `px` pixels are.
    pub fn span_of(&self, px: f32) -> Tick {
        ticks(f64::from(px) / self.pps)
    }
}

fn total_secs(duration: Tick) -> f64 {
    secs(duration).max(MIN_SPAN_SECS) * ROOM
}

/// The scale at which the whole timeline fits in `lane_w` pixels.
pub fn fit_pps(lane_w: f32, duration: Tick) -> f64 {
    (f64::from(lane_w.max(1.0)) / total_secs(duration)).clamp(MIN_PPS, MAX_PPS)
}

/// Keep the view inside the timeline.
pub fn clamp_scroll(scroll: f64, pps: f64, lane_w: f32, duration: Tick) -> f64 {
    let max = (total_secs(duration) - f64::from(lane_w) / pps).max(0.0);
    scroll.clamp(0.0, max)
}

/// Zoom by `factor` keeping the time under `anchor_x` where it is.
pub fn zoom_about(axis: Axis, factor: f64, anchor_x: f32, lane_w: f32, duration: Tick) -> Axis {
    let factor = if factor.is_finite() && factor > 0.0 { factor } else { 1.0 };
    let anchor = f64::from(anchor_x - axis.left) / axis.pps + axis.scroll;
    let pps = (axis.pps * factor).clamp(MIN_PPS.min(fit_pps(lane_w, duration)), MAX_PPS);
    let scroll = clamp_scroll(anchor - f64::from(anchor_x - axis.left) / pps, pps, lane_w, duration);
    Axis { pps, scroll, ..axis }
}

/// Spacing of the ruler's labels in seconds: the first nice step at least 70 px apart.
pub fn ruler_step(pps: f64) -> f64 {
    [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0].into_iter().find(|s| s * pps >= 70.0).unwrap_or(3600.0)
}

/// `t`, or the candidate within `reach` of it that is nearest.
pub fn snap(t: Tick, candidates: &[Tick], reach: Tick) -> Tick {
    candidates.iter().copied().filter(|c| (c.0 - t.0).abs() <= reach.0).min_by_key(|c| (c.0 - t.0).abs()).unwrap_or(t)
}

/// Where a span starting at `start` and `duration` long should start so that its start or its end
/// lands on a candidate, if either is within `reach`; otherwise `start`.
pub fn snap_span(start: Tick, duration: Tick, candidates: &[Tick], reach: Tick) -> Tick {
    let by_start = snap(start, candidates, reach);
    let by_end = snap(start + duration, candidates, reach) - duration;
    match (by_start != start, by_end != start) {
        (true, true) => {
            if (by_start.0 - start.0).abs() <= (by_end.0 - start.0).abs() {
                by_start
            } else {
                by_end
            }
        }
        (true, false) => by_start,
        (false, true) => by_end,
        (false, false) => start,
    }
}

/// Where a clip of `duration` wanted at `desired` can sit without covering any of the `others`
/// (start, end): `desired` if that is free, else the free spot nearest to it, never before 0.
pub fn place_without_overlap(desired: Tick, duration: Tick, others: &[(Tick, Tick)]) -> Tick {
    let desired = Tick(desired.0.max(0));
    let free = |s: Tick| s.0 >= 0 && others.iter().all(|(a, b)| s + duration <= *a || s >= *b);
    if free(desired) {
        return desired;
    }
    let mut spots = vec![Tick::ZERO];
    for (a, b) in others {
        spots.push(*b);
        spots.push(*a - duration);
    }
    spots.into_iter().filter(|s| free(*s)).min_by_key(|s| (s.0 - desired.0).abs()).unwrap_or(desired)
}

/// Where a clip is drawn.
#[derive(Clone, Copy, Debug)]
pub struct ClipGeom {
    pub id: ClipId,
    pub rect: Rect,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hit {
    /// The grab handle at one end of a clip.
    Edge(ClipId, Edge),
    Body(ClipId),
    Empty,
}

/// What is under `pos`. Later clips are on top. A clip too narrow to have separate handles has
/// only a body.
pub fn hit(geoms: &[ClipGeom], pos: Pos2, edge_px: f32) -> Hit {
    for g in geoms.iter().rev() {
        if !g.rect.contains(pos) {
            continue;
        }
        if g.rect.width() >= edge_px * 3.0 {
            if pos.x <= g.rect.left() + edge_px {
                return Hit::Edge(g.id, Edge::Start);
            }
            if pos.x >= g.rect.right() - edge_px {
                return Hit::Edge(g.id, Edge::End);
            }
        }
        return Hit::Body(g.id);
    }
    Hit::Empty
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::pos2;

    fn s(x: f64) -> Tick {
        ticks(x)
    }

    #[test]
    fn time_and_pixels_round_trip() {
        let a = Axis { left: 100.0, pps: 50.0, scroll: 2.0 };
        assert_eq!(a.x_of(s(2.0)), 100.0);
        assert_eq!(a.x_of(s(4.0)), 200.0);
        assert_eq!(a.time_at(200.0), s(4.0));
        assert_eq!(a.time_at(0.0), s(0.0), "left of the lanes reads as the start when scrolled to 0");
        assert_eq!(Axis { scroll: 0.0, ..a }.time_at(10.0), Tick::ZERO, "never negative");
        assert_eq!(a.span_of(100.0), s(2.0));
    }

    #[test]
    fn fit_shows_everything_with_room_to_spare() {
        let pps = fit_pps(1000.0, s(60.0));
        assert!(60.0 * pps < 1000.0 && 60.0 * pps > 900.0, "{pps}");
        assert_eq!(fit_pps(1000.0, Tick::ZERO), 1000.0 / (10.0 * ROOM), "an empty timeline shows ten seconds");
        assert!(fit_pps(0.0, s(60.0)) >= MIN_PPS, "a degenerate width does not divide by zero");
    }

    #[test]
    fn zoom_keeps_the_time_under_the_pointer_still() {
        let a = Axis { left: 100.0, pps: 20.0, scroll: 0.0 };
        let dur = s(120.0);
        let z = zoom_about(a, 4.0, 500.0, 1000.0, dur);
        assert_eq!(z.pps, 80.0);
        let before = a.time_at(500.0);
        let after = z.time_at(500.0);
        assert!((secs(before) - secs(after)).abs() < 1e-6, "{before:?} vs {after:?}");
        // zooming far out is held at the fit scale; zooming far in at the maximum
        assert!(zoom_about(a, 0.0001, 500.0, 1000.0, dur).pps >= MIN_PPS.min(fit_pps(1000.0, dur)));
        assert_eq!(zoom_about(a, 1e9, 500.0, 1000.0, dur).pps, MAX_PPS);
        assert_eq!(zoom_about(a, f64::NAN, 500.0, 1000.0, dur).pps, 20.0, "a bad factor changes nothing");
    }

    #[test]
    fn scrolling_stays_inside_the_timeline() {
        let dur = s(100.0);
        assert_eq!(clamp_scroll(-5.0, 10.0, 500.0, dur), 0.0);
        let max = clamp_scroll(1e9, 10.0, 500.0, dur);
        assert!((max - (100.0 * ROOM - 50.0)).abs() < 1e-9, "{max}");
        assert_eq!(clamp_scroll(5.0, 1.0, 500.0, dur), 0.0, "when everything fits there is nothing to scroll");
    }

    #[test]
    fn snapping_picks_the_nearest_candidate_in_reach_only() {
        let c = [s(1.0), s(2.0), s(10.0)];
        assert_eq!(snap(s(1.04), &c, s(0.1)), s(1.0));
        assert_eq!(snap(s(1.6), &c, s(0.5)), s(2.0));
        assert_eq!(snap(s(5.0), &c, s(0.5)), s(5.0), "nothing close: stays put");
        assert_eq!(snap(s(1.0), &[], s(1.0)), s(1.0));
    }

    #[test]
    fn a_span_snaps_by_whichever_end_is_closer() {
        let c = [s(10.0)];
        // the start is 0.05 from 10; the end (start + 4) is far from it
        assert_eq!(snap_span(s(10.05), s(4.0), &c, s(0.1)), s(10.0));
        // the end is 0.05 from 10, so the start moves to 6
        assert_eq!(snap_span(s(5.95), s(4.0), &c, s(0.1)), s(6.0));
        assert_eq!(snap_span(s(3.0), s(4.0), &c, s(0.1)), s(3.0));
    }

    #[test]
    fn a_dragged_clip_never_lands_on_another() {
        let others = [(s(2.0), s(4.0)), (s(6.0), s(8.0))];
        let d = s(1.5);
        assert_eq!(place_without_overlap(s(4.0), d, &others), s(4.0), "free where it is");
        assert_eq!(place_without_overlap(s(4.2), d, &others), s(4.2));
        // wanting to sit on the first clip: the nearest free spot is just after it
        assert_eq!(place_without_overlap(s(3.0), d, &others), s(4.0));
        // wanting to sit on the first clip's left half: just before it
        assert_eq!(place_without_overlap(s(1.0), d, &others), s(0.5));
        // a gap too small for the clip is skipped
        assert_eq!(place_without_overlap(s(5.0), s(3.0), &others), s(8.0));
        // never before the start
        assert_eq!(place_without_overlap(Tick(-5), d, &others), Tick::ZERO);
        assert_eq!(place_without_overlap(s(7.0), d, &[]), s(7.0));
    }

    #[test]
    fn hits_prefer_edges_then_bodies_and_top_clips() {
        let g = [
            ClipGeom { id: ClipId(1), rect: Rect::from_min_max(pos2(0.0, 0.0), pos2(100.0, 40.0)) },
            ClipGeom { id: ClipId(2), rect: Rect::from_min_max(pos2(200.0, 0.0), pos2(210.0, 40.0)) },
        ];
        assert_eq!(hit(&g, pos2(3.0, 10.0), 6.0), Hit::Edge(ClipId(1), Edge::Start));
        assert_eq!(hit(&g, pos2(97.0, 10.0), 6.0), Hit::Edge(ClipId(1), Edge::End));
        assert_eq!(hit(&g, pos2(50.0, 10.0), 6.0), Hit::Body(ClipId(1)));
        assert_eq!(hit(&g, pos2(150.0, 10.0), 6.0), Hit::Empty);
        assert_eq!(hit(&g, pos2(205.0, 10.0), 6.0), Hit::Body(ClipId(2)), "a narrow clip has no handles");
        assert_eq!(hit(&g, pos2(50.0, 80.0), 6.0), Hit::Empty);
    }

    #[test]
    fn the_ruler_picks_a_readable_step() {
        assert_eq!(ruler_step(100.0), 1.0);
        assert_eq!(ruler_step(10.0), 10.0);
        assert_eq!(ruler_step(2.0), 60.0);
        assert_eq!(ruler_step(0.001), 3600.0);
    }
}
