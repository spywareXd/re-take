//! Cut modes: how the sentences the user removes become the ranges that are cut.
//!
//! Without `refine` a removed sentence is cut at its own start and end. A service that offers
//! `refine` (`POST /v1/refine`) looks at the audio and the words and answers with exact ranges in
//! file time, in one of three modes. The ranges come from outside, so [`plan_cuts`] checks every one
//! and never lets a bad one change the video: a range that fails is replaced by the whole sentences
//! it touches, and the user is told.

use filmcraft_time::Tick;
use serde::Deserialize;

use crate::retake::{CutProposal, END_TOLERANCE_SECS, MAX_DOCUMENT_BYTES, MIN_CUT_SECS};
use crate::review::Sentence;

const TICKS_PER_SECOND: f64 = 254_016_000_000.0;
/// How far outside its sentences a range may reach in `anchored` and `retake_only` (dead air around
/// the removal, never another sentence's worth).
pub const WIDEN_SECS: f64 = 1.0;
const MAX_RANGES: usize = 10_000;

/// How the removals become cuts. The retake detection is the same in all three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CutMode {
    /// Keep only spoken sentences: everything without speech goes, silent gameplay included.
    Sentences,
    /// Remove the retakes and the silence or noise around them. Recommended.
    #[default]
    Anchored,
    /// Remove only the retake itself. The least intrusive.
    RetakeOnly,
}

impl CutMode {
    pub const ALL: [CutMode; 3] = [CutMode::Anchored, CutMode::RetakeOnly, CutMode::Sentences];

    /// The value the service expects in `mode`.
    pub fn as_str(self) -> &'static str {
        match self {
            CutMode::Sentences => "sentences",
            CutMode::Anchored => "anchored",
            CutMode::RetakeOnly => "retake_only",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            CutMode::Sentences => "Speech only",
            CutMode::Anchored => "Retakes and dead air",
            CutMode::RetakeOnly => "Retakes only",
        }
    }

    /// One line for the interface.
    pub fn description(self) -> &'static str {
        match self {
            CutMode::Sentences => "Keeps only spoken sentences. Anything without speech is removed, including silent gameplay.",
            CutMode::Anchored => "Removes retakes and the silence or noise around them. Recommended.",
            CutMode::RetakeOnly => "Removes only the retake itself. Least intrusive.",
        }
    }

    /// Shown when the mode is chosen, if it can remove more than the retakes.
    pub fn warning(self) -> Option<&'static str> {
        match self {
            CutMode::Sentences => Some("Parts of the video with no speech (silent gameplay, screen action, pauses) will be removed."),
            _ => None,
        }
    }
}

/// What to send to `refine`: only the removed sentences whose spans the user has not adjusted by
/// hand (those are cut exactly as set, so the service must not see them as removed).
pub fn ids_to_refine(removed: &[Sentence], edited: &[String]) -> Vec<String> {
    removed.iter().filter(|s| !edited.contains(&s.id)).map(|s| s.id.clone()).collect()
}

/// The cuts to make, and what the user should be told about them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RefinedCuts {
    pub cuts: Vec<CutProposal>,
    pub notes: Vec<String>,
}

#[derive(Deserialize)]
struct RawReply {
    ranges: Vec<RawRange>,
}

#[derive(Deserialize)]
struct RawRange {
    start: f64,
    end: f64,
}

fn secs(t: Tick) -> f64 {
    t.0 as f64 / TICKS_PER_SECOND
}

/// Turn the removed sentences into cuts.
///
/// * `reply` is the service's answer (`{"ranges": [{"start","end"}]}`), or None when there was no
///   refinement: every removed sentence is then cut at its own span.
/// * `edited` are the sentences whose span the user changed by hand: always cut exactly as set.
/// * A range must be finite, inside the file, at least [`MIN_CUT_SECS`] long and after the one
///   before it. In `Anchored` and `RetakeOnly` it must also lie within a removed sentence's span
///   widened by [`WIDEN_SECS`] (neighbouring widened spans count as one). `Sentences` mode is
///   allowed to remove stretches without speech anywhere, so that check is skipped.
/// * A range that fails is replaced by the whole sentences it touches; a removed sentence that no
///   valid range touches is cut whole. Either way a note says so.
pub fn plan_cuts(mode: CutMode, removed: &[Sentence], edited: &[String], reply: Option<&str>, media_duration: Tick) -> RefinedCuts {
    let (hand, plain): (Vec<&Sentence>, Vec<&Sentence>) = removed.iter().partition(|s| edited.contains(&s.id));
    let mut out = RefinedCuts { cuts: hand.iter().map(|s| s.as_cut()).collect(), notes: Vec::new() };
    let Some(reply) = reply else {
        out.cuts.extend(plain.iter().map(|s| s.as_cut()));
        return out;
    };
    if !hand.is_empty() {
        out.notes.push(format!(
            "{} sentence{} you adjusted by hand {} cut exactly as you set {}.",
            hand.len(),
            plural(hand.len()),
            if hand.len() == 1 { "is" } else { "are" },
            if hand.len() == 1 { "it" } else { "them" }
        ));
    }
    let ranges = match parse(reply) {
        Ok(r) => r,
        Err(why) => {
            out.notes.push(format!("The cut ranges from the service could not be used ({why}), so whole sentences are cut instead."));
            out.cuts.extend(plain.iter().map(|s| s.as_cut()));
            return out;
        }
    };

    let duration = secs(media_duration);
    let allowed = widened(&plain);
    let mut covered = vec![false; plain.len()];
    let mut whole: Vec<bool> = vec![false; plain.len()];
    let (mut bad, mut previous_end) = (0usize, 0.0f64);
    let mut valid_cuts: Vec<CutProposal> = Vec::new();
    for r in &ranges {
        let verdict = check(r, previous_end, duration, mode, &allowed);
        if r.end.is_finite() {
            previous_end = previous_end.max(r.end);
        }
        match verdict {
            Ok((start, end)) => {
                let touched: Vec<usize> = overlapping(&plain, start, end);
                for i in &touched {
                    if let Some(c) = covered.get_mut(*i) {
                        *c = true;
                    }
                }
                let text = touched.iter().filter_map(|i| plain.get(*i)).map(|s| s.text.as_str()).collect::<Vec<_>>().join(" / ");
                valid_cuts.push(CutProposal {
                    id: format!("refined:{}", valid_cuts.len() + 1),
                    start: Tick::from_seconds_f64(start),
                    end: Tick::from_seconds_f64(end),
                    confidence: 1.0,
                    decision: crate::retake::Decision::Remove,
                    reason: format!("{} cut", mode.as_str()),
                    removed_text: text,
                    kept_text: String::new(),
                    kept: None,
                    source: Some("refine".into()),
                    group: None,
                });
            }
            Err(()) => {
                bad += 1;
                if r.start.is_finite() && r.end.is_finite() {
                    for i in overlapping(&plain, r.start, r.end) {
                        if let Some(w) = whole.get_mut(i) {
                            *w = true;
                        }
                    }
                }
            }
        }
    }
    // a sentence no valid range touches is cut whole: the user asked for it to go
    let mut uncovered = 0usize;
    for ((s, covered), by_bad_range) in plain.iter().zip(&covered).zip(&whole) {
        if !covered {
            out.cuts.push(s.as_cut());
            if !by_bad_range {
                uncovered += 1;
            }
        }
    }
    out.cuts.extend(valid_cuts);
    if bad > 0 {
        out.notes.push(format!(
            "{bad} cut range{} from the service could not be used, so the whole sentence{} {} cut instead.",
            plural(bad),
            plural(bad),
            if bad == 1 { "is" } else { "are" }
        ));
    }
    if uncovered > 0 {
        out.notes.push(format!(
            "{uncovered} removed sentence{} had no cut range from the service, so {} cut whole.",
            plural(uncovered),
            if uncovered == 1 { "it is" } else { "they are" }
        ));
    }
    out
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn parse(reply: &str) -> Result<Vec<RawRange>, String> {
    if reply.len() > MAX_DOCUMENT_BYTES {
        return Err("the answer is too large".into());
    }
    let raw: RawReply = serde_json::from_str(reply).map_err(|_| "it is not in the expected form".to_string())?;
    if raw.ranges.len() > MAX_RANGES {
        return Err("there are too many ranges".into());
    }
    Ok(raw.ranges)
}

/// The removed sentences' spans, each widened by [`WIDEN_SECS`], neighbours merged.
fn widened(plain: &[&Sentence]) -> Vec<(f64, f64)> {
    let mut spans: Vec<(f64, f64)> = plain.iter().map(|s| (secs(s.start) - WIDEN_SECS, secs(s.end) + WIDEN_SECS)).collect();
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (a, b) in spans {
        match merged.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    merged
}

fn overlapping(plain: &[&Sentence], start: f64, end: f64) -> Vec<usize> {
    plain.iter().enumerate().filter(|(_, s)| secs(s.start) < end && secs(s.end) > start).map(|(i, _)| i).collect()
}

/// The range as it will be cut (its end clamped to the file), or Err when it is not usable.
fn check(r: &RawRange, previous_end: f64, duration: f64, mode: CutMode, allowed: &[(f64, f64)]) -> Result<(f64, f64), ()> {
    if !r.start.is_finite() || !r.end.is_finite() || r.start < 0.0 || r.end > duration + END_TOLERANCE_SECS {
        return Err(());
    }
    let (start, end) = (r.start, r.end.min(duration));
    if end - start < MIN_CUT_SECS || start < previous_end - 1e-9 {
        return Err(());
    }
    if mode != CutMode::Sentences && !allowed.iter().any(|(a, b)| start >= a - 1e-6 && end <= b + 1e-6) {
        return Err(());
    }
    Ok((start, end))
}
