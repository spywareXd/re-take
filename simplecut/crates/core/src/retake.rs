//! The retake-analysis contract (`simplecut.retakes/1`, see `docs/retake-contract.md`): reading
//! what a retake-detection pipeline found, checking it, and cutting the approved parts out of the
//! timeline.
//!
//! Nothing here knows how the pipeline works. It only knows the document. Times in the document
//! are seconds on the media file's own clock; applying converts them to timeline positions per
//! clip, so trimmed, moved and repeated clips are all handled.

use serde::{Deserialize, Serialize};

use filmcraft_project::TrackId;
use filmcraft_time::{Tick, TimeRange};

use crate::editor::{Editor, ItemId};
use crate::error::{EditorError, Result};

/// The only schema this version reads.
pub const SCHEMA: &str = "simplecut.retakes/1";
/// A document larger than this is refused unread.
pub const MAX_DOCUMENT_BYTES: usize = 64 * 1024 * 1024;
const MAX_PROPOSALS: usize = 100_000;
/// How far the document's media length may differ from the file's before it is "another file".
pub(crate) const DURATION_TOLERANCE_SECS: f64 = 0.5;
/// An `end` this far past the end of the file is clamped instead of rejected.
pub(crate) const END_TOLERANCE_SECS: f64 = 0.05;
/// The shortest cut worth proposing.
pub(crate) const MIN_CUT_SECS: f64 = 0.01;

// ── the document, as written by the pipeline ───────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetakeDocument {
    pub schema: String,
    pub media: DocMedia,
    #[serde(default)]
    pub generator: Option<Generator>,
    #[serde(default)]
    pub transcript: Option<DocTranscript>,
    pub proposals: Vec<RawProposal>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocMedia {
    #[serde(default)]
    pub path: String,
    pub duration_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Generator {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocTranscript {
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub words: Vec<DocWord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocWord {
    pub text: String,
    pub start: f64,
    pub end: f64,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub speaker: Option<String>,
}

/// What the pipeline recommends for a proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Decision {
    /// Cut it (pre-selected in the review list when confidence is high).
    #[default]
    Remove,
    /// Not sure, or could not decide: the user must choose (never pre-selected).
    Review,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawProposal {
    pub id: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f64,
    #[serde(default)]
    pub decision: Decision,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub removed_text: String,
    #[serde(default)]
    pub kept_text: String,
    #[serde(default)]
    pub kept_start: Option<f64>,
    #[serde(default)]
    pub kept_end: Option<f64>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub group: Option<String>,
}

/// Read a document. Fails with a plain message if it is not valid JSON, is too big, or has another
/// schema.
pub fn parse(json: &str) -> Result<RetakeDocument> {
    if json.len() > MAX_DOCUMENT_BYTES {
        return Err(EditorError::Invalid("The retake analysis is too large to read.".into()));
    }
    // look at the schema first, so a newer format gets a clear message instead of a parse error
    let value: serde_json::Value = serde_json::from_str(json).map_err(|e| EditorError::Invalid(format!("The retake analysis is not valid JSON: {e}")))?;
    match value.get("schema").and_then(serde_json::Value::as_str) {
        Some(SCHEMA) => {}
        Some(other) => {
            return Err(EditorError::Invalid(format!("The retake analysis uses the format “{other}”, but this version of SimpleCut reads “{SCHEMA}”.")));
        }
        None => return Err(EditorError::Invalid("The retake analysis has no “schema” field, so it cannot be recognised.".into())),
    }
    serde_json::from_value(value).map_err(|e| EditorError::Invalid(format!("The retake analysis is incomplete or has a wrong value: {e}")))
}

// ── the checked result ─────────────────────────────────────────────────────

/// One candidate cut, checked, with times in the media file's own time.
#[derive(Debug, Clone, PartialEq)]
pub struct CutProposal {
    pub id: String,
    pub start: Tick,
    pub end: Tick,
    /// 0 to 1: how sure the pipeline is that this is a retake.
    pub confidence: f32,
    pub decision: Decision,
    pub reason: String,
    pub removed_text: String,
    pub kept_text: String,
    /// Where the take that stays is, if the pipeline said.
    pub kept: Option<(Tick, Tick)>,
    pub source: Option<String>,
    pub group: Option<String>,
}

/// Whether a proposal is currently on the timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presence {
    On,
    Off,
}

/// A proposal that was dropped, and why.
#[derive(Debug, Clone, PartialEq)]
pub struct Rejected {
    pub id: String,
    pub reason: String,
}

/// A document checked against the media file it is for.
#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    pub item: ItemId,
    pub generator: Option<Generator>,
    /// Sorted by start.
    pub proposals: Vec<CutProposal>,
    pub rejected: Vec<Rejected>,
}

/// The number of distinct colors in the retake group palette.
pub const PALETTE_SIZE: usize = 8;

/// A group of retake proposals (e.g. discarded takes before the same kept take).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    /// The proposal `group`, or a unique key made from its id when it has none (a group of one).
    pub key: String,
    /// Friendly label: "Retake 1", "Retake 2", etc.
    pub label: String,
    /// Position in order modulo [`PALETTE_SIZE`].
    pub color: usize,
    /// Indexes into `Analysis.proposals`, ordered by start.
    pub members: Vec<usize>,
}

/// Group proposals in `analysis` by retake group.
///
/// Proposals sharing a `group` key belong to the same group. A proposal without a `group`
/// is its own group. Groups are ordered by the earliest start of any member.
pub fn groups(analysis: &Analysis) -> Vec<Group> {
    groups_with_offset(analysis, 0)
}

/// Group proposals with numbering and colors starting from `offset`.
pub fn groups_with_offset(analysis: &Analysis, offset: usize) -> Vec<Group> {
    let mut order: Vec<String> = Vec::new();
    let mut map: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();

    for (idx, p) in analysis.proposals.iter().enumerate() {
        let key = p.group.as_ref().filter(|g| !g.trim().is_empty()).cloned().unwrap_or_else(|| p.id.clone());
        if !map.contains_key(&key) {
            order.push(key.clone());
        }
        map.entry(key).or_default().push(idx);
    }

    order.sort_by(|a, b| {
        let min_a = map[a].iter().map(|&idx| analysis.proposals[idx].start).min().unwrap_or(Tick::ZERO);
        let min_b = map[b].iter().map(|&idx| analysis.proposals[idx].start).min().unwrap_or(Tick::ZERO);
        (min_a, a).cmp(&(min_b, b))
    });

    order
        .into_iter()
        .enumerate()
        .map(|(i, key)| {
            let num = offset + i + 1;
            let color = (offset + i) % PALETTE_SIZE;
            let mut members = map.remove(&key).unwrap_or_default();
            members.sort_by_key(|&idx| (analysis.proposals[idx].start, analysis.proposals[idx].end, &analysis.proposals[idx].id));
            Group { key, label: format!("Retake {num}"), color, members }
        })
        .collect()
}

fn secs(t: f64) -> Tick {
    Tick::from_seconds_f64(t)
}

/// Check `doc` against the file it is meant for (`item`, `media_duration` long). Refuses a document
/// made for a different file; drops, with a reason, each proposal that is not usable.
pub fn validate(doc: RetakeDocument, item: ItemId, media_duration: Tick) -> Result<Analysis> {
    let file_secs = media_duration.0 as f64 / 254_016_000_000.0;
    let claimed = doc.media.duration_seconds;
    if !claimed.is_finite() || claimed <= 0.0 {
        return Err(EditorError::Invalid("The retake analysis does not say how long the video is, so it cannot be matched to your file.".into()));
    }
    if (claimed - file_secs).abs() > DURATION_TOLERANCE_SECS {
        return Err(EditorError::Invalid(format!(
            "This retake analysis was made for a different file: it is for a {claimed:.1} s video, but this one is {file_secs:.1} s."
        )));
    }
    if doc.proposals.len() > MAX_PROPOSALS {
        return Err(EditorError::Invalid("The retake analysis has too many proposals to review.".into()));
    }

    let mut proposals: Vec<CutProposal> = Vec::new();
    let mut rejected: Vec<Rejected> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for p in doc.proposals {
        let reject = |why: &str| Rejected { id: p.id.clone(), reason: why.to_string() };
        if p.id.trim().is_empty() {
            rejected.push(reject("it has no id"));
            continue;
        }
        if seen.contains(&p.id) {
            rejected.push(reject("its id is used more than once"));
            continue;
        }
        seen.push(p.id.clone());
        if !p.start.is_finite() || !p.end.is_finite() {
            rejected.push(reject("its start or end is not a number"));
            continue;
        }
        if p.start < 0.0 {
            rejected.push(reject("it starts before the beginning of the file"));
            continue;
        }
        if p.end <= p.start {
            rejected.push(reject("it ends before it starts"));
            continue;
        }
        if p.end > file_secs + END_TOLERANCE_SECS {
            rejected.push(reject("it ends after the end of the file"));
            continue;
        }
        let end = p.end.min(file_secs);
        if end - p.start < MIN_CUT_SECS {
            rejected.push(reject("it is shorter than 10 ms"));
            continue;
        }
        if !p.confidence.is_finite() || !(0.0..=1.0).contains(&p.confidence) {
            rejected.push(reject("its confidence is not a number from 0 to 1"));
            continue;
        }
        let kept = match (p.kept_start, p.kept_end) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() && a >= 0.0 && b > a => Some((secs(a), secs(b))),
            _ => None,
        };
        proposals.push(CutProposal {
            id: p.id,
            start: secs(p.start),
            end: secs(end),
            confidence: p.confidence as f32,
            decision: p.decision,
            reason: p.reason,
            removed_text: p.removed_text,
            kept_text: p.kept_text,
            kept,
            source: p.source,
            group: p.group,
        });
    }
    proposals.sort_by_key(|p| (p.start, p.end));
    Ok(Analysis { item, generator: doc.generator, proposals, rejected })
}

// ── applying ───────────────────────────────────────────────────────────────

/// Which tracks the cuts are made on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CutScope {
    /// Video and Dialogue: the picture and its own sound. Music is left alone.
    #[default]
    MainTracks,
    /// Every unlocked track, music included.
    AllTracks,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ApplyReport {
    /// How many separate pieces were cut out.
    pub cuts: usize,
    /// How much shorter the timeline is.
    pub removed: Tick,
    /// Things that were left alone, in words.
    pub notes: Vec<String>,
}

/// Merge overlapping and touching ranges.
fn merge(mut ranges: Vec<(Tick, Tick)>) -> Vec<(Tick, Tick)> {
    ranges.sort();
    let mut out: Vec<(Tick, Tick)> = Vec::new();
    for (a, b) in ranges {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

impl Editor {
    /// Check a document against the media file `item` of this project.
    pub fn read_analysis(&self, json: &str, item: ItemId) -> Result<Analysis> {
        let entry = self.media_entry(item).ok_or(EditorError::NoSuchMedia)?;
        validate(parse(json)?, item, entry.duration)
    }

    /// Which file of this project a document is for: the one whose length matches, and if several
    /// do, the one with the same file name.
    pub fn find_media_for(&self, doc: &RetakeDocument) -> Result<ItemId> {
        self.find_media_by(&doc.media.path, doc.media.duration_seconds)
    }

    /// Which file of this project a document written for `path`, `want` seconds long is for.
    pub fn find_media_by(&self, path: &str, want: f64) -> Result<ItemId> {
        let near: Vec<_> = self.media().into_iter().filter(|m| (m.duration.0 as f64 / 254_016_000_000.0 - want).abs() <= DURATION_TOLERANCE_SECS).collect();
        let file_name = path.rsplit(['/', '\\']).next().unwrap_or("").to_lowercase();
        match near.len() {
            0 => Err(EditorError::Invalid(format!("None of the files in this project matches that analysis (it is for a {want:.1} s video)."))),
            1 => Ok(near[0].id),
            _ => {
                let named: Vec<_> = near.iter().filter(|m| m.name.to_lowercase() == file_name).collect();
                match named.as_slice() {
                    [one] => Ok(one.id),
                    _ => Err(EditorError::Invalid("More than one file in this project could be the one that analysis is for.".into())),
                }
            }
        }
    }

    /// Where the part of file `item` that `p` covers is on the timeline: one range per clip of the
    /// main video track that shows it. Not rounded to frames; for drawing and for "go to".
    pub fn timeline_ranges(&self, item: ItemId, p: &CutProposal) -> Vec<(Tick, Tick)> {
        self.timeline_ranges_of(item, p.start, p.end)
    }

    /// [`Self::timeline_ranges`] for any span of the file (a sentence, a whole group).
    pub fn timeline_ranges_of(&self, item: ItemId, start: Tick, end: Tick) -> Vec<(Tick, Tick)> {
        let Some(q) = self.session().active_sequence() else { return Vec::new() };
        let Some(track) = q.video_tracks.first() else { return Vec::new() };
        let mut out = Vec::new();
        for clip in track.items.iter().filter(|c| c.item == item && (c.speed - 1.0).abs() <= f64::EPSILON && c.frame_hold.is_none()) {
            let (src_lo, src_hi) = (clip.source_in, clip.source_in + clip.duration);
            let (lo, hi) = (start.max(src_lo), end.min(src_hi));
            if lo < hi {
                out.push((clip.start + (lo - src_lo), clip.start + (hi - src_lo)));
            }
        }
        out
    }

    /// Inverse of [`Self::timeline_ranges_of`]: the media file time shown at `timeline_tick` by a clip
    /// on the main video track for `item`, honouring clip in-point and speed. Returns `None` if
    /// `timeline_tick` falls in a gap or outside any clip of `item`.
    pub fn file_time_at(&self, item: ItemId, timeline_tick: Tick) -> Option<Tick> {
        let q = self.session().active_sequence()?;
        let track = q.video_tracks.first()?;
        let matching = track
            .items
            .iter()
            .filter(|c| c.item == item && c.duration > Tick::ZERO)
            .find(|c| c.start <= timeline_tick && timeline_tick < c.start + c.duration)
            .or_else(|| track.items.iter().filter(|c| c.item == item && c.duration > Tick::ZERO).find(|c| timeline_tick == c.start + c.duration))?;

        if let Some(held) = matching.frame_hold {
            return Some(held);
        }

        let offset_timeline = timeline_tick.0.checked_sub(matching.start.0)?;
        if !matching.speed.is_finite() || matching.speed <= 0.0 {
            return None;
        }
        let offset_source =
            if (matching.speed - 1.0).abs() <= f64::EPSILON { offset_timeline } else { (offset_timeline as f64 * matching.speed).round() as i64 };
        let file_tick = matching.source_in.0.checked_add(offset_source)?;
        Some(Tick(file_tick))
    }

    /// Whether proposal `p` (for media file `item`) is currently on the timeline.
    ///
    /// Returns [`Presence::Off`] if the total covered length on the main video track is less
    /// than one frame of the sequence or less than 10 % of the proposal's own duration (preventing
    /// rounding slivers from showing as suggestions). Clips with changed speed or frame hold have
    /// no mapped ranges and are therefore `Off`.
    pub fn presence(&self, item: ItemId, p: &CutProposal) -> Presence {
        self.presence_of(item, p.start, p.end)
    }

    /// [`Self::presence`] for any span of the file. A span shorter than a frame needs half of itself
    /// on the timeline (so a very short sentence is not "off" just because it is under a frame).
    pub fn presence_of(&self, item: ItemId, start: Tick, end: Tick) -> Presence {
        let ranges = self.timeline_ranges_of(item, start, end);
        let covered: Tick = ranges.iter().fold(Tick::ZERO, |acc, (s, e)| acc + (*e - *s));
        let frame_duration =
            self.session().active_sequence().map(|q| q.settings.frame_rate.frame_duration()).unwrap_or_else(|| Tick::from_seconds_f64(1.0 / 30.0));
        let len = if end > start { Tick(end.0 - start.0) } else { Tick::ZERO };
        let tenth = Tick(len.0 / 10);
        let min_threshold = tenth.max(frame_duration.min(Tick(len.0 / 2)));
        if covered < min_threshold || covered <= Tick::ZERO { Presence::Off } else { Presence::On }
    }

    /// Whether proposal `p` (for media file `item`) falls completely outside the used source range
    /// of normal clips on the main video track.
    pub fn is_outside_used(&self, item: ItemId, p: &CutProposal) -> bool {
        self.span_outside_used(item, p.start, p.end)
    }

    /// [`Self::is_outside_used`] for any span of the file.
    pub fn span_outside_used(&self, item: ItemId, start: Tick, end: Tick) -> bool {
        let Some(q) = self.session().active_sequence() else { return true };
        let Some(track) = q.video_tracks.first() else { return true };
        let mut min_src: Option<Tick> = None;
        let mut max_src: Option<Tick> = None;
        for clip in track.items.iter().filter(|c| c.item == item && (c.speed - 1.0).abs() <= f64::EPSILON && c.frame_hold.is_none()) {
            let lo = clip.source_in;
            let hi = clip.source_in + clip.duration;
            min_src = Some(min_src.map_or(lo, |m| m.min(lo)));
            max_src = Some(max_src.map_or(hi, |m| m.max(hi)));
        }
        match (min_src, max_src) {
            (Some(lo), Some(hi)) => end <= lo || start >= hi,
            _ => true,
        }
    }

    /// Cut `proposals` (all for media file `item`) out of the timeline: see [`Editor::apply_cut_sets`].
    pub fn apply_cuts(&mut self, item: ItemId, proposals: &[CutProposal], scope: CutScope) -> Result<ApplyReport> {
        self.apply_cut_sets(&[(item, proposals.to_vec())], scope)
    }

    /// Cut the proposals of one or more media files out of the timeline as one undo step: picture
    /// and sound together, gaps closed. Each proposal is converted from the file's time to timeline
    /// time through every clip that shows that part of the file, so trimmed, moved or repeated
    /// clips are right. Overlapping cuts are merged and the result is rounded to whole frames.
    pub fn apply_cut_sets(&mut self, sets: &[(ItemId, Vec<CutProposal>)], scope: CutScope) -> Result<ApplyReport> {
        if sets.iter().all(|(_, p)| p.is_empty()) {
            return Err(EditorError::NothingSelected);
        }
        let mut notes = Vec::new();
        let (tracks, ranges, rate) = {
            let q = self.session().active_sequence().ok_or(EditorError::NoSequence)?;
            let main: Vec<TrackId> = q.video_tracks.first().into_iter().chain(q.audio_tracks.first()).map(|t| t.id).collect();
            let targets: Vec<TrackId> = match scope {
                CutScope::MainTracks => main,
                CutScope::AllTracks => q.video_tracks.iter().chain(q.audio_tracks.iter()).filter(|t| !t.locked).map(|t| t.id).collect(),
            };
            let mut ranges: Vec<(Tick, Tick)> = Vec::new();
            for track in q.video_tracks.iter().chain(q.audio_tracks.iter()).filter(|t| targets.contains(&t.id)) {
                for (item, proposals) in sets {
                    for clip in track.items.iter().filter(|c| c.item == *item) {
                        if (clip.speed - 1.0).abs() > f64::EPSILON || clip.frame_hold.is_some() {
                            let note = format!("“{}” has its speed changed, so it was left as it is.", clip.name);
                            if !notes.contains(&note) {
                                notes.push(note);
                            }
                            continue;
                        }
                        let (src_lo, src_hi) = (clip.source_in, clip.source_in + clip.duration);
                        for p in proposals {
                            let (lo, hi) = (p.start.max(src_lo), p.end.min(src_hi));
                            if lo < hi {
                                ranges.push((clip.start + (lo - src_lo), clip.start + (hi - src_lo)));
                            }
                        }
                    }
                }
            }
            (targets, ranges, q.settings.frame_rate)
        };
        // whole frames, then merge again: two cuts a hair apart may now touch
        let snapped: Vec<(Tick, Tick)> = merge(ranges).into_iter().map(|(a, b)| (rate.snap(a), rate.snap(b))).filter(|(a, b)| b > a).collect();
        let cuts = merge(snapped);
        if cuts.is_empty() {
            return Err(EditorError::NotAvailable("None of those cuts falls inside the part of the file that is on the timeline.".into()));
        }
        let removed = self.session_mut().edit_sequence("Remove retakes", |seq, ctx, _| {
            let mut total = Tick::ZERO;
            // last first, so the earlier ranges keep their positions
            for (a, b) in cuts.iter().rev() {
                filmcraft_edit::extract(seq, &tracks, TimeRange::from_bounds(*a, *b), ctx);
                total += *b - *a;
            }
            Ok(total)
        })?;
        Ok(ApplyReport { cuts: cuts.len(), removed, notes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = include_str!("../../../docs/examples/retake-sample.json");

    fn t(s: f64) -> Tick {
        secs(s)
    }

    #[test]
    fn the_documented_sample_is_valid() {
        let doc = parse(SAMPLE).unwrap();
        assert_eq!(doc.proposals.len(), 3);
        let a = validate(doc, ItemId(1), t(60.0)).unwrap();
        assert!(a.rejected.is_empty(), "{:?}", a.rejected);
        assert_eq!(a.proposals.len(), 3);
        assert_eq!(a.proposals[0].id, "span_0:0");
        assert_eq!(a.proposals[1].decision, Decision::Review);
        assert_eq!(a.proposals[2].decision, Decision::Remove, "decision defaults to remove");
        assert_eq!(a.proposals[2].reason, "", "optional text defaults to empty");
        assert_eq!(a.proposals[0].kept, Some((t(15.8), t(19.2))));
        assert_eq!(a.generator.unwrap().name, "retake");
    }

    #[test]
    fn parsing_refuses_what_it_cannot_trust() {
        assert!(matches!(parse("not json"), Err(EditorError::Invalid(m)) if m.contains("not valid JSON")));
        assert!(matches!(parse("{}"), Err(EditorError::Invalid(m)) if m.contains("no “schema”")));
        let other = r#"{"schema":"simplecut.retakes/2","media":{"duration_seconds":1},"proposals":[]}"#;
        assert!(matches!(parse(other), Err(EditorError::Invalid(m)) if m.contains("retakes/2") && m.contains("retakes/1")));
        let missing = r#"{"schema":"simplecut.retakes/1","proposals":[]}"#;
        assert!(matches!(parse(missing), Err(EditorError::Invalid(m)) if m.contains("incomplete")));
        let text_time = r#"{"schema":"simplecut.retakes/1","media":{"duration_seconds":10},"proposals":[{"id":"a","start":"1","end":2,"confidence":0.5}]}"#;
        assert!(parse(text_time).is_err(), "times must be numbers");
        // extra fields from a newer pipeline are fine
        let extra = r#"{"schema":"simplecut.retakes/1","media":{"duration_seconds":10,"fps":29.97},"proposals":[],"future":true}"#;
        assert!(parse(extra).is_ok());
        let huge = " ".repeat(MAX_DOCUMENT_BYTES + 1);
        assert!(matches!(parse(&huge), Err(EditorError::Invalid(m)) if m.contains("too large")));
    }

    fn doc(duration: f64, proposals: &str) -> RetakeDocument {
        parse(&format!(r#"{{"schema":"simplecut.retakes/1","media":{{"duration_seconds":{duration}}},"proposals":[{proposals}]}}"#)).unwrap()
    }

    #[test]
    fn a_document_for_another_file_is_refused_whole() {
        let d = doc(100.0, "");
        let e = validate(d, ItemId(1), t(60.0)).unwrap_err();
        assert!(matches!(&e, EditorError::Invalid(m) if m.contains("different file") && m.contains("100.0") && m.contains("60.0")), "{e}");
        assert!(validate(doc(60.3, ""), ItemId(1), t(60.0)).is_ok(), "a small difference is rounding, not another file");
        assert!(validate(doc(0.0, ""), ItemId(1), t(60.0)).is_err());
        let nan = RetakeDocument { media: DocMedia { path: String::new(), duration_seconds: f64::NAN }, ..doc(60.0, "") };
        assert!(validate(nan, ItemId(1), t(60.0)).is_err());
    }

    #[test]
    fn unusable_proposals_are_dropped_one_by_one_with_the_reason() {
        let d = doc(
            60.0,
            r#"
            {"id":"ok","start":1,"end":2,"confidence":0.9},
            {"id":"","start":1,"end":2,"confidence":0.9},
            {"id":"ok","start":3,"end":4,"confidence":0.9},
            {"id":"rev","start":5,"end":4,"confidence":0.9},
            {"id":"neg","start":-1,"end":4,"confidence":0.9},
            {"id":"late","start":59,"end":70,"confidence":0.9},
            {"id":"tiny","start":5,"end":5.005,"confidence":0.9},
            {"id":"conf","start":6,"end":7,"confidence":1.5},
            {"id":"clamp","start":58,"end":60.03,"confidence":0.4}
            "#,
        );
        let a = validate(d, ItemId(1), t(60.0)).unwrap();
        let ids: Vec<&str> = a.proposals.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, ["ok", "clamp"]);
        let why = |id: &str| a.rejected.iter().find(|r| r.id == id).map(|r| r.reason.clone()).unwrap_or_default();
        assert!(why("").contains("no id"));
        assert!(why("ok").contains("more than once"));
        assert!(why("rev").contains("ends before"));
        assert!(why("neg").contains("before the beginning"));
        assert!(why("late").contains("after the end"));
        assert!(why("tiny").contains("10 ms"));
        assert!(why("conf").contains("0 to 1"));
        assert!(a.proposals[1].end <= t(60.0) + t(0.05), "a hair past the end is clamped");
    }

    #[test]
    fn proposals_come_back_sorted_by_start() {
        let d = doc(60.0, r#"{"id":"b","start":20,"end":21,"confidence":0.5},{"id":"a","start":10,"end":11,"confidence":0.5}"#);
        let a = validate(d, ItemId(1), t(60.0)).unwrap();
        assert_eq!(a.proposals.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(), ["a", "b"]);
    }

    #[test]
    fn merging_joins_overlapping_and_touching_ranges_only() {
        let m = merge(vec![(t(5.0), t(6.0)), (t(1.0), t(2.0)), (t(1.5), t(3.0)), (t(3.0), t(4.0)), (t(8.0), t(9.0))]);
        assert_eq!(m, vec![(t(1.0), t(4.0)), (t(5.0), t(6.0)), (t(8.0), t(9.0))]);
        assert!(merge(vec![]).is_empty());
    }
}
