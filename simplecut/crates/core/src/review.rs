//! The retake contract, version 2 (`simplecut.retakes/2`, see `docs/retake-contract.md`).
//!
//! The pipeline only finds sentences and groups them; it never says what to keep and never calls
//! an LLM. Two documents come from it: **candidates** (sentences and the pairs of them it thinks
//! may be the same line, so the user can answer the uncertain ones) and, after the user's answers
//! went through the cross-encoder, **groups** (sentences, each group with its span and a `likely`
//! flag). The editor checks both strictly: whatever cannot be trusted is dropped with a reason, and
//! nothing is guessed.

use serde::{Deserialize, Serialize};

use filmcraft_time::Tick;

use crate::editor::ItemId;
use crate::error::{EditorError, Result};
use crate::retake::{CutProposal, DURATION_TOLERANCE_SECS, Decision, END_TOLERANCE_SECS, Generator, MAX_DOCUMENT_BYTES, MIN_CUT_SECS, Rejected};

/// The schema this module reads.
pub const SCHEMA_V2: &str = "simplecut.retakes/2";
const MAX_SENTENCES: usize = 200_000;
const MAX_PAIRS: usize = 500_000;
/// A group span may differ from its sentences by this much (seconds) before the group is refused.
const SPAN_TOLERANCE_SECS: f64 = 0.01;

// ── as written by the pipeline ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocSentence {
    pub id: String,
    pub start: f64,
    pub end: f64,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub confidence: Option<f64>,
}

/// How sure the pipeline is about a pair of sentences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PairClass {
    Likely,
    Ambiguous,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocPair {
    pub id: String,
    pub a: String,
    pub b: String,
    pub class: PairClass,
    pub score: f64,
    /// The model's leaning ("retake" or "different"). Scores are not calibrated probabilities, so
    /// this is what the user is shown, never a percentage.
    #[serde(default)]
    pub lean: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocGroup {
    pub id: String,
    pub start: f64,
    pub end: f64,
    /// The pipeline considers this a retake with high confidence.
    pub likely: bool,
    pub sentences: Vec<DocSentence>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    Candidates,
    Groups,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocMedia {
    #[serde(default)]
    pub path: String,
    pub duration_seconds: f64,
    /// Hash of the file (see [`fingerprint_file`]). When present it must match the project's file.
    #[serde(default)]
    pub fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocV2 {
    pub schema: String,
    pub stage: Stage,
    #[serde(default)]
    pub job: String,
    pub media: DocMedia,
    #[serde(default)]
    pub generator: Option<Generator>,
    #[serde(default)]
    pub sentences: Vec<DocSentence>,
    #[serde(default)]
    pub pairs: Vec<DocPair>,
    #[serde(default)]
    pub groups: Vec<DocGroup>,
}

/// Read a version 2 document. Fails with a plain message when it is not JSON, is too big, has no
/// stage, or is another version.
pub fn parse_v2(json: &str) -> Result<DocV2> {
    if json.len() > MAX_DOCUMENT_BYTES {
        return Err(EditorError::Invalid("The retake analysis is too large to read.".into()));
    }
    let value: serde_json::Value = serde_json::from_str(json).map_err(|e| EditorError::Invalid(format!("The retake analysis is not valid JSON: {e}")))?;
    match value.get("schema").and_then(serde_json::Value::as_str) {
        Some(SCHEMA_V2) => {}
        Some(other) => {
            return Err(EditorError::Invalid(format!("The retake analysis uses the format “{other}”, but this version of SimpleCut reads “{SCHEMA_V2}”.")));
        }
        None => return Err(EditorError::Invalid("The retake analysis has no “schema” field, so it cannot be recognised.".into())),
    }
    if value.get("stage").is_none() {
        return Err(EditorError::Invalid("The retake analysis does not say whether it holds candidates or groups (“stage”).".into()));
    }
    serde_json::from_value(value).map_err(|e| EditorError::Invalid(format!("The retake analysis is incomplete or has a wrong value: {e}")))
}

// ── checked ────────────────────────────────────────────────────────────────

/// One sentence of the video, with times on the media file's own clock.
#[derive(Debug, Clone, PartialEq)]
pub struct Sentence {
    pub id: String,
    pub start: Tick,
    pub end: Tick,
    pub text: String,
    pub confidence: Option<f32>,
}

impl Sentence {
    /// This sentence as the unit the cutting code works with.
    pub fn as_cut(&self) -> CutProposal {
        CutProposal {
            id: self.id.clone(),
            start: self.start,
            end: self.end,
            confidence: self.confidence.unwrap_or(1.0),
            decision: Decision::Remove,
            reason: String::new(),
            removed_text: self.text.clone(),
            kept_text: String::new(),
            kept: None,
            source: None,
            group: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pair {
    pub id: String,
    pub a: String,
    pub b: String,
    pub class: PairClass,
    pub score: f32,
    pub lean: Option<Verdict>,
}

/// The pipeline's first answer: sentences and the pairs it wants the user to look at.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidates {
    pub item: ItemId,
    pub job: String,
    pub generator: Option<Generator>,
    /// In time order.
    pub sentences: Vec<Sentence>,
    pub pairs: Vec<Pair>,
    pub rejected: Vec<Rejected>,
}

impl Candidates {
    pub fn sentence(&self, id: &str) -> Option<&Sentence> {
        self.sentences.iter().find(|s| s.id == id)
    }

    /// The pairs the user is asked about, in video order (by the earlier sentence, then the later).
    pub fn ambiguous(&self) -> Vec<&Pair> {
        let mut v: Vec<&Pair> = self.pairs.iter().filter(|p| p.class == PairClass::Ambiguous).collect();
        v.sort_by_key(|p| {
            let (a, b) = (self.sentence(&p.a).map(|s| s.start), self.sentence(&p.b).map(|s| s.start));
            (a.min(b), a.max(b), p.id.clone())
        });
        v
    }
}

/// A set of sentences the pipeline believes are attempts at one line.
#[derive(Debug, Clone, PartialEq)]
pub struct ReviewGroup {
    pub id: String,
    pub start: Tick,
    pub end: Tick,
    pub likely: bool,
    /// In time order.
    pub sentences: Vec<Sentence>,
}

impl ReviewGroup {
    /// Update the span of sentence `sentence_id` within this group.
    ///
    /// Rules:
    /// - `start < end` and span is at least [`MIN_CUT_SECS`] long.
    /// - Inside `0.0..=media_duration`.
    /// - Rejects NaN, infinite, negative, or reversed input.
    /// - Clamps against neighbouring sentences in the same group so spans cannot cross.
    /// - Recomputes the group's total start and end from its sentences.
    pub fn set_span(&mut self, sentence_id: &str, start: f64, end: f64, media_duration: f64) -> Result<()> {
        if !start.is_finite() || !end.is_finite() || !media_duration.is_finite() {
            return Err(EditorError::Invalid("Span start, end, and media duration must be finite numbers.".into()));
        }
        if media_duration <= 0.0 {
            return Err(EditorError::Invalid("Media duration must be positive.".into()));
        }
        if start < 0.0 {
            return Err(EditorError::Invalid("Span start cannot be negative.".into()));
        }
        if end > media_duration {
            return Err(EditorError::Invalid(format!("Span end ({end:.3}s) cannot be after media duration ({media_duration:.3}s).")));
        }
        if start >= end {
            return Err(EditorError::Invalid(format!("Span start ({start:.3}s) must be before end ({end:.3}s).")));
        }
        if end - start < MIN_CUT_SECS {
            return Err(EditorError::Invalid(format!("Span must be at least {MIN_CUT_SECS} seconds.")));
        }

        let idx = self
            .sentences
            .iter()
            .position(|s| s.id == sentence_id)
            .ok_or_else(|| EditorError::Invalid(format!("Sentence “{sentence_id}” not found in group “{}”.", self.id)))?;

        let mut start_tick = Tick::from_seconds_f64(start);
        let mut end_tick = Tick::from_seconds_f64(end);

        // Clamp to neighbouring sentences of the same group
        if let Some(prev) = idx.checked_sub(1).and_then(|i| self.sentences.get(i)) {
            let prev_end = prev.end;
            if start_tick < prev_end {
                start_tick = prev_end;
            }
        }
        if let Some(next) = self.sentences.get(idx + 1) {
            let next_start = next.start;
            if end_tick > next_start {
                end_tick = next_start;
            }
        }

        let min_cut_ticks = Tick::from_seconds_f64(MIN_CUT_SECS);
        if end_tick.0 <= start_tick.0 || (end_tick.0 - start_tick.0) < min_cut_ticks.0 {
            return Err(EditorError::Invalid("Span cannot cross neighbouring sentence or become shorter than minimum cut length.".into()));
        }

        let Some(target) = self.sentences.get_mut(idx) else { return Err(EditorError::Invalid("Sentence not found.".into())) };
        target.start = start_tick;
        target.end = end_tick;

        if let (Some(min_s), Some(max_e)) = (self.sentences.iter().map(|s| s.start).min(), self.sentences.iter().map(|s| s.end).max()) {
            self.start = min_s;
            self.end = max_e;
        }

        Ok(())
    }
}

/// The pipeline's second answer: groups of sentences.
#[derive(Debug, Clone, PartialEq)]
pub struct Groups {
    pub item: ItemId,
    pub job: String,
    pub generator: Option<Generator>,
    /// In time order.
    pub groups: Vec<ReviewGroup>,
    pub rejected: Vec<Rejected>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Checked {
    Candidates(Candidates),
    Groups(Groups),
}

fn secs(t: f64) -> Tick {
    Tick::from_seconds_f64(t)
}

/// A sentence is usable if its times are real numbers inside the file; `file_secs` is the file's
/// length. The end may run up to 50 ms past the file and is then clamped.
fn check_sentence(s: DocSentence, file_secs: f64) -> std::result::Result<Sentence, String> {
    if s.id.trim().is_empty() {
        return Err("it has no id".into());
    }
    if !s.start.is_finite() || !s.end.is_finite() {
        return Err("its start or end is not a number".into());
    }
    if s.start < 0.0 {
        return Err("it starts before the beginning of the file".into());
    }
    if s.end <= s.start {
        return Err("it ends before it starts".into());
    }
    if s.end > file_secs + END_TOLERANCE_SECS {
        return Err("it ends after the end of the file".into());
    }
    let end = s.end.min(file_secs);
    if end - s.start < MIN_CUT_SECS {
        return Err("it is shorter than 10 ms".into());
    }
    let confidence = match s.confidence {
        Some(c) if c.is_finite() && (0.0..=1.0).contains(&c) => Some(c as f32),
        Some(_) => return Err("its confidence is not a number from 0 to 1".into()),
        None => None,
    };
    Ok(Sentence { id: s.id, start: secs(s.start), end: secs(end), text: s.text, confidence })
}

/// Check a version 2 document against the file `item` it is for (`media_duration` long).
///
/// Refused as a whole: a document for another file, a candidates document with no job id, one with
/// more sentences or pairs than can be reviewed. Dropped one by one, with a reason: a sentence, pair
/// or group that cannot be trusted.
pub fn validate_v2(doc: DocV2, item: ItemId, media_duration: Tick) -> Result<Checked> {
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
    if doc.sentences.len() > MAX_SENTENCES || doc.pairs.len() > MAX_PAIRS {
        return Err(EditorError::Invalid("The retake analysis has too many sentences to review.".into()));
    }
    let mut rejected: Vec<Rejected> = Vec::new();
    match doc.stage {
        Stage::Candidates => {
            if doc.job.trim().is_empty() {
                return Err(EditorError::Invalid("The retake analysis has no job id, so your answers could not be sent back.".into()));
            }
            let mut sentences: Vec<Sentence> = Vec::new();
            for s in doc.sentences {
                let id = s.id.clone();
                if sentences.iter().any(|x| x.id == id) {
                    rejected.push(Rejected { id, reason: "its id is used more than once".into() });
                    continue;
                }
                match check_sentence(s, file_secs) {
                    Ok(ok) => sentences.push(ok),
                    Err(why) => rejected.push(Rejected { id, reason: why }),
                }
            }
            sentences.sort_by(|a, b| (a.start, &a.id).cmp(&(b.start, &b.id)));
            let mut pairs: Vec<Pair> = Vec::new();
            for p in doc.pairs {
                let why = if p.id.trim().is_empty() {
                    Some("it has no id")
                } else if pairs.iter().any(|x| x.id == p.id) {
                    Some("its id is used more than once")
                } else if p.a == p.b {
                    Some("it pairs a sentence with itself")
                } else if !sentences.iter().any(|s| s.id == p.a) || !sentences.iter().any(|s| s.id == p.b) {
                    Some("it names a sentence that is not usable")
                } else if !p.score.is_finite() || !(0.0..=1.0).contains(&p.score) {
                    Some("its score is not a number from 0 to 1")
                } else if p.lean.as_deref().is_some_and(|l| l != "retake" && l != "different") {
                    Some("its lean is not \"retake\" or \"different\"")
                } else {
                    None
                };
                match why {
                    Some(w) => rejected.push(Rejected { id: p.id, reason: w.into() }),
                    None => {
                        let lean = match p.lean.as_deref() {
                            Some("retake") => Some(Verdict::Retake),
                            Some("different") => Some(Verdict::Different),
                            _ => None,
                        };
                        pairs.push(Pair { id: p.id, a: p.a, b: p.b, class: p.class, score: p.score as f32, lean });
                    }
                }
            }
            Ok(Checked::Candidates(Candidates { item, job: doc.job, generator: doc.generator, sentences, pairs, rejected }))
        }
        Stage::Groups => {
            let mut groups: Vec<ReviewGroup> = Vec::new();
            for g in doc.groups {
                let gid = g.id.clone();
                match check_group(g, file_secs, &groups) {
                    Ok(ok) => groups.push(ok),
                    Err(why) => rejected.push(Rejected { id: gid, reason: why }),
                }
            }
            groups.sort_by(|a, b| (a.start, &a.id).cmp(&(b.start, &b.id)));
            Ok(Checked::Groups(Groups { item, job: doc.job, generator: doc.generator, groups, rejected }))
        }
    }
}

fn check_group(g: DocGroup, file_secs: f64, earlier: &[ReviewGroup]) -> std::result::Result<ReviewGroup, String> {
    if g.id.trim().is_empty() {
        return Err("it has no id".into());
    }
    if earlier.iter().any(|x| x.id == g.id) {
        return Err("its id is used more than once".into());
    }
    if g.sentences.len() < 2 {
        return Err("it has fewer than two sentences".into());
    }
    if !g.start.is_finite() || !g.end.is_finite() || g.end <= g.start {
        return Err("its start and end are not a valid span".into());
    }
    let mut sentences: Vec<Sentence> = Vec::new();
    for s in g.sentences {
        let sid = s.id.clone();
        if sentences.iter().any(|x| x.id == sid) {
            return Err(format!("sentence “{sid}” appears twice"));
        }
        if let Some(other) = earlier.iter().find(|x| x.sentences.iter().any(|y| y.id == sid)) {
            return Err(format!("sentence “{sid}” is already in group “{}”", other.id));
        }
        sentences.push(check_sentence(s, file_secs).map_err(|why| format!("sentence “{sid}”: {why}"))?);
    }
    sentences.sort_by(|a, b| (a.start, &a.id).cmp(&(b.start, &b.id)));
    let (lo, hi) = (sentences.iter().map(|s| s.start).min().unwrap_or_default(), sentences.iter().map(|s| s.end).max().unwrap_or_default());
    let tol = secs(SPAN_TOLERANCE_SECS);
    let (start, end) = (secs(g.start), secs(g.end.min(file_secs + END_TOLERANCE_SECS)));
    if start > lo + tol || end + tol < hi {
        return Err("its start and end do not cover all of its sentences".into());
    }
    Ok(ReviewGroup { id: g.id, start: start.min(lo), end: end.max(hi), likely: g.likely, sentences })
}

// ── the user's answers ─────────────────────────────────────────────────────

/// What the user said about a pair of sentences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// The same line, said again.
    Retake,
    /// Two different lines.
    Different,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Label {
    pub pair: String,
    pub label: Verdict,
    /// Always `"user"`: only human answers are training labels.
    pub source: String,
}

/// What the editor sends when the user presses Next.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelsRequest {
    pub job: String,
    pub labels: Vec<Label>,
    pub skipped: Vec<String>,
}

impl LabelsRequest {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".into())
    }
}

// ── the optional AI button ─────────────────────────────────────────────────

/// What the service can do beyond finding groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Capabilities {
    #[serde(default)]
    pub adjudicate: bool,
}

/// The AI's advice for one group: which sentences to keep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Advice {
    pub group: String,
    pub keep: Vec<String>,
    pub reason: String,
}

#[derive(Deserialize)]
struct RawAdvice {
    group: String,
    keep: Vec<String>,
    #[serde(default)]
    reason: String,
}

/// Read the AI's reply for `groups`. Strict: a reply for an unknown group, one that keeps nothing,
/// or one that names a sentence that is not in the group is discarded and reported in the notes,
/// never repaired.
pub fn parse_advice(json: &str, groups: &[ReviewGroup]) -> Result<(Vec<Advice>, Vec<String>)> {
    if json.len() > MAX_DOCUMENT_BYTES {
        return Err(EditorError::Invalid("The AI reply is too large to read.".into()));
    }
    let raw: Vec<RawAdvice> = serde_json::from_str(json).map_err(|e| EditorError::Invalid(format!("The AI reply is not in the expected form: {e}")))?;
    let mut advice: Vec<Advice> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    for r in raw {
        let Some(group) = groups.iter().find(|g| g.id == r.group) else {
            notes.push(format!("The AI answered for “{}”, which is not one of the groups it was asked about.", r.group));
            continue;
        };
        if advice.iter().any(|a| a.group == r.group) {
            notes.push(format!("The AI answered twice for “{}”; the first answer was used.", r.group));
            continue;
        }
        if r.keep.is_empty() {
            notes.push(format!("The AI kept nothing in “{}”, so its answer was ignored.", r.group));
            continue;
        }
        if let Some(bad) = r.keep.iter().find(|k| !group.sentences.iter().any(|s| &s.id == *k)) {
            notes.push(format!("The AI named a sentence (“{bad}”) that is not in “{}”, so its answer for that group was ignored.", r.group));
            continue;
        }
        let mut keep = r.keep;
        keep.dedup();
        advice.push(Advice { group: r.group, keep, reason: r.reason });
    }
    Ok((advice, notes))
}

/// The groups formatted as a full version 2 "groups" document (seconds, as in the contract).
pub fn groups_to_json(job: &str, media_path: &str, duration_seconds: f64, groups: &[ReviewGroup]) -> String {
    let secs = |t: Tick| t.0 as f64 / 254_016_000_000.0;
    let value = serde_json::json!({
        "schema": SCHEMA_V2,
        "stage": "groups",
        "job": job,
        "media": {
            "path": media_path,
            "duration_seconds": duration_seconds,
        },
        "groups": groups.iter().map(|g| serde_json::json!({
            "id": g.id,
            "start": secs(g.start),
            "end": secs(g.end),
            "likely": g.likely,
            "sentences": g.sentences.iter().map(|s| serde_json::json!({
                "id": s.id,
                "start": secs(s.start),
                "end": secs(s.end),
                "text": s.text,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    });
    serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string())
}

/// The identity of a media file: SHA-256, lowercase hex, over the file size (u64, little endian), the
/// first 1 MiB and the last 1 MiB (the whole file when it is smaller than 2 MiB). Cheap on any disk.
/// The retake service must compute the same value.
pub fn fingerprint_file(path: &std::path::Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Seek, SeekFrom};
    const PART: u64 = 1 << 20;
    let mut file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    let mut hasher = Sha256::new();
    hasher.update(size.to_le_bytes());
    let mut buf = Vec::new();
    if size <= 2 * PART {
        file.read_to_end(&mut buf)?;
        hasher.update(&buf);
    } else {
        buf.resize(PART as usize, 0);
        file.read_exact(&mut buf)?;
        hasher.update(&buf);
        file.seek(SeekFrom::Start(size - PART))?;
        file.read_exact(&mut buf)?;
        hasher.update(&buf);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

impl crate::editor::Editor {
    /// Read and check a version 2 document for the media file `item` of this project.
    pub fn read_review(&self, json: &str, item: ItemId) -> Result<Checked> {
        let entry = self.media_entry(item).ok_or(EditorError::NoSuchMedia)?;
        let doc = parse_v2(json)?;
        if let (Some(want), Some(have)) = (doc.media.fingerprint.as_deref(), self.fingerprint_of(item))
            && !want.trim().eq_ignore_ascii_case(&have)
        {
            return Err(EditorError::Invalid("That analysis was made for a different file (its fingerprint does not match this one).".into()));
        }
        validate_v2(doc, item, entry.duration)
    }

    /// The fingerprint of the project's file `item`; None when it has no path or cannot be read.
    pub fn fingerprint_of(&self, item: ItemId) -> Option<String> {
        let path = self.media_entry(item)?.path?;
        fingerprint_file(std::path::Path::new(&path)).ok()
    }

    /// Which file of this project a version 2 document is for (by length, then by file name).
    pub fn find_media_for_review(&self, json: &str) -> Result<ItemId> {
        let doc = parse_v2(json)?;
        // a fingerprint is a far stronger identity than the length: the file that matches it wins
        if let Some(want) = doc.media.fingerprint.as_deref().map(str::trim).filter(|f| !f.is_empty()) {
            let hit = self.media().into_iter().find(|m| self.fingerprint_of(m.id).is_some_and(|f| f.eq_ignore_ascii_case(want)));
            return hit
                .map(|m| m.id)
                .ok_or_else(|| EditorError::Invalid("None of the files in this project matches that analysis (its fingerprint is for another file).".into()));
        }
        self.find_media_by(&doc.media.path, doc.media.duration_seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANDIDATES: &str = include_str!("../../../docs/examples/retake-candidates.json");
    const LABELS: &str = include_str!("../../../docs/examples/retake-labels.json");
    const GROUPS: &str = include_str!("../../../docs/examples/retake-groups.json");
    const ADVICE: &str = include_str!("../../../docs/examples/retake-adjudication.json");
    const CAPS: &str = include_str!("../../../docs/examples/retake-capabilities.json");

    fn t(s: f64) -> Tick {
        secs(s)
    }

    fn checked(json: &str) -> Checked {
        validate_v2(parse_v2(json).unwrap(), ItemId(1), t(60.0)).unwrap()
    }

    #[test]
    fn the_documented_candidates_are_valid() {
        let Checked::Candidates(c) = checked(CANDIDATES) else { panic!("not candidates") };
        assert!(c.rejected.is_empty(), "{:?}", c.rejected);
        assert_eq!(c.job, "j_8f3a");
        assert_eq!(c.sentences.len(), 14);
        assert_eq!(c.pairs.len(), 9);
        let asked: Vec<&str> = c.ambiguous().iter().map(|p| p.id.as_str()).collect();
        assert_eq!(asked, ["9:0|11:0", "13:0|14:0", "17:0|18:0", "21:0|22:0"], "the questions come in video order");
        assert!(c.sentences.windows(2).all(|w| w[0].start <= w[1].start));
    }

    #[test]
    fn the_documented_groups_are_valid() {
        let Checked::Groups(g) = checked(GROUPS) else { panic!("not groups") };
        assert!(g.rejected.is_empty(), "{:?}", g.rejected);
        assert_eq!(g.groups.len(), 6);
        let likely: Vec<bool> = g.groups.iter().map(|x| x.likely).collect();
        assert_eq!(likely, [true, true, true, true, false, true]);
        assert_eq!((g.groups[0].start, g.groups[0].end), (t(4.1), t(14.6)));
        assert_eq!(g.groups[0].sentences.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), ["3:0", "4:0", "5:0"]);
    }

    #[test]
    fn the_documented_labels_advice_and_capabilities_are_valid() {
        let l: LabelsRequest = serde_json::from_str(LABELS).unwrap();
        assert_eq!(l.job, "j_8f3a");
        assert_eq!(l.labels.len(), 2);
        assert_eq!(l.labels[0].label, Verdict::Retake);
        assert_eq!(l.labels[1].label, Verdict::Different);
        assert_eq!(l.skipped, ["13:0|14:0", "17:0|18:0"]);
        assert_eq!(serde_json::from_str::<LabelsRequest>(&l.to_json()).unwrap(), l, "it round-trips");

        let Checked::Groups(g) = checked(GROUPS) else { panic!("not groups") };
        let (advice, notes) = parse_advice(ADVICE, &g.groups).unwrap();
        assert!(notes.is_empty(), "{notes:?}");
        assert_eq!(advice, vec![Advice { group: "span_4".into(), keep: vec!["18:0".into()], reason: advice[0].reason.clone() }]);
        assert!(!advice[0].reason.is_empty());

        assert!(serde_json::from_str::<Capabilities>(CAPS).unwrap().adjudicate);
        assert!(!serde_json::from_str::<Capabilities>("{}").unwrap().adjudicate, "a service that says nothing offers nothing");
    }

    #[test]
    fn parsing_refuses_what_it_cannot_trust() {
        assert!(matches!(parse_v2("nope"), Err(EditorError::Invalid(m)) if m.contains("not valid JSON")));
        assert!(matches!(parse_v2("{}"), Err(EditorError::Invalid(m)) if m.contains("no “schema”")));
        let v1 = r#"{"schema":"simplecut.retakes/1","media":{"duration_seconds":1},"proposals":[]}"#;
        assert!(matches!(parse_v2(v1), Err(EditorError::Invalid(m)) if m.contains("retakes/1") && m.contains("retakes/2")));
        let no_stage = r#"{"schema":"simplecut.retakes/2","media":{"duration_seconds":1}}"#;
        assert!(matches!(parse_v2(no_stage), Err(EditorError::Invalid(m)) if m.contains("stage")));
        let bad_stage = r#"{"schema":"simplecut.retakes/2","stage":"pairs","media":{"duration_seconds":1}}"#;
        assert!(parse_v2(bad_stage).is_err());
        let text_time = r#"{"schema":"simplecut.retakes/2","stage":"groups","media":{"duration_seconds":10},"groups":[{"id":"g","start":"1","end":2,"likely":true,"sentences":[]}]}"#;
        assert!(parse_v2(text_time).is_err(), "times must be numbers");
        let extra = r#"{"schema":"simplecut.retakes/2","stage":"groups","media":{"duration_seconds":10,"fps":30},"groups":[],"future":1}"#;
        assert!(parse_v2(extra).is_ok(), "fields it does not know are ignored");
    }

    fn doc(stage: &str, extra: &str) -> DocV2 {
        parse_v2(&format!(r#"{{"schema":"simplecut.retakes/2","stage":"{stage}","job":"j","media":{{"duration_seconds":60.0}},{extra}}}"#)).unwrap()
    }

    #[test]
    fn a_document_for_another_file_or_without_a_job_is_refused() {
        let wrong = parse_v2(&CANDIDATES.replace("60.0", "99.0")).unwrap();
        let e = validate_v2(wrong, ItemId(1), t(60.0)).unwrap_err();
        assert!(matches!(&e, EditorError::Invalid(m) if m.contains("different file")), "{e}");
        let no_job = parse_v2(&CANDIDATES.replace("\"job\": \"j_8f3a\",", "")).unwrap();
        assert!(matches!(validate_v2(no_job, ItemId(1), t(60.0)), Err(EditorError::Invalid(m)) if m.contains("no job id")));
        // a groups document needs no job to be read
        assert!(validate_v2(doc("groups", r#""groups":[]"#), ItemId(1), t(60.0)).is_ok());
    }

    #[test]
    fn unusable_sentences_and_pairs_are_dropped_with_the_reason() {
        let d = doc(
            "candidates",
            r#""sentences":[
                {"id":"a","start":1,"end":2,"text":"one"},
                {"id":"a","start":3,"end":4,"text":"again"},
                {"id":"","start":3,"end":4},
                {"id":"rev","start":5,"end":4},
                {"id":"late","start":59,"end":70},
                {"id":"tiny","start":5,"end":5.005},
                {"id":"b","start":6,"end":7,"confidence":3},
                {"id":"c","start":8,"end":9}
            ],
            "pairs":[
                {"id":"a|c","a":"a","b":"c","class":"likely","score":0.9},
                {"id":"a|c","a":"a","b":"c","class":"likely","score":0.9},
                {"id":"a|a","a":"a","b":"a","class":"likely","score":0.9},
                {"id":"a|rev","a":"a","b":"rev","class":"ambiguous","score":0.5},
                {"id":"a|c2","a":"a","b":"c","class":"ambiguous","score":1.5}
            ]"#,
        );
        let Checked::Candidates(c) = validate_v2(d, ItemId(1), t(60.0)).unwrap() else { panic!() };
        assert_eq!(c.sentences.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), ["a", "c"]);
        assert_eq!(c.pairs.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(), ["a|c"]);
        let why = |id: &str| c.rejected.iter().filter(|r| r.id == id).map(|r| r.reason.clone()).collect::<Vec<_>>().join(" / ");
        assert!(why("a").contains("more than once"));
        assert!(why("").contains("no id"));
        assert!(why("rev").contains("ends before"));
        assert!(why("late").contains("after the end"));
        assert!(why("tiny").contains("10 ms"));
        assert!(why("b").contains("confidence"));
        assert!(why("a|c").contains("more than once"));
        assert!(why("a|a").contains("itself"));
        assert!(why("a|rev").contains("not usable"));
        assert!(why("a|c2").contains("score"));
    }

    #[test]
    fn unusable_groups_are_dropped_and_good_ones_kept() {
        let s = |id: &str, a: f64, b: f64| format!(r#"{{"id":"{id}","start":{a},"end":{b}}}"#);
        let d = doc(
            "groups",
            &format!(
                r#""groups":[
                {{"id":"good","start":1,"end":5,"likely":true,"sentences":[{},{}]}},
                {{"id":"good","start":10,"end":12,"likely":true,"sentences":[{},{}]}},
                {{"id":"one","start":20,"end":22,"likely":true,"sentences":[{}]}},
                {{"id":"span","start":2,"end":3,"likely":true,"sentences":[{},{}]}},
                {{"id":"shared","start":1,"end":5,"likely":false,"sentences":[{},{}]}},
                {{"id":"dup","start":30,"end":35,"likely":true,"sentences":[{},{}]}},
                {{"id":"nan","start":40,"end":30,"likely":true,"sentences":[{},{}]}},
                {{"id":"bad","start":40,"end":50,"likely":true,"sentences":[{},{}]}}
            ]"#,
                s("a", 1.0, 2.0),
                s("b", 4.0, 5.0),
                s("c", 10.0, 11.0),
                s("d", 11.5, 12.0),
                s("e", 20.0, 22.0),
                s("f", 30.0, 31.0),
                s("g", 40.0, 41.0),
                s("a", 1.0, 2.0),
                s("h", 4.0, 5.0),
                s("x", 30.0, 31.0),
                s("x", 33.0, 34.0),
                s("y", 41.0, 42.0),
                s("z", 43.0, 44.0),
                s("w", 45.0, 49.0),
                s("v", 49.0, 50.0),
            ),
        );
        let Checked::Groups(g) = validate_v2(d, ItemId(1), t(60.0)).unwrap() else { panic!() };
        assert_eq!(g.groups.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), ["good", "bad"]);
        let why = |id: &str| g.rejected.iter().filter(|r| r.id == id).map(|r| r.reason.clone()).collect::<Vec<_>>().join(" / ");
        assert!(why("good").contains("more than once"));
        assert!(why("one").contains("fewer than two"));
        assert!(why("span").contains("do not cover"));
        assert!(why("shared").contains("already in group"));
        assert!(why("dup").contains("appears twice"));
        assert!(why("nan").contains("not a valid span"));
    }

    #[test]
    fn a_groups_span_may_be_a_hair_off_but_never_short() {
        let ok = doc(
            "groups",
            r#""groups":[{"id":"g","start":0.995,"end":5.005,"likely":true,"sentences":[{"id":"a","start":1,"end":2},{"id":"b","start":4,"end":5}]}]"#,
        );
        let Checked::Groups(g) = validate_v2(ok, ItemId(1), t(60.0)).unwrap() else { panic!() };
        assert_eq!((g.groups[0].start, g.groups[0].end), (t(0.995), t(5.005)));
        let short =
            doc("groups", r#""groups":[{"id":"g","start":1.5,"end":5,"likely":true,"sentences":[{"id":"a","start":1,"end":2},{"id":"b","start":4,"end":5}]}]"#);
        let Checked::Groups(g) = validate_v2(short, ItemId(1), t(60.0)).unwrap() else { panic!() };
        assert!(g.groups.is_empty() && g.rejected.len() == 1);
    }

    #[test]
    fn the_ais_reply_is_read_strictly() {
        let Checked::Groups(g) = checked(GROUPS) else { panic!("not groups") };
        let reply = r#"[
            {"group":"span_4","keep":["18:0"],"reason":"more specific"},
            {"group":"span_4","keep":["17:0"]},
            {"group":"nope","keep":["1"]},
            {"group":"span_0","keep":[]},
            {"group":"span_1","keep":["9:0","made-up"]},
            {"group":"span_2","keep":["14:0","14:0"],"reason":""}
        ]"#;
        let (advice, notes) = parse_advice(reply, &g.groups).unwrap();
        assert_eq!(advice.iter().map(|a| a.group.as_str()).collect::<Vec<_>>(), ["span_4", "span_2"]);
        assert_eq!(advice[1].keep, ["14:0"], "a repeated id is collapsed");
        assert_eq!(notes.len(), 4, "{notes:?}");
        assert!(notes.iter().any(|n| n.contains("twice")) && notes.iter().any(|n| n.contains("not one of the groups")));
        assert!(notes.iter().any(|n| n.contains("kept nothing")) && notes.iter().any(|n| n.contains("made-up")));
        assert!(parse_advice("not json", &g.groups).is_err());
        assert!(parse_advice(r#"{"group":"x"}"#, &g.groups).is_err(), "it must be a list");
    }

    #[test]
    fn groups_are_sent_to_the_ai_in_seconds() {
        let Checked::Groups(g) = checked(GROUPS) else { panic!("not groups") };
        let sent: serde_json::Value = serde_json::from_str(&groups_to_json("j_8f3a", "C:/videos/take1.mp4", 60.0, &g.groups[..1])).unwrap();
        assert_eq!(sent["job"], "j_8f3a");
        assert_eq!(sent["groups"][0]["id"], "span_0");
        assert_eq!(sent["groups"][0]["sentences"].as_array().unwrap().len(), 3);
        assert!((sent["groups"][0]["sentences"][2]["end"].as_f64().unwrap() - 14.6).abs() < 1e-9);
        assert!(sent["groups"][0]["sentences"][0].get("keep").is_none(), "the AI is not told what to keep");
    }

    #[test]
    fn groups_to_json_produces_valid_v2_groups_document() {
        let Checked::Groups(g) = checked(GROUPS) else { panic!("not groups") };
        let json = groups_to_json(&g.job, "C:/videos/take1.mp4", 60.0, &g.groups);
        let doc = parse_v2(&json).expect("valid json schema");
        let validated = validate_v2(doc, ItemId(1), Tick::from_seconds_f64(60.0)).expect("valid groups doc");
        let Checked::Groups(out) = validated else { panic!("not groups") };
        assert_eq!(out.groups.len(), g.groups.len());
        assert_eq!(out.groups[0].id, g.groups[0].id);
        assert_eq!(out.groups[0].sentences.len(), g.groups[0].sentences.len());
    }

    #[test]
    fn a_sentence_becomes_a_cut_for_the_apply_code() {
        let s = Sentence { id: "x".into(), start: t(1.0), end: t(2.0), text: "hello".into(), confidence: None };
        let c = s.as_cut();
        assert_eq!((c.id.as_str(), c.start, c.end, c.removed_text.as_str()), ("x", t(1.0), t(2.0), "hello"));
    }

    #[test]
    fn review_group_set_span_hostile_numbers_and_neighbour_clamping() {
        let make_group = || ReviewGroup {
            id: "g1".into(),
            start: t(1.0),
            end: t(5.0),
            likely: true,
            sentences: vec![
                Sentence { id: "s1".into(), start: t(1.0), end: t(2.0), text: "first".into(), confidence: None },
                Sentence { id: "s2".into(), start: t(3.0), end: t(5.0), text: "second".into(), confidence: None },
            ],
        };

        let media_dur = 60.0;

        // 1. Hostile inputs: NaN, infinity, reversed, negative, huge, zero length, too short
        let hostile_cases = [
            (f64::NAN, 4.0, "NaN start"),
            (3.0, f64::NAN, "NaN end"),
            (f64::INFINITY, 4.0, "+inf start"),
            (3.0, f64::INFINITY, "+inf end"),
            (f64::NEG_INFINITY, 4.0, "-inf start"),
            (4.0, 2.0, "reversed"),
            (-1.0, 4.0, "negative start"),
            (100.0, 120.0, "huge past media duration"),
            (3.0, 65.0, "end past media duration"),
            (3.0, 3.0, "zero length"),
            (3.0, 3.005, "under MIN_CUT_SECS"),
        ];

        for (st, en, label) in hostile_cases {
            let mut g = make_group();
            assert!(g.set_span("s2", st, en, media_dur).is_err(), "hostile input should fail: {label}");
            assert_eq!(g, make_group(), "failed edit must leave group unchanged");
        }

        // Hostile media duration
        let mut g = make_group();
        assert!(g.set_span("s2", 3.0, 4.0, f64::NAN).is_err());
        assert!(g.set_span("s2", 3.0, 4.0, -10.0).is_err());
        assert!(g.set_span("s2", 3.0, 4.0, 0.0).is_err());

        // Non-existent sentence
        assert!(g.set_span("nonexistent", 3.0, 4.0, media_dur).is_err());

        // 2. Normal edit within bounds updates sentence and recomputes group span
        let mut g = make_group();
        g.set_span("s2", 2.5, 6.0, media_dur).unwrap();
        assert_eq!(g.sentences[1].start, t(2.5));
        assert_eq!(g.sentences[1].end, t(6.0));
        assert_eq!(g.start, t(1.0));
        assert_eq!(g.end, t(6.0), "group end expanded");

        // 3. Moving sentence 1 start earlier updates group start
        g.set_span("s1", 0.5, 2.0, media_dur).unwrap();
        assert_eq!(g.sentences[0].start, t(0.5));
        assert_eq!(g.start, t(0.5), "group start expanded earlier");

        // 4. Neighbour clamping:
        // Trying to move s2 start before s1 end (2.0): clamps to 2.0
        let mut g = make_group();
        g.set_span("s2", 1.5, 4.0, media_dur).unwrap();
        assert_eq!(g.sentences[1].start, t(2.0), "clamped to prev sentence end");
        assert_eq!(g.sentences[1].end, t(4.0));

        // Trying to move s1 end past s2 start (3.0): clamps to 3.0
        let mut g = make_group();
        g.set_span("s1", 1.0, 3.5, media_dur).unwrap();
        assert_eq!(g.sentences[0].start, t(1.0));
        assert_eq!(g.sentences[0].end, t(3.0), "clamped to next sentence start");

        // Trying to move s2 completely before s1: clamped start = 2.0, end = 1.5 -> rejected
        let mut g = make_group();
        assert!(g.set_span("s2", 0.5, 1.5, media_dur).is_err(), "cannot cross neighbour completely");
        assert_eq!(g, make_group(), "group unchanged after error");

        // Trying to move s1 completely after s2: clamped end = 3.0, start = 3.5 -> rejected
        let mut g = make_group();
        assert!(g.set_span("s1", 3.5, 4.5, media_dur).is_err(), "cannot cross neighbour completely");
        assert_eq!(g, make_group(), "group unchanged after error");

        // Clamping leaving less than MIN_CUT_SECS: rejected
        let mut g = make_group();
        assert!(g.set_span("s2", 1.0, 2.005, media_dur).is_err(), "clamped span < MIN_CUT_SECS");
        assert_eq!(g, make_group());
    }
}
