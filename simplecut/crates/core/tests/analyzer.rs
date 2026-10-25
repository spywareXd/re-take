//! The retake service interface with the stand-in service (candidates, then groups after the
//! user's answers, then the optional AI advice), what the editor asks a service to look at, and the
//! mapping from file time to the timeline used by the review.
//!
//! Skipped (with a message) when ffmpeg / ffprobe are not installed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::{Duration, Instant};

use common::{have, make_clip, scratch};
use simplecut_core::retake::{CutProposal, CutScope, Decision};
use simplecut_core::review::{self, parse_advice};
use simplecut_core::{
    AdjudicateRequest, AnalysisJob, Candidates, Checked, Edge, Editor, EditorError, FakeService, Groups, ItemId, JobState, Label, LabelsRequest, Lane, PlaceAt,
    ResolveRequest, RetakeService, Tick, TrimMode, Verdict,
};

const TPS: f64 = 254_016_000_000.0;

fn secs(s: f64) -> Tick {
    Tick::from_seconds_f64(s)
}

fn s_of(t: Tick) -> f64 {
    t.0 as f64 / TPS
}

fn proposal(id: &str, start: f64, end: f64) -> CutProposal {
    CutProposal {
        id: id.into(),
        start: secs(start),
        end: secs(end),
        confidence: 0.9,
        decision: Decision::Remove,
        reason: String::new(),
        removed_text: String::new(),
        kept_text: String::new(),
        kept: None,
        source: None,
        group: None,
    }
}

/// A project with 10 s clips of the given names, placed one after the other.
fn project(names: &[&str]) -> Option<(Editor, Vec<ItemId>)> {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("SKIPPED: ffmpeg / ffprobe are not installed");
        return None;
    }
    let dir = scratch("analyzer");
    let mut e = Editor::new();
    e.new_project("T").unwrap();
    let mut items = Vec::new();
    for n in names {
        let id = e.import(&[make_clip(&dir, n, 10, true)]).unwrap().imported[0];
        e.place(id, Lane::Main, PlaceAt::End).unwrap();
        items.push(id);
    }
    e.select(&[]).unwrap();
    Some((e, items))
}

fn finish(job: &mut Box<dyn AnalysisJob>) -> (Vec<f32>, JobState) {
    let mut fractions = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        match job.poll() {
            JobState::Running { fraction, .. } => fractions.push(fraction),
            other => return (fractions, other),
        }
        assert!(Instant::now() < deadline, "the request never finished");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn done(job: &mut Box<dyn AnalysisJob>) -> String {
    match finish(job).1 {
        JobState::Done(text) => text,
        other => panic!("not done: {other:?}"),
    }
}

fn analyse(e: &Editor, item: ItemId, svc: &FakeService) -> (String, Candidates) {
    let request = e.analysis_requests().unwrap().into_iter().find(|r| r.item == item).unwrap();
    let mut job = svc.analyze(request);
    let text = done(&mut job);
    let Checked::Candidates(c) = e.read_review(&text, item).unwrap() else { panic!("not candidates") };
    (text, c)
}

fn resolve(e: &Editor, item: ItemId, svc: &FakeService, text: &str, c: &Candidates, labels: &[(&str, Verdict)], skipped: &[&str]) -> Groups {
    let labels = LabelsRequest {
        job: c.job.clone(),
        labels: labels.iter().map(|(p, v)| Label { pair: (*p).into(), label: *v, source: "user".into() }).collect(),
        skipped: skipped.iter().map(|s| (*s).into()).collect(),
    };
    let mut job = svc.resolve(ResolveRequest { item, labels, candidates: Some(text.into()) });
    let Checked::Groups(g) = e.read_review(&done(&mut job), item).unwrap() else { panic!("not groups") };
    g
}

fn ids(g: &Groups) -> Vec<(Vec<String>, bool)> {
    g.groups.iter().map(|x| (x.sentences.iter().map(|s| s.id.clone()).collect(), x.likely)).collect()
}

fn v(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn analyzing_returns_candidates_scaled_to_the_file_and_the_questions_in_video_order() {
    let Some((e, items)) = project(&["a.mp4"]) else { return };
    let (_, c) = analyse(&e, items[0], &FakeService::default());
    assert_eq!(c.sentences.len(), 14);
    assert!(c.rejected.is_empty(), "{:?}", c.rejected);
    assert!(s_of(c.sentences.last().unwrap().end) < 10.0, "a 10 s file gets 10 s of sentences: {:?}", c.sentences.last());
    let asked: Vec<&str> = c.ambiguous().iter().map(|p| p.id.as_str()).collect();
    assert_eq!(asked, ["9:0|11:0", "13:0|14:0", "17:0|18:0", "21:0|22:0"]);
}

#[test]
fn the_documented_answers_make_the_documented_groups() {
    let Some((e, items)) = project(&["a.mp4"]) else { return };
    let svc = FakeService::default();
    let (text, c) = analyse(&e, items[0], &svc);
    let g = resolve(&e, items[0], &svc, &text, &c, &[("9:0|11:0", Verdict::Retake), ("21:0|22:0", Verdict::Different)], &["13:0|14:0", "17:0|18:0"]);
    assert_eq!(
        ids(&g),
        vec![
            (v(&["3:0", "4:0", "5:0"]), true),
            (v(&["9:0", "11:0"]), true),
            (v(&["13:0", "14:0"]), true),
            (v(&["15:0", "16:0"]), true),
            (v(&["17:0", "18:0"]), false),
            (v(&["20:0", "21:0"]), true),
        ],
        "the user's answer joined 9:0 and 11:0, split 22:0 off, the model decided 13:0|14:0 and was unsure about 17:0|18:0"
    );
    assert_eq!(g.job, c.job);
    assert!(g.groups.iter().all(|x| x.sentences.iter().all(|s| s.id != "22:0")), "a sentence with no partner is in no group");
}

#[test]
fn other_answers_regroup_the_sentences() {
    let Some((e, items)) = project(&["a.mp4"]) else { return };
    let svc = FakeService::default();
    let (text, c) = analyse(&e, items[0], &svc);

    // "different" on 9:0|11:0 leaves them out of any group
    let g = resolve(&e, items[0], &svc, &text, &c, &[("9:0|11:0", Verdict::Different)], &[]);
    assert!(ids(&g).iter().all(|(s, _)| !s.contains(&"9:0".to_string())));

    // "retake" on 21:0|22:0 makes a group of three
    let g = resolve(&e, items[0], &svc, &text, &c, &[("21:0|22:0", Verdict::Retake)], &["17:0|18:0"]);
    assert!(ids(&g).contains(&(v(&["20:0", "21:0", "22:0"]), true)));

    // "retake" on the unsure pair makes that group likely
    let g = resolve(&e, items[0], &svc, &text, &c, &[("17:0|18:0", Verdict::Retake)], &[]);
    assert!(ids(&g).contains(&(v(&["17:0", "18:0"]), true)));

    // answering nothing: the pretend model decides everything it can
    let g = resolve(&e, items[0], &svc, &text, &c, &[], &[]);
    assert!(ids(&g).contains(&(v(&["17:0", "18:0"]), false)), "still unsure");
    assert!(ids(&g).contains(&(v(&["9:0", "11:0"]), true)), "the model said yes");
}

#[test]
fn the_service_never_says_what_to_keep() {
    let Some((e, items)) = project(&["a.mp4"]) else { return };
    let svc = FakeService::default();
    let (text, c) = analyse(&e, items[0], &svc);
    let request = ResolveRequest { item: items[0], labels: LabelsRequest { job: c.job.clone(), labels: vec![], skipped: vec![] }, candidates: Some(text) };
    let json = done(&mut svc.resolve(request));
    for forbidden in ["suggested_keep", "keep", "note", "reason", "score", "pairs"] {
        assert!(!json.contains(&format!("\"{forbidden}\"")), "the groups document must not contain “{forbidden}”: {json}");
    }
}

#[test]
fn the_service_can_answer_with_groups_at_once_cancel_fail_or_answer_for_another_file() {
    let Some((e, items)) = project(&["a.mp4"]) else { return };
    let req = || e.analysis_requests().unwrap().remove(0);

    let mut job = FakeService { skip_questions: true, ..Default::default() }.analyze(req());
    let Checked::Groups(g) = e.read_review(&done(&mut job), items[0]).unwrap() else { panic!("not groups") };
    assert!(!g.groups.is_empty(), "nothing to ask: groups straight away");

    let mut job = FakeService { steps: 100, step_delay: Duration::from_millis(30), ..Default::default() }.analyze(req());
    std::thread::sleep(Duration::from_millis(100));
    job.cancel();
    assert_eq!(finish(&mut job).1, JobState::Cancelled);

    let mut job = FakeService { fail_analyze: Some("the model is not installed".into()), ..Default::default() }.analyze(req());
    assert_eq!(finish(&mut job).1, JobState::Failed("the model is not installed".into()));

    let mut job = FakeService { wrong_length: true, ..Default::default() }.analyze(req());
    let text = done(&mut job);
    assert!(matches!(e.read_review(&text, items[0]), Err(EditorError::Invalid(m)) if m.contains("different file")));

    let svc = FakeService { fail_resolve: Some("out of memory".into()), ..Default::default() };
    let (t, c) = analyse(&e, items[0], &FakeService::default());
    let labels = LabelsRequest { job: c.job, labels: vec![], skipped: vec![] };
    let mut job = svc.resolve(ResolveRequest { item: items[0], labels, candidates: Some(t) });
    assert_eq!(finish(&mut job).1, JobState::Failed("out of memory".into()));
}

#[test]
fn the_ai_is_optional_and_its_advice_is_checked() {
    let Some((e, items)) = project(&["a.mp4"]) else { return };
    assert!(FakeService::default().capabilities().adjudicate);
    let off = FakeService { adjudicate: false, ..Default::default() };
    assert!(!off.capabilities().adjudicate);
    let g = {
        let (text, c) = analyse(&e, items[0], &off);
        resolve(&e, items[0], &off, &text, &c, &[], &[])
    };
    assert!(
        off.adjudicate(AdjudicateRequest { item: items[0], job: g.job.clone(), groups: g.groups.clone() }).is_none(),
        "a service without the extension has no AI"
    );

    let on = FakeService::default();
    let mut job = on.adjudicate(AdjudicateRequest { item: items[0], job: g.job.clone(), groups: g.groups.clone() }).unwrap();
    let (advice, notes) = parse_advice(&done(&mut job), &g.groups).unwrap();
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(advice.len(), g.groups.len());
    for a in &advice {
        let group = g.groups.iter().find(|x| x.id == a.group).unwrap();
        assert_eq!(a.keep, vec![group.sentences.last().unwrap().id.clone()]);
    }
    let _ = review::SCHEMA_V2;
}

#[test]
fn the_request_lists_each_file_once_with_the_parts_that_are_used() {
    let Some((mut e, items)) = project(&["a.mp4"]) else { return };
    e.place(items[0], Lane::Main, PlaceAt::End).unwrap(); // the same file again
    let clip = e.sequence().unwrap().tracks[0].clips[0].id;
    e.trim(clip, Edge::Start, secs(2.0), TrimMode::Ripple).unwrap(); // the first use shows 2 s to 10 s
    let r = e.analysis_requests().unwrap();
    assert_eq!(r.len(), 1, "one file, though used twice");
    assert_eq!(r[0].name, "a.mp4");
    assert!(r[0].path.ends_with("a.mp4"));
    assert_eq!(r[0].used.len(), 1, "2-10 and 0-10 overlap, so they are one used part: {:?}", r[0].used);
    assert!(s_of(r[0].used[0].0).abs() < 0.04 && (s_of(r[0].used[0].1) - 10.0).abs() < 0.04);

    let mut empty = Editor::new();
    empty.new_project("E").unwrap();
    assert!(matches!(empty.analysis_requests(), Err(EditorError::NotAvailable(_))));
}

#[test]
fn a_file_that_cannot_be_read_is_not_sent_to_the_service() {
    let Some((mut e, _)) = project(&["a.mp4"]) else { return };
    let dir = scratch("analyzer-offline");
    let file = dir.join("p.fcproj");
    e.save_as(&file).unwrap();
    let media = e.media()[0].path.clone().unwrap();
    std::fs::rename(&media, format!("{media}.moved")).unwrap();
    let mut again = Editor::new();
    again.open(&file).unwrap();
    assert!(matches!(again.analysis_requests(), Err(EditorError::FileMissing(_))));
}

#[test]
fn file_time_maps_to_where_it_is_on_the_timeline_and_short_spans_are_still_present() {
    let Some((mut e, items)) = project(&["a.mp4"]) else { return };
    let clip = e.sequence().unwrap().tracks[0].clips[0].id;
    e.trim(clip, Edge::Start, secs(1.0), TrimMode::Ripple).unwrap();
    let start = s_of(e.sequence().unwrap().tracks[0].clips[0].start);
    let r = e.timeline_ranges(items[0], &proposal("a", 3.0, 4.0));
    assert_eq!(r.len(), 1);
    assert!((s_of(r[0].0) - (start + 2.0)).abs() < 0.04 && (s_of(r[0].1) - (start + 3.0)).abs() < 0.04, "{r:?}");
    assert!(e.timeline_ranges(items[0], &proposal("gone", 0.1, 0.9)).is_empty(), "before the used part");
    assert!(e.timeline_ranges(ItemId(987_654), &proposal("x", 3.0, 4.0)).is_empty());

    // a span under one frame is still "on the timeline" when it is
    use simplecut_core::retake::Presence;
    assert_eq!(e.presence_of(items[0], secs(5.0), secs(5.02)), Presence::On);
    assert_eq!(e.presence_of(items[0], secs(0.1), secs(0.12)), Presence::Off);
    assert_eq!(e.presence_of(items[0], secs(5.0), secs(5.0)), Presence::Off, "an empty span is nowhere");
}

#[test]
fn an_analysis_finds_its_own_media_by_length_then_by_name() {
    let Some((e, items)) = project(&["first.mp4", "second.mp4"]) else { return };
    assert_eq!(e.find_media_by("C:\\videos\\SECOND.mp4", 10.0).unwrap(), items[1]);
    assert_eq!(e.find_media_by("/home/me/first.mp4", 10.1).unwrap(), items[0]);
    assert!(matches!(e.find_media_by("other.mp4", 10.0), Err(EditorError::Invalid(m)) if m.contains("More than one")));
    assert!(matches!(e.find_media_by("first.mp4", 99.0), Err(EditorError::Invalid(m)) if m.contains("None of the files")));
    let doc = r#"{"schema":"simplecut.retakes/2","stage":"groups","media":{"path":"x/second.mp4","duration_seconds":10.0},"groups":[]}"#.to_string();
    assert_eq!(e.find_media_for_review(&doc).unwrap(), items[1]);
}

#[test]
fn cuts_in_several_files_are_one_undo_step() {
    let Some((mut e, items)) = project(&["a.mp4", "b.mp4"]) else { return };
    let steps = e.session().history.undo.len();
    let r = e.apply_cut_sets(&[(items[0], vec![proposal("a", 2.0, 3.0)]), (items[1], vec![proposal("b", 5.0, 6.0)])], CutScope::MainTracks).unwrap();
    assert_eq!(r.cuts, 2);
    assert!((s_of(r.removed) - 2.0).abs() < 0.04);
    assert!((s_of(e.sequence().unwrap().duration) - 18.0).abs() < 0.05);
    assert_eq!(e.session().history.undo.len(), steps + 1);
    e.undo().unwrap();
    assert!((s_of(e.sequence().unwrap().duration) - 20.0).abs() < 0.05);
    assert_eq!(e.apply_cut_sets(&[(items[0], vec![])], CutScope::MainTracks).unwrap_err(), EditorError::NothingSelected);
}

#[test]
fn pairs_carry_a_lean_and_no_percentage_is_needed() {
    let Some((e, items)) = project(&["a.mp4"]) else { return };
    let (_, c) = analyse(&e, items[0], &FakeService::default());
    assert!(c.pairs.iter().all(|p| p.lean.is_some()), "the stand-in service says which way each pair leans");
    let q = c.ambiguous();
    assert_eq!(q.iter().find(|p| p.id == "17:0|18:0").unwrap().lean, Some(Verdict::Retake), "0.52 leans towards a retake");

    // a document without lean still loads, one with a bad lean loses only that pair
    let text = serde_json::json!({
        "schema": "simplecut.retakes/2", "stage": "candidates", "job": "j",
        "media": {"path": "x/a.mp4", "duration_seconds": 10.0},
        "sentences": [
            {"id": "a", "start": 1.0, "end": 2.0, "text": "one", "internal_retake": {"start": 1.2, "end": 1.5}},
            {"id": "b", "start": 3.0, "end": 4.0, "text": "two"}],
        "pairs": [
            {"id": "a|b", "a": "a", "b": "b", "class": "ambiguous", "score": 0.5},
            {"id": "b|a", "a": "b", "b": "a", "class": "ambiguous", "score": 0.5, "lean": "maybe"}]
    })
    .to_string();
    let Checked::Candidates(c) = e.read_review(&text, items[0]).unwrap() else { panic!("not candidates") };
    assert_eq!(c.sentences.len(), 2, "an internal_retake field on a sentence is ignored");
    assert_eq!(c.pairs.len(), 1);
    assert_eq!(c.pairs[0].lean, None);
    assert!(c.rejected.iter().any(|r| r.id == "b|a" && r.reason.contains("lean")), "{:?}", c.rejected);
}

#[test]
fn the_fingerprint_decides_which_file_an_analysis_is_for() {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("SKIPPED: ffmpeg / ffprobe are not installed");
        return;
    }
    let dir = scratch("fingerprint-project");
    let mut e = Editor::new();
    e.new_project("T").unwrap();
    let a = e.import(&[make_clip(&dir, "a.mp4", 10, true)]).unwrap().imported[0];
    let b = e.import(&[make_clip(&dir, "b.mp4", 12, true)]).unwrap().imported[0];
    e.place(a, Lane::Main, PlaceAt::End).unwrap();
    e.place(b, Lane::Main, PlaceAt::End).unwrap();
    e.select(&[]).unwrap();

    let (text, _) = analyse(&e, b, &FakeService::default());
    let mut doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(doc["media"]["fingerprint"], serde_json::Value::String(e.fingerprint_of(b).unwrap()), "the stand-in service echoes the file's fingerprint");
    assert_eq!(e.find_media_for_review(&text).unwrap(), b);

    // the fingerprint wins over a length that points at the other file
    doc["media"]["duration_seconds"] = 10.0.into();
    assert_eq!(e.find_media_for_review(&doc.to_string()).unwrap(), b, "length says a, fingerprint says b");
    doc["media"]["fingerprint"] = serde_json::Value::Null;
    assert_eq!(e.find_media_for_review(&doc.to_string()).unwrap(), a, "without a fingerprint the length decides, as before");

    // the right file accepts the document, the other one refuses it, a made-up fingerprint matches nothing
    let mut doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(e.read_review(&doc.to_string(), b).is_ok());
    let wrong = e.read_review(&doc.to_string(), a).unwrap_err();
    assert!(wrong.to_string().contains("different") || wrong.to_string().contains("None"), "{wrong}");
    doc["media"]["fingerprint"] = "00".repeat(32).into();
    assert!(e.read_review(&doc.to_string(), b).unwrap_err().to_string().contains("different file"));
    assert!(matches!(e.find_media_for_review(&doc.to_string()), Err(EditorError::Invalid(m)) if m.contains("fingerprint")));
}

#[test]
fn the_fingerprint_recipe_is_size_plus_first_and_last_mebibyte() {
    let dir = scratch("fingerprint");
    let write = |name: &str, data: &[u8]| {
        let p = dir.join(name);
        std::fs::write(&p, data).unwrap();
        p
    };
    let fp = |p: &std::path::Path| review::fingerprint_file(p).unwrap();

    // small file: all of it counts
    let small = write("small.bin", b"hello");
    let small2 = write("small2.bin", b"hellp");
    assert_eq!(fp(&small).len(), 64);
    assert_ne!(fp(&small), fp(&small2));
    assert_eq!(fp(&small), fp(&write("small_copy.bin", b"hello")), "same bytes, same fingerprint");

    // big file: the first and last MiB and the size count, the middle does not
    let mut big = vec![7u8; 3 << 20];
    let base = fp(&write("big.bin", &big));
    big[(3 << 20) / 2] = 9;
    assert_eq!(fp(&write("big_mid.bin", &big)), base, "the middle is not read");
    big[10] = 9;
    assert_ne!(fp(&write("big_head.bin", &big)), base);
    big[10] = 7;
    big[(3 << 20) - 1] = 9;
    assert_ne!(fp(&write("big_tail.bin", &big)), base);
    assert_ne!(fp(&write("big_longer.bin", &vec![7u8; (3 << 20) + 1])), base, "the size counts");

    // a known value pins the recipe the backend must follow: sha256(8 bytes LE size || bytes)
    assert_eq!(fp(&write("empty.bin", b"")), "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc");
    assert!(review::fingerprint_file(&dir.join("nope.bin")).is_err(), "an unreadable file is an error, not a panic");
}
