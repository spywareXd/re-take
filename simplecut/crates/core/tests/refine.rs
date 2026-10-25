//! Cut modes: every way a range from the service can be wrong must end in whole-sentence cuts,
//! never in a bad cut. The ranges come from outside and are not trusted.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use simplecut_core::{AnalysisJob, CutMode, FakeService, ItemId, JobState, RefineRequest, RetakeService, Sentence, Tick, WIDEN_SECS, ids_to_refine, plan_cuts};

const TPS: f64 = 254_016_000_000.0;

fn t(secs: f64) -> Tick {
    Tick::from_seconds_f64(secs)
}

fn sentence(id: &str, a: f64, b: f64) -> Sentence {
    Sentence { id: id.into(), start: t(a), end: t(b), text: format!("text of {id}"), confidence: None }
}

/// two removed sentences, 10 to 12 s and 20 to 23 s, in a 60 s file
fn removed() -> Vec<Sentence> {
    vec![sentence("a", 10.0, 12.0), sentence("b", 20.0, 23.0)]
}

fn seconds(c: &simplecut_core::CutProposal) -> (f64, f64) {
    (c.start.0 as f64 / TPS, c.end.0 as f64 / TPS)
}

fn plan(mode: CutMode, reply: &str) -> simplecut_core::RefinedCuts {
    plan_cuts(mode, &removed(), &[], Some(reply), t(60.0))
}

fn whole_sentences(cuts: &[simplecut_core::CutProposal]) -> bool {
    let mut spans: Vec<(f64, f64)> = cuts.iter().map(seconds).collect();
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    spans.len() == 2
        && (spans[0].0 - 10.0).abs() < 1e-6
        && (spans[0].1 - 12.0).abs() < 1e-6
        && (spans[1].0 - 20.0).abs() < 1e-6
        && (spans[1].1 - 23.0).abs() < 1e-6
}

#[test]
fn without_a_reply_each_sentence_is_cut_at_its_own_span() {
    let out = plan_cuts(CutMode::Anchored, &removed(), &[], None, t(60.0));
    assert!(whole_sentences(&out.cuts) && out.notes.is_empty());
}

#[test]
fn good_ranges_are_used_as_they_are() {
    let out = plan(CutMode::Anchored, r#"{"ranges":[{"start":9.6,"end":12.4},{"start":19.8,"end":23.5}]}"#);
    assert!(out.notes.is_empty(), "{:?}", out.notes);
    let mut spans: Vec<_> = out.cuts.iter().map(seconds).collect();
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert_eq!(spans, vec![(9.6, 12.4), (19.8, 23.5)].into_iter().map(|(a, b)| (t(a).0 as f64 / TPS, t(b).0 as f64 / TPS)).collect::<Vec<_>>());
    assert!(out.cuts.iter().all(|c| c.removed_text.starts_with("text of")));
}

#[test]
fn one_range_may_cover_two_neighbouring_sentences() {
    let removed = vec![sentence("a", 10.0, 12.0), sentence("b", 12.5, 14.0)];
    let out = plan_cuts(CutMode::RetakeOnly, &removed, &[], Some(r#"{"ranges":[{"start":10.0,"end":14.0}]}"#), t(60.0));
    assert_eq!(out.cuts.len(), 1);
    assert!(out.notes.is_empty());
}

#[test]
fn every_broken_range_falls_back_to_the_whole_sentences_and_says_so() {
    let broken = [
        r#"{"ranges":[{"start":12.0,"end":10.0},{"start":19.8,"end":23.5}]}"#, // backwards
        r#"{"ranges":[{"start":9.6,"end":12.4},{"start":19.8,"end":99.0}]}"#,  // past the end of the file
        r#"{"ranges":[{"start":-1.0,"end":12.4},{"start":19.8,"end":23.5}]}"#, // before the start
        r#"{"ranges":[{"start":9.6,"end":9.605},{"start":19.8,"end":23.5}]}"#, // shorter than the shortest cut
        r#"{"ranges":[{"start":9.6,"end":12.4},{"start":12.0,"end":13.0},{"start":19.8,"end":23.5}]}"#, // overlaps the one before
        r#"{"ranges":[{"start":0.0,"end":59.0}]}"#,                            // far outside the sentences
    ];
    for (i, reply) in broken.iter().enumerate() {
        let out = plan(CutMode::Anchored, reply);
        assert!(!out.notes.is_empty(), "case {i}: the user must be told");
        // whatever survives, nothing may reach past a removed sentence's widened span
        for c in &out.cuts {
            let (s, e) = seconds(c);
            let inside = |a: f64, b: f64| s >= a - WIDEN_SECS - 1e-6 && e <= b + WIDEN_SECS + 1e-6;
            assert!(inside(10.0, 12.0) || inside(20.0, 23.0), "case {i}: cut {s}..{e} is outside both sentences");
        }
    }
    // the first case: the bad range maps to sentence a, the good one stays
    let out = plan(CutMode::Anchored, broken[0]);
    assert_eq!(out.cuts.len(), 2);
}

#[test]
fn numbers_that_are_not_numbers_never_cut_anything() {
    for reply in [r#"{"ranges":[{"start":1e999,"end":2.0}]}"#, r#"{"ranges":[{"start":"x","end":2.0}]}"#, r#"{"ranges":null}"#, "[]", "not json", ""] {
        let out = plan(CutMode::Anchored, reply);
        assert!(whole_sentences(&out.cuts), "{reply}: {:?}", out.cuts);
        assert!(!out.notes.is_empty(), "{reply}");
    }
}

#[test]
fn a_removed_sentence_no_range_touches_is_cut_whole() {
    let out = plan(CutMode::Anchored, r#"{"ranges":[{"start":9.6,"end":12.4}]}"#);
    assert_eq!(out.cuts.len(), 2);
    assert!(out.cuts.iter().any(|c| c.id == "b" && seconds(c) == (20.0, 23.0)));
    assert_eq!(out.notes.len(), 1);
    let none = plan(CutMode::Anchored, r#"{"ranges":[]}"#);
    assert!(whole_sentences(&none.cuts) && none.notes.len() == 1);
}

#[test]
fn sentences_mode_may_remove_silence_far_from_the_retakes() {
    let reply = r#"{"ranges":[{"start":0.0,"end":3.0},{"start":9.7,"end":12.3},{"start":19.9,"end":23.2},{"start":40.0,"end":58.0}]}"#;
    let out = plan(CutMode::Sentences, reply);
    assert!(out.notes.is_empty(), "{:?}", out.notes);
    assert_eq!(out.cuts.len(), 4);
    // the same reply is refused in the other modes, where the first and last ranges are not near a retake
    let anchored = plan(CutMode::Anchored, reply);
    assert!(!anchored.notes.is_empty());
    assert!(anchored.cuts.iter().all(|c| seconds(c).1 <= 24.0));
}

#[test]
fn ranges_must_be_inside_the_file_in_every_mode() {
    let out = plan(CutMode::Sentences, r#"{"ranges":[{"start":9.7,"end":12.3},{"start":19.9,"end":61.0}]}"#);
    assert!(out.cuts.iter().all(|c| seconds(c).1 <= 60.0 + 1e-6));
    assert!(!out.notes.is_empty());
    // within the file's end tolerance the end is clamped, not refused
    let near = plan(CutMode::Sentences, r#"{"ranges":[{"start":9.7,"end":12.3},{"start":19.9,"end":60.03}]}"#);
    assert!(near.notes.is_empty());
    assert!((seconds(near.cuts.last().unwrap()).1 - 60.0).abs() < 1e-6 || near.cuts.iter().any(|c| (seconds(c).1 - 60.0).abs() < 1e-6));
}

#[test]
fn hand_edited_sentences_are_never_refined() {
    let edited = vec!["b".to_string()];
    assert_eq!(ids_to_refine(&removed(), &edited), vec!["a".to_string()]);
    let out = plan_cuts(CutMode::Anchored, &removed(), &edited, Some(r#"{"ranges":[{"start":9.6,"end":12.4}]}"#), t(60.0));
    assert!(out.cuts.iter().any(|c| c.id == "b" && seconds(c) == (20.0, 23.0)), "b is cut exactly as the user set it");
    assert!(out.cuts.iter().any(|c| seconds(c) == (t(9.6).0 as f64 / TPS, t(12.4).0 as f64 / TPS)));
    assert_eq!(out.cuts.len(), 2);
    // a range that reaches into the edited sentence is not accepted for it
    let reach = plan_cuts(CutMode::Anchored, &removed(), &edited, Some(r#"{"ranges":[{"start":9.6,"end":12.4},{"start":19.5,"end":23.5}]}"#), t(60.0));
    assert!(reach.cuts.iter().any(|c| c.id == "b"));
    assert!(reach.notes.iter().any(|n| n.contains("adjusted by hand")));
}

#[test]
fn a_huge_number_of_ranges_is_refused() {
    let many: Vec<String> = (0..10_001).map(|i| format!(r#"{{"start":{},"end":{}}}"#, f64::from(i) * 0.001, f64::from(i) * 0.001 + 0.0005)).collect();
    let out = plan(CutMode::Sentences, &format!(r#"{{"ranges":[{}]}}"#, many.join(",")));
    assert!(whole_sentences(&out.cuts) && !out.notes.is_empty());
}

#[test]
fn modes_have_service_names_and_only_sentences_warns() {
    assert_eq!(CutMode::default(), CutMode::Anchored);
    let names: Vec<&str> = CutMode::ALL.iter().map(|m| m.as_str()).collect();
    assert_eq!(names, ["anchored", "retake_only", "sentences"]);
    assert!(CutMode::Sentences.warning().is_some_and(|w| w.contains("no speech")));
    assert!(CutMode::Anchored.warning().is_none() && CutMode::RetakeOnly.warning().is_none());
}

fn run(job: Option<Box<dyn AnalysisJob>>) -> JobState {
    let mut job = job.expect("the stand-in service offers refine");
    for _ in 0..400 {
        match job.poll() {
            JobState::Running { .. } => std::thread::sleep(std::time::Duration::from_millis(10)),
            other => return other,
        }
    }
    panic!("never finished");
}

fn request(mode: CutMode) -> RefineRequest {
    RefineRequest {
        item: ItemId(1),
        job: "j".into(),
        mode,
        remove: vec!["a".into(), "b".into()],
        spans: vec![("a".into(), t(10.0), t(12.0)), ("b".into(), t(20.0), t(23.0))],
    }
}

#[test]
fn the_stand_in_service_answers_in_every_mode_and_its_ranges_pass_the_checks() {
    let service = FakeService { steps: 1, step_delay: std::time::Duration::from_millis(1), ..Default::default() };
    assert!(service.capabilities().refine);
    for mode in CutMode::ALL {
        let JobState::Done(reply) = run(service.refine(request(mode))) else { panic!("{mode:?} should finish") };
        let out = plan_cuts(mode, &removed(), &[], Some(&reply), t(60.0));
        assert!(out.notes.is_empty(), "{mode:?}: {:?}", out.notes);
        assert_eq!(out.cuts.len(), 2, "{mode:?}");
    }
}

#[test]
fn the_stand_in_can_misbehave_and_the_checks_hold() {
    let service = FakeService { steps: 1, step_delay: std::time::Duration::from_millis(1), bad_refine: true, ..Default::default() };
    let JobState::Done(reply) = run(service.refine(request(CutMode::Anchored))) else { panic!() };
    let out = plan_cuts(CutMode::Anchored, &removed(), &[], Some(&reply), t(60.0));
    assert!(!out.notes.is_empty());
    assert!(out.cuts.iter().all(|c| seconds(c).1 <= 24.0));
}

#[test]
fn a_service_without_refine_says_none() {
    let service = FakeService { refine: false, ..Default::default() };
    assert!(!service.capabilities().refine);
    assert!(service.refine(request(CutMode::Anchored)).is_none());
}
