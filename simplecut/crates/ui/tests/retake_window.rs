//! Remove Retakes in the window, step by step: the button, the questions, the groups, the keep
//! ticks, the markers on the timeline, removing what is not kept, and the optional AI. The service
//! is the stand-in one; everything around it is real.
//!
//! Skipped (with a message) when ffmpeg / ffprobe are not installed.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::harness::{Script, click, item, skip, step_until, window, window_with, with_media_and_service};
use egui::{Pos2, pos2, vec2};
use egui_kittest::kittest::{NodeT, Queryable};
use simplecut_core::{CutMode, FakeService, Lane, Tick, Verdict};
use simplecut_ui::Action;

const TPS: f64 = 254_016_000_000.0;

fn s_of(t: Tick) -> f64 {
    t.0 as f64 / TPS
}

type Win = common::harness::Window;

fn ready_with_script(service: FakeService) -> (Win, Script, std::path::PathBuf) {
    let (mut h, script, dir) = with_media_and_service(service);
    let video = item(&h, "talk.mp4");
    h.state_mut().perform(Action::AddToTimeline(video, Lane::Main));
    h.state_mut().perform(Action::SelectClips { clips: vec![], add: false });
    h.step();
    h.step();
    (h, script, dir)
}

fn drag_mouse(h: &mut Win, from: Pos2, to: Pos2) {
    h.hover_at(from);
    h.step();
    h.drag_at(from);
    h.step();
    for i in 1..=5 {
        h.hover_at(from + (to - from) * (i as f32 / 5.0));
        h.step();
    }
    h.drop_at(to);
    h.step();
    h.step();
}

fn ready(service: FakeService) -> (Win, std::path::PathBuf) {
    let (mut h, _, dir) = with_media_and_service(service);
    let video = item(&h, "talk.mp4");
    h.state_mut().perform(Action::AddToTimeline(video, Lane::Main));
    h.state_mut().perform(Action::SelectClips { clips: vec![], add: false });
    h.step();
    h.step();
    (h, dir)
}

fn stage(h: &mut Win, what: &'static str) {
    step_until(h, what, |a| a.review_stage() == what);
}

/// Start, answer the way the documentation does, press Next and wait for the groups.
fn to_review(h: &mut Win) {
    click(h, "Remove Retakes");
    stage(h, "questions");
    let ids = h.state().question_ids();
    assert_eq!(ids, ["9:0|11:0", "13:0|14:0", "17:0|18:0", "21:0|22:0"]);
    h.state_mut().perform(Action::AnswerPair { pair: ids[0].clone(), verdict: Some(Verdict::Retake) });
    h.state_mut().perform(Action::AnswerPair { pair: ids[1].clone(), verdict: None });
    h.state_mut().perform(Action::AnswerPair { pair: ids[2].clone(), verdict: None });
    h.state_mut().perform(Action::AnswerPair { pair: ids[3].clone(), verdict: Some(Verdict::Different) });
    h.step();
    h.state_mut().perform(Action::QuestionsNext);
    h.step();
    h.state_mut().perform(Action::QuestionsSendAnyway);
    stage(h, "review");
}

fn video_total(h: &Win) -> f64 {
    let v = h.state().editor().sequence().unwrap();
    v.tracks[0].clips.iter().map(|c| s_of(c.duration)).sum()
}

#[test]
fn the_button_appears_only_with_a_service_and_a_video_on_the_timeline() {
    if skip() {
        return;
    }
    let mut plain = window(Script::default());
    plain.step();
    assert!(plain.query_by_label("Remove Retakes").is_none(), "no service, no button");

    let mut with = window_with(Script::default(), Some(Box::new(FakeService::default())));
    with.step();
    assert!(with.get_by_label("Remove Retakes").accesskit_node().is_disabled(), "nothing on the timeline to look at");

    let (h, _) = ready(FakeService::default());
    assert!(!h.get_by_label("Remove Retakes").accesskit_node().is_disabled());
}

#[test]
fn the_steps_go_questions_then_groups_and_nothing_is_cut_until_the_user_chooses() {
    if skip() {
        return;
    }
    let (mut h, dir) = ready(FakeService::default());
    h.state_mut().set_label_log(Some(dir.join("labels.jsonl")));
    let steps = h.state().editor().session().history.undo.len();
    let before = video_total(&h);

    click(&mut h, "Remove Retakes");
    assert!(h.state().is_analyzing());
    h.get_by_label("Cancel analysis");
    stage(&mut h, "questions");
    assert_eq!(h.state().question_summary(), Some((4, 0, 0, 0)));
    h.get_by_label("Same line");
    h.get_by_label("Different lines");
    h.get_by_label("The model leans: same line.");
    assert!(h.query_by_label_contains("% sure").is_none(), "scores are not probabilities, so no percentage is shown");

    // answering with the real buttons moves to the next card
    click(&mut h, "Same line");
    assert_eq!(h.state().question_summary(), Some((4, 1, 0, 1)));
    click(&mut h, "Skip");
    assert_eq!(h.state().question_summary(), Some((4, 1, 1, 2)));

    // Next with questions left over asks first
    click(&mut h, "Next");
    assert_eq!(h.state().review_stage(), "questions", "waiting for the user to confirm");
    h.get_by_label("Send anyway");
    click(&mut h, "Send anyway");
    assert!(h.state().is_analyzing(), "the model is working");
    stage(&mut h, "review");

    // the answers were logged before they were sent
    let log = std::fs::read_to_string(dir.join("labels.jsonl")).unwrap();
    assert_eq!(log.lines().count(), 1, "{log}");
    assert!(log.contains("\"label\":\"retake\"") && log.contains("9:0|11:0") && log.contains("\"source\":\"user\""), "{log}");

    // likely groups first; nothing is cut yet, all sentences are kept
    let groups = h.state().review_groups();
    assert!(groups.len() >= 5, "{groups:?}");
    assert!(groups[0].2 && groups.iter().any(|g| !g.2), "there are likely and possible groups");
    assert_eq!(h.state().review_cut_count(), 0);
    assert!(groups.iter().flat_map(|g| g.3.iter()).all(|id| h.state().review_kept(id) == Some(true)));
    assert!(h.state().review_kept_rects().len() >= 10, "kept sentences are outlined");
    assert!(h.state().review_band_rects().len() >= 5, "every group has a band");
    assert!(h.state().review_marker_rects().is_empty(), "nothing will be cut");
    assert_eq!(h.state().editor().session().history.undo.len(), steps);
    h.get_by_label("Remove 0 sentences");

    // untick the first take of the first group
    let first = groups[0].3[0].clone();
    h.state_mut().perform(Action::SetKeep { sentence: first.clone(), keep: false });
    h.step();
    h.step();
    assert_eq!(h.state().review_kept(&first), Some(false));
    assert_eq!(h.state().review_origin(&first), Some("user"));
    assert_eq!(h.state().review_cut_count(), 1);
    assert_eq!(h.state().review_marker_rects().len(), 1);
    assert_eq!(h.state().review_marker_colors(), [groups[0].1]);
    h.get_by_label("Remove 1 sentence");

    click(&mut h, "Remove 1 sentence");
    step_until(&mut h, "removed", |a| a.banner().is_some_and(|b| !b.error && b.text.contains("Removed")));
    assert!(video_total(&h) < before - 0.02, "{} vs {before}", video_total(&h));
    assert_eq!(h.state().editor().session().history.undo.len(), steps + 1, "one undo step");
    assert!(h.state().banner().is_some_and(|b| !b.error && b.text.contains("Removed")), "{:?}", h.state().banner());

    click(&mut h, "Undo");
    assert!((video_total(&h) - before).abs() < 0.05);
    assert_eq!(h.state().review_stage(), "review");
    assert_eq!(h.state().review_groups().len(), groups.len(), "undo brings the group back");
}

#[test]
fn at_least_one_sentence_of_a_group_stays_and_the_helpers_work() {
    if skip() {
        return;
    }
    let (mut h, _) = ready(FakeService::default());
    to_review(&mut h);
    let groups = h.state().review_groups();
    let ids = groups[0].3.clone();
    assert_eq!(ids.len(), 3);

    // keep last only
    let gid = "span_0".to_string();
    h.state_mut().perform(Action::KeepLastOnly(gid.clone()));
    h.step();
    assert_eq!(h.state().review_kept(&ids[0]), Some(false));
    assert_eq!(h.state().review_kept(&ids[1]), Some(false));
    assert_eq!(h.state().review_kept(&ids[2]), Some(true));
    assert_eq!(h.state().review_cut_count(), 2);

    // unticking the last sentence now works, cutting the whole group
    h.state_mut().perform(Action::SetKeep { sentence: ids[2].clone(), keep: false });
    h.step();
    assert_eq!(h.state().review_kept(&ids[2]), Some(false));
    assert_eq!(h.state().review_cut_count(), ids.len());
    assert!(!h.state().banner().is_some_and(|b| b.error));

    // remove all helper works directly
    h.state_mut().perform(Action::KeepAllOf(gid.clone()));
    h.step();
    assert_eq!(h.state().review_cut_count(), 0);
    h.state_mut().perform(Action::RemoveAllOf(gid.clone()));
    h.step();
    assert_eq!(h.state().review_cut_count(), ids.len());
    assert_eq!(h.state().review_kept(&ids[0]), Some(false));
    assert_eq!(h.state().review_kept(&ids[1]), Some(false));
    assert_eq!(h.state().review_kept(&ids[2]), Some(false));

    // buttons exist
    h.get_by_label("Remove all in Retake 1");
    h.get_by_label("Keep all in Retake 1");
    h.get_by_label("Keep last only in Retake 1");

    // keep all undoes the choice
    h.state_mut().perform(Action::KeepAllOf(gid));
    h.step();
    assert_eq!(h.state().review_cut_count(), 0);

    // one click for every likely group
    h.state_mut().perform(Action::KeepLastInLikely);
    h.step();
    let likely_cuts: usize = h.state().review_groups().iter().filter(|g| g.2).map(|g| g.3.len() - 1).sum();
    assert_eq!(h.state().review_cut_count(), likely_cuts);
    assert!(h.state().review_cut_count() > 0);
    assert!(
        h.state().review_groups().iter().filter(|g| !g.2).flat_map(|g| g.3.iter()).all(|id| h.state().review_kept(id) == Some(true)),
        "possible groups are left alone"
    );
}

#[test]
fn the_questions_can_be_asked_again_and_an_answer_can_be_changed() {
    if skip() {
        return;
    }
    let (mut h, _) = ready(FakeService::default());
    to_review(&mut h);
    h.state_mut().perform(Action::RedoQuestions);
    h.step();
    assert_eq!(h.state().review_stage(), "questions");
    assert_eq!(h.state().question_summary().map(|q| (q.0, q.1)), Some((4, 2)), "the answers are still there");
    h.state_mut().perform(Action::AnswerPair { pair: "17:0|18:0".into(), verdict: Some(Verdict::Retake) });
    h.state_mut().perform(Action::QuestionsNext);
    h.state_mut().perform(Action::QuestionsSendAnyway);
    stage(&mut h, "review");
    assert!(h.state().review_groups().iter().any(|g| g.3 == ["17:0", "18:0"] && g.2), "the changed answer made it likely");
}

#[test]
fn the_ai_button_exists_only_when_the_service_offers_it_and_only_fills_in_ticks() {
    if skip() {
        return;
    }
    let (mut none, _) = ready(FakeService { adjudicate: false, ..Default::default() });
    to_review(&mut none);
    assert!(none.query_by_label("Ask AI to choose").is_none());
    assert!(!none.state().ai_available());

    let (mut h, _) = ready(FakeService { adjudicate: true, ..Default::default() });
    to_review(&mut h);
    let before = video_total(&h);
    let steps = h.state().editor().session().history.undo.len();
    click(&mut h, "Ask AI to choose");
    h.get_by_label("Continue");
    assert!(!h.state().is_analyzing(), "nothing is sent before the user agrees");
    click(&mut h, "Continue");
    step_until(&mut h, "the AI", |a| !a.is_analyzing() && a.banner().is_some_and(|b| b.text.contains("AI chose")));
    assert!(h.state().banner().is_some_and(|b| !b.error && b.text.contains("Nothing is cut")), "{:?}", h.state().banner());
    let tagged: Vec<String> = h.state().review_groups().iter().flat_map(|g| g.3.clone()).filter(|id| h.state().review_origin(id) == Some("ai")).collect();
    assert!(!tagged.is_empty(), "the AI's ticks are tagged");
    assert!((video_total(&h) - before).abs() < 1e-9, "nothing was cut");
    assert_eq!(h.state().editor().session().history.undo.len(), steps);

    // the user can override
    let group = h.state().review_groups().iter().find(|g| g.3.contains(&tagged[0])).map(|g| g.3.clone()).unwrap();
    h.state_mut().perform(Action::SetKeep { sentence: tagged[0].clone(), keep: true });
    h.step();
    assert_eq!(h.state().review_origin(&tagged[0]), Some("user"));
    assert!(group.len() >= 2);
}

#[test]
fn a_service_with_nothing_to_ask_goes_straight_to_the_groups() {
    if skip() {
        return;
    }
    let (mut h, _) = ready(FakeService { skip_questions: true, ..Default::default() });
    click(&mut h, "Remove Retakes");
    stage(&mut h, "review");
    assert!(!h.state().review_groups().is_empty());
}

#[test]
fn cancelling_failing_and_wrong_files_leave_nothing_behind() {
    if skip() {
        return;
    }
    let (mut h, _) = ready(FakeService { steps: 50, ..Default::default() });
    click(&mut h, "Remove Retakes");
    click(&mut h, "Cancel analysis");
    assert_eq!(h.state().review_stage(), "none");
    assert!(!h.state().is_analyzing());

    let (mut h, _) = ready(FakeService { fail_analyze: Some("model not found".into()), ..Default::default() });
    click(&mut h, "Remove Retakes");
    step_until(&mut h, "the failure", |a| a.banner().is_some_and(|b| b.error));
    assert!(h.state().banner().unwrap().text.contains("model not found"));
    assert_eq!(h.state().review_stage(), "none");

    let (mut h, _) = ready(FakeService { wrong_length: true, ..Default::default() });
    click(&mut h, "Remove Retakes");
    step_until(&mut h, "the refusal", |a| a.banner().is_some_and(|b| b.error));
    assert_eq!(h.state().review_stage(), "none");

    // a failing model keeps the answers
    let (mut h, _) = ready(FakeService { fail_resolve: Some("out of memory".into()), ..Default::default() });
    click(&mut h, "Remove Retakes");
    stage(&mut h, "questions");
    h.state_mut().perform(Action::AnswerPair { pair: "9:0|11:0".into(), verdict: Some(Verdict::Retake) });
    h.state_mut().perform(Action::QuestionsNext);
    h.state_mut().perform(Action::QuestionsSendAnyway);
    step_until(&mut h, "the failure", |a| a.banner().is_some_and(|b| b.error && b.text.contains("out of memory")));
    stage(&mut h, "questions");
    assert_eq!(h.state().question_summary().map(|q| q.1), Some(1), "the answer is kept");
}

#[test]
fn an_analysis_file_made_by_hand_can_be_loaded() {
    if skip() {
        return;
    }
    let (mut h, script, dir) = common::harness::with_media();
    let video = item(&h, "talk.mp4");
    h.state_mut().perform(Action::AddToTimeline(video, Lane::Main));
    h.step();

    let good = dir.join("talk.groups.json");
    let json = serde_json::json!({
        "schema": "simplecut.retakes/2",
        "stage": "groups",
        "job": "j_1",
        "media": { "path": "D:/elsewhere/TALK.mp4", "duration_seconds": 4.0 },
        "groups": [
            { "id": "g0", "start": 0.5, "end": 2.0, "likely": true, "sentences": [
                { "id": "a:0", "start": 0.5, "end": 1.0, "text": "Hello there" },
                { "id": "b:0", "start": 1.5, "end": 2.0, "text": "Hello there everyone" } ] },
            { "id": "g1", "start": 3.0, "end": 2.0, "likely": true, "sentences": [
                { "id": "c:0", "start": 3.0, "end": 3.2, "text": "x" },
                { "id": "d:0", "start": 3.4, "end": 3.6, "text": "y" } ] }
        ]
    });
    std::fs::write(&good, json.to_string()).unwrap();
    script.analysis.lock().unwrap().push_back(good);
    h.state_mut().perform(Action::OpenAnalysisFile);
    h.step();
    assert_eq!(h.state().review_stage(), "review");
    assert_eq!(h.state().review_groups().len(), 1);
    assert_eq!(h.state().review_ignored(), 1, "the reversed group is ignored and reported");

    // a file for a different video, one that is not JSON, and one that is missing leave the review alone
    let other = dir.join("other.json");
    std::fs::write(&other, json.to_string().replace("4.0", "99.0")).unwrap();
    h.state_mut().perform(Action::LoadAnalysis(other));
    h.step();
    assert!(h.state().banner().unwrap().error);
    let junk = dir.join("junk.json");
    std::fs::write(&junk, "not json").unwrap();
    h.state_mut().perform(Action::LoadAnalysis(junk));
    h.step();
    assert!(h.state().banner().unwrap().error);
    h.state_mut().perform(Action::LoadAnalysis(dir.join("nope.json")));
    h.step();
    assert!(h.state().banner().unwrap().text.contains("could not be found"));
    assert_eq!(h.state().review_groups().len(), 1);
}

#[test]
fn a_group_that_is_cut_moves_to_already_handled_and_undo_brings_it_back() {
    if skip() {
        return;
    }
    let (mut h, _) = ready(FakeService::default());
    to_review(&mut h);
    let n = h.state().review_groups().len();
    h.state_mut().perform(Action::KeepLastOnly("span_0".into()));
    h.state_mut().perform(Action::ApplyRetakes);
    step_until(&mut h, "applied", |a| a.review_groups().len() == n - 1);
    assert_eq!(h.state().review_groups().len(), n - 1, "only one take of that line is on the timeline now");
    assert_eq!(h.state().review_handled().len(), 1);

    click(&mut h, "Undo");
    assert_eq!(h.state().review_groups().len(), n);
    assert!(h.state().review_handled().is_empty());
}

#[test]
fn preview_skips_unticked_sentences_checkbox_toggle_remove_and_undo() {
    if skip() {
        return;
    }
    let (mut h, _) = ready(FakeService::default());
    to_review(&mut h);

    // Initial state: review is open, "Skip unticked sentences when playing" default ON
    assert!(h.state().review_skips_unticked());
    // Untick sentences of span_0 (keep last only)
    h.state_mut().perform(Action::KeepLastOnly("span_0".into()));
    h.step();

    // Skips are set in player from Cut marks
    assert!(!h.state().player().current_skips().is_empty(), "player has skips from unticked sentences");

    // Toggle checkbox OFF
    click(&mut h, "Skip unticked sentences when playing");
    h.step();
    assert!(!h.state().review_skips_unticked());
    assert!(h.state().player().current_skips().is_empty(), "skips cleared when checkbox off");

    // Toggle checkbox back ON
    click(&mut h, "Skip unticked sentences when playing");
    h.step();
    assert!(h.state().review_skips_unticked());
    assert!(!h.state().player().current_skips().is_empty(), "skips restored when checkbox on");

    // After Remove: nothing left to skip
    h.state_mut().perform(Action::ApplyRetakes);
    step_until(&mut h, "skips empty", |a| a.player().current_skips().is_empty());
    assert!(h.state().player().current_skips().is_empty(), "nothing left to skip after Remove");

    // Undo brings the skips back
    click(&mut h, "Undo");
    h.step();
    assert!(!h.state().player().current_skips().is_empty(), "Undo brings skips back");

    // Close review clears skips
    h.state_mut().perform(Action::CloseReview);
    h.step();
    assert!(h.state().player().current_skips().is_empty(), "closing review clears skips");
}

#[test]
fn drag_sentence_edge_with_real_mouse_updates_sentence_group_span_and_rects() {
    if skip() {
        return;
    }
    let (mut h, _) = ready(FakeService::default());
    to_review(&mut h);

    let (s_orig, e_orig) = h.state().sentence_span_f64("3:0").expect("sentence 3:0 exists");

    // Untick 3:0 so it becomes Cut
    h.state_mut().perform(Action::SetKeep { sentence: "3:0".into(), keep: false });
    h.step();

    let rects_before = h.state().review_marker_rects();
    assert!(!rects_before.is_empty(), "marker rects exist for cut sentence");
    let r3 = rects_before[0];
    let from = pos2(r3.right(), r3.center().y);
    let to = pos2(r3.right() + 15.0, r3.center().y);

    drag_mouse(&mut h, from, to);

    let (s_new, e_new) = h.state().sentence_span_f64("3:0").expect("sentence 3:0 exists");
    assert!((s_new - s_orig).abs() < 1e-4, "start stayed at {s_orig}");
    assert!(e_new > e_orig, "end was dragged later: {e_new} > {e_orig}");

    let g_span = h.state().review_group_spans()[0];
    assert!((s_of(g_span.0) - s_orig).abs() < 1e-4);

    let rects_after = h.state().review_marker_rects();
    assert!(rects_after[0].width() > r3.width(), "marker rect widened");
    assert!(h.state().is_sentence_span_edited("3:0"));
}

#[test]
fn remove_cuts_exact_edited_span_with_exact_source_offsets() {
    if skip() {
        return;
    }
    let (mut h, _) = ready(FakeService::default());
    to_review(&mut h);

    let (s_orig, _e_orig) = h.state().sentence_span_f64("3:0").expect("sentence 3:0 exists");
    // Untick 3:0 and edit its span to frame-aligned seconds (0.30s and 0.40s at 30fps)
    let new_s = 0.30;
    let new_e = 0.40;
    assert!(new_s > s_orig, "new_s is different from original start");
    h.state_mut().perform(Action::SetKeep { sentence: "3:0".into(), keep: false });
    h.state_mut().perform(Action::SetSentenceSpan { sentence: "3:0".into(), start: new_s, end: new_e });
    h.step();

    // Remove cuts 3:0
    h.state_mut().perform(Action::ApplyRetakes);
    h.step();

    let seq = h.state().editor().sequence().expect("sequence exists");
    let v_track = &seq.tracks[0];
    assert_eq!(v_track.clips.len(), 2, "video was cut into 2 pieces");

    let c0 = &v_track.clips[0];
    assert_eq!(s_of(c0.start), 0.0);
    assert!((s_of(c0.duration) - new_s).abs() < 1e-4);
    assert_eq!(s_of(c0.source_in), 0.0);

    let c1 = &v_track.clips[1];
    assert!((s_of(c1.start) - new_s).abs() < 1e-4);
    assert!((s_of(c1.source_in) - new_e).abs() < 1e-4);
    assert!((s_of(c1.duration) - (4.0 - new_e)).abs() < 1e-4);
}

#[test]
fn numeric_edit_clamping_neighbour_reset_span_and_log_content() {
    if skip() {
        return;
    }
    let (mut h, _, dir) = ready_with_script(FakeService::default());
    let log_path = dir.join("span_edits.jsonl");
    h.state_mut().set_span_log(Some(log_path.clone()));

    to_review(&mut h);

    let (s0, e0) = h.state().sentence_span_f64("3:0").unwrap();
    let (s1, _) = h.state().sentence_span_f64("4:0").unwrap();

    // Edit start of 3:0
    let target_s = s0 + 0.03;
    h.state_mut().perform(Action::SetSentenceSpan { sentence: "3:0".into(), start: target_s, end: e0 });
    h.step();
    let (s, e) = h.state().sentence_span_f64("3:0").unwrap();
    assert!((s - target_s).abs() < 1e-4);
    assert!((e - e0).abs() < 1e-4);
    assert!(h.state().is_sentence_span_edited("3:0"));

    // Clamping against neighbour 4:0 (which starts at s1)
    let past_neighbour = s1 + 0.10;
    h.state_mut().perform(Action::SetSentenceSpan { sentence: "3:0".into(), start: target_s, end: past_neighbour });
    h.step();
    let (_, e_clamped) = h.state().sentence_span_f64("3:0").unwrap();
    assert!((e_clamped - s1).abs() < 1e-4, "end clamped to neighbour start: {e_clamped} vs {s1}");

    // Reset span
    click(&mut h, "Reset span of 3:0");
    h.step();
    let (s_reset, e_reset) = h.state().sentence_span_f64("3:0").unwrap();
    assert!((s_reset - s0).abs() < 1e-4);
    assert!((e_reset - e0).abs() < 1e-4);
    assert!(!h.state().is_sentence_span_edited("3:0"));

    // Verify span_edits.jsonl content
    assert!(log_path.exists());
    let text = std::fs::read_to_string(&log_path).expect("read span log");
    let lines: Vec<serde_json::Value> = text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).unwrap()).collect();
    assert!(lines.len() >= 3, "logged the edits and reset");
    for line in &lines {
        assert_eq!(line["sentence"], "3:0");
        assert_eq!(line["source"], "user");
        assert!(line["old"]["start"].is_number());
        assert!(line["old"]["end"].is_number());
        assert!(line["new"]["start"].is_number());
        assert!(line["new"]["end"].is_number());
    }
}

#[test]
fn saved_groups_file_is_accepted_by_read_review_and_holds_edited_times() {
    if skip() {
        return;
    }
    let (mut h, script, dir) = ready_with_script(FakeService::default());
    to_review(&mut h);

    let (s0, e0) = h.state().sentence_span_f64("3:0").unwrap();
    let target_s = s0 + 0.04;
    let target_e = e0 - 0.02;

    // Edit 3:0 span
    h.state_mut().perform(Action::SetSentenceSpan { sentence: "3:0".into(), start: target_s, end: target_e });
    h.step();

    let save_path = dir.join("saved_groups.json");
    script.groups_save.lock().unwrap().push_back(save_path.clone());

    click(&mut h, "Save groups...");
    h.step();

    assert!(save_path.exists(), "groups file was written");
    let json = std::fs::read_to_string(&save_path).expect("read saved groups");

    let video = item(&h, "talk.mp4");
    let checked = h.state().editor().read_review(&json, video).expect("accepted by read_review");
    let simplecut_core::Checked::Groups(g) = checked else { panic!("not groups") };
    let s3 = g.groups[0].sentences.iter().find(|s| s.id == "3:0").expect("found 3:0 in saved file");
    assert!((s_of(s3.start) - target_s).abs() < 1e-4);
    assert!((s_of(s3.end) - target_e).abs() < 1e-4);
}

#[test]
fn sentence_edge_coinciding_with_clip_edge_wins_over_clip_trim() {
    if skip() {
        return;
    }
    let (mut h, _) = ready(FakeService::default());

    // Sentence 3:0 starts at 0.27s (or Tick(68_584_320_000))
    let t_split = Tick::from_seconds_f64(0.27);
    h.state_mut().perform(Action::SeekTimeline(t_split));
    h.state_mut().perform(Action::Split);
    h.step();

    to_review(&mut h);

    let (s0, _) = h.state().sentence_span_f64("3:0").unwrap();

    // Untick 3:0 so it is Cut
    h.state_mut().perform(Action::SetKeep { sentence: "3:0".into(), keep: false });
    h.step();

    // At s0 on the video lane, clip 1 starts AND sentence 3:0 starts.
    // Dragging at this edge must drag the sentence edge, NOT trim the clip.
    let rects = h.state().review_marker_rects();
    assert!(!rects.is_empty());
    let r3 = rects[0];
    let edge_start = pos2(r3.left(), r3.center().y);
    let initial_clip1_start = h.state().editor().sequence().unwrap().tracks[0].clips[1].start;

    drag_mouse(&mut h, edge_start, edge_start + vec2(15.0, 0.0));

    // Sentence 3:0 start was updated
    let (s_new, _) = h.state().sentence_span_f64("3:0").unwrap();
    assert!(s_new > s0, "sentence start changed: {s_new} > {s0}");

    // Clip 1 start on the timeline was NOT trimmed
    let seq = h.state().editor().sequence().unwrap();
    let clip1 = &seq.tracks[0].clips[1];
    assert_eq!(clip1.start, initial_clip1_start, "clip start was NOT trimmed");

    // But clip trimming still works elsewhere:
    // Drag the end of clip 1 (at 4.0s)
    let clip1_rect = h.state().timeline_clip_rect(clip1.id).expect("clip rect");
    let clip_end_pos = pos2(clip1_rect.right(), clip1_rect.center().y);
    drag_mouse(&mut h, clip_end_pos, clip_end_pos - vec2(20.0, 0.0));

    let seq_after = h.state().editor().sequence().unwrap();
    let clip1_after = &seq_after.tracks[0].clips[1];
    assert!(clip1_after.end() < clip1.end(), "clip trim still works where no sentence edge is");
}

#[test]
fn rejected_span_edits_are_not_logged_and_clamped_ones_log_the_stored_value() {
    if skip() {
        return;
    }
    let (mut h, _, dir) = ready_with_script(FakeService::default());
    let log_path = dir.join("span_edits.jsonl");
    h.state_mut().set_span_log(Some(log_path.clone()));
    to_review(&mut h);
    let (s0, e0) = h.state().sentence_span_f64("3:0").unwrap();
    let (s1, _) = h.state().sentence_span_f64("4:0").unwrap();
    let lines = |p: &std::path::Path| std::fs::read_to_string(p).map(|t| t.lines().filter(|l| !l.trim().is_empty()).count()).unwrap_or(0);

    // reversed, not a number and past the file are refused: nothing changes, nothing is logged, the user is told
    for (start, end) in [(e0, s0), (f64::NAN, e0), (s0, 1.0e9)] {
        h.state_mut().perform(Action::SetSentenceSpan { sentence: "3:0".into(), start, end });
        h.step();
        assert_eq!(h.state().sentence_span_f64("3:0"), Some((s0, e0)), "unchanged");
        assert_eq!(lines(&log_path), 0, "a refused edit is not an edit");
        assert!(h.state().banner().is_some_and(|b| b.error), "{:?}", h.state().banner());
    }

    // an end past the neighbour is clamped; the log holds what was stored, not what was asked
    h.state_mut().perform(Action::SetSentenceSpan { sentence: "3:0".into(), start: s0, end: s1 + 0.1 });
    h.step();
    let text = std::fs::read_to_string(&log_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert!((v["new"]["end"].as_f64().unwrap() - s1).abs() < 1e-4, "{v}");
    assert_eq!(lines(&log_path), 1);
}

#[test]
fn mode_selector_present_only_with_capability_and_shows_warnings() {
    if skip() {
        return;
    }
    // 1. Without refine capability: selector absent, flow unchanged
    let no_refine_service = FakeService { refine: false, ..Default::default() };
    let (mut h_no, _) = ready(no_refine_service);
    to_review(&mut h_no);
    assert!(!h_no.state().refine_available());
    assert!(h_no.query_by_label("Cut mode: Retakes and dead air").is_none());
    assert!(h_no.query_by_label("Cut mode: Speech only").is_none());
    assert!(h_no.query_by_label("Cut mode: Retakes only").is_none());

    // 2. With refine capability: selector present, default Anchored
    let (mut h, _) = ready(FakeService::default());
    to_review(&mut h);
    assert!(h.state().refine_available());
    assert_eq!(h.state().review_cut_mode(), Some(CutMode::Anchored));
    h.get_by_label("Cut mode: Retakes and dead air");
    h.get_by_label("Cut mode: Speech only");
    h.get_by_label("Cut mode: Retakes only");

    // Anchored and RetakeOnly have no warning
    assert!(CutMode::Anchored.warning().is_none());
    assert!(CutMode::RetakeOnly.warning().is_none());
    let sent_warn = CutMode::Sentences.warning().expect("Sentences has warning");

    // Sentences shows warning text
    click(&mut h, "Cut mode: Speech only");
    assert_eq!(h.state().review_cut_mode(), Some(CutMode::Sentences));
    h.get_by_label(sent_warn);

    // Switching back to Anchored hides warning
    click(&mut h, "Cut mode: Retakes and dead air");
    assert_eq!(h.state().review_cut_mode(), Some(CutMode::Anchored));
    assert!(h.query_by_label(sent_warn).is_none());
}

#[test]
fn remove_in_each_mode_cuts_exactly_planned_ranges_and_ordering() {
    if skip() {
        return;
    }
    let mut durations_removed = Vec::new();

    for mode in [CutMode::RetakeOnly, CutMode::Anchored, CutMode::Sentences] {
        let (mut h, _) = ready(FakeService::default());
        to_review(&mut h);
        let before = video_total(&h);

        // Set mode
        h.state_mut().perform(Action::SetCutMode(mode));
        assert_eq!(h.state().review_cut_mode(), Some(mode));

        // Untick span_0
        h.state_mut().perform(Action::KeepLastOnly("span_0".into()));
        h.step();

        // Wait for debounce refine so review_planned_ranges() holds the refined plan
        let initial_planned = h.state().review_planned_ranges();
        std::thread::sleep(std::time::Duration::from_millis(600));
        step_until(&mut h, "debounce refined", |a| a.review_planned_ranges() != initial_planned);
        h.step();

        let planned_ranges = h.state().review_planned_ranges();
        assert!(!planned_ranges.is_empty(), "has planned ranges");

        // Apply retakes (uses the cached refined plan)
        h.state_mut().perform(Action::ApplyRetakes);
        step_until(&mut h, "applied", |a| a.banner().is_some_and(|b| !b.error && b.text.contains("Removed")));

        let after = video_total(&h);
        let removed = before - after;
        durations_removed.push(removed);

        let seq = h.state().editor().sequence().expect("sequence exists");
        let v_track = &seq.tracks[0];
        let total_clip_dur: f64 = v_track.clips.iter().map(|c| s_of(c.duration)).sum();
        assert!((total_clip_dur - after).abs() < 1e-4);

        // Assert that the removed ranges on the timeline match review_planned_ranges()
        let mut actual_removed: Vec<(f64, f64)> = Vec::new();
        if s_of(v_track.clips[0].source_in) > 1e-4 {
            actual_removed.push((0.0, s_of(v_track.clips[0].source_in)));
        }
        for w in v_track.clips.windows(2) {
            let cut_start = s_of(w[0].source_in) + s_of(w[0].duration);
            let cut_end = s_of(w[1].source_in);
            actual_removed.push((cut_start, cut_end));
        }
        let last_clip = &v_track.clips[v_track.clips.len() - 1];
        if s_of(last_clip.source_in) + s_of(last_clip.duration) < 4.0 - 1e-4 {
            actual_removed.push((s_of(last_clip.source_in) + s_of(last_clip.duration), 4.0));
        }
        let planned_secs: Vec<(f64, f64)> = planned_ranges.iter().map(|(s, e)| (s_of(*s), s_of(*e))).collect();
        assert_eq!(actual_removed.len(), planned_secs.len(), "cut pieces count matches planned ranges");
        for (act, exp) in actual_removed.iter().zip(&planned_secs) {
            assert!((act.0 - exp.0).abs() < 0.035, "cut start matches planned start: {} vs {}", act.0, exp.0);
            assert!((act.1 - exp.1).abs() < 0.035, "cut end matches planned end: {} vs {}", act.1, exp.1);
        }

        match mode {
            CutMode::RetakeOnly => {
                assert_eq!(s_of(v_track.clips[0].source_in), 0.0);
                assert_eq!(v_track.clips.len(), 3);
            }
            CutMode::Anchored => {
                assert_eq!(s_of(v_track.clips[0].source_in), 0.0);
                assert_eq!(v_track.clips.len(), 2);
            }
            CutMode::Sentences => {
                assert!(s_of(v_track.clips[0].source_in) > 0.0);
                assert_eq!(v_track.clips.len(), 1);
            }
        }
    }

    // the three modes remove strictly more in the order retake_only < anchored < sentences
    assert!(durations_removed[0] < durations_removed[1], "RetakeOnly ({}) should remove less than Anchored ({})", durations_removed[0], durations_removed[1]);
    assert!(durations_removed[1] < durations_removed[2], "Anchored ({}) should remove less than Sentences ({})", durations_removed[1], durations_removed[2]);
}

#[test]
fn bad_refine_cuts_inside_allowed_bounds_notes_in_info_and_one_undo() {
    if skip() {
        return;
    }
    let service = FakeService { bad_refine: true, ..Default::default() };
    let (mut h, _) = ready(service);
    to_review(&mut h);
    let steps = h.state().editor().session().history.undo.len();
    let before = video_total(&h);

    // Get spans of sentences 3:0 and 4:0 before removing them
    let (s3, e3) = h.state().sentence_span_f64("3:0").expect("span 3:0");
    let (s4, e4) = h.state().sentence_span_f64("4:0").expect("span 4:0");
    let widen = simplecut_core::WIDEN_SECS;
    let allowed_bounds = [(s3 - widen, e3 + widen), (s4 - widen, e4 + widen)];

    h.state_mut().perform(Action::KeepLastOnly("span_0".into()));
    h.step();

    h.state_mut().perform(Action::ApplyRetakes);
    step_until(&mut h, "applied", |a| a.banner().is_some_and(|b| !b.error && b.text.contains("Removed")));

    // Remove still cuts
    assert!(video_total(&h) < before);
    // Exactly one undo step
    assert_eq!(h.state().editor().session().history.undo.len(), steps + 1);
    // Info text carries the note
    let banner = h.state().banner().expect("banner present");
    assert!(!banner.error);
    assert!(banner.text.contains("whole sentences are cut instead") || banner.text.contains("could not be used"));

    // Assert that every removed piece lies inside some removed sentence's span widened by WIDEN_SECS
    let seq = h.state().editor().sequence().expect("sequence exists");
    let v_track = &seq.tracks[0];
    let mut actual_removed: Vec<(f64, f64)> = Vec::new();
    if s_of(v_track.clips[0].source_in) > 1e-4 {
        actual_removed.push((0.0, s_of(v_track.clips[0].source_in)));
    }
    for w in v_track.clips.windows(2) {
        let cut_start = s_of(w[0].source_in) + s_of(w[0].duration);
        let cut_end = s_of(w[1].source_in);
        actual_removed.push((cut_start, cut_end));
    }
    let last_clip = &v_track.clips[v_track.clips.len() - 1];
    if s_of(last_clip.source_in) + s_of(last_clip.duration) < 4.0 - 1e-4 {
        actual_removed.push((s_of(last_clip.source_in) + s_of(last_clip.duration), 4.0));
    }
    assert!(!actual_removed.is_empty(), "some cuts were applied");
    for (cs, ce) in &actual_removed {
        let inside = allowed_bounds.iter().any(|(b_start, b_end)| *cs >= *b_start - 1e-3 && *ce <= *b_end + 1e-3);
        assert!(inside, "removed piece [{cs}, {ce}] must lie inside widened sentence span bounds: {allowed_bounds:?}");
    }
}

#[test]
fn hand_edited_sentence_span_is_cut_exactly_as_set_in_every_mode() {
    if skip() {
        return;
    }
    for mode in [CutMode::RetakeOnly, CutMode::Anchored, CutMode::Sentences] {
        // Case 1: single sentence removed and hand-edited
        let (mut h, _) = ready(FakeService::default());
        to_review(&mut h);

        h.state_mut().perform(Action::SetCutMode(mode));
        let (s_orig, _) = h.state().sentence_span_f64("3:0").expect("sentence 3:0 exists");
        let new_s = 0.30;
        let new_e = 0.40;
        assert!(new_s > s_orig);

        h.state_mut().perform(Action::SetKeep { sentence: "3:0".into(), keep: false });
        h.state_mut().perform(Action::SetSentenceSpan { sentence: "3:0".into(), start: new_s, end: new_e });
        h.step();

        h.state_mut().perform(Action::ApplyRetakes);
        h.step();

        let seq = h.state().editor().sequence().expect("sequence exists");
        let v_track = &seq.tracks[0];
        assert_eq!(v_track.clips.len(), 2, "video was cut into 2 pieces in mode {mode:?}");

        let c0 = &v_track.clips[0];
        assert_eq!(s_of(c0.start), 0.0);
        assert!((s_of(c0.duration) - new_s).abs() < 1e-4);

        let c1 = &v_track.clips[1];
        assert!((s_of(c1.start) - new_s).abs() < 1e-4);
        assert!((s_of(c1.source_in) - new_e).abs() < 1e-4);
        assert!((s_of(c1.duration) - (4.0 - new_e)).abs() < 1e-4);

        // Case 2: TWO removed sentences, one edited and one not. Refine is called for the unedited one.
        let count = Arc::new(AtomicUsize::new(0));
        let last_req = Arc::new(Mutex::new(None));
        let service = CountingService { inner: FakeService::default(), refine_count: count.clone(), last_request: last_req.clone() };
        let (mut h2, _) = ready_any(Box::new(service));
        to_review(&mut h2);

        h2.state_mut().perform(Action::SetCutMode(mode));
        let new_s2 = 0.30;
        let new_e2 = 0.40;
        // Untick both 3:0 (in group 0) and 9:0 (in group 1)
        h2.state_mut().perform(Action::SetKeep { sentence: "3:0".into(), keep: false });
        h2.state_mut().perform(Action::SetKeep { sentence: "9:0".into(), keep: false });
        // Only 3:0 is edited
        h2.state_mut().perform(Action::SetSentenceSpan { sentence: "3:0".into(), start: new_s2, end: new_e2 });
        h2.step();

        // Wait for debounce so review_planned_ranges holds the refined plan
        let initial_planned = h2.state().review_planned_ranges();
        std::thread::sleep(std::time::Duration::from_millis(600));
        step_until(&mut h2, "debounce refined 2", |a| a.review_planned_ranges() != initial_planned);
        h2.step();

        // Get planned ranges just before Remove
        let planned_ranges = h2.state().review_planned_ranges();

        h2.state_mut().perform(Action::ApplyRetakes);
        step_until(&mut h2, "applied case 2", |a| a.banner().is_some_and(|b| !b.error && b.text.contains("Removed")));

        // Assert refine request carried only the unedited id "9:0"
        let req = last_req.lock().unwrap().clone().expect("refine request recorded");
        assert_eq!(req.remove, vec!["9:0"], "refine carried only unedited sentence id in mode {mode:?}");

        // Assert the cuts on the timeline:
        let seq2 = h2.state().editor().sequence().expect("sequence exists");
        let v_track2 = &seq2.tracks[0];
        let mut actual_removed: Vec<(f64, f64)> = Vec::new();
        if s_of(v_track2.clips[0].source_in) > 1e-4 {
            actual_removed.push((0.0, s_of(v_track2.clips[0].source_in)));
        }
        for w in v_track2.clips.windows(2) {
            let cut_start = s_of(w[0].source_in) + s_of(w[0].duration);
            let cut_end = s_of(w[1].source_in);
            actual_removed.push((cut_start, cut_end));
        }
        let last_clip = &v_track2.clips[v_track2.clips.len() - 1];
        if s_of(last_clip.source_in) + s_of(last_clip.duration) < 4.0 - 1e-4 {
            actual_removed.push((s_of(last_clip.source_in) + s_of(last_clip.duration), 4.0));
        }

        // First cut is the edited sentence [new_s2, new_e2] cut exactly as set
        assert!((actual_removed[0].0 - new_s2).abs() < 1e-3, "first cut start matches edited start: {} vs {}", actual_removed[0].0, new_s2);
        assert!((actual_removed[0].1 - new_e2).abs() < 1e-3, "first cut end matches edited end: {} vs {}", actual_removed[0].1, new_e2);

        // Remaining cut is refined range for 9:0
        let planned_secs: Vec<(f64, f64)> = planned_ranges.iter().map(|(s, e)| (s_of(*s), s_of(*e))).collect();
        assert_eq!(actual_removed.len(), planned_secs.len());
        for (act, exp) in actual_removed.iter().zip(&planned_secs) {
            assert!((act.0 - exp.0).abs() < 0.035, "cut start matches planned: {} vs {}", act.0, exp.0);
            assert!((act.1 - exp.1).abs() < 0.035, "cut end matches planned: {} vs {}", act.1, exp.1);
        }
    }
}

#[test]
fn cancel_during_refining_leaves_timeline_and_undo_stack_unchanged() {
    if skip() {
        return;
    }
    let (mut h, _) = ready(FakeService::default());
    to_review(&mut h);
    let before_len = video_total(&h);
    let steps = h.state().editor().session().history.undo.len();

    h.state_mut().perform(Action::KeepLastOnly("span_0".into()));
    h.step();

    // Start refining
    h.state_mut().perform(Action::ApplyRetakes);
    h.step();
    assert!(h.state().review_refining());

    // Cancel while refining
    h.state_mut().perform(Action::CancelAnalysis);
    h.step();
    assert!(!h.state().review_refining());

    // Timeline and undo stack are untouched
    assert_eq!(video_total(&h), before_len);
    assert_eq!(h.state().editor().session().history.undo.len(), steps);
}

#[test]
fn changing_tick_or_mode_while_refine_in_flight_never_applies_old_plan() {
    if skip() {
        return;
    }
    // Variant 1: change mode while refine is in flight
    let (mut h, _) = ready(FakeService::default());
    to_review(&mut h);
    let before_len = video_total(&h);
    let steps = h.state().editor().session().history.undo.len();

    // Untick sentences of span_0
    h.state_mut().perform(Action::KeepLastOnly("span_0".into()));
    h.step();

    // Start refine
    h.state_mut().perform(Action::ApplyRetakes);
    h.step();
    assert!(h.state().review_refining());

    // Change mode while refine is in flight
    h.state_mut().perform(Action::SetCutMode(CutMode::RetakeOnly));
    h.step();

    // Wait until background refine finishes
    step_until(&mut h, "refine settled", |a| !a.review_refining());

    // The old plan must NOT have been applied to the timeline
    assert_eq!(video_total(&h), before_len);
    assert_eq!(h.state().editor().session().history.undo.len(), steps);

    // Variant 2: change tick while refine is in flight
    let (mut h2, _) = ready(FakeService::default());
    to_review(&mut h2);
    let before_len2 = video_total(&h2);
    let steps2 = h2.state().editor().session().history.undo.len();

    h2.state_mut().perform(Action::KeepLastOnly("span_0".into()));
    h2.step();

    h2.state_mut().perform(Action::ApplyRetakes);
    h2.step();
    assert!(h2.state().review_refining());

    // Change tick while refine is in flight
    h2.state_mut().perform(Action::SetKeep { sentence: "3:0".into(), keep: true });
    h2.step();

    // Wait until background refine finishes
    step_until(&mut h2, "refine settled 2", |a| !a.review_refining());

    assert_eq!(video_total(&h2), before_len2);
    assert_eq!(h2.state().editor().session().history.undo.len(), steps2);

    let banner = h2.state().banner().expect("info banner shown");
    assert!(!banner.error);
    assert_eq!(banner.text, "The selection changed while the cuts were being refined. Press Remove again.");
}

#[test]
fn refine_job_fails_shows_message_nothing_cut_cut_whole_sentences_works() {
    if skip() {
        return;
    }
    let service = FakeService { fail_refine: Some("Audio refine service crashed".into()), ..Default::default() };
    let (mut h, _) = ready(service);
    to_review(&mut h);
    let before_len = video_total(&h);
    let steps = h.state().editor().session().history.undo.len();

    h.state_mut().perform(Action::KeepLastOnly("span_0".into()));
    h.step();

    // Remove starts refine, which will fail
    h.state_mut().perform(Action::ApplyRetakes);
    step_until(&mut h, "failed", |a| a.banner().is_some_and(|b| b.error));

    // Message is shown
    let banner = h.state().banner().expect("banner");
    assert!(banner.text.contains("Audio refine service crashed"));
    assert!(banner.error);
    assert_eq!(banner.action.as_ref().map(|(l, _)| l.as_str()), Some("Cut whole sentences instead"));

    // Nothing was cut yet
    assert_eq!(video_total(&h), before_len);
    assert_eq!(h.state().editor().session().history.undo.len(), steps);

    // "Cut whole sentences instead" works
    click(&mut h, "Cut whole sentences instead");
    step_until(&mut h, "cut applied", |a| a.banner().is_some_and(|b| !b.error && b.text.contains("Removed")));

    // Video was cut, banner is info
    assert!(video_total(&h) < before_len);
    assert_eq!(h.state().editor().session().history.undo.len(), steps + 1);
}

#[test]
fn preview_before_and_after_debounce_matches_planned_ranges_and_rects() {
    if skip() {
        return;
    }
    let (mut h, _) = ready(FakeService::default());
    to_review(&mut h);

    let item = h.state().review_item().expect("review item exists");

    // Untick sentences of span_0 (sentences 3:0 and 4:0 removed, 5:0 kept)
    h.state_mut().perform(Action::KeepLastOnly("span_0".into()));
    h.step();

    // Before reply/debounce: skips equal whole sentence spans mapped to timeline
    let s3 = h.state().review_sentence_span("3:0").expect("span 3:0");
    let s4 = h.state().review_sentence_span("4:0").expect("span 4:0");
    let mut expected_before = Vec::new();
    for (s, e) in [s3, s4] {
        expected_before.extend(h.state().editor().timeline_ranges_of(item, s, e));
    }
    expected_before.sort_by_key(|r| r.0);
    let mut merged_before: Vec<(Tick, Tick)> = Vec::new();
    for (s, e) in expected_before {
        if let Some(last) = merged_before.last_mut()
            && s <= last.1
        {
            last.1 = last.1.max(e);
            continue;
        }
        merged_before.push((s, e));
    }
    assert_eq!(h.state().player().current_skips(), merged_before.as_slice(), "before reply: skips equal sentence spans mapped to timeline");

    let planned_before = h.state().review_planned_ranges();

    // Sleep 600ms so debounce fires, then step_until refine completes
    std::thread::sleep(std::time::Duration::from_millis(600));
    step_until(&mut h, "debounce refined", |a| {
        let planned = a.review_planned_ranges();
        planned != planned_before
    });
    h.step();

    // After debounce: player current_skips equals planned ranges mapped to timeline, merged and sorted
    let planned_after = h.state().review_planned_ranges();
    let mut expected_after = Vec::new();
    for (s, e) in &planned_after {
        expected_after.extend(h.state().editor().timeline_ranges_of(item, *s, *e));
    }
    expected_after.sort_by_key(|r| r.0);
    let mut merged_after: Vec<(Tick, Tick)> = Vec::new();
    for (s, e) in expected_after {
        if let Some(last) = merged_after.last_mut()
            && s <= last.1
        {
            last.1 = last.1.max(e);
            continue;
        }
        merged_after.push((s, e));
    }
    assert_eq!(h.state().player().current_skips(), merged_after.as_slice(), "after reply: skips equal planned ranges mapped, merged, and sorted");

    // review_marker_rects() has one rect per planned range on the timeline and x-extent matches
    let rects = h.state().review_marker_rects();
    let mut timeline_ranges = Vec::new();
    for (s, e) in &planned_after {
        timeline_ranges.extend(h.state().editor().timeline_ranges_of(item, *s, *e));
    }
    assert_eq!(rects.len(), timeline_ranges.len(), "one marker rect per planned range on timeline");
    for (rect, (s, e)) in rects.iter().zip(&timeline_ranges) {
        let x0 = h.state().timeline_x(*s).expect("x0 from timeline hook");
        let x1 = h.state().timeline_x(*e).expect("x1 from timeline hook");
        assert!((rect.min.x - x0).abs() < 1.0, "rect min.x ({}) matches timeline_x start ({})", rect.min.x, x0);
        assert!((rect.max.x - x1).abs() < 1.0, "rect max.x ({}) matches timeline_x end ({})", rect.max.x, x1);
    }
}

#[test]
fn refine_remove_is_one_undo_step_and_undo_redo_with_review_open_works() {
    if skip() {
        return;
    }
    let (mut h, _) = ready(FakeService::default());
    to_review(&mut h);
    let before_len = video_total(&h);
    let groups_count = h.state().review_groups().len();
    let steps = h.state().editor().session().history.undo.len();

    h.state_mut().perform(Action::KeepLastOnly("span_0".into()));
    h.step();

    // Apply retakes with refine
    h.state_mut().perform(Action::ApplyRetakes);
    step_until(&mut h, "removed", |a| a.review_groups().len() == groups_count - 1);

    // Exactly one undo step
    assert_eq!(h.state().editor().session().history.undo.len(), steps + 1);
    let cut_len = video_total(&h);
    assert!(cut_len < before_len);

    // Undo brings it back
    click(&mut h, "Undo");
    h.step();
    assert!((video_total(&h) - before_len).abs() < 1e-3);
    assert_eq!(h.state().review_groups().len(), groups_count);
    assert_eq!(h.state().review_stage(), "review");

    // Redo re-applies the cuts
    click(&mut h, "Redo");
    h.step();
    assert!((video_total(&h) - cut_len).abs() < 1e-3);
    assert_eq!(h.state().review_groups().len(), groups_count - 1);
    assert_eq!(h.state().review_stage(), "review");
}

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

struct CountingService {
    inner: FakeService,
    refine_count: Arc<AtomicUsize>,
    last_request: Arc<Mutex<Option<simplecut_core::RefineRequest>>>,
}

impl simplecut_core::RetakeService for CountingService {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn capabilities(&self) -> simplecut_core::Capabilities {
        self.inner.capabilities()
    }
    fn analyze(&self, request: simplecut_core::AnalysisRequest) -> Box<dyn simplecut_core::AnalysisJob> {
        self.inner.analyze(request)
    }
    fn resolve(&self, request: simplecut_core::ResolveRequest) -> Box<dyn simplecut_core::AnalysisJob> {
        self.inner.resolve(request)
    }
    fn adjudicate(&self, request: simplecut_core::AdjudicateRequest) -> Option<Box<dyn simplecut_core::AnalysisJob>> {
        self.inner.adjudicate(request)
    }
    fn refine(&self, request: simplecut_core::RefineRequest) -> Option<Box<dyn simplecut_core::AnalysisJob>> {
        self.refine_count.fetch_add(1, Ordering::SeqCst);
        *self.last_request.lock().unwrap() = Some(request.clone());
        self.inner.refine(request)
    }
}

fn ready_any(service: Box<dyn simplecut_core::RetakeService>) -> (Win, std::path::PathBuf) {
    let dir = common::scratch("window-retakes-counting");
    let script = Script::default();
    script.media.lock().unwrap().push_back(vec![common::make_clip(&dir, "talk.mp4", 4), common::make_music(&dir, "bed.wav", 6)]);
    let mut h = window_with(script.clone(), Some(service));
    click(&mut h, "Import");
    step_until(&mut h, "the import", |a| a.editor().media().len() == 2);
    let video = item(&h, "talk.mp4");
    h.state_mut().perform(Action::AddToTimeline(video, Lane::Main));
    h.state_mut().perform(Action::SelectClips { clips: vec![], add: false });
    h.step();
    h.step();
    (h, dir)
}

#[test]
fn failing_refine_is_not_retried_in_a_loop_when_idle() {
    if skip() {
        return;
    }
    let count = Arc::new(AtomicUsize::new(0));
    let last_req = Arc::new(Mutex::new(None));
    let service = CountingService {
        inner: FakeService { fail_refine: Some("Model failed".into()), ..Default::default() },
        refine_count: count.clone(),
        last_request: last_req.clone(),
    };
    let (mut h, _) = ready_any(Box::new(service));
    to_review(&mut h);

    // Untick sentences of span_0 to cause refine to be needed
    h.state_mut().perform(Action::KeepLastOnly("span_0".into()));
    h.step();

    // Idle for 3 seconds of frames
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(3000);
    while std::time::Instant::now() < deadline {
        h.step();
        std::thread::sleep(std::time::Duration::from_millis(16));
    }

    assert!(count.load(Ordering::SeqCst) <= 1, "failing refine retried in a loop: got {} calls", count.load(Ordering::SeqCst));
}

#[test]
fn successful_refine_is_not_requested_again_when_idle() {
    if skip() {
        return;
    }
    let count = Arc::new(AtomicUsize::new(0));
    let last_req = Arc::new(Mutex::new(None));
    let service =
        CountingService { inner: FakeService { fail_refine: None, ..Default::default() }, refine_count: count.clone(), last_request: last_req.clone() };
    let (mut h, _) = ready_any(Box::new(service));
    to_review(&mut h);

    // Untick sentences of span_0 to cause refine to be needed
    h.state_mut().perform(Action::KeepLastOnly("span_0".into()));
    h.step();

    // Idle for 3 seconds of frames
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(3000);
    while std::time::Instant::now() < deadline {
        h.step();
        std::thread::sleep(std::time::Duration::from_millis(16));
    }

    assert_eq!(count.load(Ordering::SeqCst), 1, "successful refine requested multiple times: got {} calls", count.load(Ordering::SeqCst));
}

#[test]
fn failing_refine_retries_once_when_tick_changes() {
    if skip() {
        return;
    }
    let count = Arc::new(AtomicUsize::new(0));
    let last_req = Arc::new(Mutex::new(None));
    let service = CountingService {
        inner: FakeService { fail_refine: Some("Model failed".into()), ..Default::default() },
        refine_count: count.clone(),
        last_request: last_req.clone(),
    };
    let (mut h, _) = ready_any(Box::new(service));
    to_review(&mut h);

    // Untick sentences of span_0 to cause refine to be needed
    h.state_mut().perform(Action::KeepLastOnly("span_0".into()));
    h.step();

    // Wait until the first request completes (fails)
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(1500);
    while std::time::Instant::now() < deadline {
        h.step();
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
    assert_eq!(count.load(Ordering::SeqCst), 1);

    // Change a tick: untick sentence "18:0" in another group
    h.state_mut().perform(Action::SetKeep { sentence: "18:0".into(), keep: false });
    h.step();

    // Idle for 2 seconds
    let deadline2 = std::time::Instant::now() + std::time::Duration::from_millis(2000);
    while std::time::Instant::now() < deadline2 {
        h.step();
        std::thread::sleep(std::time::Duration::from_millis(16));
    }

    // Exactly one more request for the new key, not a loop
    assert_eq!(count.load(Ordering::SeqCst), 2, "changing tick caused {} refine calls, expected 2", count.load(Ordering::SeqCst));
}

#[test]
fn failing_refine_retries_once_when_remove_pressed_and_shows_banner() {
    if skip() {
        return;
    }
    let count = Arc::new(AtomicUsize::new(0));
    let last_req = Arc::new(Mutex::new(None));
    let service = CountingService {
        inner: FakeService { fail_refine: Some("Model failed".into()), ..Default::default() },
        refine_count: count.clone(),
        last_request: last_req.clone(),
    };
    let (mut h, _) = ready_any(Box::new(service));
    to_review(&mut h);

    // Untick sentences of span_0
    h.state_mut().perform(Action::KeepLastOnly("span_0".into()));
    h.step();

    // Wait for the background request to fail silently
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(1500);
    while std::time::Instant::now() < deadline {
        h.step();
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert!(h.state().banner().is_none_or(|b| !b.error), "background failure must stay silent (no error banner)");

    // User clicks Remove
    h.state_mut().perform(Action::ApplyRetakes);
    step_until(&mut h, "failed with banner", |a| a.banner().is_some_and(|b| b.error));

    // Exactly one more request was made (total 2)
    assert_eq!(count.load(Ordering::SeqCst), 2);

    let banner = h.state().banner().expect("banner");
    assert!(banner.text.contains("Model failed"));
    assert_eq!(banner.action.as_ref().map(|(l, _)| l.as_str()), Some("Cut whole sentences instead"));
}

// ── keeper choice ──────────────────────────────────────────────────────────

struct DoneJob(Option<simplecut_core::JobState>);

impl simplecut_core::AnalysisJob for DoneJob {
    fn poll(&mut self) -> simplecut_core::JobState {
        self.0.take().unwrap_or(simplecut_core::JobState::Cancelled)
    }
    fn cancel(&mut self) {}
}

/// A stand-in service with a keeper choice. Its AI answers: keep the last sentence of the first
/// group asked about, say so is unclear, and name the keeper that answered.
struct KeeperService {
    inner: FakeService,
    current: Mutex<String>,
    asked: Arc<Mutex<Vec<String>>>,
}

impl KeeperService {
    fn info(&self) -> simplecut_core::KeeperInfo {
        let current = self.current.lock().unwrap().clone();
        let p = |id: &str, available: bool, reason: &str| simplecut_core::KeeperProvider { id: id.into(), available, reason: reason.into() };
        simplecut_core::KeeperInfo {
            current,
            default: "finetuned".into(),
            providers: vec![p("finetuned", true, "local model"), p("gpt-oss", false, "no LLM key is configured"), p("keep-last", true, "always available")],
        }
    }
}

impl simplecut_core::RetakeService for KeeperService {
    fn name(&self) -> String {
        self.inner.name()
    }
    fn capabilities(&self) -> simplecut_core::Capabilities {
        self.inner.capabilities()
    }
    fn analyze(&self, request: simplecut_core::AnalysisRequest) -> Box<dyn simplecut_core::AnalysisJob> {
        self.inner.analyze(request)
    }
    fn resolve(&self, request: simplecut_core::ResolveRequest) -> Box<dyn simplecut_core::AnalysisJob> {
        self.inner.resolve(request)
    }
    fn keeper(&self) -> Option<simplecut_core::KeeperInfo> {
        Some(self.info())
    }
    fn set_keeper(&self, id: &str) -> Option<Box<dyn simplecut_core::AnalysisJob>> {
        let ok = self.info().provider(id).is_some_and(|p| p.available);
        let state = if ok {
            *self.current.lock().unwrap() = id.to_string();
            simplecut_core::JobState::Done("{}".into())
        } else {
            simplecut_core::JobState::Failed(format!("Keeper '{id}' is not available."))
        };
        Some(Box::new(DoneJob(Some(state))))
    }
    fn adjudicate(&self, request: simplecut_core::AdjudicateRequest) -> Option<Box<dyn simplecut_core::AnalysisJob>> {
        let current = self.current.lock().unwrap().clone();
        self.asked.lock().unwrap().push(current.clone());
        let g = request.groups.first()?;
        let last = g.sentences.last()?.id.clone();
        let reply = format!(r#"[{{"group":"{}","keep":["{last}"],"reason":"r","provider":"{current}","decided_by":"empty_fallback","unclear":true}}]"#, g.id);
        Some(Box::new(DoneJob(Some(simplecut_core::JobState::Done(reply)))))
    }
}

fn keeper_window(current: &str) -> (Win, Arc<Mutex<Vec<String>>>) {
    let asked = Arc::new(Mutex::new(Vec::new()));
    let service = KeeperService { inner: FakeService { adjudicate: true, ..Default::default() }, current: Mutex::new(current.into()), asked: asked.clone() };
    let (mut h, _) = ready_any(Box::new(service));
    to_review(&mut h);
    (h, asked)
}

#[test]
fn the_local_keeper_asks_no_consent_and_a_group_it_doubts_is_flagged_for_a_check() {
    if skip() {
        return;
    }
    let (mut h, asked) = keeper_window("finetuned");
    h.get_by_label("Keeper");
    click(&mut h, "Ask AI to choose");
    assert!(h.query_by_label("Continue").is_none(), "nothing leaves the machine, so no consent is asked");
    step_until(&mut h, "the AI", |a| !a.is_analyzing() && a.banner().is_some_and(|b| b.text.contains("AI chose")));
    assert_eq!(asked.lock().unwrap().as_slice(), ["finetuned"]);
    assert!(h.state().banner().is_some_and(|b| b.text.contains("1 group needs a check")), "{:?}", h.state().banner());
    let unclear = h.state().review_unclear();
    assert_eq!(unclear.len(), 1);
    // the suggestion is only a tick; nothing was cut
    let tagged = h.state().review_groups().iter().flat_map(|g| g.3.clone()).filter(|id| h.state().review_origin(id) == Some("ai")).count();
    assert!(tagged >= 2, "the suggestion fills in ticks");
}

#[test]
fn the_hosted_keeper_still_asks_for_consent() {
    if skip() {
        return;
    }
    let (mut h, asked) = keeper_window("gpt-oss");
    click(&mut h, "Ask AI to choose");
    h.get_by_label("Continue");
    assert!(asked.lock().unwrap().is_empty(), "nothing was asked before the user agreed");
}

#[test]
fn switching_the_keeper_works_and_a_refused_switch_says_why_and_changes_nothing() {
    if skip() {
        return;
    }
    let (mut h, _) = keeper_window("finetuned");
    h.state_mut().perform(Action::SetKeeper("keep-last".into()));
    step_until(&mut h, "the switch", |a| a.keeper_info().is_some_and(|k| k.current == "keep-last"));
    assert!(h.state().banner().is_some_and(|b| !b.error && b.text.contains("Keep last take")), "{:?}", h.state().banner());

    h.state_mut().perform(Action::SetKeeper("gpt-oss".into()));
    step_until(&mut h, "the refusal", |a| a.banner().is_some_and(|b| b.error && b.text.contains("not available")));
    assert_eq!(h.state().keeper_info().unwrap().current, "keep-last");
}

#[test]
fn a_service_that_does_not_report_keepers_shows_no_selector() {
    if skip() {
        return;
    }
    let (mut h, _) = ready(FakeService { adjudicate: true, ..Default::default() });
    to_review(&mut h);
    assert!(h.query_by_label("Keeper").is_none());
    assert!(h.state().keeper_info().is_none());
}
