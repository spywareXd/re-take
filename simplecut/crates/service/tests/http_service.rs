//! `HttpService` against a tiny scripted HTTP server on a local port: no Python needed.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use simplecut_core::{
    AdjudicateRequest, AnalysisJob, AnalysisRequest, CutMode, ItemId, JobState, LabelsRequest, RefineRequest, ResolveRequest, RetakeService, ReviewGroup,
    Sentence, Tick, review,
};
use simplecut_service::HttpService;

type Handler = dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync;
type Seen = Arc<Mutex<Vec<(String, String, String)>>>;

struct Mock {
    url: String,
    seen: Seen,
    stop: Arc<AtomicBool>,
}

impl Drop for Mock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.url.trim_start_matches("http://"));
    }
}

fn mock(handler: impl Fn(&str, &str, &str) -> (u16, String) + Send + Sync + 'static) -> Mock {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let seen: Seen = Arc::default();
    let stop = Arc::new(AtomicBool::new(false));
    let handler: Arc<Handler> = Arc::new(handler);
    let (thread_seen, thread_stop) = (Arc::clone(&seen), Arc::clone(&stop));
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            if thread_stop.load(Ordering::SeqCst) {
                return;
            }
            let Ok(stream) = stream else { continue };
            let (handler, seen) = (Arc::clone(&handler), Arc::clone(&thread_seen));
            std::thread::spawn(move || serve(stream, &*handler, &seen));
        }
    });
    Mock { url, seen, stop }
}

fn serve(mut stream: TcpStream, handler: &Handler, seen: &Seen) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let (head_end, length) = loop {
        let n = stream.read(&mut chunk).unwrap_or(0);
        if n == 0 {
            return;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..pos]).to_ascii_lowercase();
            let length = head.lines().find_map(|l| l.strip_prefix("content-length:")).and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0);
            break (pos + 4, length);
        }
    };
    while buf.len() < head_end + length {
        let n = stream.read(&mut chunk).unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut first = head.lines().next().unwrap_or("").split_whitespace();
    let (method, path) = (first.next().unwrap_or("").to_string(), first.next().unwrap_or("").to_string());
    let body = String::from_utf8_lossy(&buf[head_end..]).to_string();
    seen.lock().unwrap().push((method.clone(), path.clone(), body.clone()));
    let (status, reply) = handler(&method, &path, &body);
    let text = format!("HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}", reply.len());
    let _ = stream.write_all(text.as_bytes());
}

fn request() -> AnalysisRequest {
    AnalysisRequest {
        item: ItemId(7),
        name: "take.mp4".into(),
        path: "C:/clips/take.mp4".into(),
        duration: Tick(254_016_000_000 * 20),
        fingerprint: Some("ab".repeat(32)),
        used: vec![(Tick(0), Tick(254_016_000_000 * 10))],
    }
}

fn labels() -> LabelsRequest {
    serde_json::from_str(r#"{"job":"j1","labels":[{"pair":"a|b","label":"retake","source":"user"}],"skipped":["c|d"]}"#).unwrap()
}

fn wait(job: &mut dyn AnalysisJob) -> JobState {
    let end = Instant::now() + Duration::from_secs(20);
    while Instant::now() < end {
        match job.poll() {
            JobState::Running { .. } => std::thread::sleep(Duration::from_millis(20)),
            other => return other,
        }
    }
    panic!("job did not finish");
}

fn happy(method: &str, path: &str, _body: &str) -> (u16, String) {
    match (method, path) {
        ("GET", "/v1/capabilities") => (200, r#"{"adjudicate":true,"refine":true}"#.into()),
        ("POST", "/v1/analyze") => (202, r#"{"job":"j1"}"#.into()),
        ("POST", "/v1/resolve") => (202, r#"{"job":"r_1"}"#.into()),
        ("GET", "/v1/jobs/j1") => {
            (200, r#"{"state":"done","fraction":1.0,"status":"Done","result":{"stage":"candidates","job":"j1","extra":{"keep":1}}}"#.into())
        }
        ("GET", "/v1/jobs/r_1") => (200, r#"{"state":"done","fraction":1.0,"status":"Done","result":{"stage":"groups"}}"#.into()),
        _ => (404, r#"{"error":"Not found."}"#.into()),
    }
}

#[test]
fn capabilities_are_read_once_at_connect() {
    let m = mock(happy);
    let s = HttpService::connect(&m.url);
    assert!(s.is_reachable());
    assert!(s.capabilities().adjudicate);
    assert!(s.name().contains(&m.url));
}

#[test]
fn capabilities_are_refreshed_by_every_request() {
    let asked = Arc::new(Mutex::new(0));
    let a2 = Arc::clone(&asked);
    let m = mock(move |me, p, b| {
        if p == "/v1/capabilities" {
            let mut n = a2.lock().unwrap();
            *n += 1;
            // the first answer says no AI, as a service whose keys were not loaded yet would
            return (200, format!(r#"{{"adjudicate":{}}}"#, *n > 1));
        }
        happy(me, p, b)
    });
    let s = HttpService::connect(&m.url);
    assert!(!s.capabilities().adjudicate);
    let _ = wait(&mut *s.analyze(request()));
    assert!(s.capabilities().adjudicate);
}

fn group() -> ReviewGroup {
    let s = |id: &str, a: i64, b: i64, text: &str| Sentence {
        id: id.into(),
        start: Tick(a * 254_016_000_000),
        end: Tick(b * 254_016_000_000),
        text: text.into(),
        confidence: None,
    };
    ReviewGroup {
        id: "g1".into(),
        start: Tick(0),
        end: Tick(9 * 254_016_000_000),
        likely: true,
        sentences: vec![s("s1", 0, 4, "Welcome back."), s("s2", 5, 9, "Welcome back everyone.")],
    }
}

#[test]
fn adjudicate_sends_the_groups_with_their_text_and_returns_the_advice_array() {
    let m = mock(|me, p, b| match (me, p) {
        ("POST", "/v1/adjudicate") => (202, r#"{"job":"a1"}"#.into()),
        ("GET", "/v1/jobs/a1") => (200, r#"{"state":"done","fraction":1.0,"result":[{"group":"g1","keep":["s2"],"reason":"cleaner"}]}"#.into()),
        _ => happy(me, p, b),
    });
    let s = HttpService::connect(&m.url);
    let _ = wait(&mut *s.analyze(request()));
    let req = AdjudicateRequest { item: ItemId(7), job: "j1".into(), groups: vec![group()] };
    let JobState::Done(text) = wait(&mut *s.adjudicate(req).expect("the service offers advice")) else { panic!("should be done") };
    let (advice, notes) = review::parse_advice(&text, &[group()]).unwrap();
    assert!(notes.is_empty() && advice.len() == 1 && advice[0].keep == ["s2"], "{advice:?} {notes:?}");
    let seen = m.seen.lock().unwrap();
    let (_, _, body) = seen.iter().find(|(me, p, _)| me == "POST" && p == "/v1/adjudicate").unwrap();
    let sent: Value = serde_json::from_str(body).unwrap();
    assert_eq!((sent["path"].as_str(), sent["job"].as_str()), (Some("C:/clips/take.mp4"), Some("j1")));
    assert_eq!(sent["groups"][0]["sentences"][1]["text"], "Welcome back everyone.");
    assert_eq!(sent["groups"][0]["sentences"][1]["start"], 5.0);
}

#[test]
fn no_advice_when_the_service_cannot_give_it() {
    let m = mock(|me, p, b| if p == "/v1/capabilities" { (200, r#"{"adjudicate":false,"refine":true}"#.into()) } else { happy(me, p, b) });
    let s = HttpService::connect(&m.url);
    let req = AdjudicateRequest { item: ItemId(7), job: "j1".into(), groups: vec![group()] };
    assert!(s.adjudicate(req).is_none());
}

#[test]
fn advice_for_an_unknown_file_fails_without_calling_the_service() {
    let m = mock(happy);
    let s = HttpService::connect(&m.url);
    let req = AdjudicateRequest { item: ItemId(99), job: "j1".into(), groups: vec![group()] };
    assert!(matches!(wait(&mut *s.adjudicate(req).unwrap()), JobState::Failed(_)));
    assert!(m.seen.lock().unwrap().iter().all(|(_, p, _)| p != "/v1/adjudicate"));
}

fn refine_request(mode: CutMode) -> RefineRequest {
    RefineRequest { item: ItemId(7), job: "j1".into(), mode, remove: vec!["s1".into()], spans: vec![("s1".into(), Tick(0), Tick(254_016_000_000 * 4))] }
}

fn refine_service(reply: impl Fn() -> (u16, String) + Send + Sync + 'static) -> Mock {
    mock(move |me, p, b| if (me, p) == ("POST", "/v1/refine") { reply() } else { happy(me, p, b) })
}

#[test]
fn refine_sends_the_mode_and_the_ids_and_returns_the_ranges() {
    let m = refine_service(|| (200, r#"{"ranges":[{"start":0.0,"end":4.2}]}"#.into()));
    let s = HttpService::connect(&m.url);
    let _ = wait(&mut *s.analyze(request()));
    for mode in CutMode::ALL {
        let JobState::Done(text) = wait(&mut *s.refine(refine_request(mode)).expect("the service offers refine")) else { panic!("should be done") };
        assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["ranges"][0]["end"], 4.2);
    }
    let seen = m.seen.lock().unwrap();
    let sent: Vec<Value> = seen.iter().filter(|(me, p, _)| me == "POST" && p == "/v1/refine").map(|(_, _, b)| serde_json::from_str(b).unwrap()).collect();
    assert_eq!(sent.iter().map(|v| v["mode"].as_str().unwrap()).collect::<Vec<_>>(), ["anchored", "retake_only", "sentences"]);
    assert_eq!(sent[0]["path"], "C:/clips/take.mp4");
    assert_eq!(sent[0]["remove"], serde_json::json!(["s1"]));
}

#[test]
fn refine_errors_are_shown_as_the_service_wrote_them() {
    let m = refine_service(|| (400, r#"{"error":"This file has not been analysed yet. Run Remove Retakes first."}"#.into()));
    let s = HttpService::connect(&m.url);
    let _ = wait(&mut *s.analyze(request()));
    let JobState::Failed(why) = wait(&mut *s.refine(refine_request(CutMode::Anchored)).unwrap()) else { panic!() };
    assert!(why.contains("not been analysed"), "{why}");
}

#[test]
fn refine_for_an_unknown_file_fails_and_without_the_capability_there_is_none() {
    let m = mock(happy);
    let s = HttpService::connect(&m.url);
    let mut other = refine_request(CutMode::Anchored);
    other.item = ItemId(99);
    assert!(matches!(wait(&mut *s.refine(other).unwrap()), JobState::Failed(_)));
    let off = mock(|me, p, b| if p == "/v1/capabilities" { (200, r#"{"adjudicate":true,"refine":false}"#.into()) } else { happy(me, p, b) });
    assert!(HttpService::connect(&off.url).refine(refine_request(CutMode::Anchored)).is_none());
}

#[test]
fn cancelling_refine_reads_as_cancelled_at_once_and_the_late_answer_is_dropped() {
    let m = refine_service(|| {
        std::thread::sleep(Duration::from_millis(800));
        (200, r#"{"ranges":[{"start":0.0,"end":4.2}]}"#.into())
    });
    let s = HttpService::connect(&m.url);
    let _ = wait(&mut *s.analyze(request()));
    let mut job = s.refine(refine_request(CutMode::Anchored)).unwrap();
    std::thread::sleep(Duration::from_millis(200));
    assert!(matches!(job.poll(), JobState::Running { .. }));
    job.cancel();
    assert_eq!(job.poll(), JobState::Cancelled);
    std::thread::sleep(Duration::from_millis(1200));
    assert_eq!(job.poll(), JobState::Cancelled, "the answer that arrived later must not bring the job back");
}

#[test]
fn a_dead_service_is_unreachable_and_requests_fail_in_plain_words() {
    let dead = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", l.local_addr().unwrap())
    };
    let s = HttpService::connect(&dead);
    assert!(!s.is_reachable());
    assert!(!s.capabilities().adjudicate);
    let JobState::Failed(why) = wait(&mut *s.analyze(request())) else { panic!("should fail") };
    assert!(why.contains("Could not reach the retake service"), "{why}");
}

#[test]
fn analyze_sends_seconds_and_returns_the_document_text() {
    let m = mock(happy);
    let s = HttpService::connect(&m.url);
    let JobState::Done(text) = wait(&mut *s.analyze(request())) else { panic!("should be done") };
    let doc: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(doc["stage"], "candidates");
    let seen = m.seen.lock().unwrap();
    let (_, _, body) = seen.iter().find(|(me, p, _)| me == "POST" && p == "/v1/analyze").unwrap();
    let sent: Value = serde_json::from_str(body).unwrap();
    assert_eq!(sent["path"], "C:/clips/take.mp4");
    assert_eq!(sent["duration_seconds"], 20.0);
    assert_eq!(sent["used"], serde_json::json!([[0.0, 10.0]]));
    assert_eq!(sent["fingerprint"].as_str().unwrap().len(), 64);
}

#[test]
fn resolve_echoes_the_candidates_with_their_unknown_fields_and_the_path() {
    let m = mock(happy);
    let s = HttpService::connect(&m.url);
    let JobState::Done(candidates) = wait(&mut *s.analyze(request())) else { panic!() };
    let req = ResolveRequest { item: ItemId(7), labels: labels(), candidates: Some(candidates) };
    let JobState::Done(text) = wait(&mut *s.resolve(req)) else { panic!("should be done") };
    assert_eq!(serde_json::from_str::<Value>(&text).unwrap()["stage"], "groups");
    let seen = m.seen.lock().unwrap();
    let (_, _, body) = seen.iter().find(|(me, p, _)| me == "POST" && p == "/v1/resolve").unwrap();
    let sent: Value = serde_json::from_str(body).unwrap();
    assert_eq!(sent["path"], "C:/clips/take.mp4");
    assert_eq!(sent["candidates"]["extra"]["keep"], 1, "fields the editor does not model must survive");
    assert_eq!(sent["labels"]["labels"][0]["label"], "retake");
    assert_eq!(sent["labels"]["skipped"][0], "c|d");
}

#[test]
fn resolve_without_a_known_file_or_candidates_fails_without_calling_the_service() {
    let m = mock(happy);
    let s = HttpService::connect(&m.url);
    let none = ResolveRequest { item: ItemId(9), labels: labels(), candidates: Some("{}".into()) };
    assert!(matches!(wait(&mut *s.resolve(none)), JobState::Failed(_)));
    let _ = wait(&mut *s.analyze(request()));
    let no_doc = ResolveRequest { item: ItemId(7), labels: labels(), candidates: None };
    assert!(matches!(wait(&mut *s.resolve(no_doc)), JobState::Failed(_)));
    let bad_doc = ResolveRequest { item: ItemId(7), labels: labels(), candidates: Some("not json".into()) };
    assert!(matches!(wait(&mut *s.resolve(bad_doc)), JobState::Failed(_)));
    assert!(m.seen.lock().unwrap().iter().all(|(_, p, _)| p != "/v1/resolve"));
}

#[test]
fn a_400_with_an_error_message_is_shown_as_is() {
    let m = mock(|me, p, b| if p == "/v1/analyze" { (400, r#"{"error":"duration_seconds must be a positive number"}"#.into()) } else { happy(me, p, b) });
    let s = HttpService::connect(&m.url);
    assert_eq!(wait(&mut *s.analyze(request())), JobState::Failed("duration_seconds must be a positive number".into()));
}

#[test]
fn an_error_without_a_message_gets_a_plain_sentence() {
    let m = mock(|me, p, b| if p == "/v1/analyze" { (500, "<html>boom</html>".into()) } else { happy(me, p, b) });
    let s = HttpService::connect(&m.url);
    let JobState::Failed(why) = wait(&mut *s.analyze(request())) else { panic!() };
    assert!(why.contains("HTTP 500") && !why.contains("html"), "{why}");
}

#[test]
fn a_failed_job_carries_the_services_sentence() {
    let m = mock(
        |me, p, b| if p == "/v1/jobs/j1" { (200, r#"{"state":"failed","fraction":0.5,"error":"Could not read the file"}"#.into()) } else { happy(me, p, b) },
    );
    let s = HttpService::connect(&m.url);
    assert_eq!(wait(&mut *s.analyze(request())), JobState::Failed("Could not read the file".into()));
}

#[test]
fn hostile_progress_numbers_are_clamped_and_unknown_states_fail() {
    let polls = Arc::new(Mutex::new(0));
    let p2 = Arc::clone(&polls);
    let m = mock(move |me, p, b| {
        if p == "/v1/jobs/j1" {
            let mut n = p2.lock().unwrap();
            *n += 1;
            return match *n {
                1 => (200, r#"{"state":"running","fraction":7.5,"status":"Transcribing"}"#.into()),
                _ => (200, r#"{"state":"exploded"}"#.into()),
            };
        }
        happy(me, p, b)
    });
    let s = HttpService::connect(&m.url);
    let mut job = s.analyze(request());
    let mut saw_clamped = false;
    let end = Instant::now() + Duration::from_secs(20);
    let last = loop {
        match job.poll() {
            JobState::Running { fraction, .. } => {
                saw_clamped |= fraction == 1.0;
                std::thread::sleep(Duration::from_millis(5));
            }
            other => break other,
        }
        assert!(Instant::now() < end, "never ended");
    };
    assert!(saw_clamped, "7.5 must show as 1.0");
    assert!(matches!(last, JobState::Failed(_)));
}

#[test]
fn a_done_job_without_a_result_is_a_failure() {
    let m = mock(|me, p, b| if p == "/v1/jobs/j1" { (200, r#"{"state":"done","fraction":1.0}"#.into()) } else { happy(me, p, b) });
    let s = HttpService::connect(&m.url);
    assert!(matches!(wait(&mut *s.analyze(request())), JobState::Failed(_)));
}

#[test]
fn truncated_json_from_the_service_fails_cleanly() {
    let m = mock(|me, p, b| if p == "/v1/jobs/j1" { (200, r#"{"state":"done","result":{"sta"#.into()) } else { happy(me, p, b) });
    let s = HttpService::connect(&m.url);
    assert!(matches!(wait(&mut *s.analyze(request())), JobState::Failed(_)));
}

#[test]
fn an_unknown_job_means_the_service_restarted() {
    let m = mock(|me, p, b| if p == "/v1/jobs/j1" { (404, r#"{"error":"Unknown job."}"#.into()) } else { happy(me, p, b) });
    let s = HttpService::connect(&m.url);
    let JobState::Failed(why) = wait(&mut *s.analyze(request())) else { panic!() };
    assert!(why.contains("restarted"), "{why}");
}

#[test]
fn cancel_sends_delete_and_ends_cancelled() {
    let m = mock(|me, p, b| match (me, p) {
        ("GET", "/v1/jobs/j1") => (200, r#"{"state":"running","fraction":0.2,"status":"Transcribing"}"#.into()),
        ("DELETE", "/v1/jobs/j1") => (204, String::new()),
        _ => happy(me, p, b),
    });
    let s = HttpService::connect(&m.url);
    let mut job = s.analyze(request());
    std::thread::sleep(Duration::from_millis(300));
    assert!(matches!(job.poll(), JobState::Running { .. }));
    job.cancel();
    assert_eq!(wait(&mut *job), JobState::Cancelled);
    assert!(m.seen.lock().unwrap().iter().any(|(me, p, _)| me == "DELETE" && p == "/v1/jobs/j1"));
}

#[test]
fn a_service_cancelling_the_job_ends_cancelled() {
    let m = mock(|me, p, b| if p == "/v1/jobs/j1" { (200, r#"{"state":"cancelled","fraction":0.1}"#.into()) } else { happy(me, p, b) });
    let s = HttpService::connect(&m.url);
    assert_eq!(wait(&mut *s.analyze(request())), JobState::Cancelled);
}

// ── keeper choice ──────────────────────────────────────────────────────────

const KEEPER_CAPS: &str = r#"{"adjudicate": true, "refine": true, "keeper": {"current": "finetuned", "default": "finetuned", "providers": {
    "finetuned": {"available": true, "reason": "decision model"},
    "gpt-oss": {"available": false, "reason": "no LLM key is configured"},
    "keep-last": {"available": true, "reason": "always available"}}}}"#;

fn keeper_service() -> Mock {
    mock(|method, path, body| match (method, path) {
        ("GET", "/v1/capabilities") => (200, KEEPER_CAPS.into()),
        ("PUT", "/v1/keeper") if body.contains("keep-last") => {
            (200, r#"{"current": "keep-last", "default": "finetuned", "providers": {"keep-last": {"available": true, "reason": "always available"}}}"#.into())
        }
        ("PUT", "/v1/keeper") => (400, r#"{"error": "Keeper 'gpt-oss' is not available: no LLM key is configured."}"#.into()),
        _ => (404, "{}".into()),
    })
}

#[test]
fn the_keepers_are_read_from_the_capabilities() {
    let server = keeper_service();
    let service = HttpService::connect(&server.url);
    let info = service.keeper().unwrap();
    assert_eq!(info.current, "finetuned");
    assert_eq!(info.providers.len(), 3);
    assert!(info.provider("finetuned").unwrap().available);
    let off = info.provider("gpt-oss").unwrap();
    assert!(!off.available);
    assert_eq!(off.reason, "no LLM key is configured");
}

#[test]
fn an_older_service_has_no_keepers_and_cannot_switch() {
    let server = mock(|_, _, _| (200, r#"{"adjudicate": true, "refine": true}"#.into()));
    let service = HttpService::connect(&server.url);
    assert!(service.keeper().is_none());
    assert!(service.set_keeper("keep-last").is_none());
    assert!(server.seen.lock().unwrap().iter().all(|(m, _, _)| m != "PUT"));
}

#[test]
fn switching_the_keeper_sends_the_id_and_updates_the_state() {
    let server = keeper_service();
    let service = HttpService::connect(&server.url);
    let mut job = service.set_keeper("keep-last").unwrap();
    assert!(matches!(wait(job.as_mut()), JobState::Done(_)));
    let seen = server.seen.lock().unwrap().clone();
    let put = seen.iter().find(|(m, p, _)| m == "PUT" && p == "/v1/keeper").unwrap();
    assert_eq!(serde_json::from_str::<Value>(&put.2).unwrap()["provider"], "keep-last");
}

#[test]
fn a_refused_switch_shows_the_services_sentence_and_changes_nothing() {
    let server = keeper_service();
    let service = HttpService::connect(&server.url);
    let mut job = service.set_keeper("gpt-oss").unwrap();
    let JobState::Failed(why) = wait(job.as_mut()) else { panic!("should have failed") };
    assert!(why.contains("no LLM key is configured"), "{why}");
    assert_eq!(service.keeper().unwrap().current, "finetuned");
}

#[test]
fn hostile_keeper_text_is_cut_and_junk_is_skipped() {
    let long = "x".repeat(5000);
    let caps = format!(
        r#"{{"adjudicate": true, "keeper": {{"current": "{long}", "providers": {{"": {{"available": true}}, "a": {{"available": "yes", "reason": "{long}"}}}}}}}}"#
    );
    let server = mock(move |_, _, _| (200, caps.clone()));
    let service = HttpService::connect(&server.url);
    let info = service.keeper().unwrap();
    assert_eq!(info.current.chars().count(), 300);
    assert_eq!(info.providers.len(), 1);
    assert!(!info.providers[0].available);
    assert_eq!(info.providers[0].reason.chars().count(), 300);
}
