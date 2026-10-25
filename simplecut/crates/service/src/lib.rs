//! The editor's HTTP client for the local retake service (`python -m service`, 127.0.0.1:8765).
//!
//! It implements [`RetakeService`], so the rest of the editor cannot tell it from the `FakeService`.
//! Every request is a job: `POST` returns `202 {"job"}`, then a background thread polls
//! `GET /v1/jobs/{job}` about every 200 ms into a shared [`JobState`]; cancelling sends `DELETE`.
//! `poll()` never blocks. The service is on this machine but is still treated as hostile: every
//! number and string it sends is checked, every failure becomes a plain sentence.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use serde_json::{Value, json};
use simplecut_core::{AnalysisJob, AnalysisRequest, Capabilities, ItemId, JobState, ResolveRequest, RetakeService, Tick};

/// The address the service listens on unless told otherwise.
pub const DEFAULT_URL: &str = "http://127.0.0.1:8765";

const TICKS_PER_SECOND: f64 = 254_016_000_000.0;
const POLL_EVERY: Duration = Duration::from_millis(200);
/// Polls that may fail in a row (service busy or restarting) before the request is given up.
const MAX_MISSED_POLLS: u32 = 10;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_REPLY_BYTES: u64 = 32 * 1024 * 1024;
const MAX_STATUS_CHARS: usize = 120;

/// A client for the retake service.
pub struct HttpService {
    base: String,
    agent: ureq::Agent,
    capabilities: Mutex<Capabilities>,
    reachable: AtomicBool,
    /// Where each analysed file is: `resolve` does not carry the path, the service needs it.
    paths: Arc<Mutex<Vec<(ItemId, String)>>>,
}

impl HttpService {
    /// Create a client and ask the service what it can do. Never fails: when nobody answers, the
    /// client says so through [`HttpService::is_reachable`] and every request fails with a plain
    /// message. Call [`HttpService::refresh`] to ask again.
    pub fn connect(base_url: &str) -> Self {
        let service = Self {
            base: base_url.trim().trim_end_matches('/').to_string(),
            agent: agent(),
            capabilities: Mutex::new(Capabilities::default()),
            reachable: AtomicBool::new(false),
            paths: Arc::new(Mutex::new(Vec::new())),
        };
        service.refresh();
        service
    }

    /// Ask the service again what it can do (after it was started late, for instance).
    pub fn refresh(&self) {
        let reply = send(&self.agent, Method::Get, &format!("{}/v1/capabilities", self.base), None);
        let caps = match reply {
            Ok(r) if r.status == 200 => {
                serde_json::from_str::<Value>(&r.body).ok().map(|v| Capabilities { adjudicate: v["adjudicate"].as_bool().unwrap_or(false) })
            }
            Ok(r) => {
                log::warn!("capabilities: HTTP {}", r.status);
                None
            }
            Err(e) => {
                log::warn!("capabilities: {e}");
                None
            }
        };
        self.reachable.store(caps.is_some(), Ordering::SeqCst);
        *self.capabilities.lock().unwrap_or_else(PoisonError::into_inner) = caps.unwrap_or_default();
    }

    /// Whether the last attempt to reach the service worked.
    pub fn is_reachable(&self) -> bool {
        self.reachable.load(Ordering::SeqCst)
    }

    /// The address this client talks to.
    pub fn base_url(&self) -> &str {
        &self.base
    }

    fn start(&self, endpoint: &str, body: String) -> Box<dyn AnalysisJob> {
        let state = Arc::new(Mutex::new(JobState::Running { fraction: 0.0, status: "Contacting the retake service".into() }));
        let cancel = Arc::new(AtomicBool::new(false));
        let (agent, base, endpoint) = (self.agent.clone(), self.base.clone(), endpoint.to_string());
        let (thread_state, thread_cancel) = (Arc::clone(&state), Arc::clone(&cancel));
        let spawned = std::thread::Builder::new().name("retake-http".into()).spawn(move || {
            let end = run_job(&agent, &base, &endpoint, &body, &thread_state, &thread_cancel);
            set(&thread_state, end);
        });
        if spawned.is_err() {
            return failed("Could not start the request.");
        }
        Box::new(HttpJob { state, cancel })
    }

    fn path_of(&self, item: ItemId) -> Option<String> {
        self.paths.lock().unwrap_or_else(PoisonError::into_inner).iter().find(|(i, _)| *i == item).map(|(_, p)| p.clone())
    }
}

impl RetakeService for HttpService {
    fn name(&self) -> String {
        format!("retake service ({})", self.base)
    }

    fn capabilities(&self) -> Capabilities {
        *self.capabilities.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn analyze(&self, request: AnalysisRequest) -> Box<dyn AnalysisJob> {
        {
            let mut paths = self.paths.lock().unwrap_or_else(PoisonError::into_inner);
            paths.retain(|(i, _)| *i != request.item);
            paths.push((request.item, request.path.clone()));
        }
        let used: Vec<Value> = request.used.iter().map(|(a, b)| json!([seconds(*a), seconds(*b)])).collect();
        let mut body = json!({
            "path": request.path,
            "name": request.name,
            "duration_seconds": seconds(request.duration),
            "used": used,
        });
        if let (Some(fp), Some(map)) = (&request.fingerprint, body.as_object_mut()) {
            map.insert("fingerprint".into(), json!(fp));
        }
        self.start("/v1/analyze", body.to_string())
    }

    fn resolve(&self, request: ResolveRequest) -> Box<dyn AnalysisJob> {
        let Some(path) = self.path_of(request.item) else {
            return failed("The retake service does not know this file yet. Run Remove Retakes again.");
        };
        // the candidates go back exactly as they came (the service needs its own extra fields)
        let Some(candidates) = request.candidates.as_deref().and_then(|c| serde_json::from_str::<Value>(c).ok()) else {
            return failed("The questions are missing, so they cannot be sent back. Run Remove Retakes again.");
        };
        let Ok(labels) = serde_json::to_value(&request.labels) else {
            return failed("Could not read the answers.");
        };
        self.start("/v1/resolve", json!({ "path": path, "labels": labels, "candidates": candidates }).to_string())
    }
}

// ── jobs ───────────────────────────────────────────────────────────────────

struct HttpJob {
    state: Arc<Mutex<JobState>>,
    cancel: Arc<AtomicBool>,
}

impl AnalysisJob for HttpJob {
    fn poll(&mut self) -> JobState {
        self.state.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn cancel(&mut self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
}

fn failed(why: &str) -> Box<dyn AnalysisJob> {
    Box::new(HttpJob { state: Arc::new(Mutex::new(JobState::Failed(why.to_string()))), cancel: Arc::new(AtomicBool::new(false)) })
}

fn set(state: &Mutex<JobState>, to: JobState) {
    *state.lock().unwrap_or_else(PoisonError::into_inner) = to;
}

/// Send the request, then follow the job until it ends. Returns the final state.
fn run_job(agent: &ureq::Agent, base: &str, endpoint: &str, body: &str, state: &Mutex<JobState>, cancel: &AtomicBool) -> JobState {
    let started = match send(agent, Method::Post, &format!("{base}{endpoint}"), Some(body)) {
        Ok(r) => r,
        Err(e) => return JobState::Failed(e),
    };
    if started.status != 202 {
        return JobState::Failed(error_of(&started));
    }
    let Some(job) = serde_json::from_str::<Value>(&started.body).ok().and_then(|v| v["job"].as_str().map(str::to_string)).filter(|j| !j.is_empty()) else {
        return JobState::Failed("The retake service answered in a form this editor does not understand.".into());
    };
    let url = format!("{base}/v1/jobs/{job}");
    let mut missed = 0u32;
    loop {
        if cancel.load(Ordering::SeqCst) {
            if let Err(e) = send(agent, Method::Delete, &url, None) {
                log::warn!("cancel: {e}");
            }
            return JobState::Cancelled;
        }
        match send(agent, Method::Get, &url, None) {
            Ok(r) if r.status == 200 => {
                missed = 0;
                match read_state(&r.body) {
                    Some(JobState::Running { fraction, status }) => set(state, JobState::Running { fraction, status }),
                    Some(end) => return end,
                    None => return JobState::Failed("The retake service answered in a form this editor does not understand.".into()),
                }
            }
            Ok(r) if r.status == 404 => return JobState::Failed("The retake service lost this request (it may have restarted). Try again.".into()),
            Ok(r) => return JobState::Failed(error_of(&r)),
            Err(e) => {
                missed += 1;
                if missed >= MAX_MISSED_POLLS {
                    return JobState::Failed(e);
                }
            }
        }
        std::thread::sleep(POLL_EVERY);
    }
}

/// One `GET /v1/jobs/{id}` body as a state. None when it is not a state at all.
fn read_state(body: &str) -> Option<JobState> {
    let v: Value = serde_json::from_str(body).ok()?;
    match v["state"].as_str()? {
        "running" => {
            let fraction = v["fraction"].as_f64().filter(|f| f.is_finite()).unwrap_or(0.0).clamp(0.0, 1.0) as f32;
            let status: String = v["status"].as_str().unwrap_or("Working").chars().take(MAX_STATUS_CHARS).collect();
            Some(JobState::Running { fraction, status })
        }
        "done" => match v.get("result") {
            Some(r) if !r.is_null() => Some(JobState::Done(r.to_string())),
            _ => Some(JobState::Failed("The retake service finished without an answer.".into())),
        },
        "failed" => {
            let why = v["error"].as_str().filter(|e| !e.trim().is_empty()).unwrap_or("The retake service could not finish.");
            Some(JobState::Failed(why.chars().take(500).collect()))
        }
        "cancelled" => Some(JobState::Cancelled),
        _ => None,
    }
}

// ── http ───────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Method {
    Get,
    Post,
    Delete,
}

struct Reply {
    status: u16,
    body: String,
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().timeout_global(Some(REQUEST_TIMEOUT)).http_status_as_error(false).build().into()
}

fn send(agent: &ureq::Agent, method: Method, url: &str, body: Option<&str>) -> Result<Reply, String> {
    let unreachable = |e: &dyn std::fmt::Display| {
        log::warn!("{url}: {e}");
        "Could not reach the retake service. Is it running?".to_string()
    };
    let result = match method {
        Method::Get => agent.get(url).call(),
        Method::Delete => agent.delete(url).call(),
        Method::Post => agent.post(url).header("Content-Type", "application/json").send(body.unwrap_or("{}")),
    };
    let mut response = result.map_err(|e| unreachable(&e))?;
    let status = response.status().as_u16();
    let body = response.body_mut().with_config().limit(MAX_REPLY_BYTES).read_to_string().map_err(|e| unreachable(&e))?;
    Ok(Reply { status, body })
}

/// The service's `{"error": "..."}`, or a plain sentence when it sent something else.
fn error_of(reply: &Reply) -> String {
    serde_json::from_str::<Value>(&reply.body)
        .ok()
        .and_then(|v| v["error"].as_str().map(|e| e.chars().take(500).collect::<String>()))
        .filter(|e| !e.trim().is_empty())
        .unwrap_or_else(|| format!("The retake service answered with an error (HTTP {}).", reply.status))
}

fn seconds(t: Tick) -> f64 {
    t.0.max(0) as f64 / TICKS_PER_SECOND
}
