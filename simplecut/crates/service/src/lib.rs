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
use simplecut_core::{
    AdjudicateRequest, AnalysisJob, AnalysisRequest, Capabilities, ExportRequest, ItemId, JobState, KeeperInfo, ProjectCall, RefineRequest, ResolveRequest,
    RetakeService, ServiceOptions, StageEvent, Tick, UserCall, WorkCall,
};

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
    capabilities: Arc<Mutex<Capabilities>>,
    /// The keepers the service offers, as last reported (None: not reported, e.g. an older service).
    keeper: Arc<Mutex<Option<KeeperInfo>>>,
    /// The settings the service offers (models, silence-gap range), as last reported.
    options: Arc<Mutex<ServiceOptions>>,
    /// The project folder the editor has open; sent with every request so the service keeps its data in it.
    project: Arc<Mutex<Option<String>>>,
    reachable: Arc<AtomicBool>,
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
            capabilities: Arc::new(Mutex::new(Capabilities::default())),
            keeper: Arc::new(Mutex::new(None)),
            options: Arc::new(Mutex::new(ServiceOptions::default())),
            project: Arc::new(Mutex::new(None)),
            reachable: Arc::new(AtomicBool::new(false)),
            paths: Arc::new(Mutex::new(Vec::new())),
        };
        service.refresh();
        service
    }

    /// Ask the service again what it can do (after it was started late, for instance). Every
    /// request also does this first, in its own thread.
    pub fn refresh(&self) {
        refresh_capabilities(&self.agent, &self.base, &self.capabilities, &self.keeper, &self.options, &self.reachable);
    }

    /// Whether the last attempt to reach the service worked.
    pub fn is_reachable(&self) -> bool {
        self.reachable.load(Ordering::SeqCst)
    }

    /// The address this client talks to.
    pub fn base_url(&self) -> &str {
        &self.base
    }

    fn project(&self) -> Option<String> {
        self.project.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// The request body with the open project folder added (when there is one).
    fn tagged(&self, mut body: Value) -> String {
        if let (Some(p), Some(map)) = (self.project(), body.as_object_mut()) {
            map.insert("project".into(), json!(p));
        }
        body.to_string()
    }

    /// One request, one reply, for the project calls: the result is `{"status": <HTTP status>, "body": <reply>}` whatever the status, so a "please confirm" (409) or a
    /// refusal (400) reaches the editor with its details. Only a failure to reach the service is a failed job.
    fn start_call(&self, method: Method, path: String, body: String) -> Box<dyn AnalysisJob> {
        let state = Arc::new(Mutex::new(JobState::Running { fraction: 0.5, status: "Asking the retake service".into() }));
        let (agent, url, thread_state) = (self.agent.clone(), format!("{}{path}", self.base), Arc::clone(&state));
        let spawned = std::thread::Builder::new().name("retake-call".into()).spawn(move || {
            let end = match send(&agent, method, &url, Some(&body)) {
                Ok(r) => {
                    let parsed = serde_json::from_str::<Value>(&r.body).unwrap_or(Value::Null);
                    JobState::Done(json!({ "status": r.status, "body": parsed }).to_string())
                }
                Err(e) => JobState::Failed(e),
            };
            set(&thread_state, end);
        });
        if spawned.is_err() {
            return failed("Could not start the request.");
        }
        Box::new(HttpJob { state, cancel: Arc::new(AtomicBool::new(false)), immediate: true, events: Arc::new(Mutex::new(Vec::new())), taken: 0 })
    }

    fn start(&self, endpoint: &str, body: String) -> Box<dyn AnalysisJob> {
        self.start_as(Method::Post, endpoint, body, false)
    }

    /// `immediate`: the service answers in one reply (no job to follow), so Cancel cannot reach it;
    /// the job reads as cancelled at once and the late answer is dropped.
    fn start_as(&self, method: Method, endpoint: &str, body: String, immediate: bool) -> Box<dyn AnalysisJob> {
        let state = Arc::new(Mutex::new(JobState::Running { fraction: 0.0, status: "Contacting the retake service".into() }));
        let events: EventBox = Arc::new(Mutex::new(Vec::new()));
        let thread_events = Arc::clone(&events);
        let cancel = Arc::new(AtomicBool::new(false));
        let (agent, base, endpoint) = (self.agent.clone(), self.base.clone(), endpoint.to_string());
        let (thread_state, thread_cancel) = (Arc::clone(&state), Arc::clone(&cancel));
        let (caps, keeper, reachable, options) =
            (Arc::clone(&self.capabilities), Arc::clone(&self.keeper), Arc::clone(&self.reachable), Arc::clone(&self.options));
        let spawned = std::thread::Builder::new().name("retake-http".into()).spawn(move || {
            refresh_capabilities(&agent, &base, &caps, &keeper, &options, &reachable);
            let end = if immediate {
                run_once(&agent, method, &base, &endpoint, &body, &thread_state)
            } else {
                run_job(&agent, &base, &endpoint, &body, &thread_state, &thread_cancel, &thread_events)
            };
            if let JobState::Done(reply) = &end
                && method == Method::Put
                && let Some(info) = serde_json::from_str::<Value>(reply).ok().as_ref().and_then(KeeperInfo::from_json)
            {
                *keeper.lock().unwrap_or_else(PoisonError::into_inner) = Some(info);
                refresh_capabilities(&agent, &base, &caps, &keeper, &options, &reachable);
            }
            if !(immediate && thread_cancel.load(Ordering::SeqCst)) {
                set(&thread_state, end);
            }
        });
        if spawned.is_err() {
            return failed("Could not start the request.");
        }
        Box::new(HttpJob { state, cancel, immediate, events, taken: 0 })
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

    fn keeper(&self) -> Option<KeeperInfo> {
        self.keeper.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn options(&self) -> ServiceOptions {
        self.options.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn set_project(&self, folder: Option<std::path::PathBuf>) {
        *self.project.lock().unwrap_or_else(PoisonError::into_inner) = folder.map(|f| f.to_string_lossy().into_owned());
    }

    fn export(&self, request: ExportRequest) -> Option<Box<dyn AnalysisJob>> {
        let path = request.path;
        if self.project().is_none() {
            return Some(failed("Save the project first: exports are written into the project folder."));
        }
        let mut body = request.body;
        if let Some(map) = body.as_object_mut() {
            map.insert("path".into(), json!(path));
        }
        Some(self.start("/v1/export", self.tagged(body)))
    }

    fn work_call(&self, call: WorkCall) -> Option<Box<dyn AnalysisJob>> {
        let (endpoint, what, path) = match call {
            WorkCall::Reset { what, path } => ("/v1/work/reset", what, path),
            WorkCall::Restore { what, path } => ("/v1/work/restore", what, path),
        };
        // the project folder (when there is one) is added; without one the service works on its old per-machine cache for this video
        Some(self.start_call(Method::Post, endpoint.to_string(), self.tagged(json!({ "what": what.id(), "path": path }))))
    }

    fn user_call(&self, call: UserCall) -> Option<Box<dyn AnalysisJob>> {
        Some(match call {
            UserCall::ExportOptions => self.start_call(Method::Get, "/v1/export/options".into(), String::new()),
            UserCall::SavePreset { name, preset } => self.start_call(Method::Put, format!("/v1/export/presets/{}", urlencode(&name)), preset.to_string()),
            UserCall::DeletePreset { name } => self.start_call(Method::Delete, format!("/v1/export/presets/{}", urlencode(&name)), String::new()),
        })
    }

    fn project_call(&self, call: ProjectCall) -> Option<Box<dyn AnalysisJob>> {
        let Some(project) = self.project() else { return Some(failed("Save the project first: its settings are kept in the project folder.")) };
        let q = urlencode(&project);
        let (method, path, body) = match call {
            ProjectCall::Status => (Method::Get, format!("/v1/project/status?project={q}"), String::new()),
            ProjectCall::SetSettings { model, silence_gap_s } => {
                let mut b = json!({ "project": project });
                if let Some(map) = b.as_object_mut() {
                    if let Some(m) = model {
                        map.insert("transcription_model".into(), json!(m));
                    }
                    if let Some(g) = silence_gap_s {
                        map.insert("silence_gap_s".into(), json!(g));
                    }
                }
                (Method::Put, "/v1/project/settings".to_string(), b.to_string())
            }
            ProjectCall::UnitsPreview { silence_gap_s } => {
                (Method::Get, format!("/v1/project/units-preview?project={q}&silence_gap_s={silence_gap_s}"), String::new())
            }
            ProjectCall::ImportLegacy { path } => {
                (Method::Post, "/v1/project/import-legacy".to_string(), json!({ "project": project, "path": path }).to_string())
            }
            ProjectCall::Resplit { silence_gap_s, confirm } => {
                (Method::Post, "/v1/project/resplit".to_string(), json!({ "project": project, "silence_gap_s": silence_gap_s, "confirm": confirm }).to_string())
            }
        };
        Some(self.start_call(method, path, body))
    }

    fn set_keeper(&self, id: &str) -> Option<Box<dyn AnalysisJob>> {
        // an older service has no /v1/keeper: do not offer a switch it cannot make
        self.keeper()?;
        Some(self.start_as(Method::Put, "/v1/keeper", json!({ "provider": id }).to_string(), true))
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
        if let Some(map) = body.as_object_mut() {
            if let Some(m) = &request.settings.model {
                map.insert("model".into(), json!(m));
            }
            if let Some(g) = request.settings.silence_gap_s {
                map.insert("silence_gap_s".into(), json!(g));
            }
            if request.settings.replace_existing {
                map.insert("replace_existing".into(), json!(true));
            }
        }
        self.start("/v1/analyze", self.tagged(body))
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
        self.start("/v1/resolve", self.tagged(json!({ "path": path, "labels": labels, "candidates": candidates })))
    }

    fn adjudicate(&self, request: AdjudicateRequest) -> Option<Box<dyn AnalysisJob>> {
        if !self.capabilities().adjudicate {
            return None;
        }
        let Some(path) = self.path_of(request.item) else {
            return Some(failed("The retake service does not know this file yet. Run Remove Retakes again."));
        };
        // the groups as the editor holds them now (spans the user changed included); the service needs the text
        let groups: Vec<Value> = request
            .groups
            .iter()
            .map(|g| {
                json!({
                    "id": g.id,
                    "start": seconds(g.start),
                    "end": seconds(g.end),
                    "likely": g.likely,
                    "sentences": g.sentences.iter().map(|s| json!({
                        "id": s.id, "start": seconds(s.start), "end": seconds(s.end), "text": s.text,
                    })).collect::<Vec<_>>(),
                })
            })
            .collect();
        Some(self.start("/v1/adjudicate", self.tagged(json!({ "path": path, "job": request.job, "groups": groups }))))
    }

    fn refine(&self, request: RefineRequest) -> Option<Box<dyn AnalysisJob>> {
        if !self.capabilities().refine {
            return None;
        }
        let Some(path) = self.path_of(request.item) else {
            return Some(failed("The retake service does not know this file yet. Run Remove Retakes again."));
        };
        let body = json!({ "path": path, "job": request.job, "remove": request.remove, "mode": request.mode.as_str() });
        Some(self.start_as(Method::Post, "/v1/refine", self.tagged(body), true))
    }
}

// ── jobs ───────────────────────────────────────────────────────────────────

/// The events fetched so far, shared with the polling thread.
type EventBox = Arc<Mutex<Vec<StageEvent>>>;

struct HttpJob {
    state: Arc<Mutex<JobState>>,
    cancel: Arc<AtomicBool>,
    immediate: bool,
    events: EventBox,
    /// How many of `events` were already handed out.
    taken: usize,
}

impl AnalysisJob for HttpJob {
    fn poll(&mut self) -> JobState {
        self.state.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn events(&mut self) -> Vec<StageEvent> {
        let all = self.events.lock().unwrap_or_else(PoisonError::into_inner);
        let new = all.get(self.taken..).map(<[StageEvent]>::to_vec).unwrap_or_default();
        self.taken = all.len();
        new
    }

    fn cancel(&mut self) {
        self.cancel.store(true, Ordering::SeqCst);
        if self.immediate {
            let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            if matches!(*state, JobState::Running { .. }) {
                *state = JobState::Cancelled;
            }
        }
    }
}

fn refresh_capabilities(
    agent: &ureq::Agent,
    base: &str,
    into: &Mutex<Capabilities>,
    keeper: &Mutex<Option<KeeperInfo>>,
    options: &Mutex<ServiceOptions>,
    reachable: &AtomicBool,
) {
    let mut info = None;
    let mut offered = ServiceOptions::default();
    let caps = match send(agent, Method::Get, &format!("{base}/v1/capabilities"), None) {
        Ok(r) if r.status == 200 => serde_json::from_str::<Value>(&r.body).ok().map(|v| {
            info = v.get("keeper").and_then(KeeperInfo::from_json);
            offered = ServiceOptions::from_capabilities(&v);
            Capabilities { adjudicate: v["adjudicate"].as_bool().unwrap_or(false), refine: v["refine"].as_bool().unwrap_or(false) }
        }),
        Ok(r) => {
            log::warn!("capabilities: HTTP {}", r.status);
            None
        }
        Err(e) => {
            log::warn!("capabilities: {e}");
            None
        }
    };
    reachable.store(caps.is_some(), Ordering::SeqCst);
    *into.lock().unwrap_or_else(PoisonError::into_inner) = caps.unwrap_or_default();
    *keeper.lock().unwrap_or_else(PoisonError::into_inner) = info;
    *options.lock().unwrap_or_else(PoisonError::into_inner) = offered;
}

fn failed(why: &str) -> Box<dyn AnalysisJob> {
    Box::new(HttpJob {
        state: Arc::new(Mutex::new(JobState::Failed(why.to_string()))),
        cancel: Arc::new(AtomicBool::new(false)),
        immediate: false,
        events: Arc::new(Mutex::new(Vec::new())),
        taken: 0,
    })
}

/// One request, one reply (`/v1/refine`): the reply body is the result.
fn run_once(agent: &ureq::Agent, method: Method, base: &str, endpoint: &str, body: &str, state: &Mutex<JobState>) -> JobState {
    let status = if method == Method::Put { "Switching the keeper" } else { "Refining the cuts" };
    set(state, JobState::Running { fraction: 0.5, status: status.into() });
    match send(agent, method, &format!("{base}{endpoint}"), Some(body)) {
        Ok(r) if r.status == 200 => JobState::Done(r.body),
        Ok(r) => JobState::Failed(error_of(&r)),
        Err(e) => JobState::Failed(e),
    }
}

fn set(state: &Mutex<JobState>, to: JobState) {
    *state.lock().unwrap_or_else(PoisonError::into_inner) = to;
}

/// Send the request, then follow the job until it ends. Returns the final state.
fn run_job(agent: &ureq::Agent, base: &str, endpoint: &str, body: &str, state: &Mutex<JobState>, cancel: &AtomicBool, events: &EventBox) -> JobState {
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
    let events_url = format!("{url}/events");
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
                    Some(JobState::Running { fraction, status }) => {
                        set(state, JobState::Running { fraction, status });
                        fetch_events(agent, &events_url, events);
                    }
                    Some(end) => {
                        fetch_events(agent, &events_url, events); // the last events (an error message, say) arrive with the end
                        return end;
                    }
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

/// Fetch the events after the ones already held. Never fails the job: a service without the endpoint, a bad reply or a dropped connection just means no new events.
fn fetch_events(agent: &ureq::Agent, url: &str, into: &Mutex<Vec<StageEvent>>) {
    let after = into.lock().unwrap_or_else(PoisonError::into_inner).last().map_or(0, |e| e.seq);
    let Ok(reply) = send(agent, Method::Get, &format!("{url}?after={after}"), None) else { return };
    if reply.status != 200 {
        return;
    }
    let Some(list) = serde_json::from_str::<Value>(&reply.body).ok().and_then(|v| v.get("events").and_then(Value::as_array).cloned()) else { return };
    let mut held = into.lock().unwrap_or_else(PoisonError::into_inner);
    for ev in list.iter().filter_map(StageEvent::from_json) {
        if held.last().is_none_or(|last| ev.seq > last.seq) && held.len() < simplecut_core::progress::MAX_EVENTS {
            held.push(ev);
        }
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Method {
    Get,
    Post,
    Put,
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
        Method::Put => agent.put(url).header("Content-Type", "application/json").send(body.unwrap_or("{}")),
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

/// Percent-encode a value for a URL query.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn seconds(t: Tick) -> f64 {
    t.0.max(0) as f64 / TICKS_PER_SECOND
}
