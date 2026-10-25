//! Draws preview pictures on a worker thread so the window never waits for a decode.
//!
//! Only the newest request matters: while the user scrubs, requests that have been overtaken are
//! dropped, not drawn.

use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use filmcraft_time::Tick;

use crate::snapshot::{PreviewFrame, Snapshot};

/// A finished picture and what it was asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedFrame {
    /// The id [`PreviewRenderer::request`] returned.
    pub id: u64,
    pub time: Tick,
    pub frame: PreviewFrame,
}

struct Request {
    id: u64,
    snapshot: Snapshot,
    time: Tick,
    scale: f32,
}

#[derive(Default)]
struct State {
    pending: Option<Request>,
    done: Option<RenderedFrame>,
    next_id: u64,
    busy: bool,
    quit: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    ready: Condvar,
}

fn lock(m: &Mutex<State>) -> MutexGuard<'_, State> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub struct PreviewRenderer {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl Default for PreviewRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl PreviewRenderer {
    pub fn new() -> Self {
        let shared = Arc::new(Shared { state: Mutex::default(), wake: Condvar::new(), ready: Condvar::new() });
        let for_thread = shared.clone();
        let worker = std::thread::Builder::new().name("simplecut-preview".into()).spawn(move || work(&for_thread)).ok();
        if worker.is_none() {
            log::error!("could not start the preview thread; pictures will be drawn on request only");
        }
        Self { shared, worker }
    }

    /// Ask for the picture at `time`, replacing any request that has not started yet. Returns the
    /// request's id.
    pub fn request(&self, snapshot: Snapshot, time: Tick, scale: f32) -> u64 {
        let mut st = lock(&self.shared.state);
        st.next_id += 1;
        let id = st.next_id;
        st.pending = Some(Request { id, snapshot, time, scale });
        self.shared.wake.notify_one();
        id
    }

    /// The newest finished picture, if one has arrived since the last call.
    pub fn take_result(&self) -> Option<RenderedFrame> {
        lock(&self.shared.state).done.take()
    }

    /// Wait up to `timeout` for a finished picture.
    pub fn wait_result(&self, timeout: Duration) -> Option<RenderedFrame> {
        let deadline = Instant::now() + timeout;
        let mut st = lock(&self.shared.state);
        loop {
            if let Some(f) = st.done.take() {
                return Some(f);
            }
            let left = deadline.checked_duration_since(Instant::now())?;
            let (guard, timed_out) = self.shared.ready.wait_timeout(st, left).unwrap_or_else(PoisonError::into_inner);
            st = guard;
            if timed_out.timed_out() && st.done.is_none() {
                return None;
            }
        }
    }

    /// True when nothing is waiting or being drawn.
    pub fn is_idle(&self) -> bool {
        let st = lock(&self.shared.state);
        st.pending.is_none() && !st.busy
    }
}

impl Drop for PreviewRenderer {
    fn drop(&mut self) {
        lock(&self.shared.state).quit = true;
        self.shared.wake.notify_all();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

fn work(shared: &Shared) {
    loop {
        let req = {
            let mut st = lock(&shared.state);
            loop {
                if st.quit {
                    return;
                }
                if let Some(r) = st.pending.take() {
                    st.busy = true;
                    break r;
                }
                st = shared.wake.wait(st).unwrap_or_else(PoisonError::into_inner);
            }
        };
        // a damaged file must never take the window down: draw black instead
        let frame = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| req.snapshot.render_frame(req.time, req.scale))).unwrap_or_else(|_| {
            log::error!("drawing the preview failed at {:?}", req.time);
            PreviewFrame::black(2, 2)
        });
        let mut st = lock(&shared.state);
        st.busy = false;
        // a result older than one already waiting is not worth showing
        if st.done.as_ref().is_none_or(|d| d.id < req.id) {
            st.done = Some(RenderedFrame { id: req.id, time: req.time, frame });
        }
        shared.ready.notify_all();
    }
}
