//! Library thumbnails, drawn one after another on a worker thread.
//!
//! Unlike the preview renderer, every request matters: each imported file gets a thumbnail, in the
//! order they were asked for. A file that has no picture (audio) or cannot be read is reported as
//! having none, so the library shows an icon instead of waiting forever.

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use filmcraft_project::ItemId;
use filmcraft_time::Tick;

use crate::snapshot::{PreviewFrame, Snapshot};

/// Thumbnails are drawn at this fraction of full size.
const THUMB_SCALE: f32 = 0.2;

#[derive(Default)]
struct State {
    queue: VecDeque<(ItemId, Snapshot)>,
    asked: HashSet<ItemId>,
    ready: Vec<(ItemId, Option<PreviewFrame>)>,
    busy: bool,
    quit: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    done: Condvar,
}

fn lock(m: &Mutex<State>) -> MutexGuard<'_, State> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub struct Thumbnails {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl Default for Thumbnails {
    fn default() -> Self {
        Self::new()
    }
}

impl Thumbnails {
    pub fn new() -> Self {
        let shared = Arc::new(Shared { state: Mutex::default(), wake: Condvar::new(), done: Condvar::new() });
        let for_thread = shared.clone();
        let worker = std::thread::Builder::new().name("simplecut-thumbnails".into()).spawn(move || work(&for_thread)).ok();
        if worker.is_none() {
            log::error!("could not start the thumbnail thread; the library will show icons only");
        }
        Self { shared, worker }
    }

    /// Ask for `item`'s thumbnail, once: asking again for the same item does nothing until
    /// [`Self::forget`].
    pub fn request(&self, snapshot: Snapshot, item: ItemId) {
        let mut st = lock(&self.shared.state);
        if st.asked.insert(item) {
            st.queue.push_back((item, snapshot));
            self.shared.wake.notify_one();
        }
    }

    /// Allow `item` to be asked for again (after it was relinked, say).
    pub fn forget(&self, item: ItemId) {
        lock(&self.shared.state).asked.remove(&item);
    }

    /// Thumbnails finished since the last call. `None` means the file has no picture to show.
    pub fn take_ready(&self) -> Vec<(ItemId, Option<PreviewFrame>)> {
        std::mem::take(&mut lock(&self.shared.state).ready)
    }

    /// True when every requested thumbnail has been drawn and handed over or is waiting to be.
    pub fn is_idle(&self) -> bool {
        let st = lock(&self.shared.state);
        st.queue.is_empty() && !st.busy
    }

    /// Wait up to `timeout` until the queue is empty.
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut st = lock(&self.shared.state);
        while !(st.queue.is_empty() && !st.busy) {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else { return false };
            st = self.shared.done.wait_timeout(st, left).unwrap_or_else(PoisonError::into_inner).0;
        }
        true
    }
}

impl Drop for Thumbnails {
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
        let (item, snapshot) = {
            let mut st = lock(&shared.state);
            loop {
                if st.quit {
                    return;
                }
                if let Some(next) = st.queue.pop_front() {
                    st.busy = true;
                    break next;
                }
                st = shared.wake.wait(st).unwrap_or_else(PoisonError::into_inner);
            }
        };
        // a frame a little way in, so a clip that starts on black still shows something
        let at = Tick(snapshot.media_duration(item).0 / 10).min(Tick::from_seconds_f64(1.0));
        let frame = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| snapshot.render_media_frame(item, at, THUMB_SCALE))).unwrap_or_else(|_| {
            log::error!("drawing a thumbnail failed");
            None
        });
        let mut st = lock(&shared.state);
        st.busy = false;
        st.ready.push((item, frame));
        shared.done.notify_all();
    }
}
