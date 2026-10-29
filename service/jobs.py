"""In-memory job state machine: one worker thread per job, a cancel flag checked at every progress tick (between stages), and a lock that lets
only ONE heavy analyze job run at a time (the others wait in the queue; resolve / adjudicate run alongside)."""
from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .events import EventLog

logger = logging.getLogger("service.jobs")

RUNNING, DONE, FAILED, CANCELLED = "running", "done", "failed", "cancelled"


class JobCancelled(Exception):
    pass


class UserError(Exception):
    """An error whose message is already a plain sentence the user can read (shown verbatim)."""


@dataclass
class Job:
    id: str
    kind: str
    state: str = RUNNING
    fraction: float = 0.0
    status: str = "Starting"
    result: Any = None
    error: Optional[str] = None
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    events: EventLog = field(default_factory=lambda: EventLog(""))

    def snapshot(self) -> dict:
        with self.lock:
            out = {"state": self.state, "fraction": round(self.fraction, 4), "status": self.status}
            if self.state == DONE:
                out["result"] = self.result
            elif self.state == FAILED:
                out["error"] = self.error
            return out


class JobContext:
    def __init__(self, job: Job):
        self.job = job

    @property
    def id(self) -> str:
        return self.job.id

    @property
    def events(self) -> EventLog:
        return self.job.events

    def check(self) -> None:
        if self.job.cancel.is_set():
            raise JobCancelled()

    def progress(self, status: str, fraction: float) -> None:
        """Called between stages: records progress and honours a cancel request."""
        self.check()
        with self.job.lock:
            if self.job.state == RUNNING:
                self.job.status, self.job.fraction = status, max(0.0, min(1.0, float(fraction)))


class JobManager:
    def __init__(self, max_kept: int = 100):
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._mu = threading.Lock()
        self._heavy = threading.Lock()
        self._max_kept = max_kept

    def new_id(self, prefix: str = "j") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def get(self, job_id: str) -> Optional[Job]:
        with self._mu:
            return self._jobs.get(job_id)

    def submit(self, kind: str, fn: Callable[[JobContext], Any], heavy: bool = False, job_id: Optional[str] = None, user_message: str = "The retake service could not finish",
               events_path: Optional[Path] = None) -> str:
        job = Job(id=job_id or self.new_id("j" if heavy else "r"), kind=kind)
        job.events = EventLog(job.id, events_path)          # events_path: <project>/data/progress.jsonl, so the log outlives the job
        with self._mu:
            self._jobs[job.id] = job
            self._evict()
        threading.Thread(target=self._run, args=(job, fn, heavy), name=f"job-{job.id}", daemon=True).start()
        return job.id

    def cancel(self, job_id: str) -> None:
        """Idempotent: unknown or finished jobs are left alone."""
        job = self.get(job_id)
        if job is None:
            return
        job.cancel.set()
        with job.lock:
            if job.state == RUNNING:
                job.status = "Cancelling"

    def _evict(self) -> None:
        while len(self._jobs) > self._max_kept:
            for jid, j in self._jobs.items():
                if j.state != RUNNING:
                    del self._jobs[jid]
                    break
            else:
                return

    def _finish(self, job: Job, state: str, **kw: Any) -> None:
        with job.lock:
            if job.state != RUNNING:
                return
            job.state = state
            job.result, job.error = kw.get("result"), kw.get("error")
            if state == DONE:
                job.fraction, job.status = 1.0, "Done"
            elif state == CANCELLED:
                job.status = "Cancelled"

    def _run(self, job: Job, fn: Callable[[JobContext], Any], heavy: bool) -> None:
        ctx = JobContext(job)
        acquired = False
        try:
            if heavy:
                ctx.progress("Waiting for another analysis", 0.0)
                while not self._heavy.acquire(timeout=0.2):
                    ctx.check()                                    # a queued job can be cancelled before it starts
                acquired = True
            ctx.check()
            result = fn(ctx)
            ctx.check()
            self._finish(job, DONE, result=result)
        except JobCancelled:
            self._finish(job, CANCELLED)
        except UserError as exc:
            self._finish(job, FAILED, error=str(exc))
        except FileNotFoundError:
            self._finish(job, FAILED, error="Could not read the file")
        except Exception:                                          # never leak stack traces or server paths to the editor
            logger.exception("job %s (%s) failed", job.id, job.kind)
            # the editor adds "The retake service could not finish:" itself; the job id is the key into the service log
            self._finish(job, FAILED, error=f"Something went wrong while processing the file (job {job.id}). See the service log.")
        finally:
            job.events.close()
            if acquired:
                self._heavy.release()
