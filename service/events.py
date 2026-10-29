"""Structured pipeline events: {seq, job, stage, status (started|progress|done|error), pct (optional), message, timestamp}.

An EventLog keeps the events of one job in memory (so a client can poll or stream them) and appends each one to the project's progress.jsonl when the job belongs to a
project, so the log is still there after the job and after a restart. Emitting never blocks on a client: it takes a short lock and appends. A stage that was skipped
because its result already existed is reported as status "done" with cached=true and the message "cached" (shown as "cached" in the UI, never hidden).
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

STARTED, PROGRESS, DONE, ERROR = "started", "progress", "done", "error"
STATUSES = (STARTED, PROGRESS, DONE, ERROR)

# the stepper, in order: (id, label)
STAGES = (
    ("model_download", "Download model"),
    ("model_load", "Load model"),
    ("audio_extraction", "Extract audio"),
    ("vad", "Detect speech (VAD)"),
    ("transcription", "Transcribe"),
    ("alignment", "Align words"),
    ("unit_splitting", "Split sentences"),
    ("pair_scoring", "Score sentence pairs"),
    ("keeper_selection", "Choose keepers"),
    ("cutlist_build", "Build cut list"),
    ("export_render", "Render export"),
)
STAGE_IDS = tuple(s for s, _ in STAGES)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class EventLog:
    def __init__(self, job: str, persist_path: Optional[Path] = None):
        self.job = job
        self._events: list[dict] = []
        self._lock = threading.Lock()
        self._persist = Path(persist_path) if persist_path else None
        self._closed = False

    def emit(self, stage: str, status: str, message: str = "", pct: Optional[float] = None, **extra: Any) -> dict:
        if stage not in STAGE_IDS:
            raise ValueError(f"unknown stage '{stage}'")
        if status not in STATUSES:
            raise ValueError(f"unknown status '{status}'")
        with self._lock:
            ev = {"seq": len(self._events) + 1, "job": self.job, "stage": stage, "status": status, "message": message, "timestamp": now_iso()}
            if pct is not None:
                ev["pct"] = round(max(0.0, min(100.0, float(pct))), 1)
            ev.update(extra)
            self._events.append(ev)
            if self._persist is not None:
                try:
                    self._persist.parent.mkdir(parents=True, exist_ok=True)
                    with open(self._persist, "a", encoding="utf-8") as f:
                        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                except OSError:
                    pass                                                       # a full disk must not fail the job; the in-memory log still has the event
            return ev

    def since(self, seq: int = 0) -> list[dict]:
        with self._lock:
            return list(self._events[seq:])

    def close(self) -> None:
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    # ---- helpers for the pipeline code ----
    @contextmanager
    def stage(self, stage: str, message: str = "") -> Iterator["StageHandle"]:
        """started -> (progress)* -> done; an exception becomes an error event with the exact message and is re-raised."""
        h = StageHandle(self, stage)
        self.emit(stage, STARTED, message)
        try:
            yield h
        except BaseException as exc:                                           # includes JobCancelled: shown as an error on the stage that was running
            name = type(exc).__name__
            self.emit(stage, ERROR, "Cancelled" if name == "JobCancelled" else f"{name}: {exc}" if str(exc) else name)
            raise
        self.emit(stage, DONE, h.done_message, pct=100.0)

    def cached(self, stage: str) -> None:
        self.emit(stage, DONE, "cached", pct=100.0, cached=True)


class StageHandle:
    def __init__(self, log: EventLog, stage: str):
        self.log, self.stage, self.done_message = log, stage, ""
        self._last = -1.0

    def progress(self, pct: float, message: str = "") -> None:
        if pct - self._last >= 1.0 or pct >= 100.0:                            # at most one event per percent
            self._last = pct
            self.log.emit(self.stage, PROGRESS, message, pct=pct)


class NullEvents(EventLog):
    """Used when nobody listens (scripts, old callers): the same interface, nothing stored."""

    def __init__(self):
        super().__init__("none")

    def emit(self, stage: str, status: str, message: str = "", pct: Optional[float] = None, **extra: Any) -> dict:
        return {}


def read_log(path: Path) -> list[dict]:
    """The persisted events of a project (progress.jsonl); unreadable lines are skipped."""
    out: list[dict] = []
    if Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def sse_format(ev: dict) -> str:
    return f"id: {ev['seq']}\nevent: stage\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"


def make_throttle(fn: Callable[[float], None], min_interval_s: float = 0.2) -> Callable[[float], None]:
    last = [0.0]

    def call(pct: float) -> None:
        t = time.monotonic()
        if pct >= 100.0 or t - last[0] >= min_interval_s:
            last[0] = t
            fn(pct)
    return call
