"""
Rich label storage: every user answer is appended to labels.jsonl BEFORE it is used to build groups (training data: same intended line = 1,
different = 0). Pairs the user skipped go to a separate file with label "skipped". LLM advice is never written here, and "source" is always "user".
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

from .config import ReviewConfig

logger = logging.getLogger(__name__)

SOURCE_USER = "user"
ASKED_BECAUSE = "unsure_band"


def _read_keys(path: Path) -> set[tuple]:
    keys: set[tuple] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                keys.add((r.get("job"), r.get("pair"), r.get("label")))
            except ValueError:
                continue
    return keys


def append_jsonl(path: Union[str, Path], record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def label_record(candidates: dict, pair: dict, label: str, now: Optional[float] = None) -> dict:
    """One labels.jsonl line for `pair` (an entry of candidates['pairs'])."""
    sents = {s["id"]: s for s in candidates.get("sentences", [])}
    side = lambda i: {"id": i, "text": sents[i]["text"], "start": sents[i]["start"], "end": sents[i]["end"]}
    th = candidates.get("thresholds") or {}
    score = pair.get("score")
    return {
        "time": time.time() if now is None else now,
        "job": candidates["job"],
        "pair": pair["id"],
        "label": label,
        "source": SOURCE_USER,
        "a": side(pair["a"]),
        "b": side(pair["b"]),
        "score": score,
        "ce": pair.get("ce", score),                           # cross-encoder probability
        "stacker_p": pair.get("stacker_p"),
        "model": (candidates.get("generator") or {}).get("model"),
        "thresholds_version": th.get("version"),
        "was_asked_because": ASKED_BECAUSE,
    }


def record_answers(candidates: dict, labels_request: Optional[dict], cfg: Optional[ReviewConfig] = None, now: Optional[float] = None) -> dict[str, int]:
    """Append the user's answers (and skipped pairs) to the label files. Repeatable: an identical (job, pair, label) line is written once."""
    cfg = cfg or ReviewConfig()
    by_id = {p["id"]: p for p in candidates.get("pairs", [])}
    done, skipped_done = _read_keys(Path(cfg.labels_path)), _read_keys(Path(cfg.skipped_path))
    written = skipped_written = 0
    for e in (labels_request or {}).get("labels", []) or []:
        pid, lab = e.get("pair"), e.get("label")
        if pid not in by_id or lab not in ("retake", "different"):
            logger.warning("[Labels] answer for %r (%r) ignored: unknown pair or label", pid, lab)
            continue
        rec = label_record(candidates, by_id[pid], lab, now)
        if (rec["job"], pid, lab) not in done:
            append_jsonl(cfg.labels_path, rec)
            done.add((rec["job"], pid, lab))
            written += 1
    for pid in (labels_request or {}).get("skipped", []) or []:
        if pid not in by_id:
            logger.warning("[Labels] skipped pair %r ignored: unknown pair", pid)
            continue
        rec = label_record(candidates, by_id[pid], "skipped", now)
        if (rec["job"], pid, "skipped") not in skipped_done:
            append_jsonl(cfg.skipped_path, rec)
            skipped_done.add((rec["job"], pid, "skipped"))
            skipped_written += 1
    return {"labels_written": written, "skipped_written": skipped_written}


def resolve_request(request: dict, cfg: Optional[ReviewConfig] = None, now: Optional[float] = None) -> dict:
    """`/v1/resolve` without HTTP: save the answers FIRST, then build the groups from the echoed candidates document alone."""
    from .json_export import build_groups

    cfg = cfg or ReviewConfig()
    candidates = request["candidates"]
    labels_request = request.get("labels")
    record_answers(candidates, labels_request, cfg, now)
    return build_groups(candidates, labels_request, cfg)
