"""Python mirror of the editor's validation (backend-handoff.md section 6). The editor DROPS what fails and shows "N unusable entries ignored",
so every document this package returns must produce an empty issue list here."""
from __future__ import annotations

import json
import math
from typing import Any, Optional

from .config import SCHEMA, STAGE_CANDIDATES, STAGE_GROUPS

MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
DURATION_TOLERANCE_S = 0.5
END_BEYOND_FILE_S = 0.05
SPAN_TOLERANCE_S = 0.01


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _sentence_issues(s: Any, where: str, duration: float, seen: set) -> list[str]:
    if not isinstance(s, dict):
        return [f"{where}: not an object"]
    out = []
    sid = s.get("id")
    if not isinstance(sid, str) or not sid:
        return [f"{where}: empty or missing id"]
    if sid in seen:
        out.append(f"{where} {sid}: duplicate id")
    start, end = s.get("start"), s.get("end")
    if not (_num(start) and _num(end)):
        out.append(f"{where} {sid}: non-finite or missing times")
    else:
        if start >= end:
            out.append(f"{where} {sid}: start {start} is not before end {end}")
        if start < 0:
            out.append(f"{where} {sid}: start {start} < 0")
        if end > duration + END_BEYOND_FILE_S:
            out.append(f"{where} {sid}: end {end} is beyond the file ({duration}) by more than {END_BEYOND_FILE_S}")
    if not isinstance(s.get("text"), str):
        out.append(f"{where} {sid}: missing text")
    conf = s.get("confidence")
    if conf is not None and not (_num(conf) and 0.0 <= conf <= 1.0):
        out.append(f"{where} {sid}: confidence {conf} is not a number from 0 to 1")
    seen.add(sid)
    return out


def validate_document(doc: Any, file_duration: Optional[float] = None, expected_job: Optional[str] = None) -> list[str]:
    """Return a list of problems (empty = the editor would accept every entry). `file_duration` is the project file's length in seconds."""
    if not isinstance(doc, dict):
        return ["document is not a JSON object"]
    issues: list[str] = []
    if len(json.dumps(doc, ensure_ascii=False).encode("utf-8")) > MAX_DOCUMENT_BYTES:
        issues.append("document larger than 64 MiB")
    if doc.get("schema") != SCHEMA:
        issues.append(f"schema is {doc.get('schema')!r}, expected {SCHEMA!r}")
    stage = doc.get("stage")
    if stage not in (STAGE_CANDIDATES, STAGE_GROUPS):
        issues.append(f"stage is {stage!r}, expected 'candidates' or 'groups'")
    if not isinstance(doc.get("job"), str) or not doc.get("job"):
        issues.append("job missing")
    elif expected_job is not None and doc["job"] != expected_job:
        issues.append(f"job {doc['job']!r} differs from the question's job {expected_job!r}")
    media = doc.get("media")
    duration = None
    if not isinstance(media, dict) or not isinstance(media.get("path"), str) or not _num(media.get("duration_seconds")):
        issues.append("media.path / media.duration_seconds missing")
    else:
        duration = float(media["duration_seconds"])
        if file_duration is not None and abs(duration - file_duration) > DURATION_TOLERANCE_S:
            issues.append(f"media.duration_seconds {duration} differs from the file ({file_duration}) by more than {DURATION_TOLERANCE_S} s")
    if not isinstance(doc.get("generator"), dict) or not doc["generator"].get("name") or not doc["generator"].get("version"):
        issues.append("generator.name / generator.version missing")
    dur = duration if duration is not None else float("inf")

    if stage == STAGE_CANDIDATES:
        sentences, seen = doc.get("sentences"), set()
        if not isinstance(sentences, list):
            return issues + ["sentences missing"]
        by_id: dict[str, dict] = {}
        for s in sentences:
            issues += _sentence_issues(s, "sentence", dur, seen)
            if isinstance(s, dict) and isinstance(s.get("id"), str):
                by_id[s["id"]] = s
        pairs, pseen = doc.get("pairs"), set()
        if not isinstance(pairs, list):
            return issues + ["pairs missing"]
        for p in pairs:
            if not isinstance(p, dict):
                issues.append("pair: not an object")
                continue
            a, b, pid = p.get("a"), p.get("b"), p.get("id")
            if a not in by_id or b not in by_id:
                issues.append(f"pair {pid}: a sentence is missing")
                continue
            if a == b:
                issues.append(f"pair {pid}: a == b")
            if pid != f"{a}|{b}":
                issues.append(f"pair {pid}: id must be '<a>|<b>' ({a}|{b})")
            if pid in pseen:
                issues.append(f"pair {pid}: duplicate id")
            pseen.add(pid)
            if p.get("class") not in ("likely", "ambiguous"):
                issues.append(f"pair {pid}: class {p.get('class')!r} is not 'likely' or 'ambiguous'")
            if not (_num(p.get("score")) and 0.0 <= p["score"] <= 1.0):
                issues.append(f"pair {pid}: score {p.get('score')!r} is not a number from 0 to 1")
            if p.get("lean") is not None and p["lean"] not in ("retake", "different"):
                issues.append(f"pair {pid}: lean {p['lean']!r} is not 'retake' or 'different'")
            sa, sb = by_id[a], by_id[b]
            if _num(sa.get("start")) and _num(sb.get("start")) and (sa["start"], a) > (sb["start"], b):
                issues.append(f"pair {pid}: a must be the earlier sentence")
    elif stage == STAGE_GROUPS:
        groups = doc.get("groups")
        if not isinstance(groups, list):
            return issues + ["groups missing"]
        used: dict[str, str] = {}
        gseen: set = set()
        for g in groups:
            if not isinstance(g, dict):
                issues.append("group: not an object")
                continue
            gid = g.get("id")
            if not isinstance(gid, str) or not gid or gid in gseen:
                issues.append(f"group {gid!r}: empty or duplicate id")
            gseen.add(gid)
            sents = g.get("sentences")
            if not isinstance(sents, list) or len(sents) < 2:
                issues.append(f"group {gid}: fewer than 2 sentences")
                continue
            local: set = set()
            ok = []
            for s in sents:
                issues += _sentence_issues(s, f"group {gid} sentence", dur, local)
                if isinstance(s, dict) and _num(s.get("start")) and _num(s.get("end")):
                    ok.append(s)
                if isinstance(s, dict) and isinstance(s.get("id"), str):
                    if s["id"] in used and used[s["id"]] != gid:
                        issues.append(f"group {gid}: sentence {s['id']} is already in group {used[s['id']]}")
                    used[s["id"]] = gid
            if [s["start"] for s in ok] != sorted(s["start"] for s in ok):
                issues.append(f"group {gid}: sentences are not in video order")
            if not (_num(g.get("start")) and _num(g.get("end"))):
                issues.append(f"group {gid}: non-finite start/end")
            elif ok and (g["start"] > min(s["start"] for s in ok) + SPAN_TOLERANCE_S or g["end"] < max(s["end"] for s in ok) - SPAN_TOLERANCE_S):
                issues.append(f"group {gid}: span {g['start']}-{g['end']} does not cover its sentences")
            if not isinstance(g.get("likely"), bool):
                issues.append(f"group {gid}: 'likely' must be true or false")
    return issues


def assert_valid(doc: Any, file_duration: Optional[float] = None, expected_job: Optional[str] = None) -> None:
    issues = validate_document(doc, file_duration, expected_job)
    if issues:
        raise ValueError("document would be rejected or trimmed by the editor: " + "; ".join(issues[:8]))
