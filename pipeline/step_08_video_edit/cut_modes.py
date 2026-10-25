"""
Cut modes: three ways to turn the SAME resolved discard decisions into keep spans.

    retake_only   complement of the discarded words (least intrusive)
    anchored      removals extended outward through dead air next to them, keeping a short stitch pause (default)
    sentences     only the kept words (+ padding) survive; long gaps between them are shortened

Every mode returns removal intervals that go through the same finalize() step (snap to the quietest frame, merge, invert, drop
fragments), so the cutlist JSON and the audio fades are identical in shape across modes.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from models.retake import CutList, TimeSpan

from .boundary_refiner import load_envelope, reference_level
from .cut_config import CutConfig
from .cut_generator import _extract_dict, compute_keep_spans, generate_cuts_from_adjudicated_spans, merge_spans, resolved_discard_unit_ids
from .visual_motion import motion_series

logger = logging.getLogger(__name__)

FRAME_S = 0.01      # must match boundary_refiner.load_envelope's default frame size
EPS = 1e-6


@dataclass
class Word:
    start: float
    end: float
    removed: bool = False


@dataclass
class SideInfo:
    stopped_by: str = "none"          # pad | word | energy | motion | cap | none
    absorbed_s: float = 0.0
    motion_max: float | None = None   # max visual motion seen in the candidate window (None: not measured)
    cap_truncated: bool = False


@dataclass
class Removal:
    start: float
    end: float
    left: SideInfo = field(default_factory=SideInfo)
    right: SideInfo = field(default_factory=SideInfo)
    lo: float = 0.0                   # snapping limits (never cut into a kept word / leave audio of a removed word)
    hi: float = 0.0
    min_start: float = 0.0            # earliest start must stay <= first removed word start
    max_end: float = 0.0              # end must stay >= last removed word end


# ----------------------------------------------------------------------------- words


def collect_words(transcript: Any, units: Sequence[Any], discard_unit_ids: Sequence[str]) -> tuple[list[Word], list[dict]]:
    """Words with a removed flag (word midpoint inside a discarded unit). Falls back to one pseudo-word per unit if the ASR has no word times."""
    udicts = [_extract_dict(u) for u in units]
    discard = set(discard_unit_ids)
    discard_spans = [(float(u["start"]), float(u["end"])) for u in udicts if str(u.get("unit_id")) in discard]
    segments = transcript["segments"] if isinstance(transcript, dict) else getattr(transcript, "segments", [])
    raw: list[tuple[float, float]] = []
    for seg in segments:
        for w in (seg.get("words") if isinstance(seg, dict) else seg.words) or []:
            s, e = (w["start"], w["end"]) if isinstance(w, dict) else (w.start, w.end)
            if s is not None and e is not None:
                raw.append((float(s), float(e)))
    if not raw:
        logger.warning("[CutModes] transcript has no word timestamps: using unit bounds as one word per unit")
        raw = [(float(u["start"]), float(u["end"])) for u in udicts]
    raw.sort()
    words = [Word(s, e, any(a - 0.02 <= (s + e) / 2.0 <= b + 0.02 for a, b in discard_spans)) for s, e in raw]
    return words, udicts


def _discard_word_bounds(words: Sequence[Word], udicts: Sequence[dict], discard_unit_ids: Sequence[str]) -> list[tuple[float, float]]:
    """Per discarded unit: first removed word start .. last removed word end (unit bounds if it has no words)."""
    out = []
    discard = set(discard_unit_ids)
    for u in udicts:
        if str(u.get("unit_id")) not in discard:
            continue
        a, b = float(u["start"]), float(u["end"])
        inside = [w for w in words if w.removed and a - 0.02 <= (w.start + w.end) / 2.0 <= b + 0.02]
        out.append((min(w.start for w in inside), max(w.end for w in inside)) if inside else (a, b))
    spans = merge_spans([TimeSpan(a, b) for a, b in out], tolerance=0.05)
    return [(s.start, s.end) for s in spans]


# ----------------------------------------------------------------------------- audio helpers


def _fi(t: float, n: int) -> int:
    return min(max(int(round(t / FRAME_S)), 0), n - 1)


def _quietest(env: np.ndarray, a: float, b: float) -> float:
    lo, hi = _fi(min(a, b), len(env)), _fi(max(a, b), len(env))
    return (lo + int(np.argmin(env[lo : hi + 1]))) * FRAME_S


def _speech_level(env: np.ndarray, word: Word | None) -> float:
    if word is not None:
        seg = env[_fi(word.start, len(env)) : _fi(word.end, len(env)) + 1]
        if seg.size:
            return float(np.percentile(seg, 90))
    return reference_level(env)


def dead_air_mask(env: np.ndarray, lo_t: float, hi_t: float, ref_db: float, cfg: CutConfig) -> tuple[int, np.ndarray]:
    """
    Boolean per 10 ms frame over [lo_t, hi_t]: energy-wise dead air. A frame is dead when >= dead_air_rel_db below the speech level; a short run
    (<= blip_max_s) of louder frames that still peaks >= blip_rel_db below speech (a breath or click) counts as non-speech as well.
    Returns (index of first frame, mask).
    """
    margin = int(cfg.blip_max_s / FRAME_S) + 2
    a = max(_fi(lo_t, len(env)) - margin, 0)
    b = min(_fi(hi_t, len(env)) + margin + 1, len(env))
    seg = env[a:b]
    quiet = seg < (ref_db - cfg.dead_air_rel_db)
    dead = quiet.copy()
    i, n = 0, len(seg)
    while i < n:
        if quiet[i]:
            i += 1
            continue
        j = i
        while j < n and not quiet[j]:
            j += 1
        bounded = i > 0 and j < n                                    # a run touching the window edge may be longer than we can see
        if bounded and (j - i) * FRAME_S <= cfg.blip_max_s and seg[i:j].max() < ref_db - cfg.blip_rel_db:
            dead[i:j] = True
        i = j
    return a, dead


# ----------------------------------------------------------------------------- finalize


def _constraints(removals: list[Removal], words: Sequence[Word], duration: float, cfg: CutConfig) -> None:
    kept = [w for w in words if not w.removed]
    for r in removals:
        before = [w.end for w in kept if w.end <= r.start + 0.02]
        after = [w.start for w in kept if w.start >= r.end - 0.02]
        r.lo = max(before) if before else 0.0
        r.hi = min(after) if after else duration
        inside = [w for w in words if w.removed and r.start - 0.02 <= (w.start + w.end) / 2.0 <= r.end + 0.02]
        r.min_start = min((w.start for w in inside), default=r.start)
        r.max_end = max((w.end for w in inside), default=r.end)
        if not inside:                                   # pure dead-air removal: either edge may move, but the stitch padding stays
            r.lo = r.lo + cfg.tail_pad_s if before else r.lo
            r.hi = r.hi - cfg.head_pad_s if after else r.hi
            r.min_start, r.max_end = r.end, r.start


def finalize(
    removals: list[Removal],
    words: Sequence[Word],
    env: np.ndarray,
    duration: float,
    cfg: CutConfig,
) -> tuple[list[TimeSpan], list[TimeSpan], list[Removal]]:
    """Snap every cut to the quietest frame within +/- snap_ms (never into a kept word), merge removals, invert into keep spans."""
    _constraints(removals, words, duration, cfg)
    snap = cfg.snap_ms / 1000.0
    snapped: list[Removal] = []
    for r in removals:
        s, e = r.start, r.end
        if s > EPS:
            a, b = max(r.lo, s - snap), min(r.min_start, s + snap)
            if a <= b:
                s = _quietest(env, a, b)
        if e < duration - EPS:
            a, b = max(r.max_end, e - snap), min(r.hi, e + snap)
            if a <= b:
                e = _quietest(env, a, b)
        if e > s + EPS:
            snapped.append(Removal(round(s, 4), round(e, 4), r.left, r.right, r.lo, r.hi, r.min_start, r.max_end))
    snapped.sort(key=lambda r: r.start)
    merged: list[Removal] = []
    for r in snapped:
        if merged and r.start <= merged[-1].end + EPS:
            m = merged[-1]
            if r.end >= m.end:
                m.right = r.right
            m.end = max(m.end, r.end)
        else:
            merged.append(Removal(r.start, r.end, r.left, r.right, r.lo, r.hi, r.min_start, r.max_end))
    discards = [TimeSpan(r.start, r.end, "cut") for r in merged]
    keep = compute_keep_spans(discards, source_duration=duration, min_fragment_duration=cfg.min_fragment_s) if discards else [TimeSpan(0.0, round(duration, 4), "keep_all")]
    return keep, discards, merged


# ----------------------------------------------------------------------------- modes


def removals_retake_only(words: Sequence[Word], bounds: Sequence[tuple[float, float]]) -> list[Removal]:
    return [Removal(a, b) for a, b in bounds]


def _gap_removals(words: Sequence[Word], env: np.ndarray, cfg: CutConfig, longer_than: float) -> list[Removal]:
    """
    For every gap between consecutive kept words: if it holds a removed word, remove everything between the stitch paddings; otherwise, if
    it is longer than `longer_than` (and than max_kept_pause_s), remove the middle so max_kept_pause_s remains, centred on the quietest frame
    (never leaving less than the stitch padding at either side).
    """
    kept = [w for w in words if not w.removed]
    out: list[Removal] = []
    for a, b in zip(kept, kept[1:]):
        gap = b.start - a.end
        between = [w for w in words if w.removed and w.start >= a.end - EPS and w.end <= b.start + EPS]
        if between:                                     # padding never reaches into the removed words' audio
            s = min(a.end + cfg.tail_pad_s, min(w.start for w in between))
            e = max(b.start - cfg.head_pad_s, max(w.end for w in between))
            if e > s + EPS:
                out.append(Removal(round(s, 4), round(e, 4)))
            continue
        if gap <= longer_than or gap <= cfg.max_kept_pause_s + EPS:
            continue
        cut = gap - cfg.max_kept_pause_s
        lo_c = a.end + cfg.tail_pad_s + cut / 2.0
        hi_c = b.start - cfg.head_pad_s - cut / 2.0
        if hi_c < lo_c:
            lo_c = hi_c = (a.end + b.start) / 2.0
        c = _quietest(env, lo_c, hi_c) if hi_c > lo_c else lo_c
        c = min(max(c, lo_c), hi_c)
        out.append(Removal(round(c - cut / 2.0, 4), round(c + cut / 2.0, 4)))
    return out


def removals_sentences(words: Sequence[Word], env: np.ndarray, duration: float, cfg: CutConfig) -> list[Removal]:
    kept = [w for w in words if not w.removed]
    if not kept:
        return [Removal(0.0, duration)]
    out: list[Removal] = []
    first = max(kept[0].start - cfg.head_pad_s, 0.0)
    if first > EPS:
        out.append(Removal(0.0, first))
    last = min(kept[-1].end + cfg.tail_pad_s, duration)
    if duration - last > EPS:
        out.append(Removal(last, duration))
    out += _gap_removals(words, env, cfg, longer_than=0.0)
    return out


def _walk(
    env: np.ndarray, words: Sequence[Word], start: float, direction: int, limit: float, ref_db: float, cfg: CutConfig,
    video_path: Any, duration: float,
) -> tuple[float, SideInfo]:
    """Extend a removal edge from `start` outward (direction -1 left, +1 right) through dead air; returns (new edge, why it stopped)."""
    info = SideInfo()
    cap = cfg.max_absorb_s
    if direction < 0:
        reach = max(limit, start - cap, 0.0)
        if reach >= start - EPS:
            info.stopped_by = "pad"
            return start, info
        win = (reach, start)
    else:
        reach = min(limit, start + cap, duration)
        if reach <= start + EPS:
            info.stopped_by = "pad"
            return start, info
        win = (start, reach)
    base, dead = dead_air_mask(env, win[0], win[1], ref_db, cfg)
    motion = None
    if cfg.use_visual_motion and video_path is not None:
        try:
            motion = motion_series(video_path, win[0], win[1], cfg.motion_fps)
        except Exception as exc:                                    # no video stream / ffmpeg problem: audio-only decision
            logger.warning("[CutModes] motion sampling failed (%s); using audio only", exc)
    if motion is not None and motion.size:
        info.motion_max = 0.0
    k = int(round(start / FRAME_S))
    n = len(env)

    def frame_ok(idx: int) -> tuple[bool, str]:
        t0, t1 = idx * FRAME_S, (idx + 1) * FRAME_S
        if any(w.end > t0 + EPS and w.start < t1 - EPS for w in words):
            return False, "word"
        if idx < 0 or idx >= n or not dead[min(max(idx - base, 0), len(dead) - 1)]:
            return False, "energy"
        if motion is not None and motion.size:
            m = float(motion[min(max(int((((t0 + t1) / 2.0) - win[0]) * cfg.motion_fps), 0), motion.size - 1)])
            info.motion_max = max(info.motion_max or 0.0, m)
            if m > cfg.motion_threshold:
                return False, "motion"
        return True, ""

    reason = "pad"
    while True:
        idx = k - 1 if direction < 0 else k
        t_far = idx * FRAME_S if direction < 0 else (idx + 1) * FRAME_S
        beyond = (t_far < win[0] - EPS) if direction < 0 else (t_far > win[1] + EPS)
        if beyond:
            reason = "pad" if abs((win[0] if direction < 0 else win[1]) - limit) < 1e-3 else "cap"
            break
        ok, why = frame_ok(idx)
        if not ok:
            reason = why
            break
        k += direction
    edge = k * FRAME_S
    if reason == "pad":
        edge = limit                                                 # exact stitch padding
    if reason == "cap":
        nxt = k - 1 if direction < 0 else k
        info.cap_truncated = frame_ok(nxt)[0] and (nxt * FRAME_S >= limit - EPS if direction < 0 else (nxt + 1) * FRAME_S <= limit + EPS)
    info.stopped_by = reason
    info.absorbed_s = round(abs(edge - start), 4)
    return round(edge, 4), info


def removals_anchored(
    words: Sequence[Word], bounds: Sequence[tuple[float, float]], env: np.ndarray, duration: float, cfg: CutConfig, video_path: Any,
) -> list[Removal]:
    kept = [w for w in words if not w.removed]
    out: list[Removal] = []
    for s, e in bounds:
        prev = max((w for w in kept if w.end <= s + 0.02), key=lambda w: w.end, default=None)
        nxt = min((w for w in kept if w.start >= e - 0.02), key=lambda w: w.start, default=None)
        limit_l = (prev.end + cfg.tail_pad_s) if prev else 0.0
        limit_r = (nxt.start - cfg.head_pad_s) if nxt else duration
        limit_l, limit_r = min(limit_l, s), max(limit_r, e)
        new_s, left = _walk(env, words, s, -1, limit_l, _speech_level(env, prev), cfg, video_path, duration)
        new_e, right = _walk(env, words, e, +1, limit_r, _speech_level(env, nxt), cfg, video_path, duration)
        logger.info("[Anchored] removal %.2f-%.2f -> %.2f-%.2f | left %s (%.2fs, motion %s) right %s (%.2fs, motion %s)", s, e, new_s, new_e,
                    left.stopped_by, left.absorbed_s, left.motion_max, right.stopped_by, right.absorbed_s, right.motion_max)
        out.append(Removal(new_s, new_e, left, right))
    return out


# ----------------------------------------------------------------------------- entry point


def build_cutlist(
    cut_mode: str,
    adjudicated_spans: Sequence[Any],
    units: Sequence[Any],
    transcript: Any,
    source: str,
    source_duration: float,
    envelope: np.ndarray | None = None,
    video_path: Any = None,
    cfg: CutConfig | None = None,
) -> tuple[CutList, list[dict]]:
    """Build the CutList for `cut_mode` from the resolved adjudicated spans. Returns (cutlist, csv_rows)."""
    cfg = cfg or CutConfig(cut_mode=cut_mode)
    if cfg.cut_mode != cut_mode:
        cfg = CutConfig(**{**cfg.__dict__, "cut_mode": cut_mode})
    base = generate_cuts_from_adjudicated_spans(adjudicated_spans, units, source, source_duration)   # retake_groups + resolver (unchanged)
    discard_ids = resolved_discard_unit_ids(adjudicated_spans)
    env = envelope if envelope is not None else load_envelope(video_path)
    words, udicts = collect_words(transcript, units, discard_ids)
    bounds = _discard_word_bounds(words, udicts, discard_ids)

    if cut_mode == "retake_only":
        removals = removals_retake_only(words, bounds)
    elif cut_mode == "sentences":
        removals = removals_sentences(words, env, source_duration, cfg)
    else:
        removals = removals_anchored(words, bounds, env, source_duration, cfg, video_path)
        if cfg.trim_long_silence:
            removals += _gap_removals(words, env, cfg, longer_than=cfg.trim_long_silence_s)

    keep, discards, merged = finalize(removals, words, env, source_duration, cfg)
    removed = round(sum(d.duration for d in discards), 4)
    cutlist = CutList(source=str(source), source_duration=round(float(source_duration), 4), keep_spans=keep, discard_spans=discards,
                      retake_groups=base.retake_groups, mode=cut_mode, removed_seconds=removed)
    rows = keep_vs_units_rows(cutlist, words, udicts, discard_ids, merged, video_path, cfg)
    return cutlist, rows


# ----------------------------------------------------------------------------- CSV


CSV_COLUMNS = ["row_type", "mode", "start", "end", "n_kept_words", "kept_units", "lead_s", "trail_s", "touches_cut",
               "start_join_stopped_by", "start_join_motion", "start_join_cap_truncated", "end_join_stopped_by", "end_join_motion",
               "end_join_cap_truncated", "note"]


def keep_vs_units_rows(cutlist: CutList, words: Sequence[Word], udicts: Sequence[dict], discard_ids: Sequence[str],
                       removals: Sequence[Removal], video_path: Any, cfg: CutConfig) -> list[dict]:
    kept_words = [w for w in words if not w.removed]
    discard = set(discard_ids)
    rows: list[dict] = []
    dur = cutlist.source_duration
    for k in cutlist.keep_spans:
        inside = [w for w in kept_words if w.start >= k.start - 0.01 and w.end <= k.end + 0.01]
        units_in = [str(u["unit_id"]) for u in udicts if str(u["unit_id"]) not in discard
                    and min(u["end"], k.end) - max(u["start"], k.start) > 0.5 * (u["end"] - u["start"])]
        before = next((r for r in removals if abs(r.end - k.start) < 0.02), None)
        after = next((r for r in removals if abs(r.start - k.end) < 0.02), None)
        rows.append({
            "row_type": "keep_span", "mode": cutlist.mode, "start": round(k.start, 3), "end": round(k.end, 3), "n_kept_words": len(inside),
            "kept_units": " ".join(units_in), "lead_s": round(inside[0].start - k.start, 3) if inside else "",
            "trail_s": round(k.end - inside[-1].end, 3) if inside else "",
            "touches_cut": bool(before or after),
            "start_join_stopped_by": before.right.stopped_by if before else "", "start_join_motion": _fmt(before.right.motion_max) if before else "",
            "start_join_cap_truncated": before.right.cap_truncated if before else "",
            "end_join_stopped_by": after.left.stopped_by if after else "", "end_join_motion": _fmt(after.left.motion_max) if after else "",
            "end_join_cap_truncated": after.left.cap_truncated if after else "", "note": "" if inside else "no kept words",
        })
    for d in cutlist.discard_spans:
        if d.duration <= cfg.lost_gameplay_min_s or any(w.end > d.start and w.start < d.end for w in words):
            continue
        m = None
        if video_path is not None:
            try:
                series = motion_series(video_path, d.start, d.end, cfg.motion_fps)
                m = float(series.mean()) if series.size else None
            except Exception:
                m = None
        if m is not None and m > cfg.motion_threshold:
            rows.append({"row_type": "removed_flag", "mode": cutlist.mode, "start": round(d.start, 3), "end": round(d.end, 3),
                         "note": f"removed {d.duration:.1f}s with no ASR words and mean motion {m:.2f} (> {cfg.motion_threshold}): possible lost footage"})
    return rows


def _fmt(v: float | None) -> Any:
    return "" if v is None else round(v, 3)


def write_keep_spans_vs_units(rows: Sequence[dict], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CSV_COLUMNS})
    return p
