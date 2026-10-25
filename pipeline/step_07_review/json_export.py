"""
Builders for the editor's review documents (backend-handoff.md, schema "simplecut.retakes/2"):

    build_candidates(...)  sentences + pairs (likely / ambiguous) from scored pairs; the unsure band decides which pairs are asked
    build_groups(...)      groups from the ECHOED candidates document alone + the user's labels (no model, no earlier job state)
    analyze_media(...)     transcribe the WHOLE file once, cache transcript + units + pair scores by media fingerprint, return candidates

The service finds and groups sentences; it never says what to keep.
"""
from __future__ import annotations

import logging
import math
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from pipeline.step_05_span_aggregation.constrained_clustering import Edge, cluster_with_constraints

from .cache import MediaCache, media_fingerprint
from .config import GENERATOR_NAME, GENERATOR_VERSION, SCHEMA, STAGE_CANDIDATES, STAGE_GROUPS, ReviewConfig

logger = logging.getLogger(__name__)

Used = Optional[Sequence[Sequence[float]]]


# ----------------------------------------------------------------------------- sentences (word-based times)


def _words_by_segment(transcript: dict) -> dict[int, list[tuple[float, float]]]:
    out: dict[int, list[tuple[float, float]]] = {}
    for seg in transcript.get("segments", []):
        ws = [(float(w["start"]), float(w["end"])) for w in (seg.get("words") or []) if w.get("start") is not None and w.get("end") is not None]
        out[int(seg["id"])] = sorted(ws)
    return out


def sentence_bounds(unit: dict, words: dict[int, list[tuple[float, float]]]) -> tuple[float, float, bool]:
    """(start, end, from_words): first word start .. last word end of the unit's words; unit bounds (from_words=False) if it has none."""
    u0, u1 = float(unit["start"]), float(unit["end"])
    inside = [w for w in words.get(int(unit.get("source_segment_id", -1)), []) if u0 - 0.02 <= (w[0] + w[1]) / 2.0 <= u1 + 0.02]
    if inside:
        return inside[0][0], inside[-1][1], True
    return u0, u1, False


def _in_used(start: float, end: float, used: Used) -> bool:
    if not used:
        return True
    return any(end > float(r[0]) and start < float(r[1]) for r in used)


def _finite01(x: Any) -> Optional[float]:
    return float(min(max(x, 0.0), 1.0)) if isinstance(x, (int, float)) and math.isfinite(x) else None


# ----------------------------------------------------------------------------- candidates


def thresholds_block(engine: Any) -> dict[str, Any]:
    """Everything build_groups needs from the model side, echoed inside the candidates document."""
    return {"high": engine.high, "low": engine.low, "t_lo": engine.t_lo, "t_hi": engine.t_hi, "t_amb": engine.t_amb,
            "unsure_band_enabled": engine.unsure_band_enabled, "max_questions": engine.max_questions, "version": engine.thresholds_version,
            "scorer": engine.scorer_name}


def build_candidates(
    units: Sequence[dict],
    scores: Sequence[dict],
    transcript: dict,
    engine: Any,
    media: dict,
    job: str,
    used: Used = None,
    cfg: Optional[ReviewConfig] = None,
) -> dict:
    """
    units: unit dicts (unit_id, source_segment_id, start, end, raw_text, confidence).
    scores: one dict per scored pair {a, b, ce_prob, stacker_p, plain_stage2} for the WHOLE file (pairs are filtered by `used` here, AFTER scoring).
    engine: a FinalScorer (or any object with its thresholds and resolve()); decides likely / ambiguous / dropped per pair.
    """
    cfg = cfg or ReviewConfig()
    words = _words_by_segment(transcript)
    sentences: dict[str, dict] = {}
    n_fallback = 0
    for u in units:
        s, e, from_words = sentence_bounds(u, words)
        n_fallback += 0 if from_words else 1
        if not (math.isfinite(s) and math.isfinite(e) and e > s):
            continue
        if not _in_used(s, e, used):
            continue
        rec = {"id": str(u["unit_id"]), "start": round(s, 4), "end": round(e, 4), "text": str(u.get("raw_text", "")).strip()}
        conf = _finite01(u.get("confidence"))
        if conf is not None:
            rec["confidence"] = round(conf, 4)
        sentences[rec["id"]] = rec
    if n_fallback:
        logger.info("[Review] %d sentence(s) had no word timestamps: unit bounds used", n_fallback)

    rows = []
    for sc in scores:
        a, b = str(sc["a"]), str(sc["b"])
        if a not in sentences or b not in sentences or a == b:
            continue                                                       # filtered by `used` after scoring
        if (sentences[a]["start"], a) > (sentences[b]["start"], b):
            a, b = b, a
        rows.append((sentences[a]["start"], sentences[b]["start"], a, b, sc))
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    seen: set = set()
    rows = [r for r in rows if not ((r[2], r[3]) in seen or seen.add((r[2], r[3])))]

    p = np.array([float(r[4]["stacker_p"]) for r in rows])
    ce = np.array([float(r[4]["ce_prob"]) if r[4].get("ce_prob") is not None else float("nan") for r in rows])
    decisions, resolved_by = engine.resolve(p, ce) if len(rows) else ([], [])
    pairs = []
    for (_, _, a, b, sc), pv, cv, dec, by in zip(rows, p, ce, decisions, resolved_by):
        if dec == "retake":
            cls = "likely"
        elif dec == "ambiguous":
            cls = "ambiguous"
        else:
            continue                                                       # decided "different": dropped
        ce_ok = math.isfinite(cv)
        entry = {"id": f"{a}|{b}", "a": a, "b": b, "class": cls, "stacker_p": float(pv), "resolved_by": by}
        if ce_ok:
            entry["ce"] = float(cv)
        if cls == "likely":
            entry["score"] = round(_finite01(pv), 6)                    # confident pairs: the stacker probability
        else:
            entry["score"] = round(_finite01(cv) if ce_ok else _finite01(pv), 6)      # asked pairs: the cross-encoder probability
            mid = (engine.t_lo + engine.t_hi) / 2.0 if engine.t_lo is not None else 0.5
            entry["lean"] = "retake" if (cv if ce_ok else pv) >= mid else "different"
        pairs.append(entry)
    ordered = sorted(sentences.values(), key=lambda s: (s["start"], s["id"]))
    return {
        "schema": SCHEMA, "stage": STAGE_CANDIDATES, "job": job, "media": media,
        "generator": {"name": cfg.generator_name, "version": cfg.generator_version, "model": engine.model_version or engine.scorer_name},
        "thresholds": thresholds_block(engine),
        "sentences": ordered, "pairs": pairs,
    }


# ----------------------------------------------------------------------------- groups


def labels_by_pair(labels_request: Optional[dict]) -> tuple[dict[str, str], set[str]]:
    """({pair id: 'retake' | 'different'}, {skipped pair ids}). Later entries win; entries with another label value are ignored."""
    lab: dict[str, str] = {}
    for e in (labels_request or {}).get("labels", []) or []:
        if isinstance(e, dict) and e.get("label") in ("retake", "different") and isinstance(e.get("pair"), str):
            lab[e["pair"]] = e["label"]
    return lab, {s for s in (labels_request or {}).get("skipped", []) or [] if isinstance(s, str)}


def build_groups(candidates: dict, labels_request: Optional[dict] = None, cfg: Optional[ReviewConfig] = None) -> dict:
    """
    Groups from the echoed candidates document alone plus the labels. Likely pairs are edges (weight = stacker_p); a user "retake" is a must-link
    (weight 1.0), a user "different" a cannot-link; skipped / unanswered ambiguous pairs are decided by score >= t_hi. A group is "likely" only if
    every edge in it is confident (stacker_p >= high, or a user "retake"); an edge decided by the cross-encoder alone makes it false.
    """
    cfg = cfg or ReviewConfig()
    th = candidates.get("thresholds") or {}
    high = float(th["high"]) if th.get("high") is not None else float("inf")
    t_hi = float(th["t_hi"]) if th.get("t_hi") is not None else (float(th["t_amb"]) if th.get("t_amb") is not None else float("inf"))
    sents = {s["id"]: s for s in candidates.get("sentences", [])}
    labels, _skipped = labels_by_pair(labels_request)
    known = {p["id"] for p in candidates.get("pairs", [])}
    for pid in labels:
        if pid not in known:
            logger.warning("[Review] label for unknown pair %s ignored", pid)

    model_edges: list[Edge] = []
    must: list[Edge] = []
    cannot: list[tuple[str, str]] = []
    for p in candidates.get("pairs", []):
        a, b, pid = p["a"], p["b"], p["id"]
        if a not in sents or b not in sents:
            continue
        lab = labels.get(pid)
        if lab == "retake":
            must.append(Edge(a, b, 1.0, "user", True))
        elif lab == "different":
            cannot.append((a, b))
        else:
            sp = p.get("stacker_p")
            if not isinstance(sp, (int, float)) and p.get("class") == "likely":
                sp = p.get("score")                                      # for likely pairs the score IS the stacker probability
            weight = float(sp) if isinstance(sp, (int, float)) else float(p.get("score", 0.0))
            decided = p.get("class") == "likely" or (p.get("class") == "ambiguous" and float(p.get("score", 0.0)) >= t_hi)
            if decided:
                model_edges.append(Edge(a, b, weight, "model", p.get("class") == "likely" and isinstance(sp, (int, float)) and float(sp) >= high))
    res = cluster_with_constraints(model_edges, must, cannot, min_size=2)
    confident = {frozenset((e.a, e.b)): bool(e.tag) for e in res.accepted}
    groups = []
    for members in res.components:
        m = set(members)
        ss = sorted((sents[i] for i in members), key=lambda s: (s["start"], s["id"]))
        edges_in = [e for e in res.accepted if e.a in m and e.b in m]
        groups.append({
            "id": f"g_{ss[0]['id']}",
            "start": min(s["start"] for s in ss), "end": max(s["end"] for s in ss),          # exact: earliest member start / latest member end
            "likely": bool(edges_in) and all(confident[frozenset((e.a, e.b))] for e in edges_in),
            "sentences": [{"id": s["id"], "start": s["start"], "end": s["end"], "text": s["text"]} for s in ss],
        })
    groups.sort(key=lambda g: (g["start"], g["id"]))
    return {"schema": SCHEMA, "stage": STAGE_GROUPS, "job": candidates["job"], "media": candidates["media"],
            "generator": candidates.get("generator", {"name": GENERATOR_NAME, "version": GENERATOR_VERSION}), "groups": groups}


# ----------------------------------------------------------------------------- analyze (transcribe once, cache, score)


def default_transcribe(cfg: ReviewConfig) -> Callable[[Path], Any]:
    def run(path: Path):
        from pipeline.step_01_transcription import TimestampGenerator
        return TimestampGenerator(**cfg.transcription).process(media_path=str(path), output_json_path=None)
    return run


def _decision_engine(scorer: Any, cfg: ReviewConfig) -> Any:
    """The scorer when we have one, else a thresholds-only view (mock=True never loads weights; thresholds come from the same file)."""
    if scorer is not None:
        return scorer
    from models.final.final_scorer import FinalScorer
    return FinalScorer(model_dir=cfg.scorer_model_dir, mock=True, unsure_band_enabled=cfg.unsure_band_enabled, max_questions=cfg.max_questions)


def analyze_media(
    path: Union[str, Path],
    name: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    used: Used = None,
    cfg: Optional[ReviewConfig] = None,
    scorer: Any = None,
    transcribe_fn: Optional[Callable[[Path], Any]] = None,
    job: Optional[str] = None,
    progress: Optional[Callable[[str, float], None]] = None,
    scorer_provider: Optional[Callable[[], Any]] = None,
) -> dict:
    """The whole file is transcribed once (never a trimmed part), so sentence ids are file-global and independent of `used`. Transcript, units and
    pair scores are cached under cfg.cache_dir/<media fingerprint>; a second run on the same file reuses them and gives identical ids."""
    cfg = cfg or ReviewConfig()
    tick = progress or (lambda status, frac: None)
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError("Could not read the file")
    tick("Reading the file", 0.02)
    fp = media_fingerprint(p)
    cache = MediaCache(cfg.cache_dir, fp)

    transcript = cache.get("transcript")
    if transcript is None:
        tick("Transcribing", 0.05)
        t = (transcribe_fn or default_transcribe(cfg))(p)
        transcript = t if isinstance(t, dict) else t.model_dump()
        cache.put("transcript", transcript)
    else:
        logger.info("[Review] transcript from cache (%s)", fp[:12])

    units = cache.get("units")
    unit_objs = None
    if units is None:
        unit_objs = _load_units(transcript)
        tick("Splitting sentences", 0.3)
        units = [{"unit_id": u.unit_id, "source_segment_id": u.source_segment_id, "start": u.start, "end": u.end, "speaker": u.speaker,
                  "raw_text": u.raw_text, "confidence": u.confidence, "is_below_min_token_length": u.is_below_min_token_length} for u in unit_objs]
        cache.put("units", units)

    engine = _decision_engine(scorer, cfg)
    model_key = (engine.model_version or engine.scorer_name or "model").replace("/", "_")
    scores = cache.get(f"scores__{model_key}")
    if scores is None:
        tick("Scoring sentence pairs", 0.4)
        scores = _score_pairs(unit_objs if unit_objs is not None else _load_units(transcript), scorer or (scorer_provider() if scorer_provider else None), cfg)
        cache.put(f"scores__{model_key}", scores)
    else:
        logger.info("[Review] pair scores from cache (%d pairs)", len(scores))

    if duration_seconds is None:
        duration_seconds = _probe_duration(p) or float(transcript.get("duration", 0.0))
    tick("Building the candidates", 0.95)
    doc = build_candidates(units, scores, transcript, engine,
                           {"path": str(path), "duration_seconds": float(duration_seconds), "fingerprint": fp},
                           job or f"j_{uuid.uuid4().hex[:8]}", used, cfg)
    tick("Done", 1.0)
    return doc


def _load_units(transcript: dict) -> list:
    from models.transcript import Transcript
    from pipeline.step_03_token_matching import PAUSE_SPLIT_THRESHOLD_SECONDS, load_transcript_units
    return load_transcript_units(Transcript.model_validate(transcript), pause_threshold=PAUSE_SPLIT_THRESHOLD_SECONDS)


def _score_pairs(unit_objs: list, scorer: Any, cfg: ReviewConfig) -> list[dict]:
    """Score EVERY candidate pair of the whole file (Stage 3 candidates -> Stage 4 scorer). Decisions are made later, per request."""
    from pipeline.step_03_token_matching import default_config as token_config
    from pipeline.step_03_token_matching import generate_candidate_pairs
    from pipeline.step_04_semantic_matching import get_final_scorer, run_stage4

    scorer = scorer or get_final_scorer(model_dir=cfg.scorer_model_dir, device=cfg.scorer_device,
                                        unsure_band_enabled=cfg.unsure_band_enabled, max_questions=cfg.max_questions)
    out = []
    for c in run_stage4(generate_candidate_pairs(unit_objs, config=token_config), scorer=scorer):
        ua, ub = sorted((c["unit_a"], c["unit_b"]), key=lambda u: (float(u["start"]), str(u["unit_id"])))
        out.append({"a": str(ua["unit_id"]), "b": str(ub["unit_id"]), "plain_stage2": c["plain_stage2_score"], "ce_prob": c.get("ce_prob"),
                    "stacker_p": c["stacker_p_raw"]})
    return out


def _probe_duration(p: Path) -> Optional[float]:
    try:
        from pipeline.step_08_video_edit import probe_media
        return probe_media(p)[0]
    except Exception:
        return None
