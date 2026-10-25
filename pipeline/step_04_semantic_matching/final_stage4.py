"""
Stage 4 with the FINAL SCORER (stacker + cross-encoder), replacing the old Stage 2 classification as the decision rule.

For every candidate pair (from step 03):
  1. `run_stage2` (unchanged) supplies the semantic score and the OLD recency-adjusted score / class; those are kept only for audit as
     `legacy_stage2_score` and `legacy_class` (the old key names `stage2_score` / `classification` are removed so nothing reads them by mistake).
  2. plain_stage2_score = 0.50*semantic + 0.30*lexical_score + 0.20*containment, built from the pair's OWN fields. This is the score the
     stacker and thresholds were trained on; it is NOT the pipeline's recency-adjusted value.
  3. The cross-encoder (saved 3-seed average, raw text, earlier unit first, max length 128) gives ce_prob / ce_logit; the stacker gives stacker_p.
  4. Classification: stacker_p >= high -> retake, < low -> not a retake, otherwise ambiguous.
  5. Ambiguous pairs go through the cross-encoder UNSURE BAND: ce_prob >= t_hi (= t_amb) -> retake, <= t_lo -> not_retake (resolved_by
     "cross_encoder"); in between the pair stays `ambiguous` with resolved_by = "unsure" (a person is asked), at most `max_questions` per call. With
     unsure_band_enabled=False the old single t_amb threshold decides and every pair is `retake` or `not_retake`. resolved_by is "stacker",
     "cross_encoder", "unsure" (or "stage2_fallback" when the scorer fell back to plain Stage 2).
The old lexical-vs-semantic disagreement rule is no longer used for classification; its value stays stored in the candidate.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from .semantic_matching import run_stage2

logger = logging.getLogger(__name__)


def _earlier_first(c: dict[str, Any]) -> tuple[str, str]:
    """raw texts of the two units, earlier unit first (the order the cross-encoder was trained on)."""
    a, b = c["unit_a"], c["unit_b"]
    if float(a.get("start", 0.0)) > float(b.get("start", 0.0)):
        a, b = b, a
    return str(a.get("raw_text", "")), str(b.get("raw_text", ""))


def get_final_scorer(model_dir: Optional[str] = None, device: Optional[str] = None, batch_size: int = 64, mock: bool = False,
                     unsure_band_enabled: bool = True, max_questions: int = 25):
    """Build the FinalScorer from models/final (imported lazily; heavy imports only when Stage 4 runs)."""
    from models.final.final_scorer import FinalScorer
    return FinalScorer(model_dir=model_dir, device=device, batch_size=batch_size, mock=mock,
                       unsure_band_enabled=unsure_band_enabled, max_questions=max_questions)


def scorer_envelope(scorer) -> dict[str, Any]:
    """Metadata written into every candidate and into the JSON envelope."""
    return {"scorer_name": scorer.scorer_name, "thresholds_file": scorer.thresholds_file, "model_version": scorer.model_version,
            "feature_version": scorer.feature_version, "label_version": scorer.label_version,
            "thresholds": {"high": scorer.high, "low": scorer.low, "t_amb": scorer.t_amb, "t_lo": scorer.t_lo, "t_hi": scorer.t_hi,
                           "unsure_band_enabled": scorer.unsure_band_enabled, "version": scorer.thresholds_version}}


def run_stage4(
    candidates: Sequence[Any],
    scorer: Any = None,
    model_dir: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Score, classify and resolve every candidate pair. Returns dicts that keep all step-03 fields."""
    if not candidates:
        return []
    scorer = scorer or get_final_scorer(model_dir=model_dir, device=device, batch_size=batch_size)
    enriched = run_stage2(candidates)                                       # old pipeline fields, unchanged code
    pairs = []
    for c in enriched:
        ta, tb = _earlier_first(c)
        pairs.append({"lexical_score": c["lexical_score"], "semantic_score": c["semantic_score"],
                      "containment": c["scores"]["containment"], "text_a": ta, "text_b": tb})
    scored = scorer.score_detailed(pairs)                                   # plain formula from the pair's own fields + cross-encoder + stacker
    decisions, resolved_by = scorer.resolve(scored["p"], scored["ce_prob"])
    meta = {k: v for k, v in scorer_envelope(scorer).items() if k != "thresholds"}
    out = []
    for i, c in enumerate(enriched):
        d = dict(c)
        d["legacy_stage2_score"] = d.pop("stage2_score")                    # recency-adjusted value of the old pipeline: audit only
        d["legacy_class"] = d.pop("classification")
        d["plain_stage2_score"] = round(float(scored["plain_stage2"][i]), 6)
        d["ce_logit"] = None if scored["ce_logit"][i] != scored["ce_logit"][i] else round(float(scored["ce_logit"][i]), 6)
        d["stacker_p"] = round(float(scored["p"][i]), 6)                     # in a fallback mode this is the plain Stage 2 score
        d["stacker_p_raw"] = float(scored["p"][i])                           # unrounded (the review documents compare it with the thresholds)
        d["ce_prob"] = None if scored["ce_prob"][i] != scored["ce_prob"][i] else float(scored["ce_prob"][i])
        d["class"] = decisions[i]                                           # "retake" | "not_retake"; "ambiguous" only for unsure-band pairs that are asked
        d["resolved_by"] = resolved_by[i]
        d.update(meta)
        out.append(d)
    n_ce = sum(1 for r in resolved_by if r == "cross_encoder")
    logger.info("[Stage 4] scorer=%s: %d pairs, %d retake, %d resolved by the cross-encoder", scorer.scorer_name, len(out), decisions.count("retake"), n_ce)
    return out
