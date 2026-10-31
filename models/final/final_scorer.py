"""Final scorer: stacker over [Stage 2 score, logit(cross-encoder prob)] with two thresholds -> retake / ambiguous / unlikely.

    from models.final.final_scorer import score_pairs, classify
    p = score_pairs(pairs)            # pairs: list of dicts with text_a/text_b (and optional precomputed scores)
    classify(p)                       # -> ['retake' | 'ambiguous' | 'unlikely', ...]

A pair dict may carry precomputed values (used by the tests and by callers that already ran Stage 2):
    lexical_score, semantic_score, containment -> Stage 2 = 0.5*semantic + 0.3*lexical + 0.2*containment (always preferred; a pipeline
                            candidate's own `stage2_score` is recency-adjusted, differs from the training score, and is IGNORED with a warning)
    stage2_score            -> used as is ONLY when no parts are given; it must then be the plain value
    ce_prob                 -> cross-encoder probability (mean over seeds); skips the cross-encoder
Otherwise Stage 2 and the cross-encoder are computed from text_a / text_b (earlier unit first).

Fallback: if thresholds.json or any cross-encoder weights are missing, the scorer logs a WARNING and uses the Stage 2 formula
alone with the OLD rules (retake >= 0.65, unlikely < 0.35). `mock=True` replaces the cross-encoder and the embedding model with
cheap deterministic stand-ins so pytest runs without a GPU (mock scores are NOT comparable with the real ones).

This module does NOT touch the keeper / LLM step: spans come out with a status and `keeper_unit_id = None`.
"""
from __future__ import annotations

import json
import logging
import math
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
logger = logging.getLogger("final_scorer")
RETAKE, AMBIGUOUS, UNLIKELY = "retake", "ambiguous", "unlikely"
OLD_HIGH, OLD_LOW = 0.65, 0.35
DEFAULT_TUNED_PATH = REPO / "data" / "training" / "results_v2" / "final" / "fair_comparison.json"   # tuned Stage 2 thresholds saved by the evaluation
LEX_W = {"sequence": 0.30, "jaccard": 0.25, "containment": 0.25, "bigram": 0.20}      # pipeline lexical weights (step 03)
S2_W = {"semantic": 0.50, "lexical": 0.30, "containment": 0.20}                       # pipeline Stage 2 weights (step 04)
_TOK = re.compile(r"[a-z0-9']+")


def stage2_from_scores(semantic: float, lexical: float, containment: float) -> float:
    return S2_W["semantic"] * semantic + S2_W["lexical"] * lexical + S2_W["containment"] * containment


def _simple_scores(a: str, b: str) -> dict[str, float]:
    """Cheap token scores used only by the mock path."""
    ta, tb = _TOK.findall(a.lower()), _TOK.findall(b.lower())
    sa, sb = set(ta), set(tb)
    jac = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    cont = len(sa & sb) / min(len(sa), len(sb)) if sa and sb else 0.0
    bg = lambda t: {tuple(t[i:i + 2]) for i in range(len(t) - 1)}
    ba, bb = bg(ta), bg(tb)
    big = len(ba & bb) / len(ba | bb) if ba | bb else 0.0
    seq = SequenceMatcher(a=ta, b=tb).ratio()
    return {"sequence": seq, "jaccard": jac, "containment": cont, "bigram": big}


class FinalScorer:
    """mode / scorer_name:  "stacker" (stacker_cross_encoder)  >  "tuned_stage2_fallback"  >  "legacy_stage2_fallback".
    The fallbacks exist only for robustness (missing/failed weights or thresholds): plain Stage 2 with thresholds read from the saved
    tuned file (results_v2/final/fair_comparison.json), then the legacy 0.65 / 0.35 rules, each with a WARNING."""

    def __init__(self, model_dir: Optional[str | Path] = None, mock: bool = False, device: Optional[str] = None, batch_size: int = 64,
                 thresholds_path: Optional[str | Path] = None, tuned_thresholds_path: Optional[str | Path] = None,
                 unsure_band_enabled: bool = True, max_questions: int = 25):
        self.model_dir = Path(model_dir) if model_dir else HERE
        self.mock, self.device, self.batch_size = mock, device, int(batch_size)
        self.thresholds_path = Path(thresholds_path) if thresholds_path else self.model_dir / "thresholds.json"
        self.tuned_path = Path(tuned_thresholds_path) if tuned_thresholds_path else DEFAULT_TUNED_PATH
        self.mode, self.scorer_name = "stacker", "stacker_cross_encoder"
        self.high, self.low, self.t_amb = OLD_HIGH, OLD_LOW, None
        self._ce = None
        self.art: dict[str, Any] = {}
        missing = []
        if not self.thresholds_path.exists():
            missing.append(str(self.thresholds_path))
        else:
            try:
                self.art = json.loads(self.thresholds_path.read_text(encoding="utf-8"))
                if not mock:
                    for d in self.art["models"]["cross_encoder_weights"]:
                        if not (self.model_dir / d / "config.json").exists():
                            missing.append(str(self.model_dir / d))
            except Exception as e:                                               # unreadable / malformed thresholds file
                missing.append(f"{self.thresholds_path} ({e})")
        if not missing and not mock:
            try:
                self._load_ce()                                                  # weights that exist but fail to load also trigger the fallback
            except Exception as e:
                missing.append(f"cross-encoder weights failed to load ({e})")
        if missing:
            self._enter_fallback(missing)
        else:
            self.high, self.low = float(self.art["thresholds"]["high"]), float(self.art["thresholds"]["low"])
            self.t_amb = self.art["thresholds"].get("t_amb")
            if self.t_amb is None:
                self._enter_fallback([f"t_amb missing in {self.thresholds_path} (run models/final/fit_t_amb.py)"])
            else:
                self.t_amb = float(self.t_amb)
        # unsure band: cross-encoder prob >= t_hi -> retake, <= t_lo -> different, in between -> ambiguous (ask the user). Only meaningful in stacker mode
        # and only when thresholds.json carries "unsure_band"; with unsure_band_enabled=False the old single threshold t_amb decides (old behaviour).
        self.max_questions = int(max_questions)
        band = self.art.get("unsure_band") if self.mode == "stacker" else None
        self.t_lo = float(band["t_lo"]) if band else None
        self.t_hi = float(band["t_hi"]) if band else self.t_amb
        self.unsure_band_enabled = bool(unsure_band_enabled) and band is not None
        if unsure_band_enabled and self.mode == "stacker" and band is None:
            logger.warning("final scorer: thresholds.json has no unsure_band (run models/final/fit_unsure_band.py); using the single t_amb threshold")
        self.thresholds_version = str(self.art.get("thresholds_version", "1")) if self.mode == "stacker" else ""
        self.model_version = self.art.get("model_version", "") if self.mode == "stacker" else ""
        self.feature_version = self.art.get("feature_version", "v1")
        self.label_version = self.art.get("label_version", "") if self.mode == "stacker" else ""
        self.thresholds_file = str(self.thresholds_path if self.mode == "stacker" else (self.tuned_path if self.mode == "tuned_stage2_fallback" else "legacy 0.65/0.35 rules"))

    def _enter_fallback(self, missing: list[str]) -> None:
        try:
            lo, hi = json.loads(self.tuned_path.read_text(encoding="utf-8"))["final_thresholds"]["stage2"]
            self.mode, self.scorer_name, self.low, self.high = "tuned_stage2_fallback", "tuned_stage2_fallback", float(lo), float(hi)
            logger.warning("final scorer: problem with %s; FALLING BACK to plain tuned Stage 2 (low %.4f, high %.4f, read from %s)", "; ".join(missing), self.low, self.high, self.tuned_path)
        except Exception:
            self.mode, self.scorer_name, self.low, self.high = "legacy_stage2_fallback", "legacy_stage2_fallback", OLD_LOW, OLD_HIGH
            logger.warning("final scorer: problem with %s and the tuned thresholds file %s is missing/unreadable; FALLING BACK to the legacy Stage 2 rules %.2f / %.2f",
                           "; ".join(missing), self.tuned_path, OLD_HIGH, OLD_LOW)
        self.t_amb = None

    # ---------------- Stage 2 ----------------
    def stage2_score(self, pair: dict[str, Any]) -> float:
        """The PLAIN Stage 2 score the stacker and thresholds were trained on: 0.5*semantic + 0.3*lexical_score + 0.2*containment.
        Priority: (1) lexical_score/semantic_score/containment -> the plain formula, ALWAYS (a supplied `stage2_score` is ignored, with a warning
        if it differs: the pipeline's run_stage2 value is recency-adjusted and runs about 0.04 lower, which would shift every score);
        (2) `stage2_score`, which the caller promises is the plain value; (3) compute from text_a / text_b."""
        has_parts = all(pair.get(k) is not None for k in ("lexical_score", "semantic_score", "containment"))
        if has_parts:
            plain = stage2_from_scores(float(pair["semantic_score"]), float(pair["lexical_score"]), float(pair["containment"]))
            sv = pair.get("stage2_score")
            if sv is not None and abs(float(sv) - plain) > 0.005:
                logger.warning("pair %s: supplied stage2_score %.4f differs from the plain formula %.4f (the pipeline's value is recency-adjusted); "
                               "using the plain formula the stacker was trained on", pair.get("pair_id"), float(sv), plain)
            return plain
        if pair.get("stage2_score") is not None:
            return float(pair["stage2_score"])
        a, b = str(pair["text_a"]), str(pair["text_b"])
        if self.mock:
            s = _simple_scores(a, b)
            lex = sum(LEX_W[k] * s[k] for k in LEX_W)
            return stage2_from_scores(s["jaccard"], lex, s["containment"])         # mock semantic = jaccard
        return self._stage2_real(a, b)

    def _stage2_real(self, a: str, b: str) -> float:
        """The pipeline's own lexical functions + all-MiniLM-L6-v2 cosine mapped to [0,1] (same as the training data)."""
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from pipeline.step_03_token_matching.config import TokenMatchingConfig
        from pipeline.step_03_token_matching.edit_distance import containment_score, sequence_similarity_score
        from pipeline.step_03_token_matching.exact_match import exact_match_score, near_exact_match_score
        from pipeline.step_03_token_matching.jaccard import jaccard_score
        from pipeline.step_03_token_matching.ngram_match import ngram_overlap_score
        from pipeline.step_02_preprocessing.normalize import tokenize
        from pipeline.step_02_preprocessing.schemas import ComparableUnit
        from pipeline.step_04_semantic_matching.semantic_matching import get_model, semantic_similarity_score

        def cu(t):
            tk = tokenize(t)
            return ComparableUnit(unit_id="x", source_segment_id=0, start=0.0, end=0.0, raw_text=t, normalized_text=" ".join(tk), tokens=tk, token_set=set(tk))

        ua, ub = cu(a), cu(b)
        lex = max(LEX_W["sequence"] * sequence_similarity_score(ua, ub) + LEX_W["jaccard"] * jaccard_score(ua, ub)
                  + LEX_W["containment"] * containment_score(ua, ub) + LEX_W["bigram"] * ngram_overlap_score(ua, ub, 2),
                  exact_match_score(ua, ub), near_exact_match_score(ua, ub, set(TokenMatchingConfig().filler_words)))
        ea, eb = get_model("all-MiniLM-L6-v2").encode([a, b], show_progress_bar=False)
        return stage2_from_scores(semantic_similarity_score(ea, eb), lex, containment_score(ua, ub))

    # ---------------- cross-encoder ----------------
    def _load_ce(self) -> None:
        """Load the saved seed models (GPU if present unless `device` was given)."""
        if self._ce is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._ce = [(AutoTokenizer.from_pretrained(self.model_dir / d), AutoModelForSequenceClassification.from_pretrained(self.model_dir / d).to(dev).eval(), dev)
                    for d in self.art["models"]["cross_encoder_weights"]]

    def ce_probs(self, pairs: Sequence[dict[str, Any]]) -> np.ndarray:
        """mean over the 3 seeds of sigmoid(logit); precomputed `ce_prob` in a pair dict is used as is."""
        out = np.full(len(pairs), np.nan)
        todo = [i for i, p in enumerate(pairs) if p.get("ce_prob") is None]
        for i, p in enumerate(pairs):
            if p.get("ce_prob") is not None:
                out[i] = float(p["ce_prob"])
        if not todo:
            return out
        if self.mock:
            for i in todo:
                s = _simple_scores(str(pairs[i]["text_a"]), str(pairs[i]["text_b"]))
                lex = sum(LEX_W[k] * s[k] for k in LEX_W)
                out[i] = 1 / (1 + math.exp(-10 * (lex - 0.5)))                  # deterministic stand-in
            return out
        self._load_ce()
        import torch
        firsts, seconds = [str(pairs[i]["text_a"]) for i in todo], [str(pairs[i]["text_b"]) for i in todo]
        seed_probs = []
        B = self.batch_size
        for tok, model, dev in self._ce:
            ps = []
            with torch.no_grad():
                for j in range(0, len(todo), B):
                    enc = tok(firsts[j:j + B], seconds[j:j + B], truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)
                    ps.append(torch.sigmoid(model(**enc).logits.squeeze(-1)).float().cpu().numpy())
            seed_probs.append(np.concatenate(ps))
        out[todo] = np.mean(seed_probs, axis=0)
        return out

    # ---------------- stacker, scoring, classification ----------------
    def stacker_prob(self, stage2: np.ndarray, ce_prob: np.ndarray) -> np.ndarray:
        s = self.art["stacker"]
        eps = s["logit_clip_eps"]
        pc = np.clip(ce_prob, eps, 1 - eps)
        x = np.stack([stage2, np.log(pc / (1 - pc))], axis=1)
        z = (x - np.array(s["scaler_mean"])) / np.array(s["scaler_scale"])
        return 1 / (1 + np.exp(-(z @ np.array(s["coef_standardized"]) + s["intercept"])))

    def score_detailed(self, pairs: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
        """plain_stage2, ce_prob (seed mean), ce_logit, p (stacker probability; in a fallback mode p IS the plain Stage 2 score and ce_* are NaN)."""
        stage2 = np.array([self.stage2_score(p) for p in pairs], dtype=float)
        if self.mode != "stacker":
            nan = np.full(len(pairs), np.nan)
            return {"plain_stage2": stage2, "ce_prob": nan, "ce_logit": nan.copy(), "p": stage2}
        ce = self.ce_probs(pairs)
        eps = self.art["stacker"]["logit_clip_eps"]
        pc = np.clip(ce, eps, 1 - eps)
        return {"plain_stage2": stage2, "ce_prob": ce, "ce_logit": np.log(pc / (1 - pc)), "p": self.stacker_prob(stage2, ce)}

    def score_pairs(self, pairs: Sequence[dict[str, Any]]) -> np.ndarray:
        return self.score_detailed(pairs)["p"]

    def band_decision(self, ce_prob: float) -> str:
        """The unsure band applied to a cross-encoder probability: 'retake' (>= t_hi), 'different' (<= t_lo) or 'ambiguous' (in between)."""
        if ce_prob >= self.t_hi:
            return "retake"
        if ce_prob <= self.t_lo:
            return "different"
        return "ambiguous"

    def resolve(self, p, ce_prob) -> tuple[list[str], list[str]]:
        """Final decision for every pair: (decision, resolved_by).
        stacker mode: p >= high -> 'retake' and p < low -> 'not_retake', both resolved_by 'stacker'; in between the cross-encoder decides.
          unsure_band_enabled (default): ce >= t_hi -> 'retake', ce <= t_lo -> 'not_retake' (both resolved_by 'cross_encoder'); otherwise 'ambiguous'
            with resolved_by 'unsure' (to be asked). At most `max_questions` pairs per call stay ambiguous: those closest to the middle of the band;
            the rest are decided by ce >= t_hi, i.e. 'not_retake' / 'cross_encoder', and are never emitted as ambiguous. Call once per video.
          unsure_band_enabled=False (old behaviour): retake if ce >= t_amb else 'not_retake', resolved_by 'cross_encoder'.
        fallback modes (plain Stage 2): >= high retake, < low not_retake; an in-between pair is NOT cut (not_retake), resolved_by 'stage2_fallback'."""
        decisions, by = [], []
        ce_arr = np.asarray(ce_prob, dtype=float)
        for pv, cp in zip(np.asarray(p, dtype=float), ce_arr):
            tier = self.classify(float(pv))
            if tier == RETAKE:
                decisions.append("retake"); by.append("stacker" if self.mode == "stacker" else "stage2_fallback")
            elif tier == UNLIKELY:
                decisions.append("not_retake"); by.append("stacker" if self.mode == "stacker" else "stage2_fallback")
            elif self.mode != "stacker":
                decisions.append("not_retake"); by.append("stage2_fallback")
            elif not self.unsure_band_enabled:
                decisions.append("retake" if cp >= self.t_amb else "not_retake"); by.append("cross_encoder")
            else:
                d = self.band_decision(float(cp))
                if d == "retake":
                    decisions.append("retake"); by.append("cross_encoder")
                elif d == "different":
                    decisions.append("not_retake"); by.append("cross_encoder")
                else:
                    decisions.append("ambiguous"); by.append("unsure")
        if self.unsure_band_enabled and self.mode == "stacker":
            asked = [i for i, r in enumerate(by) if r == "unsure"]
            if len(asked) > self.max_questions:
                mid = (self.t_lo + self.t_hi) / 2.0
                asked.sort(key=lambda i: (abs(ce_arr[i] - mid), i))
                for i in asked[self.max_questions:]:               # over the cap: decided by ce >= t_hi, which is False inside the band
                    decisions[i], by[i] = "not_retake", "cross_encoder"
        return decisions, by

    def classify(self, p):
        """retake: p >= high;  unlikely: p < low;  ambiguous otherwise. Accepts a scalar or an array."""
        if np.isscalar(p):
            return RETAKE if p >= self.high else (UNLIKELY if p < self.low else AMBIGUOUS)
        return [RETAKE if v >= self.high else (UNLIKELY if v < self.low else AMBIGUOUS) for v in np.asarray(p)]

    def classify_pairs(self, pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        p = self.score_pairs(pairs)
        return [{"pair_id": pr.get("pair_id"), "video_id": pr.get("video_id", ""), "unit_a_id": pr.get("unit_a_id"), "unit_b_id": pr.get("unit_b_id"),
                 "start_a": pr.get("start_a"), "end_a": pr.get("end_a"), "start_b": pr.get("start_b"), "end_b": pr.get("end_b"),
                 "p": float(v), "status": s, "mode": self.mode} for pr, v, s in zip(pairs, p, self.classify(p))]


def to_spans(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cluster retake/ambiguous pairs into spans (union-find over unit ids, like step 05). Span status = 'ambiguous' if any pair in it is
    ambiguous (it needs a human/LLM decision later), else 'retake'. The keeper is deliberately left unset."""
    parent: dict[tuple, tuple] = {}
    def find(x):                                        # union-find with path halving
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    keep = [r for r in results if r["status"] in (RETAKE, AMBIGUOUS)]
    times: dict[tuple, tuple] = {}
    for r in keep:
        a, b = (r["video_id"], r["unit_a_id"]), (r["video_id"], r["unit_b_id"])
        times[a], times[b] = (r.get("start_a"), r.get("end_a")), (r.get("start_b"), r.get("end_b"))
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    comps: dict[tuple, list[dict[str, Any]]] = {}
    for r in keep:
        comps.setdefault(find((r["video_id"], r["unit_a_id"])), []).append(r)
    spans = []
    for rs in comps.values():
        units = sorted({(r["video_id"], u) for r in rs for u in (r["unit_a_id"], r["unit_b_id"])}, key=lambda k: (times[k][0] if times[k][0] is not None else 0.0, str(k[1])))
        starts = [times[k][0] for k in units if times[k][0] is not None]
        ends = [times[k][1] for k in units if times[k][1] is not None]
        spans.append({"video_id": rs[0]["video_id"], "unit_ids": [k[1] for k in units], "start": min(starts) if starts else None, "end": max(ends) if ends else None,
                      "pair_ids": [r["pair_id"] for r in rs], "status": AMBIGUOUS if any(r["status"] == AMBIGUOUS for r in rs) else RETAKE,
                      "keeper_unit_id": None, "units_to_remove": None})
    spans.sort(key=lambda s: (s["video_id"], s["start"] if s["start"] is not None else 0.0))
    for i, s in enumerate(spans):
        s["span_id"] = f"span_{i}"
    return spans


# module-level convenience wrappers around a lazily created default scorer
_default: Optional[FinalScorer] = None


def _get(scorer: Optional[FinalScorer]) -> FinalScorer:
    global _default
    if scorer is not None:
        return scorer
    if _default is None:
        _default = FinalScorer()
    return _default


def score_pairs(pairs: Sequence[dict[str, Any]], scorer: Optional[FinalScorer] = None) -> np.ndarray:
    return _get(scorer).score_pairs(pairs)


def classify(p, scorer: Optional[FinalScorer] = None):
    return _get(scorer).classify(p)
