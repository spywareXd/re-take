"""
Score keeper-selection decision-model outputs the same way for every model.

Input: for each row the model's per-unit probabilities {"keep_<unit id>": p} (or {"<unit id>": p}), as returned by FastDecisionModel.predict
(the probability of True of a noul question).

    decide(probs)           -> {"keep": [...], "keep_none": bool, "unclear": bool}
        keep      = units with p >= 0.5
        keep_none = the keep set is empty (no forced fallback)
        unclear   = the top two units are within 0.15 of each other, or the max p is below 0.6
    exact_match(keep, gold) -> True iff the predicted keeper set equals the labelled set (gold: {"keep_<id>": bool}); two empty sets match (keep_none rows)
    score(gold_rows, prob_rows) -> {"n", "exact_match" (overall), "exact_match_keep_none_rows", "exact_match_rows_with_keepers", "n_keep_none", ...}
"""
from __future__ import annotations

import json
from typing import Mapping, Sequence

THRESHOLD = 0.5
UNCLEAR_MARGIN = 0.15
UNCLEAR_MAX_BELOW = 0.6


def _unit_id(key: str) -> str:
    return key[len("keep_"):] if key.startswith("keep_") else key


def decide(probs: Mapping[str, float]) -> dict:
    p = {_unit_id(k): float(v) for k, v in probs.items()}
    keep = [u for u, v in p.items() if v >= THRESHOLD]                       # unit order of the input
    ranked = sorted(p.values(), reverse=True)
    unclear = bool(ranked) and (ranked[0] < UNCLEAR_MAX_BELOW or (len(ranked) > 1 and ranked[0] - ranked[1] <= UNCLEAR_MARGIN))
    return {"keep": keep, "keep_none": not keep, "unclear": unclear if ranked else True}


def gold_keep(gold: Mapping[str, bool] | str) -> list[str]:
    g = json.loads(gold) if isinstance(gold, str) else gold
    return [_unit_id(k) for k, v in g.items() if v]


def exact_match(keep: Sequence[str], gold: Mapping[str, bool] | str) -> bool:
    return set(keep) == set(gold_keep(gold))


def _rate(hits: int, n: int) -> float | None:
    return None if n == 0 else hits / n


def score(gold_rows: Sequence[Mapping[str, bool] | str], prob_rows: Sequence[Mapping[str, float]]) -> dict:
    """gold_rows[i] is the row's gold (dict or JSON string); prob_rows[i] its per-unit probabilities. keep-last baseline = labelled set == [last unit]."""
    if len(gold_rows) != len(prob_rows):
        raise ValueError("gold_rows and prob_rows differ in length")
    hits = {"all": 0, "none": 0, "keepers": 0}
    counts = {"none": 0, "keepers": 0}
    unclear = baseline = 0
    for g, probs in zip(gold_rows, prob_rows):
        d = decide(probs)
        gk = gold_keep(g)
        kind = "none" if not gk else "keepers"
        ok = exact_match(d["keep"], g)
        hits["all"] += ok
        hits[kind] += ok
        counts[kind] += 1
        unclear += d["unclear"]
        units = [_unit_id(k) for k in (json.loads(g) if isinstance(g, str) else g)]
        baseline += gk == [units[-1]]
    n = len(gold_rows)
    return {"n": n, "exact_match": _rate(hits["all"], n), "exact_match_keep_none_rows": _rate(hits["none"], counts["none"]),
            "exact_match_rows_with_keepers": _rate(hits["keepers"], counts["keepers"]), "n_keep_none": counts["none"],
            "unclear_rate": _rate(unclear, n), "keep_last_baseline": _rate(baseline, n)}
