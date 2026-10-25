"""Fit the cross-encoder "unsure band" ONCE from the saved out-of-fold scores and store it in thresholds.json under "unsure_band".

Ambiguous pairs (low <= out-of-fold stacker p < high) are decided by the cross-encoder. Until now one threshold (t_amb) did it; the unsure band
adds a lower threshold: ce >= t_hi (= t_amb) -> retake, ce <= t_lo -> different, in between -> ask the user.
t_lo = the highest value (floored to 8 decimals) such that at most 3% of the true retakes in the ambiguous band have ce <= t_lo.
Only the "unsure_band" and "thresholds_version" keys are written. Idempotent: refuses to overwrite unless --refit is given.
The estimate rests on about 36 retakes in 376 pairs, so t_lo and fraction_asked are rough."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parents[1] / "data" / "training"
art_path = HERE / "thresholds.json"
art = json.loads(art_path.read_text(encoding="utf-8"))
if "unsure_band" in art and "--refit" not in sys.argv:
    raise SystemExit(f"unsure_band already fitted ({art['unsure_band']['t_lo']}); pass --refit to redo")

MAX_MISSED = 0.03
lo, hi = art["thresholds"]["low"], art["thresholds"]["high"]
t_hi = float(art["thresholds"]["t_amb"])
df = pd.read_csv(TRAIN / "results_v2" / "final" / "final_oof_scores.csv", dtype={"pair_id": str})
amb = df[(df.stacker_prob_oof >= lo) & (df.stacker_prob_oof < hi)]
y, ce = amb.label.values, amb.ce_prob_mean.values
retake_ce = np.sort(ce[y == 1])
allowed = int(math.floor(MAX_MISSED * len(retake_ce) + 1e-9))             # retakes we may drop below t_lo (3% of 36 -> 1)
# highest t with at most `allowed` retakes at ce <= t: just under the (allowed+1)-th lowest retake score, floored to 8 decimals
t_lo = math.floor(float(retake_ce[allowed]) * 1e8) / 1e8
if (retake_ce <= t_lo).sum() > allowed:
    t_lo = math.floor((float(retake_ce[allowed]) - 1e-8) * 1e8) / 1e8
asked = (ce > t_lo) & (ce < t_hi)
art["unsure_band"] = {
    "t_lo": t_lo,
    "t_hi": t_hi,
    "method": (f"t_lo = highest value (floored to 8 decimals) such that at most {MAX_MISSED:.0%} of the true retakes among the ambiguous-band pairs have "
               "cross-encoder prob <= t_lo; t_hi = t_amb. Pairs with t_lo < ce < t_hi are escalated to the user. Fitted on out-of-fold scores."),
    "n_pairs": int(len(amb)),
    "n_retakes": int(y.sum()),
    "n_retakes_below_t_lo": int((retake_ce <= t_lo).sum()),
    "fraction_asked": round(float(asked.mean()), 4),
}
art["thresholds_version"] = "2"                                              # 1 = before the unsure band existed (no key was stored)
art_path.write_text(json.dumps(art, indent=2), encoding="utf-8")
b = art["unsure_band"]
print(f"t_lo = {b['t_lo']} (t_hi = {b['t_hi']:.8f}); {b['n_retakes_below_t_lo']} of {b['n_retakes']} retakes at or below t_lo; "
      f"fraction_asked = {b['fraction_asked']:.1%} of {b['n_pairs']} band pairs ({int(asked.sum())} pairs; {int(asked.sum())}/{len(df)} = {asked.sum() / len(df):.1%} of all fit pairs)")
