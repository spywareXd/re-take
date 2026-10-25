"""Fit t_amb ONCE from the saved out-of-fold scores and store it (with the number of pairs used) in thresholds.json.

Ambiguous pairs (low <= stacker p < high) are resolved by the cross-encoder: retake if prob_mean >= t_amb. t_amb is the threshold that
maximises F1 (retake class) among the out-of-fold pairs whose out-of-fold stacker probability lies in that band; ties go to the lowest
threshold. Also records the scorer name and version fields that the pipeline writes into every candidate envelope. No report is produced.
Run:  python models/final/fit_t_amb.py   (idempotent: refuses to overwrite an existing t_amb unless --refit is given)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parents[1] / "data" / "training"
art_path = HERE / "thresholds.json"
art = json.loads(art_path.read_text(encoding="utf-8"))
if "t_amb" in art["thresholds"] and "--refit" not in sys.argv:
    raise SystemExit(f"t_amb already fitted ({art['thresholds']['t_amb']:.6f}); pass --refit to redo")

df = pd.read_csv(TRAIN / "results_v2" / "final" / "final_oof_scores.csv", dtype={"pair_id": str})
lo, hi = art["thresholds"]["low"], art["thresholds"]["high"]
amb = df[(df.stacker_prob_oof >= lo) & (df.stacker_prob_oof < hi)]
y, ce = amb.label.values, amb.ce_prob_mean.values
best_t, best_f1 = None, -1.0
for t in np.unique(ce):
    pred = ce >= t
    tp = int((pred & (y == 1)).sum())
    f1 = 2 * tp / max(1, int(pred.sum()) + int((y == 1).sum()))            # F1 = 2TP / (predicted positives + actual positives)
    if f1 > best_f1 + 1e-12:                                                # strict: ties keep the lowest threshold
        best_t, best_f1 = float(t), f1
art["thresholds"]["t_amb"] = best_t
art["thresholds"]["t_amb_rule"] = ("ambiguous pairs (low <= stacker p < high) are resolved by the cross-encoder: retake if prob_mean >= t_amb; "
                                   "t_amb maximises F1 among the out-of-fold pairs whose out-of-fold stacker probability lies in that band")
art["thresholds"]["t_amb_fit_pairs"] = int(len(amb))
art["thresholds"]["t_amb_fit_retakes"] = int(y.sum())
art["scorer_name"] = "stacker_cross_encoder"
art["model_version"] = "final-ce-3seed-v2labels+stacker-v1"
art["label_version"] = f"pairs_v2 (sha256 {art['label_file']['sha256_12']})"
art_path.write_text(json.dumps(art, indent=2), encoding="utf-8")
print(f"t_amb = {best_t:.6f} fitted on {len(amb)} pairs ({int(y.sum())} retakes)")
