"""Fit the final stacker + two thresholds and estimate honest tier performance. No cross-encoder training, no API calls.

Stacker = StandardScaler + L2 logistic regression (class_weight balanced) on [stage2_score, logit(cross_encoder prob_mean)],
using the out-of-fold cross-encoder scores in results_v2/ (labels: pairs_v2.csv, label 0/1 only; folds: folds.json).
Thresholds on the stacker probability p:  high = lowest t with precision(p >= t) >= 0.95;  low = highest t with recall(p >= t) >= 0.98.
Tiers: retake (p >= high), ambiguous (low <= p < high), unlikely (p < low).
Writes models/final/thresholds.json, results_v2/final/final_oof_scores.csv and results_v2/final/fit_results.json.
Caveat (same as combine.md): the cross-encoder OOF scores of the training folds came from models that saw the held-out fold's labels.
"""
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRAIN = REPO / "data" / "training"
OUT = TRAIN / "results_v2" / "final"
OUT.mkdir(parents=True, exist_ok=True)
C_GRID = [0.01, 0.1, 1, 10, 100]
PREC_TARGET, RECALL_TARGET = 0.95, 0.98
S2_HIGH, S2_LOW = 0.65, 0.35
N_BOOT, SEED, INNER = 1000, 0, 5
EPS = 1e-6                                           # clip for logit(prob)

# ---------------- data ----------------
labels_path, folds_path = TRAIN / "pairs_v2.csv", TRAIN / "folds.json"
base = pd.read_csv(TRAIN / "results_v2" / "baseline_scores.csv", dtype={"pair_id": str, "video_id": str})
ce = pd.read_csv(TRAIN / "results_v2" / "ce_scores.csv", dtype={"pair_id": str}).drop_duplicates("pair_id")
df = base[["pair_id", "video_id", "fold", "label", "current_stage2"]].merge(ce[["pair_id", "prob_mean"]], on="pair_id", validate="1:1")
lab = pd.read_csv(labels_path, dtype=str, keep_default_na=False).set_index("pair_id").label
assert (df.set_index("pair_id").label.astype(str) == lab.reindex(df.pair_id).values).all(), "labels differ from pairs_v2.csv"
assert df.label.isin([0, 1]).all()
df = df.rename(columns={"current_stage2": "stage2_score", "prob_mean": "ce_prob_mean"}).reset_index(drop=True)
p_ = df.ce_prob_mean.clip(EPS, 1 - EPS)
df["ce_logit"] = np.log(p_ / (1 - p_))
X, y, groups, fold = df[["stage2_score", "ce_logit"]].values, df.label.values, df.video_id.values, df.fold.values
FOLDS = sorted(np.unique(fold))


# ---------------- stacker ----------------
def make(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty="l2", class_weight="balanced", max_iter=3000))


def fit_stacker(Xt, yt, gt):
    """C chosen by inner GroupKFold by video (mean PR-AUC, ties -> smaller C), then refit on all given rows."""
    sc = {C: [] for C in C_GRID}
    for tr, va in GroupKFold(n_splits=INNER).split(Xt, yt, gt):
        if yt[va].sum() == 0:
            continue
        for C in C_GRID:
            sc[C].append(average_precision_score(yt[va], make(C).fit(Xt[tr], yt[tr]).predict_proba(Xt[va])[:, 1]))
    C = max(C_GRID, key=lambda c: (np.mean(sc[c]), -c))
    return make(C).fit(Xt, yt), C


def oof_probs(Xa, ya, ga, fa, folds):
    """each row predicted by a stacker fitted on the OTHER folds among `folds`"""
    p = np.zeros(len(Xa))
    for k in folds:
        te = fa == k
        m, _ = fit_stacker(Xa[~te], ya[~te], ga[~te])
        p[te] = m.predict_proba(Xa[te])[:, 1]
    return p


# ---------------- thresholds and tiers ----------------
def choose_thresholds(yt, p):
    """high: lowest t with precision(p>=t) >= 0.95;  low: highest t with recall(p>=t) >= 0.98."""
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], yt[order]
    tp, n = np.cumsum(ys), np.arange(1, len(ps) + 1)
    last = np.r_[ps[1:] != ps[:-1], True]                  # a threshold t = ps[i] includes all rows with p >= t
    prec, rec = tp[last] / n[last], tp[last] / max(1, ys.sum())
    thr = ps[last]
    ok_hi = np.where(prec >= PREC_TARGET)[0]
    high = float(thr[ok_hi.max()]) if len(ok_hi) else float("inf")      # thr is descending: the lowest t is the largest index
    ok_lo = np.where(rec >= RECALL_TARGET)[0]
    low = float(thr[ok_lo.min()]) if len(ok_lo) else float("-inf")      # highest t = smallest index
    return min(low, high), high


def tiers(p, low, high):
    return np.where(p >= high, "retake", np.where(p < low, "unlikely", "ambiguous"))


def tier_stats(yt, tr):
    """precision/recall/F1 of the retake tier (positive class), NPV/recall/F1 of the unlikely tier (negative class), shares, confusion."""
    yt = np.asarray(yt)
    n = len(yt)
    P, N = int((yt == 1).sum()), int((yt == 0).sum())
    out = {"n": n}
    for t in ("retake", "ambiguous", "unlikely"):
        m = tr == t
        out[f"{t}_n"], out[f"{t}_share"] = int(m.sum()), float(m.sum() / n) if n else 0.0
        out[f"{t}_retakes"], out[f"{t}_nonretakes"] = int(((yt == 1) & m).sum()), int(((yt == 0) & m).sum())
    pr = out["retake_retakes"] / max(1, out["retake_n"])
    rc = out["retake_retakes"] / max(1, P)
    npv = out["unlikely_nonretakes"] / max(1, out["unlikely_n"])
    nrc = out["unlikely_nonretakes"] / max(1, N)
    f = lambda a, b: 2 * a * b / (a + b) if a + b else 0.0
    out.update({"retake_precision": pr, "retake_recall": rc, "retake_f1": f(pr, rc),
                "unlikely_precision": npv, "unlikely_recall": nrc, "unlikely_f1": f(npv, nrc),
                "retakes_in_unlikely": out["unlikely_retakes"], "nonretakes_in_retake": out["retake_nonretakes"]})
    return out


# ---------------- 1. final stacker on ALL out-of-fold rows ----------------
final_model, final_C = fit_stacker(X, y, groups)
sc_, lr_ = final_model[0], final_model[-1]
p_final = final_model.predict_proba(X)[:, 1]                           # final stacker applied to the OOF inputs (in-sample for the stacker)

# ---------------- 2. thresholds ----------------
p_oof = oof_probs(X, y, groups, fold, FOLDS)                           # each row from a stacker fitted on the other 3 folds
low_f, high_f = choose_thresholds(y, p_oof)                            # FINAL thresholds, fitted on all OOF rows
# honest nested estimate: for outer fold k, thresholds from an inner OOF pass over the other folds only
nested_tier = np.empty(len(df), dtype=object)
nested_thr, nested_C = {}, {}
for k in FOLDS:
    te = fold == k
    inner_folds = [j for j in FOLDS if j != k]
    p_inner = oof_probs(X[~te], y[~te], groups[~te], fold[~te], inner_folds)
    lo, hi = choose_thresholds(y[~te], p_inner)
    m, C = fit_stacker(X[~te], y[~te], groups[~te])
    nested_thr[int(k)], nested_C[int(k)] = (lo, hi), C
    nested_tier[te] = tiers(m.predict_proba(X[te])[:, 1], lo, hi)
s2_tier = tiers(df.stage2_score.values, S2_LOW, S2_HIGH)
final_tier = tiers(p_oof, low_f, high_f)

# ---------------- metrics: pooled, per fold, bootstrap vs Stage 2 ----------------
res = {"nested": {"pooled": tier_stats(y, nested_tier)}, "final_thr_on_oof": {"pooled": tier_stats(y, final_tier)}, "stage2": {"pooled": tier_stats(y, s2_tier)}}
for k in FOLDS:
    m = fold == k
    for name, t in (("nested", nested_tier), ("final_thr_on_oof", final_tier), ("stage2", s2_tier)):
        res[name][f"fold{k}"] = tier_stats(y[m], t[m])
rng = np.random.default_rng(SEED)
vids = df.video_id.unique()
idx_by_v = {v: np.where(groups == v)[0] for v in vids}
KEYS = ["ambiguous_share", "retake_precision", "retake_recall", "retake_f1", "unlikely_precision", "unlikely_recall", "retake_share", "unlikely_share"]
boot = {s: {k: [] for k in KEYS} for s in ("stacker", "stage2")}
for _ in range(N_BOOT):
    idx = np.concatenate([idx_by_v[v] for v in rng.choice(vids, len(vids), replace=True)])
    for s, t in (("stacker", nested_tier), ("stage2", s2_tier)):
        st = tier_stats(y[idx], t[idx])
        for k in KEYS:
            boot[s][k].append(st[k])
ci = {}
for k in KEYS:
    d = np.array(boot["stacker"][k]) - np.array(boot["stage2"][k])
    lo, hi = np.percentile(d, [2.5, 97.5])
    ci[k] = {"diff": res["nested"]["pooled"][k] - res["stage2"]["pooled"][k], "lo": float(lo), "hi": float(hi)}

# ---------------- 4. artifacts ----------------
sha12 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
art = {
    "created": date.today().isoformat(),
    "stacker": {"features": ["stage2_score", "logit(cross_encoder_prob)"], "logit_clip_eps": EPS,
                "scaler_mean": sc_.mean_.tolist(), "scaler_scale": sc_.scale_.tolist(),
                "coef_standardized": lr_.coef_[0].tolist(), "intercept": float(lr_.intercept_[0]), "C": final_C,
                "penalty": "l2", "class_weight": "balanced", "fit_rows": int(len(df))},
    "thresholds": {"high": high_f, "low": low_f, "rule": "high = lowest t with precision(p>=t)>=0.95; low = highest t with recall(p>=t)>=0.98, both on out-of-fold stacker probabilities",
                   "tiers": "retake: p>=high; ambiguous: low<=p<high; unlikely: p<low"},
    "fallback": {"high": S2_HIGH, "low": S2_LOW, "note": "Stage 2 formula 0.5*semantic+0.3*lexical+0.2*containment with the old rules, used when model files are missing"},
    "models": {"cross_encoder_base": "cross-encoder/ms-marco-MiniLM-L-6-v2", "cross_encoder_weights": ["ce_seed0", "ce_seed1", "ce_seed2"],
               "seed_averaging": "mean of sigmoid(logit) over the 3 seeds (prob_mean), then logit(clip(prob_mean))",
               "cross_encoder_hyperparameters": {"max_length": 128, "lr": 2e-5, "batch_size": 16, "epochs": 3, "warmup": 0.10, "weight_decay": 0.01,
                                                 "loss": "BCE with pos_weight=#neg/#pos", "input": "(text_a, text_b) raw text, earlier unit first"},
               "embedding_model": "all-MiniLM-L6-v2", "stage2_formula": "0.50*semantic_score + 0.30*lexical_score + 0.20*containment"},
    "feature_version": "v1",
    "label_file": {"name": "pairs_v2.csv", "sha256_12": sha12(labels_path)}, "folds_file": {"name": "folds.json", "sha256_12": sha12(folds_path)},
    "training_rows": int(len(df)), "retakes": int(y.sum()),
}
(HERE / "thresholds.json").write_text(json.dumps(art, indent=2), encoding="utf-8")
sc_out = df.copy()
sc_out["stacker_prob_oof"], sc_out["stacker_prob_final"] = p_oof, p_final
sc_out["tier_nested"], sc_out["tier_final_thr"], sc_out["tier_stage2"] = nested_tier, final_tier, s2_tier
sc_out.to_csv(OUT / "final_oof_scores.csv", index=False)
(OUT / "fit_results.json").write_text(json.dumps({"results": res, "bootstrap_diff_vs_stage2": ci, "nested_thresholds": {k: list(v) for k, v in nested_thr.items()},
                                                  "nested_C": nested_C, "final_C": final_C, "final_thresholds": {"low": low_f, "high": high_f}}, indent=2), encoding="utf-8")
print("final C", final_C, "| coef", np.round(lr_.coef_[0], 3), "| thresholds low/high", round(low_f, 4), round(high_f, 4))
print("nested thresholds per fold:", {k: (round(a, 3), round(b, 3)) for k, (a, b) in nested_thr.items()})
for name in ("nested", "stage2", "final_thr_on_oof"):
    r = res[name]["pooled"]
    print(f"{name}: retake {r['retake_n']} (P {r['retake_precision']:.3f} R {r['retake_recall']:.3f}) | ambiguous {r['ambiguous_share']:.1%} | unlikely {r['unlikely_n']} (NPV {r['unlikely_precision']:.3f}, retakes lost {r['retakes_in_unlikely']})")
