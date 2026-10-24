"""Part 1: Stage 2 error analysis (no training).  Part 2: gradient boosting, nested speaker-grouped CV.

Real rows only (label != -1), folds from folds.json. Reads pairs_v1.csv and
results/baseline_scores.csv; never modifies inputs.
Writes results/stage2_errors.csv, results/gbm.md, results/gbm_scores.csv.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
THRESHOLD = 0.65
N_BOOT, SEED, INNER_SPLITS = 1000, 0, 5

# ---------------- load (ids as strings, scores as floats) ----------------
df = pd.read_csv(HERE / "pairs_v1.csv", dtype=str, keep_default_na=False)
folds = json.loads((HERE / "folds.json").read_text())["folds"]
df = df[(df.is_synthetic == "0") & (df.label != "-1")].copy()
NUM = ["sequence", "jaccard", "containment", "bigram", "lexical_score", "semantic_score",
       "recency_weight", "restart_phrase"]
for c in NUM:
    df[c] = pd.to_numeric(df[c], errors="raise")
    assert df[c].notna().all(), f"NaN in {c}"
df["label"] = df.label.astype(int)
df["fold"] = df.video_id.map(folds).astype(int)
base = pd.read_csv(OUT / "baseline_scores.csv", dtype={"pair_id": str, "video_id": str})
df = df.merge(base[["pair_id", "current_stage2"]], on="pair_id", how="inner").reset_index(drop=True)
assert len(df) == len(base), "row set differs from baseline_scores.csv"

# ======================= Part 1: Stage 2 errors =======================
df["pred"] = (df.current_stage2 >= THRESHOLD).astype(int)
tok = lambda s: len(s.split())
df["tokens_a"], df["tokens_b"] = df.text_a.map(tok), df.text_b.map(tok)


def guess(r):
    """Simple rule-based guess at why Stage 2 got this row wrong (for manual review)."""
    if r.tokens_a < 4 and r.tokens_b < 4:
        return "very_short"
    if r.containment >= 0.8 and r.sequence < 0.6:
        return "truncation"                      # one unit is mostly contained in the other
    if r.lexical_score < 0.5 and r.semantic_score >= 0.8:
        return "paraphrase"
    if r.label == 0 and r.lexical_score >= 0.8:
        return "near_duplicate_not_retake"       # FP: very similar text but labelled not a retake
    if r.label == 1 and r.lexical_score < 0.5:
        return "low_overlap_retake"
    return "other"


errs = df[df.pred != df.label].copy()
errs["error"] = np.where(errs.label == 1, "FN", "FP")
errs["error_type_guess"] = errs.apply(guess, axis=1)
errs["dist_from_threshold"] = (errs.current_stage2 - THRESHOLD).abs()
errs = errs.sort_values("dist_from_threshold", ascending=False)
cols = ["error", "pair_id", "video_id", "fold", "label", "current_stage2", "text_a", "text_b", "lexical_score",
        "semantic_score", "containment", "recency_weight", "pipeline_class", "notes", "error_type_guess",
        "dist_from_threshold"]
errs[cols].to_csv(OUT / "stage2_errors.csv", index=False)

print(f"Stage 2 @ {THRESHOLD}: {int((errs.error == 'FN').sum())} FN, {int((errs.error == 'FP').sum())} FP")
print(pd.crosstab(errs.fold, errs.error).to_string(), "\n")
print(pd.crosstab(errs.error_type_guess, errs.error).to_string(), "\n")
for kind in ("FN", "FP"):
    print(f"=== 25 most confident {kind} ===")
    for r in errs[errs.error == kind].head(25).itertuples():
        print(f"[{r.current_stage2:.3f}] {r.pair_id} fold{r.fold} {r.error_type_guess} "
              f"lex={r.lexical_score:.2f} sem={r.semantic_score:.2f} cont={r.containment:.2f} "
              f"rec={r.recency_weight:.2f} class={r.pipeline_class} notes={r.notes!r}\n"
              f"    A: {r.text_a}\n    B: {r.text_b}")

# ======================= Part 2: gradient boosting =======================
# extra features. Lengths are token counts of the two texts. gap_units = difference in
# transcript segment number between the two units, derivable from pair_id ("<video>__<seg>-<sub>__<seg>-<sub>").
df["len_a"], df["len_b"] = df.tokens_a, df.tokens_b
df["len_ratio"] = np.minimum(df.len_a, df.len_b) / np.maximum(np.maximum(df.len_a, df.len_b), 1)
seg = lambda s: int(s.split("-")[0])
parts = df.pair_id.str.split("__", expand=True)
df["gap_units"] = parts[2].map(seg) - parts[1].map(seg)
assert (df.gap_units >= 0).all(), "unit order unexpected"
FEATS = ["sequence", "jaccard", "containment", "bigram", "semantic_score", "recency_weight", "restart_phrase",
         "len_a", "len_b", "len_ratio", "gap_units"]
MAX_ITER, LEAF_GRID = 400, [20, 100]


def make_gbm(n_iter, leaf):
    # fixed max_depth 3, lr 0.05; n_iter picked by inner grouped early stopping
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=n_iter,
                                          min_samples_leaf=leaf, class_weight="balanced",
                                          early_stopping=False, random_state=0)


def tune(train):
    """Inner GroupKFold by video: for each min_samples_leaf, track validation PR-AUC after every
    boosting iteration (staged predictions = early stopping on a grouped inner validation set),
    average over inner folds, take the best (leaf, n_iter). Training rows only."""
    best = (-1, None, None)
    for leaf in LEAF_GRID:
        curves = []
        for tr_idx, va_idx in GroupKFold(n_splits=INNER_SPLITS).split(train, train.label, train.video_id):
            tr, va = train.iloc[tr_idx], train.iloc[va_idx]
            if va.label.sum() == 0:
                continue
            m = make_gbm(MAX_ITER, leaf).fit(tr[FEATS], tr.label)
            curves.append([average_precision_score(va.label, p[:, 1]) for p in m.staged_predict_proba(va[FEATS])])
        mean_curve = np.mean(curves, axis=0)
        n_iter = int(np.argmax(mean_curve)) + 1
        if mean_curve[n_iter - 1] > best[0]:
            best = (mean_curve[n_iter - 1], leaf, n_iter)
    return best[1], best[2]


def recall_at_precision(y, s, p):
    prec, rec, _ = precision_recall_curve(y, s)
    ok = prec >= p
    return float(rec[ok].max()) if ok.any() else 0.0


def rank_metrics(y, s):
    return {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
            "recall_at_p90": recall_at_precision(y, s, 0.90), "recall_at_p80": recall_at_precision(y, s, 0.80)}


oof = np.zeros(len(df))
imp_rows, cfg = [], []
for k in sorted(df.fold.unique()):
    te = (df.fold == k).values
    train, test = df[~te].reset_index(drop=True), df[te]
    leaf, n_iter = tune(train)
    m = make_gbm(n_iter, leaf).fit(train[FEATS], train.label)
    oof[te] = m.predict_proba(test[FEATS])[:, 1]
    # permutation importance on the held-out fold (drop in PR-AUC); not used for any tuning
    pi = permutation_importance(m, test[FEATS], test.label, scoring="average_precision",
                                n_repeats=10, random_state=0)
    for f, mu, sd in zip(FEATS, pi.importances_mean, pi.importances_std):
        imp_rows.append({"fold": int(k), "feature": f, "importance": mu, "std": sd})
    cfg.append((int(k), leaf, n_iter))
    print(f"fold {k}: min_samples_leaf={leaf}, n_iter={n_iter}")

df["gbm"] = oof
df[["pair_id", "video_id", "fold", "label", "gbm"]].rename(columns={"gbm": "prob"}).to_csv(
    OUT / "gbm_scores.csv", index=False)

# ---------------- metrics + paired bootstrap vs Stage 2 ----------------
models = {"current_stage2": "current_stage2", "gbm": "gbm"}
scopes = [("pooled", df)] + [(f"fold{k}", df[df.fold == k]) for k in sorted(df.fold.unique())]
metrics = {(n, s, m): v for s, g in scopes for n, col in models.items()
           for m, v in rank_metrics(g.label, g[col]).items()}

rng = np.random.default_rng(SEED)
vids = df.video_id.unique()
by_vid = {v: g for v, g in df.groupby("video_id")}
bs = {n: [] for n in models}
for _ in range(N_BOOT):
    samp = pd.concat([by_vid[v] for v in rng.choice(vids, len(vids), replace=True)])
    for n, col in models.items():
        bs[n].append(average_precision_score(samp.label, samp[col]) if samp.label.sum() else np.nan)
bs = {n: np.array(v) for n, v in bs.items()}
ci = {n: np.nanpercentile(bs[n], [2.5, 97.5]) for n in models}
diff = bs["gbm"] - bs["current_stage2"]
d_lo, d_hi = np.nanpercentile(diff, [2.5, 97.5])
d_pt = metrics[("gbm", "pooled", "pr_auc")] - metrics[("current_stage2", "pooled", "pr_auc")]

# ---------------- report ----------------
sc = ["pooled", "fold1", "fold2", "fold3", "fold4"]
md = ["# Gradient boosting results", "",
      f"HistGradientBoostingClassifier (max_depth=3, lr=0.05, class_weight=balanced). Real rows, label -1 excluded: "
      f"{len(df)} rows, {int(df.label.sum())} positives. Nested speaker-grouped CV; inner GroupKFold({INNER_SPLITS}) by "
      f"video chooses min_samples_leaf from {LEAF_GRID} and the number of boosting iterations (early stopping on the "
      "grouped inner validation PR-AUC). Features: the 7 sub-scores plus len_a, len_b, len_ratio (token counts) and "
      "gap_units (segment-number difference taken from pair_id; derivable, so included).", "",
      "Chosen per fold (min_samples_leaf, n_iter): " + "; ".join(f"fold{k}: ({l}, {n})" for k, l, n in cfg), ""]
for title, m in [("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc"), ("Recall at 90% precision", "recall_at_p90"),
                 ("Recall at 80% precision", "recall_at_p80")]:
    md += [f"### {title}", "", "| model | " + " | ".join(sc) + " |", "|---|" + "---|" * len(sc)]
    md += [f"| {n} | " + " | ".join(f"{metrics[(n, s, m)]:.3f}" for s in sc) + " |" for n in models]
    md.append("")
md += ["Pooled PR-AUC 95% CI (1000 video-level bootstrap resamples, seed 0): "
       + "; ".join(f"{n} [{ci[n][0]:.3f}, {ci[n][1]:.3f}]" for n in models), "",
       f"**Paired difference in pooled PR-AUC (gbm minus current_stage2): {d_pt:+.3f}, 95% CI [{d_lo:+.3f}, {d_hi:+.3f}]**", "",
       "### Permutation importance (drop in PR-AUC when the feature is shuffled, on the held-out fold; mean of 10 repeats)", ""]
imp = pd.DataFrame(imp_rows)
imp.to_csv(OUT / "gbm_importance.csv", index=False)
tbl = imp.pivot(index="feature", columns="fold", values="importance").reindex(FEATS)
tbl["mean"] = tbl.mean(axis=1)
md += ["| feature | " + " | ".join(f"fold{k}" for k in tbl.columns[:-1]) + " | mean |", "|---|" + "---|" * len(tbl.columns)]
md += [f"| {f} | " + " | ".join(f"{v:+.4f}" for v in r.values) + " |" for f, r in tbl.sort_values("mean", ascending=False).iterrows()]
verdict = "outside the noise (CI excludes 0)" if (d_lo > 0 or d_hi < 0) else "inside the noise (CI includes 0)"
md += ["", "## Summary", "",
       f"1. GBM pooled PR-AUC {metrics[('gbm', 'pooled', 'pr_auc')]:.3f} vs Stage 2 {metrics[('current_stage2', 'pooled', 'pr_auc')]:.3f}: "
       f"diff {d_pt:+.3f}, 95% CI [{d_lo:+.3f}, {d_hi:+.3f}], {verdict}.",
       f"2. Stage 2 errors at {THRESHOLD}: {int((errs.error == 'FN').sum())} FN and {int((errs.error == 'FP').sum())} FP "
       "(see results/stage2_errors.csv; error_type_guess is rule-based and needs manual review).",
       "3. Per-fold differences are small-sample; do not over-read them."]
(OUT / "gbm.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
