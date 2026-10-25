"""Part B2-B5: does adding neighbor / gap / duplicate-structure features improve on the Stage 2 + cross-encoder stacker?

Models M0-M4 exactly as in results/neighbor/preregistration.md. Only existing out-of-fold scores are used (no cross-encoder
retraining, no API calls). For held-out fold k everything fitted (median imputation, scaler, C, weights, threshold) uses the
OTHER folds' rows only. Writes results/neighbor/neighbor.md, changed_pairs.csv and status.txt.
"""
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "neighbor"
RES = HERE / "results"
STATUS = OUT / "status.txt"
C_GRID, INNER, N_BOOT, SEED, S2_T = [0.01, 0.1, 1, 10, 100], 5, 1000, 0, 0.65
GAP = ["gap_units", "gap_seconds", "pause_before_b", "pause_after_a", "a_is_followed_by_b"]
DUP = ["dup_degree_a", "dup_degree_b", "rank_a", "rank_b", "b_best_for_a", "same_open_2", "same_open_3"]
NBR = ["nbr_sim_prev", "nbr_sim_next", "mid_has_restart_cue"]
BASE_F = ["current_stage2", "ce_logit"]
SETS = {"M0": BASE_F, "M1": BASE_F + GAP, "M2": BASE_F + DUP, "M3": BASE_F + NBR}
# M4 = all new features of the pre-registered B1 list. len_ratio, a_trunc are also B1 features not in any of M1-M3; M4 = "all new".
ALL_NEW = GAP + DUP + NBR + ["len_ratio", "a_trunc"]
SETS["M4"] = BASE_F + ALL_NEW


def main():
    t0 = time.time()
    # ---------------- load and align ----------------
    base = pd.read_csv(RES / "baseline_scores.csv", dtype={"pair_id": str, "video_id": str})
    ce = pd.read_csv(RES / "ce_scores.csv", dtype={"pair_id": str}).drop_duplicates("pair_id")
    feat = pd.read_csv(OUT / "neighbor_features.csv", dtype={"pair_id": str})
    df = base.merge(ce[["pair_id", "prob_mean"]], on="pair_id", validate="1:1").merge(feat, on="pair_id", validate="1:1")
    assert len(df) == 3570
    p = np.clip(df.prob_mean, 1e-6, 1 - 1e-6)
    df["ce_logit"] = np.log(p / (1 - p))
    y, folds = df.label.values, sorted(df.fold.unique())
    pv = pd.read_csv(HERE / "pairs_v1.csv", dtype=str, keep_default_na=False)[["pair_id", "text_a", "text_b", "notes"]]
    df = df.merge(pv, on="pair_id", how="left")
    errs = pd.read_csv(RES / "stage2_errors.csv", dtype=str)[["pair_id", "error", "error_type_guess"]]
    s2_pred = (df.current_stage2 >= S2_T).astype(int).values

    # ---------------- helpers ----------------
    def best_f1_thr(yt, s):
        prec, rec, thr = precision_recall_curve(yt, s)
        f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
        return float(thr[np.argmax(f1)])

    def prf(yt, pr):
        tp = int(((pr == 1) & (yt == 1)).sum())
        pp = tp / pr.sum() if pr.sum() else 0.0
        rr = tp / (yt == 1).sum() if (yt == 1).sum() else 0.0
        return pp, rr, (2 * pp * rr / (pp + rr) if pp + rr else 0.0)

    def rec_at_p(yt, s, pt):
        pr_, rc_, _ = precision_recall_curve(yt, s)
        ok = pr_ >= pt
        return float(rc_[ok].max()) if ok.any() else 0.0

    def make(C):
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty="l2", class_weight="balanced", max_iter=3000))

    def prep(train, test, feats):
        """Median imputation fitted on the training rows only (only matters for the NaN neighbor features)."""
        med = train[feats].median()
        return train[feats].fillna(med), test[feats].fillna(med)

    def choose_C(train, feats):
        """Inner GroupKFold by video on the training rows only; mean PR-AUC; ties -> smaller C."""
        sc = {C: [] for C in C_GRID}
        for tr, va in GroupKFold(n_splits=INNER).split(train, train.label, train.video_id):
            a, b = train.iloc[tr], train.iloc[va]
            if b.label.sum() == 0:
                continue
            Xa, Xb = prep(a, b, feats)
            for C in C_GRID:
                sc[C].append(average_precision_score(b.label, make(C).fit(Xa, a.label).predict_proba(Xb)[:, 1]))
        return max(C_GRID, key=lambda C: (np.mean(sc[C]), -C))

    def run(feats, fixed_C=None, importance=False):
        """Out-of-fold probabilities, decisions at a training-chosen threshold, coefficients, permutation importance."""
        score, dec = np.zeros(len(df)), np.zeros(len(df), int)
        coefs, imps, chosen = [], [], {}
        for k in folds:
            te = (df.fold == k).values
            tr_df, te_df = df[~te], df[te]
            C = fixed_C or choose_C(tr_df, feats)
            Xtr, Xte = prep(tr_df, te_df, feats)
            m = make(C).fit(Xtr, tr_df.label)
            score[te] = m.predict_proba(Xte)[:, 1]
            thr = best_f1_thr(tr_df.label, m.predict_proba(Xtr)[:, 1])     # threshold from training folds only
            dec[te] = (score[te] >= thr).astype(int)
            chosen[int(k)] = C
            for f, c in zip(feats, m[-1].coef_[0]):
                coefs.append({"fold": k, "feature": f, "coef": c})
            if importance:                                             # permutation importance on the held-out fold (PR-AUC drop)
                rng = np.random.default_rng(SEED)
                base_ap = average_precision_score(te_df.label, score[te])
                for f in feats:
                    drops = []
                    for _ in range(10):
                        Xp = Xte.copy()
                        Xp[f] = rng.permutation(Xp[f].values)
                        drops.append(base_ap - average_precision_score(te_df.label, m.predict_proba(Xp)[:, 1]))
                    imps.append({"fold": k, "feature": f, "importance": float(np.mean(drops))})
        return score, dec, pd.DataFrame(coefs), pd.DataFrame(imps), chosen

    # ---------------- B2: models ----------------
    out = {}
    for name, feats in SETS.items():
        out[name] = run(feats, importance=name != "M0")
        print(f"{name} done ({time.time() - t0:.0f}s) C per fold {out[name][4]}")
    score = {"current_stage2": df.current_stage2.values, **{n: out[n][0] for n in SETS}}
    dec = {"current_stage2": s2_pred, **{n: out[n][1] for n in SETS}}

    # M0 reproduction check vs results/combine.md (stack = LR on stage2 + ce_logit, C=1, no C search)
    m0_fix = run(BASE_F, fixed_C=1)[0]
    comb = pd.read_csv(RES / "combine_scores.csv", dtype={"pair_id": str}).set_index("pair_id")
    ref = df.pair_id.map(comb.score_stack).values
    repro_ok = bool(np.allclose(m0_fix, ref, atol=1e-9))
    pr_fixed, pr_ref = average_precision_score(y, m0_fix), average_precision_score(y, ref)

    # ---------------- B3: metrics ----------------
    scopes = [("pooled", np.ones(len(df), bool))] + [(f"fold{k}", (df.fold == k).values) for k in folds]
    names = list(score)

    def table(metric_fn, fmt="{:.3f}"):
        return ["| model | " + " | ".join(s for s, _ in scopes) + " |", "|---|" + "---|" * len(scopes)] + [
            f"| {n} | " + " | ".join(fmt.format(metric_fn(n, m)) for _, m in scopes) + " |" for n in names]

    md = ["# Neighbor / gap features vs the Stage 2 + cross-encoder stacker", "",
          "Pre-registered in preregistration.md before any model was run; those are the only comparisons. Anything else is labelled exploratory. "
          "Out-of-fold over the 4 speaker-grouped folds, same 3570 rows as baselines. For each held-out fold: median imputation, StandardScaler, "
          "L2, class_weight=balanced, C from an inner GroupKFold(5) by video on the training folds only; decision threshold = best F1 on the training "
          "folds' in-sample probabilities. Caveat as in combine.md: the cross-encoder OOF scores of the training folds came from models that saw the "
          "held-out fold's labels, so the stacker is mildly optimistic for every model including M0. M4 = M0 + all 17 new features "
          "(M1-M3 sets plus len_ratio and a_trunc).", "",
          "## M0 reproduction check", "",
          f"M0 with C fixed to 1 (as in combine.py) reproduces combine_scores.csv `stack` exactly: {repro_ok} "
          f"(pooled PR-AUC {pr_fixed:.4f} vs {pr_ref:.4f} in combine.md). The official M0 below picks C by inner CV, so it can differ slightly "
          f"(chosen C per fold: {out['M0'][4]}).", ""]
    for title, fn in [("PR-AUC", lambda n, m: average_precision_score(y[m], score[n][m])),
                      ("Recall at 90% precision", lambda n, m: rec_at_p(y[m], score[n][m], 0.90)),
                      ("Recall at 80% precision", lambda n, m: rec_at_p(y[m], score[n][m], 0.80)),
                      ("Precision at training-chosen threshold (current_stage2 uses 0.65)", lambda n, m: prf(y[m], dec[n][m])[0]),
                      ("Recall at training-chosen threshold", lambda n, m: prf(y[m], dec[n][m])[1]),
                      ("F1 at training-chosen threshold", lambda n, m: prf(y[m], dec[n][m])[2])]:
        md += [f"### {title}", ""] + table(fn) + [""]
    md += ["Chosen C per fold: " + "; ".join(f"{n}: {out[n][4]}" for n in SETS), ""]

    # paired video-level bootstrap of the pooled PR-AUC difference
    rng = np.random.default_rng(SEED)
    vids = df.video_id.unique()
    ix = {v: np.where(df.video_id.values == v)[0] for v in vids}
    boot = {n: [] for n in names}
    for _ in range(N_BOOT):
        idx = np.concatenate([ix[v] for v in rng.choice(vids, len(vids), replace=True)])
        for n in names:
            boot[n].append(average_precision_score(y[idx], score[n][idx]) if y[idx].sum() else np.nan)
    boot = {n: np.array(v) for n, v in boot.items()}
    comps = [("M1", "M0"), ("M2", "M0"), ("M3", "M0"), ("M4", "M0"), ("M0", "current_stage2"), ("M4", "current_stage2")]
    md += ["### Pre-registered paired video-level bootstrap: pooled PR-AUC difference (1000 resamples, seed 0)", "",
           "| comparison | diff | 95% CI | CI excludes 0 |", "|---|---|---|---|"]
    ap = {n: average_precision_score(y, score[n]) for n in names}
    wins = []
    for a, b in comps:
        d = boot[a] - boot[b]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        ex = bool(lo > 0 or hi < 0)
        md.append(f"| {a} - {b} | {ap[a] - ap[b]:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {ex} |")
        if ex and ap[a] > ap[b]:
            wins.append(f"{a} - {b}")
    md.append("")

    # coefficients and permutation importance of the new features
    md += ["### Standardized coefficients of the new features per fold (sign flips flagged)", ""]
    for n in ["M1", "M2", "M3", "M4"]:
        co = out[n][2]
        new = [f for f in SETS[n] if f not in BASE_F]
        md += [f"**{n}**", "", "| feature | " + " | ".join(f"fold{k}" for k in folds) + " | sign flip |", "|---|" + "---|" * (len(folds) + 1)]
        for f in new:
            v = co[co.feature == f].sort_values("fold").coef.values
            md.append(f"| {f} | " + " | ".join(f"{x:+.3f}" for x in v) + f" | {'YES' if (v > 0).any() and (v < 0).any() else 'no'} |")
        v0 = [co[co.feature == b_].sort_values('fold').coef.values for b_ in BASE_F]
        md += [f"| (stage2, ce_logit for reference) | " + "; ".join(" ".join(f"{x:+.2f}" for x in vv) for vv in v0) + " | |", ""]
    md += ["### Permutation importance on the held-out fold (mean PR-AUC drop over folds, 10 shuffles per fold; new features only)", "",
           "| feature | M1 | M2 | M3 | M4 |", "|---|---|---|---|---|"]
    imp = {n: out[n][3].groupby("feature").importance.mean() for n in ["M1", "M2", "M3", "M4"]}
    for f in ALL_NEW:
        md.append(f"| {f} | " + " | ".join(f"{imp[n].get(f, np.nan):+.4f}" if f in SETS[n] else "-" for n in ["M1", "M2", "M3", "M4"]) + " |")
    md.append("")

    # ---------------- B4: error analysis ----------------
    e = errs.set_index("pair_id")
    df["s2_err"] = df.pair_id.map(e.error).fillna("")
    df["guess"] = df.pair_id.map(e.error_type_guess).fillna("")
    md += ["### B4 Errors vs the 79 Stage 2 errors (Stage 2 @0.65: 49 FN, 30 FP)", "",
           "| model | FN | FP | fixed FN | fixed FP | new FN | new FP |", "|---|---|---|---|---|---|---|"]
    changed = []
    for n in ["M0", "M4"]:
        pr_ = dec[n]
        fixed_fn = (y == 1) & (s2_pred == 0) & (pr_ == 1); fixed_fp = (y == 0) & (s2_pred == 1) & (pr_ == 0)
        new_fn = (y == 1) & (s2_pred == 1) & (pr_ == 0); new_fp = (y == 0) & (s2_pred == 0) & (pr_ == 1)
        md.append(f"| {n} | {int(((y == 1) & (pr_ == 0)).sum())} | {int(((y == 0) & (pr_ == 1)).sum())} | {int(fixed_fn.sum())} | {int(fixed_fp.sum())} | {int(new_fn.sum())} | {int(new_fp.sum())} |")
        for st, m in [("fixed_FN", fixed_fn), ("fixed_FP", fixed_fp), ("new_FN", new_fn), ("new_FP", new_fp)]:
            c = df[m][["pair_id", "video_id", "fold", "label", "current_stage2", "notes", "guess", "text_a", "text_b"]].copy()
            c["model"], c["status"] = n, st
            c["model_score"] = score[n][m]
            changed.append(c)
    ch = pd.concat(changed)
    ch.to_csv(OUT / "changed_pairs.csv", index=False)
    md += ["", "Fixed Stage 2 errors by error_type_guess (guess comes from stage2_errors.csv):", "",
           "| model | " + " | ".join(sorted(df.guess[df.guess != ''].unique())) + " |", "|---|" + "---|" * df.guess[df.guess != ''].nunique()]
    for n in ["M0", "M4"]:
        fx = ch[(ch.model == n) & ch.status.str.startswith("fixed")]
        tot = df[df.guess != ""].groupby("guess").size()
        md.append(f"| {n} | " + " | ".join(f"{int((fx.guess == g).sum())}/{tot[g]}" for g in sorted(tot.index)) + " |")
    md.append("")
    grp = {"TP (Stage 2 right, label 1)": (y == 1) & (s2_pred == 1), "FN": (y == 1) & (s2_pred == 0), "FP": (y == 0) & (s2_pred == 1),
           "FP: video_02 poetry": (y == 0) & (s2_pred == 1) & (df.video_id == "video_02").values & (df.notes == "poetry").values,
           "FP: speaker_a_6": (y == 0) & (s2_pred == 1) & (df.video_id == "speaker_a_6").values}
    keyf = ["dup_degree_a", "dup_degree_b", "rank_a", "rank_b", "b_best_for_a", "same_open_2", "gap_units"]
    md += ["### B4 Medians of structure features (do they separate repeated-structure FPs from true positives?)", "",
           "| group | n | " + " | ".join(keyf) + " |", "|---|---|" + "---|" * len(keyf)]
    for g, m in grp.items():
        md.append(f"| {g} | {int(m.sum())} | " + " | ".join(f"{df.loc[m, f].median():.1f}" if m.sum() else "-" for f in keyf) + " |")
    md.append("")

    # ---------------- B5: leave-one-feature-out within M4 ----------------
    md += ["### B5 Leave-one-feature-out within M4 (pooled PR-AUC change when the feature is removed; negative = the feature helped)", "",
           "| removed feature | PR-AUC without it | change vs M4 |", "|---|---|---|"]
    for f in ALL_NEW:
        s = run([x for x in SETS["M4"] if x != f])[0]
        md.append(f"| {f} | {average_precision_score(y, s):.4f} | {average_precision_score(y, s) - ap['M4']:+.4f} |")
    md.append("")

    # ---------------- summary ----------------
    best = max(["M1", "M2", "M3", "M4"], key=lambda n: ap[n])
    d_best = boot[best] - boot["M0"]
    lo, hi = np.nanpercentile(d_best, [2.5, 97.5])
    beats = [w for w in wins if w.endswith("- M0")]
    md += ["## Summary", "",
           f"1. M0 (Stage 2 + cross-encoder stacker) pooled PR-AUC {ap['M0']:.3f} vs Stage 2 {ap['current_stage2']:.3f}; M4 {ap['M4']:.3f}.",
           f"2. Best new-feature model by point estimate: {best} {ap[best]:.3f} ({ap[best] - ap['M0']:+.4f} vs M0, 95% CI [{lo:+.4f}, {hi:+.4f}]).",
           ("3. At least one new-feature model beats M0 with a CI excluding 0: " + ", ".join(beats) + "." if beats else
            "3. No new-feature model beats M0 with a paired-bootstrap CI that excludes 0."),
           f"4. M0 reproduces combine.md's stacker exactly at C=1: {repro_ok}; official M0 uses inner-CV C.",
           f"5. Errors fixed/new vs Stage 2: see B4 table; features' coefficients with sign flips are flagged above. Runtime {time.time() - t0:.0f}s."]
    (OUT / "neighbor.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    try:
        main()
        STATUS.write_text("done\n")
    except Exception:
        traceback.print_exc()
        STATUS.write_text("failed\n" + traceback.format_exc())
        raise
