"""PART A: matched-precision comparison of S1 Stage 2, S2 cross-encoder alone, S3 stacker, S4 rank average (reference).
Everything is fixed in results_v2/final/preregistration_matched.md (written before any number was computed). Uses only saved out-of-fold
scores; no retraining, no API calls. Writes results_v2/final/matched.md, matched_scores.csv, pr_curves_matched.png, status_matched.txt."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parents[1] / "data" / "training"
OUT = TRAIN / "results_v2" / "final"
STATUS = OUT / "status_matched.txt"
TARGETS = [0.90, 0.95, 0.98, 0.99]
LOW_RECALL = 0.98
N_BOOT, SEED = 1000, 0
SYS = {"S1": ("Stage 2 plain formula", "stage2_score"), "S2": ("Cross-encoder alone", "ce_prob_mean"),
       "S3": ("Stacker (Stage 2 + CE logit)", "stacker_prob_oof"), "S4": ("Rank average (reference)", "score_rank_avg")}

try:
    df = pd.read_csv(OUT / "final_oof_scores.csv", dtype={"pair_id": str, "video_id": str})
    cb = pd.read_csv(TRAIN / "results_v2" / "combine_scores.csv", dtype={"pair_id": str})[["pair_id", "score_rank_avg"]]
    df = df.merge(cb, on="pair_id", validate="1:1")
    assert len(df) == 3562 and df.label.isin([0, 1]).all()
    y, fold, vid = df.label.values, df.fold.values, df.video_id.values
    FOLDS = sorted(np.unique(fold))

    # ---------------- thresholds ----------------
    def curve(yt, s):
        order = np.argsort(-s, kind="stable")
        ss, yy = s[order], yt[order]
        tp, n = np.cumsum(yy), np.arange(1, len(ss) + 1)
        last = np.r_[ss[1:] != ss[:-1], True]               # threshold t = ss[i] includes every row with score >= t
        return ss[last], tp[last] / n[last], tp[last] / max(1, yy.sum())

    def high_for(yt, s, target):
        thr, prec, _ = curve(yt, s)
        ok = np.where(prec >= target)[0]
        return float(thr[ok.max()]) if len(ok) else float("inf")       # lowest t reaching the target; inf = unreachable

    def low_for(yt, s):
        thr, _, rec = curve(yt, s)
        ok = np.where(rec >= LOW_RECALL)[0]
        return float(thr[ok.min()]) if len(ok) else float("-inf")      # highest t with recall >= 0.98

    tiers, thr_tab = {}, []
    for sid, (_, col) in SYS.items():
        s = df[col].values
        for tg in TARGETS:
            t = np.empty(len(df), dtype=object)
            for k in FOLDS:
                te = fold == k
                hi, lo = high_for(y[~te], s[~te], tg), low_for(y[~te], s[~te])
                lo = min(lo, hi)
                t[te] = np.where(s[te] >= hi, "retake", np.where(s[te] < lo, "unlikely", "ambiguous"))
                thr_tab.append({"system": sid, "target": tg, "fold": int(k), "high": hi, "low": lo, "unreachable": bool(np.isinf(hi))})
            tiers[(sid, tg)] = t
    thr_df = pd.DataFrame(thr_tab)

    # ---------------- metrics ----------------
    def metrics(mask, t):
        yy, tt = y[mask], t[mask]
        P = int((yy == 1).sum())
        cut, amb, unl = tt == "retake", tt == "ambiguous", tt == "unlikely"
        auto, tp = int(cut.sum()), int(((yy == 1) & cut).sum())
        return {"auto_cuts": auto, "true_cuts": tp, "false_cuts": auto - tp, "precision": tp / auto if auto else float("nan"), "recall": tp / max(1, P),
                "retakes_in_review": int(((yy == 1) & amb).sum()), "retakes_in_unlikely": int(((yy == 1) & unl).sum()), "review_queue": int(amb.sum()),
                "ambiguous_share": float(amb.mean()), "n": int(mask.sum()), "retakes": P}

    M = {}
    for (sid, tg), t in tiers.items():
        M[(sid, tg, "pooled")] = metrics(np.ones(len(df), bool), t)
        for k in FOLDS:
            M[(sid, tg, int(k))] = metrics(fold == k, t)

    # ---------------- paired video-level bootstrap (tiers fixed per row) ----------------
    vids = pd.unique(vid)
    per_v = {}
    for key, t in tiers.items():
        rows = []
        for v in vids:
            m = vid == v
            yy, tt = y[m], t[m]
            rows.append([int(((yy == 1) & (tt == "retake")).sum()), int(((yy == 0) & (tt == "retake")).sum()), int((tt == "ambiguous").sum()), int((yy == 1).sum())])
        per_v[key] = np.array(rows)
    v_fold = np.array([fold[vid == v][0] for v in vids])
    rng = np.random.default_rng(SEED)
    draws = [rng.choice(len(vids), len(vids), replace=True) for _ in range(N_BOOT)]
    f3_idx = np.where(v_fold == 3)[0]
    f3_draws = [rng.choice(f3_idx, len(f3_idx), replace=True) for _ in range(N_BOOT)]

    def stat(counts, idx):
        c = counts[idx].sum(axis=0)
        tp, fp, amb, P = c
        return {"recall": tp / max(1, P), "precision": tp / (tp + fp) if tp + fp else np.nan, "false_cuts": fp, "review_queue": amb}

    def boot_diff(a, b, tg, draw_set):
        out = {k: [] for k in ("recall", "precision", "false_cuts", "review_queue")}
        for idx in draw_set:
            sa, sb = stat(per_v[(a, tg)], idx), stat(per_v[(b, tg)], idx)
            for k in out:
                out[k].append(sa[k] - sb[k])
        res = {}
        for k, v in out.items():
            v = np.array(v, dtype=float)
            lo, hi = np.nanpercentile(v, [2.5, 97.5])
            res[k] = (lo, hi)
        return res

    full = np.arange(len(vids))
    BD, F3 = {}, {}
    for a in ("S3", "S2"):
        for tg in TARGETS:
            BD[(a, tg)] = boot_diff(a, "S1", tg, draws)
            F3[(a, tg)] = boot_diff(a, "S1", tg, f3_draws)

    # ---------------- plot ----------------
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors, marks = {"S1": "#1f77b4", "S2": "#d62728", "S3": "#2ca02c"}, {0.90: "o", 0.95: "s", 0.98: "^", 0.99: "D"}
    for sid in ("S1", "S2", "S3"):
        prec, rec, _ = precision_recall_curve(y, df[SYS[sid][1]].values)
        ax.plot(rec, prec, color=colors[sid], lw=1.6, label=f"{sid} {SYS[sid][0]}")
        for tg in TARGETS:
            m = M[(sid, tg, "pooled")]
            if not np.isnan(m["precision"]):
                ax.scatter(m["recall"], m["precision"], marker=marks[tg], color=colors[sid], edgecolor="black", s=60, zorder=5)
    for tg in TARGETS:
        ax.axhline(tg, color="grey", lw=0.5, ls=":")
        ax.scatter([], [], marker=marks[tg], color="white", edgecolor="black", label=f"nested operating point, target {tg:.2f}")
    ax.set_xlabel("recall (share of all retakes)"); ax.set_ylabel("precision (retake class)")
    ax.set_title("Pooled out-of-fold PR curves; markers = nested operating points", fontsize=10)
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.6, 1.01); ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(OUT / "pr_curves_matched.png", dpi=150); plt.close(fig)

    # ---------------- decision rule and summary (rule fixed in the pre-registration) ----------------
    pos_ci = lambda a, tg: BD[(a, tg)]["recall"][0] > 0
    s3_beats = [tg for tg in TARGETS if pos_ci("S3", tg)]
    s2_beats = [tg for tg in TARGETS if pos_ci("S2", tg)]
    rule_met = pos_ci("S3", 0.95) and pos_ci("S3", 0.99)
    f3_gain = {tg: M[("S3", tg, 3)]["recall"] - M[("S1", tg, 3)]["recall"] for tg in TARGETS}
    f3_value = sum(g > 0 for g in f3_gain.values()) >= 3 and F3[("S3", 0.95)]["recall"][0] > 0
    recommendation = ("STACKER as the default scorer" if rule_met else "TUNED STAGE 2 as the default scorer; the stacker as an optional mode")

    pct = lambda x: f"{100 * x:.1f}%"
    fm = lambda x: "n/a" if x != x else f"{x:.3f}"
    md = ["# Matched-precision comparison (Part A)", "",
          "Pre-registered in `preregistration_matched.md` before any number was computed. Saved out-of-fold scores only, 3,562 labelled rows / 340 retakes, nested over the 4 speaker folds "
          "(thresholds for each held-out fold from the other folds only). **Caveat:** the other folds' out-of-fold scores came from models that saw the held-out fold's labels (mild optimism, same for every system). "
          "Systems are matched on the precision TARGET; the precision actually reached out of sample is shown and differs between systems.", "",
          "## Summary", ""]
    md += [f"1. **S3 (stacker) vs S1 (tuned Stage 2), recall at matched precision target, CI excluding 0 (positive): " + (", ".join(f"{t:.2f}" for t in s3_beats) if s3_beats else "none of 0.90 / 0.95 / 0.98 / 0.99") + ".**  "
           + "; ".join(f"{tg:.2f}: {BD[('S3', tg)]['recall'][0] + 0:+.3f}..{BD[('S3', tg)]['recall'][1]:+.3f} (point {M[('S3', tg, 'pooled')]['recall'] - M[('S1', tg, 'pooled')]['recall']:+.3f})" for tg in TARGETS) + ".",
           f"2. S2 (cross-encoder alone) vs S1: recall CI excluding 0 (positive) at " + (", ".join(f"{t:.2f}" for t in s2_beats) if s2_beats else "no target") + "; point differences " + ", ".join(f"{tg:.2f}: {M[('S2', tg, 'pooled')]['recall'] - M[('S1', tg, 'pooled')]['recall']:+.3f}" for tg in TARGETS) + ".",
           f"3. **Fold 3:** S3 minus S1 recall " + ", ".join(f"{tg:.2f}: {g:+.3f}" for tg, g in f3_gain.items()) + f"; bootstrap CI over the 3 fold-3 videos at 0.95: [{F3[('S3', 0.95)]['recall'][0]:+.3f}, {F3[('S3', 0.95)]['recall'][1]:+.3f}]. "
           + ("The cross-encoder ADDS value in fold 3 under the pre-registered criterion." if f3_value else "No clear value from the cross-encoder in fold 3 under the pre-registered criterion (>= 3 of 4 targets with a positive gain AND a 0.95 CI above 0)."),
           f"4. **Recommendation (pre-registered rule, unchanged): {recommendation}.** Rule: stacker is the default only if S3 beats S1 in recall with a CI excluding 0 at BOTH 0.95 and 0.99; "
           f"here 0.95: {'yes' if pos_ci('S3', 0.95) else 'no'} (CI [{BD[('S3', 0.95)]['recall'][0]:+.3f}, {BD[('S3', 0.95)]['recall'][1]:+.3f}]), 0.99: {'yes' if pos_ci('S3', 0.99) else 'no'} (CI [{BD[('S3', 0.99)]['recall'][0]:+.3f}, {BD[('S3', 0.99)]['recall'][1]:+.3f}]).",
           "5. Achieved out-of-fold precision at the four targets (S1 / S2 / S3): " + "; ".join(f"{tg:.2f}: {fm(M[('S1', tg, 'pooled')]['precision'])} / {fm(M[('S2', tg, 'pooled')]['precision'])} / {fm(M[('S3', tg, 'pooled')]['precision'])}" for tg in TARGETS)
           + f"; unreachable folds (S1 / S2 / S3): " + ", ".join(f"{int(thr_df[(thr_df.system == s) & thr_df.unreachable].shape[0])}" for s in ("S1", "S2", "S3")) + " (of 16 fold-target pairs each).", ""]

    md += ["## Pooled results (counts summed over the 4 held-out folds)", "",
           "| target | system | achieved precision | recall | auto-cuts | false cuts | retakes in review queue | retakes in 'unlikely' | ambiguous share | review-queue pairs | unreachable folds |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tg in TARGETS:
        for sid in SYS:
            m = M[(sid, tg, "pooled")]
            un = int(thr_df[(thr_df.system == sid) & (thr_df.target == tg) & thr_df.unreachable].shape[0])
            md.append(f"| {tg:.2f} | {sid} {SYS[sid][0]} | {fm(m['precision'])} | {m['recall']:.3f} | {m['auto_cuts']} | {m['false_cuts']} | {m['retakes_in_review']} | {m['retakes_in_unlikely']} | {pct(m['ambiguous_share'])} | {m['review_queue']} | {un} |")
    md += ["", "## Per fold: precision / recall (auto-cuts, false cuts)", ""]
    for tg in TARGETS:
        md += [f"### Target {tg:.2f}", "", "| system | " + " | ".join(f"fold{k}" for k in FOLDS) + " |", "|---|" + "---|" * len(FOLDS)]
        for sid in SYS:
            cells = []
            for k in FOLDS:
                m = M[(sid, tg, int(k))]
                unr = bool(thr_df[(thr_df.system == sid) & (thr_df.target == tg) & (thr_df.fold == int(k))].unreachable.iloc[0])
                cells.append("unreachable (no auto-cuts)" if unr else f"{fm(m['precision'])} / {m['recall']:.3f} ({m['auto_cuts']}, {m['false_cuts']})")
            md.append(f"| {sid} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## Thresholds chosen on the other folds (high; 'inf' = unreachable)", "", "| system | target | " + " | ".join(f"fold{k} high / low" for k in FOLDS) + " |", "|---|---|" + "---|" * len(FOLDS)]
    for sid in SYS:
        for tg in TARGETS:
            g = thr_df[(thr_df.system == sid) & (thr_df.target == tg)].sort_values("fold")
            md.append(f"| {sid} | {tg:.2f} | " + " | ".join(f"{('inf' if np.isinf(r.high) else format(r.high, '.4f'))} / {r.low:.4f}" for r in g.itertuples()) + " |")
    md += ["", "## Paired video-level bootstrap vs S1 (1000 resamples, seed 0; difference, 95% CI; * = CI excludes 0)", "",
           "| target | comparison | recall | review-queue size (pairs) | achieved precision (supplementary) | false cuts (supplementary) |", "|---|---|---|---|---|---|"]
    for tg in TARGETS:
        for a in ("S3", "S2"):
            pa, pb = M[(a, tg, "pooled")], M[("S1", tg, "pooled")]
            cells = []
            for k, key in (("recall", "recall"), ("review", "review_queue"), ("prec", "precision"), ("fc", "false_cuts")):
                bk = {"recall": "recall", "review": "review_queue", "prec": "precision", "fc": "false_cuts"}[k]
                lo, hi = BD[(a, tg)][bk]
                point = pa[key] - pb[key]
                cells.append(f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if lo > 0 or hi < 0 else ''}")
            md.append(f"| {tg:.2f} | {a} minus S1 | " + " | ".join(cells) + " |")
    md += ["", "## Fold 3 only: recall of S3 minus S1 (and S2 minus S1) at each target", "", "| target | S3 - S1 point | CI over the 3 fold-3 videos | S2 - S1 point | CI |", "|---|---|---|---|---|"]
    for tg in TARGETS:
        md.append(f"| {tg:.2f} | {M[('S3', tg, 3)]['recall'] - M[('S1', tg, 3)]['recall']:+.3f} | [{F3[('S3', tg)]['recall'][0]:+.3f}, {F3[('S3', tg)]['recall'][1]:+.3f}] | "
                  f"{M[('S2', tg, 3)]['recall'] - M[('S1', tg, 3)]['recall']:+.3f} | [{F3[('S2', tg)]['recall'][0]:+.3f}, {F3[('S2', tg)]['recall'][1]:+.3f}] |")
    md += ["", "(A bootstrap over only 3 videos is very coarse; treat these intervals as a rough guide.)", "",
           "## Plot", "", "![Pooled out-of-fold precision-recall curves](pr_curves_matched.png)", "",
           "Curves: S1, S2, S3 on the pooled out-of-fold scores; markers: the pooled nested operating point of each system at each target (circle 0.90, square 0.95, triangle 0.98, diamond 0.99). A marker well below the dotted target line means the threshold chosen on the other folds did not transfer.", ""]
    (OUT / "matched.md").write_text("\n".join(md), encoding="utf-8")

    sc = df[["pair_id", "video_id", "fold", "label"] + [c for _, c in SYS.values()]].copy()
    for (sid, tg), t in tiers.items():
        sc[f"tier_{sid}_{tg:.2f}"] = t
    sc.to_csv(OUT / "matched_scores.csv", index=False)
    STATUS.write_text("done\n")
    print("\n".join(md[6:13]))
except Exception as e:                       # report instead of failing silently
    import traceback
    STATUS.write_text("failed: " + traceback.format_exc()[-500:] + "\n")
    raise
