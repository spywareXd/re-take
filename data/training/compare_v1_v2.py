"""Build results_v2/v1_vs_v2.md: every headline number on the v1 labels (results/) next to the v2 labels (results_v2/).
Reads only; writes results_v2/v1_vs_v2.md. Metrics are recomputed from the per-row score files with one shared function,
so v1 and v2 are treated identically. No model and no API call."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

HERE = Path(__file__).resolve().parent
R1, R2 = HERE / "results", HERE / "results_v2"
N_BOOT, SEED = 1000, 0
rd = lambda p: pd.read_csv(p, dtype={"pair_id": str, "video_id": str})


def rec_at_p(y, s, p=0.90):
    pr, rc, _ = precision_recall_curve(y, s)
    ok = pr >= p
    return float(rc[ok].max()) if ok.any() else 0.0


def load(root):
    """model -> DataFrame(pair_id, video_id, fold, label, score)"""
    b = rd(root / "baseline_scores.csv")
    out = {"stage2": b.rename(columns={"current_stage2": "score"}), "lexical_only": b.rename(columns={"lexical_only": "score"}),
           "semantic_only": b.rename(columns={"semantic_only": "score"})}
    for k in ("stage2", "lexical_only", "semantic_only"):
        out[k] = out[k][["pair_id", "video_id", "fold", "label", "score"]]
    lg = rd(root / "logreg_scores.csv"); out["logreg_7feat"] = lg.rename(columns={"prob": "score"})[["pair_id", "video_id", "fold", "label", "score"]]
    gb = rd(root / "gbm_scores.csv"); out["gbm"] = gb.rename(columns={"prob": "score"})[["pair_id", "video_id", "fold", "label", "score"]]
    ce = rd(root / "ce_scores.csv").drop_duplicates("pair_id"); out["cross_encoder"] = ce.rename(columns={"prob_mean": "score"})[["pair_id", "video_id", "fold", "label", "score"]]
    cb = rd(root / "combine_scores.csv")
    for src, name in (("score_stack", "stacker"), ("score_rank_avg", "rank_avg")):
        out[name] = cb.rename(columns={src: "score"})[["pair_id", "video_id", "fold", "label", "score"]]
    return out


def boot_ci(dfs, base="stage2"):
    """pooled PR-AUC CI per model and paired diff vs `base`, video-level, shared resamples"""
    ref = dfs[base]
    vids = ref.video_id.unique()
    rng = np.random.default_rng(SEED)
    idx = {m: {v: np.where(d.video_id.values == v)[0] for v in vids} for m, d in dfs.items()}
    res = {m: [] for m in dfs}
    for _ in range(N_BOOT):
        pick = rng.choice(vids, len(vids), replace=True)
        for m, d in dfs.items():
            ii = np.concatenate([idx[m][v] for v in pick])
            y = d.label.values[ii]
            res[m].append(average_precision_score(y, d.score.values[ii]) if y.sum() else np.nan)
    out = {}
    for m in dfs:
        lo, hi = np.nanpercentile(res[m], [2.5, 97.5])
        dd = np.array(res[m]) - np.array(res[base])
        dlo, dhi = np.nanpercentile(dd, [2.5, 97.5])
        out[m] = (lo, hi, dlo, dhi)
    return out


def headline(dfs):
    rows = {}
    ci = boot_ci(dfs)
    for m, d in dfs.items():
        y, s = d.label.values, d.score.values
        pooled = average_precision_score(y, s)
        folds = {k: average_precision_score(g.label, g.score) for k, g in d.groupby("fold")}
        r90 = {k: rec_at_p(g.label.values, g.score.values) for k, g in d.groupby("fold")}
        diff = pooled - average_precision_score(dfs["stage2"].label, dfs["stage2"].score)
        rows[m] = {"n": len(d), "pos": int(y.sum()), "pooled": pooled, "ci": ci[m][:2], "diff": diff, "diff_ci": ci[m][2:], "folds": folds,
                   "r90": rec_at_p(y, s), "r90_f3": r90.get(3, np.nan), "pr_f3": folds.get(3, np.nan)}
    return rows


v1, v2 = load(R1), load(R2)
H1, H2 = headline(v1), headline(v2)


def f(x, n=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


md = ["# Results on the v1 labels vs the corrected v2 labels", "",
      "v1 = results/ (labels at scoring time, pairs_v1_original_labels.csv); v2 = results_v2/ (pairs_v2.csv). Same folds, seeds, hyperparameters, threshold-selection "
      "procedures and pre-registered comparisons; the cross-encoder was retrained (3 seeds x 4 folds) and everything else recomputed. LLM verdicts are the cached ones (no new "
      "API calls). Headline metrics here are recomputed from the per-row score files with one shared function (video-level bootstrap, 1000 resamples, seed 0). "
      "Caveat: the per-script reports in results_v2/ still print some fixed captions from v1 (e.g. '3570 rows', '79 Stage 2 errors', '861 pairs'); the numbers in their tables are v2.", ""]

# ---------------- what changed ----------------
ch = pd.read_csv(HERE / "label_changes.csv", dtype=str)
old_err = pd.read_csv(R1 / "stage2_errors.csv", dtype=str)
new_err = pd.read_csv(R2 / "stage2_errors.csv", dtype=str)
fixed_err = old_err[old_err.pair_id.isin(ch.pair_id)]
# caption values computed from the data (were hard-coded)
fold_txt = "/".join(sorted(ch.fold.unique()))
vid_txt = ", ".join(f"{v} {n}" for v, n in ch.video_id.value_counts().sort_index().items())
dir_txt = " and ".join(f"{n} went {k}" for k, n in (ch.old_label + "->" + ch.new_label).value_counts().sort_index().items())
f3_fn, f3_fp = int(((old_err.fold == "3") & (old_err.error == "FN")).sum()), int(((old_err.fold == "3") & (old_err.error == "FP")).sum())
unsure = ch[ch.new_label == "-1"]
n_band_unsure = int(((unsure.stage2_score.astype(float) >= 0.45) & (unsure.stage2_score.astype(float) < 0.65)).sum())
n_set_unsure = int((unsure.stage2_score.astype(float) >= 0.65).sum())
vs = {}
for tag, root in (("v1", R1), ("v2", R2)):
    bb = rd(root / "baseline_scores.csv"); bb = bb[bb.current_stage2 >= 0.65]
    vs[tag] = (len(bb), int(bb.label.sum()), int((bb.label == 0).sum()))
import json as _json
_fs = _json.loads((R2 / "fewshot_examples.json").read_text())
_changed_examples = sorted({e["pair_id"] for v in _fs.values() for e in v if e["pair_id"] in set(ch.pair_id)})
md += ["## The label change", "",
       f"- {len(ch)} labels changed, all in fold {fold_txt} ({vid_txt}): {dir_txt}. The row count with a 0/1 label fell from {len(v1['stage2'])} to {len(v2['stage2'])}; retakes from {H1['stage2']['pos']} to {H2['stage2']['pos']}.",
       f"- **Old Stage 2 errors that were label fixes:** {len(fixed_err)} of the {len(old_err)} old errors ({int((fixed_err.error == 'FN').sum())} FN, {int((fixed_err.error == 'FP').sum())} FP) "
       f"were pairs whose label you changed, so they are no longer errors. The corrected set has {len(new_err)} Stage 2 errors ({int((new_err.error == 'FN').sum())} FN, {int((new_err.error == 'FP').sum())} FP); "
       f"all new errors were old errors: {set(new_err.pair_id) <= set(old_err.pair_id)}.",
       f"- Of the old errors not changed, fold 3 had {int((old_err.fold == '3').sum())} ({f3_fn} FN, {f3_fp} FP); now {int((new_err.fold == '3').sum())}.",
       f"- The {len(ch)} changed pairs by direction and by model correctness are in label_changes.md. {n_band_unsure} band pairs (Stage 2 in [0.45, 0.65)) and {n_set_unsure} pairs in the veto set (Stage 2 >= 0.65) became -1 and left those evaluations.",
       f"- **Review coverage:** the label review covered all folds and found changes only in fold {fold_txt} ({vid_txt}), per the author's review. The few-shot LLM variants (llama3 and Gemini) were not rerun because a fold-1 few-shot example ({', '.join(_changed_examples)}) changed label.", ""]

# ---------------- headline ranking ----------------
order = ["stage2", "lexical_only", "semantic_only", "logreg_7feat", "gbm", "cross_encoder", "stacker", "rank_avg"]
md += ["## Pooled PR-AUC (95% video-bootstrap CI) and the paired difference vs Stage 2", "",
       "| model | v1 pooled [CI] | v2 pooled [CI] | v1 diff vs Stage 2 [CI] | v2 diff vs Stage 2 [CI] | CI excludes 0 (v1 / v2) |", "|---|---|---|---|---|---|"]
changed = []
for m in order:
    a, b = H1[m], H2[m]
    e1 = a["diff_ci"][0] > 0 or a["diff_ci"][1] < 0
    e2 = b["diff_ci"][0] > 0 or b["diff_ci"][1] < 0
    if m != "stage2" and e1 != e2:
        changed.append(f"{m}: paired CI vs Stage 2 {'excluded' if e1 else 'included'} 0 on v1 and {'excludes' if e2 else 'includes'} 0 on v2")
    md.append(f"| {m} | {f(a['pooled'])} [{f(a['ci'][0])}, {f(a['ci'][1])}] | {f(b['pooled'])} [{f(b['ci'][0])}, {f(b['ci'][1])}] | "
              + ("-" if m == "stage2" else f"{a['diff']:+.3f} [{a['diff_ci'][0]:+.3f}, {a['diff_ci'][1]:+.3f}]") + " | "
              + ("-" if m == "stage2" else f"{b['diff']:+.3f} [{b['diff_ci'][0]:+.3f}, {b['diff_ci'][1]:+.3f}]") + f" | {'-' if m == 'stage2' else str(e1) + ' / ' + str(e2)} |")
r1 = sorted(order, key=lambda m: -H1[m]["pooled"]); r2 = sorted(order, key=lambda m: -H2[m]["pooled"])
md += ["", f"Ranking by pooled PR-AUC, v1: {' > '.join(r1)}", "", f"Ranking by pooled PR-AUC, v2: {' > '.join(r2)}", ""]

md += ["## Per fold PR-AUC (v1 -> v2) and recall at 90% precision", "",
       "| model | fold1 | fold2 | **fold3** | fold4 | pooled recall@P90 v1 -> v2 | fold3 recall@P90 v1 -> v2 |", "|---|---|---|---|---|---|---|"]
for m in order:
    a, b = H1[m], H2[m]
    md.append(f"| {m} | " + " | ".join((f"**{f(a['folds'][k])} -> {f(b['folds'][k])}**" if k == 3 else f"{f(a['folds'][k])} -> {f(b['folds'][k])}") for k in (1, 2, 3, 4))
              + f" | {f(a['r90'])} -> {f(b['r90'])} | {f(a['r90_f3'])} -> {f(b['r90_f3'])} |")
md.append("")

# ---------------- operating points: Stage 2 @0.65 and combine decisions ----------------
def prf(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0), tp, fp, fn


md += ["## F1 at the operating point (Stage 2 at 0.65; stackers at the training-fold-chosen threshold)", "",
       "| system | v1 P / R / F1 | v2 P / R / F1 | v1 FN / FP | v2 FN / FP |", "|---|---|---|---|---|"]
for name, col in (("stage2 @0.65", None), ("stacker", "pred_stack"), ("rank_avg", "pred_rank_avg")):
    cells = []
    for root in (R1, R2):
        if col is None:
            d = rd(root / "baseline_scores.csv"); pred = (d.current_stage2 >= 0.65).astype(int); y = d.label
        else:
            d = rd(root / "combine_scores.csv"); pred = d[col].astype(int); y = d.label
        cells.append(prf(y, pred))
    md.append(f"| {name} | {cells[0][0]:.3f} / {cells[0][1]:.3f} / {cells[0][2]:.3f} | {cells[1][0]:.3f} / {cells[1][1]:.3f} / {cells[1][2]:.3f} | {cells[0][5]} / {cells[0][4]} | {cells[1][5]} / {cells[1][4]} |")
md.append("")

# ---------------- band adjudication ----------------
def band_f1(path_scores, cols, llm_prefix_cols):
    d = pd.read_csv(path_scores, dtype={"pair_id": str})
    d = d[d.band_flag.astype(str) == "True"]
    out = {}
    for name, col in cols.items():
        v = d[col].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, 1: 1.0, 0: 0.0}) if d[col].dtype == object else d[col]
        v = pd.to_numeric(v, errors="coerce")
        m = v.notna()
        pr, rc, f1, tp, fp, fn = prf(d.label[m], v[m])
        out[name] = (pr, rc, f1, tp, fp, fn, int(m.sum()))
    return out, len(d), int(d.label.sum())


md += ["## Ambiguous band [0.45, 0.65): precision / recall / F1 (pooled)", ""]
sets = [("llama3 8B (local)", R1 / "adjudication_scores.csv", R2 / "adjudication_scores_llama3.csv", {"cross_encoder": "ce_decision", "llm_zero": "llm_zero", "llm_few": "llm_few"}),
        ("Gemini flash-lite", R1 / "adjudication_scores_gemini_flashlite.csv", R2 / "adjudication_scores_gemini_flashlite.csv", {"cross_encoder": "ce_decision", "llm_zero": "llm_zero", "llm_few": "llm_few"})]
for title, p1, p2, cols in sets:
    c1, n1, pos1 = band_f1(p1, cols, None)
    c2cols = {k: v for k, v in cols.items() if k != "llm_few"}
    c2, n2, pos2 = band_f1(p2, c2cols, None)
    md += [f"**{title}** (band: v1 {n1} pairs / {pos1} retakes; v2 {n2} pairs / {pos2} retakes)", "",
           "| system | v1 P / R / F1 (TP FP FN) | v2 P / R / F1 (TP FP FN) |", "|---|---|---|"]
    for k in cols:
        a = c1[k]
        b = c2.get(k)
        md.append(f"| {k} | {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f} ({a[3]} {a[4]} {a[5]}) | "
                  + (f"{b[0]:.3f} / {b[1]:.3f} / {b[2]:.3f} ({b[3]} {b[4]} {b[5]})" if b else "**not rerun** (fold-1 few-shot example label changed)") + " |")
    md.append("")


def parse_rows(path, header_contains, first_cell_prefix=None):
    txt = path.read_text(encoding="utf-8").splitlines()
    out, on = [], False
    for line in txt:
        if line.startswith("###"):
            on = header_contains in line
            continue
        if on and line.startswith("|") and not line.startswith("|---") and not line.startswith("| system") and not line.startswith("| rule"):
            out.append([c.strip() for c in line.strip("|").split("|")])
    return out


md += ["### Band adjudication, end to end (F1 minus Stage 2 alone, paired video bootstrap; cached LLM verdicts)", "",
       "| backend | system | v1 F1 diff [CI] | v2 F1 diff [CI] |", "|---|---|---|---|"]
for title, p1, p2 in (("llama3", R1 / "adjudication.md", R2 / "adjudication_llama3.md"), ("Gemini flash-lite", R1 / "adjudication_gemini_flashlite.md", R2 / "adjudication_gemini_flashlite.md")):
    a = {r[0]: r for r in parse_rows(p1, "Paired video-level bootstrap")}
    b = {r[0]: r for r in parse_rows(p2, "Paired video-level bootstrap")}
    for k in a:
        md.append(f"| {title} | {k} | {a[k][1]} {a[k][2]} (excl. 0: {a[k][3]}) | " + (f"{b[k][1]} {b[k][2]} (excl. 0: {b[k][3]})" if k in b else "not rerun") + " |")
md.append("")

# ---------------- veto ----------------
md += ["## Veto test (V0-V3; pairs with Stage 2 >= 0.65)", "",
       f"v1 set: {vs['v1'][0]} pairs ({vs['v1'][1]} retakes, {vs['v1'][2]} non-retakes). v2 set: {vs['v2'][0]} pairs ({vs['v2'][1]} retakes, {vs['v2'][2]} non-retakes): {vs['v1'][1] - vs['v2'][1]} retakes became unsure and left the set.", "",
       "| rule | v1 FPs removed / retakes dropped | v1 F1 | v2 FPs removed / retakes dropped | v2 F1 |", "|---|---|---|---|---|"]
a = {r[0]: r for r in parse_rows(R1 / "veto" / "veto.md", "Per rule")}
b = {r[0]: r for r in parse_rows(R2 / "veto" / "veto.md", "Per rule")}
for k in ("V0", "V1", "V2", "V3"):
    md.append(f"| {k} | {a[k][1]} / {a[k][2]} | {a[k][-1]} | {b[k][1]} / {b[k][2]} | {b[k][-1]} |")
md += ["", "Paired bootstrap vs V0 (F1 diff [95% CI]; recall change; meets the pre-registered criterion = CI excludes 0 in the positive direction AND recall drop <= 0.010):", "",
       "| rule | v1 F1 diff [CI] | v1 recall change | v2 F1 diff [CI] | v2 recall change |", "|---|---|---|---|---|"]
a = {r[0]: r for r in parse_rows(R1 / "veto" / "veto.md", "Paired video-level bootstrap")}
b = {r[0]: r for r in parse_rows(R2 / "veto" / "veto.md", "Paired video-level bootstrap")}
for k in ("V1", "V2", "V3"):
    md.append(f"| {k} | {a[k][1]} | {a[k][3]} | {b[k][1]} | {b[k][3]} |")
md.append("")

# ---------------- flags ----------------
md += ["## Items to flag", "",
       "- **Few-shot not rerun:** the fold-1 few-shot example `video_07__21-1__23-0` was shown to the LLM as a retake (label 1) and is now -1, so the few-shot variant (llama3 and Gemini) was NOT rerun; v1 few-shot numbers stand as measured on the v1 labels.",
       "- **Cached verdicts:** all zero-shot LLM answers were reused; 0 new API calls. Fold-3 pairs whose label changed keep their old model answers (the models never saw labels).",
       "- **Per-script reports in results_v2/** keep some fixed v1 captions (see the note at the top); use this file for headline numbers.", ""]
md += ["## Conclusions that changed (rules applied to the numbers above)", ""] + ([f"- {c}" for c in changed] or ["- No paired-CI-vs-Stage-2 verdict changed."])
(R2 / "v1_vs_v2.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
