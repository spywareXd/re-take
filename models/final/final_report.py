"""Assemble results_v2/final/final.md from the saved results (fit_results.json, fair_comparison.json, operating_points.csv, stage2_consistency.json,
ce_logit_shift.json, thresholds.json) and a live pytest run. No fitting here."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "data" / "training" / "results_v2" / "final"
fit = json.loads((OUT / "fit_results.json").read_text())
fair = json.loads((OUT / "fair_comparison.json").read_text())
cons = json.loads((OUT / "stage2_consistency.json").read_text())
shift = json.loads((OUT / "ce_logit_shift.json").read_text())
art = json.loads((HERE / "thresholds.json").read_text())
op = pd.read_csv(OUT / "operating_points.csv")
t = subprocess.run([sys.executable, "-m", "pytest", str(HERE / "test_final_scorer.py"), "-q"], capture_output=True, text=True, cwd=REPO)
test_line = [l for l in t.stdout.strip().splitlines() if "passed" in l or "failed" in l or "error" in l.lower()][-1]

R, FR, BC = fit["results"], fair["results"], fair["bootstrap"]
st = art["stacker"]
nested, old_s2, fin = R["nested"]["pooled"], R["stage2"]["pooled"], R["final_thr_on_oof"]["pooled"]
tuned, ce_t = FR["stage2_tuned_same_procedure"]["pooled"], FR["cross_encoder_alone_tuned_same_procedure"]["pooled"]
B_t = BC["stacker_minus_stage2_tuned_same_procedure"]
pct = lambda x: f"{100 * x:.1f}%"
cc = cons["summary"]
c_train = [v for k, v in cc.items() if k.startswith("training")][0]
c_prod = [v for k, v in cc.items() if k.startswith("production")][0]
n_ret = art["retakes"]

md = ["# Final scorer: stacker + two thresholds", "",
      f"Built {art['created']} from labels `{art['label_file']['name']}` (sha256 {art['label_file']['sha256_12']}) and folds `{art['folds_file']['name']}` "
      f"(sha256 {art['folds_file']['sha256_12']}): {art['training_rows']} labelled rows, {art['retakes']} retakes. No LLM or API calls. Code and artifacts: models/final/.", "",
      "## Summary (corrected after review)", "",
      f"1. **Stacker** = StandardScaler + L2 logistic regression (C = {st['C']}) on [Stage 2 score, logit(cross-encoder prob)], fitted on all {st['fit_rows']} out-of-fold rows; thresholds from out-of-fold stacker "
      f"probabilities: high = {art['thresholds']['high']:.4f}, low = {art['thresholds']['low']:.4f}.",
      f"2. **The earlier '84% -> 10.5% ambiguous band' claim is withdrawn as a stacker effect.** Stage 2's old 0.35/0.65 cutoffs were never tuned. Stage 2 with thresholds chosen by the SAME nested "
      f"procedure (95% precision / 98% recall) already has a {pct(tuned['ambiguous_share'])} band, against the stacker's {pct(nested['ambiguous_share'])} "
      f"(difference {B_t['ambiguous_share']['diff'] * 100:+.1f} points, 95% CI [{B_t['ambiguous_share']['lo'] * 100:+.1f}, {B_t['ambiguous_share']['hi'] * 100:+.1f}]): the reduction is almost entirely threshold tuning.",
      f"3. **What the stacker actually changes (vs tuned Stage 2):** retake-tier precision {nested['retake_precision']:.3f} vs {tuned['retake_precision']:.3f} ({B_t['retake_precision']['diff'] * 100:+.1f} points, CI excludes 0), "
      f"recall {nested['retake_recall']:.3f} vs {tuned['retake_recall']:.3f} ({B_t['retake_recall']['diff'] * 100:+.1f} points, CI [{B_t['retake_recall']['lo'] * 100:+.1f}, {B_t['retake_recall']['hi'] * 100:+.1f}] includes 0). "
      f"It trades precision for recall; the 0.95 precision target is met only in-sample ({fin['retake_precision']:.3f}). {pct(nested['retakes_reach_cut_or_review'] if 'retakes_reach_cut_or_review' in nested else 1 - nested['unlikely_retakes'] / n_ret)} of retakes "
      f"({n_ret - nested['unlikely_retakes']} of {n_ret}) reach an automatic cut or a review; {nested['unlikely_retakes']} ({pct(nested['unlikely_retakes'] / n_ret)}) land in 'unlikely'. Ambiguous pairs that nobody reviews stay uncut by default.",
      f"4. **Weak speakers:** nested retake-tier precision is {R['nested']['fold3']['retake_precision']:.2f} in fold 3 ({R['nested']['fold3']['retake_nonretakes']} of {R['nested']['fold3']['retake_n']} auto-cuts would be false cuts) and "
      f"recall is {R['nested']['fold4']['retake_recall']:.2f} in fold 4. The operating-point table below is for choosing a conservative default; I have NOT changed the thresholds.",
      f"5. **Stage 2 consistency (production risk):** the pipeline's own Stage 2 score is recency-adjusted and was lower than the training score for every compared pair (mean abs gap {c_train['pipeline_vs_csv_mean_abs']:.3f}, max {c_train['pipeline_vs_csv_max_abs']:.3f}). "
      f"The scorer now builds the plain score from lexical/semantic/containment and ignores (with a warning) a supplied pipeline `stage2_score`; tests enforce it. Tests: {test_line}.", ""]

# ---------- fair comparison ----------
md += ["## 1. Fair baseline: same threshold procedure for every system (nested over folds, pooled)", "",
       "Thresholds for each held-out fold are chosen from the other folds only: high = lowest t with precision >= 0.95, low = highest t with recall >= 0.98. A raw score (Stage 2, cross-encoder) needs no model fit; "
       "the stacker's thresholds come from an inner out-of-fold pass.", "",
       "| system | ambiguous share | retake tier n | retake P / R / F1 | retakes in unlikely | retakes reaching cut-or-review | non-retakes auto-cut |", "|---|---|---|---|---|---|---|"]
labels = {"stage2_old_rules_0.65_0.35": "Stage 2, old rules 0.65 / 0.35 (never tuned)", "stage2_tuned_same_procedure": "Stage 2, thresholds tuned by the same procedure",
          "cross_encoder_alone_tuned_same_procedure": "Cross-encoder alone, tuned by the same procedure", "stacker_tuned_same_procedure": "**Stacker**, tuned by the same procedure"}
for k, name in labels.items():
    r = FR[k]["pooled"]
    md.append(f"| {name} | {pct(r['ambiguous_share'])} | {r['retake_n']} | {r['retake_precision']:.3f} / {r['retake_recall']:.3f} / {r['retake_f1']:.3f} | {r['unlikely_retakes']} | {pct(r['retakes_reach_cut_or_review'])} | {r['retake_nonretakes']} |")
md += ["", f"Tuned Stage 2 thresholds per held-out fold (low, high): " + "; ".join(f"fold{k}: ({lo:.3f}, {hi:.3f})" for k, (lo, hi) in fair["nested_thresholds"]["stage2"].items())
       + f". On all rows: low {fair['final_thresholds']['stage2'][0]:.3f}, high {fair['final_thresholds']['stage2'][1]:.3f}: the old 0.35 floor was far too low, which is why almost everything sat in the ambiguous band.", "",
       "Paired video-level bootstrap (1000 resamples, seed 0), stacker minus each comparison system:", "", "| statistic | vs old rules | vs tuned Stage 2 | vs tuned cross-encoder alone |", "|---|---|---|---|"]
for k in ("ambiguous_share", "retake_precision", "retake_recall", "unlikely_recall", "retakes_reach_cut_or_review"):
    cells = []
    for other in ("stage2_old_rules_0.65_0.35", "stage2_tuned_same_procedure", "cross_encoder_alone_tuned_same_procedure"):
        v = BC[f"stacker_minus_{other}"][k]
        cells.append(f"{v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]{' *' if v['lo'] > 0 or v['hi'] < 0 else ''}")
    md.append(f"| {k} | " + " | ".join(cells) + " |")
md += ["", "`*` = 95% CI excludes 0. Reading it: the stacker clearly beats the cross-encoder alone on recall and band size, but against a properly tuned Stage 2 it is not better on band size and is worse on precision with a (non-significant) recall gain.", ""]

# ---------- operating points ----------
md += ["## 2. Operating points: choose the high threshold deliberately", "",
       f"Stacker out-of-fold probabilities, low threshold fixed at {art['thresholds']['low']:.4f} (so 'unlikely' always holds the same {int(round((1 - op.retakes_reach_cut_or_review.iloc[0]) * n_ret))} retakes = {pct(1 - op.retakes_reach_cut_or_review.iloc[0])}). "
       "The sweep is descriptive: each global threshold is applied in-sample to out-of-fold probabilities, so precision is optimistic.", "",
       "| high | ambiguous share | auto-cut pairs | precision | recall | retakes left to review | retakes reaching cut-or-review | fold 3 false cuts | fold 4 recall |", "|---|---|---|---|---|---|---|---|---|"]
for r in op.itertuples():
    md.append(f"| {r.high:.3f}{' (final)' if r.is_final else ''} | {pct(r.ambiguous_share)} | {r.retake_n} | {r.retake_precision:.3f} | {r.retake_recall:.3f} | {r.retakes_in_ambiguous} | {pct(r.retakes_reach_cut_or_review)} | {r.fold3_false_cuts} | {r.fold4_recall:.2f} |")
cons_row = op[op.high == 0.97].iloc[0] if (op.high == 0.97).any() else None
md += ["", "Moving the high threshold up buys precision and fewer false cuts in the weak folds, at the cost of a bigger review queue and lower automatic recall (a fold's recall can fall a lot, e.g. fold 4). "
       + (f"A conservative mode at high = 0.97 would give precision {cons_row.retake_precision:.3f}, recall {cons_row.retake_recall:.3f}, {int(cons_row.fold3_false_cuts)} false cuts in fold 3 and "
          f"{cons_row.retakes_in_ambiguous} retakes in the review queue ({pct(cons_row.ambiguous_share)} ambiguous). This is a proposal for you to choose from, not a setting I applied." if cons_row is not None else ""), ""]

# ---------- nested per fold (honest) ----------
def tier_table(name, title, source=R):
    rows = [f"### {title}", "", "| scope | n | retake tier n (retakes / non-retakes) | retake P / R / F1 | ambiguous n (retakes / non-retakes) | ambiguous share | unlikely n (retakes / non-retakes) | unlikely NPV / recall(neg) |", "|---|---|---|---|---|---|---|---|"]
    for scope, r in source[name].items():
        rows.append(f"| {scope} | {r['n']} | {r['retake_n']} ({r['retake_retakes']} / {r['retake_nonretakes']}) | {r['retake_precision']:.3f} / {r['retake_recall']:.3f} / {r['retake_f1']:.3f} | "
                    f"{r['ambiguous_n']} ({r['ambiguous_retakes']} / {r['ambiguous_nonretakes']}) | {pct(r['ambiguous_share'])} | {r['unlikely_n']} ({r['unlikely_retakes']} / {r['unlikely_nonretakes']}) | {r['unlikely_precision']:.3f} / {r['unlikely_recall']:.3f} |")
    return rows + [""]


md += ["## 3. Per-fold honest performance (nested)", ""]
md += tier_table("nested", "Stacker, nested folds", R)
md += tier_table("stage2_tuned_same_procedure", "Stage 2 tuned by the same procedure, nested folds", FR)
md += ["Stacker thresholds per outer fold (low, high): " + "; ".join(f"fold{k}: ({lo:.3f}, {hi:.3f})" for k, (lo, hi) in fit["nested_thresholds"].items()) + ".", ""]

# ---------- stage 2 consistency ----------
md += ["## 4. Stage 2 consistency between training and the pipeline", "",
       f"The real pipeline code (cached transcript -> units -> candidates -> `run_stage2`) was run on `{cons['video']}` and compared with the plain-formula Stage 2 score stored with the labelled pairs "
       "(`pairs_v2.csv`: 0.5*semantic + 0.3*lexical_score + 0.2*containment).", "",
       "| pipeline config | labelled pairs compared | mean abs gap | max abs gap | share gap > 0.01 | share gap > 0.05 | pipeline higher | tier flips if the pipeline value were used | plain formula from the pipeline's own fields vs training: max gap |", "|---|---|---|---|---|---|---|---|---|"]
for tag, c in cc.items():
    md.append(f"| {tag} | {c['pairs_compared']} | {c['pipeline_vs_csv_mean_abs']:.3f} | {c['pipeline_vs_csv_max_abs']:.3f} | {pct(c['share_abs_gt_0.01'])} | {pct(c['share_abs_gt_0.05'])} | {pct(c['pipeline_higher_share'])} | "
              f"{c['tier_flips_if_pipeline_score_used']} ({pct(c['tier_flip_share'])}) | {c['bridge_plain_vs_csv_max_abs']:.4f} |")
md += ["", "- The pipeline's `stage2_score` uses `combined_score` (lexical x recency weight, +0.05 with a restart phrase), so it is lower than the training score whenever the two lines are far apart in time. It was lower for every compared pair here.",
       "- The gap rarely flips a tier on this video (1 pair), but it shifts every score and would shift the thresholds' meaning. The plain formula rebuilt from the pipeline's own candidate fields matches the training score within the CSV rounding (< 0.0005).",
       "- **Integration rule:** feed the scorer `lexical_score`, `semantic_score` and `containment` (or the raw texts); never the pipeline's `stage2_score`. The scorer enforces this and logs a warning if a conflicting `stage2_score` is supplied. Two tests enforce it against `pairs_v2.csv` (text path, and real pipeline candidates).",
       "- One video was compared (21 to 69 labelled pairs); the integration should repeat this check on a second video before release.", ""]

# ---------- final CE ----------
md += ["## 5. Final cross-encoder (trained on all labelled v2 rows)", "",
       f"- Same hyperparameters as the out-of-fold runs: {json.dumps(art['models']['cross_encoder_hyperparameters'])}; seeds 0, 1, 2; weights in models/final/ce_seed0, ce_seed1, ce_seed2 (88 MB each, now git-ignored).",
       f"- Seed-averaging rule: {art['models']['seed_averaging']}.", "",
       "| quantity | out-of-fold (what the stacker was fitted on) | final model on its own training rows |", "|---|---|---|",
       f"| logit mean (std) | {shift['oof_logit']['mean']:.2f} ({shift['oof_logit']['std']:.2f}) | {shift['final_model_logit_on_training_rows']['mean']:.2f} ({shift['final_model_logit_on_training_rows']['std']:.2f}) |",
       f"| retakes: mean (std) | {shift['by_class']['1']['oof']['mean']:.2f} ({shift['by_class']['1']['oof']['std']:.2f}) | {shift['by_class']['1']['final']['mean']:.2f} ({shift['by_class']['1']['final']['std']:.2f}) |",
       f"| non-retakes: mean (std) | {shift['by_class']['0']['oof']['mean']:.2f} ({shift['by_class']['0']['oof']['std']:.2f}) | {shift['by_class']['0']['final']['mean']:.2f} ({shift['by_class']['0']['final']['std']:.2f}) |",
       f"| share with abs(logit) > 8 | {pct(shift['fraction_abs_logit_gt_8']['oof'])} | {pct(shift['fraction_abs_logit_gt_8']['final'])} |", "",
       "The final model has seen those rows, so its logits there are more extreme; on NEW pairs it should behave like the out-of-fold models. Do not re-score the training pairs against the thresholds.", ""]

md += ["## 6. Packaging, tests and housekeeping", "",
       "- `models/final/thresholds.json`: stacker coefficients, scaler, C, thresholds, fallback rules, model names, feature version, label/folds checksums, embedding model, date.",
       "- `models/final/final_scorer.py`: `score_pairs`, `classify`, `FinalScorer`, `to_spans`. Missing files -> WARNING and Stage 2 with the old 0.65 / 0.35 rules. Spans carry `status` and `keeper_unit_id = None`; the keeper / LLM step is untouched.",
       f"- Tests (`pytest models/final/test_final_scorer.py`): fallback, boundaries, golden labelled pairs, reproduction of saved out-of-fold values, mock path, spans, Stage 2 text-path and real-pipeline consistency, CPU real-model smoke test. Result: {test_line}.",
       "- `.gitignore` now excludes `models/final/ce_seed*/`. For a public release publish the weights on Hugging Face Hub or a GitHub release with a model card, and check the base model's (cross-encoder/ms-marco-MiniLM-L-6-v2) license and what you may say about the training data.", "",
       "## Caveats", "",
       "- The cross-encoder out-of-fold scores of the training folds came from models that saw the held-out fold's labels, so the stacker and the nested estimates are mildly optimistic (as in combine.md).",
       "- Speaker-level generalisation is the main uncertainty: precision and recall vary widely by fold (section 3), and there are only 5 speakers. A drift warning (compare a new video's score distribution with the training one) is a sensible addition before auto-cutting; it is not implemented.",
       "- The 12 corrected labels are all in fold 3; everything here uses pairs_v2.csv (label 0/1 only). Nothing was committed.", ""]
(OUT / "final.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md[:16]))
