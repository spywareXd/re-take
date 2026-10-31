"""PART B: Stage 2 parity check on two more labelled videos (video_07 = fold 3, video_02 = another speaker), as pre-registered in
results_v2/final/preregistration_matched.md. Runs the REAL pipeline code on cached transcripts (no ASR, no GPU step, no API calls):
load_transcript_units -> generate_candidate_pairs -> run_stage2, then compares with pairs_v2.csv and the saved out-of-fold values.
Writes results_v2/final/parity.md and parity_rows.csv (and appends to status_matched.txt)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRAIN = REPO / "data" / "training"
OUT = TRAIN / "results_v2" / "final"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from final_scorer import FinalScorer  # noqa: E402
from models.transcript import Transcript  # noqa: E402
from pipeline.step_02_preprocessing import load_transcript_units  # noqa: E402
from pipeline.step_03_token_matching import TokenMatchingConfig, generate_candidate_pairs  # noqa: E402
from pipeline.step_04_semantic_matching import run_stage2  # noqa: E402

VIDEOS = ["video_07", "video_02"]                              # chosen in the pre-registration; fall back to video_08 / video_06 only if a cache is missing
FALLBACK = {"video_07": "video_08", "video_02": "video_06"}
CONFIGS = [("training_builder (min_lexical 0.10, window 300 s)", TokenMatchingConfig(candidate_threshold=0.10, window_seconds=300.0)),
           ("production_default (min_lexical 0.35, window 120 s)", TokenMatchingConfig())]
art = json.loads((HERE / "thresholds.json").read_text())
HI, LO = art["thresholds"]["high"], art["thresholds"]["low"]
S = art["stacker"]
oof = pd.read_csv(OUT / "final_oof_scores.csv", dtype={"pair_id": str, "video_id": str}).set_index("pair_id")
csv_all = pd.read_csv(TRAIN / "pairs_v2.csv", dtype=str, keep_default_na=False)
csv_all = csv_all[csv_all.is_synthetic == "0"].copy()
for c in ("lexical_score", "semantic_score", "containment"):
    csv_all[c] = pd.to_numeric(csv_all[c])
csv_all["csv_plain"] = 0.5 * csv_all.semantic_score + 0.3 * csv_all.lexical_score + 0.2 * csv_all.containment
csv_all = csv_all.set_index("pair_id")

stack = lambda stage2, ce_p: 1 / (1 + np.exp(-(((np.stack([stage2, np.log(np.clip(ce_p, 1e-6, 1 - 1e-6) / (1 - np.clip(ce_p, 1e-6, 1 - 1e-6)))], 1)
                                               - np.array(S["scaler_mean"])) / np.array(S["scaler_scale"])) @ np.array(S["coef_standardized"]) + S["intercept"])))
tier = lambda p, lo, hi: np.where(np.asarray(p) >= hi, "retake", np.where(np.asarray(p) < lo, "unlikely", "ambiguous"))
all_rows, summaries, notes = [], [], []

for video in VIDEOS:
    used = video
    if not (TRAIN / "processed" / video / "transcript.json").exists():
        used = FALLBACK[video]
        notes.append(f"{video}: no cached transcript, used {used} instead")
    units = load_transcript_units(Transcript.from_json_file(str(TRAIN / "processed" / used / "transcript.json")))
    for tag, cfg in CONFIGS:
        rows = []
        for c in run_stage2(generate_candidate_pairs(units, config=cfg)):
            a, b = sorted((c["unit_a"], c["unit_b"]), key=lambda u: u["start"])
            pid = f"{used}__{a['unit_id'].replace(':', '-')}__{b['unit_id'].replace(':', '-')}"
            plain_fields = 0.5 * c["semantic_score"] + 0.3 * c["lexical_score"] + 0.2 * c["scores"]["containment"]
            rows.append({"video": used, "config": tag, "pair_id": pid, "pipeline_stage2": c["stage2_score"], "pipeline_class": c["classification"], "plain_from_pipeline_fields": plain_fields,
                         "lexical_score": c["lexical_score"], "semantic_score": c["semantic_score"], "containment": c["scores"]["containment"]})
        d = pd.DataFrame(rows)
        d["in_csv"] = d.pair_id.isin(csv_all.index)
        d["csv_plain"] = d.pair_id.map(csv_all.csv_plain)
        d["in_oof"] = d.pair_id.isin(oof.index)
        g = d[d.in_csv].copy()
        n_total_csv = int((csv_all.video_id == used).sum())
        g["gap_pipeline_vs_plain"] = g.pipeline_stage2 - g.plain_from_pipeline_fields
        g["gap_plain_vs_csv"] = g.plain_from_pipeline_fields - g.csv_plain
        # (i) old rules on Stage 2 alone
        g["tier_old_plain"], g["tier_old_pipeline"] = tier(g.plain_from_pipeline_fields, 0.35, 0.65), tier(g.pipeline_stage2, 0.35, 0.65)
        # (ii) final thresholds, cross-encoder probability held at its out-of-fold value (labelled pairs only)
        lab = g[g.in_oof].copy()
        ce_p = oof.loc[lab.pair_id, "ce_prob_mean"].values
        lab["p_plain"], lab["p_pipeline"] = stack(lab.plain_from_pipeline_fields.values, ce_p), stack(lab.pipeline_stage2.values, ce_p)
        lab["tier_final_plain"], lab["tier_final_pipeline"] = tier(lab.p_plain, LO, HI), tier(lab.p_pipeline, LO, HI)
        # (a) scorer fed the pipeline's fields + the saved out-of-fold cross-encoder probability, vs the saved final-stacker probability / tier
        sc_mock = FinalScorer(mock=True)
        pa = sc_mock.score_pairs([{"pair_id": r.pair_id, "lexical_score": r.lexical_score, "semantic_score": r.semantic_score, "containment": r.containment, "ce_prob": cp}
                                  for r, cp in zip(lab.itertuples(), ce_p)]) if len(lab) else np.array([])
        lab["p_scorer_oofce"] = pa
        lab["p_saved_final_stacker"] = oof.loc[lab.pair_id, "stacker_prob_final"].values
        lab["p_saved_oof"] = oof.loc[lab.pair_id, "stacker_prob_oof"].values
        lab["tier_saved_final"] = tier(lab.p_saved_final_stacker, LO, HI)
        lab["tier_saved_oof"] = tier(lab.p_saved_oof, LO, HI)
        s = {"video": used, "config": tag, "csv_rows_for_video": n_total_csv, "pipeline_pairs": len(d), "pairs_compared_with_csv": len(g), "pairs_without_csv_row": int((~d.in_csv).sum()),
             "labelled_pairs_with_oof": len(lab),
             "mean_abs_gap_pipeline_vs_plain": float(g.gap_pipeline_vs_plain.abs().mean()), "max_abs_gap_pipeline_vs_plain": float(g.gap_pipeline_vs_plain.abs().max()),
             "pipeline_higher_share": float((g.gap_pipeline_vs_plain > 0).mean()), "share_gap_gt_0.01": float((g.gap_pipeline_vs_plain.abs() > 0.01).mean()),
             "max_abs_gap_plain_vs_csv": float(g.gap_plain_vs_csv.abs().max()),
             "old_rule_tier_changes": int((g.tier_old_plain != g.tier_old_pipeline).sum()),
             "final_threshold_tier_changes": int((lab.tier_final_plain != lab.tier_final_pipeline).sum()),
             "scorer_oofce_tier_agreement_with_saved_final": float((tier(lab.p_scorer_oofce, LO, HI) == lab.tier_saved_final).mean()) if len(lab) else float("nan"),
             "scorer_oofce_max_abs_dp_vs_saved_final": float(np.abs(lab.p_scorer_oofce - lab.p_saved_final_stacker).max()) if len(lab) else float("nan"),
             "scorer_oofce_max_abs_dp_vs_saved_oof": float(np.abs(lab.p_scorer_oofce - lab.p_saved_oof).max()) if len(lab) else float("nan")}
        # (b) full scorer with the FINAL cross-encoder on raw text (CPU), training-builder config only (the labelled set), vs saved out-of-fold stacker values
        if tag.startswith("training") and len(lab):
            real = FinalScorer(device="cpu")
            texts = csv_all.loc[lab.pair_id]
            pb = real.score_pairs([{"pair_id": pid, "text_a": ta, "text_b": tb} for pid, ta, tb in zip(lab.pair_id, texts.text_a, texts.text_b)])
            lab["p_full_scorer_final_ce"] = pb
            s.update({"full_scorer_tier_agreement_with_saved_oof_tier": float((tier(pb, LO, HI) == lab.tier_saved_oof).mean()),
                      "full_scorer_tier_disagreements": int((tier(pb, LO, HI) != lab.tier_saved_oof).sum()),
                      "full_scorer_mean_abs_dp_vs_saved_oof": float(np.abs(pb - lab.p_saved_oof).mean()), "full_scorer_max_abs_dp_vs_saved_oof": float(np.abs(pb - lab.p_saved_oof).max()),
                      "full_scorer_mean_signed_dp": float((pb - lab.p_saved_oof).mean())})
        summaries.append(s)
        all_rows.append(g.merge(lab[["pair_id"] + [c for c in lab.columns if c not in g.columns]], on="pair_id", how="left") if len(lab) else g)   # merge by pair_id (rows without an out-of-fold value stay NaN)

pd.concat(all_rows).to_csv(OUT / "parity_rows.csv", index=False)
f = lambda x, n=4: "n/a" if x != x else f"{x:.{n}f}"
md = ["# Stage 2 parity check on two more videos (Part B)", "",
      "Pre-registered in `preregistration_matched.md`. The REAL pipeline code was run on cached transcripts (no ASR / GPU step, no API calls); its own `stage2_score` is compared with the plain formula the stacker was trained on. "
      "Videos: " + ", ".join(sorted({s['video'] for s in summaries})) + (". Notes: " + "; ".join(notes) if notes else "") + ".", "",
      "Definitions: *pipeline score* = `run_stage2` `stage2_score` (recency-adjusted); *plain from pipeline fields* = 0.5*semantic + 0.3*lexical_score + 0.2*containment from the pipeline's own candidate fields; "
      "*csv plain* = the same formula from the rounded features in pairs_v2.csv. Final tiers use the final thresholds (low %.4f, high %.4f); old rules are 0.35 / 0.65." % (LO, HI), "",
      "## Stage 2 score parity", "",
      "| video | pipeline config | pipeline pairs | compared with pairs_v2 | mean abs gap (pipeline - plain) | max abs gap | pipeline higher | share gap > 0.01 | max abs gap (plain from pipeline fields - csv plain) | tier changes, old 0.35/0.65 rules | tier changes, final thresholds (labelled pairs) |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for s in summaries:
    md.append(f"| {s['video']} | {s['config']} | {s['pipeline_pairs']} | {s['pairs_compared_with_csv']} | {f(s['mean_abs_gap_pipeline_vs_plain'])} | {f(s['max_abs_gap_pipeline_vs_plain'])} | {100 * s['pipeline_higher_share']:.0f}% | "
              f"{100 * s['share_gap_gt_0.01']:.0f}% | {f(s['max_abs_gap_plain_vs_csv'], 5)} | {s['old_rule_tier_changes']} | {s['final_threshold_tier_changes']} (of {s['labelled_pairs_with_oof']}) |")
md += ["", "(Pairs the pipeline produced without a pairs_v2.csv row: " + "; ".join(f"{s['video']} / {s['config'].split(' ')[0]}: {s['pairs_without_csv_row']}" for s in summaries) + ". They have no labelled counterpart, so they are not compared.)", "",
       "## Final scorer vs the saved out-of-fold values (labelled pairs, training-builder config)", "",
       "(a) scorer fed the pipeline's own lexical/semantic/containment fields and the SAVED out-of-fold cross-encoder probability; (b) the full scorer: Stage 2 and the FINAL cross-encoder computed from the raw texts (CPU). The final cross-encoder saw these rows in training, so (b) is not an out-of-sample test; tiers and probabilities are reported separately.", "",
       "| video | labelled pairs | (a) tier agreement with saved final-stacker tier | (a) max abs dp vs saved final stacker | (a) max abs dp vs saved out-of-fold stacker | (b) tier agreement with saved out-of-fold tier | (b) tier disagreements | (b) mean abs dp | (b) max abs dp | (b) mean signed dp |", "|---|---|---|---|---|---|---|---|---|---|"]
for s in summaries:
    if "full_scorer_tier_agreement_with_saved_oof_tier" in s:
        md.append(f"| {s['video']} | {s['labelled_pairs_with_oof']} | {100 * s['scorer_oofce_tier_agreement_with_saved_final']:.1f}% | {f(s['scorer_oofce_max_abs_dp_vs_saved_final'], 5)} | {f(s['scorer_oofce_max_abs_dp_vs_saved_oof'])} | "
                  f"{100 * s['full_scorer_tier_agreement_with_saved_oof_tier']:.1f}% | {s['full_scorer_tier_disagreements']} | {f(s['full_scorer_mean_abs_dp_vs_saved_oof'])} | {f(s['full_scorer_max_abs_dp_vs_saved_oof'])} | {f(s['full_scorer_mean_signed_dp'])} |")
md += ["", "## Reading it", "",
       "- The plain formula rebuilt from the pipeline's own fields must equal the pairs_v2.csv score within the CSV's 3-decimal rounding (< 0.001); the pipeline's own `stage2_score` is lower because it multiplies the lexical score by a recency weight.",
       "- Tier changes show how often feeding the pipeline's value instead of the plain one would alter a decision; the integration rule stays: pass lexical_score, semantic_score and containment (or raw text), never the pipeline's stage2_score.",
       "- (b) disagreements, if any, come from the final cross-encoder differing from the out-of-fold models; they are expected to be near the thresholds.", ""]
(OUT / "parity.md").write_text("\n".join(md), encoding="utf-8")
with open(OUT / "status_matched.txt", "a", encoding="utf-8") as fh:
    fh.write("parity: done\n")
print("\n".join(md[8:20]))
