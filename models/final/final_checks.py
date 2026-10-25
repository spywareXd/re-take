"""Review follow-ups for the final scorer. No training, no API calls.

A. Fair baseline: Stage 2 (and the cross-encoder alone) with thresholds chosen by the SAME procedure as the stacker (95% precision / 98% recall,
   nested over the speaker folds), so the reduction of the ambiguous band can be split into 'tuning' and 'stacker'.
B. Operating-point table: band share, tier precision, tier recall at several high thresholds of the stacker.
C. Stage 2 consistency: run the real pipeline code (cached transcript -> units -> candidates -> run_stage2) on one labelled video and compare the pipeline's
   Stage 2 score with the plain-formula score the stacker and thresholds were trained on (pairs_v2.csv features).
Writes results_v2/final/fair_comparison.json, operating_points.csv, stage2_consistency.json and stage2_consistency_rows.csv.
"""
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
PREC_T, REC_T, N_BOOT, SEED = 0.95, 0.98, 1000, 0
df = pd.read_csv(OUT / "final_oof_scores.csv", dtype={"pair_id": str, "video_id": str})
y, fold, groups = df.label.values, df.fold.values, df.video_id.values
FOLDS = sorted(np.unique(fold))
art = json.loads((HERE / "thresholds.json").read_text())


# ---- same functions as fit_final.py (copied so this script does not re-run the fit) ----
def choose_thresholds(yt, p):
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], yt[order]
    tp, n = np.cumsum(ys), np.arange(1, len(ps) + 1)
    last = np.r_[ps[1:] != ps[:-1], True]
    prec, rec, thr = tp[last] / n[last], tp[last] / max(1, ys.sum()), ps[last]
    ok_hi = np.where(prec >= PREC_T)[0]
    high = float(thr[ok_hi.max()]) if len(ok_hi) else float("inf")
    ok_lo = np.where(rec >= REC_T)[0]
    low = float(thr[ok_lo.min()]) if len(ok_lo) else float("-inf")
    return min(low, high), high


def tiers(p, low, high):
    return np.where(p >= high, "retake", np.where(p < low, "unlikely", "ambiguous"))


def tier_stats(yt, tr):
    yt = np.asarray(yt)
    n, P, N = len(yt), int((yt == 1).sum()), int((yt == 0).sum())
    o = {"n": n}
    for t in ("retake", "ambiguous", "unlikely"):
        m = tr == t
        o[f"{t}_n"], o[f"{t}_share"] = int(m.sum()), float(m.sum() / n) if n else 0.0
        o[f"{t}_retakes"], o[f"{t}_nonretakes"] = int(((yt == 1) & m).sum()), int(((yt == 0) & m).sum())
    pr, rc = o["retake_retakes"] / max(1, o["retake_n"]), o["retake_retakes"] / max(1, P)
    npv, nrc = o["unlikely_nonretakes"] / max(1, o["unlikely_n"]), o["unlikely_nonretakes"] / max(1, N)
    f = lambda a, b: 2 * a * b / (a + b) if a + b else 0.0
    o.update({"retake_precision": pr, "retake_recall": rc, "retake_f1": f(pr, rc), "unlikely_precision": npv, "unlikely_recall": nrc,
              "retakes_reach_cut_or_review": 1 - o["unlikely_retakes"] / max(1, P)})
    return o


def nested_tiers(score):
    """thresholds for outer fold k come only from the other folds' scores (no model is fitted for a single raw score)"""
    out = np.empty(len(score), dtype=object)
    thr = {}
    for k in FOLDS:
        te = fold == k
        lo, hi = choose_thresholds(y[~te], score[~te])
        thr[int(k)] = (lo, hi)
        out[te] = tiers(score[te], lo, hi)
    return out, thr


# ================= A. fair comparison =================
systems = {}
t_s2, thr_s2 = nested_tiers(df.stage2_score.values)
t_ce, thr_ce = nested_tiers(df.ce_prob_mean.values)
systems["stage2_old_rules_0.65_0.35"] = df.tier_stage2.values
systems["stage2_tuned_same_procedure"] = t_s2
systems["cross_encoder_alone_tuned_same_procedure"] = t_ce
systems["stacker_tuned_same_procedure"] = df.tier_nested.values
final_thr = {"stage2": choose_thresholds(y, df.stage2_score.values), "cross_encoder": choose_thresholds(y, df.ce_prob_mean.values),
             "stacker": (art["thresholds"]["low"], art["thresholds"]["high"])}
res = {name: {"pooled": tier_stats(y, t), **{f"fold{k}": tier_stats(y[fold == k], t[fold == k]) for k in FOLDS}} for name, t in systems.items()}

rng = np.random.default_rng(SEED)
vids = pd.unique(groups)
idx_by_v = {v: np.where(groups == v)[0] for v in vids}
KEYS = ["ambiguous_share", "retake_precision", "retake_recall", "unlikely_recall", "unlikely_precision", "retakes_reach_cut_or_review"]
boot = {n: {k: [] for k in KEYS} for n in systems}
for _ in range(N_BOOT):
    idx = np.concatenate([idx_by_v[v] for v in rng.choice(vids, len(vids), replace=True)])
    for n, t in systems.items():
        st = tier_stats(y[idx], t[idx])
        for k in KEYS:
            boot[n][k].append(st[k])
ci = {}
for other in ("stage2_old_rules_0.65_0.35", "stage2_tuned_same_procedure", "cross_encoder_alone_tuned_same_procedure"):
    ci[f"stacker_minus_{other}"] = {}
    for k in KEYS:
        d = np.array(boot["stacker_tuned_same_procedure"][k]) - np.array(boot[other][k])
        lo, hi = np.percentile(d, [2.5, 97.5])
        ci[f"stacker_minus_{other}"][k] = {"diff": res["stacker_tuned_same_procedure"]["pooled"][k] - res[other]["pooled"][k], "lo": float(lo), "hi": float(hi)}
(OUT / "fair_comparison.json").write_text(json.dumps({"results": res, "bootstrap": ci, "nested_thresholds": {"stage2": {k: list(v) for k, v in thr_s2.items()},
                                                      "cross_encoder": {k: list(v) for k, v in thr_ce.items()}}, "final_thresholds": {k: list(v) for k, v in final_thr.items()}}, indent=2), encoding="utf-8")

# ================= B. operating points (stacker out-of-fold probabilities) =================
p = df.stacker_prob_oof.values
low = art["thresholds"]["low"]
rows = []
for hi in [0.70, 0.80, art["thresholds"]["high"], 0.90, 0.93, 0.95, 0.97, 0.98, 0.99]:
    tr = tiers(p, low, hi)
    s = tier_stats(y, tr)
    f3, f4 = tier_stats(y[fold == 3], tr[fold == 3]), tier_stats(y[fold == 4], tr[fold == 4])
    rows.append({"high": round(hi, 4), "is_final": abs(hi - art["thresholds"]["high"]) < 1e-12, "retake_n": s["retake_n"], "ambiguous_share": s["ambiguous_share"],
                 "retake_precision": s["retake_precision"], "retake_recall": s["retake_recall"], "retakes_in_ambiguous": s["ambiguous_retakes"],
                 "retakes_reach_cut_or_review": s["retakes_reach_cut_or_review"], "fold3_precision": f3["retake_precision"], "fold3_false_cuts": f3["retake_nonretakes"],
                 "fold4_recall": f4["retake_recall"]})
op = pd.DataFrame(rows)
op.to_csv(OUT / "operating_points.csv", index=False)

# ================= C. Stage 2 consistency with the real pipeline =================
from models.retake import TimeSpan  # noqa: E402,F401  (repo import check)
from models.transcript import Transcript  # noqa: E402
from pipeline.step_03_token_matching import TokenMatchingConfig, generate_candidate_pairs, load_transcript_units  # noqa: E402
from pipeline.step_04_semantic_matching import run_stage2  # noqa: E402

VIDEO = "speaker_a_3"
tr_json = TRAIN / "processed" / VIDEO / "transcript.json"
units = load_transcript_units(Transcript.from_json_file(str(tr_json)))
cons_rows = []
for tag, cfg in (("training_builder_config(min_lexical 0.10, window 300s)", TokenMatchingConfig(candidate_threshold=0.10, window_seconds=300.0)),
                 ("production_default_config(min_lexical 0.35, window 120s)", TokenMatchingConfig())):
    cands = run_stage2(generate_candidate_pairs(units, config=cfg))
    for c in cands:
        a, b = c["unit_a"], c["unit_b"]
        if a["start"] > b["start"]:
            a, b = b, a
        pid = f"{VIDEO}__{a['unit_id'].replace(':', '-')}__{b['unit_id'].replace(':', '-')}"
        plain = 0.5 * c["semantic_score"] + 0.3 * c["lexical_score"] + 0.2 * c["scores"]["containment"] if isinstance(c.get("scores"), dict) else np.nan
        cons_rows.append({"config": tag, "pair_id": pid, "pipeline_stage2": c["stage2_score"], "plain_from_pipeline_fields": plain,
                          "lexical_score": c["lexical_score"], "combined_score": c["combined_score"], "semantic_score": c["semantic_score"],
                          "recency_weight": c["recency_weight"], "restart_phrase": bool(c.get("restart_phrase_detected")), "pipeline_class": c["classification"]})
cons = pd.DataFrame(cons_rows)
csv_s2 = df.set_index("pair_id").stage2_score
cons["csv_stage2_plain"] = cons.pair_id.map(csv_s2)
cons["labeled"] = cons.csv_stage2_plain.notna()
lab = cons[cons.labeled].copy()
lab["diff_pipeline_minus_csv"] = lab.pipeline_stage2 - lab.csv_stage2_plain
lab["diff_plain_bridge_minus_csv"] = lab.plain_from_pipeline_fields - lab.csv_stage2_plain
hi_t, lo_t = art["thresholds"]["high"], art["thresholds"]["low"]
cons_summary = {}
for tag, g in lab.groupby("config"):
    d, bd = g.diff_pipeline_minus_csv, g.diff_plain_bridge_minus_csv
    # tier flips if the pipeline value were fed to the scorer in place of the plain one (CE held fixed at its OOF value)
    ce_p = df.set_index("pair_id").loc[g.pair_id, "ce_prob_mean"].values
    ce_l = np.log(np.clip(ce_p, 1e-6, 1 - 1e-6) / (1 - np.clip(ce_p, 1e-6, 1 - 1e-6)))
    s = art["stacker"]

    def stack(x):
        z = (np.stack([x, ce_l], 1) - np.array(s["scaler_mean"])) / np.array(s["scaler_scale"])
        return 1 / (1 + np.exp(-(z @ np.array(s["coef_standardized"]) + s["intercept"])))

    t_plain, t_pipe = tiers(stack(g.csv_stage2_plain.values), lo_t, hi_t), tiers(stack(g.pipeline_stage2.values), lo_t, hi_t)
    cons_summary[tag] = {"pairs_compared": int(len(g)), "pipeline_vs_csv_mean_abs": float(d.abs().mean()), "pipeline_vs_csv_max_abs": float(d.abs().max()),
                         "share_abs_gt_0.01": float((d.abs() > 0.01).mean()), "share_abs_gt_0.05": float((d.abs() > 0.05).mean()),
                         "pipeline_higher_share": float((d > 0).mean()), "bridge_plain_vs_csv_max_abs": float(bd.abs().max()),
                         "tier_flips_if_pipeline_score_used": int((t_plain != t_pipe).sum()), "tier_flip_share": float((t_plain != t_pipe).mean()),
                         "pairs_without_label_row": int((cons.config == tag).sum() - len(g))}
lab.to_csv(OUT / "stage2_consistency_rows.csv", index=False)
(OUT / "stage2_consistency.json").write_text(json.dumps({"video": VIDEO, "summary": cons_summary}, indent=2), encoding="utf-8")
print(json.dumps(cons_summary, indent=1))
for name in systems:
    r = res[name]["pooled"]
    print(f"{name}: ambiguous {r['ambiguous_share']:.1%} | retake P {r['retake_precision']:.3f} R {r['retake_recall']:.3f} | unlikely retakes {r['unlikely_retakes']} | reach cut-or-review {r['retakes_reach_cut_or_review']:.1%}")
print(op.round(3).to_string(index=False))
