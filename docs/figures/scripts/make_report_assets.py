"""
Figures and tables for docs/finetune_reports/REPORT.md (the 4B keeper decision model that would replace the Groq keeper call).
Reads saved outputs only; trains nothing, calls no API.

    python docs/finetune_reports/make_report_assets.py

Inputs : runs/keeper_loop/ (git-ignored): iter_01/eval_metrics.json, holdout1_fold{1,2}/{eval,data,train_log_history.json,train_result.json},
         baselines/groq_results.jsonl
Outputs: docs/finetune_reports/figures/f1..f6 (png), data/summary.json, data/per_group.csv, data/tables.md
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOOP = REPO / "runs" / "keeper_loop"
OUT = REPO / "docs" / "finetune_reports"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "data").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(LOOP / "scripts"))
import common as C  # noqa: E402
import metrics as M  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"model": "#1f77b4", "groq": "#d62728", "keep_last": "#7f7f7f"}
NAME = {"model": "4B fine-tune", "groq": "Groq gpt-oss-120b", "keep_last": "keep-last"}
w = 0.26


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def smooth(ys, a=0.08):
    """Exponential moving average (bias-corrected, in log space so near-zero tails stay readable)."""
    out, m, w = [], 0.0, 0.0
    for y in np.log10(ys):
        m = (1 - a) * m + a * y
        w = (1 - a) * w + a
        out.append(10 ** (m / w))
    return out


groq = {d["group_id"]: d for d in jl(LOOP / "baselines/groq_results.jsonl")}

# ---- per-group results: DEV (iter_01 model) and HOLDOUT folds 1, 2 (one model per fold)
ok = {"model": {}, "groq": {}, "keep_last": {}}
split_of, kind_of, probs_of, syn = {}, {}, {}, {}
dev_meta = J(LOOP / "iter_01/data/groups_meta.json")
dev = M.score_set(jl(LOOP / "iter_01/eval/dev_probs.jsonl"), dev_meta)
for r in dev["rows"]:
    ok["model"][r["group_id"]] = r["ok"]; split_of[r["group_id"]] = "DEV"; kind_of[r["group_id"]] = r["kind"]; probs_of[r["group_id"]] = r["probs"]
syn["iter_01"] = M.score_set(jl(LOOP / "iter_01/eval/synthdev_probs.jsonl"), dev_meta)
allmeta = dict(dev_meta)
for f in (1, 2):
    D = LOOP / f"holdout1_fold{f}"
    meta = J(D / "data/groups_meta.json")
    allmeta.update(meta)
    s = M.score_set(jl(D / "eval/dev_probs.jsonl"), meta)
    for r in s["rows"]:
        ok["model"][r["group_id"]] = r["ok"]; split_of[r["group_id"]] = f"fold{f}"; kind_of[r["group_id"]] = r["kind"]; probs_of[r["group_id"]] = r["probs"]
    syn[f"fold{f}"] = M.score_set(jl(D / "eval/synthdev_probs.jsonl"), meta)
for g in ok["model"]:
    k = groq[g]
    ok["groq"][g] = k["status"] == "resolved" and C.exact_match(k["keep"], allmeta[g]["gold"])
    ok["keep_last"][g] = C.exact_match([allmeta[g]["unit_ids"][-1]], allmeta[g]["gold"])


def grp(names):
    return sorted(g for g in ok["model"] if split_of[g] in names)


splits = {"DEV (folds 3+4)": grp({"DEV"}), "HOLDOUT fold 1": grp({"fold1"}), "HOLDOUT fold 2": grp({"fold2"}), "HOLDOUT pooled": grp({"fold1", "fold2"})}


def count(s, sysn, pred=lambda g: True):
    return sum(ok[sysn][g] for g in s if pred(g)), sum(1 for g in s if pred(g))


def bars(ax, groups, preds_or_splits, label_fs=7):
    for i, sysn in enumerate(("model", "groq", "keep_last")):
        xs = np.arange(len(groups)) + (i - 1) * w
        vals = [count(*gp, sysn=sysn) if False else count(gp[0], sysn, gp[1]) for gp in groups]
        ax.bar(xs, [a / b for a, b in vals], w, color=COL[sysn], label=NAME[sysn])
        for x, (a, b) in zip(xs, vals):
            ax.text(x, a / b + 0.012, f"{a}/{b}", ha="center", fontsize=label_fs)


# ---- Figure 1: accuracy by split
fig, ax = plt.subplots(figsize=(8.5, 3.9))
bars(ax, [(s, lambda g: True) for s in splits.values()], None)
ax.set(xticks=range(len(splits)), xticklabels=list(splits), ylim=(0.6, 1.08), ylabel="exact keeper-set match")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
ax.set_title("Natural groups: 4B fine-tune vs the Groq keeper call vs keep-last")
fig.tight_layout(); fig.savefig(OUT / "figures/f1_accuracy_by_split.png"); plt.close(fig)

# ---- Figure 2: nuanced vs keep-last groups, DEV + HOLDOUT
allg = sorted(ok["model"])
sl = [("Nuanced groups (keeper not last)", lambda g: kind_of[g] != "last"), ("Keep-last groups", lambda g: kind_of[g] == "last")]
fig, ax = plt.subplots(figsize=(7, 3.7))
bars(ax, [(allg, p) for _, p in sl], None, 7.5)
ax.set(xticks=range(2), xticklabels=[n for n, _ in sl], ylim=(0, 1.12), ylabel="exact keeper-set match")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)
ax.set_title("DEV + HOLDOUT folds 1-2 (131 groups)")
fig.tight_layout(); fig.savefig(OUT / "figures/f2_nuanced_vs_keeplast.png"); plt.close(fig)

# ---- Figure 3: paired differences with bootstrap intervals
rows_ci = []
for sn, s in splits.items():
    for other in ("groq", "keep_last"):
        d, lo, hi = C.bootstrap_diff([ok["model"][g] for g in s], [ok[other][g] for g in s])
        rows_ci.append({"split": sn, "versus": other, "n": len(s), "diff_points": 100 * d, "lo": 100 * lo, "hi": 100 * hi,
                        "model_correct": count(s, "model")[0], "other_correct": count(s, other)[0]})
fig, ax = plt.subplots(figsize=(8.5, 4.4))
for i, r in enumerate(rows_ci):
    ax.errorbar(r["diff_points"], i, xerr=[[r["diff_points"] - r["lo"]], [r["hi"] - r["diff_points"]]], fmt="o", color=COL[r["versus"]], capsize=3)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(range(len(rows_ci)))
ax.set_yticklabels([f"{r['split']} vs {NAME[r['versus']]} (n={r['n']})" for r in rows_ci], fontsize=7.5)
ax.invert_yaxis()
ax.set(xlabel="accuracy difference, model minus comparator (points); paired bootstrap 95% interval")
ax.set_title("An interval that touches or crosses zero is not evidence of a difference")
fig.tight_layout(); fig.savefig(OUT / "figures/f3_paired_differences.png"); plt.close(fig)

# ---- Figure 4: training and evaluation loss
fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
runs = {"DEV model (trained on folds 1+2); eval set = 59 natural DEV groups": "iter_01", "fold-1 model; eval set = 22 synthetic groups": "holdout1_fold1", "fold-2 model; eval set = 22 synthetic groups": "holdout1_fold2"}
for (n, d), c in zip(runs.items(), ("#1f77b4", "#2ca02c", "#9467bd")):
    h = J(LOOP / d / "train_log_history.json")
    tr = [(x["step"], max(x["loss"], 1e-6)) for x in h if "loss" in x and "step" in x]
    ev = [(x["step"], max(x["eval_loss"], 1e-6)) for x in h if "eval_loss" in x]
    ax[0].plot(*zip(*tr), color=c, lw=0.5, alpha=0.2)
    ax[0].plot([t[0] for t in tr], smooth([t[1] for t in tr]), color=c, lw=1.7, label=n)
    if ev:
        ax[1].plot(*zip(*ev), "o-", color=c, ms=3, lw=0.9, label=n)
ax[0].set(xlabel="step", ylabel="training loss (soft cross-entropy)", title="Training loss (smoothed; raw faint)", yscale="log")
ax[1].set(xlabel="step", ylabel="evaluation loss (eval set differs by run, see legend)", title="Evaluation loss", yscale="log")
for a in ax:
    a.legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(OUT / "figures/f4_training_loss.png"); fig.savefig(OUT / "figures/f4_training_loss.pdf"); plt.close(fig)

# ---- Figure 5: held-out synthetic accuracy by rule type
ty = {k: {t[5:]: v for t, v in s["accuracy"].items() if t.startswith("type:")} for k, s in syn.items()}
types = sorted(ty["fold1"])
kls = M.score_set(M.keep_last_records([r["group_id"] for r in syn["fold1"]["rows"]], allmeta), allmeta)
kl = {t[5:]: v for t, v in kls["accuracy"].items() if t.startswith("type:")}
fig, ax = plt.subplots(figsize=(9.5, 4.2))
x = np.arange(len(types))
for off, (lab, d, c) in zip((-0.27, 0, 0.27), (("fold-1 model", ty["fold1"], "#2ca02c"), ("fold-2 model", ty["fold2"], "#9467bd"), ("keep-last", kl, COL["keep_last"]))):
    ax.bar(x + off, [d[t]["acc"] for t in types], 0.27, color=c, label=lab)
ax.set(xticks=x, ylim=(0, 1.2), ylabel="exact keeper-set match", title="Held-out synthetic dev (220 groups, 20 per rule type)")
ax.set_xticklabels([t.replace("->", " -> ").replace("_", " ") for t in types], rotation=35, ha="right", fontsize=7.5)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "figures/f5_synthetic_by_type.png"); fig.savefig(OUT / "figures/f5_synthetic_by_type.pdf"); plt.close(fig)

# ---- Figure 6: p(keep) histogram on HOLDOUT units
hp = [(p, u in allmeta[g]["gold"]) for g in grp({"fold1", "fold2"}) for u, p in probs_of[g].items()]
fig, ax = plt.subplots(figsize=(6.5, 3.5))
bins = np.linspace(0, 1, 11)
ax.hist([p for p, y in hp if y], bins, alpha=0.7, color="#2ca02c", label="unit is a keeper")
ax.hist([p for p, y in hp if not y], bins, alpha=0.7, color="#d62728", label="unit is cut")
ax.set(xlabel="model p(keep)", ylabel="units (log scale)", yscale="log", title=f"HOLDOUT p(keep), {len(hp)} units")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "figures/f6_probability_histogram.png"); fig.savefig(OUT / "figures/f6_probability_histogram.pdf"); plt.close(fig)
sat = sum(1 for p, _ in hp if p < 0.05 or p > 0.95) / len(hp)

# ---- tables
tr = {d: J(LOOP / d / "train_result.json") for d in runs.values()}
T = ["| split | n | 4B fine-tune | Groq gpt-oss-120b | keep-last |", "|---|---|---|---|---|"]
for sn, s in splits.items():
    c = {k: count(s, k)[0] for k in ok}
    T.append(f"| {sn} | {len(s)} | " + " | ".join(f"{c[k]} ({100*c[k]/len(s):.1f}%)" for k in ok) + " |")
T += ["", "| slice (DEV + HOLDOUT) | n | 4B fine-tune | Groq | keep-last |", "|---|---|---|---|---|"]
for n, p in sl:
    c = {k: count(allg, k, p) for k in ok}
    T.append(f"| {n} | {c['model'][1]} | {c['model'][0]} | {c['groq'][0]} | {c['keep_last'][0]} |")
T += ["", "| split | comparator | model correct | comparator correct | difference (points) | 95% interval |", "|---|---|---|---|---|---|"]
for r in rows_ci:
    T.append(f"| {r['split']} | {NAME[r['versus']]} | {r['model_correct']} | {r['other_correct']} | {r['diff_points']:+.1f} | [{r['lo']:+.1f}, {r['hi']:+.1f}] |")
T += ["", "| run | train groups (natural + synthetic) | train decisions | steps | train time (min) | peak GPU memory (GB) | mean train loss |", "|---|---|---|---|---|---|---|"]
for d, t in tr.items():
    h = J(LOOP / d / "train_log_history.json")
    mc = J(LOOP / d / 'data/manifest.json')['counts']
    T.append(f"| {d} | {mc['train_rows']} ({mc['train_natural_rows']} + {mc['train_synthetic_rows']}) | {t['dataset_report']['total']} | {t['steps']} | {t['train_seconds']/60:.0f} | {t['peak_memory_gb']} | {h[-1].get('train_loss', float('nan')):.3f} |")
(OUT / "data/tables.md").write_text("\n".join(T) + "\n", encoding="utf-8")
with open(OUT / "data/per_group.csv", "w", newline="", encoding="utf-8") as f:
    cw = csv.writer(f)
    cw.writerow(["group_id", "split", "kind", "n_units", "gold", "model_ok", "groq_ok", "keep_last_ok"])
    for g in sorted(ok["model"]):
        cw.writerow([g, split_of[g], kind_of[g], allmeta[g]["n_units"], "|".join(allmeta[g]["gold"]), ok["model"][g], ok["groq"][g], ok["keep_last"][g]])
summary = {"ci": rows_ci, "saturated_share": sat, "holdout_units": len(hp),
           "synthetic": {k: {"correct": v["accuracy"]["all"]["correct"], "n": v["n"]} for k, v in syn.items()},
           "wrong_groups": {sn: [g for g in s if not ok["model"][g]] for sn, s in splits.items()},
           "auc3": {"iter_01_dev": dev["auc_within_3plus"]}}
(OUT / "data/summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("\n".join(T))
print("saturated share", round(sat, 4), "of", len(hp))
print(json.dumps(summary["wrong_groups"], indent=1))
print(summary["synthetic"], summary["auc3"])


# ---- four-fold run: one frozen-recipe model per outer fold (folds 1, 2 clean; folds 3, 4 were the tuning set)
folds4 = {"fold 1 (clean)": "holdout1_fold1", "fold 2 (clean)": "holdout1_fold2", "fold 3 (tuned on)": "outer_fold3", "fold 4 (tuned on)": "outer_fold4"}
ok4 = {"model": {}, "groq": {}, "keep_last": {}}
rows4, per_fold = [], {}
for lab, d in folds4.items():
    meta = J(LOOP / d / "data/groups_meta.json")
    recs = jl(LOOP / d / "eval/dev_probs.jsonl")
    keys = [r["group_id"] for r in recs]
    sm = M.score_set(recs, meta)
    res4 = {"model": sm, "keep_last": M.score_set(M.keep_last_records(keys, meta), meta),
            "groq": M.score_set([{"group_id": k, "keep": groq[k]["keep"], "probs": None, "error": None if groq[k]["status"] == "resolved" else "f"} for k in keys], meta)}
    for n, sc in res4.items():
        for r in sc["rows"]:
            ok4[n][r["group_id"]] = r["ok"]
    syn4 = M.score_set(jl(LOOP / d / "eval/synthdev_probs.jsonl"), meta)
    per_fold[lab] = {n: sum(r["ok"] for r in sc["rows"]) for n, sc in res4.items()} | {"n": len(keys), "auc3": sm["auc_within_3plus"], "auc_groups": sm["auc_groups_3plus"],
                                                                                         "nuanced": sm["accuracy"].get("informative"), "synth": syn4["accuracy"]["all"]["correct"]}
ks = sorted(ok4["model"])
pooled4 = {n: sum(ok4[n][g] for g in ks) for n in ok4}
ci4 = {}
for o in ("groq", "keep_last"):
    d_, lo, hi = C.bootstrap_diff([ok4["model"][g] for g in ks], [ok4[o][g] for g in ks])
    ci4[o] = [100 * d_, 100 * lo, 100 * hi]
fig, ax = plt.subplots(figsize=(9, 3.9))
labs = list(folds4) + ["pooled (131)"]
for i, sysn in enumerate(("model", "groq", "keep_last")):
    xs = np.arange(len(labs)) + (i - 1) * w
    vals = [(per_fold[l][sysn], per_fold[l]["n"]) for l in folds4] + [(pooled4[sysn], len(ks))]
    ax.bar(xs, [a / b for a, b in vals], w, color=COL[sysn], label=NAME[sysn])
    for x, (a, b) in zip(xs, vals):
        ax.text(x, a / b + 0.01, f"{a}/{b}", ha="center", fontsize=7)
ax.set(xticks=range(len(labs)), xticklabels=labs, ylim=(0.6, 1.08), ylabel="exact keeper-set match")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
ax.set_title("Four-fold run: one model per outer fold, frozen recipe")
fig.tight_layout(); fig.savefig(OUT / "figures/f7_four_fold.png"); fig.savefig(OUT / "figures/f7_four_fold.pdf"); plt.close(fig)
T4 = ["| fold | n | 4B fine-tune | Groq | keep-last | nuanced (model) | AUC 3+ (groups) | synthetic dev |", "|---|---|---|---|---|---|---|---|"]
for l, v in per_fold.items():
    nu = v["nuanced"]
    T4.append(f"| {l} | {v['n']} | {v['model']} | {v['groq']} | {v['keep_last']} | {nu['correct']}/{nu['n']} | {v['auc3']:.3f} ({v['auc_groups']}) | {v['synth']}/220 |")
T4.append(f"| **pooled** | {len(ks)} | **{pooled4['model']}** ({100*pooled4['model']/len(ks):.1f}%) | {pooled4['groq']} ({100*pooled4['groq']/len(ks):.1f}%) | {pooled4['keep_last']} ({100*pooled4['keep_last']/len(ks):.1f}%) | | | |")
T4.append(f"\nPooled 131, model minus Groq: {ci4['groq'][0]:+.1f} points, 95% interval [{ci4['groq'][1]:+.1f}, {ci4['groq'][2]:+.1f}]. Model minus keep-last: {ci4['keep_last'][0]:+.1f} points, [{ci4['keep_last'][1]:+.1f}, {ci4['keep_last'][2]:+.1f}].")
(OUT / "data/tables_four_fold.md").write_text("\n".join(T4) + "\n", encoding="utf-8")
(OUT / "data/summary_four_fold.json").write_text(json.dumps({"per_fold": per_fold, "pooled": pooled4, "n": len(ks), "ci_points": ci4}, indent=1), encoding="utf-8")
print("\n".join(T4))
