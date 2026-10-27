"""
Figures and tables for docs/finetune_final/REPORT.md: the final all-data keeper model, the second-seed check, and the keep-last fallback.
Reads saved outputs only; trains nothing, calls no API.

    python docs/finetune_final/make_final_assets.py

Inputs : runs/keeper_loop/ (git-ignored): final_all/{eval,data,train_log_history.json,train_result.json}, seed2_fold4/, outer_fold4/,
         holdout1_fold{1,2}/, outer_fold3/, baselines/groq_results.jsonl
Outputs: docs/finetune_final/figures/g1..g5 (png + pdf), data/final_numbers.json, data/tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOOP = REPO / "runs" / "keeper_loop"
OUT = REPO / "docs" / "finetune_final"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "data").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(LOOP / "scripts"))
import common as C  # noqa: E402
import metrics as M  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {"model": "#1f77b4", "groq": "#d62728", "keep_last": "#7f7f7f", "hybrid": "#2ca02c"}


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{name}.png")
    fig.savefig(OUT / "figures" / f"{name}.pdf")
    plt.close(fig)


def smooth(ys, a=0.08):
    """Exponential moving average (bias-corrected, in log space so near-zero tails stay readable)."""
    out, m, w = [], 0.0, 0.0
    for y in np.log10(ys):
        m = (1 - a) * m + a * y
        w = (1 - a) * w + a
        out.append(10 ** (m / w))
    return out


groq = {d["group_id"]: d for d in jl(LOOP / "baselines/groq_results.jsonl")}
num = {}

# ---- g1: training and evaluation loss of the final model
h = J(LOOP / "final_all/train_log_history.json")
tr = [(x["step"], max(x["loss"], 1e-6)) for x in h if "loss" in x and "step" in x]
ev = [(x["step"], max(x["eval_loss"], 1e-6)) for x in h if "eval_loss" in x]
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(*zip(*tr), color=COL["model"], lw=0.6, alpha=0.25)
ax.plot([t[0] for t in tr], smooth([t[1] for t in tr]), color=COL["model"], lw=1.8, label="training loss (smoothed, EMA 0.08 of log loss; raw faint)")
ax.plot(*zip(*ev), "o-", color="#ff7f0e", ms=3, lw=0.9, label="evaluation loss (22 synthetic groups)")
ax.set(xlabel="step", ylabel="soft cross-entropy (log scale)", yscale="log", title="Final model: all 131 natural groups + 1,200 synthetic")
ax.legend(frameon=False)
save(fig, "g1_final_loss")
res = J(LOOP / "final_all/train_result.json")
mc = J(LOOP / "final_all/data/manifest.json")["counts"]
num["final_training"] = {"steps": res["steps"], "train_seconds": res["train_seconds"], "peak_gb": res["peak_memory_gb"], "train_groups": mc["train_rows"],
                         "natural": mc["train_natural_rows"], "synthetic": mc["train_synthetic_rows"], "train_decisions": res["dataset_report"]["total"],
                         "mean_train_loss": h[-1].get("train_loss"), "final_eval_loss": ev[-1][1]}

# ---- g2: held-out synthetic accuracy by rule type, final vs the fold models and keep-last
meta_f = J(LOOP / "final_all/data/groups_meta.json")
sets = {"final model": jl(LOOP / "final_all/eval/synthdev_probs.jsonl")}
for lab, d in (("fold-1 model", "holdout1_fold1"), ("fold-2 model", "holdout1_fold2"), ("fold-3 model", "outer_fold3"), ("fold-4 model", "outer_fold4")):
    sets[lab] = jl(LOOP / d / "eval/synthdev_probs.jsonl")
metas = {"final model": meta_f, "fold-1 model": J(LOOP / "holdout1_fold1/data/groups_meta.json"), "fold-2 model": J(LOOP / "holdout1_fold2/data/groups_meta.json"),
         "fold-3 model": J(LOOP / "outer_fold3/data/groups_meta.json"), "fold-4 model": J(LOOP / "outer_fold4/data/groups_meta.json")}
sc = {k: M.score_set(v, metas[k]) for k, v in sets.items()}
types = sorted(t[5:] for t in sc["final model"]["accuracy"] if t.startswith("type:"))
kls = M.score_set(M.keep_last_records([r["group_id"] for r in sets["final model"]], meta_f), meta_f)
fig, ax = plt.subplots(figsize=(10, 4.3))
x = np.arange(len(types))
cols = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", COL["keep_last"]]
labs = list(sc) + ["keep-last"]
data = [sc[k] for k in sc] + [kls]
bw = 0.13
for i, (lab, s, c) in enumerate(zip(labs, data, cols)):
    ax.bar(x + (i - 2.5) * bw, [s["accuracy"].get("type:" + t, {"acc": 0})["acc"] for t in types], bw, color=c, label=lab)
ax.set(xticks=x, ylim=(0, 1.22), ylabel="exact keeper-set match", title="Held-out synthetic dev (220 groups, 20 per rule type)")
ax.set_xticklabels([t.replace("->", " -> ").replace("_", " ") for t in types], rotation=35, ha="right", fontsize=7.5)
ax.legend(frameon=False, ncol=6, loc="upper center", fontsize=7.5)
save(fig, "g2_synthetic_by_type")
num["synthetic_dev"] = {k: {"correct": s["accuracy"]["all"]["correct"], "n": s["n"], "misses": [r["group_id"] for r in s["rows"] if not r["ok"]]} for k, s in sc.items()}
num["synthetic_dev"]["keep_last"] = {"correct": kls["accuracy"]["all"]["correct"], "n": kls["n"]}

# ---- g3: second seed on fold 4
m4 = J(LOOP / "outer_fold4/data/groups_meta.json")
s1 = M.score_set(jl(LOOP / "outer_fold4/eval/dev_probs.jsonl"), m4)
s2 = M.score_set(jl(LOOP / "seed2_fold4/eval/dev_probs.jsonl"), m4)
keys = [r["group_id"] for r in s1["rows"]]
kl4 = M.score_set(M.keep_last_records(keys, m4), m4)
gq4 = M.score_set([{"group_id": k, "keep": groq[k]["keep"], "probs": None, "error": None if groq[k]["status"] == "resolved" else "f"} for k in keys], m4)
tot = lambda s: sum(r["ok"] for r in s["rows"])
both_right = sum(a["ok"] and b["ok"] for a, b in zip(s1["rows"], s2["rows"]))
both_wrong = sum((not a["ok"]) and (not b["ok"]) for a, b in zip(s1["rows"], s2["rows"]))
only1 = sum(a["ok"] and not b["ok"] for a, b in zip(s1["rows"], s2["rows"]))
only2 = sum(b["ok"] and not a["ok"] for a, b in zip(s1["rows"], s2["rows"]))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
names = ["seed 3407", "seed 1234", "Groq", "keep-last"]
vals = [tot(s1), tot(s2), tot(gq4), tot(kl4)]
ax[0].bar(names, [v / len(keys) for v in vals], color=[COL["model"], "#6baed6", COL["groq"], COL["keep_last"]])
for i, v in enumerate(vals):
    ax[0].text(i, v / len(keys) + 0.01, f"{v}/{len(keys)}", ha="center", fontsize=8)
ax[0].set(ylim=(0.5, 1.0), ylabel="exact keeper-set match", title="Fold 4 (34 groups): two training seeds")
parts = [("right in both", both_right, "#2ca02c"), ("right only seed 3407", only1, "#1f77b4"), ("right only seed 1234", only2, "#6baed6"), ("wrong in both", both_wrong, "#d62728")]
left = 0
for lab, v, c in parts:
    ax[1].barh([0], [v], left=left, color=c, label=f"{lab} ({v})")
    left += v
ax[1].set(yticks=[], xlim=(0, len(keys)), xlabel="groups", title="Per-group agreement between the two seeds")
ax[1].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
save(fig, "g3_second_seed_fold4")
num["second_seed_fold4"] = {"seed_3407": tot(s1), "seed_1234": tot(s2), "groq": tot(gq4), "keep_last": tot(kl4), "n": len(keys), "right_both": both_right,
                            "only_3407": only1, "only_1234": only2, "wrong_both": both_wrong, "auc_3407": s1["auc_within_3plus"], "auc_1234": s2["auc_within_3plus"],
                            "nuanced_3407": s1["accuracy"].get("informative"), "nuanced_1234": s2["accuracy"].get("informative")}

# ---- g4: keep-last fallback on the four-fold predictions
folds = {"fold 1": "holdout1_fold1", "fold 2": "holdout1_fold2", "fold 3": "outer_fold3", "fold 4": "outer_fold4"}
per = {}
changes = []
for lab, d in folds.items():
    meta = J(LOOP / d / "data/groups_meta.json")
    c = {"model": 0, "hybrid": 0, "keep_last": 0, "groq": 0, "n": 0}
    for r in jl(LOOP / d / "eval/dev_probs.jsonl"):
        m = meta[r["group_id"]]
        gold, ids = m["gold"], m["unit_ids"]
        pred = C.keep_from_probs(r["probs"])["keep"]
        hyb = pred or [ids[-1]]
        c["n"] += 1
        c["model"] += C.exact_match(pred, gold)
        c["hybrid"] += C.exact_match(hyb, gold)
        c["keep_last"] += C.exact_match([ids[-1]], gold)
        gk = groq[r["group_id"]]
        c["groq"] += gk["status"] == "resolved" and C.exact_match(gk["keep"], gold)
        if not pred:
            changes.append({"group": r["group_id"], "fold": lab, "kind": m["kind"], "model_ok": C.exact_match(pred, gold), "hybrid_ok": C.exact_match(hyb, gold)})
    per[lab] = c
pooled = {k: sum(per[f][k] for f in per) for k in ("model", "hybrid", "keep_last", "groq", "n")}
fig, ax = plt.subplots(figsize=(9, 3.9))
labs = list(per) + ["pooled (131)"]
w = 0.2
for i, k in enumerate(("model", "hybrid", "keep_last", "groq")):
    vals = [per[f][k] / per[f]["n"] for f in per] + [pooled[k] / pooled["n"]]
    cnt = [per[f][k] for f in per] + [pooled[k]]
    xs = np.arange(len(labs)) + (i - 1.5) * w
    ax.bar(xs, vals, w, color=COL[k], label={"model": "model alone", "hybrid": "model + keep-last fallback", "keep_last": "keep-last", "groq": "Groq"}[k])
    for xx, v, cn in zip(xs, vals, cnt):
        ax.text(xx, v + 0.008, str(cn), ha="center", fontsize=7)
ax.set(xticks=range(len(labs)), xticklabels=labs, ylim=(0.6, 1.07), ylabel="exact keeper-set match", title="Fallback: if the model keeps nothing, keep the last take")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)
save(fig, "g4_keep_last_fallback")
num["fallback"] = {"per_fold": per, "pooled": pooled, "rule_changes_groups": changes,
                   "note": "rule chosen after seeing these results; not a clean test"}

# ---- g5: the whole run at a glance (four folds, pooled)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
names = ["4B model\n(four folds)", "model + fallback", "keep-last", "Groq"]
vals = [pooled["model"], pooled["hybrid"], pooled["keep_last"], pooled["groq"]]
ax.bar(names, [v / pooled["n"] for v in vals], color=[COL["model"], COL["hybrid"], COL["keep_last"], COL["groq"]])
for i, v in enumerate(vals):
    ax.text(i, v / pooled["n"] + 0.006, f"{v}/{pooled['n']}", ha="center", fontsize=8)
ax.set(ylim=(0.7, 0.96), ylabel="exact keeper-set match", title="131 natural groups, one model per outer fold (folds 3 and 4 were the tuning set)")
save(fig, "g5_pooled_summary")

# ---- tables
T = ["| item | value |", "|---|---|"]
f = num["final_training"]
T += [f"| training groups | {f['train_groups']} ({f['natural']} natural + {f['synthetic']} synthetic) |", f"| training decisions | {f['train_decisions']} |",
      f"| steps / time / peak memory | {f['steps']} / {f['train_seconds']/60:.0f} min / {f['peak_gb']} GB |", f"| mean training loss | {f['mean_train_loss']:.3f} |",
      f"| final evaluation loss (22 synthetic groups) | {f['final_eval_loss']:.4f} |",
      f"| synthetic dev | {num['synthetic_dev']['final model']['correct']}/{num['synthetic_dev']['final model']['n']} |"]
T += ["", "| fold | n | model | model + fallback | keep-last | Groq |", "|---|---|---|---|---|---|"]
for lab, c in per.items():
    T.append(f"| {lab} | {c['n']} | {c['model']} | {c['hybrid']} | {c['keep_last']} | {c['groq']} |")
T.append(f"| pooled | {pooled['n']} | {pooled['model']} | {pooled['hybrid']} | {pooled['keep_last']} | {pooled['groq']} |")
(OUT / "data/tables.md").write_text("\n".join(T) + "\n", encoding="utf-8")
(OUT / "data/final_numbers.json").write_text(json.dumps(num, indent=1, default=str), encoding="utf-8")
print("\n".join(T))
print(json.dumps({k: num[k] for k in ("synthetic_dev", "second_seed_fold4")}, indent=1, default=str))
print(changes)
