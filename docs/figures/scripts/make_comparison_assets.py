"""
Hybrid keeper (fine-tuned model + rule policy) vs the model alone vs Groq gpt-oss-120b vs keep-last, on the 131 labelled natural groups.
Reads data/policy_scores.json (written when the policy was scored on the stored fold-model probabilities) and the saved Groq answers. No model or API calls.

    python docs/finetune_final/make_comparison_assets.py

Inputs : docs/finetune_final/data/policy_scores.json, runs/keeper_loop/baselines/groq_results.jsonl and scripts/common.py (git-ignored),
         runs/e2e_sample_b_{hybrid,gptoss}_sentences/summary.json + keeper_decisions.json
Outputs: docs/finetune_final/figures/h1..h4 (png + pdf), data/comparison.json, data/comparison_tables.md
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOOP = REPO / "runs" / "keeper_loop"
OUT = REPO / "docs" / "finetune_final"
sys.path.insert(0, str(LOOP / "scripts"))
import common as C  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
SYS = ["hybrid", "model", "groq", "keep_last", "rules"]
NAME = {"hybrid": "Hybrid (model + rules)", "model": "Fine-tuned model alone", "groq": "Groq gpt-oss-120b", "keep_last": "keep-last",
        "rules": "Rules only"}
COL = {"hybrid": "#2ca02c", "model": "#1f77b4", "groq": "#d62728", "keep_last": "#7f7f7f", "rules": "#bcbd22"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{name}.png")
    fig.savefig(OUT / "figures" / f"{name}.pdf")
    plt.close(fig)


pol = json.loads((OUT / "data" / "policy_scores.json").read_text(encoding="utf-8"))
rows = {r["group_id"]: r for r in C.load_groups()}
groq = {d["group_id"]: d for d in map(json.loads, open(LOOP / "baselines" / "groq_results.jsonl", encoding="utf-8"))}
G = []
for p in pol["per_group"]:
    r = rows[p["group"]]
    gold, ids = C.gold_keep(r), C.unit_ids(r)
    q = groq[p["group"]]
    G.append({"group": p["group"], "fold": p["fold"], "kind": C.group_kind(r), "n_units": len(ids), "decided_by": p["decided_by"], "status": p["status"],
              "hybrid": p["policy_ok"], "model": p["model_ok"], "rules": p["rules_ok"],
              "groq": q["status"] == "resolved" and C.exact_match(q["keep"], gold), "keep_last": C.exact_match([ids[-1]], gold)})
N = len(G)
tot = {s: sum(g[s] for g in G) for s in SYS}

# ---- h1: overall accuracy
fig, ax = plt.subplots(figsize=(6.5, 3.8))
h1 = [("hybrid", "Hybrid\n(fine-tuned Qwen3-4B + rules)"), ("keep_last", "keep-last"), ("groq", "gpt-oss-120b")]
ax.bar([lab for _, lab in h1], [tot[s] / N for s, _ in h1], color=[COL[s] for s, _ in h1], width=0.55)
for i, (s, _) in enumerate(h1):
    ax.text(i, tot[s] / N + 0.005, f"{tot[s]}/{N}", ha="center", fontsize=8)
ax.set(ylim=(0.75, 0.95), ylabel="exact keeper-set match", title="131 labelled natural groups (one model per outer fold)")
save(fig, "h1_overall")
order = sorted(SYS, key=lambda s: -tot[s])                                    # all five systems stay in the tables

# ---- h2: by fold
folds = sorted({g["fold"] for g in G})
fig, ax = plt.subplots(figsize=(9, 3.9))
w = 0.2
four = ["hybrid", "model", "groq", "keep_last"]
for i, s in enumerate(four):
    xs = np.arange(len(folds)) + (i - 1.5) * w
    vals = [(sum(g[s] for g in G if g["fold"] == f), sum(1 for g in G if g["fold"] == f)) for f in folds]
    ax.bar(xs, [a / b for a, b in vals], w, color=COL[s], label=NAME[s])
    for x, (a, b) in zip(xs, vals):
        ax.text(x, a / b + 0.008, str(a), ha="center", fontsize=7)
ax.set(xticks=range(len(folds)), xticklabels=[f"fold {f} (n={sum(1 for g in G if g['fold'] == f)})" + (" clean" if f in (1, 2) else " tuned on") for f in folds],
       ylim=(0.6, 1.06), ylabel="exact keeper-set match", title="By outer fold (folds 3 and 4 were the tuning set)")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
save(fig, "h2_by_fold")

# ---- h3: nuanced vs keep-last groups
sl = [("Nuanced groups\n(keeper not last)", lambda g: g["kind"] != "last"), ("Keep-last groups", lambda g: g["kind"] == "last")]
fig, ax = plt.subplots(figsize=(8, 3.9))
for i, s in enumerate(four):
    xs = np.arange(2) + (i - 1.5) * w
    vals = [(sum(g[s] for g in G if p(g)), sum(1 for g in G if p(g))) for _, p in sl]
    ax.bar(xs, [a / b for a, b in vals], w, color=COL[s], label=NAME[s])
    for x, (a, b) in zip(xs, vals):
        ax.text(x, a / b + 0.015, f"{a}/{b}", ha="center", fontsize=7)
ax.set(xticks=range(2), xticklabels=[n for n, _ in sl], ylim=(0, 1.15), ylabel="exact keeper-set match", title="Where the systems differ")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
save(fig, "h3_nuanced_vs_keeplast")

# ---- h4: paired bootstrap intervals, hybrid minus each comparator
ci = []
for o in ("model", "groq", "keep_last"):
    d, lo, hi = C.bootstrap_diff([g["hybrid"] for g in G], [g[o] for g in G])
    both = sum(g["hybrid"] and g[o] for g in G)
    ci.append({"versus": o, "diff_points": 100 * d, "lo": 100 * lo, "hi": 100 * hi, "hybrid_only": sum(g["hybrid"] and not g[o] for g in G),
               "other_only": sum(g[o] and not g["hybrid"] for g in G), "both_right": both})
fig, ax = plt.subplots(figsize=(7.5, 2.9))
for i, r in enumerate(ci):
    ax.errorbar(r["diff_points"], i, xerr=[[r["diff_points"] - r["lo"]], [r["hi"] - r["diff_points"]]], fmt="o", color=COL[r["versus"]], capsize=3)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(range(len(ci)))
ax.set_yticklabels([f"hybrid vs {NAME[r['versus']]}" for r in ci])
ax.invert_yaxis()
ax.set(xlabel="accuracy difference (points), paired bootstrap 95% interval, 131 groups", title="Intervals that cross zero show no reliable difference")
save(fig, "h4_paired_intervals")

# ---- tables
dec = Counter(g["decided_by"] for g in G)
dec_ok = {k: sum(g["hybrid"] for g in G if g["decided_by"] == k) for k in dec}
T = ["| system | correct of 131 | %% | nuanced (of %d) | keep-last groups (of %d) |" % (sum(g["kind"] != "last" for g in G), sum(g["kind"] == "last" for g in G)),
     "|---|---|---|---|---|"]
for s in order:
    T.append(f"| {NAME[s]} | {tot[s]} | {100 * tot[s] / N:.1f} | {sum(g[s] for g in G if g['kind'] != 'last')} | {sum(g[s] for g in G if g['kind'] == 'last')} |")
T += ["", "| fold | n | hybrid | model | Groq | keep-last | rules only |", "|---|---|---|---|---|---|---|"]
for f in folds:
    sub = [g for g in G if g["fold"] == f]
    T.append(f"| {f}{' (clean)' if f in (1, 2) else ' (tuned on)'} | {len(sub)} | " + " | ".join(str(sum(g[s] for g in sub)) for s in ("hybrid", "model", "groq", "keep_last", "rules")) + " |")
T += ["", "| comparison | difference (points) | 95% interval | groups only hybrid right | groups only comparator right |", "|---|---|---|---|---|"]
for r in ci:
    T.append(f"| hybrid vs {NAME[r['versus']]} | {r['diff_points']:+.1f} | [{r['lo']:+.1f}, {r['hi']:+.1f}] | {r['hybrid_only']} | {r['other_only']} |")
T += ["", "| hybrid layer | groups | correct |", "|---|---|---|"]
for k, v in dec.most_common():
    T.append(f"| {k} | {v} | {dec_ok[k]} |")
diff_groq = [g for g in G if g["hybrid"] != g["groq"]]
T += ["", "| group | fold | kind | hybrid | Groq | decided by |", "|---|---|---|---|---|---|"]
for g in sorted(diff_groq, key=lambda g: (g["fold"], g["group"])):
    T.append(f"| {g['group']} | {g['fold']} | {g['kind']} | {'right' if g['hybrid'] else 'wrong'} | {'right' if g['groq'] else 'wrong'} | {g['decided_by']} |")

# ---- e2e sample_b (unlabelled)
e2e = {}
for name, d in (("hybrid", "e2e_sample_b_hybrid_sentences"), ("gpt-oss", "e2e_sample_b_gptoss_sentences")):
    p = REPO / "runs" / d
    if (p / "summary.json").exists():
        e2e[name] = {"summary": json.loads((p / "summary.json").read_text(encoding="utf-8")),
                     "meta": {k: v for k, v in json.loads((p / "run_meta.json").read_text(encoding="utf-8")).items() if k in ("stage_timings", "external_connections")},
                     "keep": {x["span_id"]: x["keep"] for x in json.loads((p / "keeper_decisions.json").read_text(encoding="utf-8"))}}
if len(e2e) == 2:
    agree = sum(e2e["hybrid"]["keep"][k] == e2e["gpt-oss"]["keep"].get(k) for k in e2e["hybrid"]["keep"])
    e2e["agreement"] = {"agree": agree, "groups": len(e2e["hybrid"]["keep"]),
                        "differ": [k for k in e2e["hybrid"]["keep"] if e2e["hybrid"]["keep"][k] != e2e["gpt-oss"]["keep"].get(k)]}

(OUT / "data" / "comparison_tables.md").write_text("\n".join(T) + "\n", encoding="utf-8")
(OUT / "data" / "comparison.json").write_text(json.dumps({"n": N, "totals": tot, "intervals": ci, "decided_by": dec, "decided_by_correct": dec_ok,
                                                         "per_group": G, "e2e_sample_b": e2e}, indent=1, default=str), encoding="utf-8")
print("\n".join(T))
print(json.dumps(e2e.get("agreement"), indent=1))
