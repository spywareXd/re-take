"""
Aggregate parity metrics: standalone scores vs the reference (original Unsloth model). Prints JSON of AGGREGATE numbers and group ids only (no text).

    python training/standalone_export/parity_report.py --mode 4bit [--tag ""]

Criteria (stated before the run): keeper agreement (argmax of p_yes over a group's units) 100% on natural groups; <= 1% disagreement on synthetic groups, all of them
near-ties (top-2 reference gap < 0.05); mean |dp| < 0.01 (default mode); no errors.
"""
import argparse
import json
from pathlib import Path

import numpy as np

PRIV = Path(__file__).resolve().parents[2] / "runs" / "standalone_export" / "private"
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--ref", default="reference_scores_run1.jsonl")
args = ap.parse_args()


def read(p):
    return [json.loads(line) for line in (PRIV / p).read_text(encoding="utf-8").splitlines() if line.strip()]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


parity = {r["id"]: r for r in read("parity_set.jsonl")}
ref = {r["id"]: r for r in read(args.ref)}
new = {r["id"]: r for r in read(f"standalone_scores_{args.mode}{args.tag}.jsonl")}
timing = json.loads((PRIV / f"standalone_timing_{args.mode}{args.tag}.json").read_text(encoding="utf-8"))
NEAR = 0.05
TIE = 1e-3
per_kind: dict = {}
all_dp, errors = [], []
for gid, n in new.items():
    r, p = ref[gid], parity[gid]
    if n["error"] or r["error"]:
        errors.append(gid)
        continue
    a, b = np.array(r["p_yes"]), np.array(n["p_yes"])
    d = np.abs(a - b)
    all_dp.extend(d.tolist())
    k = per_kind.setdefault(p["kind"], {"groups": 0, "disagree": [], "near_tie_disagree": [], "dp": [], "ref_correct": 0, "new_correct": 0, "gold_groups": 0, "sp": []})
    k["groups"] += 1
    k["dp"].extend(d.tolist())
    ia, ib = int(a.argmax()), int(b.argmax())
    # tie-aware: p_yes saturates at 1.0, so a "different" argmax between equal-probability units is not a different decision. Disagreement = the standalone's pick
    # has a reference p_yes more than TIE below the reference's best. The strict argmax count is reported as well.
    k["strict"] = k.get("strict", 0) + int(ia != ib)
    if a[ib] < a.max() - TIE:
        k["disagree"].append(gid)
        s = np.sort(a)[::-1]
        if len(s) > 1 and s[0] - s[1] < NEAR:
            k["near_tie_disagree"].append(gid)
    k["sp"].append(spearman(a, b))
    if p["gold"] is not None and p["kind"] == "natural":
        ids = [q[len("keep_"):] for q in p["questions"]]
        gold = set(p["gold"])
        k["gold_groups"] += 1
        k["ref_correct"] += int(ids[ia] in gold)
        k["new_correct"] += int(ids[ib] in gold)
res = {"mode": args.mode, "errors": errors, "overall": {"units": len(all_dp), "max_dp": float(np.max(all_dp)), "mean_dp": float(np.mean(all_dp)), "p99_dp": float(np.percentile(all_dp, 99))},
       "by_kind": {}, "timing": timing}
for kind, k in per_kind.items():
    dp = np.array(k["dp"])
    res["by_kind"][kind] = {"groups": k["groups"], "keeper_disagree": len(k["disagree"]), "strict_argmax_differs": k.get("strict", 0), "disagree_ids": k["disagree"], "near_tie_disagree": len(k["near_tie_disagree"]),
                            "max_dp": float(dp.max()), "mean_dp": float(dp.mean()), "p99_dp": float(np.percentile(dp, 99)),
                            "spearman_mean": float(np.nanmean(k["sp"])), "gold_groups": k["gold_groups"], "ref_keeper_correct": k["ref_correct"], "new_keeper_correct": k["new_correct"]}
nat, syn = res["by_kind"].get("natural"), res["by_kind"].get("synthetic")
checks = {"no_errors": not errors, "natural_agreement_100": bool(nat and nat["keeper_disagree"] == 0),
          "synthetic_le_1pct_all_near_ties": bool(syn and syn["keeper_disagree"] <= 0.01 * syn["groups"] and syn["near_tie_disagree"] == syn["keeper_disagree"]),
          "mean_dp_lt_0.01": res["overall"]["mean_dp"] < 0.01}
res["checks"], res["PASS"] = checks, all(checks.values())
print(json.dumps(res, indent=1))
