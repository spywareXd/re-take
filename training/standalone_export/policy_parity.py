"""
Policy-level parity: run the FULL hybrid keeper policy (cue rule, exact gate, model, empty fallback, unclear) once with the Unsloth reference scores and once
with the standalone scores, and compare the FINAL decisions. Aggregates and group ids only. Repo venv, no GPU needed.

    python training/standalone_export/policy_parity.py [--standalone standalone_scores_4bit_v2.jsonl] [--kind natural]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pipeline.step_06_llm_adjudication.keeper_policy import KeeperConfig, decide_finetuned  # noqa: E402

PRIV = REPO / "runs" / "standalone_export" / "private"
ap = argparse.ArgumentParser()
ap.add_argument("--standalone", default="standalone_scores_4bit_v2.jsonl")
ap.add_argument("--reference", default="reference_scores_run1.jsonl")
ap.add_argument("--kinds", default="natural,synthetic")
args = ap.parse_args()


def read(name):
    return [json.loads(line) for line in (PRIV / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def unit(t):
    return {"unit_id": t["id"], "start": t["start"], "end": t["end"], "raw_text": t["text"]}


rows = {r["id"]: r for r in read("parity_set.jsonl")}
scores = {"reference": {r["id"]: r for r in read(args.reference)}, "standalone": {r["id"]: r for r in read(args.standalone)}}
cfg = KeeperConfig()
results = {}
for kind in args.kinds.split(","):
    agree = Counter()
    diffs, by_layer = [], Counter()
    gold_ok = {"reference": 0, "standalone": 0}
    gold_n = 0
    for gid, row in rows.items():
        if row["kind"] != kind:
            continue
        state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
        members = [unit(t) for t in state["takes"]]
        ctx = state.get("context") or {}
        before = unit(ctx["before"]) if ctx.get("before") else None
        after = unit(ctx["after"]) if ctx.get("after") else None
        ids = [m["unit_id"] for m in members]
        decisions = {}
        for who in ("reference", "standalone"):
            s = scores[who].get(gid)
            if not s or s["error"]:
                decisions[who] = None
                continue
            probs = {f"keep_{i}": p for i, p in zip(row["unit_ids"] if "unit_ids" in row else [q[len("keep_"):] for q in row["questions"]], s["p_yes"])}
            decisions[who] = decide_finetuned(gid, members, before, after, lambda st, q, probs=probs: probs, cfg, who)
        a, b = decisions["reference"], decisions["standalone"]
        if a is None or b is None:
            agree["missing"] += 1
            continue
        by_layer[a.decided_by] += 1
        same_keep, same_status, same_layer = a.keep == b.keep, a.status == b.status, a.decided_by == b.decided_by
        agree["groups"] += 1
        agree["same_keep"] += same_keep
        agree["same_status_flag"] += same_status
        agree["same_layer"] += same_layer
        agree["identical_everything"] += same_keep and same_status and same_layer
        if not same_keep:
            diffs.append(gid)
        if kind == "natural" and row["gold"] is not None:
            gold_n += 1
            for who, d in decisions.items():
                gold_ok[who] += int(sorted(d.keep) == sorted(row["gold"]))
    results[kind] = {"groups": agree["groups"], "final_keep_agreement": f"{agree['same_keep']}/{agree['groups']}", "same_unclear_flag": f"{agree['same_status_flag']}/{agree['groups']}",
                     "same_deciding_layer": f"{agree['same_layer']}/{agree['groups']}", "identical_decisions": f"{agree['identical_everything']}/{agree['groups']}",
                     "missing": agree["missing"], "final_keep_differs_ids": diffs, "deciding_layer_counts_reference": dict(by_layer)}
    if kind == "natural":
        results[kind]["gold_exact_set_match_IN_SAMPLE"] = {"groups_with_gold": gold_n, **gold_ok}
print(json.dumps(results, indent=1))
