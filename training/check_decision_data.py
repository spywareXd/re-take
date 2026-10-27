"""
One-shot checks + report for the decision-model files in data/training/sft_decision (run once after build_decision_data.py).

    python training/check_decision_data.py [--dir data/training/sft_decision] [--model Qwen/Qwen3.5-4B]
Token length = tokens of the model input that the data defines: state + questions (the gold is the label, not input). This is an approximation of the
sequence the DecisionTrainer builds from them (its own template/special tokens are not reproduced).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from training.decision_eval import gold_keep  # noqa: E402


def read(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_row(row: dict, where: str, rich: bool = False) -> list[str]:
    errs = []
    if set(row) != {"state", "questions", "gold"}:
        return [f"{where}: columns {sorted(row)} != state/questions/gold"]
    if not all(isinstance(row[k], str) for k in row):
        return [f"{where}: state / questions / gold must be strings"]
    state, questions, gold = json.loads(row["state"]), json.loads(row["questions"]), json.loads(row["gold"])
    take_ids = [t["id"] for t in state["takes"]]
    ctx_ids = {c["id"] for c in state["context"].values() if c}
    if list(gold) != [f"keep_{u}" for u in take_ids] or list(questions) != list(gold):
        errs.append(f"{where}: questions / gold keys {list(gold)} are not exactly one per unit {take_ids}")
    if any(f"keep_{c}" in gold or f"keep_{c}" in questions for c in ctx_ids):
        errs.append(f"{where}: a context id has a question or a gold entry")
    takes = state["takes"]
    for i, (qid, q) in enumerate(questions.items()):
        if rich:                                                          # v2 questions: the unit's own text and "take k of n"
            from training.build_decision_data_v2 import rich_instruction
            want = {"type": "noul", "instructions": rich_instruction(takes[i], i + 1, len(takes))} if i < len(takes) else None
        else:
            want = {"type": "noul", "instructions": f"Should take {qid[len('keep_'):]} stay in the final video?"}
        if q != want or qid != f"keep_{takes[i]['id']}":
            errs.append(f"{where}: unexpected question {qid}: {q}")
    rules = state["rules"]
    if "JSON only" in rules or '"reason"' in rules or "keep_none" in rules.split("Answer each yes/no question")[0].splitlines()[1:2]:
        errs.append(f"{where}: the rules still contain the JSON-output instruction")
    if "Answer each yes/no question: should this take stay in the final video?" not in rules:
        errs.append(f"{where}: the rules lack the yes/no instruction")
    if not all(isinstance(v, bool) for v in gold.values()):
        errs.append(f"{where}: gold values must be booleans")
    if not any(c.get("note") == "[context, not in this group]" for c in state["context"].values() if c) and any(state["context"].values()):
        errs.append(f"{where}: a context unit lacks the '[context, not in this group]' marker")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "data" / "training" / "sft_decision"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4], help="folds present in --dir")
    ap.add_argument("--rich-questions", action="store_true", help="expect the v2 questions (unit text + position) instead of the id-only questions")
    args = ap.parse_args()
    d = Path(args.dir)
    meta = {m["group_id"]: m for m in read(d / "decision_meta.jsonl")}
    errors: list[str] = []
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    def keys(name: str) -> list[str]:
        return (d / name.replace(".jsonl", ".keys.txt")).read_text(encoding="utf-8").split()

    print(f"{'fold':<5}{'train (nat+syn)':<18}{'test':<6}{'keep_none':<11}{'questions/row mean':<20}{'max':<5}{'tokens max/mean/p95':<22}{'keep-last baseline (test)'}")
    all_lens, all_q = [], []
    for k in args.folds:
        tr_name, te_name = f"decision_fold{k}_train.jsonl", f"decision_fold{k}_test.jsonl"
        tr, te = read(d / tr_name), read(d / te_name)
        tr_k, te_k = keys(tr_name), keys(te_name)
        if len(tr) != len(tr_k) or len(te) != len(te_k):
            errors.append(f"fold {k}: keys files do not line up with the rows")
        for name, rows, ks in ((tr_name, tr, tr_k), (te_name, te, te_k)):
            for i, (r, g) in enumerate(zip(rows, ks)):
                errors += check_row(r, f"{name}[{i}] {g}", args.rich_questions)
        if set(tr_k) & set(te_k):
            errors.append(f"fold {k}: a group is in both train and test")
        if any(meta[g]["synthetic"] for g in te_k):
            errors.append(f"fold {k}: synthetic row in the test file")
        sp_te, sp_tr = {meta[g]["speaker"] for g in te_k}, {meta[g]["speaker"] for g in tr_k if meta[g]["speaker"]}
        if sp_te & sp_tr:
            errors.append(f"fold {k}: speaker in both train and test: {sp_te & sp_tr}")
        if any(meta[g]["fold"] != k for g in te_k) or any(meta[g]["fold"] == k for g in tr_k):
            errors.append(f"fold {k}: fold labels inconsistent")
        lens = np.array([len(tok(r["state"] + "\n" + r["questions"])["input_ids"]) for r in tr + te])
        qn = np.array([len(json.loads(r["questions"])) for r in tr + te])
        all_lens.append(lens)
        all_q.append(qn)
        last = sum(gold_keep(r["gold"]) == [json.loads(r["state"])["takes"][-1]["id"]] for r in te)
        n_syn = sum(meta[g]["synthetic"] for g in tr_k)
        tokens = f"{lens.max()}/{lens.mean():.0f}/{int(np.percentile(lens, 95))}"
        n_none = sum(meta[g]["keep_none"] for g in te_k)
        print(f"{k:<5}{f'{len(tr_k) - n_syn} + {n_syn}':<18}{len(te):<6}{n_none:<11}{f'{qn.mean():.2f}':<20}{qn.max():<5}{tokens:<22}{last}/{len(te)} = {last / max(len(te), 1):.1%}")
    # per-row rules over EVERY file: non-keep_none rows have at least one True, keep_none rows have none
    for f in sorted(d.glob("decision_*.jsonl")):
        if f.name == "decision_meta.jsonl":
            continue
        for r, g in zip(read(f), keys(f.name)):
            gk = gold_keep(r["gold"])
            if meta[g]["keep_none"] and gk:
                errors.append(f"{f.name}/{g}: keep_none row has a True in gold")
            if not meta[g]["keep_none"] and not gk:
                errors.append(f"{f.name}/{g}: no True in gold but the row is not keep_none")
    lens, qn = np.concatenate(all_lens), np.concatenate(all_q)
    print(f"\noverall (every train/test row, one count per appearance): tokens max {lens.max()} mean {lens.mean():.0f} p95 {int(np.percentile(lens, 95))}"
          f" -> recommended max_seq_length {int((lens.max() // 256 + 1) * 256)} (notebook default 2048); questions per row mean {qn.mean():.2f} max {qn.max()}")
    src_name = "decision_all_natural.jsonl" if (d / "decision_all_natural.jsonl").exists() else f"decision_fold{args.folds[0]}_train.jsonl"
    nat = read(d / src_name)
    nk = keys(src_name)
    ex_nat = next(r for r, g in zip(nat, nk) if not meta[g]["synthetic"] and all(json.loads(r["state"])["context"].values()))
    ex_syn = next(r for r, g in zip(nat, nk) if meta[g]["synthetic"] and all(json.loads(r["state"])["context"].values()))
    for title, r in (("NATURAL", ex_nat), ("SYNTHETIC", ex_syn)):
        print(f"\n{'=' * 90}\nEXAMPLE ({title})\n{'=' * 90}")
        for key in ("state", "questions", "gold"):
            print(f"--- {key} ---\n{json.dumps(json.loads(r[key]), indent=2, ensure_ascii=False)}")
    if errors:
        print(f"\nFAILED: {len(errors)} problem(s)")
        for e in errors[:40]:
            print(" -", e)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
