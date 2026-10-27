"""
Decision-model data v2: the same rows, split and gold labels as data/training/sft_decision, but with richer questions.

    python training/build_decision_data_v2.py --fold 2        # writes data/training/sft_decision_v2/fold2/

v1 asked  "Should take 12:0 stay in the final video?"  (an id only; the head had to match the id to a take through attention). v2 puts the unit's own text and
its position in the question:
    Should take 12:0 ("So the main idea is that…", take 3 of 4) stay in the final video?
text = the take's text cut to MAX_WORDS words (an ellipsis inside the quotes marks a cut); "take k of n" = its position in time order inside the group (the takes in `state`
are chronological). The question keys (keep_<id>), the type (noul), the `state` (rules prefix included) and the gold values are unchanged and are asserted to be so.
Nothing in sft_decision is modified. Also checks: gold == the keeper labels of groups_v1.csv, no synthetic text equals a real unit, no test take text appears in training.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "data" / "training"
MAX_WORDS = 40
ELLIPSIS = "…"


def truncate_words(text: str, max_words: int = MAX_WORDS) -> str:
    words = text.split()
    return text.strip() if len(words) <= max_words else " ".join(words[:max_words]) + ELLIPSIS


def rich_instruction(take: dict, k: int, n: int) -> str:
    return f'Should take {take["id"]} ("{truncate_words(take["text"])}", take {k} of {n}) stay in the final video?'


def rich_questions(state: dict) -> dict:
    takes = state["takes"]
    return {f'keep_{t["id"]}': {"type": "noul", "instructions": rich_instruction(t, i + 1, len(takes))} for i, t in enumerate(takes)}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", s.lower())).strip()


def read(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def convert(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        state = json.loads(r["state"])
        new_q = rich_questions(state)
        old_q = json.loads(r["questions"])
        assert list(new_q) == list(old_q) and all(v["type"] == "noul" for v in new_q.values()), "question keys / type changed"
        out.append({"state": r["state"], "questions": json.dumps(new_q, ensure_ascii=False), "gold": r["gold"]})   # state and gold are passed through untouched
    return out


def texts_of(rows: list[dict]) -> list[tuple[str, str]]:
    """(kind, text) for every take and context unit of the rows."""
    out = []
    for r in rows:
        s = json.loads(r["state"])
        out += [("take", t["text"]) for t in s["takes"]]
        out += [("context", c["text"]) for c in s["context"].values() if c]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--src", default=str(TRAIN / "sft_decision"))
    ap.add_argument("--out", default=None, help="default: data/training/sft_decision_v2/fold<k>")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="tokenizer for the length report")
    ap.add_argument("--groups", default=str(TRAIN / "groups_v1.csv"), help="groups CSV the gold labels are checked against")
    args = ap.parse_args()
    k = args.fold
    src = Path(args.src)
    out = Path(args.out) if args.out else TRAIN / "sft_decision_v2" / f"fold{k}"
    out.mkdir(parents=True, exist_ok=True)
    meta_all = {m["group_id"]: m for m in read(src / "decision_meta.jsonl")}
    names = {"train": f"decision_fold{k}_train.jsonl", "test": f"decision_fold{k}_test.jsonl"}
    old, new, keys = {}, {}, {}
    for split, name in names.items():
        old[split] = read(src / name)
        keys[split] = (src / name.replace(".jsonl", ".keys.txt")).read_text(encoding="utf-8").split()
        assert len(old[split]) == len(keys[split]), f"{name}: keys file does not line up"
        new[split] = convert(old[split])
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new[split]), encoding="utf-8", newline="\n")
        (out / name.replace(".jsonl", ".keys.txt")).write_text("".join(g + "\n" for g in keys[split]), encoding="utf-8", newline="\n")
    used = set(keys["train"]) | set(keys["test"])
    (out / "decision_meta.jsonl").write_text("".join(json.dumps(meta_all[g], ensure_ascii=False) + "\n" for g in sorted(used)), encoding="utf-8", newline="\n")

    problems: list[str] = []
    # 1. same split, same labels
    for split in names:
        for o, n_, g in zip(old[split], new[split], keys[split]):
            if (o["state"], o["gold"]) != (n_["state"], n_["gold"]) or list(json.loads(o["questions"])) != list(json.loads(n_["questions"])):
                problems.append(f"{split}/{g}: state, gold or question keys differ from sft_decision")
    import csv
    groups_csv = Path(args.groups)
    with open(groups_csv, encoding="utf-8-sig", newline="") as f:
        gv = {r["group_id"]: r for r in csv.DictReader(f)}
    checked = bad = 0
    for split in names:
        for r, g in zip(new[split], keys[split]):
            gold_ids = [q[len("keep_"):] for q, v in json.loads(r["gold"]).items() if v]
            lab = [x for x in gv[g]["keeper_unit_ids"].split("|") if x]
            checked += 1
            if gold_ids != lab:
                bad += 1
                problems.append(f"{split}/{g}: gold {gold_ids} != {groups_csv.name} keeper labels {lab}")
    print(f"labels: gold == {groups_csv.name} keeper_unit_ids in {checked - bad}/{checked} groups")
    if set(keys["train"]) & set(keys["test"]):
        problems.append("a group is in both train and test")

    # 2. counts per split
    nat = lambda split: sum(not meta_all[g]["synthetic"] for g in keys[split])
    print(f"counts: train {len(new['train'])} groups ({nat('train')} natural + {len(new['train']) - nat('train')} synthetic), "
          f"test {len(new['test'])} groups ({nat('test')} natural, {len(new['test']) - nat('test')} synthetic); the test file is also the validation set Studio evaluates on")

    # 3. token lengths
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"token length of state + questions ({args.model} tokenizer):")
    allmax = 0
    for split in names:
        for label, rows in (("v1 questions", old[split]), ("v2 questions", new[split])):
            lens = np.array([len(tok(r["state"] + "\n" + r["questions"])["input_ids"]) for r in rows])
            allmax = max(allmax, int(lens.max())) if label.startswith("v2") else allmax
            print(f"   {split:<5} {label}: max {lens.max()} mean {lens.mean():.0f} p95 {int(np.percentile(lens, 95))} | rows over 2048: {int((lens > 2048).sum())}")
    print(f"   recommended max_seq_length: {int((allmax // 256 + 1) * 256)} (v2 max {allmax})")

    # 4. leakage checks
    tr_texts, te_texts = texts_of(new["train"]), texts_of(new["test"])
    real_units = {norm(t) for (g, r) in ((g, r) for g, r in zip(keys["train"], new["train"]) if not meta_all[g]["synthetic"]) for _, t in texts_of([r])}
    real_units |= {norm(t) for _, t in te_texts}
    syn_takes = defaultdict(list)
    for g, r in zip(keys["train"], new["train"]):
        if meta_all[g]["synthetic"]:
            for t in json.loads(r["state"])["takes"]:
                syn_takes[norm(t["text"])].append(g)
    syn_eq = {t: gs for t, gs in syn_takes.items() if t in real_units}
    print(f"synthetic takes equal to a real unit (takes or context of the natural train rows, or of the test rows): {len(syn_eq)} of {len(syn_takes)} distinct synthetic texts")
    for t, gs in list(syn_eq.items())[:5]:
        print(f"      {t!r} in {gs[:3]}")
    train_norm = defaultdict(set)
    for g, r in zip(keys["train"], new["train"]):
        s = json.loads(r["state"])
        for t in s["takes"]:
            train_norm[norm(t["text"])].add(("take", g))
        for c in s["context"].values():
            if c:
                train_norm[norm(c["text"])].add(("context", g))
    hits = []
    for g, r in zip(keys["test"], new["test"]):
        for t in json.loads(r["state"])["takes"]:
            n_ = norm(t["text"])
            if n_ in train_norm:
                hits.append((g, t["text"], sorted(train_norm[n_])[:2], len(n_.split())))
    long_hits = [h for h in hits if h[3] >= 4]
    print(f"test take texts that also appear (exactly, after normalisation) as a take or context unit in the training rows: {len(hits)} "
          f"({len(long_hits)} with 4+ words)")
    for h in hits[:8]:
        print(f"      {h[0]}: {h[1]!r} ({h[3]} words) also in {h[2]}")
    if syn_eq:
        problems.append(f"{len(syn_eq)} synthetic text(s) equal a real unit")
    if long_hits:
        problems.append(f"{len(long_hits)} test take text(s) of 4+ words appear in training")

    # 5. three complete rows
    for i in (0, 60, 130):
        r = new["train"][i]
        print(f"\n--- row {i} of {names['train']} (group {keys['train'][i]}) ---")
        print("STATE:", json.dumps(json.loads(r["state"]), ensure_ascii=False))
        print("QUESTIONS:", json.dumps(json.loads(r["questions"]), indent=1, ensure_ascii=False))
        print("GOLD:", r["gold"])
    print("\nfiles:")
    for p in sorted(out.iterdir()):
        print("  ", p)
    if problems:
        print(f"\nFAILED: {len(problems)} problem(s)")
        for p in problems[:30]:
            print(" -", p)
        return 1
    print("\nBUILD CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
