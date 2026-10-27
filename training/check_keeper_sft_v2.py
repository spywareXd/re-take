"""
One-shot validity check of the v2 keeper SFT files (data/training/sft_v2/<variant>/...). Exits non-zero on the first kind of failure it finds
after listing them all. Prints (and writes to examples.txt) two full example rows per variant: a natural row with context and a synthetic row.

    python training/check_keeper_sft_v2.py [--dir data/training/sft_v2]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.step_06_llm_adjudication.keeper_prompt import CONTEXT_NOTICE, rules_for  # noqa: E402

KEYS = {"keep", "keep_none", "unclear"}
UNIT_IDS = re.compile(r"^Unit ids \(the only valid answers\): \[(.*)\]$", re.M)
CTX_ID = re.compile(r"\[context, not in this group\] \[([^\]]+)\]")
TAKE_LINE = re.compile(r"^  \[([^\]]+)\] \(.*\) gap_before=(none|\d+\.\d\d):", re.M)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_row(r: dict, variant: str, with_reason: bool, where: str) -> list[str]:
    errs = []
    m = r["messages"]
    if [x["role"] for x in m] != ["system", "user", "assistant"]:
        return [f"{where}: roles are {[x['role'] for x in m]}"]
    if m[0]["content"] != rules_for(variant, with_reason):
        errs.append(f"{where}: system message is not the {variant} rules")
    user = m[1]["content"]
    ids_m = UNIT_IDS.search(user)
    if not ids_m or CONTEXT_NOTICE not in user or "Return ONLY the JSON object." not in user:
        return errs + [f"{where}: user message is missing the unit-id line / context notice / closing line"]
    group_ids = re.findall(r'"([^"]+)"', ids_m.group(1))
    ctx_ids = set(CTX_ID.findall(user))
    takes = TAKE_LINE.findall(user)
    if [t[0] for t in takes] != group_ids:
        errs.append(f"{where}: take lines {[t[0] for t in takes]} differ from the unit-id line {group_ids}")
    if ctx_ids & set(group_ids):
        errs.append(f"{where}: a context id is also a group id: {ctx_ids & set(group_ids)}")
    if "before:" not in user or "after:" not in user:
        errs.append(f"{where}: before/after context lines missing (a 'none' line is required when absent)")
    try:
        a = json.loads(m[2]["content"])
    except ValueError as exc:
        return errs + [f"{where}: assistant JSON does not parse ({exc})"]
    want = KEYS | ({"reason"} if with_reason else set())
    if set(a) != want:
        errs.append(f"{where}: assistant keys {sorted(a)} != {sorted(want)}")
        return errs
    if not isinstance(a["keep"], list) or not all(isinstance(k, str) for k in a["keep"]):
        errs.append(f"{where}: keep must be a list of strings")
        return errs
    if not isinstance(a["keep_none"], bool) or not isinstance(a["unclear"], bool):
        errs.append(f"{where}: keep_none / unclear must be booleans")
    bad = [k for k in a["keep"] if k not in group_ids]
    if bad:
        errs.append(f"{where}: keep ids {bad} are not group ids" + (" (context ids!)" if set(bad) & ctx_ids else ""))
    if len(set(a["keep"])) != len(a["keep"]):
        errs.append(f"{where}: duplicate ids in keep")
    if a["keep_none"] and a["keep"] != []:
        errs.append(f"{where}: keep_none is true but keep is not empty")
    if not a["keep_none"] and not a["keep"]:
        errs.append(f"{where}: keep is empty but keep_none is false")
    if with_reason and not (isinstance(a["reason"], str) and a["reason"].strip()):
        errs.append(f"{where}: empty reason")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "data" / "training" / "sft_v2"))
    args = ap.parse_args()
    root = Path(args.dir)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    with_reason = bool(summary["with_reason"])
    errors: list[str] = []
    examples: list[str] = []
    for variant in sorted(p.name for p in root.iterdir() if p.is_dir()):
        vdir = root / variant
        n_rows = 0
        for f in sorted(vdir.glob("*.jsonl")):
            rows = read(f)
            n_rows += len(rows)
            gids = [r["meta"]["group_id"] for r in rows]
            if len(gids) != len(set(gids)):
                errors.append(f"{variant}/{f.name}: duplicate group_id")
            for r in rows:
                errors += check_row(r, variant, with_reason, f"{variant}/{f.name}/{r['meta']['group_id']}")
            if f.name.endswith("_test.jsonl") and any(r["meta"]["synthetic"] for r in rows):
                errors.append(f"{variant}/{f.name}: synthetic row in a test file")
        for k in (1, 2, 3, 4):
            tr, te = read(vdir / f"keeper_sft_fold{k}_train.jsonl"), read(vdir / f"keeper_sft_fold{k}_test.jsonl")
            if {r["meta"]["group_id"] for r in tr} & {r["meta"]["group_id"] for r in te}:
                errors.append(f"{variant} fold {k}: a group is in both train and test")
            sp_te, sp_tr = {r["meta"]["speaker"] for r in te}, {r["meta"]["speaker"] for r in tr if r["meta"]["speaker"]}
            if sp_te & sp_tr:
                errors.append(f"{variant} fold {k}: speaker in both train and test: {sp_te & sp_tr}")
            if any(r["meta"]["fold"] != k for r in te) or any(r["meta"]["fold"] == k for r in tr):
                errors.append(f"{variant} fold {k}: fold labels are inconsistent")
        allrows = read(vdir / "keeper_sft_all_natural.jsonl")
        nat = next(r for r in allrows if not r["meta"]["synthetic"] and "[context, not in this group]" in r["messages"][1]["content"])
        syn = next(r for r in allrows if r["meta"]["synthetic"])
        for r in (nat, syn):
            examples.append(f"{'=' * 100}\nVARIANT {variant} | {r['meta']['group_id']} | synthetic={r['meta']['synthetic']} | meta={json.dumps(r['meta'])}\n{'=' * 100}\n" +
                            "\n".join(f"--- {x['role'].upper()} ---\n{x['content']}" for x in r["messages"]))
        print(f"{variant}: {n_rows} rows checked in {len(list(vdir.glob('*.jsonl')))} files")
    (root / "examples.txt").write_text("\n\n".join(examples), encoding="utf-8")
    print("\n\n".join(examples))
    if errors:
        print(f"\nFAILED: {len(errors)} problem(s)")
        for e in errors[:50]:
            print(" -", e)
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
