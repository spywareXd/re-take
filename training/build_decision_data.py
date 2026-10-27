"""
Convert the keeper-selection SFT v2 data to the Unsloth decision-model format (FastDecisionModel / DecisionTrainer).

Schema (from unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth, the Qwen3.5-4B Decision notebook, the typed-decisions /
faizaltkl/decision-model-dataset rows and unslothai/unsloth PR 12772): one row = {"state": str, "questions": JSON str, "gold": JSON str}.
    questions = {"<question id>": {"type": "noul", "instructions": "..."}}   (note: "instructions", plural; "noul" has no "criteria")
    gold      = {"<question id>": true | false}                               (noul answers are booleans)
`state` is a plain string; here it is a JSON-encoded object (the rules once, the takes, the context). `questions` and `gold` are stored as JSON strings,
as in the public dataset. Nothing but state / questions / gold is in a dataset row; group_id, fold, synthetic, speaker live in the meta files.

One row per group; one noul question per GROUP unit ("keep_<unit id>": "Should take <unit id> stay in the final video?"), none for context units.
gold: true for every unit in the labelled keeper set, false for all others (keep_none rows: all false).

The rows are converted from data/training/sft_v2/short (not modified), so the folds, the synthetic-rows-only-in-train rule, the hidden-context training
copies and the real-context test copies are exactly those of sft_v2.

    python training/build_decision_data.py [--src data/training/sft_v2/short] [--out data/training/sft_decision]
Writes decision_fold{1..4}_{train,test}.jsonl, decision_all_natural.jsonl (+ <name>.keys.txt = group_id per row, same order) and decision_meta.jsonl.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.step_06_llm_adjudication.keeper_prompt import rules_for  # noqa: E402

TRAIN = REPO / "data" / "training"
CONTEXT_MARK = "[context, not in this group]"
_TAKE = re.compile(r'^  \[([^\]]+)\] \((-?\d+\.\d+)s -> (-?\d+\.\d+)s\) gap_before=(none|\d+\.\d+): "(.*)"$')
_CTX = re.compile(r'^  (before|after): (?:none|\[context, not in this group\] \[([^\]]+)\] \((-?\d+\.\d+)s -> (-?\d+\.\d+)s\): "(.*)")$')


def parse_user(user: str) -> tuple[list[dict], dict[str, dict | None]]:
    """The SFT v2 user message -> (takes, {"before": ctx | None, "after": ctx | None}). Raises on anything it does not recognise."""
    takes, ctx = [], {}
    for line in user.splitlines():
        m = _CTX.match(line)
        if m:
            side, cid, s, e, text = m.groups()
            ctx[side] = None if cid is None else {"id": cid, "start": float(s), "end": float(e), "text": text, "note": CONTEXT_MARK}
            continue
        m = _TAKE.match(line)
        if m:
            uid, s, e, gap, text = m.groups()
            takes.append({"id": uid, "start": float(s), "end": float(e), "gap_before_s": None if gap == "none" else float(gap), "text": text})
    if not takes or set(ctx) != {"before", "after"}:
        raise ValueError("could not parse the SFT user message")
    ids_line = re.search(r"^Unit ids \(the only valid answers\): \[(.*)\]$", user, flags=re.M)
    if [t["id"] for t in takes] != re.findall(r'"([^"]+)"', ids_line.group(1)):
        raise ValueError("take lines differ from the unit-id line")
    return takes, ctx


ANSWER_LINE = "Answer each yes/no question: should this take stay in the final video?"


def decision_rules() -> str:
    """The short rules for the decision model: the JSON-output line (keep / keep_none / unclear format) is replaced by the yes/no instruction."""
    lines = rules_for("short", with_reason=False).split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith("JSON only:")]
    if len(idx) != 1:
        raise ValueError("expected exactly one 'JSON only:' line in the short rules")
    lines[idx[0]] = ANSWER_LINE
    return "\n".join(lines)


def to_row(rec: dict, rules: str) -> dict:
    takes, ctx = parse_user(rec["messages"][1]["content"])
    answer = json.loads(rec["messages"][2]["content"])
    keep = set(answer["keep"])
    state = {"rules": rules, "takes": takes, "context": ctx}
    questions = {f"keep_{t['id']}": {"type": "noul", "instructions": f"Should take {t['id']} stay in the final video?"} for t in takes}
    gold = {f"keep_{t['id']}": t["id"] in keep for t in takes}
    return {"state": json.dumps(state, ensure_ascii=False), "questions": json.dumps(questions, ensure_ascii=False), "gold": json.dumps(gold)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(TRAIN / "sft_v2" / "short"))
    ap.add_argument("--out", default=str(TRAIN / "sft_decision"))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rules = decision_rules()
    meta: dict[str, dict] = {}
    n_files = n_rows = 0
    for f in sorted(src.glob("keeper_sft_*.jsonl")):
        name = f.name.replace("keeper_sft_", "decision_")
        recs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [to_row(r, rules) for r in recs]
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
        (out / name.replace(".jsonl", ".keys.txt")).write_text("".join(r["meta"]["group_id"] + "\n" for r in recs), encoding="utf-8", newline="\n")
        for r in recs:                                     # keyed by group_id; the training copy's hidden side is recorded where it applies
            m = meta.setdefault(r["meta"]["group_id"], {k: r["meta"][k] for k in ("group_id", "fold", "synthetic", "speaker", "video_id", "synth_type")})
            m["keep_none"] = bool(json.loads(r["messages"][2]["content"])["keep_none"])
            if r["meta"].get("context_dropped"):
                m["context_dropped_in_training_copy"] = r["meta"]["context_dropped"]
        n_files += 1
        n_rows += len(rows)
    (out / "decision_meta.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in meta.values()), encoding="utf-8", newline="\n")
    print(f"wrote {n_files} dataset files ({n_rows} rows) + keys files + decision_meta.jsonl ({len(meta)} groups) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
