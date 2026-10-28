"""
Build the fixed parity set for the standalone decision-model export. ONLY reads existing files and writes to the git-ignored private folder.

    python training/standalone_export/build_parity_set.py

Output: runs/standalone_export/private/parity_set.jsonl, one JSON object per group, in a fixed order:
    {"id", "kind": "natural" | "synthetic" | "edge", "synth_type", "state": <string>, "questions": {...}, "gold": [kept unit ids] | null, "n_units"}
Order: the 131 labelled natural groups (trainprobe order), the 132nd live natural group (video_08__g8, no gold), the 220 held-out synthetic groups
(synthdev order, all 11 synth types), then the edge cases. Nothing here is random except the edge texts, which are fixed strings.
The rows have the exact shape the final model was trained on (rules prefix, takes with gap_before_s, before/after context, rich noul questions).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from training.build_decision_data import decision_rules  # noqa: E402
from training.build_decision_data_v2 import rich_questions  # noqa: E402
from training.build_keeper_sft_v2 import context_for, group_units  # noqa: E402

LOOP = REPO / "runs" / "keeper_loop" / "final_all" / "data"
OUT = REPO / "runs" / "standalone_export" / "private"
MAX_LEN = 16384          # FastDecisionModel.predict serves up to 16384 tokens


def read(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def state_of(takes_text: list[str], before: str | None = "Here is the sentence before the group.", after: str | None = "And here is the sentence after it.") -> dict:
    takes, t = [], 10.0
    for i, text in enumerate(takes_text):
        takes.append({"id": f"{i + 1}:0", "start": round(t, 2), "end": round(t + 2.0, 2), "gap_before_s": 0.8 if i else 1.2, "text": text})
        t += 3.0
    ctx = {"before": None if before is None else {"id": "0:0", "start": 5.0, "end": 8.0, "text": before, "note": "[context, not in this group]"},
           "after": None if after is None else {"id": "99:0", "start": t + 1.0, "end": t + 3.0, "text": after, "note": "[context, not in this group]"}}
    return {"rules": decision_rules(), "takes": takes, "context": ctx}


def edge_rows(tok) -> list[dict]:
    words = "the quick brown fox jumps over the lazy dog while the speaker repeats this line again and again".split()
    filler = lambda n: " ".join(words[i % len(words)] for i in range(n))
    # a state that fits 4096 tokens only just, and one that must be cut
    def long_state(n_words: int) -> dict:
        return state_of([filler(n_words), filler(n_words // 2)])
    near = None
    lo, hi = 200, 20000
    while lo < hi:                                      # largest word count whose full prompt still fits MAX_LEN
        mid = (lo + hi + 1) // 2
        s = long_state(mid)
        n = len(tok(json.dumps(s, ensure_ascii=False))["input_ids"]) + len(tok(json.dumps(rich_questions(s), ensure_ascii=False))["input_ids"]) + 400
        lo, hi = (mid, hi) if n <= MAX_LEN else (lo, mid - 1)
    near = long_state(lo)
    over = long_state(int(lo * 1.6))
    cases = {
        "edge_1_unit": state_of(["Only one take in this group, so there is nothing to choose between."]),
        "edge_2_unit": state_of(["Let me say that again.", "Let me say that one more time, properly."]),
        "edge_6_unit": state_of([f"Attempt number {i}, still not right." for i in range(1, 6)] + ["Attempt number six is the final one."]),
        "edge_12_unit": state_of([f"Take {i} of the opening line." for i in range(1, 13)]),
        "edge_near_max_len": near,
        "edge_over_max_len": over,
        "edge_empty_text": state_of(["", "   ", "A real sentence after two empty takes."]),
        "edge_odd_text": state_of(['She said "never\\ever" \U0001F600 שלום مرحبا', "tab\there\nnewline and 100% {braces} [brackets] <tags> &amp;", "…"]),
        "edge_cue_phrases": state_of(["Let's go.", "Here we go.", "Okay, here we go.", "From the top."]),
        "edge_no_context": state_of(["First take of the line.", "Second take of the line."], before=None, after=None),
        "edge_only_before": state_of(["One take.", "Another take."], after=None),
        "edge_long_single_word": state_of(["a" * 400, "Short take."]),
    }
    rows = []
    for name, s in cases.items():
        rows.append({"id": name, "kind": "edge", "synth_type": "", "state": json.dumps(s, ensure_ascii=False), "questions": rich_questions(s), "gold": None,
                     "n_units": len(s["takes"])})
    return rows


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    meta = json.loads((LOOP / "groups_meta.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    for fname, kind in (("trainprobe", "natural"), ("synthdev", "synthetic")):
        data, keys = read(LOOP / f"{fname}.jsonl"), (LOOP / f"{fname}.keys.txt").read_text().split()
        assert len(data) == len(keys)
        for r, g in zip(data, keys):
            gold = [q[len("keep_"):] for q, v in json.loads(r["gold"]).items() if v]
            rows.append({"id": g, "kind": kind, "synth_type": meta[g].get("synth_type", ""), "state": r["state"], "questions": json.loads(r["questions"]),
                         "gold": gold, "n_units": len(json.loads(r["questions"]))})
        if kind == "natural":                                              # the 132nd live natural group has no dataset row (superseded / no gold)
            with open(REPO / "data" / "training" / "groups_v1.csv", encoding="utf-8-sig", newline="") as f:
                live = [x for x in csv.DictReader(f) if x["source"] != "synthetic" and "STALE" not in x["notes"]]
            have = {r["id"] for r in rows}
            for x in live:
                if x["group_id"] not in have:
                    members = group_units(x)
                    before, after, status = context_for(x, members, {})
                    s = {"rules": decision_rules(), "context": {"before": None if before is None else {"id": before["unit_id"], "start": before["start"], "end": before["end"], "text": before["raw_text"], "note": "[context, not in this group]"},
                                                                  "after": None if after is None else {"id": after["unit_id"], "start": after["start"], "end": after["end"], "text": after["raw_text"], "note": "[context, not in this group]"}}}
                    takes, prev_end = [], before["end"] if before else None
                    for u in members:
                        takes.append({"id": u["unit_id"], "start": u["start"], "end": u["end"], "gap_before_s": None if prev_end is None else round(max(0.0, u["start"] - prev_end), 2), "text": u["raw_text"]})
                        prev_end = u["end"]
                    state = {"rules": s["rules"], "takes": takes, "context": s["context"]}
                    rows.append({"id": x["group_id"], "kind": "natural", "synth_type": "", "state": json.dumps(state, ensure_ascii=False), "questions": rich_questions(state),
                                 "gold": None, "n_units": len(takes)})
    rows += edge_rows(tok)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parity_set.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    from collections import Counter
    print("rows:", len(rows), dict(Counter(r["kind"] for r in rows)))
    print("synthetic by type:", dict(Counter(r["synth_type"] for r in rows if r["kind"] == "synthetic")))
    print("natural with gold:", sum(1 for r in rows if r["kind"] == "natural" and r["gold"] is not None), "| natural without gold:", [r["id"] for r in rows if r["kind"] == "natural" and r["gold"] is None])
    print("units per group (min/max):", min(r["n_units"] for r in rows), max(r["n_units"] for r in rows))
    for r in rows:
        if r["kind"] == "edge" and r["id"] in ("edge_near_max_len", "edge_over_max_len"):
            n = len(tok(r["state"])["input_ids"]) + len(tok(json.dumps(r["questions"]))["input_ids"])
            print(r["id"], "state+questions tokens (plain count):", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
