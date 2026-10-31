"""
Keeper-selection SFT data for Unsloth, v2 (system rules / user group / assistant answer), one directory per prompt variant.

    python training/build_keeper_sft_v2.py [--prompt-variant full|short|both] [--model Qwen/Qwen3-4B-Instruct-2507] [--with-reason]
                                            [--groups data/training/groups_v1.csv] [--out-dir data/training/sft_v2]

Rows: groups with a keeper (or keep_none == 1), no "STALE" in notes, minus video_07__g7 / video_08__g8 / video_08__g10 (merged / superseded).
Answer: {"keep": [...], "keep_none": bool, "unclear": bool}; --with-reason adds "reason" only from a filtered, cleaned source (see reason_for).
User message: the group's takes with gap_before, plus the unit before and after the group in the same video as NEIGHBOR CONTEXT
(natural rows; rebuilt from data/training/processed/<video>/units.json). Synthetic rows get "none" context lines (same template).
Files per variant: keeper_sft_fold{1..4}_{train,test}.jsonl (test k = natural rows of fold k; train k = the other natural folds + ALL synthetic rows),
keeper_sft_all_natural.jsonl (all natural + synthetic, for the final fit). Each row: {"messages": [system, user, assistant], "meta": {...}}.
Also writes summary.json (token lengths, per-fold table with the keep-last baseline, skips, filter changes).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.step_06_llm_adjudication.keeper_prompt import PROMPT_VARIANTS, build_keeper_messages  # noqa: E402

TRAIN = REPO / "data" / "training"
SKIP_GROUPS = {"video_07__g7", "video_08__g8", "video_08__g10"}
_TEXT_PREFIX = re.compile(r"^\[\d+\]\s*")
REASON_MAX = 220
GENERIC_REASON = "Chose the last complete take."
SYNTH_REASONS = {
    "identical": "Identical text; kept the last take.",
    "truncated->complete": "Kept the complete take; the others are cut off.",
    "stutter->clean": "Kept the clean take; the others stumble.",
    "paraphrase": "These are paraphrases of one line; kept the later take.",
    "continuation": "The kept units form one continued sentence.",
    "last_take_truncated": "The last take is cut off; kept the earlier complete take.",
    "mistake->corrected": "Kept the corrected take.",
    "cue_phrases": "These are cue phrases, not script content.",
}
_BANNED = re.compile(r"not in group|pipeline|verify|audio|stale|__g\d+", re.I)
_UNIT_ID = re.compile(r"\b\d+:\d+\b")


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:               # every cell stays a string: "4:0" is never parsed as a time
        return list(csv.DictReader(f))


def group_units(row: dict[str, str]) -> list[dict]:
    ids = row["unit_ids"].split("|")
    texts = [_TEXT_PREFIX.sub("", t.strip()) for t in row["texts"].split(" || ")]
    starts, ends = [float(x) for x in row["starts"].split("|")], [float(x) for x in row["ends"].split("|")]
    if not (len(ids) == len(texts) == len(starts) == len(ends)):
        raise ValueError(f"{row['group_id']}: unit_ids / texts / starts / ends differ in length")
    return [{"unit_id": i, "start": s, "end": e, "raw_text": t} for i, t, s, e in zip(ids, texts, starts, ends)]


def clean_note(note: str) -> str:
    """A hand note as a single clean sentence (candidate reason): SYNTHETIC tag and 'verify audio' removed, capitalised, full stop."""
    n = re.sub(r"^\s*SYNTHETIC\s*", "", note.strip())
    n = re.sub(r";?\s*verify audio\s*$", "", n, flags=re.I).strip(" ;")[:REASON_MAX].rstrip(" ;,")
    if not n:
        return ""
    n = n[0].upper() + n[1:]
    return n if n[-1] in ".!?" else n + "."


def reason_passes(reason: str, shown_ids: set[str]) -> bool:
    """No group ids, no 'not in group' / 'pipeline' / 'verify' / 'audio', no unit ids that are not in the prompt."""
    if not reason or _BANNED.search(reason):
        return False
    return all(u in shown_ids for u in _UNIT_ID.findall(reason))


def reason_for(row: dict[str, str], shown_ids: set[str]) -> tuple[str, str, bool]:
    """(reason, source, changed): the cleaned hand note if it passes the filter, else the generic template for the synth_type / the generic sentence.
    `changed` is True when a note-derived candidate existed but was replaced."""
    cand = clean_note(row["notes"])
    if cand and reason_passes(cand, shown_ids):
        return cand, "cleaned_note", False
    template = SYNTH_REASONS.get(row.get("synth_type", ""), GENERIC_REASON) if row["source"] == "synthetic" else GENERIC_REASON
    return template, "template", bool(cand)


def target_for(row: dict[str, str]) -> dict:
    keep = [k for k in row["keeper_unit_ids"].split("|") if k]
    keep_none = row["keep_none"].strip() == "1" and not keep
    unclear = bool(re.search(r"unclear|paraphrase", row["notes"], flags=re.I)) or row.get("synth_type") == "paraphrase"
    return {"keep": keep, "keep_none": keep_none, "unclear": unclear}


def load_video_units(video: str, cache: dict) -> list[dict] | None:
    if video not in cache:
        p = TRAIN / "processed" / video / "units.json"
        cache[video] = sorted(json.loads(p.read_text(encoding="utf-8")), key=lambda u: (u["start"], u["end"])) if p.exists() else None
    return cache[video]


def context_for(row: dict[str, str], members: list[dict], cache: dict) -> tuple[dict | None, dict | None, str]:
    """(before, after, status): the unit before the first member and after the last member in the same video. status: ok | no_units_file | member_mismatch."""
    units = load_video_units(row["video_id"], cache)
    if units is None:
        return None, None, "no_units_file"
    pos = {u["unit_id"]: i for i, u in enumerate(units)}
    if any(m["unit_id"] not in pos or units[pos[m["unit_id"]]]["raw_text"].strip() != m["raw_text"].strip() for m in members):
        return None, None, "member_mismatch"
    lo, hi = min(pos[m["unit_id"]] for m in members), max(pos[m["unit_id"]] for m in members)
    pick = lambda u: None if u is None else {"unit_id": u["unit_id"], "start": u["start"], "end": u["end"], "raw_text": u["raw_text"]}
    return pick(units[lo - 1]) if lo > 0 else None, pick(units[hi + 1]) if hi + 1 < len(units) else None, "ok"


def synth_pool(all_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    """Unrelated base sentences per synthetic video: every distinct text of the video in synthetic_pairs_v1.csv (the video's own base sentence, its
    paraphrases and hard negatives) plus the longest (most complete) take of every synthetic group of that video."""
    pool: dict[str, set[str]] = {}
    pairs_csv = REPO / "data" / "synthetic" / "synthetic_pairs_v1.csv"
    if pairs_csv.exists():
        for r in read_csv(pairs_csv):
            pool.setdefault(r["video_id"], set()).update(t.strip() for t in (r["text_a"], r["text_b"]) if t.strip())
    for r in all_rows:
        if r["source"] == "synthetic":
            pool.setdefault(r["video_id"], set()).add(max(group_units(r), key=lambda u: len(u["raw_text"]))["raw_text"].strip())
    return {v: sorted(s) for v, s in pool.items()}


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", s.lower()))


def _unrelated(text: str, members: list[dict]) -> bool:
    w = _words(text)
    return all(len(w & _words(m["raw_text"])) / max(len(w | _words(m["raw_text"])), 1) < 0.35 for m in members)


def synth_context(row: dict[str, str], members: list[dict], all_rows: list[dict[str, str]], pool: dict[str, list[str]], seed: int) -> tuple[dict | None, dict | None]:
    """A previous and a next context unit for a synthetic group: an unrelated base sentence of the same video, with a plausible length (about 0.38 s per
    word) and a gap of 0.4-1.6 s from the group, placed so it overlaps no other group of the video. A side with no room stays None."""
    rng = random.Random(f"{seed}:ctx:{row['group_id']}")
    own = {m["unit_id"] for m in members}
    others = []
    for r in all_rows:
        if r["video_id"] == row["video_id"] and r["group_id"] != row["group_id"]:
            us = group_units(r)
            others.append((min(u["start"] for u in us), max(u["end"] for u in us)))
    candidates = [s for s in pool.get(row["video_id"], []) if _unrelated(s, members)] or [s for v in sorted(pool) for s in pool[v] if _unrelated(s, members)]
    segs = sorted(int(i.split(":")[0]) for i in own)
    s0, e1 = min(m["start"] for m in members), max(m["end"] for m in members)
    chosen: list[str] = []

    def make(side: str) -> dict | None:
        if not candidates:
            return None
        text = rng.choice([c for c in candidates if c not in chosen] or candidates)
        dur = min(max(0.38 * len(text.split()) + rng.uniform(-0.3, 0.3), 1.2), 6.0)
        gap = rng.uniform(0.4, 1.6)
        for g, d in ((gap, dur), (0.4, dur), (0.4, 1.2)):                 # shrink until it fits between the neighbouring groups
            start, end = (s0 - g - d, s0 - g) if side == "before" else (e1 + g, e1 + g + d)
            if start >= 0 and not any(start < oe + 0.2 and end > os_ - 0.2 for os_, oe in others):
                chosen.append(text)
                sid = (segs[0] - 1) if side == "before" else (segs[-1] + 1)
                uid = f"{sid}:0"
                if sid < 0 or uid in own:
                    return None
                return {"unit_id": uid, "start": round(start, 2), "end": round(end, 2), "raw_text": text}
        return None

    return make("before"), make("after")


def drop_choice(group_id: str, before: dict | None, after: dict | None, seed: int, rate: float) -> str | None:
    """Per row, seeded: with probability `rate` hide ONE side of the context ("before" / "after") in the training copy of the row."""
    rng = random.Random(f"{seed}:drop:{group_id}")
    roll, side = rng.random(), rng.choice(["before", "after"])
    sides = [s for s, c in (("before", before), ("after", after)) if c]
    if roll >= rate or not sides:
        return None
    return side if side in sides else sides[0]


def context_kind(msg_user: str) -> str:
    b, a = "before: none" not in msg_user, "after: none" not in msg_user
    return "both" if b and a else "before_only" if b else "after_only" if a else "none"


def tokens_stats(rows: list[dict], tok) -> dict:
    lens = np.array([len(tok(tok.apply_chat_template(r["messages"], tokenize=False))["input_ids"]) for r in rows])
    return {"n": len(lens), "max": int(lens.max()), "mean": round(float(lens.mean()), 1), "p95": int(np.percentile(lens, 95)),
            "recommended_max_seq_length": int((lens.max() // 256 + 1) * 256)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", default=str(TRAIN / "groups_v1.csv"))
    ap.add_argument("--out-dir", default=str(TRAIN / "sft_v2"))
    ap.add_argument("--prompt-variant", choices=[*PROMPT_VARIANTS, "both"], default="both")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="tokenizer used for the length report")
    ap.add_argument("--with-reason", action="store_true", help='add a filtered "reason" to the answer')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-rate", type=float, default=0.10, help="share of rows whose TRAINING copy hides one context side (test files never drop)")
    args = ap.parse_args()
    variants = list(PROMPT_VARIANTS) if args.prompt_variant == "both" else [args.prompt_variant]

    folds = json.loads((TRAIN / "folds.json").read_text(encoding="utf-8"))
    fold_of, speaker_of = folds["folds"], folds["speaker_of_video"]
    cache: dict = {}
    skipped, no_context, filter_changed, base_rows = Counter(), [], [], []
    all_rows = read_csv(Path(args.groups))
    pool = synth_pool(all_rows)
    for row in all_rows:
        gid = row["group_id"]
        if gid in SKIP_GROUPS:
            skipped["superseded / merged (skip list)"] += 1
            continue
        if "STALE" in row["notes"]:
            skipped["STALE in notes"] += 1
            continue
        if not row["keeper_unit_ids"].strip() and row["keep_none"].strip() != "1":
            skipped["no keeper and keep_none != 1"] += 1
            continue
        members = group_units(row)
        ids = [m["unit_id"] for m in members]
        target = target_for(row)
        if any(k not in ids for k in target["keep"]):
            skipped["keeper is not a group unit"] += 1
            continue
        natural = row["source"] != "synthetic"
        if natural:
            before, after, status = context_for(row, members, cache)
            if status != "ok":
                no_context.append((gid, status))
        else:
            before, after = synth_context(row, members, all_rows, pool, args.seed)
            status = "generated"
        dropped = drop_choice(gid, before, after, args.seed, args.drop_rate)
        shown = set(ids) | {c["unit_id"] for c in (before, after) if c}
        answer = dict(target)
        reason_source = None
        if args.with_reason:
            answer["reason"], reason_source, changed = reason_for(row, shown)
            if changed:
                filter_changed.append(gid)
        base_rows.append({"row": row, "members": members, "before": before, "after": after, "answer": answer, "natural": natural,
                          "meta": {"group_id": gid, "fold": fold_of.get(row["video_id"]) if natural else None, "synthetic": not natural, "video_id": row["video_id"],
                                   "speaker": speaker_of.get(row["video_id"]) if natural else None, "synth_type": row.get("synth_type", ""), "context": status,
                                   "context_dropped": dropped, "reason_source": reason_source},
                          "train_before": None if dropped == "before" else before, "train_after": None if dropped == "after" else after})

    out = Path(args.out_dir)
    summary: dict = {"model": args.model, "with_reason": args.with_reason, "skipped": dict(skipped), "rows_used": len(base_rows),
                     "natural": sum(b["natural"] for b in base_rows), "synthetic": sum(not b["natural"] for b in base_rows),
                     "rows_without_context": [{"group_id": g, "why": w} for g, w in no_context],
                     "reason_filter_changed": len(filter_changed), "reason_filter_changed_groups": filter_changed, "variants": {}}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    for variant in variants:
        vdir = out / variant
        vdir.mkdir(parents=True, exist_ok=True)
        recs, recs_train = [], []                     # recs: real/generated context (test files, token stats); recs_train: with the seeded one-side drops
        for b in base_rows:
            answer = {"role": "assistant", "content": json.dumps(b["answer"], ensure_ascii=False)}
            full = build_keeper_messages(variant, b["meta"]["group_id"], b["members"], b["before"], b["after"], args.with_reason) + [answer]
            trn = build_keeper_messages(variant, b["meta"]["group_id"], b["members"], b["train_before"], b["train_after"], args.with_reason) + [answer]
            recs.append({"messages": full, "meta": {**b["meta"], "context_dropped": None}})
            recs_train.append({"messages": trn, "meta": b["meta"]})
        natural, synth = [r for r in recs if not r["meta"]["synthetic"]], [r for r in recs if r["meta"]["synthetic"]]
        natural_tr, synth_tr = [r for r in recs_train if not r["meta"]["synthetic"]], [r for r in recs_train if r["meta"]["synthetic"]]

        def dump(name: str, rows: list[dict]) -> None:
            with open(vdir / name, "w", encoding="utf-8", newline="\n") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        table = {}
        for k in (1, 2, 3, 4):
            test = [r for r in natural if r["meta"]["fold"] == k]
            train = [r for r in natural_tr if r["meta"]["fold"] != k] + synth_tr
            assert not {r["meta"]["group_id"] for r in test} & {r["meta"]["group_id"] for r in train}, f"fold {k}: a group is in both train and test"
            assert not any(r["meta"]["synthetic"] for r in test), f"fold {k}: synthetic row in the test file"
            sp_test, sp_train = {r["meta"]["speaker"] for r in test}, {r["meta"]["speaker"] for r in train if r["meta"]["speaker"]}
            assert not sp_test & sp_train, f"fold {k}: speaker in both train and test: {sp_test & sp_train}"
            dump(f"keeper_sft_fold{k}_train.jsonl", train)
            dump(f"keeper_sft_fold{k}_test.jsonl", test)
            last = sum(json.loads(r["messages"][-1]["content"])["keep"] == [re.findall(r"^  \[([^\]]+)\] \(", r["messages"][1]["content"], flags=re.M)[-1]] for r in test)
            table[k] = {"train_natural": len(train) - len(synth_tr), "train_synthetic": len(synth_tr), "test": len(test),
                        "test_speakers": sorted(sp_test), "keep_last_single_baseline": f"{last}/{len(test)} = {last / max(len(test), 1):.1%}"}
        dump("keeper_sft_all_natural.jsonl", natural_tr + synth_tr)
        kinds = lambda rows: dict(Counter(context_kind(r["messages"][1]["content"]) for r in rows))
        summary["variants"][variant] = {"tokens_all_rows": tokens_stats(recs, tok), "tokens_natural_rows": tokens_stats(natural, tok), "folds": table,
                                        "context_kinds": {"training_copies_natural": kinds(natural_tr), "training_copies_synthetic": kinds(synth_tr),
                                                          "test_copies_natural": kinds(natural),
                                                          "dropped_by_seed": dict(Counter(str(r["meta"]["context_dropped"]) for r in recs_train)),
                                                          "natural_rows_missing_a_real_side": sum(1 for r in natural if context_kind(r["messages"][1]["content"]) != "both"),
                                                          "synthetic_rows_with_a_side_that_had_no_room": sum(1 for r in synth if context_kind(r["messages"][1]["content"]) != "both")}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
