"""
Build the supervised fine-tuning dataset for the keeper-selection model (the replacement for the Gemini keeper step) from the hand-labelled
groups_v1.csv.

One record per labelled retake group, in the chat "messages" JSONL format that TRL / Unsloth / Axolotl read directly:
    user      = build_keeper_prompt(...)  (pipeline/step_06_llm_adjudication/keeper_prompt.py: the keeper rules, then the group's takes)
    assistant = {"keep": [...], "keep_none": bool, "unclear": bool, "reason": "<one sentence>"}
plus metadata (group_id, video_id, speaker, fold, split, synthetic, n_units, reason_source).

Targets come from the hand labels:
    keep       <- keeper_unit_ids (one or several ids; several = continuation, rule 5)
    keep_none  <- keep_none == 1 (cue phrases, rule 7)
    unclear    <- the group's notes say "unclear" or "paraphrase" (rules 4, 8, 9)
    reason     <- the hand note when there is one (cleaned to one sentence), otherwise a neutral template that only describes the label
                  (reason_source = "hand_note" | "template"). The labels carry no model-written reasoning.
Excluded (written to keeper_sft_excluded.jsonl with a reason): STALE groups and groups with no keeper / keep_none label yet.
Split: speaker-grouped, reusing data/training/folds.json; the validation fold is a whole fold of real videos, synthetic groups are always training.
The prompt has no score evidence (rule 9: judge the text only) and no neighbour context.
Run:  python training/build_keeper_sft.py   (writes data/training/sft/)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.step_06_llm_adjudication.keeper_prompt import build_keeper_prompt  # noqa: E402

TRAIN = REPO / "data" / "training"
DEFAULT_OUT = TRAIN / "sft"
_TEXT_PREFIX = re.compile(r"^\[\d+\]\s*")
REASON_MAX = 220


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def group_units(row: dict[str, str]) -> list[dict]:
    ids = row["unit_ids"].split("|")
    texts = [_TEXT_PREFIX.sub("", t.strip()) for t in row["texts"].split(" || ")]
    starts = [float(x) for x in row["starts"].split("|")]
    ends = [float(x) for x in row["ends"].split("|")]
    if not (len(ids) == len(texts) == len(starts) == len(ends)):
        raise ValueError(f"{row['group_id']}: unit_ids / texts / starts / ends differ in length")
    return [{"unit_id": i, "start": s, "end": e, "raw_text": t} for i, t, s, e in zip(ids, texts, starts, ends)]


def clean_reason(note: str) -> str:
    """One sentence from a hand note: drop the SYNTHETIC tag and 'verify audio' reminders, capitalise, end with a full stop."""
    n = re.sub(r"^\s*SYNTHETIC\s*", "", note.strip())
    n = re.sub(r";?\s*verify audio\s*$", "", n, flags=re.I).strip(" ;")
    n = n[:REASON_MAX].rstrip(" ;,")
    if not n:
        return ""
    n = n[0].upper() + n[1:]
    return n if n[-1] in ".!?" else n + "."


def template_reason(keep: list[str], keep_none: bool, ids: list[str]) -> str:
    if keep_none:
        return "These are cue phrases, not script content."
    if len(keep) > 1:
        return "The kept units form one continued sentence."
    return "Keeping the last take." if keep[0] == ids[-1] else "Keeping an earlier take."


def target_for(row: dict[str, str], ids: list[str]) -> tuple[dict, str]:
    keep = [k for k in row["keeper_unit_ids"].split("|") if k]
    keep_none = bool(row["keep_none"].strip()) and not keep
    note = row["notes"]
    reason = clean_reason(note)
    source = "hand_note"
    if not reason:
        reason, source = template_reason(keep, keep_none, ids), "template"
    unclear = bool(re.search(r"unclear|paraphrase", note, flags=re.I))
    return {"keep": keep, "keep_none": keep_none, "unclear": unclear, "reason": reason}, source


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", default=str(TRAIN / "groups_v1.csv"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--val-fold", type=int, default=None, help="fold used for validation (default: the fold whose real groups are closest to 20%%)")
    args = ap.parse_args()

    folds = json.loads((TRAIN / "folds.json").read_text(encoding="utf-8"))
    fold_of, synthetic = folds["folds"], set(folds["synthetic_videos"])

    usable, excluded = [], []
    for row in read_csv(Path(args.groups)):
        has_label = bool(row["keeper_unit_ids"].strip()) or bool(row["keep_none"].strip())
        reason = "stale group (unit set no longer matches the pair labels)" if "STALE" in row["notes"] else (None if has_label else "no keeper labelled yet")
        units = group_units(row)
        ids = [u["unit_id"] for u in units]
        if reason is None and any(k not in ids for k in row["keeper_unit_ids"].split("|") if k):
            reason = "a keeper is not one of the group's units"
        if reason:
            excluded.append({"group_id": row["group_id"], "video_id": row["video_id"], "reason": reason, "keeper_unit_ids": row["keeper_unit_ids"], "notes": row["notes"]})
            continue
        target, source = target_for(row, ids)
        vid = row["video_id"]
        usable.append({
            "messages": [{"role": "user", "content": build_keeper_prompt(row["group_id"], units)},
                         {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}],
            "group_id": row["group_id"], "video_id": vid, "speaker": folds["speaker_of_video"].get(vid), "fold": fold_of.get(vid),
            "synthetic": vid in synthetic, "n_units": len(ids), "reason_source": source,
        })

    real_by_fold = Counter(r["fold"] for r in usable if not r["synthetic"])
    n_real = sum(real_by_fold.values())
    val_fold = args.val_fold if args.val_fold is not None else min(real_by_fold, key=lambda f: abs(real_by_fold[f] / n_real - 0.20))
    for r in usable:
        r["split"] = "val" if (not r["synthetic"] and r["fold"] == val_fold) else "train"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, rows: list[dict]) -> None:
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump("keeper_sft_all.jsonl", usable)
    dump("keeper_sft_train.jsonl", [r for r in usable if r["split"] == "train"])
    dump("keeper_sft_val.jsonl", [r for r in usable if r["split"] == "val"])
    dump("keeper_sft_excluded.jsonl", excluded)
    tg = [json.loads(r["messages"][1]["content"]) for r in usable]
    ids_of = [re.findall(r'^  \[([^\]]+)\]', r["messages"][0]["content"], flags=re.M) for r in usable]
    stats = {
        "usable": len(usable), "train": sum(r["split"] == "train" for r in usable), "val": sum(r["split"] == "val" for r in usable),
        "train_synthetic": sum(r["synthetic"] for r in usable), "val_fold": val_fold,
        "val_videos": sorted({r["video_id"] for r in usable if r["split"] == "val"}), "val_speakers": sorted({r["speaker"] for r in usable if r["split"] == "val"}),
        "excluded": dict(Counter(e["reason"] for e in excluded)),
        "targets": {"single_keeper": sum(len(t["keep"]) == 1 for t in tg), "multi_keeper": sum(len(t["keep"]) > 1 for t in tg),
                    "keep_none": sum(t["keep_none"] for t in tg), "unclear": sum(t["unclear"] for t in tg)},
        "reason_source": dict(Counter(r["reason_source"] for r in usable)),
        "single_keeper_is_last_unit": f'{sum(t["keep"] == [i[-1]] for t, i in zip(tg, ids_of))} of {sum(len(t["keep"]) == 1 for t in tg)}',
    }
    (out / "keeper_sft_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
