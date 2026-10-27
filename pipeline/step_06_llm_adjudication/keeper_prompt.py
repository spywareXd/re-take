"""
Keeper-selection prompt for the fine-tuned keeper model (the replacement for the Gemini keeper step).

ONE function builds the prompt, so the fine-tuning data (training/build_keeper_sft.py) and inference see byte-identical prompts:
the keeper rules first, then the group of takes. The model answers with
    {"keep": ["<unit id>", ...], "keep_none": false, "unclear": false, "reason": "<one sentence>"}
This does not replace `prompt_builder.build_adjudication_prompt` (the existing single-keeper Gemini prompt); that one is unchanged.
"""
from __future__ import annotations

from typing import Any, Sequence

KEEPER_RULES = '''You are the keeper selector for a video retake remover. A creator recorded the same
spoken line several times. You are given one GROUP of takes of that line (each is a
transcript unit with an id, text, start and end time in seconds). Decide which take(s)
stay in the final video. All other takes in the group are deleted.

Return ONLY JSON: {"keep": ["<unit id>", ...], "keep_none": false, "unclear": false, "reason": "<one sentence>"}
Use unit ids exactly as given. Never invent ids. Keep at least one id unless keep_none is true.

HOW TO CHOOSE (apply in this order)
1. Drop takes that are truncated, stumbling or abandoned: they end mid-phrase, contain
   false starts or repeated words ("the, the"), or are a fragment of a later take.
2. Drop takes with a factual or wording mistake that a later take corrects
   (wrong year, wrong name, wrong word that no longer fits the context).
3. Among complete, fluent takes, prefer the LATER one. Speakers retake until they are
   satisfied, and a restart that begins within about a second of the previous take
   ending means the previous take failed. If two takes have identical text, keep the last.
4. If the later take is complete but clearly drops important content that an earlier
   complete take has, prefer the earlier one and set "unclear": true.
5. Continuation: if the best take is only one part of a sentence and the next unit in the
   same group continues it without repeating its words (a pause of roughly 1 s or less,
   no overlapping text), keep both ids. If the next unit repeats the words, keep only one.
6. Use NEIGHBOR context if provided: prefer the take that connects smoothly to the unit
   before it and the unit after it, and keep ids that are part of the final chain of takes.
7. If units are spoken cue phrases rather than script content ("let's go", "here we go",
   "let me start from the top"), they are not content: keep none and set keep_none true.
8. If the group seems to contain two different lines that were joined by mistake,
   keep the last take of each line and set "unclear": true.
9. Judge the text only. Do not reward longer or more formal wording by itself. When two
   complete takes are just paraphrases of each other, choose the later one and set
   "unclear": true. Never assume content you cannot see.

Be conservative about deletion: a wrong "keep" costs a few seconds of duplicate speech,
but a wrong "drop" removes the only good version of a line.'''

_RULE = "=" * 70


def build_keeper_prompt(span_id: str, units: Sequence[dict[str, Any]], neighbors: str | None = None) -> str:
    """units: dicts with unit_id, start, end, raw_text (any order; shown chronologically). neighbors: optional text block for rule 6."""
    ordered = sorted(units, key=lambda u: (float(u["start"]), float(u["end"]), str(u["unit_id"])))
    lines = "\n".join(f'  [{u["unit_id"]}] ({float(u["start"]):.2f}s -> {float(u["end"]):.2f}s): "{u["raw_text"]}"' for u in ordered)
    ids = ", ".join(f'"{u["unit_id"]}"' for u in ordered)
    parts = [KEEPER_RULES, "", _RULE, f"GROUP {span_id}", _RULE, f"Unit ids: [{ids}]", "", "TAKES (IN CHRONOLOGICAL ORDER):", lines]
    if neighbors:
        parts += ["", "NEIGHBOR CONTEXT:", neighbors]
    parts += ["", "Return ONLY the JSON object."]
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------------------------------------------------
# v2: rules in the SYSTEM message, the group (with neighbour context and gaps) in the USER message, two rule variants ("full" / "short").
# Training and inference must use the same variant.
# ---------------------------------------------------------------------------------------------------------------------------------------
PROMPT_VARIANTS = ("full", "short")
_REASON_FIELD = ', "reason": "<one sentence>"'

KEEPER_RULES_SHORT = '''Choose which takes of a repeated line stay; the rest are deleted.
JSON only: {"keep": ["<id>", ...], "keep_none": false, "unclear": false, "reason": "<one sentence>"}
Use only the given ids, never a [context] unit. Keep at least one id unless keep_none.
1. Drop truncated, stumbling or abandoned takes.
2. Drop takes with a mistake a later take fixes.
3. Of complete takes keep the LATER (a restart within ~1 s means the earlier failed); identical: the last.
4. Later take drops key content: keep the earlier, unclear.
5. Next unit continues the sentence (gap <= ~1 s, no repeated words): keep both.
6. Prefer the take that fits the context.
7. Cue phrases ("let's go", "here we go"): keep [], keep_none true.
8. Two lines mixed: keep the last of each, unclear.
9. Text only; paraphrases: keep the later, unclear.
Be conservative: never drop the only good take.'''

CONTEXT_NOTICE = "NEIGHBOR CONTEXT (reference only: these units are NOT in this group and their ids are never valid answers):"


def rules_for(variant: str, with_reason: bool = False) -> str:
    """The system message. The answer format line drops the "reason" field unless with_reason (the rest of the text is unchanged)."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"prompt variant must be one of {PROMPT_VARIANTS}, got {variant!r}")
    text = KEEPER_RULES if variant == "full" else KEEPER_RULES_SHORT
    return text if with_reason else text.replace(_REASON_FIELD, "")


def _ctx_line(label: str, u: dict[str, Any] | None) -> str:
    if u is None:
        return f"  {label}: none"
    return f'  {label}: [context, not in this group] [{u["unit_id"]}] ({float(u["start"]):.2f}s -> {float(u["end"]):.2f}s): "{u["raw_text"]}"'


def build_keeper_user(span_id: str, units: Sequence[dict[str, Any]], before: dict[str, Any] | None = None, after: dict[str, Any] | None = None) -> str:
    """The USER message: the group's takes (chronological, each with gap_before in seconds) between two optional context units."""
    ordered = sorted(units, key=lambda u: (float(u["start"]), float(u["end"]), str(u["unit_id"])))
    lines, prev_end = [], (float(before["end"]) if before else None)
    for u in ordered:
        gap = "none" if prev_end is None else f"{max(0.0, float(u['start']) - prev_end):.2f}"
        lines.append(f'  [{u["unit_id"]}] ({float(u["start"]):.2f}s -> {float(u["end"]):.2f}s) gap_before={gap}: "{u["raw_text"]}"')
        prev_end = float(u["end"])
    ids = ", ".join(f'"{u["unit_id"]}"' for u in ordered)
    return "\n".join([f"GROUP {span_id}", f"Unit ids (the only valid answers): [{ids}]", "", CONTEXT_NOTICE, _ctx_line("before", before), _ctx_line("after", after),
                      "", "TAKES (IN CHRONOLOGICAL ORDER):", *lines, "", "Return ONLY the JSON object."])


def build_keeper_messages(variant: str, span_id: str, units: Sequence[dict[str, Any]], before: dict[str, Any] | None = None,
                          after: dict[str, Any] | None = None, with_reason: bool = False) -> list[dict[str, str]]:
    """[system, user] chat messages; the training data appends the assistant answer."""
    return [{"role": "system", "content": rules_for(variant, with_reason)}, {"role": "user", "content": build_keeper_user(span_id, units, before, after)}]
