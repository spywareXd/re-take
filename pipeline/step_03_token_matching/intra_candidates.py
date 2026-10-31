"""
Intra-unit retake candidates (OPTIONAL, suggestion-only, for the review UI).

Sometimes the ASR keeps a restart inside ONE unit because there is no pause or punctuation:
    "Now I use beard, [redacted sentence]."
This module flags such units and proposes where the abandoned first attempt ends.

A unit is suggested only when BOTH rules fire (and it has >= min_tokens tokens):
    R1 prefix restart  : the first k (2..5) tokens reappear later in the same unit
    R2 repeated n-gram : an n-gram (n >= 3) occurs twice, with a token between the copies
                         or a silence shorter than short_gap_seconds
Measured on a 63-row review sheet R1+R2 together were right on 9 of 12 flagged units; R1 alone 1/6 and
R2 alone 0/4. R3/R4 from the experiment script are dropped. Known false alarms are intentional parallel
phrasing ("I love X and I love Y"). Recall is unknown, and ASR may drop repeated words, which a text
rule cannot see.

Nothing here changes the default pipeline: candidates are never fed to candidate generation, span
aggregation, LLM adjudication or the cutlist. They stay status="suggested" until a user decision is applied
(see pipeline/step_08_video_edit/intra_decisions.py).

The R1/R2 logic is ported from data/training/intra_detect.py (which is left unchanged).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from .config import IntraUnitConfig, default_intra_config
from pipeline.step_02_preprocessing.unit_splitter import _parse_segments, extract_segment_words

DEFAULT_INTRA_CANDIDATES_FILENAME = "intra_unit_candidates.json"
_TOKEN_STRIP_RE = re.compile(r"[^a-z0-9']")


@dataclass(slots=True)
class IntraUnitCandidate:
    """One suggested restart inside a single unit. `words` carries the unit's word timestamps so a
    later 'adjust' decision can recompute the span without needing the transcript."""

    candidate_id: str
    unit_id: str
    text: str
    split_word_index: int            # index (within the unit's words) of the first word of the restart
    discard_text: str
    kept_text: str
    proposed_discard_start: float    # seconds, start of the first attempt
    proposed_discard_end: float      # seconds, start of the restart
    pause_at_split_seconds: float
    mean_conf_discard: Optional[float]
    rules_fired: str                 # "R1+R2"
    detector_version: str
    status: str = "suggested"
    first_half_complete: bool = False   # ranking hint only, never used to filter
    video_id: Optional[str] = None
    unit_start: float = 0.0
    unit_end: float = 0.0
    words: list[dict[str, Any]] = field(default_factory=list)   # [{word, start, end, confidence}]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntraUnitCandidate":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


# ---------------------------------------------------------------------------
# token helpers (config-driven)
# ---------------------------------------------------------------------------


def _clean_token(word: str) -> str:
    """Lowercase, keep letters/digits/apostrophes ("beard," -> "beard")."""
    return _TOKEN_STRIP_RE.sub("", word.lower())


def _is_number(tok: str, cfg: IntraUnitConfig) -> bool:
    return tok.isdigit() or tok in cfg.number_words


def _has_content(gram: Sequence[str], cfg: IntraUnitConfig) -> bool:
    """An n-gram/prefix counts only if it has a non-function, non-number word (avoids 'and the ... and the')."""
    return any(t not in cfg.function_words and not _is_number(t, cfg) for t in gram)


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------


def detect_r1(toks: Sequence[str], cfg: IntraUnitConfig = default_intra_config) -> Optional[int]:
    """R1 prefix restart. Returns j, the index where the repeated prefix starts again (largest k wins), or None.
    A prefix made of one repeated word ('very very') or an intentional doubling is not a restart."""
    n = len(toks)
    for k in range(min(cfg.prefix_max_tokens, n // 2), cfg.prefix_min_tokens - 1, -1):
        pre = list(toks[:k])
        if len(set(pre)) == 1 or tuple(pre) in cfg.doublings or not _has_content(pre, cfg):
            continue
        j = next((j for j in range(k, n - k + 1) if list(toks[j:j + k]) == pre), None)
        if j is not None:
            return j
    return None


def detect_r2(
    toks: Sequence[str],
    words: Sequence[Any],
    cfg: IntraUnitConfig = default_intra_config,
) -> Optional[tuple[int, int]]:
    """R2 repeated n-gram (n >= ngram_min). Returns (i, j) of the first copy and the repeat, or None.
    Adjacent copies only count if the silence between them is shorter than short_gap_seconds."""
    n = len(toks)
    for g in range(min(cfg.ngram_max, n // 2), cfg.ngram_min - 1, -1):
        for i in range(0, n - g + 1):
            gram = list(toks[i:i + g])
            if not _has_content(gram, cfg):
                continue
            for j in range(i + g, n - g + 1):
                if list(toks[j:j + g]) != gram:
                    continue
                gap = words[j].start - words[i + g - 1].end
                if j > i + g or gap < cfg.short_gap_seconds:
                    return i, j
    return None


def first_half_complete(discard_text: str, cfg: IntraUnitConfig = default_intra_config) -> bool:
    """Simple ranking hint: the abandoned first half ends with terminal punctuation, or contains a
    subject-like word followed within a few tokens by a verb-like word. NOT used to filter candidates."""
    stripped = discard_text.strip()
    if stripped and stripped[-1] in cfg.terminal_punctuation:
        return True
    toks = [t for t in (_clean_token(w) for w in stripped.split()) if t]
    for idx, tok in enumerate(toks):
        if tok in cfg.subject_words:
            for nxt in toks[idx + 1: idx + 1 + cfg.subject_verb_window]:
                if nxt in cfg.common_verbs or (len(nxt) > 4 and nxt.endswith(cfg.verb_suffixes)):
                    return True
    return False


# ---------------------------------------------------------------------------
# unit / word lookup
# ---------------------------------------------------------------------------


def _field(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _unit_words(unit: Any, segments_by_id: dict[int, Any]) -> list[Any]:
    """The unit's words: those of its source ASR segment whose times fall inside [unit.start, unit.end]
    (+-0.03 s tolerance). ComparableUnit has no word list of its own."""
    seg = segments_by_id.get(int(_field(unit, "source_segment_id", -1)))
    if seg is None:
        return []
    start, end = float(_field(unit, "start", 0.0)), float(_field(unit, "end", 0.0))
    return [
        w for w in extract_segment_words(seg)
        if _clean_token(w.word) and w.start >= start - 0.03 and w.end <= end + 0.03
    ]


def _segments_by_id(transcript: Any) -> dict[int, Any]:
    out: dict[int, Any] = {}
    for idx, seg in enumerate(_parse_segments(transcript), start=1):
        out[int(_field(seg, "id", idx))] = seg
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def find_intra_unit_candidates(
    units: Sequence[Any],
    transcript: Union[str, Path, dict[str, Any], list[Any], Any],
    video_id: Optional[str] = None,
    config: IntraUnitConfig = default_intra_config,
) -> list[IntraUnitCandidate]:
    """
    Suggest restarts inside single units. `units` are ComparableUnits (or dicts with the same fields);
    `transcript` is a Transcript, dict, list of segments or a path (the source of the word timestamps).
    Only units with >= config.min_tokens tokens where R1 AND R2 both fire are returned; the proposed split
    is the R1 split (the first attempt starts at the unit's first word, the restart at the repeated prefix).
    """
    segs = _segments_by_id(transcript)
    out: list[IntraUnitCandidate] = []
    for unit in units:
        words = _unit_words(unit, segs)
        toks = [_clean_token(w.word) for w in words]
        if len(toks) < config.min_tokens:
            continue
        j = detect_r1(toks, config)
        if j is None or detect_r2(toks, words, config) is None:
            continue

        uid = str(_field(unit, "unit_id"))
        discard_words, kept_words = words[:j], words[j:]
        discard_text = " ".join(w.word for w in discard_words)
        confs = [w.confidence for w in discard_words if w.confidence is not None]
        out.append(
            IntraUnitCandidate(
                candidate_id=f"intra__{video_id + '__' if video_id else ''}{uid.replace(':', '-')}",
                unit_id=uid,
                text=" ".join(w.word for w in words),
                split_word_index=j,
                discard_text=discard_text,
                kept_text=" ".join(w.word for w in kept_words),
                proposed_discard_start=round(float(words[0].start), 3),
                proposed_discard_end=round(float(words[j].start), 3),
                pause_at_split_seconds=round(float(words[j].start - words[j - 1].end), 3),
                mean_conf_discard=round(sum(confs) / len(confs), 3) if confs else None,
                rules_fired="R1+R2",
                detector_version=config.detector_version,
                status="suggested",
                first_half_complete=first_half_complete(discard_text, config),
                video_id=video_id,
                unit_start=round(float(_field(unit, "start", words[0].start)), 3),
                unit_end=round(float(_field(unit, "end", words[-1].end)), 3),
                words=[
                    {"word": w.word, "start": round(float(w.start), 3), "end": round(float(w.end), 3),
                     "confidence": w.confidence}
                    for w in words
                ],
            )
        )
    return out


def save_intra_unit_candidates(
    candidates: Sequence[IntraUnitCandidate],
    output_path: Union[str, Path],
    video_id: Optional[str] = None,
    config: IntraUnitConfig = default_intra_config,
) -> Path:
    """Write the artifact JSON (metadata envelope + candidates)."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "detector_version": config.detector_version,
        "video_id": video_id,
        "note": "Suggestions only. Nothing is cut until a user decision is applied.",
        "candidate_count": len(candidates),
        "candidates": [c.to_dict() for c in candidates],
    }
    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_p


def load_intra_unit_candidates(input_path: Union[str, Path]) -> list[IntraUnitCandidate]:
    """Read candidates back from the artifact JSON."""
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return [IntraUnitCandidate.from_dict(c) for c in data.get("candidates", [])]
