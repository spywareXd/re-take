"""
Hybrid keeper policy for the fine-tuned decision model (Step 06, provider `finetuned`).

Layers, in order (each behind its own KeeperConfig flag, all on by default):
  a. cue rule        every take is short and lexically close to a cue (keeper_cues.json) -> keep none        decided_by=cue_rule
  b. exact gate      every take has the same normalised text                            -> keep the last    decided_by=exact_gate
  c. model           the decision model's p(keep) per take, keep = p >= 0.5              -> its keep set     decided_by=model
  d. empty fallback  the model keeps nothing (and the cue rule did not apply)           -> keep the last    decided_by=empty_fallback
  e. unclear         top two p within 0.15 or max p below 0.6 (training/decision_eval)  -> keep the last, flagged unclear
  model error        the model cannot answer: on_model_error="fail_closed" (default) -> the group FAILS (nothing is cut), decided_by=model_error
                     on_model_error="keep_last" -> keep the last take, decided_by=model_error_fallback (flagged unclear)

The model's input is built with the training-time builders, unchanged: keeper_prompt.build_keeper_messages("short", ...) ->
training/build_decision_data.to_row (rules from decision_rules()) -> training/build_decision_data_v2.rich_questions.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

CUES_PATH = Path(__file__).resolve().parent / "keeper_cues.json"
PROVIDERS = ("finetuned", "gpt-oss", "keep-last")
DEFAULT_PROVIDER = "finetuned"
DEFAULT_STUDIO_URL = "http://127.0.0.1:8888"
DEFAULT_STUDIO_MODEL = "clef-ft:Qwen_final"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "decision_standalone"
ON_MODEL_ERROR = ("fail_closed", "keep_last")
LOCAL_MODES = ("4bit", "fp16")                      # 4bit_requant exists in the loader for measurement only and is not offered here


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None or v == "" else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class KeeperConfig:
    """`keeper.*` settings. Provider precedence: request field / CLI flag > RETAKE_KEEPER_PROVIDER > this default."""
    provider: str = DEFAULT_PROVIDER
    cue_rule: bool = True
    cue_threshold: float = 0.8                          # a take is a cue if its best lexical score against the cue list is at least this
    cue_max_words: int = 8                              # ... and it has at most this many words
    exact_gate: bool = True
    empty_fallback: bool = True
    unclear_default: bool = True
    rules_with_gpt_oss: bool = False                    # gpt-oss: rule layers off, so the old behaviour is unchanged
    studio_url: str = DEFAULT_STUDIO_URL                # the Unsloth Studio Decision API (localhost only)
    studio_model: str = DEFAULT_STUDIO_MODEL            # runs/keeper_loop/final_all/model, loaded in Studio
    timeout_seconds: float = 60.0
    model_path: str = str(DEFAULT_MODEL_PATH)           # the standalone decision model folder (models/decision_standalone, see export_manifest.json)
    model_repo: Optional[str] = None                    # optional Hugging Face repo id to download the model from when model_path is empty or missing
    mode: str = "4bit"                                  # 4bit (about 5.4 GB VRAM) | fp16 (needs >= 10 GB free)
    verify_model: bool = False                          # check SHA-256 of the files in use against export_manifest.json on load (always done after a download)
    unload_after: bool = True                           # free the decision model (and its VRAM) after the keeper step
    on_model_error: str = "fail_closed"                 # fail_closed: the group fails, nothing is cut | keep_last: keep the last take, flagged unclear
    extra_cues: Optional[list[str]] = None              # None = the cues in keeper_cues.json

    @classmethod
    def from_env(cls, **overrides: Any) -> "KeeperConfig":
        e = os.environ.get
        cfg = cls(provider=(e("RETAKE_KEEPER_PROVIDER") or DEFAULT_PROVIDER).strip().lower(),
                  cue_rule=_flag("RETAKE_KEEPER_CUE_RULE", True), cue_threshold=float(e("RETAKE_KEEPER_CUE_THRESHOLD") or 0.8), exact_gate=_flag("RETAKE_KEEPER_EXACT_GATE", True),
                  empty_fallback=_flag("RETAKE_KEEPER_EMPTY_FALLBACK", True), unclear_default=_flag("RETAKE_KEEPER_UNCLEAR_DEFAULT", True),
                  rules_with_gpt_oss=_flag("RETAKE_KEEPER_RULES_WITH_GPT_OSS", False),
                  studio_url=e("RETAKE_KEEPER_STUDIO_URL") or DEFAULT_STUDIO_URL, studio_model=e("RETAKE_KEEPER_MODEL") or DEFAULT_STUDIO_MODEL,
                  model_path=e("RETAKE_KEEPER_MODEL_PATH") or str(DEFAULT_MODEL_PATH), model_repo=e("RETAKE_KEEPER_MODEL_REPO") or None,
                  mode=(e("RETAKE_KEEPER_MODE") or "4bit").strip().lower(), verify_model=_flag("RETAKE_KEEPER_VERIFY_MODEL", False),
                  unload_after=_flag("RETAKE_KEEPER_UNLOAD_AFTER", True), on_model_error=(e("RETAKE_KEEPER_ON_MODEL_ERROR") or "fail_closed").strip().lower())
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode not in LOCAL_MODES:
            raise ValueError(f"keeper.mode must be one of {list(LOCAL_MODES)}, got '{self.mode}'")
        if self.on_model_error not in ON_MODEL_ERROR:
            raise ValueError(f"keeper.on_model_error must be one of {list(ON_MODEL_ERROR)}, got '{self.on_model_error}'")


# ---------------------------------------------------------------- rules

def normalise(text: str) -> str:
    """Lowercase, punctuation stripped, spaces collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']+|_", " ", str(text).lower())).strip()


def load_cues(path: Path = CUES_PATH) -> list[str]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return list(d["bank_cues"]) + [t for texts in d.get("natural_keep_none_cues", {}).values() for t in texts]


_CUES_CACHE: dict[str, set[str]] = {}


def cue_set(cfg: KeeperConfig) -> set[str]:
    if cfg.extra_cues is not None:
        return {normalise(c) for c in cfg.extra_cues}
    if "file" not in _CUES_CACHE:
        _CUES_CACHE["file"] = {normalise(c) for c in load_cues()}
    return _CUES_CACHE["file"]


def cue_score(text: str, cues: set[str]) -> float:
    """Best lexical similarity of a take to any cue: max of the character ratio (difflib) and the word Jaccard overlap, on normalised text."""
    from difflib import SequenceMatcher

    n = normalise(text)
    if n in cues:
        return 1.0
    w = set(n.split())
    best = 0.0
    for c in cues:
        cw = set(c.split())
        best = max(best, SequenceMatcher(None, n, c).ratio(), len(w & cw) / max(len(w | cw), 1))
    return best


def is_cue_take(text: str, cfg: KeeperConfig) -> bool:
    return 0 < len(normalise(text).split()) <= cfg.cue_max_words and cue_score(text, cue_set(cfg)) >= cfg.cue_threshold


def is_cue_group(texts: Sequence[str], cfg: KeeperConfig) -> bool:
    """Strong cue group: EVERY take is a short cue-like take (one real sentence among cues is enough to ask the model)."""
    return bool(texts) and all(is_cue_take(t, cfg) for t in texts)


def is_exact_group(texts: Sequence[str]) -> bool:
    norm = [normalise(t) for t in texts]
    return len(norm) >= 2 and bool(norm[0]) and all(n == norm[0] for n in norm)


# ---------------------------------------------------------------- model input (training-time builders, reused)

def _u(u: Any) -> dict:
    if isinstance(u, dict):
        return {"unit_id": str(u["unit_id"]), "start": float(u["start"]), "end": float(u["end"]), "raw_text": str(u.get("raw_text", ""))}
    return {"unit_id": str(u.unit_id), "start": float(u.start), "end": float(u.end), "raw_text": str(u.raw_text)}


def build_decision_request(span_id: str, members: Sequence[Any], before: Any = None, after: Any = None) -> tuple[dict, dict]:
    """(state, questions) exactly as the decision model saw them in training (test copies: full context, nothing dropped)."""
    from training.build_decision_data import decision_rules, to_row
    from training.build_decision_data_v2 import rich_questions
    from .keeper_prompt import build_keeper_messages

    ms = [_u(m) for m in members]
    msgs = build_keeper_messages("short", span_id, ms, _u(before) if before is not None else None, _u(after) if after is not None else None, False)
    msgs.append({"role": "assistant", "content": json.dumps({"keep": [], "keep_none": True, "unclear": False})})   # to_row needs an answer; gold is discarded
    row = to_row({"messages": msgs}, decision_rules())
    state = json.loads(row["state"])
    return state, rich_questions(state)


def decide_probs(probs: dict[str, float]) -> dict:
    """keep = p >= 0.5; unclear = top two within 0.15 or max below 0.6 (training/decision_eval.decide, the scoring rule)."""
    from training.decision_eval import decide
    return decide(probs)


# ---------------------------------------------------------------- decision record

@dataclass
class KeeperDecision:
    span_id: str
    unit_ids: list[str]                               # chronological
    keep: list[str]                                   # keeper unit ids (empty = keep none)
    status: str                                       # resolved | unclear | failed
    provider: str
    decided_by: str                                   # cue_rule | exact_gate | model | empty_fallback | model_error_fallback | gpt-oss | keep_last
    reason: str = ""
    model: Optional[str] = None
    probs: Optional[dict[str, float]] = None
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)          # provider-specific fields copied into the span (gpt-oss: the old adjudication fields)

    @property
    def keep_none(self) -> bool:
        return self.status != "failed" and not self.keep

    def span_fields(self) -> dict:
        """The Step 06 span contract (adjudication_status / keeper_unit_id / units_to_remove ...) plus the new optional keeper_* fields."""
        if self.status == "failed":
            out = {"adjudication_status": "failed", "is_genuine_retake_group": True, "keeper_unit_id": None, "units_to_remove": None,
                   "confidence": None, "reasoning": None, "adjudication_error": self.error}
        else:
            out = {"adjudication_status": "resolved", "is_genuine_retake_group": True, "keeper_unit_id": self.keep[-1] if self.keep else None,
                   "units_to_remove": [u for u in self.unit_ids if u not in self.keep], "confidence": "low" if self.status == "unclear" else "high",
                   "reasoning": self.reason}
        out.update(self.extra)
        out.update({"keeper_status": self.status, "keeper_provider": self.provider, "keeper_model": self.model, "decided_by": self.decided_by,
                    "keeper_unit_ids": list(self.keep), "keep_none": self.keep_none, "unclear": self.status == "unclear"})
        if self.probs is not None:
            out["unit_probs"] = {k: round(float(v), 6) for k, v in self.probs.items()}
        if self.error and self.status != "failed":
            out["keeper_error"] = self.error
        return out


# ---------------------------------------------------------------- policy

def apply_rules(span_id: str, members: Sequence[Any], cfg: KeeperConfig, provider: str, model: Optional[str] = None) -> Optional[KeeperDecision]:
    """Layers a and b. None = no rule applies (ask the model)."""
    ms = sorted((_u(m) for m in members), key=lambda u: (u["start"], u["end"], u["unit_id"]))
    ids, texts = [m["unit_id"] for m in ms], [m["raw_text"] for m in ms]
    if cfg.cue_rule and is_cue_group(texts, cfg):
        return KeeperDecision(span_id, ids, [], "resolved", provider, "cue_rule", "Every take is a production cue: keep none.", model)
    if cfg.exact_gate and is_exact_group(texts):
        return KeeperDecision(span_id, ids, [ids[-1]], "resolved", provider, "exact_gate", "All takes have the same words: keep the last.", model)
    return None


def decide_finetuned(span_id: str, members: Sequence[Any], before: Any, after: Any, predict: Callable[[dict, dict], dict[str, float]],
                     cfg: KeeperConfig, model: Optional[str] = None) -> KeeperDecision:
    """The full hybrid policy for one group. `predict(state, questions)` returns {"keep_<id>" or "<id>": p}; any exception is a model error."""
    ruled = apply_rules(span_id, members, cfg, "finetuned", model)
    if ruled is not None:
        return ruled
    ms = sorted((_u(m) for m in members), key=lambda u: (u["start"], u["end"], u["unit_id"]))
    ids = [m["unit_id"] for m in ms]
    try:
        state, questions = build_decision_request(span_id, ms, before, after)
        raw = predict(state, questions)
        probs = {(k[len("keep_"):] if k.startswith("keep_") else k): float(v) for k, v in raw.items()}
        if set(probs) != set(ids):
            raise ValueError(f"model answered {sorted(probs)} for takes {ids}")
    except Exception as exc:                                                     # layer f
        if cfg.on_model_error != "keep_last":                                    # fail closed (default): the group fails and nothing is cut
            logger.error("[Keeper] MODEL ERROR on group %s (%s): the group fails, nothing is cut", span_id, exc)
            return KeeperDecision(span_id, ids, [], "failed", "finetuned", "model_error", "", model, None, f"{type(exc).__name__}: {exc}")
        logger.error("[Keeper] MODEL ERROR on group %s (%s): keeping the last take %s", span_id, exc, ids[-1])
        return KeeperDecision(span_id, ids, [ids[-1]], "unclear", "finetuned", "model_error_fallback",
                              "The keeper model could not answer: keeping the last take.", model, None, str(exc))
    d = decide_probs(probs)
    keep = [u for u in ids if u in set(d["keep"])]
    if not keep and cfg.empty_fallback:
        return KeeperDecision(span_id, ids, [ids[-1]], "unclear" if d["unclear"] else "resolved", "finetuned", "empty_fallback",
                              "The model kept no take: keeping the last take.", model, probs)
    if d["unclear"] and keep and cfg.unclear_default:
        return KeeperDecision(span_id, ids, [ids[-1]], "unclear", "finetuned", "model",
                              "The model is unsure: keeping the last take; please check this group.", model, probs, extra={"model_keep": keep})
    status = "unclear" if d["unclear"] else "resolved"
    reason = "The model keeps no take." if not keep else f"The model keeps {', '.join(keep)}."
    return KeeperDecision(span_id, ids, keep, status, "finetuned", "model", reason, model, probs)
