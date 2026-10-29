"""Optional keeper advice: Step 06 (keeper-only LLM) behind /v1/adjudicate. Advice is never used as a training label."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pipeline.step_06_llm_adjudication import AdjudicationConfig
from pipeline.step_06_llm_adjudication.adjudicator import adjudicate_single_span
from pipeline.step_06_llm_adjudication.llm_client import LLMClient, MockLLMClient

from .config import ServiceConfig
from .jobs import JobContext

logger = logging.getLogger("service.llm")
REASON_MAX = 300
DEFAULT_MODELS = {"groq": "openai/gpt-oss-120b", "openai": "gpt-4o-mini"}


@dataclass
class LLMSetup:
    available: bool
    provider: str
    config: Optional[AdjudicationConfig] = None
    client: Optional[LLMClient] = None             # explicit client (the mock, or a test double)


def detect_llm(cfg: ServiceConfig, client: Optional[LLMClient] = None) -> LLMSetup:
    """adjudicate is available only if the mock is enabled or an LLM key is in the environment. Keys are checked, never printed or logged."""
    if client is not None:
        return LLMSetup(True, "injected", None, client)
    if cfg.llm_mock or (cfg.llm_provider or "").lower() == "mock":
        return LLMSetup(True, "mock", None, MockLLMClient())
    provider = (cfg.llm_provider or "").lower()
    if not provider:
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            provider = "gemini"
        elif os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
    has_key = {"gemini": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
               "groq": bool(os.environ.get("GROQ_API_KEY")), "openai": bool(os.environ.get("OPENAI_API_KEY"))}.get(provider, False)
    if not has_key:
        return LLMSetup(False, provider or "none")
    kw: dict[str, Any] = {"provider": provider}
    model = cfg.llm_model or DEFAULT_MODELS.get(provider)
    if model:
        kw["model"] = model
    if provider == "groq":
        kw["api_key"] = os.environ["GROQ_API_KEY"]          # explicit: the OpenAI-compatible client would otherwise prefer OPENAI_API_KEY
    return LLMSetup(True, provider, AdjudicationConfig(**kw))


def group_to_span(g: dict) -> tuple[dict, list[dict]]:
    sentences = sorted(g["sentences"], key=lambda s: (s["start"], s["id"]))
    units = [{"unit_id": s["id"], "start": s["start"], "end": s["end"], "raw_text": s["text"]} for s in sentences]
    span = {"span_id": g["id"], "unit_ids": [u["unit_id"] for u in units], "start": g["start"], "end": g["end"], "candidate_pairs": [],
            "classification_summary": {}, "needs_adjudication": False}
    return span, units


def adjudicate_groups(groups: list[dict], setup: LLMSetup, ctx: JobContext) -> list[dict]:
    """[{group, keep:[ids], reason}] for the groups the LLM answered. A group it fails on is omitted (never invented); keep is never empty."""
    out: list[dict] = []
    seen: set = set()
    for i, g in enumerate(groups):
        ctx.progress("Asking the AI", i / max(len(groups), 1))
        if g["id"] in seen:
            continue                                        # a group answered twice uses the first answer
        seen.add(g["id"])
        span, units = group_to_span(g)
        res = adjudicate_single_span(span, all_units_by_id=units, config=setup.config, client=setup.client)
        keeper = res.get("keeper_unit_id")
        if res.get("adjudication_status") != "resolved" or not keeper or keeper not in span["unit_ids"]:
            logger.warning("adjudication skipped group %s: %s", g["id"], res.get("adjudication_error", "no valid keeper"))
            continue
        out.append({"group": g["id"], "keep": [keeper], "reason": str(res.get("reasoning") or "")[:REASON_MAX]})
    return out


def record_decision(path: str, entry: dict) -> None:
    """Append one keeper decision (advice, never a label) to keeper_decisions.jsonl."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}, ensure_ascii=False) + "\n")


def _context_units(path: str, cache_dir: str) -> Optional[list[dict]]:
    """The analysed file's units (from the service cache), for the before/after context the decision model was trained with."""
    from pathlib import Path

    from pipeline.step_07_review.cache import media_fingerprint, open_cache

    try:
        p = Path(path)
        return open_cache(cache_dir, media_fingerprint(p)).get("units") if p.is_file() else None
    except Exception:
        return None


def keeper_advice(groups: list[dict], provider: Any, ctx: JobContext, path: str, cache_dir: Any, decisions_path: str, job: Optional[str]) -> list[dict]:
    """/v1/adjudicate with the finetuned provider: [{group, keep, reason} + optional provider / decided_by / model / unclear / probs].
    Same contract as adjudicate_groups: keep is never empty, so a keep-none decision (cue rule) is left out of the array (and recorded)."""
    from pipeline.step_06_llm_adjudication.keeper_providers import _neighbours

    units = _context_units(path, cache_dir) or []
    by_id = {str(u["unit_id"]): u for u in units}
    ordered = sorted(units, key=lambda u: (float(u["start"]), str(u["unit_id"])))
    out: list[dict] = []
    seen: set = set()
    for i, g in enumerate(groups):
        ctx.progress("Choosing the takes to keep", i / max(len(groups), 1))
        if g["id"] in seen:
            continue
        seen.add(g["id"])
        span, members = group_to_span(g)
        members = [by_id.get(m["unit_id"], m) for m in members]                     # cached units carry the same ids, times and text
        before, after = _neighbours(members, ordered) if ordered else (None, None)
        dec = provider.decide(span, members, before, after)
        fields = dec.span_fields()
        record_decision(decisions_path, {"job": job, "group": g["id"], "provider": dec.provider, "model": dec.model, "decided_by": dec.decided_by,
                                         "status": dec.status, "keep": dec.keep, "unit_ids": dec.unit_ids, "probs": fields.get("unit_probs"),
                                         "reason": dec.reason, "error": dec.error, "source": "keeper_advice"})
        if dec.status == "failed" or not dec.keep:
            logger.warning("keeper advice left out group %s: %s", g["id"], "keep none (" + dec.decided_by + ")" if not dec.keep else dec.error)
            continue
        item = {"group": g["id"], "keep": list(dec.keep), "reason": dec.reason[:REASON_MAX], "provider": dec.provider, "decided_by": dec.decided_by,
                "model": dec.model, "unclear": dec.status == "unclear"}
        if fields.get("unit_probs") is not None:
            item["probs"] = fields["unit_probs"]
        out.append(item)
    return out
