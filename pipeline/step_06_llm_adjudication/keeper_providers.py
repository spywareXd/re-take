"""
Keeper providers for Step 06: `finetuned` (default; the local decision model behind the hybrid policy), `gpt-oss` (the existing LLM keeper call,
unchanged) and `keep-last` (the last take of every group, no model). Pick one with KeeperConfig.provider / RETAKE_KEEPER_PROVIDER / --keeper-provider / the request field `keeper_provider`.

The decision model needs Unsloth, which lives in the Studio environment (Python 3.13, torch 2.13), not in this repo's environment (torch 2.5 for
WhisperX), so it cannot be loaded in-process. `finetuned` therefore calls the Unsloth Studio Decision API on localhost
(POST <studio_url>/v1/systemone, `state` sent as a JSON object) with the model loaded in Studio. No other host is ever contacted.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence
from urllib.parse import urlparse

import requests

from .adjudicator import _extract_span_dict, _api_key_available, adjudicate_single_span, run_stage6
from .config import AdjudicationConfig, default_adjudication_config
from .keeper_policy import PROVIDERS, KeeperConfig, KeeperDecision, apply_rules, decide_finetuned
from .llm_client import LLMClient

logger = logging.getLogger(__name__)
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ProviderUnavailable(RuntimeError):
    """An explicitly chosen keeper provider cannot run (the message says why)."""


class StudioDecisionBackend:
    """The Unsloth Studio Decision API on localhost. predict(state, questions) -> {"keep_<id>": p}."""

    def __init__(self, url: str, model: str, api_key: Optional[str] = None, timeout: float = 60.0, session: Optional[requests.Session] = None):
        host = urlparse(url).hostname or ""
        if host not in LOCAL_HOSTS:
            raise ValueError(f"the decision API must be on localhost, got host '{host}'")
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self._key = api_key if api_key is not None else os.environ.get("UNSLOTH_API_KEY")
        self._session = session or requests.Session()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}"} if self._key else {}

    def availability(self) -> tuple[bool, str]:
        if not self._key:
            return False, "UNSLOTH_API_KEY is not set (needed for the Studio Decision API)"
        try:
            r = self._session.get(self.url + "/api/settings/systemone", headers=self._headers(), timeout=3)
        except requests.RequestException:
            return False, f"the Studio Decision API is not reachable at {self.url}"
        if r.status_code != 200:
            return False, f"the Studio Decision API answered HTTP {r.status_code}"
        names = {m.get("name") for m in r.json().get("models", []) if m.get("available")}
        if self.model not in names:
            return False, f"model '{self.model}' is not available in Studio"
        return True, f"decision model '{self.model}' via {self.url}/v1/systemone"

    def predict(self, state: dict, questions: dict) -> dict[str, float]:
        r = self._session.post(self.url + "/v1/systemone", json={"model": self.model, "state": state, "questions": questions},
                               headers=self._headers(), timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"decision API HTTP {r.status_code}: {r.text[:200]}")
        answers = r.json().get("answers") or {}
        out: dict[str, float] = {}
        for q in questions:
            a = answers.get(q)
            v = a.get("noul") if isinstance(a, dict) else None              # p(True) of the noul question
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"no probability for question {q}")
            out[q] = float(v)
        return out


class KeeperProvider(ABC):
    name: str

    @abstractmethod
    def availability(self) -> tuple[bool, str]:
        ...

    @abstractmethod
    def decide(self, span: dict, members: list, before: Any, after: Any) -> KeeperDecision:
        ...


class FinetunedProvider(KeeperProvider):
    name = "finetuned"

    def __init__(self, cfg: KeeperConfig, backend: Any = None):
        self.cfg = cfg
        self.backend = backend if backend is not None else StudioDecisionBackend(cfg.studio_url, cfg.studio_model, timeout=cfg.timeout_seconds)
        self.model_id = getattr(self.backend, "model", None) or cfg.studio_model

    def availability(self) -> tuple[bool, str]:
        return self.backend.availability()

    def decide(self, span: dict, members: list, before: Any, after: Any) -> KeeperDecision:
        return decide_finetuned(str(span.get("span_id")), members, before, after, self.backend.predict, self.cfg, self.model_id)


class GptOssProvider(KeeperProvider):
    """The existing keeper call (adjudicate_single_span), unchanged. Rule layers only with keeper.rules_with_gpt_oss."""
    name = "gpt-oss"

    def __init__(self, cfg: KeeperConfig, adj_config: Optional[AdjudicationConfig] = None, client: Optional[LLMClient] = None):
        self.cfg, self.adj_config, self.client = cfg, adj_config or default_adjudication_config, client
        self.all_units: Optional[Sequence[Any]] = None          # set by run_keeper_stage: the prompt sees the same units as before

    def availability(self) -> tuple[bool, str]:
        if self.client is not None:
            return True, "injected LLM client"
        if _api_key_available(self.adj_config):
            return True, f"provider '{self.adj_config.provider}', model '{self.adj_config.model}' (API key present)"
        return False, f"no API key for provider '{self.adj_config.provider}'"

    def decide(self, span: dict, members: list, before: Any, after: Any) -> KeeperDecision:
        ids = [str(u) for u in span.get("unit_ids", [])]
        if self.cfg.rules_with_gpt_oss:
            ruled = apply_rules(str(span.get("span_id")), members, self.cfg, "gpt-oss", self.adj_config.model)
            if ruled is not None:
                return ruled
        res = adjudicate_single_span(span, all_units_by_id=self.all_units if self.all_units is not None else members, config=self.adj_config,
                                     client=self.client)
        if res.get("adjudication_status") != "resolved":
            return KeeperDecision(str(span.get("span_id")), ids, [], "failed", "gpt-oss", "gpt-oss", "", self.adj_config.model,
                                  error=res.get("adjudication_error"), extra=res)
        return KeeperDecision(str(span.get("span_id")), ids, [res["keeper_unit_id"]], "resolved", "gpt-oss", "gpt-oss",
                              str(res.get("reasoning") or ""), self.adj_config.model, extra=res)


class KeepLastProvider(KeeperProvider):
    """Keep the last take of every group: no model, no rules, no key, no network."""
    name = "keep-last"

    def availability(self) -> tuple[bool, str]:
        return True, "always available (keeps the last take of every group)"

    def decide(self, span: dict, members: list, before: Any, after: Any) -> KeeperDecision:
        ids = [str(u["unit_id"] if isinstance(u, dict) else u.unit_id)
               for u in sorted(members, key=lambda u: (float(u["start"] if isinstance(u, dict) else u.start), str(u["unit_id"] if isinstance(u, dict) else u.unit_id)))]
        ids = ids or [str(u) for u in span.get("unit_ids", [])]
        return KeeperDecision(str(span.get("span_id")), ids, [ids[-1]], "resolved", "keep-last", "keep_last", "Keeping the last take.", None)


def choose_provider_name(explicit: Optional[str], cfg: KeeperConfig) -> str:
    """Request field / CLI flag first, then cfg.provider (which already holds RETAKE_KEEPER_PROVIDER if built with from_env)."""
    name = (explicit or cfg.provider or "").strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"unknown keeper provider '{name}'; choose one of {list(PROVIDERS)}")
    return name


def make_provider(name: str, cfg: KeeperConfig, adj_config: Optional[AdjudicationConfig] = None, llm_client: Optional[LLMClient] = None,
                  backend: Any = None) -> KeeperProvider:
    if name == "finetuned":
        return FinetunedProvider(cfg, backend)
    if name == "gpt-oss":
        return GptOssProvider(cfg, adj_config, llm_client)
    if name == "keep-last":
        return KeepLastProvider()
    raise ValueError(f"unknown keeper provider '{name}'; choose one of {list(PROVIDERS)}")


def _neighbours(members: Sequence[Any], ordered_units: Sequence[Any]) -> tuple[Any, Any]:
    """The unit just before the first member and just after the last member in the transcript (the training context rule)."""
    uid = lambda u: str(u["unit_id"] if isinstance(u, dict) else u.unit_id)
    pos = {uid(u): i for i, u in enumerate(ordered_units)}
    idx = [pos[uid(m)] for m in members if uid(m) in pos]
    if not idx:
        return None, None
    lo, hi = min(idx), max(idx)
    return (ordered_units[lo - 1] if lo > 0 else None), (ordered_units[hi + 1] if hi + 1 < len(ordered_units) else None)


def run_keeper_stage(spans: Sequence[Any], units: Sequence[Any], provider: KeeperProvider) -> list[dict]:
    """Step 06 with a keeper provider. Output = run_stage6's span contract plus the optional keeper_* fields.
    gpt-oss without rule layers delegates to run_stage6 itself, so its output is exactly the old one."""
    if isinstance(provider, GptOssProvider) and not provider.cfg.rules_with_gpt_oss:
        return run_stage6(spans, all_units_by_id=units, config=provider.adj_config, llm_client=provider.client)
    ordered = sorted(units, key=lambda u: (float(u["start"] if isinstance(u, dict) else u.start), str(u["unit_id"] if isinstance(u, dict) else u.unit_id)))
    by_id = {str(u["unit_id"] if isinstance(u, dict) else u.unit_id): u for u in units}
    if isinstance(provider, GptOssProvider):
        provider.all_units = units
    out = []
    for item in spans:
        span = _extract_span_dict(item)
        members = [by_id[str(u)] for u in span.get("unit_ids", []) if str(u) in by_id]
        before, after = _neighbours(members, ordered)
        dec = provider.decide(span, members, before, after)
        logger.info("[Stage 6] span %s: %s by %s keep=%s%s", span.get("span_id"), dec.status, dec.decided_by, dec.keep or "none",
                    " (UNCLEAR)" if dec.status == "unclear" else "")
        merged = dict(span)
        merged.update(dec.span_fields())
        out.append(merged)
    return out
