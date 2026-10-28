"""The standalone `finetuned` keeper: fail-closed errors, settings, unload, model files, capabilities, no Unsloth/Studio imports.
No GPU and no model weights needed (the decision model is a fake)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.step_06_llm_adjudication import MockLLMClient
from pipeline.step_06_llm_adjudication.decision_standalone import DecisionModelMissing
from pipeline.step_06_llm_adjudication.keeper_model import StandaloneBackend, download_model, verify_manifest
from pipeline.step_06_llm_adjudication.keeper_policy import KeeperConfig, decide_finetuned
from pipeline.step_06_llm_adjudication.keeper_providers import FinetunedProvider, make_provider, run_keeper_stage
from service.app import Deps, create_app
from service.config import ServiceConfig

REPO = Path(__file__).resolve().parents[1]


def U(uid, start, text):
    return {"unit_id": uid, "start": float(start), "end": float(start) + 1.0, "raw_text": text}


UNITS = [U("1:0", 0, "Hello and welcome."), U("2:0", 2, "So the main idea is"), U("3:0", 4, "So the main idea is that it works."), U("4:0", 6, "Thanks for watching.")]
SPAN = {"span_id": "s1", "unit_ids": ["2:0", "3:0"], "start": 2.0, "end": 5.0, "candidate_pairs": [], "classification_summary": {}, "needs_adjudication": True}


class FakeLocal:
    """Stands in for StandaloneBackend: counts loads and unloads."""
    model = "standalone-fake"

    def __init__(self, probs=None, error=None):
        self.probs, self.error, self.unloaded, self.before_load = probs or {}, error, 0, None

    def availability(self):
        return True, "fake"

    def predict(self, state, questions):
        if self.error:
            raise self.error
        return {q: self.probs[q[len("keep_"):]] for q in questions}

    def unload(self):
        self.unloaded += 1


def test_defaults_are_fail_closed_4bit_and_unload_after():
    cfg = KeeperConfig()
    assert (cfg.provider, cfg.mode, cfg.on_model_error, cfg.unload_after) == ("finetuned", "4bit", "fail_closed", True)


def test_requant_mode_and_bad_error_policy_are_not_offered():
    for bad in ("4bit_requant", "int3"):
        with pytest.raises(ValueError, match="keeper.mode"):
            KeeperConfig.from_env(mode=bad)
        with pytest.raises(ValueError, match="keeper.mode"):
            StandaloneBackend(mode=bad)
    with pytest.raises(ValueError, match="on_model_error"):
        KeeperConfig.from_env(on_model_error="ignore")


def test_env_vars_set_the_keeper_options(monkeypatch):
    for k, v in {"RETAKE_KEEPER_MODEL_PATH": "X:/m", "RETAKE_KEEPER_MODE": "fp16", "RETAKE_KEEPER_UNLOAD_AFTER": "0", "RETAKE_KEEPER_ON_MODEL_ERROR": "keep_last",
                 "RETAKE_KEEPER_MODEL_REPO": "someone/model"}.items():
        monkeypatch.setenv(k, v)
    cfg = KeeperConfig.from_env()
    assert (cfg.model_path, cfg.mode, cfg.unload_after, cfg.on_model_error, cfg.model_repo) == ("X:/m", "fp16", False, "keep_last", "someone/model")


def test_model_error_fails_closed_by_default_and_keeps_last_when_asked():
    boom = FakeLocal(error=RuntimeError("cuda out of memory"))
    d = decide_finetuned("g", UNITS[1:3], None, None, boom.predict, KeeperConfig())
    f = d.span_fields()
    assert d.status == "failed" and d.decided_by == "model_error" and f["adjudication_status"] == "failed" and f["keeper_unit_id"] is None and f["units_to_remove"] is None
    assert "cuda out of memory" in f["adjudication_error"]
    d = decide_finetuned("g", UNITS[1:3], None, None, boom.predict, KeeperConfig(on_model_error="keep_last"))
    assert d.keep == ["3:0"] and d.status == "unclear" and d.decided_by == "model_error_fallback" and d.span_fields()["keeper_error"]


def test_rule_layers_never_call_the_model():
    boom = FakeLocal(error=RuntimeError("must not be called"))
    cue = decide_finetuned("g", [U("a", 0, "Here we go."), U("b", 2, "Okay, here we go.")], None, None, boom.predict, KeeperConfig())
    assert cue.decided_by == "cue_rule" and cue.keep == []
    exact = decide_finetuned("g", [U("a", 0, "Same words here."), U("b", 2, "same words here")], None, None, boom.predict, KeeperConfig())
    assert exact.decided_by == "exact_gate" and exact.keep == ["b"]


def test_stage_unloads_the_model_after_use_and_can_keep_it_loaded():
    for unload_after, expected in ((True, 1), (False, 0)):
        backend = FakeLocal({"2:0": 0.1, "3:0": 0.95})
        prov = FinetunedProvider(KeeperConfig(unload_after=unload_after), backend)
        out = run_keeper_stage([copy.deepcopy(SPAN)], UNITS, prov)
        assert out[0]["decided_by"] == "model" and out[0]["keeper_unit_id"] == "3:0" and backend.unloaded == expected


def test_stage_unloads_even_when_it_raises():
    backend = FakeLocal({"2:0": 0.1, "3:0": 0.95})
    prov = FinetunedProvider(KeeperConfig(), backend)
    bad = copy.deepcopy(SPAN)
    bad["unit_ids"] = 5                                                                  # not iterable -> the stage raises
    with pytest.raises(TypeError):
        run_keeper_stage([bad], UNITS, prov)
    assert backend.unloaded == 1


def test_default_provider_is_the_standalone_backend_and_gpt_oss_is_selectable():
    prov = make_provider("finetuned", KeeperConfig())
    assert isinstance(prov.backend, StandaloneBackend) and prov.model_id == "standalone-4bit" and not prov.backend.loaded
    assert prov.info()["backend"] == "standalone" and prov.info()["on_model_error"] == "fail_closed"
    assert make_provider("gpt-oss", KeeperConfig(), llm_client=MockLLMClient()).name == "gpt-oss"


def test_missing_model_is_a_clear_error(tmp_path):
    b = StandaloneBackend(tmp_path / "nope")
    ok, why = b.availability()
    assert not ok and "missing" in why
    with pytest.raises(DecisionModelMissing):
        b.predict({}, {"keep_a": {"type": "noul", "instructions": "x"}})


# ---------------------------------------------------------------- model files: manifest, download

def _make_model_dir(root: Path, mode_files: dict[str, bytes]) -> Path:
    files = {}
    for rel, data in mode_files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        files[rel] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (root / "export_manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return root


def test_manifest_verification_detects_changes_and_missing_files(tmp_path):
    data = {"head.safetensors": b"h", "head_config.json": b"{}", "adapter/adapter_model.safetensors": b"a", "base_bnb4/model.safetensors": b"b"}
    folder = _make_model_dir(tmp_path / "m", data)
    assert verify_manifest(folder, "4bit") == []
    (folder / "head.safetensors").write_bytes(b"changed")
    (folder / "base_bnb4" / "model.safetensors").unlink()
    problems = verify_manifest(folder, "4bit")
    assert any("head.safetensors" in p and "SHA-256" in p for p in problems) and any("base_bnb4/model.safetensors: missing" in p for p in problems)
    assert verify_manifest(tmp_path / "empty", "4bit") == ["export_manifest.json is missing"]


def test_download_never_writes_into_a_folder_that_has_files(tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    with pytest.raises(DecisionModelMissing, match="not empty"):
        download_model("someone/model", tmp_path, "4bit")
    assert (tmp_path / "keep.txt").read_text() == "x"


def test_a_configured_repo_makes_a_missing_folder_available(tmp_path):
    b = StandaloneBackend(tmp_path / "new", repo="someone/model")
    ok, why = b.availability()
    assert ok and "downloaded from someone/model" in why


# ---------------------------------------------------------------- capabilities

def test_capabilities_expose_provider_mode_path_and_availability(tmp_path, monkeypatch):
    monkeypatch.delenv("RETAKE_KEEPER_PROVIDER", raising=False)
    monkeypatch.setenv("RETAKE_KEEPER_MODEL_PATH", str(tmp_path / "model"))
    from models.final.final_scorer import FinalScorer

    client = TestClient(create_app(ServiceConfig(data_dir=str(tmp_path / "data")), Deps(scorer=FinalScorer(mock=True))))
    keeper = client.get("/v1/capabilities").json()["keeper"]
    assert keeper["current"] == "finetuned" and keeper["default"] == "finetuned"
    assert keeper["providers"]["finetuned"]["available"] is False                    # no model in that folder
    f = keeper["finetuned"]
    assert f["backend"] == "standalone" and f["mode"] == "4bit" and f["model_path"] == str(tmp_path / "model") and f["loaded"] is False
    assert f["on_model_error"] == "fail_closed" and f["unload_after"] is True


# ---------------------------------------------------------------- no Unsloth, no Studio imports

def test_nothing_in_the_product_code_imports_unsloth_or_studio():
    offenders = []
    for folder in ("pipeline", "service", "models", "app"):
        for p in (REPO / folder).rglob("*.py"):
            if "__pycache__" in p.parts or "unsloth_compiled_cache" in p.parts:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                t = line.strip()
                if t.startswith(("import unsloth", "from unsloth")):
                    offenders.append(f"{p.relative_to(REPO)}: {t}")
    assert not offenders, offenders
