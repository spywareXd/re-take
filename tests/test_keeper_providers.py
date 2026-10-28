"""Keeper providers and the hybrid policy (Step 06): rule layers, provider switching, gpt-oss regression, no network for the default provider.
No GPU, no API key and no Studio needed: the decision model is a fake (a function, or a tiny HTTP server on 127.0.0.1)."""
from __future__ import annotations

import copy
import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.step_06_llm_adjudication import MockLLMClient, run_stage6
from pipeline.step_06_llm_adjudication.config import AdjudicationConfig
from pipeline.step_06_llm_adjudication.keeper_policy import (KeeperConfig, build_decision_request, cue_score, cue_set, decide_finetuned, is_cue_group, is_cue_take,
                                                             is_exact_group, normalise)
from pipeline.step_06_llm_adjudication.keeper_providers import (FinetunedProvider, GptOssProvider, StudioDecisionBackend, choose_provider_name,
                                                                make_provider, run_keeper_stage)

REPO = Path(__file__).resolve().parents[1]


def U(uid, start, text):
    return {"unit_id": uid, "start": float(start), "end": float(start) + 1.0, "raw_text": text}


UNITS = [U("1:0", 0, "Hello and welcome."), U("2:0", 2, "So the main idea is"), U("3:0", 4, "So the main idea is that it works."),
         U("4:0", 6, "Thanks for watching.")]
SPAN = {"span_id": "s1", "unit_ids": ["2:0", "3:0"], "start": 2.0, "end": 5.0, "candidate_pairs": [], "classification_summary": {},
        "needs_adjudication": True}


def fixed(probs):
    calls = []

    def predict(state, questions):
        calls.append((state, questions))
        return {q: probs[q[len("keep_"):]] for q in questions}
    predict.calls = calls
    return predict


def decide(members, probs=None, cfg=None, predict=None):
    p = predict or fixed(probs or {})
    return decide_finetuned("g", members, None, None, p, cfg or KeeperConfig()), p


# ---------------------------------------------------------------- policy layers

def test_cue_rule_keeps_none_without_the_model():
    d, p = decide([U("a", 0, "Okay, here we go."), U("b", 2, "Take two!")])
    assert d.keep == [] and d.decided_by == "cue_rule" and d.status == "resolved" and not p.calls
    f = d.span_fields()
    assert f["adjudication_status"] == "resolved" and f["keeper_unit_id"] is None and f["units_to_remove"] == ["a", "b"] and f["keep_none"] is True


def test_cue_rule_needs_every_take_to_be_a_cue():
    assert not is_cue_group(["Okay, here we go.", "So the main idea is that it works."], KeeperConfig())
    assert is_cue_group(["let me start from the top"], KeeperConfig())                  # a natural cue from the training folds (keeper_cues.json)


def test_cue_rule_is_a_lexical_band():
    cfg = KeeperConfig(extra_cues=["Okay, here we go.", "Let me start again."])
    assert cue_score("Here we go.", cue_set(cfg)) >= 0.8 and is_cue_group(["Here we go!", "okay here we go"], cfg)       # near matches count
    assert not is_cue_take("[redacted sentence].", cfg)                         # real content ending in a cue: below the band
    assert not is_cue_take("Okay, here we go, so today we are going to build a shelf.", cfg)   # too long to be a cue
    assert not is_cue_group(["Let me start again.", "Let me explain the settings."], cfg)
    assert is_cue_group(["Let me start again"], cfg) and not is_cue_group(["Let me start again"], KeeperConfig(extra_cues=cfg.extra_cues, cue_threshold=1.01))


def test_exact_gate_keeps_the_last_without_the_model():
    d, p = decide([U("a", 0, "It works."), U("b", 2, "it works")])
    assert d.keep == ["b"] and d.decided_by == "exact_gate" and not p.calls
    assert is_exact_group(["It works!", "it  works"]) and not is_exact_group(["It works.", "It worked."]) and normalise("It's, OK!") == "it's ok"


def test_model_keep_set_and_multi_keepers():
    d, p = decide(UNITS[1:3], {"2:0": 0.02, "3:0": 0.98})
    assert d.keep == ["3:0"] and d.decided_by == "model" and d.status == "resolved" and len(p.calls) == 1
    d2, _ = decide(UNITS[1:3], {"2:0": 0.97, "3:0": 0.99}, cfg=KeeperConfig(unclear_default=False))
    f = d2.span_fields()
    assert d2.keep == ["2:0", "3:0"] and f["units_to_remove"] == [] and f["keeper_unit_ids"] == ["2:0", "3:0"]
    # with the unclear layer on, two confident keepers are "within 0.15" (decision_eval's margin rule): keep the last, flagged
    d3, _ = decide(UNITS[1:3], {"2:0": 0.97, "3:0": 0.99})
    assert d3.keep == ["3:0"] and d3.status == "unclear" and d3.extra["model_keep"] == ["2:0", "3:0"]


def test_empty_set_fallback_keeps_the_last():
    d, _ = decide(UNITS[1:3], {"2:0": 0.01, "3:0": 0.02})
    assert d.keep == ["3:0"] and d.decided_by == "empty_fallback" and d.probs == {"2:0": 0.01, "3:0": 0.02}


def test_unclear_keeps_the_last_and_flags():
    d, _ = decide(UNITS[1:3], {"2:0": 0.9, "3:0": 0.1}, )
    assert d.keep == ["2:0"] and d.status == "resolved"                               # confident earlier keeper: kept
    d, _ = decide(UNITS[1:3], {"2:0": 0.58, "3:0": 0.2})                              # max p below 0.6: unclear
    assert d.keep == ["3:0"] and d.status == "unclear" and d.extra["model_keep"] == ["2:0"] and d.span_fields()["unclear"] is True


def test_model_error_falls_back_to_keep_last():
    def boom(state, questions):
        raise ConnectionError("studio down")
    d, _ = decide(UNITS[1:3], predict=boom, cfg=KeeperConfig(on_model_error="keep_last"))      # the opt-in; the default is fail_closed
    assert d.keep == ["3:0"] and d.decided_by == "model_error_fallback" and "studio down" in d.error
    assert d.span_fields()["adjudication_status"] == "resolved" and d.span_fields()["keeper_error"]


def test_layers_can_be_switched_off():
    off = KeeperConfig(cue_rule=False, exact_gate=False, empty_fallback=False, unclear_default=False)
    d, p = decide([U("a", 0, "Take two."), U("b", 2, "Take two.")], {"a": 0.1, "b": 0.2}, off)
    assert p.calls and d.keep == [] and d.decided_by == "model"                          # no cue rule, no exact gate, no empty fallback
    d, _ = decide(UNITS[1:3], {"2:0": 0.58, "3:0": 0.2}, off)
    assert d.keep == ["2:0"] and d.status == "unclear"                                   # unclear flagged, model's answer kept


def test_decision_request_uses_the_training_builders():
    from training.build_decision_data import decision_rules
    state, questions = build_decision_request("g", UNITS[1:3], UNITS[0], UNITS[3])
    assert state["rules"] == decision_rules() and [t["id"] for t in state["takes"]] == ["2:0", "3:0"]
    assert state["context"]["before"]["id"] == "1:0" and state["context"]["after"]["id"] == "4:0"
    assert questions["keep_3:0"]["type"] == "noul" and "take 2 of 2" in questions["keep_3:0"]["instructions"]


def test_decision_request_matches_the_scored_rows_when_available():
    """The runtime rows equal the rows the fold models were scored on (needs the git-ignored runs/keeper_loop data)."""
    dev = REPO / "runs" / "keeper_loop" / "holdout1_fold1" / "data" / "dev.jsonl"
    if not dev.exists():
        pytest.skip("runs/keeper_loop not present")
    sys.path.insert(0, str(REPO / "runs" / "keeper_loop" / "scripts"))
    import common as C
    from training.build_keeper_sft_v2 import context_for, group_units
    rows = {r["group_id"]: r for r in C.load_groups()}
    keys = (dev.with_suffix(".keys.txt")).read_text(encoding="utf-8").split()
    cache: dict = {}
    for k, line in list(zip(keys, dev.read_text(encoding="utf-8").splitlines()))[:10]:
        d = json.loads(line)
        m = group_units(rows[k])
        b, a, _ = context_for(rows[k], m, cache)
        state, questions = build_decision_request(k, m, b, a)
        assert state == json.loads(d["state"]) and questions == json.loads(d["questions"])


# ---------------------------------------------------------------- provider switching

def test_provider_choice_precedence(monkeypatch):
    monkeypatch.delenv("RETAKE_KEEPER_PROVIDER", raising=False)
    assert KeeperConfig.from_env().provider == "finetuned" and choose_provider_name(None, KeeperConfig()) == "finetuned"
    monkeypatch.setenv("RETAKE_KEEPER_PROVIDER", "gpt-oss")
    cfg = KeeperConfig.from_env()
    assert choose_provider_name(None, cfg) == "gpt-oss" and choose_provider_name("finetuned", cfg) == "finetuned"      # request / CLI wins
    with pytest.raises(ValueError, match="unknown keeper provider"):
        choose_provider_name("rules", cfg)


def test_cli_has_the_keeper_provider_flag():
    out = subprocess.run([sys.executable, "-m", "pipeline.run_pipeline", "--help"], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert "--keeper-provider" in out.stdout and "finetuned" in out.stdout and "gpt-oss" in out.stdout


class FakeBackend:
    model = "fake-decision-model"

    def __init__(self, ok=True, probs=None):
        self.ok, self.probs, self.calls = ok, probs, 0

    def availability(self):
        return (True, "fake model loaded") if self.ok else (False, "Studio is not running")

    def predict(self, state, questions):
        self.calls += 1
        ids = [q[len("keep_"):] for q in questions]
        return {f"keep_{u}": (self.probs or {}).get(u, 1.0 if u == ids[-1] else 0.0) for u in ids}


def _service(tmp_path, backend, provider=None, llm_client=None):
    from models.final.final_scorer import FinalScorer
    from service.app import Deps, create_app
    from service.config import ServiceConfig
    cfg = ServiceConfig(data_dir=str(tmp_path / "data"), keeper_provider=provider, llm_mock=llm_client is not None)
    app = create_app(cfg, Deps(scorer=FinalScorer(mock=True), llm_client=llm_client, keeper_backend=backend))
    return TestClient(app), cfg


GROUPS = [{"id": "g1", "start": 2.0, "end": 5.0, "sentences": [{"id": "2:0", "start": 2.0, "end": 3.0, "text": "So the main idea is"},
                                                              {"id": "3:0", "start": 4.0, "end": 5.0, "text": "So the main idea is that it works."}]},
          {"id": "g2", "start": 6.0, "end": 9.0, "sentences": [{"id": "5:0", "start": 6.0, "end": 7.0, "text": "Okay, here we go."},
                                                              {"id": "6:0", "start": 8.0, "end": 9.0, "text": "Take two."}]}]


def _poll(client, job):
    for _ in range(400):
        b = client.get(f"/v1/jobs/{job}").json()
        if b["state"] != "running":
            return b
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_capabilities_and_default_finetuned_advice(tmp_path, monkeypatch):
    monkeypatch.delenv("RETAKE_KEEPER_PROVIDER", raising=False)
    backend = FakeBackend()
    client, cfg = _service(tmp_path, backend)
    caps = client.get("/v1/capabilities").json()
    assert caps["adjudicate"] is True and caps["keeper"]["default"] == "finetuned"
    assert caps["keeper"]["providers"]["finetuned"] == {"available": True, "reason": "fake model loaded"}
    assert caps["keeper"]["providers"]["gpt-oss"]["available"] in (True, False)
    media = tmp_path / "x.mp4"
    media.write_bytes(b"x")
    res = _poll(client, client.post("/v1/adjudicate", json={"path": str(media), "job": "j1", "groups": GROUPS}).json()["job"])
    assert res["state"] == "done"
    advice = {a["group"]: a for a in res["result"]}
    assert set(advice) == {"g1"}                                                          # g2 is a cue group: keep none, left out (keep is never empty)
    assert advice["g1"]["keep"] == ["3:0"] and advice["g1"]["decided_by"] == "model" and advice["g1"]["provider"] == "finetuned"
    rec = [json.loads(x) for x in Path(cfg.keeper_decisions_path).read_text(encoding="utf-8").splitlines()]
    assert {r["group"]: r["decided_by"] for r in rec} == {"g1": "model", "g2": "cue_rule"} and all(r["model"] == "fake-decision-model" for r in rec)


def test_unavailable_provider_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("RETAKE_KEEPER_PROVIDER", raising=False)
    client, _ = _service(tmp_path, FakeBackend(ok=False))
    media = tmp_path / "x.mp4"
    media.write_bytes(b"x")
    assert client.get("/v1/capabilities").json()["adjudicate"] is False
    r = client.post("/v1/adjudicate", json={"path": str(media), "job": "j", "groups": GROUPS, "keeper_provider": "finetuned"})
    assert r.status_code == 400 and "finetuned" in r.json()["error"] and "Studio is not running" in r.json()["error"]
    r = client.post("/v1/adjudicate", json={"path": str(media), "job": "j", "groups": GROUPS, "keeper_provider": "rules"})
    assert r.status_code == 400 and "keeper_provider" in r.json()["error"]


def test_request_field_selects_gpt_oss(tmp_path, monkeypatch):
    monkeypatch.delenv("RETAKE_KEEPER_PROVIDER", raising=False)
    backend = FakeBackend()
    client, _ = _service(tmp_path, backend, llm_client=MockLLMClient())
    media = tmp_path / "x.mp4"
    media.write_bytes(b"x")
    res = _poll(client, client.post("/v1/adjudicate", json={"path": str(media), "job": "j", "groups": GROUPS, "keeper_provider": "gpt-oss"}).json()["job"])
    assert res["state"] == "done" and backend.calls == 0 and {a["group"] for a in res["result"]} == {"g1", "g2"}
    assert all(set(a) == {"group", "keep", "reason"} for a in res["result"])               # the old advice shape, unchanged


# ---------------------------------------------------------------- gpt-oss regression

def test_gpt_oss_provider_output_equals_the_old_path():
    spans = [copy.deepcopy(SPAN), {**copy.deepcopy(SPAN), "span_id": "s2", "unit_ids": ["1:0", "4:0"]}]
    old = run_stage6(copy.deepcopy(spans), all_units_by_id=UNITS, config=AdjudicationConfig(provider="mock"), llm_client=MockLLMClient())
    prov = make_provider("gpt-oss", KeeperConfig(), AdjudicationConfig(provider="mock"), llm_client=MockLLMClient())
    new = run_keeper_stage(copy.deepcopy(spans), UNITS, prov)
    assert new == old


def test_gpt_oss_rules_are_off_by_default_and_optional():
    class Recording(MockLLMClient):
        def __init__(self):
            super().__init__()
            self.n = 0

        def generate_json(self, prompt):
            self.n += 1
            return super().generate_json(prompt)
    units = UNITS + [U("7:0", 10, "It works."), U("8:0", 12, "It works.")]
    span = {**copy.deepcopy(SPAN), "span_id": "s3", "unit_ids": ["7:0", "8:0"]}
    client = Recording()
    run_keeper_stage([span], units, GptOssProvider(KeeperConfig(), AdjudicationConfig(provider="mock"), client))
    assert client.n == 1                                                                  # exact group still goes to the LLM
    client = Recording()
    out = run_keeper_stage([span], units, GptOssProvider(KeeperConfig(rules_with_gpt_oss=True), AdjudicationConfig(provider="mock"), client))
    assert client.n == 0 and out[0]["decided_by"] == "exact_gate" and out[0]["keeper_unit_id"] == "8:0"


# ---------------------------------------------------------------- no network for the default provider

class _StudioHandler(BaseHTTPRequestHandler):
    hits: list = []

    def log_message(self, format, *args):
        pass

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        _StudioHandler.hits.append(self.path)
        self._send({"models": [{"name": "clef-ft:test", "available": True}]})

    def do_POST(self):
        _StudioHandler.hits.append(self.path)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert isinstance(body["state"], dict)                                            # state sent as an object, never a string
        qs = list(body["questions"])
        self._send({"model": body["model"], "answers": {q: {"noul": 0.99 if i == len(qs) - 1 else 0.01} for i, q in enumerate(qs)}})


def test_default_provider_makes_no_external_calls(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("RETAKE_KEEPER_PROVIDER", raising=False)
    monkeypatch.setenv("UNSLOTH_API_KEY", "test-key")
    server = HTTPServer(("127.0.0.1", 0), _StudioHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    real_connect = socket.socket.connect
    seen = []

    def guarded(self, address):
        seen.append(address)
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"external connection to {address}")
        return real_connect(self, address)
    monkeypatch.setattr(socket.socket, "connect", guarded)
    try:
        cfg = KeeperConfig.from_env(studio_url=f"http://127.0.0.1:{server.server_port}", studio_model="clef-ft:test")
        prov = make_provider(choose_provider_name(None, cfg), cfg, backend=StudioDecisionBackend(cfg.studio_url, cfg.studio_model))      # the Studio backend is no longer the default
        assert isinstance(prov, FinetunedProvider) and prov.availability()[0]
        out = run_keeper_stage([copy.deepcopy(SPAN)], UNITS, prov)
    finally:
        server.shutdown()
    assert out[0]["decided_by"] == "model" and out[0]["keeper_unit_id"] == "3:0" and out[0]["keeper_model"] == "clef-ft:test"
    assert seen and all(a[0] == "127.0.0.1" for a in seen) and "/v1/systemone" in _StudioHandler.hits


def test_backend_refuses_non_local_hosts():
    with pytest.raises(ValueError, match="localhost"):
        StudioDecisionBackend("https://example.com", "m")


# ---------------------------------------------------------------- keep-last provider and the service-wide switch

def test_keep_last_provider_keeps_the_last_take():
    prov = make_provider("keep-last", KeeperConfig())
    out = run_keeper_stage([copy.deepcopy(SPAN)], UNITS, prov)
    assert prov.availability()[0] and out[0]["keeper_unit_id"] == "3:0" and out[0]["units_to_remove"] == ["2:0"]
    assert out[0]["decided_by"] == "keep_last" and out[0]["keeper_provider"] == "keep-last" and "unit_probs" not in out[0]


def test_switch_keeper_provider_endpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("RETAKE_KEEPER_PROVIDER", raising=False)
    backend = FakeBackend()
    client, _ = _service(tmp_path, backend)
    k = client.get("/v1/keeper").json()
    assert k["current"] == k["default"] == "finetuned" and set(k["providers"]) == {"finetuned", "gpt-oss", "keep-last"}
    assert k["providers"]["keep-last"]["available"] is True
    r = client.put("/v1/keeper", json={"provider": "keep-last"})
    assert r.status_code == 200 and r.json()["current"] == "keep-last" and r.json()["default"] == "finetuned"
    assert client.get("/v1/capabilities").json()["keeper"]["current"] == "keep-last"
    media = tmp_path / "x.mp4"
    media.write_bytes(b"x")
    res = _poll(client, client.post("/v1/adjudicate", json={"path": str(media), "job": "j", "groups": GROUPS}).json()["job"])
    assert backend.calls == 0 and {a["group"]: (a["keep"], a["decided_by"], a["provider"]) for a in res["result"]} == {
        "g1": (["3:0"], "keep_last", "keep-last"), "g2": (["6:0"], "keep_last", "keep-last")}
    # a request field still overrides the service-wide choice
    res = _poll(client, client.post("/v1/adjudicate", json={"path": str(media), "job": "j", "groups": GROUPS, "keeper_provider": "finetuned"}).json()["job"])
    assert backend.calls == 1 and {a["group"] for a in res["result"]} == {"g1"}
    assert client.put("/v1/keeper", json={"provider": "rules"}).status_code == 400


def test_switch_to_an_unavailable_provider_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("RETAKE_KEEPER_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client, _ = _service(tmp_path, FakeBackend())
    r = client.put("/v1/keeper", json={"provider": "gpt-oss"})
    assert r.status_code == 400 and "gpt-oss" in r.json()["error"] and client.get("/v1/keeper").json()["current"] == "finetuned"
