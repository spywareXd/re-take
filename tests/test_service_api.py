"""API smoke test of the local retake service with a mock LLM: analyze -> poll -> resolve -> poll -> adjudicate -> cancel, and the unreadable-file failure."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from models.final.final_scorer import FinalScorer
from models.transcript import Transcript
from pipeline.step_06_llm_adjudication.llm_client import MockLLMClient
from pipeline.step_07_review import validate_document
from service.app import Deps, create_app
from service.config import ServiceConfig

FIXTURE = Path(__file__).parent / "fixtures" / "sample_c_transcript.json"


def _poll(client, job, timeout=120.0):
    end = time.time() + timeout
    while time.time() < end:
        r = client.get(f"/v1/jobs/{job}")
        assert r.status_code == 200
        body = r.json()
        if body["state"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


@pytest.fixture
def api(tmp_path):
    def transcribe(path):
        time.sleep(0.4)                                                # long enough for the cancel request below to land mid-job
        return Transcript.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))

    cfg = ServiceConfig(data_dir=str(tmp_path / "data"), llm_mock=True)
    app = create_app(cfg, Deps(transcribe_fn=transcribe, scorer=FinalScorer(mock=True), llm_client=MockLLMClient()))
    media = tmp_path / "take.mp4"
    media.write_bytes(b"stand-in media bytes")
    return TestClient(app), media, cfg


def test_service_flow_with_mock_llm(api):
    client, media, cfg = api
    assert client.get("/v1/capabilities").json() == {"adjudicate": True, "refine": True}

    # analyze -> poll
    r = client.post("/v1/analyze", json={"path": str(media), "name": "take.mp4", "duration_seconds": 25.8, "used": [[0.0, 25.8]]})
    assert r.status_code == 202
    job = r.json()["job"]
    done = _poll(client, job)
    assert done["state"] == "done" and done["fraction"] == 1.0 and "result" in done
    doc = done["result"]
    assert validate_document(doc, file_duration=25.8) == []
    assert doc["job"] == job and doc["media"]["fingerprint"]

    # resolve -> poll (repeatable, from the echoed document only)
    if doc["stage"] == "candidates":
        answers = [{"pair": p["id"], "label": "retake", "source": "user"} for p in doc["pairs"] if p["class"] == "ambiguous"]
        req = {"path": str(media), "candidates": doc, "labels": {"job": job, "labels": answers, "skipped": []}}
        for _ in range(2):
            rj = client.post("/v1/resolve", json=req)
            assert rj.status_code == 202
            groups = _poll(client, rj.json()["job"])
            assert groups["state"] == "done" and groups["result"]["stage"] == "groups" and groups["result"]["job"] == job
            assert validate_document(groups["result"], file_duration=25.8, expected_job=job) == []
        lines = Path(cfg.labels_path).read_text(encoding="utf-8").splitlines() if answers else []
        assert len(lines) == len(answers) and all(json.loads(x)["source"] == "user" for x in lines)      # saved once, before use
        groups_doc = groups["result"]
    else:
        groups_doc = doc

    # adjudicate (mock LLM) -> a plain array, never an empty keep
    ra = client.post("/v1/adjudicate", json={"path": str(media), "job": job, "groups": groups_doc["groups"]})
    assert ra.status_code == 202
    advice = _poll(client, ra.json()["job"])
    assert advice["state"] == "done" and isinstance(advice["result"], list)
    by_group = {g["id"]: {s["id"] for s in g["sentences"]} for g in groups_doc["groups"]}
    assert advice["result"], "the mock answers every group"
    for a in advice["result"]:
        assert a["keep"] and set(a["keep"]) <= by_group[a["group"]] and isinstance(a["reason"], str)

    # cancel: DELETE is idempotent; a new analyze job (fresh file, nothing cached) is stopped between stages
    media2 = media.parent / "other.mp4"
    media2.write_bytes(b"different bytes, different fingerprint")
    j2 = client.post("/v1/analyze", json={"path": str(media2), "duration_seconds": 25.8}).json()["job"]
    assert client.delete(f"/v1/jobs/{j2}").status_code == 204 and client.delete(f"/v1/jobs/{j2}").status_code == 204
    assert _poll(client, j2)["state"] == "cancelled"
    assert client.delete("/v1/jobs/j_unknown").status_code == 204
    assert client.get("/v1/jobs/j_unknown").status_code == 404 and client.get("/v1/jobs/j_unknown").json() == {"error": "Unknown job."}


def test_unreadable_file_and_bad_requests(api):
    client, media, _ = api
    job = client.post("/v1/analyze", json={"path": str(media.parent / "missing.mp4"), "duration_seconds": 10}).json()["job"]
    failed = _poll(client, job)
    assert failed["state"] == "failed" and failed["error"] == "Could not read the file" and "result" not in failed
    bad = client.post("/v1/analyze", json={"name": "x"})
    assert bad.status_code == 400 and set(bad.json()) == {"error"}
    assert client.post("/v1/resolve", content=b"not json").status_code == 400
