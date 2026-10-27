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

    class NoStudio:                                                    # the finetuned keeper is not used here; never touch the network
        model = "fake"

        def availability(self):
            return False, "test: no decision model"

    cfg = ServiceConfig(data_dir=str(tmp_path / "data"), llm_mock=True, keeper_provider="gpt-oss")      # this flow tests the LLM keeper path
    app = create_app(cfg, Deps(transcribe_fn=transcribe, scorer=FinalScorer(mock=True), llm_client=MockLLMClient(), keeper_backend=NoStudio()))
    media = tmp_path / "take.mp4"
    media.write_bytes(b"stand-in media bytes")
    return TestClient(app), media, cfg


def test_unexpected_failure_is_logged_in_full_but_not_leaked_to_the_editor(tmp_path, caplog):
    def broken(path):
        raise RuntimeError("secret detail Z:/private/clip.mp4 exploded")

    cfg = ServiceConfig(data_dir=str(tmp_path / "data"), llm_mock=True)
    client = TestClient(create_app(cfg, Deps(transcribe_fn=broken, scorer=FinalScorer(mock=True), llm_client=MockLLMClient())))
    media = tmp_path / "take.mp4"
    media.write_bytes(b"stand-in media bytes")
    with caplog.at_level("ERROR"):
        job = client.post("/v1/analyze", json={"path": str(media), "duration_seconds": 25.8}).json()["job"]
        failed = _poll(client, job)
    assert failed["state"] == "failed"
    assert f"job {job}" in failed["error"] and "service log" in failed["error"]            # the key into the log
    assert "secret detail" not in failed["error"] and "Traceback" not in failed["error"] and "private" not in failed["error"]
    assert not failed["error"].startswith("The retake service could not finish")            # the editor adds that sentence itself
    record = next(r for r in caplog.records if job in r.getMessage())
    assert record.exc_info and "secret detail" in "".join(__import__("traceback").format_exception(*record.exc_info))


def test_log_file_gets_service_and_pipeline_records_with_tracebacks(tmp_path):
    import logging

    from service.logsetup import log_path, setup_logging

    path = setup_logging(tmp_path)
    try:
        assert path == log_path(tmp_path) and path.parent.is_dir()
        setup_logging(tmp_path)                                                              # twice: still one file handler
        assert sum(1 for h in logging.getLogger().handlers if getattr(h, "_retake_service_file_handler", False)) == 1
        logging.getLogger("service.jobs").info("hello from the service")
        logging.getLogger("pipeline.step_01_transcription").warning("hello from the pipeline")
        try:
            raise ValueError("kaboom")
        except ValueError:
            logging.getLogger("service.jobs").exception("job j_test failed")
        for h in logging.getLogger().handlers:
            h.flush()
        text = path.read_text(encoding="utf-8")
        assert "hello from the service" in text and "hello from the pipeline" in text
        assert "job j_test failed" in text and "Traceback" in text and "ValueError: kaboom" in text
    finally:
        for h in list(logging.getLogger().handlers):
            if getattr(h, "_retake_service_file_handler", False):
                logging.getLogger().removeHandler(h)
                h.close()


def test_refine_errors_and_ranges(api, monkeypatch):
    client, media, _ = api
    body = {"path": str(media), "remove": ["x"], "mode": "anchored"}
    r = client.post("/v1/refine", json=body)
    assert r.status_code == 400 and "not been analysed" in r.json()["error"]

    done = _poll(client, client.post("/v1/analyze", json={"path": str(media), "duration_seconds": 25.8}).json()["job"])
    ids = [s["id"] for s in done["result"]["sentences"]]
    assert ids
    r = client.post("/v1/refine", json={**body, "remove": ["nope:99"]})
    assert r.status_code == 400 and "unknown sentence id" in r.json()["error"]
    r = client.post("/v1/refine", json={**body, "remove": [ids[0]], "mode": "everything"})
    assert r.status_code == 400 and "mode must be one of" in r.json()["error"]

    # the stand-in file has no audio or video, so stub the media probes; the cut-mode code itself runs for real
    import pipeline.step_08_video_edit as edit
    monkeypatch.setattr(edit, "probe_media", lambda p: (25.8, None, None))
    import numpy as np
    monkeypatch.setattr(edit, "load_envelope", lambda p: np.full(2600, -40.0))
    for mode in ("sentences", "anchored", "retake_only"):
        r = client.post("/v1/refine", json={**body, "remove": [ids[0]], "mode": mode})
        assert r.status_code == 200, r.text
        ranges = r.json()["ranges"]
        assert ranges and all(0.0 <= x["start"] < x["end"] <= 25.9 for x in ranges)
        assert all(a["end"] <= b["start"] for a, b in zip(ranges, ranges[1:]))


def test_service_flow_with_mock_llm(api):
    client, media, cfg = api
    caps = client.get("/v1/capabilities").json()
    assert {k: caps[k] for k in ("adjudicate", "refine")} == {"adjudicate": True, "refine": True}
    assert caps["keeper"]["default"] == "gpt-oss" and caps["keeper"]["providers"]["finetuned"]["available"] is False

    # analyze -> poll
    r = client.post("/v1/analyze", json={"path": str(media), "name": "take.mp4", "duration_seconds": 25.8, "used": [[0.0, 25.8]]})
    assert r.status_code == 202
    job = r.json()["job"]
    done = _poll(client, job)
    assert done["state"] == "done" and done["fraction"] == 1.0 and "result" in done
    doc = done["result"]
    assert validate_document(doc, file_duration=25.8) == []
    assert doc["job"] == job and "fingerprint" not in doc["media"]

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
