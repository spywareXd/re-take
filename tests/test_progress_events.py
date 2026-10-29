"""Item 2: stage events (started/progress/done/error + cached), persistence, SSE, failures at every stage, and a free event loop during a long job."""
from __future__ import annotations

import asyncio
import json
import sys
import time
import types
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from models.final.final_scorer import FinalScorer
from models.transcript import Transcript
from pipeline.step_06_llm_adjudication.llm_client import MockLLMClient
from service.app import Deps, create_app
from service.config import ServiceConfig
from service.events import EventLog, read_log

FIXTURE = Path(__file__).parent / "fixtures" / "sample_c_transcript.json"


def _fixture():
    return Transcript.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))


def _wait(client, job, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        b = client.get(f"/v1/jobs/{job}").json()
        if b["state"] != "running":
            return b
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def _make(tmp_path, transcribe):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = ServiceConfig(data_dir=str(tmp_path / "service_data"), llm_mock=True, keeper_provider="gpt-oss")
    app = create_app(cfg, Deps(transcribe_fn=transcribe, scorer=FinalScorer(mock=True), llm_client=MockLLMClient()))
    media = tmp_path / "take.mp4"
    media.write_bytes(b"stand-in media bytes")
    return TestClient(app), media


def _analyze(client, media, proj):
    job = client.post("/v1/analyze", json={"path": str(media), "duration_seconds": 25.8, "project": str(proj)}).json()["job"]
    return job, _wait(client, job)


def _events(client, job):
    return client.get(f"/v1/jobs/{job}/events").json()["events"]


def test_event_log_schema_ordering_and_error_message():
    log = EventLog("j_1")
    with log.stage("unit_splitting", "go") as st:
        st.progress(50, "half")
        st.done_message = "7 sentences"
    with pytest.raises(RuntimeError):
        with log.stage("pair_scoring"):
            raise RuntimeError("exact text: C:/x/y.mp4 is bad")
    log.cached("vad")
    evs = log.since(0)
    assert [(e["stage"], e["status"]) for e in evs] == [("unit_splitting", "started"), ("unit_splitting", "progress"), ("unit_splitting", "done"),
                                                        ("pair_scoring", "started"), ("pair_scoring", "error"), ("vad", "done")]
    assert [e["seq"] for e in evs] == [1, 2, 3, 4, 5, 6]
    assert all({"stage", "status", "message", "timestamp", "job"} <= set(e) for e in evs)
    assert evs[1]["pct"] == 50.0 and evs[2]["message"] == "7 sentences"
    assert evs[4]["message"] == "RuntimeError: exact text: C:/x/y.mp4 is bad"             # the exact message, per stage
    assert evs[5]["cached"] is True and evs[5]["message"] == "cached"
    with pytest.raises(ValueError):
        log.emit("nope", "started")
    with pytest.raises(ValueError):
        log.emit("vad", "weird")


def test_analyze_emits_stages_persists_them_and_marks_cached_stages(tmp_path):
    client, media = _make(tmp_path, lambda p: _fixture())
    proj = tmp_path / "P"
    job, body = _analyze(client, media, proj)
    assert body["state"] == "done"
    evs = client.get(f"/v1/jobs/{job}/events").json()
    stages_done = {e["stage"] for e in evs["events"] if e["status"] == "done"}
    assert {"unit_splitting", "pair_scoring"} <= stages_done and evs["finished"] is True
    assert read_log(proj / "data" / "progress.jsonl") == evs["events"]                   # persisted, identical, kept after completion
    assert client.get("/v1/project/events", params={"project": str(proj)}).json()["events"] == evs["events"]
    job2, _ = _analyze(client, media, proj)                                              # second run: everything comes from the project's files
    e2 = _events(client, job2)
    cached = {e["stage"] for e in e2 if e.get("cached")}
    assert {"transcription", "unit_splitting", "pair_scoring", "alignment", "vad"} <= cached
    assert len(read_log(proj / "data" / "progress.jsonl")) == len(evs["events"]) + len(e2)       # appended, nothing replaced


def test_events_after_parameter_and_sse_stream(tmp_path):
    client, media = _make(tmp_path, lambda p: _fixture())
    job, _ = _analyze(client, media, tmp_path / "P")
    allev = _events(client, job)
    assert client.get(f"/v1/jobs/{job}/events", params={"after": 2}).json()["events"] == allev[2:]
    r = client.get(f"/v1/jobs/{job}/events/stream")
    assert r.headers["content-type"].startswith("text/event-stream")
    ids = [int(line[4:]) for line in r.text.splitlines() if line.startswith("id: ")]
    assert ids == [e["seq"] for e in allev] and "event: end" in r.text
    r2 = client.get(f"/v1/jobs/{job}/events/stream", headers={"Last-Event-ID": "3"})
    assert [int(line[4:]) for line in r2.text.splitlines() if line.startswith("id: ")] == ids[3:]
    assert client.get("/v1/jobs/nope/events").status_code == 404


# ---- a failure at each stage, through the real WhisperX adapter with a fake whisperx module ----
def _fake_whisperx(monkeypatch, fail_at=None):
    from pipeline.step_01_transcription import audio_extractor, model_download

    def boom(stage):
        if fail_at == stage:
            raise RuntimeError(f"simulated failure in {stage}")

    class Model:
        def transcribe(self, audio, batch_size=16, print_progress=False):
            boom("vad")
            for pct in (25, 50, 100):
                print(f"Progress: {pct:.2f}%...")
            boom("transcription")
            return {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "hello there", "words": []}]}

    fw = types.ModuleType("whisperx")

    def load_model(*a, **k):
        boom("model_load")
        return Model()

    def load_align_model(**k):
        boom("alignment")
        return object(), {}

    aligned = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello there", "words": [{"word": "hello", "start": 0.0, "end": 0.5, "score": 0.9}]}]}
    fw.load_model, fw.load_audio, fw.load_align_model = load_model, lambda p: [0.0] * 16, load_align_model
    fw.align = lambda segs, *a, **k: aligned
    monkeypatch.setitem(sys.modules, "whisperx", fw)

    def fake_download(name, on_progress=None):
        boom("model_download")
        on_progress(40.0, 400_000_000, 1_000_000_000)
        on_progress(100.0, 1_000_000_000, 1_000_000_000)
        return True

    monkeypatch.setattr(model_download, "download", fake_download)

    def fake_extract(cls, media_path, output_wav_path=None, **k):
        boom("audio_extraction")
        return str(media_path)

    monkeypatch.setattr(audio_extractor.AudioExtractor, "extract_audio", classmethod(fake_extract))


def _real_transcribe_app(tmp_path):
    cfg = ServiceConfig(data_dir=str(tmp_path / "service_data"), llm_mock=True, keeper_provider="gpt-oss")
    app = create_app(cfg, Deps(scorer=FinalScorer(mock=True), llm_client=MockLLMClient()))           # no transcribe_fn: the real adapter path
    media = tmp_path / "take.mp4"
    media.write_bytes(b"stand-in media bytes")
    return TestClient(app), media


def test_full_stage_sequence_through_the_real_adapter(tmp_path, monkeypatch):
    _fake_whisperx(monkeypatch)
    client, media = _real_transcribe_app(tmp_path)
    job, body = _analyze(client, media, tmp_path / "P")
    assert body["state"] == "done", body
    evs = _events(client, job)
    first_seen = []
    for e in evs:
        if e["stage"] not in first_seen:
            first_seen.append(e["stage"])
    assert first_seen == ["audio_extraction", "model_download", "model_load", "vad", "transcription", "alignment", "unit_splitting", "pair_scoring"]
    assert [e["pct"] for e in evs if e["stage"] == "model_download" and e["status"] == "progress"] == [40.0, 100.0]
    assert [e["pct"] for e in evs if e["stage"] == "transcription" and e["status"] == "progress"] == [25.0, 50.0, 100.0]
    assert not [e for e in evs if e["status"] == "error"]


@pytest.mark.parametrize("stage", ["audio_extraction", "model_download", "model_load", "vad", "transcription", "alignment"])
def test_failure_at_each_transcription_stage(tmp_path, monkeypatch, stage):
    _fake_whisperx(monkeypatch, fail_at=stage)
    client, media = _real_transcribe_app(tmp_path)
    proj = tmp_path / "P"
    job, body = _analyze(client, media, proj)
    errs = [e for e in _events(client, job) if e["status"] == "error"]
    if stage == "model_download":                                                        # a failed download hook is not fatal: loading goes on
        assert body["state"] == "done" and not errs
        return
    assert body["state"] == "failed" and "simulated failure" not in body["error"]       # the editor-facing error stays sanitised
    assert [e["stage"] for e in errs] == [stage] and errs[0]["message"] == f"RuntimeError: simulated failure in {stage}"
    assert any(e["message"] == errs[0]["message"] for e in read_log(proj / "data" / "progress.jsonl"))          # and the log keeps it


@pytest.mark.parametrize("stage,attr", [("unit_splitting", "_load_units"), ("pair_scoring", "_score_pairs")])
def test_failure_in_unit_splitting_and_pair_scoring(tmp_path, monkeypatch, stage, attr):
    import pipeline.step_07_review.json_export as je

    def boom(*a, **k):
        raise ValueError(f"simulated {stage} failure")

    client, media = _make(tmp_path, lambda p: _fixture())
    monkeypatch.setattr(je, attr, boom)
    job, body = _analyze(client, media, tmp_path / "P")
    errs = [e for e in _events(client, job) if e["status"] == "error"]
    assert body["state"] == "failed" and [e["stage"] for e in errs] == [stage] and errs[0]["message"] == f"ValueError: simulated {stage} failure"


def test_failure_in_keeper_selection_and_cutlist_build(tmp_path, monkeypatch):
    client, media = _make(tmp_path, lambda p: _fixture())
    proj = tmp_path / "P"
    import service.app as sapp

    def bad_advice(*a, **k):
        raise RuntimeError("simulated keeper failure")

    monkeypatch.setattr(sapp, "adjudicate_groups", bad_advice)
    groups = [{"id": "g1", "sentences": [{"id": "s1", "start": 0.0, "end": 1.0, "text": "a"}, {"id": "s2", "start": 2.0, "end": 3.0, "text": "a"}]}]
    job = client.post("/v1/adjudicate", json={"path": str(media), "groups": groups, "project": str(proj), "keeper_provider": "gpt-oss"}).json()["job"]
    assert _wait(client, job)["state"] == "failed"
    errs = [(e["stage"], e["message"]) for e in _events(client, job) if e["status"] == "error"]
    assert errs == [("keeper_selection", "RuntimeError: simulated keeper failure")]

    def bad_refine(*a, **k):
        raise ValueError("simulated cut failure")

    monkeypatch.setattr(sapp, "refine_ranges", bad_refine)
    r = client.post("/v1/refine", json={"path": str(media), "remove": ["s1"], "project": str(proj)})
    assert r.status_code == 400
    log = read_log(proj / "data" / "progress.jsonl")
    assert any(e["stage"] == "cutlist_build" and e["status"] == "error" and e["message"] == "ValueError: simulated cut failure" for e in log)


def test_event_loop_stays_free_during_a_long_job(tmp_path):
    """A 1.5 s transcription runs while 30 requests hit the same app: none may take long, and the SSE stream delivers events while it runs."""
    def slow(p):
        time.sleep(1.5)
        return _fixture()

    cfg = ServiceConfig(data_dir=str(tmp_path / "sd"), llm_mock=True, keeper_provider="gpt-oss")
    app = create_app(cfg, Deps(transcribe_fn=slow, scorer=FinalScorer(mock=True), llm_client=MockLLMClient()))
    media = tmp_path / "take.mp4"
    media.write_bytes(b"x")

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            await c.get("/v1/capabilities")                    # warm-up: the first call imports the keeper's libraries (slow, and unrelated to the event loop)
            job = (await c.post("/v1/analyze", json={"path": str(media), "duration_seconds": 25.8, "project": str(tmp_path / "P")})).json()["job"]
            worst, streamed = 0.0, []

            async def stream():
                async with c.stream("GET", f"/v1/jobs/{job}/events/stream") as r:
                    async for line in r.aiter_lines():
                        if line.startswith("id: "):
                            streamed.append(time.monotonic())

            task = asyncio.create_task(stream())
            for _ in range(30):
                t = time.monotonic()
                assert (await c.get("/v1/capabilities")).status_code == 200
                assert (await c.get(f"/v1/jobs/{job}/events")).status_code == 200
                worst = max(worst, time.monotonic() - t)
                await asyncio.sleep(0.05)
            await asyncio.wait_for(task, 30)
            return worst, streamed

    worst, streamed = asyncio.run(go())
    assert worst < 0.5, f"a request waited {worst:.2f}s while a job was running"
    assert len(streamed) >= 3
