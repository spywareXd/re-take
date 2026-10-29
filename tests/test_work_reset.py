"""Setting work aside (Ctrl+Z of a step, the Reset buttons) and bringing it back: nothing is deleted, the user's answers are untouched, the next run starts from scratch
without asking again, the old results become numbered copies, and restore serves them again. Works for a project folder and for the old per-machine cache."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from models.final.final_scorer import FinalScorer
from models.transcript import Transcript
from pipeline.step_06_llm_adjudication.llm_client import MockLLMClient
from service.app import Deps, create_app
from service.config import ServiceConfig
from service.project_store import ProjectStore

FIXTURE = Path(__file__).parent / "fixtures" / "sample_c_transcript.json"


@pytest.fixture
def api(tmp_path):
    calls = {"transcribe": 0}

    def transcribe(path):
        calls["transcribe"] += 1
        return Transcript.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))

    cfg = ServiceConfig(data_dir=str(tmp_path / "service_data"), llm_mock=True, keeper_provider="gpt-oss")
    client = TestClient(create_app(cfg, Deps(transcribe_fn=transcribe, scorer=FinalScorer(mock=True), llm_client=MockLLMClient())))
    media = tmp_path / "take.mp4"
    media.write_bytes(b"stand-in media bytes")
    return client, media, cfg, calls, tmp_path / "Proj"


def _analyze(client, media, **extra):
    r = client.post("/v1/analyze", json={"path": str(media), "duration_seconds": 25.8, **extra})
    if r.status_code != 202:
        return r, None
    job = r.json()["job"]
    end = time.time() + 60
    while time.time() < end:
        b = client.get(f"/v1/jobs/{job}").json()
        if b["state"] != "running":
            return r, {"job": job, **b}
        time.sleep(0.02)
    raise AssertionError("timeout")


def _files(d: Path) -> dict:
    """The results in a folder. project.json is left out: it is where the reset marks themselves are recorded."""
    return {p.name: p.read_bytes() for p in d.glob("*.json") if p.name not in ("project.json", "settings.json")}


def test_reset_sets_the_transcription_aside_without_deleting_and_the_next_run_starts_over(api):
    client, media, cfg, calls, proj = api
    _analyze(client, media, project=str(proj))
    data = proj / "data"
    before = _files(data)
    ProjectStore.create(proj).put("settings", {"transcription_model": "large-v2"})
    (data / "labels.jsonl").write_text('{"label": "retake"}\n', encoding="utf-8")                     # the user's answers
    r = client.post("/v1/work/reset", json={"project": str(proj), "what": "transcription"}).json()
    assert {"transcript", "units"} <= set(r["reset"]) and any(n.startswith("pairs__") for n in r["reset"]) and r["kept"] is True
    for name, content in before.items():
        assert (data / name).read_bytes() == content, f"{name} is still there, byte for byte"
    assert (data / "labels.jsonl").read_text(encoding="utf-8") == '{"label": "retake"}\n', "the user's answers are never touched"
    st = client.get("/v1/project/status", params={"project": str(proj)}).json()
    assert "transcript" in st["stale"]
    _, body = _analyze(client, media, project=str(proj))                                              # same settings: no question, it just starts over
    assert body["state"] == "done" and calls["transcribe"] == 2, "transcribed again from scratch"
    assert (data / "stale" / "transcript.1.json").read_bytes() == before["transcript.json"], "the set-aside transcript is kept as a numbered copy"
    assert ProjectStore(proj).stale() == [] and ProjectStore(proj).reset_marks() == []


def test_restore_brings_the_old_results_back_and_nothing_is_recomputed(api):
    client, media, cfg, calls, proj = api
    _analyze(client, media, project=str(proj))
    client.post("/v1/work/reset", json={"project": str(proj), "what": "transcription"})
    r = client.post("/v1/work/restore", json={"project": str(proj), "what": "transcription"}).json()
    assert "transcript" in r["restored"]
    assert ProjectStore(proj).stale() == [] and ProjectStore(proj).reset_marks() == []
    _, body = _analyze(client, media, project=str(proj))
    assert body["state"] == "done" and calls["transcribe"] == 1, "the restored transcript is used, no new transcription"


def test_a_reset_is_not_undone_by_going_back_to_the_same_model_and_a_changed_model_does_not_ask(api):
    client, media, cfg, calls, proj = api
    _analyze(client, media, project=str(proj), model="base")
    client.post("/v1/work/reset", json={"project": str(proj), "what": "transcription"})
    client.put("/v1/project/settings", json={"project": str(proj), "transcription_model": "base"})        # same model as the transcript: used to lift stale marks
    assert "transcript" in ProjectStore(proj).stale(), "a deliberate reset stays"
    r, body = _analyze(client, media, project=str(proj), model="small")                                   # a different model after a reset: nothing to replace, so no 409
    assert r.status_code == 202 and body["state"] == "done" and calls["transcribe"] == 2


def test_resetting_the_keeper_groups_leaves_the_transcription_alone(api):
    client, media, cfg, calls, proj = api
    _analyze(client, media, project=str(proj))
    groups = [{"id": "g1", "start": 0.0, "end": 3.0, "sentences": [{"id": "s1", "start": 0.0, "end": 1.0, "text": "a"}, {"id": "s2", "start": 2.0, "end": 3.0, "text": "a"}]}]
    job = client.post("/v1/adjudicate", json={"path": str(media), "groups": groups, "project": str(proj), "keeper_provider": "gpt-oss"}).json()["job"]
    end = time.time() + 30
    while time.time() < end and client.get(f"/v1/jobs/{job}").json()["state"] == "running":
        time.sleep(0.02)
    assert ProjectStore(proj).has("keepers"), "the advice is stored as a result of the project"
    r = client.post("/v1/work/reset", json={"project": str(proj), "what": "keepers"}).json()
    assert "keepers" in r["reset"] and "transcript" not in r["reset"]
    assert ProjectStore(proj).stale() == ["keepers"] or set(ProjectStore(proj).stale()) <= {"keepers", "cutlist"}
    assert ProjectStore(proj).get("transcript") is not None, "the transcription is untouched"
    assert (proj / "data" / "keepers.json").is_file(), "set aside, not deleted"


def test_the_old_per_machine_cache_can_be_reset_and_restored_too(api):
    client, media, cfg, calls, _ = api
    _analyze(client, media)                                                                             # no project: the old cache
    (folder,) = list(Path(cfg.cache_dir).iterdir())
    before = _files(folder)
    assert "transcript.json" in before
    r = client.post("/v1/work/reset", json={"path": str(media), "what": "transcription"}).json()
    assert {"transcript", "units"} <= set(r["reset"]) and r["where"] == "per-machine cache"
    for name, content in before.items():
        assert (folder / name).read_bytes() == content, "nothing deleted"
    _analyze(client, media)
    assert calls["transcribe"] == 2, "it starts over instead of finding the old transcript"
    assert (folder / "stale" / "transcript.1.json").read_bytes() == before["transcript.json"]
    client.post("/v1/work/reset", json={"path": str(media), "what": "transcription"})
    back = client.post("/v1/work/restore", json={"path": str(media), "what": "transcription"}).json()
    assert "transcript" in back["restored"]
    _analyze(client, media)
    assert calls["transcribe"] == 2, "restored: used again"


def test_bad_requests_are_400s(api):
    client, media, cfg, calls, proj = api
    assert client.post("/v1/work/reset", json={"project": str(proj), "what": "everything"}).status_code == 400
    assert client.post("/v1/work/reset", json={"what": "transcription"}).status_code == 400                      # neither project nor video
    assert client.post("/v1/work/restore", json={"path": str(media.parent / "nope.mp4"), "what": "transcription"}).status_code == 400
