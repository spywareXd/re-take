"""Item 1: the model list comes from the installed faster-whisper, cache checks never download, VRAM warnings, compute-type fallback, and a changed model never
silently replaces an existing transcript (409, stale marks, numbered copies)."""
from __future__ import annotations

import json
import time
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from models.final.final_scorer import FinalScorer
from models.transcript import Transcript
from pipeline.step_01_transcription import model_catalog as mc
from pipeline.step_01_transcription import model_download
from pipeline.step_01_transcription.whisperx_transcriber import WhisperXTranscriber
from pipeline.step_06_llm_adjudication.llm_client import MockLLMClient
from service.app import Deps, create_app
from service.config import ServiceConfig
from service.project_store import ProjectStore

FIXTURE = Path(__file__).parent / "fixtures" / "sample_c_transcript.json"


def _fixture():
    return Transcript.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_list_is_built_from_the_installed_faster_whisper(monkeypatch):
    import faster_whisper.utils as fu
    monkeypatch.setattr(fu, "_MODELS", {"tiny": "a/tiny", "base": "a/base", "small": "a/small", "large-v2": "a/lv2", "distil-large-v3": "a/d"})
    assert list(mc.supported()) == ["tiny", "base", "small", "large-v2"]                 # medium / large-v3 / turbo are not in this "install": not listed; distil is not offered
    monkeypatch.setattr(fu, "_MODELS", {**fu._MODELS, "large-v3-turbo": "a/turbo"})
    assert mc.supported()["turbo"] == "large-v3-turbo"                                   # turbo appears when the install knows it, under whichever name it uses
    with pytest.raises(mc.UnknownModel):
        mc.resolve("medium")


def test_real_install_lists_the_wanted_sizes():
    names = list(mc.supported())
    assert {"tiny", "base", "small", "medium", "large-v2", "large-v3"} <= set(names)
    assert mc.resolve(None) == "large-v2"                                                # the default is what the pipeline uses today


def test_downloaded_check_never_touches_the_network(monkeypatch):
    import huggingface_hub
    calls = []

    def fake(repo, **kw):
        calls.append(kw)
        if not kw.get("local_files_only"):
            raise AssertionError("network access attempted")
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    assert model_download.is_downloaded("large-v2") is False
    assert calls and all(c.get("local_files_only") is True for c in calls)
    cat = mc.catalog()
    assert all(m["downloaded"] is False for m in cat["models"]) and all(c.get("local_files_only") is True for c in calls)


def test_vram_warning_warns_but_never_blocks():
    gpu8 = {"name": "x", "total_gb": 8.0, "free_gb": 7.5}
    assert mc.vram_warning("small", gpu8) is None
    assert mc.vram_warning("large-v3", gpu8) is None                                     # 5.5 GB estimate fits an 8 GB card with 7.5 GB free
    assert "free" in mc.vram_warning("large-v3", {"name": "x", "total_gb": 8.0, "free_gb": 4.0})
    assert "run out of memory" in mc.vram_warning("large-v3", {"name": "x", "total_gb": 4.0, "free_gb": 4.0})
    assert "CPU" in mc.vram_warning("small", None)
    d = mc.describe("large-v3", "large-v3", {"name": "x", "total_gb": 4.0, "free_gb": 4.0})
    assert d["warning"] and d["approx"] is True                                          # still listed and selectable


def test_compute_type_falls_back_when_the_device_refuses(monkeypatch):
    tried = []

    def load_model(name, device, compute_type=None):
        tried.append(compute_type)
        if compute_type == "int8_float16":
            raise ValueError("Requested int8_float16 compute type, but the target device does not support it")
        return object()

    fw = types.SimpleNamespace(load_model=load_model)
    t = WhisperXTranscriber(model_name="small", device="cuda", compute_type="int8_float16")
    t._load(fw, "int8_float16")
    assert tried == ["int8_float16", "float16"] and t.loaded_compute_type == "float16"
    tried.clear()
    WhisperXTranscriber(model_name="small", device="cuda", compute_type="auto")._load(fw, "auto")
    assert tried == ["int8_float16", "float16"]                                          # auto = the candidate order
    assert mc.compute_candidates("cpu") == ("int8", "float32")


@pytest.fixture
def api(tmp_path):
    cfg = ServiceConfig(data_dir=str(tmp_path / "sd"), llm_mock=True, keeper_provider="gpt-oss")
    app = create_app(cfg, Deps(transcribe_fn=lambda p: _fixture(), scorer=FinalScorer(mock=True), llm_client=MockLLMClient()))
    media = tmp_path / "take.mp4"
    media.write_bytes(b"stand-in media bytes")
    return TestClient(app), media, tmp_path / "P"


def _run(client, media, proj, **extra):
    r = client.post("/v1/analyze", json={"path": str(media), "duration_seconds": 25.8, "project": str(proj), **extra})
    if r.status_code != 202:
        return r, None
    job = r.json()["job"]
    end = time.time() + 60
    while time.time() < end:
        b = client.get(f"/v1/jobs/{job}").json()
        if b["state"] != "running":
            return r, b
        time.sleep(0.02)
    raise AssertionError("timeout")


def test_capabilities_lists_models(api):
    client, _, _ = api
    t = client.get("/v1/capabilities").json()["transcription"]
    assert t["default"] == "large-v2" and any(m["name"] == "large-v2" and m["default"] for m in t["models"])
    assert {"downloaded", "download_gb", "vram_gb", "relative_speed", "approx"} <= set(t["models"][0])


def test_default_model_is_used_and_saved_and_unknown_is_rejected(api):
    client, media, proj = api
    r, body = _run(client, media, proj)
    assert body["state"] == "done"
    assert ProjectStore(proj).settings()["transcription_model"] == "large-v2"
    assert ProjectStore(proj).get("transcript_meta") == {"model": "large-v2"}
    r2, _ = _run(client, media, proj, model="gigantic")
    assert r2.status_code == 400 and "not a model this installation supports" in r2.json()["error"]


def test_changed_model_never_silently_replaces_the_transcript(api):
    client, media, proj = api
    _run(client, media, proj, model="base")
    store = ProjectStore(proj)
    before = (proj / "data" / "transcript.json").read_bytes()
    r, _ = _run(client, media, proj, model="small")
    assert r.status_code == 409 and r.json()["code"] == "model_changed" and r.json()["transcript_model"] == "base" and "transcript" in r.json()["stale"]
    assert (proj / "data" / "transcript.json").read_bytes() == before                       # nothing replaced
    assert {"transcript", "units"} <= set(store.stale())                                    # marked stale, still on disk
    r, body = _run(client, media, proj, model="small", replace_existing=True)               # the confirmed re-transcription
    assert body["state"] == "done"
    assert (proj / "data" / "stale" / "transcript.1.json").read_bytes() == before            # the old transcript is kept as a numbered copy
    assert store.get("transcript_meta") == {"model": "small"} and store.stale() == []


def test_settings_change_marks_stale_and_asks(api):
    client, media, proj = api
    _run(client, media, proj, model="base")
    r = client.put("/v1/project/settings", json={"project": str(proj), "transcription_model": "medium"}).json()
    assert "transcript" in r["newly_stale"] and r["ask"] and r["settings"]["transcription_model"] == "medium"
    st = client.get("/v1/project/status", params={"project": str(proj)}).json()
    assert st["transcript_model"] == "base" and "transcript" in st["stale"] and st["has"]["transcript"] is True
    same = client.put("/v1/project/settings", json={"project": str(proj), "transcription_model": "base"}).json()
    assert same["ask"] is None                                                             # back to the model of the transcript: nothing newly stale
    assert client.put("/v1/project/settings", json={"project": str(proj), "transcription_model": "nope"}).status_code == 400
