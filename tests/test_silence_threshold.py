"""Item 3: the silence gap is a job parameter stored in the project, validated server-side, re-splits from the saved word timestamps without re-transcribing,
previews the unit count, marks downstream results stale, asks before replacing anything and keeps numbered copies."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from models.final.final_scorer import FinalScorer
from models.transcript import Transcript
from pipeline.step_02_preprocessing import PAUSE_SPLIT_THRESHOLD_SECONDS
from pipeline.step_06_llm_adjudication.llm_client import MockLLMClient
from service.app import Deps, create_app
from service.config import ServiceConfig
from service.project_store import ProjectStore

FIXTURE = Path(__file__).parent / "fixtures" / "sample_c_transcript.json"


@pytest.fixture
def api(tmp_path):
    calls = {"n": 0}

    def transcribe(path):
        calls["n"] += 1
        return Transcript.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))

    cfg = ServiceConfig(data_dir=str(tmp_path / "sd"), llm_mock=True, keeper_provider="gpt-oss")
    app = create_app(cfg, Deps(transcribe_fn=transcribe, scorer=FinalScorer(mock=True), llm_client=MockLLMClient()))
    media = tmp_path / "take.mp4"
    media.write_bytes(b"stand-in media bytes")
    return TestClient(app), media, tmp_path / "P", calls


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


def test_capabilities_report_range_and_todays_default(api):
    client = api[0]
    g = client.get("/v1/capabilities").json()["silence_gap"]
    assert g["default"] == PAUSE_SPLIT_THRESHOLD_SECONDS == 0.7 and g["min"] < g["default"] < g["max"] and g["unit"] == "seconds" and g["step"] > 0


@pytest.mark.parametrize("bad", [0, 0.05, 3.5, -1, "0.7", True, [0.7]])
def test_out_of_range_or_bad_values_are_rejected_server_side(api, bad):
    client, media, proj, _ = api
    r, _ = _run(client, media, proj, silence_gap_s=bad)
    assert r.status_code == 400 and "silence_gap_s" in r.json()["error"]
    r2 = client.put("/v1/project/settings", json={"project": str(proj), "silence_gap_s": bad})
    assert r2.status_code == 400
    assert not (proj / "data" / "transcript.json").exists()                              # rejected before any work


def test_default_gap_is_unchanged_and_the_chosen_gap_is_used_before_transcription(api):
    client, media, proj, _ = api
    _run(client, media, proj)
    assert ProjectStore(proj).settings()["silence_gap_s"] == 0.7
    n_default = len(ProjectStore(proj).get("units"))
    client2, media2, proj2, _ = api
    _run(client2, media2, proj2.parent / "Q", silence_gap_s=0.1)                          # set as a job parameter of the first analyze
    st = ProjectStore(proj2.parent / "Q")
    assert st.settings()["silence_gap_s"] == 0.1 and st.get("units_meta") == {"silence_gap_s": 0.1}
    assert len(st.get("units")) > n_default                                               # a smaller gap splits more


def test_preview_counts_units_without_storing_or_transcribing(api):
    client, media, proj, calls = api
    _run(client, media, proj)
    before = sorted(p.name for p in (proj / "data").iterdir())
    p01 = client.get("/v1/project/units-preview", params={"project": str(proj), "silence_gap_s": 0.1}).json()
    p30 = client.get("/v1/project/units-preview", params={"project": str(proj), "silence_gap_s": 3.0}).json()
    assert p01["units"] == 12 and p30["units"] == 8 and p01["current_units"] == 8 and p01["current_silence_gap_s"] == 0.7
    assert sorted(p.name for p in (proj / "data").iterdir()) == before and calls["n"] == 1
    assert client.get("/v1/project/units-preview", params={"project": str(proj), "silence_gap_s": 99}).status_code == 400


def test_resplit_needs_confirmation_then_keeps_numbered_copies_and_never_retranscribes(api):
    client, media, proj, calls = api
    _run(client, media, proj)
    store = ProjectStore(proj)
    store.save_result("keepers", {"g1": ["u1"]})
    store.save_result("cutlist", {"ranges": [[1, 2]]})
    old_units = (proj / "data" / "units.json").read_bytes()
    old_transcript = (proj / "data" / "transcript.json").read_bytes()
    s = client.put("/v1/project/settings", json={"project": str(proj), "silence_gap_s": 0.1}).json()
    assert {"units", "keepers", "cutlist"} <= set(s["newly_stale"]) and any(n.startswith("pairs__") for n in s["newly_stale"]) and s["ask"]
    assert "transcript" not in s["stale"]                                                  # the transcript is not affected
    r = client.post("/v1/project/resplit", json={"project": str(proj)})                    # no confirm: nothing happens
    assert r.status_code == 409 and r.json()["code"] == "resplit_needs_confirmation" and "keepers" in r.json()["will_replace"]
    assert (proj / "data" / "units.json").read_bytes() == old_units
    ok = client.post("/v1/project/resplit", json={"project": str(proj), "confirm": True}).json()
    assert ok["units"] == 12 and ok["previous_units_copy"] == "units.1.json"
    assert (proj / "data" / "stale" / "units.1.json").read_bytes() == old_units           # the previous version, byte for byte
    assert (proj / "data" / "transcript.json").read_bytes() == old_transcript and calls["n"] == 1      # no re-transcription
    assert "units" not in store.stale() and {"keepers", "cutlist"} <= set(store.stale())  # downstream stays stale (and on disk) until recomputed
    assert store.get("keepers") == {"g1": ["u1"]} and store.get("units_meta") == {"silence_gap_s": 0.1}
    _, body = _run(client, media, proj)                                                    # the next analyze rescoring pairs uses the new units, without asking again
    assert body["state"] == "done" and calls["n"] == 1
    assert len([p for p in (proj / "data" / "stale").iterdir() if p.name.startswith("pairs__")]) == 1      # old pair scores kept as a numbered copy
    assert not [n for n in store.stale() if n.startswith("pairs__")]


def test_analyze_with_a_different_gap_asks_first(api):
    client, media, proj, calls = api
    _run(client, media, proj)
    before = (proj / "data" / "units.json").read_bytes()
    r, _ = _run(client, media, proj, silence_gap_s=0.3)
    assert r.status_code == 409 and r.json()["code"] == "silence_changed" and r.json()["units_silence_gap_s"] == 0.7
    assert (proj / "data" / "units.json").read_bytes() == before
    r, body = _run(client, media, proj, silence_gap_s=0.3, replace_existing=True)
    assert body["state"] == "done" and calls["n"] == 1 and (proj / "data" / "stale" / "units.1.json").read_bytes() == before


def test_reverting_the_slider_lifts_the_stale_marks(api):
    client, media, proj, _ = api
    _run(client, media, proj)
    client.put("/v1/project/settings", json={"project": str(proj), "transcription_model": "base"})
    assert "transcript" in ProjectStore(proj).stale()
    client.put("/v1/project/settings", json={"project": str(proj), "transcription_model": "large-v2"})
    assert ProjectStore(proj).stale() == []


def test_resplit_refuses_when_the_transcript_itself_is_stale(api):
    client, media, proj, _ = api
    _run(client, media, proj)
    client.put("/v1/project/settings", json={"project": str(proj), "transcription_model": "base"})
    r = client.post("/v1/project/resplit", json={"project": str(proj), "confirm": True, "silence_gap_s": 0.3})
    assert r.status_code == 400 and "Re-transcribe first" in r.json()["error"]


def test_nan_sent_as_raw_json_is_rejected(api):
    client, media, proj, _ = api
    raw = '{"path": %s, "duration_seconds": 25.8, "project": %s, "silence_gap_s": NaN}' % (json.dumps(str(media)), json.dumps(str(proj)))
    r = client.post("/v1/analyze", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 400 and "silence_gap_s" in r.json()["error"]
