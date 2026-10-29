"""ProjectStore: layout, atomic versioned writes, legacy loading, source fingerprint, unique export names; and the service writing only into the project folder."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from models.final.final_scorer import FinalScorer
from models.transcript import Transcript
from pipeline.step_06_llm_adjudication.llm_client import MockLLMClient
from service.app import Deps, create_app
from service.config import ServiceConfig
from service.project_store import ProjectError, ProjectStore

FIXTURE = Path(__file__).parent / "fixtures" / "sample_c_transcript.json"


def _tree(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_layout_and_atomic_versioned_roundtrip(tmp_path):
    s = ProjectStore.create(tmp_path / "X")
    assert s.data == tmp_path / "X" / "data" and (s.data / "exports").is_dir() and (s.data / "stale").is_dir()
    assert s.project_file("scp") == tmp_path / "X" / "X.scp"
    s.put("transcript", {"a": 1})
    raw = json.loads((s.data / "transcript.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1 and raw["kind"] == "transcript" and raw["data"] == {"a": 1}
    assert s.get("transcript") == {"a": 1}
    assert not [p for p in s.data.iterdir() if p.name.endswith(".tmp")]                 # temp file renamed away


def test_older_unversioned_file_loads_and_newer_schema_is_refused(tmp_path):
    s = ProjectStore.create(tmp_path / "X")
    (s.data / "units.json").write_text(json.dumps([{"unit_id": "u1"}]), encoding="utf-8")     # schema 0: no envelope
    assert s.get("units") == [{"unit_id": "u1"}]
    (s.data / "cutlist.json").write_text(json.dumps({"schema_version": 99, "data": {}}), encoding="utf-8")
    with pytest.raises(ProjectError):
        s.get("cutlist")
    (s.data / "keepers.json").write_text("{not json", encoding="utf-8")
    assert s.get("keepers", "fallback") == "fallback"                                   # unreadable -> default, never a crash


def test_paths_cannot_escape_the_data_folder(tmp_path):
    s = ProjectStore.create(tmp_path / "X")
    for bad in ("../x", "a/b", "", ".hidden", "x\y"):
        with pytest.raises(ProjectError):
            s.json_path(bad)
    with pytest.raises(ProjectError):
        s.path("..", "outside.txt")


def test_replace_keeps_a_numbered_copy(tmp_path):
    s = ProjectStore.create(tmp_path / "X")
    assert s.replace_with_copy("cutlist", {"v": 1}) is None
    c2 = s.replace_with_copy("cutlist", {"v": 2})
    c3 = s.replace_with_copy("cutlist", {"v": 3})
    assert c2.name == "cutlist.1.json" and c3.name == "cutlist.2.json"
    assert json.loads(c2.read_text(encoding="utf-8"))["data"] == {"v": 1} and s.get("cutlist") == {"v": 3}


def test_source_reference_is_relative_and_detects_missing_and_changed(tmp_path):
    video = tmp_path / "clips" / "a.mp4"
    video.parent.mkdir()
    video.write_bytes(b"one" * 1000)
    s = ProjectStore.create(tmp_path / "X")
    ref = s.set_source(video)
    assert ref["path"] == "../clips/a.mp4" and ref["fingerprint"]["size"] == 3000
    assert s.check_source()["status"] == "ok"
    video.write_bytes(b"two" * 1000)
    assert s.check_source()["status"] == "changed"
    video.rename(tmp_path / "clips" / "b.mp4")
    assert s.check_source()["status"] == "missing"


def test_unique_export_names_never_overwrite(tmp_path):
    s = ProjectStore.create(tmp_path / "X")
    a = s.unique_export_path("clip", "mp4")
    a.write_bytes(b"first")
    b = s.unique_export_path("clip", "mp4")
    c = s.unique_export_path("clip", ".mp4")
    assert (a.name, b.name, c.name) == ("clip.mp4", "clip_2.mp4", "clip_3.mp4") and a.read_bytes() == b"first"
    assert s.unique_export_path("a/b:c", "srt").parent == s.exports_dir               # unsafe characters are replaced, never a sub-path


def test_import_legacy_copies_and_leaves_originals(tmp_path):
    legacy = tmp_path / "service_data" / "cache" / "abc"
    legacy.mkdir(parents=True)
    (legacy / "transcript.json").write_text('{"segments": []}', encoding="utf-8")
    (legacy / "scores__m1.json").write_text("[1, 2]", encoding="utf-8")
    before = _tree(tmp_path / "service_data")
    s = ProjectStore.create(tmp_path / "X")
    assert sorted(s.legacy_available(tmp_path / "service_data" / "cache", "abc")) == ["scores__m1", "transcript"]
    assert sorted(s.import_legacy(tmp_path / "service_data" / "cache", "abc")) == ["pairs__m1", "transcript"]
    assert s.get("transcript") == {"segments": []} and s.get("pairs__m1") == [1, 2]
    assert _tree(tmp_path / "service_data") == before                                    # originals untouched
    assert s.import_legacy(tmp_path / "service_data" / "cache", "abc") == []             # nothing replaced the second time


@pytest.fixture
def api(tmp_path):
    def transcribe(path):
        return Transcript.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))

    cfg = ServiceConfig(data_dir=str(tmp_path / "service_data"), llm_mock=True, keeper_provider="gpt-oss")
    app = create_app(cfg, Deps(transcribe_fn=transcribe, scorer=FinalScorer(mock=True), llm_client=MockLLMClient()))
    media = tmp_path / "take.mp4"
    media.write_bytes(b"stand-in media bytes")
    return TestClient(app), media, tmp_path


def test_analyze_with_project_writes_only_inside_the_project(api):
    client, media, tmp = api
    proj = tmp / "MyProject"
    job = client.post("/v1/analyze", json={"path": str(media), "duration_seconds": 25.8, "project": str(proj)}).json()["job"]
    end = time.time() + 120
    while time.time() < end:
        body = client.get(f"/v1/jobs/{job}").json()
        if body["state"] != "running":
            break
        time.sleep(0.05)
    assert body["state"] == "done", body
    names = {p.name for p in (proj / "data").iterdir()}
    assert {"transcript.json", "words.json", "units.json", "project.json"} <= names and any(n.startswith("pairs__") for n in names)
    assert not (tmp / "service_data").exists()                                           # the legacy location stayed untouched (not even created)
    assert ProjectStore(proj).check_source()["status"] == "ok"
