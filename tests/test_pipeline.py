"""Unit and integration tests for the end-to-end pipeline orchestrator and per-video assortment."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from models.transcript import Transcript
from pipeline.run_pipeline import (
    PipelineResult,
    VideoPipelinePaths,
    get_or_create_video_id,
    get_video_pipeline_paths,
    load_pipeline_result,
    run_full_pipeline,
)


def test_video_id_auto_increment(tmp_path: Path):
    video1 = tmp_path / "first_interview.mp4"
    video2 = tmp_path / "second_interview.mp4"
    video3 = tmp_path / "third_interview.mp4"
    video1.touch()
    video2.touch()
    video3.touch()

    id1 = get_or_create_video_id(video1, project_root=tmp_path)
    assert id1 == "video_1"

    # Same video should reuse video_1
    assert get_or_create_video_id(video1, project_root=tmp_path) == "video_1"

    # New video should create next video_id: video_2
    id2 = get_or_create_video_id(video2, project_root=tmp_path)
    assert id2 == "video_2"

    # Third video should create next video_id: video_3
    id3 = get_or_create_video_id(video3, project_root=tmp_path)
    assert id3 == "video_3"


def test_get_video_pipeline_paths_schema(tmp_path: Path):
    video_path = tmp_path / "videos" / "interview_take1.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()

    paths = get_video_pipeline_paths(video_path, project_root=tmp_path)

    assert paths.video_id == "video_1"
    expected_dir = tmp_path / "data" / "processed" / "video_1"
    assert paths.video_dir == expected_dir
    assert paths.candidates_dir == expected_dir / "candidates"
    assert paths.cutlists_dir == expected_dir / "cutlists"
    assert paths.outputs_dir == expected_dir / "outputs"
    assert paths.units_dir == expected_dir / "units"

    # Verify files reside inside their respective folders
    assert paths.token_candidates == expected_dir / "candidates" / "token_candidates.json"
    assert paths.semantic_candidates == expected_dir / "candidates" / "semantic_candidates.json"
    assert paths.retake_spans == expected_dir / "candidates" / "retake_spans.json"
    assert paths.adjudicated_spans == expected_dir / "candidates" / "adjudicated_spans.json"
    assert paths.cutlist == expected_dir / "cutlists" / "cutlist.json"
    assert paths.output_video == expected_dir / "outputs" / "interview_take1_cleaned.mp4"
    assert paths.units == expected_dir / "units" / "comparable_units.json"
    assert paths.transcript == expected_dir / "transcript.json"
    assert paths.pipeline_result == expected_dir / "pipeline_result.json"

    # Test mkdir creates all subdirectories
    paths.mkdir()
    assert paths.candidates_dir.is_dir()
    assert paths.cutlists_dir.is_dir()
    assert paths.outputs_dir.is_dir()
    assert paths.units_dir.is_dir()


def test_pipeline_result_serialization():
    result = PipelineResult(
        video_id="video_1",
        input_video="data/videos/test.mp4",
        output_video="data/processed/video_1/outputs/test_cleaned.mp4",
        source_duration=12.5,
        output_duration=8.2,
        time_saved=4.3,
        retake_groups_count=2,
        total_candidates=5,
        likely_retakes=3,
        ambiguous=1,
        spans_count=2,
        adjudicated_count=1,
        stage_timings={"transcription": 1.2, "semantic_matching": 0.5},
        cutlist_dict={"source": "test.mp4", "keep_spans": []},
        success=True,
    )

    d = result.to_dict()
    assert d["video_id"] == "video_1"
    assert d["input_video"] == "data/videos/test.mp4"
    assert d["source_duration"] == 12.5
    assert d["cutlist_dict"]["source"] == "test.mp4"
    assert d["success"] is True

    reconstructed = PipelineResult.from_dict(d)
    assert reconstructed.video_id == "video_1"
    assert reconstructed.input_video == result.input_video
    assert reconstructed.output_video == result.output_video
    assert reconstructed.source_duration == result.source_duration
    assert reconstructed.stage_timings == result.stage_timings
    assert reconstructed.cutlist_dict == result.cutlist_dict
    assert reconstructed.success is True


def test_load_pipeline_result(tmp_path: Path):
    video = tmp_path / "sample.mp4"
    video.touch()

    # None before saving
    assert load_pipeline_result(video, project_root=tmp_path) is None

    paths = get_video_pipeline_paths(video, project_root=tmp_path)
    paths.mkdir()
    paths.pipeline_result.write_text(
        json.dumps({
            "video_id": paths.video_id,
            "input_video": str(video),
            "output_video": str(paths.output_video),
            "source_duration": 20.0,
            "success": True,
        }),
        encoding="utf-8",
    )

    loaded = load_pipeline_result(video, project_root=tmp_path)
    assert loaded is not None
    assert loaded.video_id == paths.video_id
    assert loaded.success is True
    assert loaded.source_duration == 20.0


def test_multi_video_persistence_no_overwrite(tmp_path: Path):
    """
    Ensure running data persistence for video_a does NOT overwrite video_b,
    and each video has its own video_id directory with candidates, cutlists, outputs, units.
    """
    video_a = tmp_path / "video_a.mp4"
    video_b = tmp_path / "video_b.mp4"
    video_a.touch()
    video_b.touch()

    paths_a = get_video_pipeline_paths(video_a, project_root=tmp_path)
    paths_b = get_video_pipeline_paths(video_b, project_root=tmp_path)

    assert paths_a.video_id == "video_1"
    assert paths_b.video_id == "video_2"
    assert paths_a.video_dir != paths_b.video_dir

    paths_a.mkdir()
    paths_b.mkdir()

    # Save distinct data for video A inside its folders
    paths_a.transcript.write_text(json.dumps({"source": "video_a.mp4", "duration": 15.0}), encoding="utf-8")
    paths_a.cutlist.write_text(json.dumps({"source": "video_a.mp4", "retake_groups": [1]}), encoding="utf-8")
    paths_a.units.write_text(json.dumps({"units": ["unit_a1"]}), encoding="utf-8")
    paths_a.token_candidates.write_text(json.dumps([{"cand": "a"}]), encoding="utf-8")

    # Save distinct data for video B inside its folders
    paths_b.transcript.write_text(json.dumps({"source": "video_b.mp4", "duration": 45.0}), encoding="utf-8")
    paths_b.cutlist.write_text(json.dumps({"source": "video_b.mp4", "retake_groups": [2, 3]}), encoding="utf-8")
    paths_b.units.write_text(json.dumps({"units": ["unit_b1", "unit_b2"]}), encoding="utf-8")
    paths_b.token_candidates.write_text(json.dumps([{"cand": "b"}]), encoding="utf-8")

    # Assert Video A data is completely preserved
    data_a_trans = json.loads(paths_a.transcript.read_text(encoding="utf-8"))
    data_a_cut = json.loads(paths_a.cutlist.read_text(encoding="utf-8"))
    data_a_cand = json.loads(paths_a.token_candidates.read_text(encoding="utf-8"))
    assert data_a_trans["source"] == "video_a.mp4"
    assert data_a_cut["retake_groups"] == [1]
    assert data_a_cand[0]["cand"] == "a"

    # Assert Video B data is completely preserved
    data_b_trans = json.loads(paths_b.transcript.read_text(encoding="utf-8"))
    data_b_cut = json.loads(paths_b.cutlist.read_text(encoding="utf-8"))
    data_b_cand = json.loads(paths_b.token_candidates.read_text(encoding="utf-8"))
    assert data_b_trans["source"] == "video_b.mp4"
    assert data_b_cut["retake_groups"] == [2, 3]
    assert data_b_cand[0]["cand"] == "b"

    # Assert directory names and internal folders
    assert paths_a.video_dir.name == "video_1"
    assert paths_b.video_dir.name == "video_2"
    assert paths_a.candidates_dir.name == "candidates"
    assert paths_a.cutlists_dir.name == "cutlists"
    assert paths_a.outputs_dir.name == "outputs"
    assert paths_a.units_dir.name == "units"
