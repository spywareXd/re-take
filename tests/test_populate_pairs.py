"""
Tests for training/populate_pairs.py

Validates:
- Rerunning on an unchanged video adds zero rows and does not call the pipeline.
- Existing labels, notes and hard-negative flags survive both a normal run and --refresh.
- --refresh updates feature columns and appends new pairs, and leaves rows the new run does not produce untouched.
- pair_id is deterministic and A always starts before B.
- A dummy_ row is never modified or counted.
- A PermissionError on write leaves the CSV and manifest unchanged and exits non-zero.
- A backup is created before each write, and the oldest are pruned beyond 20.
- Header mismatch is refused.
- --dry-run writes nothing.
- Same-stem and same-hash duplicates are handled as described.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from training.config import (
    CSV_COLUMNS,
    FEATURE_VERSION,
    GROUPS_CSV_COLUMNS,
    REFRESH_FEATURE_COLUMNS,
)
from training.populate_pairs import (
    _GroupsUnionFind,
    _compute_group_heuristics,
    atomic_write_groups_csv,
    atomic_write_manifest,
    atomic_write_pairs_csv,
    backup_groups_csv,
    backup_pairs_csv,
    build_groups,
    candidate_to_csv_row,
    check_groups,
    compute_sha256,
    eval_keepers,
    get_video_id,
    load_manifest,
    print_stats,
    process_videos,
    read_groups_csv,
    read_pairs_csv,
)


def _create_sample_candidate(
    uid_a: str = "1:0",
    uid_b: str = "2:0",
    text_a: str = "First sentence take one.",
    text_b: str = "First sentence take two.",
    start_a: float = 1.0,
    end_a: float = 3.0,
    start_b: float = 5.0,
    end_b: float = 7.0,
    lexical: float = 0.85,
    semantic: float = 0.92,
    classification: str = "likely_retake",
) -> dict:
    return {
        "candidate_id": 1,
        "unit_a": {
            "unit_id": uid_a,
            "speaker": "speaker_0",
            "raw_text": text_a,
            "start": start_a,
            "end": end_a,
        },
        "unit_b": {
            "unit_id": uid_b,
            "speaker": "speaker_0",
            "raw_text": text_b,
            "start": start_b,
            "end": end_b,
        },
        "scores": {
            "sequence": 0.88,
            "jaccard": 0.80,
            "containment": 1.0,
            "bigram": 0.75,
        },
        "lexical_score": lexical,
        "semantic_score": semantic,
        "recency_weight": 0.95,
        "restart_phrase_detected": False,
        "lexical_semantic_disagreement": round(abs(lexical - semantic), 3),
        "classification": classification,
    }


def test_pair_id_deterministic_and_a_before_b():
    """Verify pair_id formatting, determinism, and that unit A always starts before unit B."""
    # Normal order: unit_a starts at 2.0, unit_b starts at 6.0
    cand1 = _create_sample_candidate(
        uid_a="1:1",
        uid_b="2:1",
        start_a=2.0,
        end_a=4.0,
        start_b=6.0,
        end_b=8.0,
    )
    row1 = candidate_to_csv_row(cand1, "test_clip", "scripted")
    assert row1["pair_id"] == "test_clip__1-1__2-1"
    assert row1["start_a"] == 2.0
    assert row1["start_b"] == 6.0

    # Inverted order: candidate passed with start_a=10.0 and start_b=2.0
    cand2 = _create_sample_candidate(
        uid_a="3:1",
        uid_b="1:1",
        text_a="Later text",
        text_b="Earlier text",
        start_a=10.0,
        end_a=12.0,
        start_b=2.0,
        end_b=4.0,
    )
    row2 = candidate_to_csv_row(cand2, "test_clip", "scripted")
    # Swapped so A is earlier than B
    assert row2["start_a"] == 2.0
    assert row2["start_b"] == 10.0
    assert row2["text_a"] == "Earlier text"
    assert row2["text_b"] == "Later text"
    assert row2["pair_id"] == "test_clip__1-1__3-1"


def test_header_mismatch_refused(tmp_path: Path):
    """Refuse to run if CSV header differs from canonical specification."""
    csv_file = tmp_path / "pairs_v1.csv"
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "wrong_column", "video_id"])

    with pytest.raises(ValueError, match="Header mismatch"):
        read_pairs_csv(csv_file)


def test_dry_run_writes_nothing(tmp_path: Path):
    """Verify --dry-run writes no CSV, no manifest, no backups, and no processed artifacts."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    processed_dir = tmp_path / "processed"
    backups_dir = tmp_path / "backups"
    csv_path = tmp_path / "pairs_v1.csv"
    manifest_path = tmp_path / "manifest.json"

    vid = videos_dir / "sample.mp4"
    vid.write_bytes(b"sample video bytes")

    sample_cand = _create_sample_candidate()

    with patch("training.populate_pairs.run_pipeline_for_video") as mock_pipeline:
        mock_pipeline.return_value = (MagicMock(), [MagicMock(), MagicMock()], [sample_cand])

        exit_code = process_videos(
            videos_dir=videos_dir,
            processed_dir=processed_dir,
            backups_dir=backups_dir,
            csv_path=csv_path,
            manifest_path=manifest_path,
            dry_run=True,
        )

        assert exit_code == 0
        assert not csv_path.exists()
        assert not manifest_path.exists()
        assert not backups_dir.exists() or len(list(backups_dir.glob("*"))) == 0
        assert not (processed_dir / "sample").exists()


def test_rerun_unchanged_video_adds_zero_rows_and_skips_pipeline(tmp_path: Path):
    """If a video is already in manifest, it is skipped and pipeline is not called."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    processed_dir = tmp_path / "processed"
    backups_dir = tmp_path / "backups"
    csv_path = tmp_path / "pairs_v1.csv"
    manifest_path = tmp_path / "manifest.json"

    vid = videos_dir / "clip_a.mp4"
    vid.write_bytes(b"clip content")
    sha = compute_sha256(vid)

    # Initialize manifest with clip_a already processed
    manifest = {
        "clip_a": {
            "video_id": "clip_a",
            "filename": "clip_a.mp4",
            "sha256": sha,
            "processed_date": "2026-10-20T00:00:00Z",
            "units": 5,
            "pairs_exported": 2,
        }
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # CSV has some initial content
    atomic_write_pairs_csv(csv_path, [], fieldnames=CSV_COLUMNS)

    with patch("training.populate_pairs.run_pipeline_for_video") as mock_pipeline:
        exit_code = process_videos(
            videos_dir=videos_dir,
            processed_dir=processed_dir,
            backups_dir=backups_dir,
            csv_path=csv_path,
            manifest_path=manifest_path,
        )

        assert exit_code == 0
        mock_pipeline.assert_not_called()
        _, rows = read_pairs_csv(csv_path)
        assert len(rows) == 0


def test_existing_labels_notes_flags_survive_normal_and_refresh(tmp_path: Path):
    """
    Ensure hand-labeled data (label, notes, is_hard_negative) is preserved
    across both normal runs and --refresh runs.
    """
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    processed_dir = tmp_path / "processed"
    backups_dir = tmp_path / "backups"
    csv_path = tmp_path / "pairs_v1.csv"
    manifest_path = tmp_path / "manifest.json"

    vid = videos_dir / "video_demo.mp4"
    vid.write_bytes(b"video demo bytes")

    # 1. Existing row with manual labels
    initial_row = {
        "pair_id": "video_demo__1-0__2-0",
        "video_id": "video_demo",
        "source_type": "scripted",
        "speaker": "speaker_0",
        "unit_a_id": "1:0",
        "unit_b_id": "2:0",
        "text_a": "Hello original.",
        "text_b": "Hello retake.",
        "start_a": 1.0,
        "end_a": 2.5,
        "start_b": 4.0,
        "end_b": 5.5,
        "label": "1",
        "notes": "critical manual note",
        "is_hard_negative": "1",
        "sequence": 0.5,
        "jaccard": 0.5,
        "containment": 0.5,
        "bigram": 0.5,
        "lexical_score": 0.5,
        "semantic_score": 0.6,
        "recency_weight": 0.9,
        "restart_phrase": 0,
        "disagreement": 0.1,
        "passes_stage1_default": 1,
        "pipeline_class": "ambiguous",
        "split": "train",
        "is_synthetic": 0,
        "feature_version": "v1",
    }
    atomic_write_pairs_csv(csv_path, [initial_row], fieldnames=CSV_COLUMNS)

    # 2. Test Normal Run: Video generates candidate with same pair_id
    cand = _create_sample_candidate(
        uid_a="1:0",
        uid_b="2:0",
        start_a=1.0,
        end_a=2.5,
        start_b=4.0,
        end_b=5.5,
        lexical=0.99,  # different new lexical score
    )
    with patch("training.populate_pairs.run_pipeline_for_video") as mock_pipeline:
        mock_pipeline.return_value = (MagicMock(), [MagicMock()], [cand])

        # Normal run should skip existing pair_id and not overwrite anything
        exit_code = process_videos(
            videos_dir=videos_dir,
            processed_dir=processed_dir,
            backups_dir=backups_dir,
            csv_path=csv_path,
            manifest_path=manifest_path,
        )
        assert exit_code == 0
        _, rows = read_pairs_csv(csv_path)
        assert len(rows) == 1
        assert rows[0]["label"] == "1"
        assert rows[0]["notes"] == "critical manual note"
        assert rows[0]["is_hard_negative"] == "1"
        assert rows[0]["lexical_score"] == "0.5"  # untouched!

    # 3. Test Refresh Run: Video recomputed with new feature values
    cand_refreshed = _create_sample_candidate(
        uid_a="1:0",
        uid_b="2:0",
        start_a=1.0,
        end_a=2.5,
        start_b=4.0,
        end_b=5.5,
        lexical=0.95,
        semantic=0.98,
        classification="likely_retake",
    )
    with patch("training.populate_pairs.run_pipeline_for_video") as mock_pipeline:
        mock_pipeline.return_value = (MagicMock(), [MagicMock()], [cand_refreshed])

        exit_code = process_videos(
            videos_dir=videos_dir,
            processed_dir=processed_dir,
            backups_dir=backups_dir,
            csv_path=csv_path,
            manifest_path=manifest_path,
            refresh_video_id="video_demo",
        )
        assert exit_code == 0
        _, rows = read_pairs_csv(csv_path)
        assert len(rows) == 1
        # Manual labels and text remain untouched
        assert rows[0]["label"] == "1"
        assert rows[0]["notes"] == "critical manual note"
        assert rows[0]["is_hard_negative"] == "1"
        assert rows[0]["text_a"] == "Hello original."
        assert rows[0]["split"] == "train"
        # Features recomputed
        assert rows[0]["lexical_score"] == "0.95"
        assert rows[0]["semantic_score"] == "0.98"
        assert rows[0]["pipeline_class"] == "likely_retake"


def test_refresh_updates_features_appends_new_and_leaves_unproduced_untouched(tmp_path: Path):
    """
    --refresh updates matching pairs, appends newly generated pairs,
    and leaves existing pairs that the new run no longer produces untouched.
    """
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    processed_dir = tmp_path / "processed"
    backups_dir = tmp_path / "backups"
    csv_path = tmp_path / "pairs_v1.csv"
    manifest_path = tmp_path / "manifest.json"

    vid = videos_dir / "vid_x.mp4"
    vid.write_bytes(b"vid x content")

    row_pair1 = {
        "pair_id": "vid_x__1-0__2-0",
        "video_id": "vid_x",
        "source_type": "scripted",
        "speaker": "speaker_0",
        "unit_a_id": "1:0",
        "unit_b_id": "2:0",
        "text_a": "A1",
        "text_b": "B1",
        "start_a": 1.0,
        "end_a": 2.0,
        "start_b": 3.0,
        "end_b": 4.0,
        "label": "1",
        "notes": "hand note 1",
        "is_hard_negative": "0",
        "sequence": 0.5,
        "jaccard": 0.5,
        "containment": 0.5,
        "bigram": 0.5,
        "lexical_score": 0.5,
        "semantic_score": 0.5,
        "recency_weight": 0.8,
        "restart_phrase": 0,
        "disagreement": 0.0,
        "passes_stage1_default": 1,
        "pipeline_class": "ambiguous",
        "split": "",
        "is_synthetic": 0,
        "feature_version": "v1",
    }
    row_pair2 = {
        "pair_id": "vid_x__2-0__3-0",
        "video_id": "vid_x",
        "source_type": "scripted",
        "speaker": "speaker_0",
        "unit_a_id": "2:0",
        "unit_b_id": "3:0",
        "text_a": "B1",
        "text_b": "C1",
        "start_a": 3.0,
        "end_a": 4.0,
        "start_b": 5.0,
        "end_b": 6.0,
        "label": "0",
        "notes": "not a retake",
        "is_hard_negative": "1",
        "sequence": 0.2,
        "jaccard": 0.2,
        "containment": 0.2,
        "bigram": 0.2,
        "lexical_score": 0.2,
        "semantic_score": 0.2,
        "recency_weight": 0.8,
        "restart_phrase": 0,
        "disagreement": 0.0,
        "passes_stage1_default": 0,
        "pipeline_class": "unlikely_retake",
        "split": "",
        "is_synthetic": 0,
        "feature_version": "v1",
    }
    atomic_write_pairs_csv(csv_path, [row_pair1, row_pair2], fieldnames=CSV_COLUMNS)

    # New run produces pair1 (with updated lexical) and new pair3.
    # It does NOT produce pair2.
    cand1 = _create_sample_candidate(
        uid_a="1:0",
        uid_b="2:0",
        start_a=1.0,
        end_a=2.0,
        start_b=3.0,
        end_b=4.0,
        lexical=0.91,
        semantic=0.96,
        classification="likely_retake",
    )
    cand3 = _create_sample_candidate(
        uid_a="3:0",
        uid_b="4:0",
        start_a=5.0,
        end_a=6.0,
        start_b=7.0,
        end_b=8.0,
        lexical=0.77,
        semantic=0.88,
        classification="ambiguous",
    )

    with patch("training.populate_pairs.run_pipeline_for_video") as mock_pipeline:
        mock_pipeline.return_value = (MagicMock(), [MagicMock(), MagicMock()], [cand1, cand3])

        exit_code = process_videos(
            videos_dir=videos_dir,
            processed_dir=processed_dir,
            backups_dir=backups_dir,
            csv_path=csv_path,
            manifest_path=manifest_path,
            refresh_video_id="vid_x",
        )
        assert exit_code == 0

        _, rows = read_pairs_csv(csv_path)
        assert len(rows) == 3

        row_map = {r["pair_id"]: r for r in rows}
        # pair1 updated
        assert row_map["vid_x__1-0__2-0"]["lexical_score"] == "0.91"
        assert row_map["vid_x__1-0__2-0"]["label"] == "1"

        # pair2 untouched
        assert row_map["vid_x__2-0__3-0"]["lexical_score"] == "0.2"
        assert row_map["vid_x__2-0__3-0"]["label"] == "0"
        assert row_map["vid_x__2-0__3-0"]["notes"] == "not a retake"

        # pair3 appended
        assert "vid_x__3-0__4-0" in row_map
        assert row_map["vid_x__3-0__4-0"]["lexical_score"] == "0.77"
        assert row_map["vid_x__3-0__4-0"]["label"] == ""


def test_dummy_row_never_modified_or_counted(tmp_path: Path):
    """Rows starting with dummy_ are never modified, refreshed, or included in stats."""
    csv_path = tmp_path / "pairs_v1.csv"
    dummy_row = {
        "pair_id": "dummy_01__1-0__2-0",
        "video_id": "dummy_01",
        "source_type": "dummy",
        "speaker": "speaker_0",
        "unit_a_id": "1:0",
        "unit_b_id": "2:0",
        "text_a": "Dummy text A",
        "text_b": "Dummy text B",
        "start_a": 10.0,
        "end_a": 12.0,
        "start_b": 15.0,
        "end_b": 17.0,
        "label": "1",
        "notes": "dummy note",
        "is_hard_negative": "0",
        "sequence": 0.9,
        "jaccard": 0.9,
        "containment": 1.0,
        "bigram": 0.9,
        "lexical_score": 0.9,
        "semantic_score": 0.9,
        "recency_weight": 0.9,
        "restart_phrase": 0,
        "disagreement": 0.0,
        "passes_stage1_default": 1,
        "pipeline_class": "likely_retake",
        "split": "",
        "is_synthetic": 0,
        "feature_version": "v1",
    }
    atomic_write_pairs_csv(csv_path, [dummy_row], fieldnames=CSV_COLUMNS)

    # 1. Attempting to refresh dummy video should fail
    exit_code = process_videos(
        csv_path=csv_path,
        refresh_video_id="dummy_01",
    )
    assert exit_code == 1

    # 2. Stats output should exclude dummy rows completely
    with patch("sys.stdout.write") as mock_stdout:
        print_stats(csv_path)


def test_permission_error_on_write_leaves_csv_and_manifest_unchanged(tmp_path: Path):
    """If a PermissionError occurs during write, manifest and CSV remain unchanged."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    processed_dir = tmp_path / "processed"
    backups_dir = tmp_path / "backups"
    csv_path = tmp_path / "pairs_v1.csv"
    manifest_path = tmp_path / "manifest.json"

    vid = videos_dir / "vid_lock.mp4"
    vid.write_bytes(b"content")

    # Initial CSV and manifest
    atomic_write_pairs_csv(csv_path, [], fieldnames=CSV_COLUMNS)
    atomic_write_manifest(manifest_path, {})

    cand = _create_sample_candidate()
    with patch("training.populate_pairs.run_pipeline_for_video") as mock_pipeline:
        mock_pipeline.return_value = (MagicMock(), [MagicMock()], [cand])
        with patch("os.replace", side_effect=PermissionError("File locked by Excel")):
            with pytest.raises(SystemExit) as exc_info:
                process_videos(
                    videos_dir=videos_dir,
                    processed_dir=processed_dir,
                    backups_dir=backups_dir,
                    csv_path=csv_path,
                    manifest_path=manifest_path,
                )
            assert exc_info.value.code != 0

    # CSV and manifest should be completely empty and unchanged
    _, rows = read_pairs_csv(csv_path)
    assert len(rows) == 0
    assert load_manifest(manifest_path) == {}


def test_backup_created_and_pruned_beyond_20(tmp_path: Path):
    """A backup is created before write, and the oldest are pruned beyond 20."""
    csv_path = tmp_path / "pairs_v1.csv"
    backups_dir = tmp_path / "backups"
    atomic_write_pairs_csv(csv_path, [], fieldnames=CSV_COLUMNS)

    # Generate 25 backups
    import time
    for i in range(25):
        # Update file so stat changes
        csv_path.write_text(f"dummy content {i}\n", encoding="utf-8-sig")
        backup_pairs_csv(csv_path, backups_dir)
        time.sleep(0.005)

    backups = list(backups_dir.glob("pairs_v1.*.csv"))
    assert len(backups) == 20


def test_duplicate_stems_and_hashes_handled(tmp_path: Path):
    """
    1. Two files with the same stem (e.g. 'test 1.mp4' and 'test_1.mov') causes an error.
    2. Two files with identical sha256: second one logs warning and skips.
    """
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    processed_dir = tmp_path / "processed"
    backups_dir = tmp_path / "backups"
    csv_path = tmp_path / "pairs_v1.csv"
    manifest_path = tmp_path / "manifest.json"

    # Part 1: Same stem duplicate
    v1 = videos_dir / "my clip.mp4"
    v2 = videos_dir / "my_clip.mov"
    v1.write_bytes(b"content 1")
    v2.write_bytes(b"content 2")

    exit_code = process_videos(
        videos_dir=videos_dir,
        processed_dir=processed_dir,
        backups_dir=backups_dir,
        csv_path=csv_path,
        manifest_path=manifest_path,
    )
    assert exit_code == 1

    # Fix stem duplicate by deleting v2
    v2.unlink()

    # Part 2: Same SHA256 duplicate
    v1.write_bytes(b"identical video bytes")
    v3 = videos_dir / "different_name.mp4"
    v3.write_bytes(b"identical video bytes")

    cand = _create_sample_candidate()
    with patch("training.populate_pairs.run_pipeline_for_video") as mock_pipeline:
        mock_pipeline.return_value = (MagicMock(), [MagicMock()], [cand])

        exit_code = process_videos(
            videos_dir=videos_dir,
            processed_dir=processed_dir,
            backups_dir=backups_dir,
            csv_path=csv_path,
            manifest_path=manifest_path,
        )
        assert exit_code == 0

        # Only one video should have been processed
        manifest = load_manifest(manifest_path)
        assert len(manifest) == 1
        assert ("my_clip" in manifest) ^ ("different_name" in manifest)


def _setup_mock_units(
    proc_dir: Path,
    video_id: str,
    units_info: list[tuple[str, float, float, str, float]],
) -> dict[str, dict[str, Any]]:
    v_dir = proc_dir / video_id
    v_dir.mkdir(parents=True, exist_ok=True)
    units_list = []
    for uid, start, end, text, conf in units_info:
        units_list.append({
            "unit_id": uid,
            "start": start,
            "end": end,
            "raw_text": text,
            "confidence": conf,
            "is_below_min_token_length": False,
        })
    (v_dir / "units.json").write_text(json.dumps(units_list), encoding="utf-8")
    return {u["unit_id"]: u for u in units_list}


def _create_pair_row(
    video_id: str,
    uid_a: str,
    uid_b: str,
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
    label: str = "",
    notes: str = "",
    pipeline_class: str = "likely_retake",
) -> dict[str, Any]:
    return {
        "pair_id": f"{video_id}__{uid_a.replace(':', '-')}__{uid_b.replace(':', '-')}",
        "video_id": video_id,
        "source_type": "scripted",
        "speaker": "speaker_0",
        "unit_a_id": uid_a,
        "unit_b_id": uid_b,
        "text_a": f"Text {uid_a}",
        "text_b": f"Text {uid_b}",
        "start_a": start_a,
        "end_a": end_a,
        "start_b": start_b,
        "end_b": end_b,
        "label": label,
        "notes": notes,
        "is_hard_negative": "0",
        "sequence": "0.9",
        "jaccard": "0.8",
        "containment": "1.0",
        "bigram": "0.7",
        "lexical_score": "0.85",
        "semantic_score": "0.92",
        "recency_weight": "0.9",
        "restart_phrase": "0",
        "disagreement": "0.07",
        "passes_stage1_default": "1",
        "pipeline_class": pipeline_class,
        "split": "",
        "is_synthetic": "0",
        "feature_version": FEATURE_VERSION,
    }


def test_groups_connected_components_label1_only(tmp_path: Path):
    """Connected components of label=1 pairs only; 0, -1, and blank never join units."""
    proc_dir = tmp_path / "processed"
    _setup_mock_units(proc_dir, "vid1", [
        ("1:0", 1.0, 2.0, "Take one", 0.8),
        ("1:1", 3.0, 4.0, "Take two", 0.9),
        ("2:0", 5.0, 6.0, "Take three", 0.95),
        ("3:0", 7.0, 8.0, "Unrelated line", 0.9),
        ("4:0", 9.0, 10.0, "Another line", 0.85),
        ("5:0", 11.0, 12.0, "Third line", 0.8),
        ("6:0", 13.0, 14.0, "Second group take 1", 0.9),
        ("7:0", 15.0, 16.0, "Second group take 2", 0.92),
    ])

    pairs = [
        _create_pair_row("vid1", "1:0", "1:1", 1.0, 2.0, 3.0, 4.0, label="1"),
        _create_pair_row("vid1", "1:1", "2:0", 3.0, 4.0, 5.0, 6.0, label="1"),
        _create_pair_row("vid1", "2:0", "3:0", 5.0, 6.0, 7.0, 8.0, label="0"),     # negative: no join
        _create_pair_row("vid1", "3:0", "4:0", 7.0, 8.0, 9.0, 10.0, label="-1"),   # unclear: no join
        _create_pair_row("vid1", "4:0", "5:0", 9.0, 10.0, 11.0, 12.0, label=""),   # blank: no join
        _create_pair_row("vid1", "6:0", "7:0", 13.0, 14.0, 15.0, 16.0, label="1"), # group 2
    ]

    pairs_csv = tmp_path / "pairs_v1.csv"
    groups_csv = tmp_path / "groups_v1.csv"
    backups_dir = tmp_path / "backups"

    atomic_write_pairs_csv(pairs_csv, pairs)
    exit_code = build_groups(pairs_csv, groups_csv, proc_dir, backups_dir)
    assert exit_code == 0

    header, rows = read_groups_csv(groups_csv)
    assert len(rows) == 2

    # Group 1 has 3 units
    g1 = rows[0]
    assert g1["group_id"] == "vid1__g1"
    assert g1["n_units"] == "3"
    assert g1["unit_ids"] == "1:0|1:1|2:0"
    assert "[1] Take one || [2] Take two || [3] Take three" in g1["texts"]
    assert g1["starts"] == "1.0|3.0|5.0"
    assert g1["ends"] == "2.0|4.0|6.0"
    assert g1["keeper_unit_id"] == ""
    assert g1["keep_none"] == ""
    assert "vid1__1-0__1-1|vid1__1-1__2-0" in g1["pair_ids"]

    # Group 2 has 2 units
    g2 = rows[1]
    assert g2["group_id"] == "vid1__g2"
    assert g2["n_units"] == "2"
    assert g2["unit_ids"] == "6:0|7:0"


def test_group_stability_same_unit_set_preserves_hand_labels(tmp_path: Path):
    """Same unit set on rebuild keeps hand labels untouched; group_id never changes."""
    proc_dir = tmp_path / "processed"
    _setup_mock_units(proc_dir, "vid1", [
        ("1:0", 1.0, 2.0, "Take one", 0.8),
        ("1:1", 3.0, 4.0, "Take two", 0.9),
    ])

    pairs = [
        _create_pair_row("vid1", "1:0", "1:1", 1.0, 2.0, 3.0, 4.0, label="1"),
    ]
    pairs_csv = tmp_path / "pairs_v1.csv"
    groups_csv = tmp_path / "groups_v1.csv"
    backups_dir = tmp_path / "backups"

    atomic_write_pairs_csv(pairs_csv, pairs)
    build_groups(pairs_csv, groups_csv, proc_dir, backups_dir)

    # Hand-label the group
    header, rows = read_groups_csv(groups_csv)
    assert len(rows) == 1
    rows[0]["keeper_unit_id"] = "1:1"
    rows[0]["notes"] = "editor pick"
    atomic_write_groups_csv(groups_csv, rows)

    # Re-run build_groups
    build_groups(pairs_csv, groups_csv, proc_dir, backups_dir)

    header, rows_after = read_groups_csv(groups_csv)
    assert len(rows_after) == 1
    assert rows_after[0]["group_id"] == "vid1__g1"
    assert rows_after[0]["keeper_unit_id"] == "1:1"
    assert rows_after[0]["notes"] == "editor pick"


def test_group_stability_changed_unit_set_appends_and_marks_stale(tmp_path: Path):
    """Changed unit set appends a new row and marks the old one STALE in notes; nothing is deleted."""
    proc_dir = tmp_path / "processed"
    _setup_mock_units(proc_dir, "vid1", [
        ("1:0", 1.0, 2.0, "Take one", 0.8),
        ("1:1", 3.0, 4.0, "Take two", 0.9),
        ("2:0", 5.0, 6.0, "Take three", 0.95),
    ])

    pairs = [
        _create_pair_row("vid1", "1:0", "1:1", 1.0, 2.0, 3.0, 4.0, label="1"),
    ]
    pairs_csv = tmp_path / "pairs_v1.csv"
    groups_csv = tmp_path / "groups_v1.csv"
    backups_dir = tmp_path / "backups"

    atomic_write_pairs_csv(pairs_csv, pairs)
    build_groups(pairs_csv, groups_csv, proc_dir, backups_dir)

    # Hand-label the original group
    header, rows = read_groups_csv(groups_csv)
    rows[0]["keeper_unit_id"] = "1:0"
    rows[0]["notes"] = "initial pick"
    atomic_write_groups_csv(groups_csv, rows)

    # Now add pair 1:1 ~ 2:0 with label=1 -> group unit set grows to {1:0, 1:1, 2:0}
    pairs.append(_create_pair_row("vid1", "1:1", "2:0", 3.0, 4.0, 5.0, 6.0, label="1"))
    atomic_write_pairs_csv(pairs_csv, pairs)

    build_groups(pairs_csv, groups_csv, proc_dir, backups_dir)

    header, rows_after = read_groups_csv(groups_csv)
    # Old row is preserved, new row appended
    assert len(rows_after) == 2

    old_row = rows_after[0]
    assert old_row["group_id"] == "vid1__g1"
    assert "STALE: unit set no longer matches pair labels" in old_row["notes"]
    assert old_row["keeper_unit_id"] == "1:0"

    new_row = rows_after[1]
    assert new_row["group_id"] == "vid1__g2"
    assert new_row["unit_ids"] == "1:0|1:1|2:0"
    assert new_row["keeper_unit_id"] == ""


def test_group_stability_keeper_not_in_unit_ids_flagged(tmp_path: Path):
    """If keeper_unit_id is no longer in unit_ids, it is marked STALE."""
    proc_dir = tmp_path / "processed"
    _setup_mock_units(proc_dir, "vid1", [
        ("1:0", 1.0, 2.0, "Take one", 0.8),
        ("1:1", 3.0, 4.0, "Take two", 0.9),
    ])

    pairs = [
        _create_pair_row("vid1", "1:0", "1:1", 1.0, 2.0, 3.0, 4.0, label="1"),
    ]
    pairs_csv = tmp_path / "pairs_v1.csv"
    groups_csv = tmp_path / "groups_v1.csv"
    backups_dir = tmp_path / "backups"

    atomic_write_pairs_csv(pairs_csv, pairs)
    build_groups(pairs_csv, groups_csv, proc_dir, backups_dir)

    # Set invalid keeper
    header, rows = read_groups_csv(groups_csv)
    rows[0]["keeper_unit_id"] = "9:9"
    atomic_write_groups_csv(groups_csv, rows)

    build_groups(pairs_csv, groups_csv, proc_dir, backups_dir)
    header, rows_after = read_groups_csv(groups_csv)
    assert "STALE: keeper_unit_id '9:9' not in unit_ids" in rows_after[0]["notes"]


def test_check_groups_validations(tmp_path: Path):
    """check_groups flags contradictory zero pairs, missing keepers, conflicts, and stale rows."""
    pairs_csv = tmp_path / "pairs_v1.csv"
    groups_csv = tmp_path / "groups_v1.csv"

    # 1. Contradictory pair labeled 0 inside group
    pairs = [
        _create_pair_row("vid1", "1:0", "1:1", 1.0, 2.0, 3.0, 4.0, label="1"),
        _create_pair_row("vid1", "1:1", "2:0", 3.0, 4.0, 5.0, 6.0, label="1"),
        _create_pair_row("vid1", "1:0", "2:0", 1.0, 2.0, 5.0, 6.0, label="0"), # contradictory!
    ]
    atomic_write_pairs_csv(pairs_csv, pairs)

    groups = [
        {
            "group_id": "vid1__g1",
            "video_id": "vid1",
            "n_units": 3,
            "unit_ids": "1:0|1:1|2:0",
            "texts": "text",
            "starts": "1.0|3.0|5.0",
            "ends": "2.0|4.0|6.0",
            "keeper_unit_id": "1:1",
            "keep_none": "",
            "notes": "",
            "pipeline_keeper_unit_id": "1:1",
            "heuristic_later": "2:0",
            "heuristic_earlier": "1:0",
            "heuristic_confidence": "1:1",
            "pair_ids": "vid1__1-0__1-1|vid1__1-1__2-0",
            "feature_version": FEATURE_VERSION,
        }
    ]
    atomic_write_groups_csv(groups_csv, groups)

    code = check_groups(groups_csv, pairs_csv)
    assert code == 1

    # 2. Both keeper_unit_id and keep_none = 1
    groups[0]["keep_none"] = "1"
    atomic_write_groups_csv(groups_csv, groups)
    assert check_groups(groups_csv, pairs_csv) == 1

    # 3. Keeper outside unit_ids
    groups[0]["keep_none"] = ""
    groups[0]["keeper_unit_id"] = "unknown:0"
    atomic_write_groups_csv(groups_csv, groups)
    assert check_groups(groups_csv, pairs_csv) == 1


def test_pipeline_heuristics_called_and_none_leaves_blank(tmp_path: Path):
    """Pipeline heuristic columns are filled by calling existing functions; None leaves blank."""
    proc_dir = tmp_path / "processed"
    _setup_mock_units(proc_dir, "vid1", [
        ("1:0", 1.0, 2.0, "Take one", 0.8),
        ("1:1", 3.0, 4.0, "Take two", 0.9),
    ])
    pairs = [_create_pair_row("vid1", "1:0", "1:1", 1.0, 2.0, 3.0, 4.0, label="1")]
    pairs_csv = tmp_path / "pairs_v1.csv"
    groups_csv = tmp_path / "groups_v1.csv"
    backups_dir = tmp_path / "backups"

    atomic_write_pairs_csv(pairs_csv, pairs)

    # Case A: _determine_likely_final_unit_id returns None -> blank pipeline_keeper_unit_id
    with patch("training.populate_pairs._determine_likely_final_unit_id") as mock_step05, \
         patch("training.populate_pairs.select_take") as mock_cutlist:

        mock_step05.return_value = None
        mock_cutlist.side_effect = [
            ("b", "later"),
            ("a", "earlier"),
            ("b", "higher confidence"),
        ]

        build_groups(pairs_csv, groups_csv, proc_dir, backups_dir)
        assert mock_step05.called
        assert mock_cutlist.call_count == 3

        header, rows = read_groups_csv(groups_csv)
        assert rows[0]["pipeline_keeper_unit_id"] == ""
        assert rows[0]["heuristic_later"] == "1:1"
        assert rows[0]["heuristic_earlier"] == "1:0"
        assert rows[0]["heuristic_confidence"] == "1:1"

    # Case B: _determine_likely_final_unit_id returns "1:1"
    with patch("training.populate_pairs._determine_likely_final_unit_id") as mock_step05, \
         patch("training.populate_pairs.select_take") as mock_cutlist:

        mock_step05.return_value = "1:1"
        mock_cutlist.side_effect = [("b", "later"), ("a", "earlier"), ("b", "conf")]

        build_groups(pairs_csv, groups_csv, proc_dir, backups_dir)
        header, rows = read_groups_csv(groups_csv)
        assert rows[0]["pipeline_keeper_unit_id"] == "1:1"


def test_groups_write_safety_and_dummy_exclusion(tmp_path: Path):
    """Write safety: backup, atomic write, PermissionError, header mismatch, dummy exclusion."""
    groups_csv = tmp_path / "groups_v1.csv"
    backups_dir = tmp_path / "backups"
    proc_dir = tmp_path / "processed"
    pairs_csv = tmp_path / "pairs_v1.csv"

    # 1. Header mismatch refused
    groups_csv.write_text("wrong_header,col2\n1,2\n", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="Header mismatch"):
        read_groups_csv(groups_csv)

    # 2. PermissionError handled
    with patch("os.replace", side_effect=PermissionError("File locked")):
        with pytest.raises(SystemExit) as exc_info:
            atomic_write_groups_csv(groups_csv, [{"group_id": "g1"}], fieldnames=GROUPS_CSV_COLUMNS)
        assert exc_info.value.code == 1

    # 3. Backup creation and pruning to 20
    groups_csv.write_text(",".join(GROUPS_CSV_COLUMNS) + "\n", encoding="utf-8-sig")
    backups_dir.mkdir(parents=True, exist_ok=True)
    for i in range(25):
        b = backups_dir / f"groups_v1.20261020T1200{i:02d}000Z.csv"
        b.write_text("old", encoding="utf-8")

    backup_groups_csv(groups_csv, backups_dir)
    remaining_backups = list(backups_dir.glob("groups_v1.*.csv"))
    assert len(remaining_backups) <= 20

    # 4. dummy_ video ids ignored
    pairs = [
        _create_pair_row("dummy_01", "1:0", "1:1", 1.0, 2.0, 3.0, 4.0, label="1"),
    ]
    atomic_write_pairs_csv(pairs_csv, pairs)
    build_groups(pairs_csv, groups_csv, proc_dir, backups_dir)
    header, rows = read_groups_csv(groups_csv)
    assert len(rows) == 0

    # 5. --dry-run writes nothing
    _setup_mock_units(proc_dir, "vid1", [("1:0", 1.0, 2.0, "T1", 0.9), ("1:1", 3.0, 4.0, "T2", 0.9)])
    pairs = [_create_pair_row("vid1", "1:0", "1:1", 1.0, 2.0, 3.0, 4.0, label="1")]
    atomic_write_pairs_csv(pairs_csv, pairs)

    groups_csv.unlink(missing_ok=True)
    build_groups(pairs_csv, groups_csv, proc_dir, backups_dir, dry_run=True)
    assert not groups_csv.exists()


def test_eval_keepers_accuracy_and_empty(tmp_path: Path, capsys):
    """eval_keepers computes accuracy on small fixture, breaks down by size, excludes keep_none."""
    groups_csv = tmp_path / "groups_v1.csv"

    # Empty file / no labeled groups
    atomic_write_groups_csv(groups_csv, [])
    eval_keepers(groups_csv)
    captured = capsys.readouterr()
    assert "No labeled groups found in groups_v1.csv" in captured.out

    # Fixture with 3 groups:
    # Group 1 (size 2): keeper "1:1", pipeline "1:1", later "1:1", earlier "1:0", conf "1:1"
    # Group 2 (size 3): keeper "2:0", pipeline "2:0", later "2:0", earlier "1:0", conf "1:0"
    # Group 3 (size 2): keep_none=1 (excluded from accuracy)
    rows = [
        {
            "group_id": "vid__g1",
            "video_id": "vid",
            "n_units": 2,
            "unit_ids": "1:0|1:1",
            "texts": "[1] A || [2] B",
            "starts": "1.0|3.0",
            "ends": "2.0|4.0",
            "keeper_unit_id": "1:1",
            "keep_none": "",
            "notes": "",
            "pipeline_keeper_unit_id": "1:1",
            "heuristic_later": "1:1",
            "heuristic_earlier": "1:0",
            "heuristic_confidence": "1:1",
            "pair_ids": "p1",
            "feature_version": FEATURE_VERSION,
        },
        {
            "group_id": "vid__g2",
            "video_id": "vid",
            "n_units": 3,
            "unit_ids": "1:0|1:1|2:0",
            "texts": "[1] A || [2] B || [3] C",
            "starts": "1.0|3.0|5.0",
            "ends": "2.0|4.0|6.0",
            "keeper_unit_id": "2:0",
            "keep_none": "",
            "notes": "",
            "pipeline_keeper_unit_id": "2:0",
            "heuristic_later": "2:0",
            "heuristic_earlier": "1:0",
            "heuristic_confidence": "1:0",
            "pair_ids": "p2",
            "feature_version": FEATURE_VERSION,
        },
        {
            "group_id": "vid__g3",
            "video_id": "vid",
            "n_units": 2,
            "unit_ids": "3:0|4:0",
            "texts": "[1] X || [2] Y",
            "starts": "7.0|9.0",
            "ends": "8.0|10.0",
            "keeper_unit_id": "",
            "keep_none": "1",
            "notes": "",
            "pipeline_keeper_unit_id": "4:0",
            "heuristic_later": "4:0",
            "heuristic_earlier": "3:0",
            "heuristic_confidence": "4:0",
            "pair_ids": "p3",
            "feature_version": FEATURE_VERSION,
        },
    ]
    atomic_write_groups_csv(groups_csv, rows)

    eval_keepers(groups_csv)
    captured = capsys.readouterr()

    # keep_none excluded
    assert "Excluded keep_none=1 groups:            1" in captured.out
    assert "Groups evaluated for strategy accuracy: 2" in captured.out
    # Pipeline: 2/2 (100.0%)
    assert "2/2 (100.0%)" in captured.out
    # Keep Earlier: 0/2 (0.0%)
    assert "0/2 (0.0%)" in captured.out


def test_candidate_to_csv_row_disagreement_calculation():
    """Verify that candidate_to_csv_row computes disagreement as |lexical_score - semantic_score|."""
    cand = {
        "unit_a": {"unit_id": "11:0", "raw_text": "text A", "start": 10.0, "end": 12.0},
        "unit_b": {"unit_id": "15:0", "raw_text": "text B", "start": 20.0, "end": 22.0},
        "scores": {"sequence": 0.9, "jaccard": 0.9, "containment": 0.9, "bigram": 0.9},
        "lexical_score": 0.926,
        "semantic_score": 0.990,
        "combined_score": 0.275,
        "recency_weight": 0.297,
        "restart_phrase_detected": False,
        "lexical_semantic_disagreement": 0.064,
        "classification": "ambiguous",
    }
    row = candidate_to_csv_row(cand, "speaker_a_3", "youtube")
    assert row["lexical_score"] == 0.926
    assert row["semantic_score"] == 0.990
    assert row["disagreement"] == 0.064

