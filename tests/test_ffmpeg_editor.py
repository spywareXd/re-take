"""Tests for Step 08: FFmpeg Video Editor.

These tests verify the FFmpeg integration functions. Most tests use
a small real video for end-to-end verification where possible,
falling back to mocking where FFmpeg is unavailable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from pipeline.step_05_cutlist.schemas import CutList, TimeSpan
from pipeline.step_08_video_edit.ffmpeg_editor import (
    extract_segment,
    concatenate_segments,
    edit_video,
    _run_ffmpeg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _test_video_path() -> Path:
    """Get path to a test video if available."""
    candidates = [
        Path("data/videos/sample_c.mp4"),
        Path("data/videos/test.mp4"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path("data/videos/test.mp4")  # Will skip if not found


SKIP_NO_FFMPEG = pytest.mark.skipif(
    not _ffmpeg_available(), reason="FFmpeg not installed"
)
SKIP_NO_VIDEO = pytest.mark.skipif(
    not _test_video_path().exists(), reason="No test video available"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractSegment:
    @SKIP_NO_FFMPEG
    @SKIP_NO_VIDEO
    def test_extract_creates_file(self, tmp_path):
        video = _test_video_path()
        output = tmp_path / "segment.mp4"
        result = extract_segment(video, output, start=0.0, end=2.0, copy_mode=False)
        assert result.exists()
        assert result.stat().st_size > 0

    @SKIP_NO_FFMPEG
    @SKIP_NO_VIDEO
    def test_extract_copy_mode(self, tmp_path):
        video = _test_video_path()
        output = tmp_path / "segment_copy.mp4"
        result = extract_segment(video, output, start=0.0, end=2.0, copy_mode=True)
        assert result.exists()
        assert result.stat().st_size > 0


class TestConcatenateSegments:
    @SKIP_NO_FFMPEG
    @SKIP_NO_VIDEO
    def test_concat_two_segments(self, tmp_path):
        video = _test_video_path()
        seg1 = tmp_path / "seg1.mp4"
        seg2 = tmp_path / "seg2.mp4"
        out = tmp_path / "concat.mp4"

        extract_segment(video, seg1, start=0.0, end=1.0, copy_mode=False)
        extract_segment(video, seg2, start=2.0, end=3.0, copy_mode=False)

        result = concatenate_segments([seg1, seg2], out)
        assert result.exists()
        assert result.stat().st_size > 0

    @SKIP_NO_FFMPEG
    @SKIP_NO_VIDEO
    def test_single_segment_copies(self, tmp_path):
        video = _test_video_path()
        seg1 = tmp_path / "seg1.mp4"
        out = tmp_path / "output.mp4"

        extract_segment(video, seg1, start=0.0, end=2.0, copy_mode=False)
        result = concatenate_segments([seg1], out)
        assert result.exists()

    def test_empty_segments_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No segments"):
            concatenate_segments([], tmp_path / "out.mp4")


class TestEditVideo:
    @SKIP_NO_FFMPEG
    @SKIP_NO_VIDEO
    def test_full_edit(self, tmp_path):
        video = _test_video_path()
        output = tmp_path / "cleaned.mp4"

        cutlist = CutList(
            source=video.name,
            source_duration=10.0,
            keep_spans=[
                TimeSpan(start=0.0, end=2.0, label="keep"),
                TimeSpan(start=5.0, end=8.0, label="keep"),
            ],
            discard_spans=[
                TimeSpan(start=2.0, end=5.0, label="discard"),
            ],
            retake_groups=[],
        )

        result = edit_video(video, cutlist, output, copy_mode=False)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_missing_video_raises(self, tmp_path):
        cutlist = CutList(
            source="missing.mp4", source_duration=10.0,
            keep_spans=[TimeSpan(0.0, 5.0)], discard_spans=[], retake_groups=[],
        )
        with pytest.raises(FileNotFoundError):
            edit_video(tmp_path / "missing.mp4", cutlist)

    def test_empty_keep_spans_raises(self, tmp_path):
        cutlist = CutList(
            source="test.mp4", source_duration=10.0,
            keep_spans=[], discard_spans=[], retake_groups=[],
        )
        # Even though video might not exist, empty keep_spans should raise first
        with pytest.raises(ValueError, match="no keep_spans"):
            edit_video(tmp_path / "test.mp4", cutlist)
