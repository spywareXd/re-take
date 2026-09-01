"""
End-to-End Pipeline Orchestrator

Chains all stages together into a single callable:
    input_video → Step 01 (transcribe) → Step 02/03 (split + token match)
                → Step 04 (semantic match) → Step 05 (cutlist) → Step 08 (FFmpeg edit)
                → output_video

Usage:
    from pipeline.run_pipeline import run_full_pipeline
    result = run_full_pipeline("data/videos/test.mp4")
    print(result["output_video"])
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Captures results from each stage of the pipeline."""

    input_video: str = ""
    output_video: str = ""
    source_duration: float = 0.0
    output_duration: float = 0.0
    time_saved: float = 0.0
    retake_groups_count: int = 0
    total_candidates: int = 0
    likely_retakes: int = 0
    ambiguous: int = 0
    stage_timings: dict[str, float] = field(default_factory=dict)
    cutlist_dict: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_video": self.input_video,
            "output_video": self.output_video,
            "source_duration": round(self.source_duration, 2),
            "output_duration": round(self.output_duration, 2),
            "time_saved": round(self.time_saved, 2),
            "retake_groups_count": self.retake_groups_count,
            "total_candidates": self.total_candidates,
            "likely_retakes": self.likely_retakes,
            "ambiguous": self.ambiguous,
            "stage_timings": {k: round(v, 3) for k, v in self.stage_timings.items()},
            "success": self.success,
            "error": self.error,
        }


def run_full_pipeline(
    input_video: str | Path,
    output_video: str | Path | None = None,
    provider: str = "whisperx",
    model_name: str = "base",
    device: str = "cuda",
    compute_type: str = "float16",
    take_strategy: str = "keep_later",
    copy_mode: bool = False,
    save_intermediates: bool = True,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> PipelineResult:
    """
    Run the complete retake detection and video editing pipeline.

    Args:
        input_video: Path to the input video file.
        output_video: Path for the cleaned output video. Auto-generated if None.
        provider: Transcription provider ('whisperx', 'whisper', 'assemblyai').
        model_name: ASR model size ('base', 'small', 'medium', 'large-v3').
        device: Compute device ('cuda', 'cpu').
        compute_type: Precision ('float16', 'int8', 'float32').
        take_strategy: Which take to keep ('keep_later', 'keep_earlier', 'keep_higher_confidence').
        copy_mode: If True, use FFmpeg stream copy (fast). If False, re-encode (precise).
        save_intermediates: If True, save intermediate JSON files.
        progress_callback: Optional callback(stage_name, progress_fraction) for UI updates.

    Returns:
        PipelineResult with all outputs and timing information.
    """
    result = PipelineResult(input_video=str(input_video))
    input_path = Path(input_video)

    if not input_path.exists():
        result.error = f"Input video not found: {input_path}"
        return result

    project_root = _find_project_root(input_path)

    def _progress(stage: str, frac: float):
        if progress_callback:
            progress_callback(stage, frac)
        logger.info(f"[Pipeline] {stage} ({frac * 100:.0f}%)")

    try:
        # ======================================================================
        # Stage 1: Transcription
        # ======================================================================
        _progress("Transcription", 0.0)
        t0 = time.time()

        from pipeline.step_01_transcription import TimestampGenerator
        from models.transcript import Transcript

        transcript_path = project_root / "data" / "raw" / "transcripts" / "output_transcript.json"

        generator = TimestampGenerator(
            provider=provider,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
        )
        transcript = generator.process(
            media_path=str(input_path),
            output_json_path=str(transcript_path) if save_intermediates else None,
        )

        result.source_duration = transcript.duration
        result.stage_timings["transcription"] = time.time() - t0
        _progress("Transcription", 1.0)

        # ======================================================================
        # Stage 2 & 3: Unit Splitting + Token Matching
        # ======================================================================
        _progress("Token Matching", 0.0)
        t0 = time.time()

        from pipeline.step_03_token_matching import (
            PAUSE_SPLIT_THRESHOLD_SECONDS,
            default_config as token_config,
            generate_candidate_pairs,
            load_transcript_units,
            save_comparable_units,
        )

        units = load_transcript_units(transcript, pause_threshold=PAUSE_SPLIT_THRESHOLD_SECONDS)

        if save_intermediates:
            units_path = project_root / "data" / "processed" / "units" / "comparable_units.json"
            save_comparable_units(units, output_path=units_path, source_transcript=str(transcript.source))

        candidates = generate_candidate_pairs(units, config=token_config)

        if save_intermediates:
            token_candidates_path = project_root / "data" / "processed" / "candidates" / "token_candidates.json"
            token_candidates_path.parent.mkdir(parents=True, exist_ok=True)
            token_candidates_path.write_text(
                json.dumps([c.to_dict() for c in candidates], indent=2), encoding="utf-8"
            )

        result.stage_timings["token_matching"] = time.time() - t0
        _progress("Token Matching", 1.0)

        # ======================================================================
        # Stage 4: Semantic Matching
        # ======================================================================
        _progress("Semantic Matching", 0.0)
        t0 = time.time()

        from pipeline.step_04_semantic_matching import run_stage2, save_semantic_candidates

        stage2_results = run_stage2(candidates)

        if save_intermediates:
            sem_path = project_root / "data" / "processed" / "candidates" / "semantic_candidates.json"
            save_semantic_candidates(stage2_results, output_path=sem_path)

        result.total_candidates = len(stage2_results)
        result.likely_retakes = sum(1 for r in stage2_results if r["classification"] == "likely_retake")
        result.ambiguous = sum(1 for r in stage2_results if r["classification"] == "ambiguous")
        result.stage_timings["semantic_matching"] = time.time() - t0
        _progress("Semantic Matching", 1.0)

        # ======================================================================
        # Stage 5: Cutlist Generation
        # ======================================================================
        _progress("Cutlist Generation", 0.0)
        t0 = time.time()

        from pipeline.step_05_cutlist import build_cutlist, save_cutlist

        cutlist = build_cutlist(
            candidates=stage2_results,
            source=str(input_path.name),
            source_duration=transcript.duration,
            strategy=take_strategy,
        )

        if save_intermediates:
            cutlist_path = project_root / "data" / "processed" / "cutlists" / "cutlist.json"
            save_cutlist(cutlist, output_path=cutlist_path)

        result.retake_groups_count = len(cutlist.retake_groups)
        result.output_duration = cutlist.output_duration
        result.time_saved = cutlist.source_duration - cutlist.output_duration
        result.cutlist_dict = cutlist.to_dict()
        result.stage_timings["cutlist_generation"] = time.time() - t0
        _progress("Cutlist Generation", 1.0)

        # ======================================================================
        # Stage 8: Video Editing
        # ======================================================================
        _progress("Video Editing", 0.0)
        t0 = time.time()

        from pipeline.step_08_video_edit import edit_video

        if output_video is None:
            out_dir = project_root / "data" / "processed" / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_video_path = out_dir / f"{input_path.stem}_cleaned.mp4"
        else:
            output_video_path = Path(output_video)

        edit_video(
            input_path=input_path,
            cutlist=cutlist,
            output_path=output_video_path,
            copy_mode=copy_mode,
        )

        result.output_video = str(output_video_path)
        result.stage_timings["video_editing"] = time.time() - t0
        _progress("Video Editing", 1.0)

        result.success = True

    except Exception as e:
        result.error = str(e)
        logger.exception(f"Pipeline failed: {e}")

    return result


def _find_project_root(reference_path: Path) -> Path:
    """Walk up from a reference path to find the project root (contains pyproject.toml)."""
    current = reference_path.resolve()
    if current.is_file():
        current = current.parent

    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback: assume the CWD is the project root
    return Path.cwd()
