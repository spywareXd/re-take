"""
End-to-End Pipeline Orchestrator

Chains all stages together into a single callable:
    input_video → Step 01 (transcribe) 
                → Step 02/03 (split + token match)
                → Step 04 (semantic match) 
                → Step 05 (span aggregation)
                → Step 06 (LLM adjudication)
                → Step 08 (cut generator + FFmpeg edit)
                → output_video

Usage:
    from pipeline.run_pipeline import run_full_pipeline
    result = run_full_pipeline("data/videos/test.mp4")
    print(result.output_video)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Captures results from each stage of the pipeline."""

    video_id: str = ""
    input_video: str = ""
    output_video: str = ""
    source_duration: float = 0.0
    output_duration: float = 0.0
    time_saved: float = 0.0
    retake_groups_count: int = 0
    total_candidates: int = 0
    likely_retakes: int = 0
    ambiguous: int = 0
    spans_count: int = 0
    adjudicated_count: int = 0
    stage_timings: dict[str, float] = field(default_factory=dict)
    cutlist_dict: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: str = ""
    # Optional: only set when run_full_pipeline(detect_intra_unit=True); None (and absent from to_dict) otherwise
    intra_unit_candidates_count: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "video_id": self.video_id,
            "input_video": self.input_video,
            "output_video": self.output_video,
            "source_duration": round(self.source_duration, 2),
            "output_duration": round(self.output_duration, 2),
            "time_saved": round(self.time_saved, 2),
            "retake_groups_count": self.retake_groups_count,
            "total_candidates": self.total_candidates,
            "likely_retakes": self.likely_retakes,
            "ambiguous": self.ambiguous,
            "spans_count": self.spans_count,
            "adjudicated_count": self.adjudicated_count,
            "stage_timings": {k: round(v, 3) for k, v in self.stage_timings.items()},
            "cutlist_dict": self.cutlist_dict,
            "success": self.success,
            "error": self.error,
        }
        if self.intra_unit_candidates_count is not None:   # key absent when the feature is off
            d["intra_unit_candidates_count"] = self.intra_unit_candidates_count
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineResult:
        return cls(
            video_id=data.get("video_id", ""),
            input_video=data.get("input_video", ""),
            output_video=data.get("output_video", ""),
            source_duration=float(data.get("source_duration", 0.0)),
            output_duration=float(data.get("output_duration", 0.0)),
            time_saved=float(data.get("time_saved", 0.0)),
            retake_groups_count=int(data.get("retake_groups_count", 0)),
            total_candidates=int(data.get("total_candidates", 0)),
            likely_retakes=int(data.get("likely_retakes", 0)),
            ambiguous=int(data.get("ambiguous", 0)),
            spans_count=int(data.get("spans_count", 0)),
            adjudicated_count=int(data.get("adjudicated_count", 0)),
            stage_timings=dict(data.get("stage_timings", {})),
            cutlist_dict=dict(data.get("cutlist_dict", {})),
            success=bool(data.get("success", False)),
            error=str(data.get("error", "")),
            intra_unit_candidates_count=data.get("intra_unit_candidates_count"),
        )


@dataclass
class VideoPipelinePaths:
    """Canonical per-video assortment of file paths under data/processed/<video_id>/"""

    video_id: str
    video_dir: Path
    candidates_dir: Path
    cutlists_dir: Path
    outputs_dir: Path
    units_dir: Path
    transcript: Path
    units: Path
    token_candidates: Path
    semantic_candidates: Path
    retake_spans: Path
    adjudicated_spans: Path
    cutlist: Path
    output_video: Path
    pipeline_result: Path
    video_info: Path

    def mkdir(self) -> None:
        """Create all subdirectories for this video_id."""
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.cutlists_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.units_dir.mkdir(parents=True, exist_ok=True)


def _load_video_registry(registry_path: Path) -> dict[str, dict[str, Any]]:
    if registry_path.exists():
        try:
            return json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read registry at {registry_path}: {e}")
    return {}


def _save_video_registry(registry_path: Path, registry: dict[str, dict[str, Any]]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def get_or_create_video_id(
    input_video: str | Path,
    project_root: Path,
    force_new_id: bool = False,
) -> str:
    """
    Finds the existing video_id (video_1, video_2, ...) for input_video or
    allocates the next available video_id.
    """
    from datetime import datetime, timezone

    input_path = Path(input_video)
    processed_root = project_root / "data" / "processed"
    registry_path = processed_root / "video_registry.json"
    registry = _load_video_registry(registry_path)

    # Prune stale registry entries if their video directories were deleted on disk
    if processed_root.exists():
        stale_keys = [k for k in list(registry.keys()) if k.startswith("video_") and not (processed_root / k).exists()]
        if stale_keys:
            for k in stale_keys:
                del registry[k]
            _save_video_registry(registry_path, registry)

    existing_nums: set[int] = set()
    for k in registry.keys():
        if k.startswith("video_"):
            try:
                existing_nums.add(int(k.split("_", 1)[1]))
            except ValueError:
                pass

    if processed_root.exists():
        for d in processed_root.iterdir():
            if d.is_dir() and d.name.startswith("video_"):
                try:
                    existing_nums.add(int(d.name.split("_", 1)[1]))
                except ValueError:
                    pass

    target_name = input_path.name
    target_stem = input_path.stem
    try:
        target_resolved = str(input_path.resolve())
    except Exception:
        target_resolved = str(input_path)

    if not force_new_id:
        for vid, info in registry.items():
            if (
                info.get("resolved_path") == target_resolved
                or info.get("filename") == target_name
                or (info.get("stem") == target_stem and target_stem != "")
            ):
                return vid

        for num in sorted(existing_nums):
            cand_id = f"video_{num}"
            cand_info_file = processed_root / cand_id / "video_info.json"
            if cand_info_file.exists():
                try:
                    info = json.loads(cand_info_file.read_text(encoding="utf-8"))
                    if (
                        info.get("resolved_path") == target_resolved
                        or info.get("filename") == target_name
                        or (info.get("stem") == target_stem and target_stem != "")
                    ):
                        registry[cand_id] = info
                        _save_video_registry(registry_path, registry)
                        return cand_id
                except Exception:
                    pass

    next_num = max(existing_nums, default=0) + 1
    new_video_id = f"video_{next_num}"

    info_payload = {
        "video_id": new_video_id,
        "input_video": str(input_video),
        "resolved_path": target_resolved,
        "filename": target_name,
        "stem": target_stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    registry[new_video_id] = info_payload
    _save_video_registry(registry_path, registry)

    video_dir = processed_root / new_video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "video_info.json").write_text(json.dumps(info_payload, indent=2), encoding="utf-8")

    return new_video_id


def get_video_pipeline_paths(
    input_video: str | Path,
    project_root: Path | None = None,
    output_video: str | Path | None = None,
    processed_dir: str | Path | None = None,
    video_id: str | None = None,
    force_new_id: bool = False,
) -> VideoPipelinePaths:
    """
    Get the canonical assortment of file paths for a specific video following the exact schema:
    data/processed/<video_id>/
        candidates/
            token_candidates.json
            semantic_candidates.json
            retake_spans.json
            adjudicated_spans.json
        cutlists/
            cutlist.json
        outputs/
            <stem>_cleaned.mp4
        units/
            comparable_units.json
        transcript.json
        pipeline_result.json
        video_info.json
    """
    input_path = Path(input_video)
    if project_root is None:
        project_root = _find_project_root(input_path)

    if video_id is None:
        if processed_dir is not None:
            v_id = Path(processed_dir).name
        else:
            v_id = get_or_create_video_id(input_path, project_root=project_root, force_new_id=force_new_id)
    else:
        v_id = video_id

    if processed_dir is not None:
        video_dir = Path(processed_dir)
    else:
        video_dir = project_root / "data" / "processed" / v_id

    candidates_dir = video_dir / "candidates"
    cutlists_dir = video_dir / "cutlists"
    outputs_dir = video_dir / "outputs"
    units_dir = video_dir / "units"

    stem = input_path.stem
    if output_video is not None:
        out_vid = Path(output_video)
    else:
        out_vid = outputs_dir / f"{stem}_cleaned.mp4"

    return VideoPipelinePaths(
        video_id=v_id,
        video_dir=video_dir,
        candidates_dir=candidates_dir,
        cutlists_dir=cutlists_dir,
        outputs_dir=outputs_dir,
        units_dir=units_dir,
        transcript=video_dir / "transcript.json",
        units=units_dir / "comparable_units.json",
        token_candidates=candidates_dir / "token_candidates.json",
        semantic_candidates=candidates_dir / "semantic_candidates.json",
        retake_spans=candidates_dir / "retake_spans.json",
        adjudicated_spans=candidates_dir / "adjudicated_spans.json",
        cutlist=cutlists_dir / "cutlist.json",
        output_video=out_vid,
        pipeline_result=video_dir / "pipeline_result.json",
        video_info=video_dir / "video_info.json",
    )


def load_pipeline_result(
    input_video: str | Path,
    project_root: Path | None = None,
    processed_dir: str | Path | None = None,
    video_id: str | None = None,
) -> PipelineResult | None:
    """Load previously persisted PipelineResult for a video if available."""
    paths = get_video_pipeline_paths(
        input_video=input_video,
        project_root=project_root,
        processed_dir=processed_dir,
        video_id=video_id,
    )
    if paths.pipeline_result.exists():
        try:
            data = json.loads(paths.pipeline_result.read_text(encoding="utf-8"))
            return PipelineResult.from_dict(data)
        except Exception as e:
            logger.warning(f"Could not load persisted pipeline result from {paths.pipeline_result}: {e}")
    return None


def run_full_pipeline(
    input_video: str | Path,
    output_video: str | Path | None = None,
    provider: str = "whisperx",
    model_name: str = "large-v2",
    device: str = "cuda",
    compute_type: str = "float16",
    llm_provider: str = "gemini",
    llm_model: str = "gemini-2.5-flash",
    copy_mode: bool = False,
    save_intermediates: bool = True,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    processed_dir: str | Path | None = None,
    reuse_existing_transcript: bool = True,
    video_id: str | None = None,
    force_new_id: bool = False,
    detect_intra_unit: bool = False,
) -> PipelineResult:
    """
    Run the complete retake detection and video editing pipeline.

    Args:
        input_video: Path to the input video file.
        output_video: Path for the cleaned output video. Defaults to data/processed/<video_id>/outputs/<stem>_cleaned.mp4.
        provider: Transcription provider ('whisperx', 'whisper', 'assemblyai').
        model_name: ASR model size ('base', 'small', 'medium', 'large-v2', 'large-v3').
        device: Compute device ('cuda', 'cpu').
        compute_type: Precision ('float16', 'int8', 'float32').
        llm_provider: LLM provider for Stage 6 ('gemini', 'openai', 'mock').
        llm_model: LLM model name.
        copy_mode: If True, use FFmpeg stream copy (fast). If False, re-encode (precise).
        save_intermediates: If True, save intermediate JSON files in the per-video processed dir.
        progress_callback: Optional callback(stage_name, progress_fraction) for UI updates.
        processed_dir: Custom output directory for processed files. Defaults to data/processed/<video_id>/.
        reuse_existing_transcript: If True and transcript.json exists, reuses it without re-transcribing.
        video_id: Explicit video_id (e.g. 'video_1'). If None, automatically assigned.
        force_new_id: If True, always allocates the next available video_id.
        detect_intra_unit: OPTIONAL, default off. If True, also suggest restarts inside single units and save
            them to <video dir>/candidates/intra_unit_candidates.json for the review UI. Suggestions never feed
            candidate generation, span aggregation, adjudication or the cutlist.

    Returns:
        PipelineResult with all outputs and timing information.
    """
    result = PipelineResult(input_video=str(input_video))
    input_path = Path(input_video)

    if not input_path.exists():
        result.error = f"Input video not found: {input_path}"
        return result

    project_root = _find_project_root(input_path)
    paths = get_video_pipeline_paths(
        input_video=input_path,
        project_root=project_root,
        output_video=output_video,
        processed_dir=processed_dir,
        video_id=video_id,
        force_new_id=force_new_id,
    )
    result.video_id = paths.video_id

    if save_intermediates:
        paths.mkdir()

    def _progress(stage: str, frac: float):
        if progress_callback:
            progress_callback(stage, frac)
        logger.info(f"[Pipeline] {stage} ({frac * 100:.0f}%)")

    try:
        # ======================================================================
        # Stage 1: Transcription (Step 01)
        # ======================================================================
        _progress("Transcription", 0.0)
        t0 = time.time()

        from models.transcript import Transcript
        from pipeline.step_01_transcription import TimestampGenerator

        transcript: Transcript | None = None
        if reuse_existing_transcript and paths.transcript.exists():
            try:
                loaded = Transcript.from_json_file(str(paths.transcript))
                if loaded.duration > 0:
                    transcript = loaded
                    logger.info(f"Reusing existing transcript from: {paths.transcript}")
            except Exception as e:
                logger.warning(f"Failed to load existing transcript from {paths.transcript}: {e}")

        # Check raw transcripts folder as fallback
        if transcript is None and reuse_existing_transcript:
            raw_fallback = project_root / "data" / "raw" / "transcripts" / f"{input_path.stem}.json"
            if raw_fallback.exists():
                try:
                    loaded = Transcript.from_json_file(str(raw_fallback))
                    if loaded.duration > 0:
                        transcript = loaded
                        logger.info(f"Reusing existing transcript from raw folder: {raw_fallback}")
                except Exception:
                    pass

        if transcript is None:
            generator = TimestampGenerator(
                provider=provider,
                model_name=model_name,
                device=device,
                compute_type=compute_type,
            )
            transcript = generator.process(
                media_path=str(input_path),
                output_json_path=str(paths.transcript) if save_intermediates else None,
            )
            # Also save per-video raw transcript for safety
            if save_intermediates:
                raw_out = project_root / "data" / "raw" / "transcripts" / f"{input_path.stem}.json"
                raw_out.parent.mkdir(parents=True, exist_ok=True)
                try:
                    transcript.to_json_file(str(raw_out))
                except Exception:
                    pass
        elif save_intermediates and not paths.transcript.exists():
            transcript.to_json_file(str(paths.transcript))

        result.source_duration = transcript.duration
        result.stage_timings["transcription"] = time.time() - t0
        _progress("Transcription", 1.0)

        # ======================================================================
        # Stage 2 & 3: Unit Splitting + Token Matching (Step 02 & Step 03)
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
            save_comparable_units(units, output_path=paths.units, source_transcript=str(transcript.source))

        # Optional, flag-gated: suggestions for the review UI only. Not passed to any later stage.
        if detect_intra_unit:
            from pipeline.step_03_token_matching import find_intra_unit_candidates, save_intra_unit_candidates

            intra_candidates = find_intra_unit_candidates(units, transcript, video_id=paths.video_id)
            result.intra_unit_candidates_count = len(intra_candidates)
            if save_intermediates:
                save_intra_unit_candidates(
                    intra_candidates,
                    output_path=paths.candidates_dir / "intra_unit_candidates.json",
                    video_id=paths.video_id,
                )

        candidates = generate_candidate_pairs(units, config=token_config)

        if save_intermediates:
            paths.token_candidates.parent.mkdir(parents=True, exist_ok=True)
            paths.token_candidates.write_text(
                json.dumps([c.to_dict() for c in candidates], indent=2), encoding="utf-8"
            )

        result.stage_timings["token_matching"] = time.time() - t0
        _progress("Token Matching", 1.0)

        # ======================================================================
        # Stage 4: Semantic Matching (Step 04)
        # ======================================================================
        _progress("Semantic Matching", 0.0)
        t0 = time.time()

        from pipeline.step_04_semantic_matching import run_stage2, save_semantic_candidates

        stage2_results = run_stage2(candidates)

        if save_intermediates:
            save_semantic_candidates(
                stage2_results,
                output_path=paths.semantic_candidates,
                source_candidates=str(paths.token_candidates),
            )

        result.total_candidates = len(stage2_results)
        result.likely_retakes = sum(1 for r in stage2_results if r["classification"] == "likely_retake")
        result.ambiguous = sum(1 for r in stage2_results if r["classification"] == "ambiguous")
        result.stage_timings["semantic_matching"] = time.time() - t0
        _progress("Semantic Matching", 1.0)

        # ======================================================================
        # Stage 5: Span Aggregation (Step 05)
        # ======================================================================
        _progress("Span Aggregation", 0.0)
        t0 = time.time()

        from pipeline.step_05_span_aggregation import build_retake_spans, save_retake_spans

        retake_spans = build_retake_spans(stage2_results, units=units)

        if save_intermediates:
            save_retake_spans(
                retake_spans,
                output_path=paths.retake_spans,
                source_candidates=str(paths.semantic_candidates),
            )

        result.spans_count = len(retake_spans)
        result.stage_timings["span_aggregation"] = time.time() - t0
        _progress("Span Aggregation", 1.0)

        # ======================================================================
        # Stage 6: LLM Adjudication (Step 06)
        # ======================================================================
        _progress("LLM Adjudication", 0.0)
        t0 = time.time()

        from pipeline.step_06_llm_adjudication import (
            AdjudicationConfig,
            run_stage6,
            save_adjudicated_spans,
        )

        adj_config = AdjudicationConfig(
            provider=llm_provider,
            model=llm_model,
        )
        adjudicated_spans = run_stage6(retake_spans, units=units, config=adj_config)

        if save_intermediates:
            save_adjudicated_spans(
                adjudicated_spans,
                output_path=paths.adjudicated_spans,
                source_spans=str(paths.retake_spans),
                model=llm_model,
            )

        result.adjudicated_count = sum(
            1 for s in adjudicated_spans if getattr(s, "adjudication_status", "") == "resolved"
        )
        result.stage_timings["llm_adjudication"] = time.time() - t0
        _progress("LLM Adjudication", 1.0)

        # ======================================================================
        # Stage 7: Cutlist Generation (Step 08 cut_generator)
        # ======================================================================
        _progress("Cutlist Generation", 0.0)
        t0 = time.time()

        from pipeline.step_08_video_edit import (
            generate_cuts_from_adjudicated_spans,
            save_cutlist,
        )

        cutlist = generate_cuts_from_adjudicated_spans(
            adjudicated_spans=adjudicated_spans,
            units=units,
            source=str(input_path.name),
            source_duration=transcript.duration,
        )

        if save_intermediates:
            save_cutlist(cutlist, output_path=paths.cutlist)

        result.retake_groups_count = len(cutlist.retake_groups)
        result.output_duration = cutlist.output_duration
        result.time_saved = cutlist.source_duration - cutlist.output_duration
        result.cutlist_dict = cutlist.to_dict()
        result.stage_timings["cutlist_generation"] = time.time() - t0
        _progress("Cutlist Generation", 1.0)

        # ======================================================================
        # Stage 8: Video Editing (Step 08 ffmpeg_editor)
        # ======================================================================
        _progress("Video Editing", 0.0)
        t0 = time.time()

        from pipeline.step_08_video_edit import edit_video

        output_video_path = paths.output_video
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        edit_video(
            input_path=input_path,
            cutlist=cutlist,
            output_path=output_video_path,
            copy_mode=copy_mode,
        )

        # Keep a copy in data/processed/outputs/ if outputs dir exists or for global access
        outputs_dir = project_root / "data" / "processed" / "outputs"
        if outputs_dir.exists() and output_video_path != (outputs_dir / f"{input_path.stem}_cleaned.mp4"):
            try:
                import shutil
                shutil.copy2(output_video_path, outputs_dir / f"{input_path.stem}_cleaned.mp4")
            except Exception as e:
                logger.debug(f"Could not copy cleaned video to outputs dir: {e}")

        result.output_video = str(output_video_path)
        result.stage_timings["video_editing"] = time.time() - t0
        _progress("Video Editing", 1.0)

        result.success = True

        # Persist full pipeline summary result envelope for this video
        if save_intermediates:
            paths.pipeline_result.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )

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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the end-to-end retake detection and video editing pipeline.")
    parser.add_argument("input_video", nargs="?", default="data/videos/sample_e.mp4", help="Path to input video (default: data/videos/sample_e.mp4)")
    parser.add_argument("--output", "-o", default=None, help="Custom output video path")
    parser.add_argument("--provider", default="whisperx", choices=["whisperx", "whisper", "assemblyai"], help="ASR provider")
    parser.add_argument("--model", default="large-v2", help="ASR model size")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Compute device")
    parser.add_argument("--llm-provider", default="gemini", choices=["gemini", "openai", "mock"], help="LLM provider")
    parser.add_argument("--llm-model", default="gemini-2.5-flash", help="LLM model name")
    parser.add_argument("--copy-mode", action="store_true", help="Fast FFmpeg stream-copy mode")
    parser.add_argument("--force-new-id", action="store_true", help="Force assigning next video_id even if previously registered")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print(f"\n=======================================================")
    print(f" Starting Retake Pipeline on: {args.input_video}")
    print(f"=======================================================\n")

    result = run_full_pipeline(
        input_video=args.input_video,
        output_video=args.output,
        provider=args.provider,
        model_name=args.model,
        device=args.device,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        copy_mode=args.copy_mode,
        force_new_id=args.force_new_id,
    )

    if result.success:
        print(f"\n[OK] Pipeline completed successfully!")
        print(f"Video ID:            {result.video_id}")
        print(f"Output Video:        {result.output_video}")
        print(f"Original Duration:   {result.source_duration:.2f}s")
        print(f"Cleaned Duration:    {result.output_duration:.2f}s")
        print(f"Time Saved:          {result.time_saved:.2f}s")
        print(f"Retake Groups Cut:   {result.retake_groups_count}")
        print(f"Persisted Folder:    data/processed/{result.video_id}/\n")
    else:
        print(f"\n[ERROR] Pipeline failed: {result.error}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
