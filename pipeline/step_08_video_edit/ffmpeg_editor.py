"""
FFmpeg Video Editor — Produces cleaned video from cutlist keep spans.

Takes the keep_spans from a CutList and uses FFmpeg to:
1. Extract each keep segment from the source video
2. Concatenate all segments into a single output video
3. Clean up temporary segment files

Supports two modes:
- Re-encode mode (default): Frame-exact cuts via re-encoding. Slower but precise.
- Copy mode: Stream-copy for speed. Cuts may not be frame-exact (keyframe-aligned).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence, Union

from pipeline.step_05_cutlist.schemas import CutList, TimeSpan

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data") / "processed" / "outputs"


def _run_ffmpeg(args: list[str], description: str = "FFmpeg") -> subprocess.CompletedProcess:
    """Run an FFmpeg command and raise on failure."""
    logger.info(f"[{description}] Running: ffmpeg {' '.join(args)}")
    result = subprocess.run(
        ["ffmpeg"] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        logger.error(f"[{description}] FFmpeg stderr:\n{result.stderr}")
        raise RuntimeError(
            f"FFmpeg failed ({description}): exit code {result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )
    return result


def extract_segment(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    start: float,
    end: float,
    copy_mode: bool = False,
) -> Path:
    """
    Extract a single time segment from the source video.

    Args:
        input_path: Source video file.
        output_path: Destination for the extracted segment.
        start: Start time in seconds.
        end: End time in seconds.
        copy_mode: If True, use stream copy (fast but keyframe-aligned).
                   If False, re-encode for frame-exact cuts.

    Returns:
        Path to the extracted segment.
    """
    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    duration = end - start

    if copy_mode:
        # Stream copy: fast but may not be frame-exact
        args = [
            "-y",
            "-ss", f"{start:.3f}",
            "-i", str(input_p),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_p),
        ]
    else:
        # Re-encode: slower but frame-exact
        args = [
            "-y",
            "-ss", f"{start:.3f}",
            "-i", str(input_p),
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            str(output_p),
        ]

    _run_ffmpeg(args, description=f"Extract segment [{start:.2f}s → {end:.2f}s]")
    return output_p


def concatenate_segments(
    segment_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    temp_dir: Union[str, Path, None] = None,
) -> Path:
    """
    Concatenate multiple video segments into one output file using FFmpeg concat demuxer.

    Args:
        segment_paths: Ordered list of segment file paths to concatenate.
        output_path: Destination for the final concatenated video.
        temp_dir: Directory for the temporary concat list file.

    Returns:
        Path to the concatenated output video.
    """
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    if not segment_paths:
        raise ValueError("No segments to concatenate")

    if len(segment_paths) == 1:
        # Single segment — just copy/move it
        import shutil
        shutil.copy2(str(segment_paths[0]), str(output_p))
        return output_p

    # Write concat list file
    if temp_dir is None:
        temp_dir = output_p.parent
    concat_list_path = Path(temp_dir) / "_concat_list.txt"

    with open(concat_list_path, "w", encoding="utf-8") as f:
        for seg_path in segment_paths:
            # FFmpeg concat requires forward slashes and escaped single quotes
            safe_path = str(Path(seg_path).resolve()).replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

    args = [
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_path),
        "-c", "copy",
        str(output_p),
    ]

    _run_ffmpeg(args, description="Concatenate segments")

    # Cleanup concat list
    try:
        concat_list_path.unlink()
    except OSError:
        pass

    return output_p


def edit_video(
    input_path: Union[str, Path],
    cutlist: CutList,
    output_path: Union[str, Path, None] = None,
    copy_mode: bool = False,
    cleanup_segments: bool = True,
) -> Path:
    """
    Main entry point: apply a CutList to a source video and produce a cleaned output.

    Args:
        input_path: Path to the source video file.
        cutlist: CutList containing keep_spans to extract and concatenate.
        output_path: Destination for the cleaned video. Auto-generated if None.
        copy_mode: If True, use stream copy (fast). If False, re-encode (precise).
        cleanup_segments: If True, remove temporary segment files after concatenation.

    Returns:
        Path to the final cleaned video.
    """
    if not cutlist.keep_spans:
        raise ValueError("CutList has no keep_spans — nothing to produce")

    input_p = Path(input_path)
    if not input_p.exists():
        raise FileNotFoundError(f"Source video not found: {input_p}")

    # Determine output path
    if output_path is None:
        stem = input_p.stem
        out_dir = DEFAULT_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        output_p = out_dir / f"{stem}_cleaned.mp4"
    else:
        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)

    # Create temp directory for intermediate segments
    temp_dir = output_p.parent / "_temp_segments"
    temp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Editing video: {input_p.name} → {output_p.name} "
        f"({len(cutlist.keep_spans)} segments, copy_mode={copy_mode})"
    )

    # Step 1: Extract each keep segment
    segment_paths: list[Path] = []
    for idx, span in enumerate(cutlist.keep_spans):
        seg_path = temp_dir / f"segment_{idx:03d}.mp4"
        extract_segment(
            input_path=input_p,
            output_path=seg_path,
            start=span.start,
            end=span.end,
            copy_mode=copy_mode,
        )
        segment_paths.append(seg_path)
        logger.info(f"  Extracted segment {idx + 1}/{len(cutlist.keep_spans)}: "
                     f"[{span.start:.2f}s → {span.end:.2f}s] ({span.duration:.2f}s)")

    # Step 2: Concatenate all segments
    concatenate_segments(
        segment_paths=segment_paths,
        output_path=output_p,
        temp_dir=temp_dir,
    )

    logger.info(f"[OK] Video editing complete: {output_p}")
    logger.info(f"  Original: {cutlist.source_duration:.2f}s → "
                f"Cleaned: {cutlist.output_duration:.2f}s "
                f"(removed {cutlist.source_duration - cutlist.output_duration:.2f}s)")

    # Step 3: Cleanup temp segment files
    if cleanup_segments:
        for seg_path in segment_paths:
            try:
                seg_path.unlink()
            except OSError:
                pass
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    return output_p
