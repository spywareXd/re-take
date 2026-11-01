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

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence, Union

from pipeline.step_05_cutlist.schemas import CutList, TimeSpan
from pipeline.step_08_video_edit.ffmpeg_compat import filter_graph_file_args

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


def probe_media(path: Union[str, Path]) -> tuple[float | None, bool]:
    """(container duration in seconds, has an audio stream) of a media file via ffprobe; (None, True) if probing fails."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        ).stdout
        data = json.loads(out)
        duration = float(data["format"]["duration"])
        has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
        return duration, has_audio
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError):
        return None, True


def build_concat_filter(spans: Sequence[TimeSpan], fade_s: float, has_audio: bool = True) -> str:
    """
    Filter graph for one inputs-per-span ffmpeg pass: each span is input i (opened with -ss/-t), its timestamps are reset,
    the audio gets a short fade in/out so a cut never lands on a non-zero sample, and the concat FILTER joins them sample-exactly
    (no per-segment AAC priming, so no drift).
    """
    lines = []
    for i, s in enumerate(spans):
        lines.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}];")
        if has_audio:
            f = min(fade_s, s.duration / 2.0)
            fade = f",afade=t=in:st=0:d={f:.4f},afade=t=out:st={s.duration - f:.4f}:d={f:.4f}" if f > 0 else ""
            lines.append(f"[{i}:a]asetpts=PTS-STARTPTS{fade}[a{i}];")
    joined = "".join(f"[v{i}][a{i}]" if has_audio else f"[v{i}]" for i in range(len(spans)))
    lines.append(f"{joined}concat=n={len(spans)}:v=1:a={1 if has_audio else 0}[outv]" + ("[outa]" if has_audio else ""))
    return "\n".join(lines)


def _edit_single_pass(input_p: Path, spans: Sequence[TimeSpan], output_p: Path, temp_dir: Path, fade_s: float) -> None:
    _, has_audio = probe_media(input_p)
    script = temp_dir / "_filter.txt"
    script.write_text(build_concat_filter(spans, fade_s, has_audio), encoding="utf-8")
    args = ["-y"]
    for s in spans:
        args += ["-ss", f"{s.start:.3f}", "-t", f"{s.duration:.3f}", "-i", str(input_p)]
    args += filter_graph_file_args(script) + ["-map", "[outv]"]
    if has_audio:
        args += ["-map", "[outa]"]
    args += ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]
    if has_audio:
        args += ["-c:a", "aac", "-b:a", "192k"]
    args.append(str(output_p))
    try:
        _run_ffmpeg(args, description=f"Single-pass edit ({len(spans)} segments)")
    finally:
        try:
            script.unlink()
        except OSError:
            pass


def edit_video(
    input_path: Union[str, Path],
    cutlist: CutList,
    output_path: Union[str, Path, None] = None,
    copy_mode: bool = False,
    cleanup_segments: bool = True,
    fade_ms: float = 12.0,
) -> Path:
    """
    Main entry point: apply a CutList to a source video and produce a cleaned output.

    Args:
        input_path: Path to the source video file.
        cutlist: CutList containing keep_spans to extract and concatenate.
        output_path: Destination for the cleaned video. Auto-generated if None.
        copy_mode: If True, use stream copy (fast). If False, re-encode (precise).
        cleanup_segments: If True, remove temporary segment files after concatenation.
        fade_ms: Re-encode mode only: audio fade in/out at every cut (ms); 0 disables.

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

    if not copy_mode:
        # One ffmpeg pass: sample-exact joins with short fades (no per-segment AAC padding, no drift)
        _edit_single_pass(input_p, cutlist.keep_spans, output_p, temp_dir, fade_s=fade_ms / 1000.0)
        try:
            temp_dir.rmdir()
        except OSError:
            pass
        logger.info(f"[OK] Video editing complete: {output_p}")
        return output_p

    # Step 1: Extract each keep segment (stream-copy mode)
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
