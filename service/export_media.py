"""Export engine: ffprobe facts (size, frame rate, VFR), the NVENC test encode, keep-span math, the ffmpeg render with progress, and the text exports (SRT, VTT, EDL,
cut-list JSON, transcript). The cuts are made exactly like the pipeline's own editor: the same concat filter (`build_concat_filter`, including its short audio fade at every
cut), so an export never cuts differently from the cleaned video the pipeline makes.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .export_options import (OptionError, plan_audio, plan_fps, plan_resolution, plan_video)

MIN_KEEP_S = 0.04                       # a kept sliver shorter than this (about one frame) is dropped


class ExportError(Exception):
    """An ffmpeg / ffprobe failure; the message is ffmpeg's own text, verbatim."""


# ---- ffprobe ----------------------------------------------------------------------------------------------------------------------------------------------
def _fraction(s: Optional[str]) -> Optional[float]:
    try:
        a, b = str(s).split("/")
        return float(a) / float(b) if float(b) else None
    except (ValueError, AttributeError):
        try:
            return float(s)
        except (TypeError, ValueError):
            return None


def ffprobe_json(path: Path, *args: str) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-of", "json", *args, str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise ExportError(r.stderr.strip() or f"ffprobe failed on {path}")
    return json.loads(r.stdout or "{}")


def detect_vfr(path: Path, max_packets: int = 400) -> bool:
    """True when the video's frame timestamps are not evenly spaced. Looks at the presentation times of the first packets (sorted: B-frame order does not matter)
    and at ffprobe's r_frame_rate vs avg_frame_rate; the two agree for constant frame rate files."""
    d = ffprobe_json(path, "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,avg_frame_rate", "-show_entries", "packet=pts_time",
                     "-read_intervals", f"%+#{max_packets}")
    st = (d.get("streams") or [{}])[0]
    r, a = _fraction(st.get("r_frame_rate")), _fraction(st.get("avg_frame_rate"))
    times = sorted(float(p["pts_time"]) for p in d.get("packets", []) if p.get("pts_time") not in (None, "N/A"))
    deltas = [b - a_ for a_, b in zip(times, times[1:]) if b - a_ > 1e-6]
    if len(deltas) >= 3:
        lo, hi = min(deltas), max(deltas)
        if hi / lo > 1.5:
            return True
        return False
    return bool(r and a and abs(r - a) / max(r, a) > 0.02)


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: Optional[float]
    duration: float
    has_audio: bool
    has_video: bool
    vfr: bool
    video_codec: str = ""
    nb_frames: Optional[int] = None


def probe_video(path: Path, check_vfr: bool = True) -> VideoInfo:
    d = ffprobe_json(path, "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames:stream_side_data=rotation")
    streams = d.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    dur = float(d.get("format", {}).get("duration") or 0.0)
    if v is None:
        return VideoInfo(str(path), 0, 0, None, dur, has_audio, False, False)
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    rot = next((abs(int(sd.get("rotation", 0))) for sd in v.get("side_data_list", []) if "rotation" in sd), 0)
    if rot % 180 == 90:                                                            # ffmpeg auto-rotates: the picture the viewer sees has the sides swapped
        w, h = h, w
    fps = _fraction(v.get("avg_frame_rate")) or _fraction(v.get("r_frame_rate"))
    return VideoInfo(str(path), w, h, fps, dur, has_audio, True, detect_vfr(path) if check_vfr else False, v.get("codec_name", ""),
                     int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None)


# ---- hardware encoders: offered only if ffmpeg lists them AND a test encode works ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _encoder_list() -> str:
    try:
        return subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


@lru_cache(maxsize=4)
def _test_encode(encoder: str) -> bool:
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=320x240:d=0.2:r=25", "-c:v", encoder, "-f", "null", "-"],
                           capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def nvenc_support() -> dict[str, bool]:
    """{"h264": bool, "h265": bool}: True only when ffmpeg reports the encoder and a real test encode succeeds (cached for the life of the process)."""
    listing = _encoder_list()
    return {"h264": "h264_nvenc" in listing and _test_encode("h264_nvenc"), "h265": "hevc_nvenc" in listing and _test_encode("hevc_nvenc")}


def encoders_available() -> dict[str, bool]:
    listing = _encoder_list()
    return {"libx264": "libx264" in listing, "libx265": "libx265" in listing, "aac": " aac " in listing, "libmp3lame": "libmp3lame" in listing}


# ---- cut math ---------------------------------------------------------------------------------------------------------------------------------------------
def keep_from_cuts(cuts: Sequence[Sequence[float]], duration: float) -> list[tuple[float, float]]:
    """The complement of the removed ranges inside [0, duration]: merged, ordered, no slivers. Numbers from the caller are validated (finite, start < end)."""
    if not (isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0):
        raise OptionError("The source duration is unknown.")
    clean = []
    for c in cuts:
        s, e = float(c[0]), float(c[1])
        if not (math.isfinite(s) and math.isfinite(e)) or e <= s:
            raise OptionError("Every cut range needs a start before its end.")
        clean.append((max(0.0, s), min(duration, e)))
    clean = sorted(c for c in clean if c[1] > c[0])
    merged: list[list[float]] = []
    for s, e in clean:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    keeps, pos = [], 0.0
    for s, e in merged:
        if s - pos >= MIN_KEEP_S:
            keeps.append((pos, s))
        pos = e
    if duration - pos >= MIN_KEEP_S:
        keeps.append((pos, duration))
    return keeps


def normalise_keeps(keeps: Sequence[Sequence[float]], duration: float) -> list[tuple[float, float]]:
    out = []
    for k in keeps:
        s, e = float(k[0]), float(k[1])
        if not (math.isfinite(s) and math.isfinite(e)) or e <= s:
            raise OptionError("Every kept span needs a start before its end.")
        out.append((max(0.0, s), min(duration, e)))
    out = sorted(k for k in out if k[1] - k[0] >= MIN_KEEP_S)
    for a, b in zip(out, out[1:]):
        if b[0] < a[1] - 1e-9:
            raise OptionError("Kept spans must not overlap.")
    return out


def remap_time(t: float, keeps: Sequence[tuple[float, float]]) -> Optional[float]:
    """A source time on the cut timeline, or None when it falls in a removed range."""
    acc = 0.0
    for s, e in keeps:
        if s - 1e-9 <= t <= e + 1e-9:
            return acc + (t - s)
        acc += e - s
    return None


def output_duration(keeps: Sequence[tuple[float, float]]) -> float:
    return sum(e - s for s, e in keeps)


# ---- the ffmpeg render ------------------------------------------------------------------------------------------------------------------------------------
def build_render_command(info: VideoInfo, keeps: Sequence[tuple[float, float]], out: Path, script: Path, video_opts: dict, nvenc: dict[str, bool]) -> tuple[list[str], dict]:
    """(ffmpeg arguments, the plan summary). `video_opts` = the effective options (video / resolution / fps / audio). Raises OptionError for bad options."""
    from pipeline.step_08_video_edit.ffmpeg_editor import build_concat_filter
    from pipeline.step_05_cutlist.schemas import TimeSpan

    res = plan_resolution(info.width, info.height, video_opts.get("resolution"))
    fps = plan_fps(video_opts.get("fps"), info.vfr, info.fps)
    vid = plan_video(video_opts.get("video"), nvenc)
    aud = plan_audio(video_opts.get("audio"))
    has_audio = info.has_audio
    graph = build_concat_filter([TimeSpan(s, e) for s, e in keeps], aud.fade_ms / 1000.0, has_audio)
    vchain = [res.vf(info.width, info.height)] + ([fps.filter] if fps.filter else []) + ["format=yuv420p"]
    graph += f";\n[outv]{','.join(vchain)}[ov]"
    if has_audio:
        graph += ";\n[outa]" + ("loudnorm=I=-16:TP=-1.5:LRA=11," if aud.loudnorm else "") + "aresample=async=1[oa]"
    script.write_text(graph, encoding="utf-8")
    args = ["-y", "-hide_banner", "-nostats", "-progress", "pipe:1"]
    for s, e in keeps:
        args += ["-ss", f"{s:.3f}", "-t", f"{e - s:.3f}", "-i", str(info.path)]
    args += ["-filter_complex_script", str(script), "-map", "[ov]"] + (["-map", "[oa]"] if has_audio else [])
    args += list(vid.args) + ["-fps_mode", fps.fps_mode_arg]
    if has_audio:
        args += ["-c:a", aud.codec, "-b:a", f"{aud.bitrate_kbps}k"]
    args += ["-movflags", "+faststart", str(out)]
    summary = {"width": res.width, "height": res.height, "fit": res.fit, "fps_mode": fps.mode, "fps": fps.fps, "encoder": vid.encoder, "hardware": vid.hardware,
               "quality": vid.quality, "audio_bitrate_kbps": aud.bitrate_kbps if has_audio else None, "loudnorm": aud.loudnorm, "fade_ms": aud.fade_ms,
               "notes": list(res.notes) + list(fps.notes)}
    return args, summary


def run_ffmpeg(args: Sequence[str], total_s: float, on_progress: Callable[[float, str], None], cancelled: Callable[[], bool] = lambda: False) -> None:
    """Run ffmpeg, reporting percent from its `-progress` output. A failure raises ExportError with ffmpeg's stderr (last 40 lines) VERBATIM. Both pipes are drained
    in threads so ffmpeg can never block on a full pipe."""
    proc = subprocess.Popen(["ffmpeg", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    err_lines: list[str] = []
    t = threading.Thread(target=lambda: err_lines.extend(proc.stderr), daemon=True)                  # type: ignore[arg-type]
    t.start()
    try:
        for line in proc.stdout:                                                                     # type: ignore[union-attr]
            if cancelled():
                proc.terminate()
                proc.wait(timeout=15)
                raise ExportError("Cancelled")
            line = line.strip()
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                try:
                    secs = float(line.split("=", 1)[1]) / 1e6
                except ValueError:
                    continue
                on_progress(min(99.0, 100.0 * secs / total_s) if total_s > 0 else 0.0, f"{secs:.1f} of {total_s:.1f} s")
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    t.join(timeout=5)
    if rc != 0:
        raise ExportError("".join(err_lines[-40:]).strip() or f"ffmpeg exited with code {rc}")


def render_video(info: VideoInfo, keeps: Sequence[tuple[float, float]], out: Path, video_opts: dict, on_progress: Callable[[float, str], None],
                 cancelled: Callable[[], bool] = lambda: False, nvenc: Optional[dict[str, bool]] = None) -> dict:
    nv = nvenc_support() if (video_opts.get("video") or {}).get("hardware") and nvenc is None else (nvenc or {"h264": False, "h265": False})
    fd, script_name = tempfile.mkstemp(suffix=".txt", prefix="_filter_", dir=out.parent)            # next to the export, inside the project folder
    os.close(fd)
    script = Path(script_name)
    try:
        args, summary = build_render_command(info, keeps, out, script, video_opts, nv)
        run_ffmpeg(args, output_duration(keeps), on_progress, cancelled)
    finally:
        try:
            script.unlink()                                                                           # our own temp file
        except OSError:
            pass
    return summary


def render_audio(info: VideoInfo, keeps: Sequence[tuple[float, float]], out: Path, fmt: str, audio_opts: Optional[dict], on_progress: Callable[[float, str], None],
                 cancelled: Callable[[], bool] = lambda: False) -> dict:
    from pipeline.step_08_video_edit.ffmpeg_editor import build_concat_filter  # noqa: F401  (same cut + fade rule, audio part)
    if not info.has_audio:
        raise OptionError("The source has no audio stream.")
    if fmt not in ("mp3", "wav"):
        raise OptionError("Audio format must be mp3 or wav.")
    aud = plan_audio(audio_opts)
    lines = []
    for i, (s, e) in enumerate(keeps):
        f = min(aud.fade_ms / 1000.0, (e - s) / 2.0)
        fade = f",afade=t=in:st=0:d={f:.4f},afade=t=out:st={(e - s) - f:.4f}:d={f:.4f}" if f > 0 else ""
        lines.append(f"[{i}:a]asetpts=PTS-STARTPTS{fade}[a{i}];")
    lines.append("".join(f"[a{i}]" for i in range(len(keeps))) + f"concat=n={len(keeps)}:v=0:a=1[oa]")
    fd, script_name = tempfile.mkstemp(suffix=".txt", prefix="_filter_", dir=out.parent)
    os.close(fd)
    script = Path(script_name)
    script.write_text("\n".join(lines), encoding="utf-8")
    args = ["-y", "-hide_banner", "-nostats", "-progress", "pipe:1"]
    for s, e in keeps:
        args += ["-ss", f"{s:.3f}", "-t", f"{e - s:.3f}", "-i", str(info.path)]
    args += ["-filter_complex_script", str(script), "-map", "[oa]"]
    args += ["-c:a", "libmp3lame", "-b:a", f"{aud.bitrate_kbps}k"] if fmt == "mp3" else ["-c:a", "pcm_s16le"]
    args.append(str(out))
    try:
        run_ffmpeg(args, output_duration(keeps), on_progress, cancelled)
    finally:
        try:
            script.unlink()
        except OSError:
            pass
    return {"format": fmt, "audio_bitrate_kbps": aud.bitrate_kbps if fmt == "mp3" else None}


def verify_output(path: Path) -> dict:
    """What the written file really is (ffprobe): size, duration, frame rate and whether it is constant or variable frame rate."""
    info = probe_video(path)
    return {"width": info.width, "height": info.height, "fps": round(info.fps, 3) if info.fps else None, "frame_rate_mode": "VFR" if info.vfr else "CFR",
            "duration_s": round(info.duration, 3), "bytes": path.stat().st_size, "has_audio": info.has_audio, "video_codec": info.video_codec}


# ---- text exports ------------------------------------------------------------------------------------------------------------------------------------------
def _ts(t: float, sep: str) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def kept_cues(units: Sequence[dict], words: Sequence[dict], keeps: Sequence[tuple[float, float]]) -> list[tuple[float, float, str]]:
    """One cue per sentence: the words that survive the cut, timed on the CUT timeline. A sentence that is cut completely gives no cue; one that is partly cut keeps only its
    surviving words. Without units the words are grouped by segment."""
    def kept(w: dict) -> Optional[tuple[float, float]]:
        mid = (float(w["start"]) + float(w["end"])) / 2.0
        if remap_time(mid, keeps) is None:
            return None
        s, e = remap_time(float(w["start"]), keeps), remap_time(float(w["end"]), keeps)
        s = s if s is not None else remap_time(mid, keeps)
        e = e if e is not None else remap_time(mid, keeps)
        return (s, max(e, s))                                                    # type: ignore[type-var]

    groups: dict[Any, list[tuple[float, float, str]]] = {}
    order: list[Any] = []
    for w in words:
        k = kept(w)
        if k is None or not str(w.get("word", "")).strip():
            continue
        mid = (float(w["start"]) + float(w["end"])) / 2.0
        key = next((u["unit_id"] for u in units if u.get("source_segment_id") == w.get("segment_id") and u["start"] - 1e-6 <= mid <= u["end"] + 1e-6), ("seg", w.get("segment_id")))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((k[0], k[1], str(w["word"]).strip()))
    cues = []
    for key in order:
        ws = groups[key]
        cues.append((ws[0][0], max(ws[-1][1], ws[0][0] + 0.2), " ".join(w[2] for w in ws)))
    cues.sort()
    for i in range(len(cues) - 1):                                               # cues never overlap
        if cues[i][1] > cues[i + 1][0]:
            cues[i] = (cues[i][0], cues[i + 1][0], cues[i][2])
    return [c for c in cues if c[1] > c[0]]


def to_srt(cues: Sequence[tuple[float, float, str]]) -> str:
    return "".join(f"{i}\n{_ts(s, ',')} --> {_ts(e, ',')}\n{t}\n\n" for i, (s, e, t) in enumerate(cues, 1))


def to_vtt(cues: Sequence[tuple[float, float, str]]) -> str:
    return "WEBVTT\n\n" + "".join(f"{_ts(s, '.')} --> {_ts(e, '.')}\n{t}\n\n" for s, e, t in cues)


def _tc(t: float, fps: int) -> str:
    frames = int(round(t * fps))
    h, r = divmod(frames, 3600 * fps)
    m, r = divmod(r, 60 * fps)
    s, f = divmod(r, fps)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def to_edl(keeps: Sequence[tuple[float, float]], source_name: str, fps: Optional[float], title: str) -> str:
    """CMX 3600 EDL, one video+audio cut event per kept span. Timecode is non-drop-frame at the nearest whole frame rate (29.97 is written as 30 NDF; the header says so)."""
    base = int(round(fps)) if fps and fps > 0 else 25
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME"]
    if fps and abs(fps - base) > 0.01:
        lines.append(f"* NOTE: source is {fps:.3f} fps; timecode is written as {base} fps non-drop (approximate)")
    rec = 0.0
    for i, (s, e) in enumerate(keeps, 1):
        d = e - s
        lines.append(f"{i:03d}  AX       AA/V  C        {_tc(s, base)} {_tc(e, base)} {_tc(rec, base)} {_tc(rec + d, base)}")
        lines.append(f"* FROM CLIP NAME: {source_name}")
        rec += d
    return "\n".join(lines) + "\n"


def cutlist_json(source: str, duration: float, keeps: Sequence[tuple[float, float]], fps: Optional[float]) -> dict:
    cuts, pos = [], 0.0
    for s, e in keeps:
        if s > pos:
            cuts.append({"start": round(pos, 3), "end": round(s, 3)})
        pos = e
    if duration > pos:
        cuts.append({"start": round(pos, 3), "end": round(duration, 3)})
    return {"schema_version": 1, "source": source, "source_duration_s": round(duration, 3), "fps": fps, "output_duration_s": round(output_duration(keeps), 3),
            "keep": [{"start": round(s, 3), "end": round(e, 3)} for s, e in keeps], "cut": cuts}


def transcript_txt(transcript: dict, keeps: Optional[Sequence[tuple[float, float]]]) -> str:
    lines = []
    for seg in transcript.get("segments", []):
        t = remap_time(float(seg["start"]), keeps) if keeps is not None else float(seg["start"])
        if t is None:
            continue
        lines.append(f"[{_ts(t, '.')[:-4]}] {str(seg.get('text', '')).strip()}")
    return "\n".join(lines) + "\n"


def safe_stem(name: str) -> str:
    return re.sub(r"[^\w.\- ]+", "_", Path(name).stem).strip() or "export"
