"""Export options: validation, resolution and frame-rate math, built-in presets and the user's saved presets. Pure Python (no ffmpeg), so it is all unit-testable.

Assumptions (Filmcraft's built-in export presets are documented in filmcraft/crates/export/src/presets.rs; this list follows its "Match Source", "Web & Social" and
"Audio Only" groups where the owner's list overlaps, and adds the options the owner asked for):
  * the broadcast formats (ProRes, DNxHR, MXF), image sequences and GIF stay in the editor's own Filmcraft export; they are not offered here;
  * CRF presets: x264 draft 28 / standard 23 / high 18; x265 draft 32 / standard 28 / high 23 (x265's scale runs about 5 higher for the same look);
  * "YouTube 1080p/4K" use CRF (not Filmcraft's fixed 16 / 45 Mbps) so the file size follows the content.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .project_store import atomic_write_bytes

MAX_DIM = 7680
MIN_DIM = 16
FPS_CHOICES = (24, 25, 30, 50, 60)
FPS_MIN, FPS_MAX = 1.0, 240.0
X264_SPEEDS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")
CRF_PRESETS = {"h264": {"draft": 28, "standard": 23, "high": 18}, "h265": {"draft": 32, "standard": 28, "high": 23}}
CRF_RANGE = (0, 51)
BITRATE_RANGE_KBPS = (100, 200_000)
AUDIO_BITRATES = (64, 96, 128, 160, 192, 256, 320, 384, 512)
VIDEO_CODECS = ("h264", "h265")
AUDIO_FORMATS = ("mp3", "wav")
TEXT_FORMATS = {"subtitles": ("srt", "vtt"), "cutlist": ("edl", "json"), "transcript": ("txt", "json")}
KINDS = ("video", "audio", "subtitles", "cutlist", "transcript")
RESOLUTION_PRESETS = {"2160p": 2160, "1440p": 1440, "1080p": 1080, "720p": 720, "480p": 480}
CANVAS_PRESETS = {"vertical_1080x1920": (1080, 1920), "square_1080x1080": (1080, 1080)}
FITS = ("letterbox", "crop", "stretch")


class OptionError(ValueError):
    """A plain-sentence problem with the requested options (the API turns it into a 400)."""


def even(n: float) -> int:
    """Nearest even integer, at least 2."""
    return max(2, int(math.floor(n / 2.0 + 0.5)) * 2)                       # round half up (Python's round() goes to the even neighbour on ties)


# ---- resolution -------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolutionPlan:
    width: int
    height: int
    fit: str                     # how the source maps to the frame: "scale" (same aspect), "letterbox", "crop", "stretch"
    notes: tuple = ()

    def vf(self, src_w: int, src_h: int) -> str:
        """The ffmpeg video filter that produces exactly width x height (always even)."""
        w, h = self.width, self.height
        if self.fit == "letterbox":
            return f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
        if self.fit == "crop":
            return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1"
        return f"scale={w}:{h},setsar=1"


def _check_dim(name: str, v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise OptionError(f"{name} must be a whole number of pixels.")
    if v % 2:
        raise OptionError(f"{name} must be an even number (H.264/H.265 need even sizes); {v} is odd. Try {v - 1} or {v + 1}.")
    if not MIN_DIM <= v <= MAX_DIM:
        raise OptionError(f"{name} must be between {MIN_DIM} and {MAX_DIM} pixels.")
    return v


def plan_resolution(src_w: int, src_h: int, spec: Optional[dict]) -> ResolutionPlan:
    """spec: {"mode": "source"} | {"mode": "preset", "preset": "1080p" | "vertical_1080x1920" | ..., "fit": "crop"|"letterbox"|"stretch"}
    | {"mode": "custom", "width": W, "height": H, "lock_aspect": bool, "fit": ...}.  None = match source."""
    if src_w < 2 or src_h < 2:
        raise OptionError("The source video has no usable frame size.")
    spec = spec or {"mode": "source"}
    mode = spec.get("mode", "source")
    fit = spec.get("fit")
    if fit is not None and fit not in FITS:
        raise OptionError(f"'fit' must be one of {', '.join(FITS)}.")
    notes = []
    if mode == "source":
        w, h = even(src_w), even(src_h)
        if (w, h) != (src_w, src_h):
            notes.append(f"The source size {src_w}x{src_h} is odd; the output is {w}x{h}.")
        return ResolutionPlan(w, h, "scale", tuple(notes))
    if mode == "preset":
        name = spec.get("preset")
        if name in RESOLUTION_PRESETS:                                         # keep the source aspect; the preset is the HEIGHT
            h = RESOLUTION_PRESETS[name]
            w = even(src_w * h / src_h)
            if h > src_h:
                notes.append(f"Upscaling from {src_h}p to {h}p: the file is bigger, not sharper.")
            return ResolutionPlan(w, h, "scale", tuple(notes))
        if name in CANVAS_PRESETS:                                             # a fixed canvas: the source is cropped or letterboxed into it
            w, h = CANVAS_PRESETS[name]
            fit = fit or "crop"
            if fit == "stretch":
                notes.append("Stretching changes the shape of the picture.")
            return ResolutionPlan(w, h, fit, tuple(notes))
        raise OptionError(f"Unknown resolution preset '{name}'. Choose one of: source, {', '.join(RESOLUTION_PRESETS)}, {', '.join(CANVAS_PRESETS)}, custom.")
    if mode == "custom":
        w, h = spec.get("width"), spec.get("height")
        lock = bool(spec.get("lock_aspect"))
        if w is None and h is None:
            raise OptionError("A custom size needs a width and/or a height.")
        if w is not None:
            _check_dim("Width", w)
        if h is not None:
            _check_dim("Height", h)
        if lock:                                                               # keep the source shape; given sizes are a bounding box
            if w is not None and h is not None:
                s = min(w / src_w, h / src_h)
                ow, oh = even(src_w * s), even(src_h * s)
            elif w is not None:
                ow, oh = w, even(src_h * w / src_w)
            else:
                ow, oh = even(src_w * h / src_h), h
            ow, oh = min(ow, MAX_DIM), min(oh, MAX_DIM)
            if max(ow, oh) > max(src_w, src_h):
                notes.append("Upscaling: the file is bigger, not sharper.")
            return ResolutionPlan(ow, oh, "scale", tuple(notes))
        if w is None or h is None:
            raise OptionError("Without the aspect-ratio lock a custom size needs both a width and a height.")
        fit = fit or "letterbox"
        if abs((w / h) - (src_w / src_h)) > 0.01 and fit == "stretch":
            notes.append("Stretching changes the shape of the picture.")
        return ResolutionPlan(w, h, fit, tuple(notes))
    raise OptionError("Resolution mode must be 'source', 'preset' or 'custom'.")


# ---- frame rate -------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FpsPlan:
    mode: str                    # "source" (frames kept as they are, VFR stays VFR) | "fixed"
    fps: Optional[float]
    notes: tuple = ()

    @property
    def filter(self) -> Optional[str]:
        return f"fps={self.fps:g}" if self.mode == "fixed" else None

    @property
    def fps_mode_arg(self) -> str:
        return "passthrough" if self.mode == "source" else "cfr"


def plan_fps(spec: Any, source_is_vfr: bool, source_fps: Optional[float]) -> FpsPlan:
    """spec: None / "source" (match source) | a number (24, 25, 30, 50, 60 or any custom value in range)."""
    if spec in (None, "source"):
        if source_is_vfr:
            return FpsPlan("source", None, ("Variable frame rate is preserved: the output is VFR like the source.",))
        return FpsPlan("source", source_fps, ("Constant frame rate, same as the source.",))
    if isinstance(spec, bool) or not isinstance(spec, (int, float)) or not math.isfinite(spec):
        raise OptionError("Frame rate must be 'source' or a number.")
    if not FPS_MIN <= spec <= FPS_MAX:
        raise OptionError(f"Frame rate must be between {FPS_MIN:g} and {FPS_MAX:g}.")
    notes = ["The output has a constant frame rate (CFR)."]
    if source_is_vfr:
        notes.append("The source is variable frame rate; frames are duplicated or dropped to reach the chosen rate.")
    return FpsPlan("fixed", float(spec), tuple(notes))


# ---- codec and quality ------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class VideoPlan:
    codec: str                   # h264 | h265
    encoder: str                 # libx264 | libx265 | h264_nvenc | hevc_nvenc
    args: tuple                  # ffmpeg output arguments for the video stream
    hardware: bool
    quality: str                 # human summary, e.g. "CRF 18" or "4000 kbps"


def plan_video(spec: Optional[dict], nvenc: dict[str, bool]) -> VideoPlan:
    """spec: {"codec": "h264"|"h265", "quality": {"crf": 18} | {"preset": "draft|standard|high"} | {"bitrate_kbps": 8000}, "speed": "medium", "hardware": bool}"""
    spec = spec or {}
    codec = spec.get("codec", "h264")
    if codec not in VIDEO_CODECS:
        raise OptionError(f"Video codec must be one of {', '.join(VIDEO_CODECS)}.")
    hw = bool(spec.get("hardware"))
    if hw and not nvenc.get(codec):
        raise OptionError("Hardware encoding (NVENC) is not available on this machine for " + codec + ".")
    q = spec.get("quality") or {"preset": "standard"}
    speed = spec.get("speed", "medium")
    if speed not in X264_SPEEDS:
        raise OptionError(f"Encoder speed must be one of {', '.join(X264_SPEEDS)}.")
    if "bitrate_kbps" in q:
        b = q["bitrate_kbps"]
        if isinstance(b, bool) or not isinstance(b, (int, float)) or not BITRATE_RANGE_KBPS[0] <= b <= BITRATE_RANGE_KBPS[1]:
            raise OptionError(f"Bitrate must be between {BITRATE_RANGE_KBPS[0]} and {BITRATE_RANGE_KBPS[1]} kbps.")
        b = int(b)
        rate = ("-b:v", f"{b}k", "-maxrate", f"{int(b * 1.5)}k", "-bufsize", f"{b * 2}k")
        label = f"{b} kbps"
        crf = None
    else:
        if "crf" in q:
            crf = q["crf"]
            if isinstance(crf, bool) or not isinstance(crf, (int, float)) or not CRF_RANGE[0] <= crf <= CRF_RANGE[1]:
                raise OptionError(f"CRF must be between {CRF_RANGE[0]} and {CRF_RANGE[1]}.")
            crf = int(crf)
        else:
            name = q.get("preset", "standard")
            if name not in CRF_PRESETS[codec]:
                raise OptionError("Quality preset must be draft, standard or high.")
            crf = CRF_PRESETS[codec][name]
        rate = None
        label = f"CRF {crf}"
    if hw:
        enc = "h264_nvenc" if codec == "h264" else "hevc_nvenc"
        nv_preset = {"ultrafast": "p1", "superfast": "p1", "veryfast": "p2", "faster": "p3", "fast": "p4", "medium": "p5", "slow": "p6", "slower": "p7", "veryslow": "p7"}[speed]
        args = ["-c:v", enc, "-preset", nv_preset] + (list(rate) if rate else ["-rc", "vbr", "-cq", str(crf), "-b:v", "0"])
    else:
        enc = "libx264" if codec == "h264" else "libx265"
        args = ["-c:v", enc, "-preset", speed] + (list(rate) if rate else ["-crf", str(crf)])
    args += ["-pix_fmt", "yuv420p"]
    if codec == "h265":
        args += ["-tag:v", "hvc1"]                                             # plays in QuickTime/Apple players
    return VideoPlan(codec, enc, tuple(args), hw, label)


@dataclass(frozen=True)
class AudioPlan:
    codec: str
    bitrate_kbps: int
    loudnorm: bool
    fade_ms: float


def plan_audio(spec: Optional[dict], default_fade_ms: float = 15.0) -> AudioPlan:
    spec = spec or {}
    br = spec.get("bitrate_kbps", 192)
    if isinstance(br, bool) or br not in AUDIO_BITRATES:
        raise OptionError(f"Audio bitrate must be one of {', '.join(map(str, AUDIO_BITRATES))} kbps.")
    fade = spec.get("fade_ms", default_fade_ms)
    if isinstance(fade, bool) or not isinstance(fade, (int, float)) or not 0 <= fade <= 100:
        raise OptionError("Crossfade at cuts must be between 0 and 100 ms (0 = off).")
    return AudioPlan("aac", int(br), bool(spec.get("normalize_loudness", False)), float(fade))      # loudness normalisation is OFF unless asked for


# ---- presets ----------------------------------------------------------------------------------------------------------------------------------------------
BUILTIN_PRESETS: dict[str, dict] = {
    "Match Source": {"category": "Match Source", "description": "Source size and frame rate, H.264 CRF 18, AAC 192 kbps",
                     "video": {"codec": "h264", "quality": {"preset": "high"}, "speed": "medium"}, "resolution": {"mode": "source"}, "fps": "source",
                     "audio": {"bitrate_kbps": 192}},
    "YouTube 1080p": {"category": "Web & Social", "description": "1080p, source frame rate, H.264 CRF 18, AAC 192 kbps",
                      "video": {"codec": "h264", "quality": {"preset": "high"}, "speed": "slow"}, "resolution": {"mode": "preset", "preset": "1080p"}, "fps": "source",
                      "audio": {"bitrate_kbps": 192}},
    "YouTube 4K": {"category": "Web & Social", "description": "2160p, source frame rate, H.264 CRF 18, AAC 256 kbps",
                   "video": {"codec": "h264", "quality": {"preset": "high"}, "speed": "medium"}, "resolution": {"mode": "preset", "preset": "2160p"}, "fps": "source",
                   "audio": {"bitrate_kbps": 256}},
    "Shorts / Reels / TikTok": {"category": "Web & Social", "description": "Vertical 1080x1920 (cropped to fill), 30 fps, H.264 CRF 20, AAC 128 kbps",
                                "video": {"codec": "h264", "quality": {"crf": 20}, "speed": "medium"},
                                "resolution": {"mode": "preset", "preset": "vertical_1080x1920", "fit": "crop"}, "fps": 30, "audio": {"bitrate_kbps": 128}},
    "Draft / Preview": {"category": "Draft", "description": "480p, 30 fps, H.264 CRF 28 on the fastest setting, AAC 96 kbps",
                        "video": {"codec": "h264", "quality": {"preset": "draft"}, "speed": "ultrafast"}, "resolution": {"mode": "preset", "preset": "480p"}, "fps": 30,
                        "audio": {"bitrate_kbps": 96}},
}
PRESET_KEYS = ("video", "resolution", "fps", "audio")


def user_config_dir() -> Path:
    """Per-user (not per-project) settings: RETAKE_USER_DIR, else SIMPLECUT_DATA_DIR (the editor's folder), else %APPDATA%\\SimpleCut, else ~/.config/simplecut."""
    for var in ("RETAKE_USER_DIR", "SIMPLECUT_DATA_DIR"):
        if os.environ.get(var):
            return Path(os.environ[var])
    if os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "SimpleCut"
    return Path.home() / ".config" / "simplecut"


def presets_path() -> Path:
    return user_config_dir() / "export_presets.json"


def load_user_presets() -> dict[str, dict]:
    p = presets_path()
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {k: v for k, v in (raw.get("presets") or {}).items() if isinstance(v, dict)}
    except (OSError, ValueError, AttributeError):
        return {}


def save_user_preset(name: str, body: dict) -> dict:
    name = (name or "").strip()
    if not re.fullmatch(r"[\w .&/()+-]{1,60}", name):
        raise OptionError("A preset name is 1 to 60 letters, digits, spaces or . & / ( ) + -.")
    if name.lower() in {k.lower() for k in BUILTIN_PRESETS}:
        raise OptionError(f"'{name}' is a built-in preset name; choose another.")
    preset = {k: body[k] for k in PRESET_KEYS if k in body}
    if not preset:
        raise OptionError("A preset needs at least one of: " + ", ".join(PRESET_KEYS) + ".")
    validate_preset(preset)
    allp = load_user_presets()
    allp[name] = {"category": "My presets", "description": body.get("description", ""), **preset}
    atomic_write_bytes(presets_path(), json.dumps({"schema_version": 1, "presets": allp}, indent=1, ensure_ascii=False).encode("utf-8"))
    return allp[name]


def delete_user_preset(name: str) -> bool:
    allp = load_user_presets()
    if name not in allp:
        return False
    del allp[name]
    atomic_write_bytes(presets_path(), json.dumps({"schema_version": 1, "presets": allp}, indent=1, ensure_ascii=False).encode("utf-8"))
    return True


def validate_preset(preset: dict) -> None:
    """Shape check only (a preset is applied to a real source later): the pieces must be valid on their own."""
    plan_video(preset.get("video"), {"h264": True, "h265": True})
    plan_audio(preset.get("audio"))
    plan_fps(preset.get("fps"), False, None)
    if preset.get("resolution") is not None:
        plan_resolution(1920, 1080, preset["resolution"])


def all_presets() -> dict[str, dict]:
    return {**{k: {**v, "builtin": True} for k, v in BUILTIN_PRESETS.items()}, **{k: {**v, "builtin": False} for k, v in load_user_presets().items()}}


def apply_preset(name: Optional[str], overrides: dict) -> dict:
    """The effective options: the preset's settings, with any key given in `overrides` (video/resolution/fps/audio) replacing the preset's."""
    base: dict = {}
    if name:
        presets = all_presets()
        if name not in presets:
            raise OptionError(f"Unknown export preset '{name}'.")
        base = {k: presets[name][k] for k in PRESET_KEYS if k in presets[name]}
    for k in PRESET_KEYS:
        if k in overrides and overrides[k] is not None:
            base[k] = {**base[k], **overrides[k]} if isinstance(base.get(k), dict) and isinstance(overrides[k], dict) else overrides[k]
    return base
