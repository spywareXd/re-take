"""The WhisperX / faster-whisper model sizes the UI offers, built from what the INSTALLED faster-whisper supports.

Membership comes from the install (faster_whisper.utils._MODELS); a wanted size that the install does not know is simply not listed. Download size, VRAM and speed are
APPROXIMATE (static table, from the published faster-whisper / Whisper benchmarks, not measured here) and are labelled that way; once a model is in the local cache its
real size on disk is reported instead. Nothing in here touches the network, and nothing imports torch (a cold torch import takes tens of seconds).
"""
from __future__ import annotations

import subprocess
from typing import Any, Optional

DEFAULT_MODEL = "large-v2"                                          # what the pipeline uses today (ReviewConfig / run_full_pipeline)
WANTED = ("tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo")
TURBO_NAMES = ("turbo", "large-v3-turbo")

# approximate: download GB (CTranslate2 fp16 files), VRAM GB with int8_float16 and with float16 (WhisperX, batch 16, incl. VAD + alignment), speed relative to large-v2
APPROX = {
    "tiny": (0.08, 1.0, 1.2, 12.0),
    "base": (0.15, 1.2, 1.5, 8.0),
    "small": (0.5, 2.0, 2.8, 4.0),
    "medium": (1.5, 3.5, 5.0, 2.0),
    "large-v2": (3.1, 5.5, 8.5, 1.0),
    "large-v3": (3.1, 5.5, 8.5, 1.0),
    "turbo": (1.6, 3.8, 5.5, 5.0),
}
COMPUTE_TYPES_GPU = ("int8_float16", "float16", "int8")           # tried in this order; today's service default is int8_float16
COMPUTE_TYPES_CPU = ("int8", "float32")


class UnknownModel(ValueError):
    pass


def supported() -> dict[str, str]:
    """{offered name: faster-whisper size name} for the installed version."""
    try:
        from faster_whisper.utils import _MODELS
    except Exception:
        return {}
    out = {}
    for name in WANTED:
        if name == "turbo":
            real = next((n for n in TURBO_NAMES if n in _MODELS), None)
            if real:
                out[name] = real
        elif name in _MODELS:
            out[name] = name
    return out


def gpu_info() -> Optional[dict[str, Any]]:
    """{name, total_gb, free_gb} from nvidia-smi, or None when there is no NVIDIA GPU / the tool is missing."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        name, total, free = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "total_gb": round(float(total) / 1024, 1), "free_gb": round(float(free) / 1024, 1)}
    except Exception:
        return None


def dir_size_gb(model_name: str) -> Optional[float]:
    """Real size of the cached model files (GB), or None when it is not downloaded."""
    from .model_download import ALLOW_PATTERNS, is_downloaded, repo_for
    if not is_downloaded(model_name):
        return None
    try:
        from pathlib import Path

        from huggingface_hub import snapshot_download
        folder = Path(snapshot_download(repo_for(model_name), allow_patterns=ALLOW_PATTERNS, local_files_only=True))
        return round(sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1e9, 2)
    except Exception:
        return None


def describe(offered: str, real: str, gpu: Optional[dict]) -> dict[str, Any]:
    from .model_download import is_downloaded
    dl, vram_i8, vram_fp16, speed = APPROX[offered]
    downloaded = is_downloaded(real)
    out = {"name": offered, "faster_whisper_name": real, "approx": True, "download_gb": dl, "vram_gb": vram_i8, "vram_gb_float16": vram_fp16,
           "relative_speed": speed, "downloaded": downloaded, "default": offered == DEFAULT_MODEL}
    if downloaded:
        out["size_on_disk_gb"] = dir_size_gb(real)
    warn = vram_warning(offered, gpu)
    if warn:
        out["warning"] = warn
    return out


def vram_warning(name: str, gpu: Optional[dict]) -> Optional[str]:
    """A warning (never a block) when the model likely does not fit. Uses the int8_float16 estimate, the lighter of the two."""
    if name not in APPROX:
        return None
    if gpu is None:
        return "No NVIDIA GPU was found: transcription will run on the CPU and be much slower."
    need = APPROX[name][1]
    if need > gpu["total_gb"] * 0.95:
        return f"Needs about {need:g} GB of GPU memory (approximate); this GPU has {gpu['total_gb']:g} GB, so it will probably run out of memory."
    if need > gpu["free_gb"]:
        return f"Needs about {need:g} GB of GPU memory (approximate); only {gpu['free_gb']:g} GB is free right now. Close other GPU programs or expect it to fail."
    return None


def catalog() -> dict[str, Any]:
    gpu = gpu_info()
    sup = supported()
    return {"default": DEFAULT_MODEL if DEFAULT_MODEL in sup else (next(iter(sup), None)), "gpu": gpu,
            "models": [describe(n, real, gpu) for n, real in sup.items()],
            "compute_type": "auto: tries " + ", ".join(COMPUTE_TYPES_GPU if gpu else COMPUTE_TYPES_CPU) + " in order and uses the first the device accepts"}


def resolve(name: Optional[str]) -> str:
    """The faster-whisper name for an offered model name; the default for None. Unknown names raise UnknownModel (the API turns that into a 400)."""
    if name is None:
        name = DEFAULT_MODEL
    sup = supported()
    if name not in sup:
        raise UnknownModel(f"'{name}' is not a model this installation supports. Choose one of: {', '.join(sup) or '(none found)'}.")
    return sup[name]


def compute_candidates(device: str, requested: Optional[str] = None) -> tuple[str, ...]:
    base = COMPUTE_TYPES_GPU if device == "cuda" else COMPUTE_TYPES_CPU
    if requested and requested != "auto":
        return (requested,) + tuple(c for c in base if c != requested)
    return base
