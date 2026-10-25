"""Low-cost visual motion measurement used to protect silent on-screen action (e.g. muted gameplay) from being cut as dead air."""
from __future__ import annotations

import subprocess

import numpy as np

W, H = 64, 36


def motion_series(video_path, start: float, end: float, fps: float = 10.0) -> np.ndarray:
    """
    Mean absolute difference (0-255) between consecutive downscaled (64x36) grayscale frames sampled at `fps` over [start, end].
    Entry i is the change between sample i and i+1, at time start + (i+1)/fps. Empty array if the window has under two frames.
    """
    if end - start < 2.0 / fps:
        return np.zeros(0)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(start, 0.0):.3f}", "-t", f"{end - start:.3f}", "-i", str(video_path), "-an",
         "-vf", f"fps={fps},scale={W}:{H}:flags=area,format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout
    n = len(raw) // (W * H)
    if n < 2:
        return np.zeros(0)
    frames = np.frombuffer(raw[: n * W * H], dtype=np.uint8).reshape(n, W * H).astype(np.float32)
    return np.abs(np.diff(frames, axis=0)).mean(axis=1)
