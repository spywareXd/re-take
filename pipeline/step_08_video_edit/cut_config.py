"""Configuration for the cut modes (how resolved discards become keep spans)."""
from __future__ import annotations

from dataclasses import dataclass

from pipeline.step_02_preprocessing.config import PAUSE_SPLIT_THRESHOLD_SECONDS

CUT_MODES = ("sentences", "anchored", "retake_only")
DEFAULT_CUT_MODE = "anchored"

# Mean absolute difference (0-255) between consecutive 64x36 grayscale frames sampled at 10 fps. Both test videos (sample_c, sample_b) are static: their max is 0.17;
# the default sits just above that. Real camera or gameplay footage needs a higher value (compression noise alone is typically 1-3).
DEFAULT_MOTION_THRESHOLD = 0.25


@dataclass
class CutConfig:
    cut_mode: str = DEFAULT_CUT_MODE
    head_pad_s: float = 0.20          # kept before the first kept word after a cut / at the file start
    tail_pad_s: float = 0.20          # kept after the last kept word before a cut / at the file end
    max_kept_pause_s: float = 0.50    # sentences: a gap between kept words longer than this is shortened to it
    # anchored: how far a removal may extend into dead air per side. Single source of truth = the gap gate used to build ComparableUnits.
    max_absorb_s: float = PAUSE_SPLIT_THRESHOLD_SECONDS
    dead_air_rel_db: float = 13.0     # a frame is dead air when this many dB below the neighbouring kept speech level
    blip_max_s: float = 0.40          # a run of sound shorter than this ...
    blip_rel_db: float = 8.0          # ... peaking at least this many dB below speech counts as non-speech
    use_visual_motion: bool = True
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD
    motion_fps: float = 10.0
    snap_ms: float = 50.0             # each cut snaps to the quietest frame within +/- this (never inside a word)
    fade_ms: float = 15.0             # audio fade-out / fade-in at every join
    trim_long_silence: bool = False   # also shorten silences between kept words that are not next to a removal
    trim_long_silence_s: float = 1.5  # ... when longer than this (shortened to max_kept_pause_s)
    min_fragment_s: float = 0.10
    lost_gameplay_min_s: float = 2.0  # removed range longer than this, with no ASR words and high motion, is flagged in the CSV

    def __post_init__(self) -> None:
        if self.cut_mode not in CUT_MODES:
            raise ValueError(f"cut_mode must be one of {CUT_MODES}, got {self.cut_mode!r}")
        assert self.tail_pad_s + self.head_pad_s < self.max_absorb_s, (
            f"tail_pad_s + head_pad_s ({self.tail_pad_s + self.head_pad_s:.2f}) must be < max_absorb_s ({self.max_absorb_s:.2f})"
        )
