"""Step 08: Video Editing — Public API."""

from models.retake import CutList, RetakeGroup, TimeSpan

from .cut_generator import (
    DEFAULT_CUTLIST_PATH,
    compute_keep_spans,
    generate_cuts_from_adjudicated_spans,
    load_cutlist,
    merge_spans,
    save_cutlist,
)
from .boundary_refiner import CutRefineConfig, load_envelope, refine_cutlist, words_from_transcript
from .cut_config import CUT_MODES, DEFAULT_CUT_MODE, CutConfig
from .cut_modes import build_cutlist, write_keep_spans_vs_units
from .intra_decisions import apply_intra_decisions
from .ffmpeg_editor import (
    DEFAULT_OUTPUT_DIR,
    concatenate_segments,
    edit_video,
    build_concat_filter,
    extract_segment,
    probe_media,
)

__all__ = [
    "CutList",
    "RetakeGroup",
    "TimeSpan",
    "DEFAULT_CUTLIST_PATH",
    "apply_intra_decisions",
    "DEFAULT_OUTPUT_DIR",
    "compute_keep_spans",
    "generate_cuts_from_adjudicated_spans",
    "load_cutlist",
    "merge_spans",
    "save_cutlist",
    "concatenate_segments",
    "edit_video",
    "extract_segment",
    "build_concat_filter",
    "probe_media",
    "CutRefineConfig",
    "load_envelope",
    "refine_cutlist",
    "words_from_transcript",
    "CUT_MODES",
    "DEFAULT_CUT_MODE",
    "CutConfig",
    "build_cutlist",
    "write_keep_spans_vs_units",
]
