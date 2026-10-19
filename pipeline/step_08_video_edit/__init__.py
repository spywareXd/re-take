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
from .ffmpeg_editor import (
    DEFAULT_OUTPUT_DIR,
    concatenate_segments,
    edit_video,
    extract_segment,
)

__all__ = [
    "CutList",
    "RetakeGroup",
    "TimeSpan",
    "DEFAULT_CUTLIST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "compute_keep_spans",
    "generate_cuts_from_adjudicated_spans",
    "load_cutlist",
    "merge_spans",
    "save_cutlist",
    "concatenate_segments",
    "edit_video",
    "extract_segment",
]
