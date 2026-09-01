"""Step 08: Video Editing — Public API."""

from .ffmpeg_editor import (
    concatenate_segments,
    edit_video,
    extract_segment,
)

__all__ = [
    "concatenate_segments",
    "edit_video",
    "extract_segment",
]
