"""How to hand FFmpeg a filter graph that lives in a file.

`-filter_complex_script <file>` was deprecated in FFmpeg 7.0 and removed in later releases (9.x fails with "Unrecognized option"); the replacement is the
`-/filter_complex <file>` form, which exists from 7.0 on. Older builds only know the old option. The right form is chosen from the installed ffmpeg.
"""
from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path

NEW_OPTION_FROM_MAJOR = 7


def _ffmpeg_text(*args: str) -> str:
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (r.stdout or "") + (r.stderr or "")


@lru_cache(maxsize=1)
def uses_slash_option() -> bool:
    """True when this ffmpeg wants `-/filter_complex <file>` (7.0 and newer); False when it only knows `-filter_complex_script`."""
    out = _ffmpeg_text("-version")
    m = re.search(r"ffmpeg version n?(\d+)\.", out)
    if m:
        return int(m.group(1)) >= NEW_OPTION_FROM_MAJOR
    if out:                                       # a development build with no version number: ask it which option it still has
        return "filter_complex_script" not in _ffmpeg_text("-h", "full")
    return False


def filter_graph_file_args(path: str | Path) -> list[str]:
    """The ffmpeg arguments that read the filter graph from `path`."""
    return ["-/filter_complex", str(path)] if uses_slash_option() else ["-filter_complex_script", str(path)]
