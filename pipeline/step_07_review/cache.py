"""Media fingerprint and on-disk cache (transcript, units, pair scores) so a second run on the same file reproduces the same ids."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

BIG_FILE_BYTES = 1 << 30            # files over 1 GB are fingerprinted from size + first and last 1 MB
EDGE_BYTES = 1 << 20
_CHUNK = 1 << 22


def media_fingerprint(path: Union[str, Path]) -> str:
    """sha256 of the file bytes; for files over 1 GB the sha256 of (size, first 1 MB, last 1 MB)."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    with open(p, "rb") as f:
        if size <= BIG_FILE_BYTES:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                h.update(chunk)
        else:
            h.update(str(size).encode("ascii"))
            h.update(f.read(EDGE_BYTES))
            f.seek(max(size - EDGE_BYTES, 0))
            h.update(f.read(EDGE_BYTES))
    return h.hexdigest()


def open_cache(where: Any, fingerprint: str) -> Any:
    """`where` is a cache folder (-> MediaCache) or an object with get/put (a project's cache), returned as it is."""
    return where if hasattr(where, "get") and hasattr(where, "put") else MediaCache(where, fingerprint)


class MediaCache:
    """<cache_dir>/<fingerprint>/<name>.json. Writes are atomic (temp file + rename)."""

    def __init__(self, cache_dir: Union[str, Path], fingerprint: str):
        self.dir = Path(cache_dir) / fingerprint
        self.fingerprint = fingerprint

    def get(self, name: str) -> Optional[Any]:
        f = self.dir / f"{name}.json"
        if not f.exists():
            return None
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("[Cache] %s unreadable (%s); recomputing", f, exc)
            return None

    def put(self, name: str, obj: Any) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        f = self.dir / f"{name}.json"
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, f)
        return f
