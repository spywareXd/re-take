"""ProjectStore: the ONE owner of every path inside a project folder.

    X/                      the project folder (X = the project name)
      X.<ext>               the project file (written by the editor, never by this class)
      data/                 everything the backend keeps for the project
        project.json        source video reference (relative path + size/hash fingerprint), schema version
        settings.json       transcription model, silence threshold, ...
        transcript.json, words.json, units.json, pairs__<model>.json, keepers.json, cutlist.json
        labels.jsonl, labels_skipped.jsonl, keeper_decisions.jsonl     (answers and decisions, append-only)
        progress.jsonl      the pipeline event log (item 2)
        audio/              extracted audio / proxy files
        cache/              other cached intermediates
        stale/              numbered copies of results that were replaced (item 3)
        exports/            exports (item 4), unique names, never overwritten

Writes are atomic (temp file in the same folder, then rename). JSON objects carry {"schema_version", "kind", "data"}; a bare legacy object (no envelope) loads as
schema 0. Nothing in here deletes or moves anything: replaced results are copied to stale/ first.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

SCHEMA_VERSION = 1
DATA_DIR = "data"
SUBDIRS = ("audio", "cache", "stale", "exports")
RESULT_NAMES = ("transcript", "words", "units", "keepers", "cutlist")          # plus pairs__<model>
FINGERPRINT_EDGE = 1 << 20
BIG_FILE = 1 << 30


class ProjectError(Exception):
    """A plain-sentence problem with the project folder."""


def fingerprint(path: Union[str, Path]) -> dict:
    """{size, sha256}: sha256 of the bytes; over 1 GB the sha256 of (size, first 1 MB, last 1 MB)."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    with open(p, "rb") as f:
        if size <= BIG_FILE:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        else:
            h.update(str(size).encode("ascii"))
            h.update(f.read(FINGERPRINT_EDGE))
            f.seek(max(size - FINGERPRINT_EDGE, 0))
            h.update(f.read(FINGERPRINT_EDGE))
    return {"size": size, "sha256": h.hexdigest()}


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")      # same folder, so the rename is atomic
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)                                                                   # our own temp file, never a user file
        except OSError:
            pass
        raise


def safe_name(name: str) -> str:
    """A cache/result name that cannot leave the data folder."""
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", name) or name.startswith("."):
        raise ProjectError(f"'{name}' is not a valid name.")
    return name


class ProjectStore:
    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)

    # ---- layout: the only place paths are built ----
    @property
    def name(self) -> str:
        return self.root.name

    @property
    def data(self) -> Path:
        return self.root / DATA_DIR

    def path(self, *parts: str) -> Path:
        p = self.data.joinpath(*parts)
        if self.data.resolve() not in p.resolve().parents and p.resolve() != self.data.resolve():
            raise ProjectError("That path is outside the project data folder.")
        return p

    def json_path(self, name: str) -> Path:
        return self.path(f"{safe_name(name)}.json")

    @property
    def settings_path(self) -> Path:
        return self.json_path("settings")

    @property
    def progress_path(self) -> Path:
        return self.path("progress.jsonl")

    @property
    def labels_path(self) -> Path:
        return self.path("labels.jsonl")

    @property
    def skipped_path(self) -> Path:
        return self.path("labels_skipped.jsonl")

    @property
    def keeper_decisions_path(self) -> Path:
        return self.path("keeper_decisions.jsonl")

    @property
    def audio_dir(self) -> Path:
        return self.path("audio")

    @property
    def cache_dir(self) -> Path:
        return self.path("cache")

    @property
    def stale_dir(self) -> Path:
        return self.path("stale")

    @property
    def exports_dir(self) -> Path:
        return self.path("exports")

    # ---- create / open ----
    @classmethod
    def create(cls, folder: Union[str, Path]) -> "ProjectStore":
        """Make X/ and X/data/ (and its subfolders). An existing folder is reused as it is; nothing in it is touched."""
        s = cls(folder)
        for sub in SUBDIRS:
            s.path(sub).mkdir(parents=True, exist_ok=True)
        return s

    def project_file(self, ext: str) -> Path:
        """Where the editor's project file goes: X/X.<ext>."""
        return self.root / f"{self.name}.{ext.lstrip('.')}"

    # ---- versioned JSON objects ----
    def put(self, name: str, obj: Any) -> Path:
        p = self.json_path(name)
        atomic_write_bytes(p, json.dumps({"schema_version": SCHEMA_VERSION, "kind": name, "data": obj}, ensure_ascii=False).encode("utf-8"))
        return p

    def get(self, name: str, default: Any = None) -> Any:
        """The stored object, or `default` when missing or unreadable. A pre-envelope object loads as it is (schema 0)."""
        p = self.json_path(name)
        if not p.is_file():
            return default
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default
        if isinstance(raw, dict) and set(raw) >= {"schema_version", "data"}:
            if not isinstance(raw["schema_version"], int) or raw["schema_version"] > SCHEMA_VERSION:
                raise ProjectError(f"{p.name} was written by a newer version of the app (schema {raw['schema_version']}).")
            return raw["data"]
        return raw

    def has(self, name: str) -> bool:
        return self.json_path(name).is_file()

    def replace_with_copy(self, name: str, obj: Any) -> Optional[Path]:
        """Store `obj` as `name`; if one already exists, first copy it to stale/<name>.<n>.json (n = 1, 2, ...). Returns the copy, if any."""
        copy = None
        old = self.json_path(name)
        if old.is_file():
            self.stale_dir.mkdir(parents=True, exist_ok=True)
            n = 1
            while (self.stale_dir / f"{name}.{n}.json").exists():
                n += 1
            copy = self.stale_dir / f"{name}.{n}.json"
            shutil.copyfile(old, copy)
        self.put(name, obj)
        return copy

    # ---- settings ----
    def settings(self) -> dict:
        return dict(self.get("settings", {}) or {})

    def update_settings(self, **changes: Any) -> dict:
        s = self.settings()
        s.update(changes)
        self.put("settings", s)
        return s

    # ---- the source video ----
    def set_source(self, video: Union[str, Path]) -> dict:
        """Reference the source by a path relative to the project folder (absolute when on another drive) plus its fingerprint. The video is not copied."""
        v = Path(video).resolve()
        try:
            rel = os.path.relpath(v, self.root.resolve())
        except ValueError:                                                                   # a different drive on Windows
            rel = str(v)
        ref = {"path": rel.replace("\\", "/"), "fingerprint": fingerprint(v)}
        self.put("project", {"source": ref})
        return ref

    def source_path(self) -> Optional[Path]:
        ref = (self.get("project", {}) or {}).get("source")
        if not ref:
            return None
        p = Path(ref["path"])
        return p if p.is_absolute() else (self.root / p).resolve()

    def check_source(self) -> dict:
        """{"status": "ok" | "missing" | "changed" | "none", "message": ...}: call when the project is reopened."""
        ref = (self.get("project", {}) or {}).get("source")
        if not ref:
            return {"status": "none", "message": "No source video is recorded for this project."}
        p = self.source_path()
        if p is None or not p.is_file():
            return {"status": "missing", "message": f"The source video was not found at {ref['path']}."}
        fp = fingerprint(p)
        if fp != ref["fingerprint"]:
            return {"status": "changed", "message": "The source video has changed since this project was saved; the transcript may not match it."}
        return {"status": "ok", "message": "The source video matches."}

    # ---- legacy data (read as a fallback, imported by copying) ----
    def legacy_cache_dir(self, legacy_root: Union[str, Path], media_fingerprint: str) -> Path:
        return Path(legacy_root) / media_fingerprint

    def legacy_available(self, legacy_root: Union[str, Path], media_fingerprint: str) -> dict[str, Path]:
        d = self.legacy_cache_dir(legacy_root, media_fingerprint)
        return {p.stem: p for p in sorted(d.glob("*.json"))} if d.is_dir() else {}

    def import_legacy(self, legacy_root: Union[str, Path], media_fingerprint: str) -> list[str]:
        """COPY the old per-fingerprint cache files into this project (transcript, units, pair scores). Originals stay; existing project files are not replaced."""
        done = []
        for stem, src in self.legacy_available(legacy_root, media_fingerprint).items():
            name = stem if stem in ("transcript", "units") else f"pairs__{stem[len('scores__'):]}" if stem.startswith("scores__") else None
            if name is None or self.has(name):
                continue
            try:
                self.put(name, json.loads(src.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
            done.append(name)
        return done

    # ---- exports ----
    def unique_export_path(self, stem: str, ext: str) -> Path:
        """exports/<stem>.<ext>, or <stem>_2.<ext>, _3, ... : never an existing file. The name is reserved by creating it exclusively."""
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^\w.\- ]+", "_", stem).strip() or "export"
        ext = ext.lstrip(".")
        n = 1
        while True:
            p = self.exports_dir / (f"{stem}.{ext}" if n == 1 else f"{stem}_{n}.{ext}")
            try:
                with open(p, "xb"):
                    return p
            except FileExistsError:
                n += 1


class ProjectCache:
    """The get/put interface of MediaCache (names 'transcript', 'units', 'scores__<model>') over a ProjectStore, so analyze_media writes into X/data/."""

    def __init__(self, store: ProjectStore):
        self.store = store

    @staticmethod
    def _name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", f"pairs__{name[len('scores__'):]}" if name.startswith("scores__") else name)      # model keys contain '+'

    def get(self, name: str) -> Optional[Any]:
        return self.store.get(self._name(name))

    def put(self, name: str, obj: Any) -> Path:
        p = self.store.put(self._name(name), obj)
        if name == "transcript":                                                             # word-level timestamps, kept apart so units can be re-split without the transcript
            words = [dict(w, segment_id=s.get("id", i)) for i, s in enumerate(obj.get("segments", [])) for w in (s.get("words") or [])]
            self.store.put("words", words)
        return p
