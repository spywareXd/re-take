"""
The local decision model as a keeper backend (provider `finetuned`): a thin wrapper around decision_standalone.StandaloneDecisionModel that adds
  - the model location (a folder, optionally downloaded from a Hugging Face repo id once),
  - SHA-256 verification of the files in use against export_manifest.json,
  - freeing other GPU models before it loads, and unload().
No Unsloth, no Studio. `predict(state, questions) -> {"keep_<id>": p_yes}` is the same call the policy used with the Studio backend.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from .decision_standalone import DecisionModelError, DecisionModelIncompatible, DecisionModelMissing, StandaloneDecisionModel
from .keeper_policy import DEFAULT_MODEL_PATH, LOCAL_MODES

logger = logging.getLogger(__name__)

# files each mode reads (relative to the model folder); the manifest lists all of them
MODE_FILES = {
    "4bit": ["head.safetensors", "head_config.json", "decision_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
             "special_tokens_map.json", "chat_template.jinja", "adapter/adapter_model.safetensors", "adapter/adapter_config.json",
             "base_bnb4/model.safetensors", "base_bnb4/config.json", "base_bnb4/generation_config.json"],
}
MODE_PATTERNS = {
    "4bit": ["export_manifest.json", "head*", "decision_config.json", "tokenizer*", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
             "chat_template.jinja", "adapter/*", "base_bnb4/*"],
    "fp16": ["export_manifest.json", "head*", "decision_config.json", "tokenizer*", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json",
             "chat_template.jinja", "config.json", "generation_config.json", "model*.safetensors", "model.safetensors.index.json"],
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def files_for(mode: str, manifest: dict, folder: Path) -> list[str]:
    """Manifest paths the mode reads."""
    listed = manifest.get("files", {})
    if mode == "4bit":
        return [f for f in MODE_FILES["4bit"] if f in listed]
    return [f for f in listed if not f.startswith(("base_bnb4/", "adapter/"))]


def verify_manifest(folder: Path, mode: str) -> list[str]:
    """SHA-256 of the files the mode uses against export_manifest.json. Returns the list of problems (empty = verified)."""
    mf = folder / "export_manifest.json"
    if not mf.is_file():
        return ["export_manifest.json is missing"]
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    problems = []
    for rel in files_for(mode, manifest, folder):
        p = folder / rel
        if not p.is_file():
            problems.append(f"{rel}: missing")
        elif sha256_of(p) != manifest["files"][rel]["sha256"]:
            problems.append(f"{rel}: SHA-256 does not match export_manifest.json")
    return problems


def download_model(repo_id: str, folder: Path, mode: str) -> None:
    """Download the files of `mode` from a Hugging Face repo into an empty or missing folder, then verify them. Never writes into a folder that has files."""
    if folder.exists() and any(folder.iterdir()):
        raise DecisionModelMissing(f"{folder} is not empty; refusing to download into it (empty it or pick another keeper.model_path)")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise DecisionModelIncompatible("huggingface_hub is needed to download the keeper model") from exc
    logger.info("[Keeper] downloading the decision model %s (%s files) into %s", repo_id, mode, folder)
    folder.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(folder), allow_patterns=MODE_PATTERNS[mode])
    problems = verify_manifest(folder, mode)
    if problems:
        raise DecisionModelIncompatible("downloaded model failed verification: " + "; ".join(problems[:5]))


class StandaloneBackend:
    """Lazy local backend. availability() never loads the model; predict() loads it on first use; unload() frees it."""

    def __init__(self, model_path: Any = None, mode: str = "4bit", repo: Optional[str] = None, verify: bool = False, device: str = "cuda",
                 before_load: Optional[Callable[[], None]] = None, model: Any = None):
        if mode not in LOCAL_MODES:
            raise ValueError(f"keeper.mode must be one of {list(LOCAL_MODES)}, got '{mode}'")
        self.path, self.mode, self.repo, self.verify = Path(model_path or DEFAULT_MODEL_PATH), mode, repo, verify
        self.before_load = before_load                                 # called right before the weights load (free the transcription model / GPU cache)
        self._model = model if model is not None else StandaloneDecisionModel(self.path, mode=mode, device=device)
        self._checked = False
        self.model = f"standalone-{mode}"                                # the id written to keeper_model / adjudicated_spans

    @property
    def loaded(self) -> bool:
        return bool(getattr(self._model, "loaded", False))

    def info(self) -> dict:
        return {"backend": "standalone", "mode": self.mode, "model_path": str(self.path), "loaded": self.loaded, "download_repo": self.repo, "verify": self.verify}

    def availability(self) -> tuple[bool, str]:
        ok, why = self._model.availability()
        if ok:
            return True, f"local decision model, mode {self.mode}, {self.path}"
        if self.repo and not (self.path.exists() and any(self.path.iterdir())):
            return True, f"the decision model will be downloaded from {self.repo} on first use"
        return False, why

    def _prepare(self) -> None:
        if self._checked:
            return
        folder_empty = not self.path.exists() or not any(self.path.iterdir())
        if folder_empty and self.repo:
            download_model(self.repo, self.path, self.mode)
        elif self.verify:
            problems = verify_manifest(self.path, self.mode)
            if problems:
                raise DecisionModelIncompatible("model files do not match export_manifest.json: " + "; ".join(problems[:5]))
        self._checked = True

    def predict(self, state: Any, questions: dict) -> dict[str, float]:
        if not self.loaded:
            self._prepare()
            if self.before_load:
                self.before_load()
        return self._model.score(state, questions)

    def unload(self) -> None:
        self._model.unload()


__all__ = ["StandaloneBackend", "verify_manifest", "download_model", "sha256_of", "DecisionModelError", "DecisionModelMissing", "DecisionModelIncompatible"]
