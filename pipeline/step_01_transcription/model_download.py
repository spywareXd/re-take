"""Model cache helpers for the ASR models: is a model already downloaded (no network), and download it with a progress hook (bytes / percent)."""
from __future__ import annotations

from typing import Callable, Optional

ALLOW_PATTERNS = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]       # the files faster-whisper itself asks for


def repo_for(model_name: str) -> Optional[str]:
    """The Hugging Face repo faster-whisper uses for a size name, or None for an unknown name / a local folder."""
    try:
        from faster_whisper.utils import _MODELS
        return _MODELS.get(model_name)
    except Exception:
        return None


def is_downloaded(model_name: str) -> bool:
    """True when the model is already in the local cache. Never touches the network."""
    repo = repo_for(model_name)
    if repo is None:
        return False
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, allow_patterns=ALLOW_PATTERNS, local_files_only=True)
        return True
    except Exception:
        return False


def download(model_name: str, on_progress: Optional[Callable[[float, int, int], None]] = None) -> bool:
    """Download into the normal Hugging Face cache, reporting (percent, bytes_done, bytes_total). Returns False when nothing had to be downloaded."""
    repo = repo_for(model_name)
    if repo is None or is_downloaded(model_name):
        return False
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    class Hook(tqdm):
        def update(self, n=1):
            r = super().update(n)
            if on_progress and self.total and self.unit == "B":
                on_progress(100.0 * self.n / self.total, int(self.n), int(self.total))
            return r

    snapshot_download(repo, allow_patterns=ALLOW_PATTERNS, tqdm_class=Hook)
    return True
