"""Configuration of the local retake service (127.0.0.1 only, no auth)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SERVICE_DIR = Path(__file__).resolve().parent


@dataclass
class ServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    data_dir: str = str(SERVICE_DIR / "data")           # labels.jsonl, labels_skipped.jsonl, cache/
    max_questions: int = 25
    unsure_band_enabled: bool = True
    llm_provider: Optional[str] = None                  # gemini | groq | openai | mock; None = pick from the keys found in the environment
    llm_model: Optional[str] = None
    llm_mock: bool = False                              # RETAKE_LLM_MOCK=1: the mock keeper client (no key, no network)
    max_jobs_kept: int = 100

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        e = os.environ.get
        return cls(host=e("RETAKE_HOST", "127.0.0.1"), port=int(e("RETAKE_PORT", "8765")), data_dir=e("RETAKE_DATA_DIR", str(SERVICE_DIR / "data")),
                   llm_provider=e("RETAKE_LLM_PROVIDER") or None, llm_model=e("RETAKE_LLM_MODEL") or None,
                   llm_mock=e("RETAKE_LLM_MOCK", "").lower() in ("1", "true", "yes"))

    @property
    def labels_path(self) -> str:
        return str(Path(self.data_dir) / "labels.jsonl")

    @property
    def skipped_path(self) -> str:
        return str(Path(self.data_dir) / "labels_skipped.jsonl")

    @property
    def cache_dir(self) -> str:
        return str(Path(self.data_dir) / "cache")


def load_dotenv(path: Path, override: bool = False) -> int:
    """Put KEY=VALUE lines of a .env file into os.environ (values are never printed). Returns the number of variables set."""
    n = 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and (override or k not in os.environ):
                os.environ[k] = v
                n += 1
    return n
