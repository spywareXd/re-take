"""Where the service writes its log: the console and a rotating file, so a failure the editor only describes in one line can be read in full."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_NAME = "service.log"
_MARK = "_retake_service_file_handler"


def log_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "logs" / LOG_NAME


def setup_logging(data_dir: str | Path, level: int = logging.INFO) -> Path:
    """Console plus <data_dir>/logs/service.log (2 MB x 5 files). Safe to call twice. Returns the log file path.

    The file gets every logger (the service, the pipeline, uvicorn, warnings) and every traceback."""
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(console)
    path = log_path(data_dir)
    if not any(getattr(h, _MARK, False) for h in root.handlers):
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(path, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8")
        handler.setFormatter(logging.Formatter(FORMAT))
        setattr(handler, _MARK, True)
        root.addHandler(handler)
    logging.captureWarnings(True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):          # uvicorn installs its own handlers; let its records reach the file too
        lg = logging.getLogger(name)
        lg.propagate = True
    return path
