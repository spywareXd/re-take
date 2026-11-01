"""
Item 7 end-to-end: run the full pipeline (cut mode sentences) on a video into a throwaway project folder and prove that nothing outside the project folder changed.

    .venv/Scripts/python training/app_changes/e2e_project_storage.py --video data/videos/sample_b.mp4

Output: runs/app_changes_e2e_<timestamp>/  (a NEW folder; refuses an existing one)
  MyProject/            the throwaway project (MyProject/data/...)
  fs_diff.json          what changed outside the project folder (must be empty)
  summary.json
Snapshot scope: the whole repo (names, sizes, mtimes) except .git, .venv, retake-editor/target, __pycache__ and the new run folder, plus the system temp folder
entries whose name starts with 'retake' or '_temp'. Global caches that are not project data (Hugging Face model cache, user config) are outside the scope on purpose.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
SKIP_DIRS = {".git", ".venv", "target", "__pycache__", ".pytest_cache", "node_modules"}


def snapshot(skip: Path) -> dict:
    snap = {}
    skip = skip.resolve()
    for dirpath, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and (Path(dirpath) / d).resolve() != skip]
        for f in files:
            p = Path(dirpath) / f
            try:
                st = p.stat()
            except OSError:
                continue
            snap[str(p.relative_to(REPO)).replace("\\", "/")] = [st.st_size, st.st_mtime_ns]
    tmp = Path(tempfile.gettempdir())
    for p in tmp.iterdir():
        if p.name.lower().startswith(("retake", "_temp")):
            snap[f"<tmp>/{p.name}"] = [0, 0]
    return snap


def diff(a: dict, b: dict) -> dict:
    return {"added": sorted(set(b) - set(a)), "removed": sorted(set(a) - set(b)), "modified": sorted(k for k in set(a) & set(b) if a[k] != b[k])}


ap = argparse.ArgumentParser()
ap.add_argument("--video", default="data/videos/sample_b.mp4")
ap.add_argument("--keeper", default="finetuned", choices=["finetuned", "keep-last", "gpt-oss"])
args = ap.parse_args()

try:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
except ImportError:
    pass
import logging  # noqa: E402

from pipeline.run_pipeline import run_full_pipeline  # noqa: E402
from service.project_store import ProjectStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
video = (REPO / args.video).resolve()
out = REPO / "runs" / f"app_changes_e2e_{datetime.now():%Y%m%d_%H%M%S}"
if out.exists():
    raise SystemExit(f"{out} exists; not overwriting")
out.mkdir(parents=True)
store = ProjectStore.create(out / "MyProject")
store.set_source(video)

before = snapshot(out)
t0 = time.time()
res = run_full_pipeline(video, output_video=store.exports_dir / "cleaned.mp4", processed_dir=store.data / "pipeline", provider="whisperx", model_name="large-v2",
                        device="cuda", llm_provider="groq", llm_model="openai/gpt-oss-120b", cut_mode="sentences", keeper_provider=args.keeper,
                        write_raw_transcript=False)
after = snapshot(out)
d = diff(before, after)
(out / "fs_diff.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
inside = sorted(str(p.relative_to(store.root)).replace("\\", "/") for p in store.root.rglob("*") if p.is_file())
summary = {"success": res.success, "error": res.error, "total_s": round(time.time() - t0, 1), "outside_project_changes": d, "files_in_project": inside,
           "source_check": store.check_source()}
(out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "files_in_project"}, indent=1))
print("project files:", len(inside), "| folder:", out)
if not res.success or any(d.values()):
    raise SystemExit("FAILED: the pipeline did not succeed, or something outside the project folder changed")
print("PASS: nothing outside the project folder changed")
