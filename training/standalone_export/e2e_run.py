"""
End-to-end run of one video with a chosen keeper provider into a NEW run folder (never overwrites; refuses an existing folder).

    .venv/Scripts/python training/standalone_export/e2e_run.py --video data/videos/sample_b.mp4 --provider finetuned --cut-mode sentences
    .venv/Scripts/python training/standalone_export/e2e_run.py --provider gpt-oss --reuse-transcript-from runs/e2e_sample_b_finetuned_<ts>

Folder: runs/e2e_<video>_<provider>_<timestamp>/  (processed data + the cleaned video + e2e_summary.json: stage timings, peak VRAM per stage, flagged groups).
"""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ap = argparse.ArgumentParser()
ap.add_argument("--video", default="data/videos/sample_b.mp4")
ap.add_argument("--provider", choices=["finetuned", "gpt-oss", "keep-last"], default="finetuned")
ap.add_argument("--cut-mode", default="sentences")
ap.add_argument("--reuse-transcript-from", default=None, help="an earlier e2e folder whose transcript.json is copied into the new folder (same unit ids)")
args = ap.parse_args()

try:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")                                       # the Groq key for gpt-oss; never printed
except ImportError:
    pass
import logging  # noqa: E402

import torch  # noqa: E402

from pipeline.run_pipeline import run_full_pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
video = (REPO / args.video).resolve()
out = REPO / "runs" / f"e2e_{video.stem}_{args.provider.replace('-', '')}_{datetime.now():%Y%m%d_%H%M%S}"
if out.exists():
    raise SystemExit(f"{out} exists; not overwriting")
out.mkdir(parents=True)
if args.reuse_transcript_from:
    (out / "processed").mkdir()
    shutil.copyfile(Path(args.reuse_transcript_from) / "processed" / "transcript.json", out / "processed" / "transcript.json")

stages: dict[str, dict] = {}
cur = {"name": None, "t": time.time()}
torch.cuda.reset_peak_memory_stats()


def on_progress(stage, frac):
    if frac == 0.0 and cur["name"] != stage:
        if cur["name"]:
            stages[cur["name"]]["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        cur["name"] = stage
        stages[stage] = {"started_s": round(time.time() - cur["t"], 1)}
        torch.cuda.reset_peak_memory_stats()


t0 = time.time()
res = run_full_pipeline(video, output_video=out / "cleaned.mp4", processed_dir=out / "processed", provider="whisperx", model_name="large-v2", device="cuda",
                        llm_provider="groq", llm_model="openai/gpt-oss-120b", cut_mode=args.cut_mode, keeper_provider=args.provider,
                        reuse_existing_transcript=bool(args.reuse_transcript_from), progress_callback=on_progress)
if cur["name"]:
    stages[cur["name"]]["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
adj_path = out / "processed" / "candidates" / "adjudicated_spans.json"
flagged, summary_groups = [], []
if adj_path.is_file():
    data = json.loads(adj_path.read_text(encoding="utf-8"))
    spans = data.get("spans", data) if isinstance(data, dict) else data
    for s in spans:
        summary_groups.append({"span_id": s.get("span_id"), "unit_ids": s.get("unit_ids"), "keep": s.get("keeper_unit_ids") or ([s["keeper_unit_id"]] if s.get("keeper_unit_id") else []),
                               "status": s.get("keeper_status") or s.get("adjudication_status"), "decided_by": s.get("decided_by")})
        if s.get("keeper_status") == "unclear" or s.get("unclear"):
            flagged.append({"span_id": s.get("span_id"), "unit_ids": s.get("unit_ids"), "kept": s.get("keeper_unit_ids"), "decided_by": s.get("decided_by"),
                            "unit_probs": s.get("unit_probs")})
summary = {"provider": args.provider, "cut_mode": args.cut_mode, "success": res.success, "error": res.error, "total_s": round(time.time() - t0, 1),
           "stage_timings_s": {k: round(v, 1) for k, v in res.stage_timings.items()}, "stage_peak_vram_gb": stages,
           "peak_vram_reserved_gb_overall": round(torch.cuda.max_memory_reserved() / 1e9, 2), "source_duration_s": res.source_duration, "output_duration_s": res.output_duration,
           "retake_groups": res.retake_groups_count, "groups": summary_groups, "flagged_unclear": flagged}
(out / "e2e_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "groups"}, indent=1, default=str))
print("folder:", out)
