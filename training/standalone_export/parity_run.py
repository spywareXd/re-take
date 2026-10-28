"""
Score the fixed parity set with the STANDALONE loader (run with the repo venv; one mode per run):

    .venv/Scripts/python training/standalone_export/parity_run.py --mode 4bit [--limit N]

Writes runs/standalone_export/private/standalone_scores_<mode>.jsonl ({"id","p_yes","logits","input_tokens","error"}) and a timing summary
runs/standalone_export/private/standalone_timing_<mode>.json. Read-only on the model folder.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from pipeline.step_06_llm_adjudication.decision_standalone import StandaloneDecisionModel  # noqa: E402

PRIV = REPO / "runs" / "standalone_export" / "private"

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["4bit", "fp16", "4bit_requant"], required=True)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--gpu-memory-gb", type=float, default=None)
ap.add_argument("--tag", default="", help="suffix for the output names (smoke tests)")
ap.add_argument("--ids", default="", help="comma-separated row ids only")
args = ap.parse_args()

rows = [json.loads(line) for line in (PRIV / "parity_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
if args.ids:
    rows = [r for r in rows if r["id"] in args.ids.split(",")]
if args.limit:
    rows = rows[: args.limit]
out_path = PRIV / f"standalone_scores_{args.mode}{args.tag}.jsonl"
if out_path.exists():
    raise SystemExit(f"{out_path} exists; not overwriting")
model = StandaloneDecisionModel(mode=args.mode, gpu_memory_gb=args.gpu_memory_gb)
torch.cuda.reset_peak_memory_stats()
lat, errors = [], 0
t0 = time.time()
with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    for i, r in enumerate(rows):
        rec = {"id": r["id"], "p_yes": None, "logits": None, "input_tokens": None, "error": None}
        t1 = time.time()
        try:
            logits, n_tok = model.score_logits(r["state"], r["questions"])
            ids = list(r["questions"])
            rec.update(logits=[logits[q] for q in ids], input_tokens=n_tok)
            rec["p_yes"] = [float(np.exp(logits[q][0] - max(logits[q])) / sum(np.exp(np.array(logits[q]) - max(logits[q])))) for q in ids]
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            errors += 1
        lat.append(time.time() - t1)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
timing = {"mode": args.mode, "rows": len(rows), "errors": errors, "load_s": round(model.load_seconds or 0, 1), "score_s": round(time.time() - t0, 1),
          "latency_median_s": round(float(np.median(lat[1:] or lat)), 3), "latency_p95_s": round(float(np.percentile(lat, 95)), 3),
          "first_call_s": round(lat[0], 2), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
          "versions": {"torch": torch.__version__, "python": sys.version.split()[0]}}
(PRIV / f"standalone_timing_{args.mode}{args.tag}.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
print(json.dumps(timing))
model.unload()
