"""
Reference scores from the ORIGINAL final decision model, with Unsloth's own code path (run with the STUDIO venv, read-only on the model folder):

    <studio python> -X utf8 training/standalone_export/reference_scores.py --run 1

The model is loaded as runs/keeper_loop/final_all/scripts/train_decision.py --no-train does: FastDecisionModel.from_pretrained(<final_all/model>, max_seq_length=4096,
load_in_4bit=True), then FastDecisionModel.for_inference. Each parity row is scored the way FastDecisionModel.predict does it (unsloth.models.decision._clef_decide, copied
step by step so the UNROUNDED probability and the raw logits are kept; predict() itself rounds to 4 decimals). The rounded value of the real predict() is also stored for
every 10th row as a consistency check.

Request used for every row (also written to reference_request.json):
    FastDecisionModel.predict(model, tokenizer, state, questions, max_length=4096)      state = the row's JSON string, questions = the row's dict
Writes runs/standalone_export/private/reference_scores_run<N>.jsonl: {"id","unit_ids","p_yes","logits"(true,false per question),"p_yes_api"?,"input_tokens","error"}
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "runs" / "keeper_loop" / "final_all" / "model"
PRIV = REPO / "runs" / "standalone_export" / "private"
MAX_LEN = 16384        # FastDecisionModel.predict reads max(config max_len, CLEF_SERVE_MAX_LEN=16384) tokens
TRAIN_MAX_LEN = 4096

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=int, default=1)
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()

import peft, transformers, unsloth, bitsandbytes  # noqa: E402,F401
from unsloth import FastDecisionModel  # noqa: E402
from unsloth.models.decision import _clef_amp_dtype  # noqa: E402
from unsloth.models._decision_common import QUESTION_TYPES, _clef_question, _parsed, _served_temperatures  # noqa: E402
from unsloth.models.clef import collate_records, encode_record  # noqa: E402

rows = [json.loads(line) for line in (PRIV / "parity_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
if args.limit:
    rows = rows[: args.limit]
t_load = time.time()
model, tokenizer = FastDecisionModel.from_pretrained(str(MODEL_DIR), max_seq_length=TRAIN_MAX_LEN, load_in_4bit=True, use_gradient_checkpointing=False)
FastDecisionModel.for_inference(model)
load_s = time.time() - t_load
torch.cuda.reset_peak_memory_stats()
(PRIV / "reference_request.json").write_text(json.dumps({
    "call": "FastDecisionModel.predict(model, tokenizer, state, questions)  (it parses a JSON-looking state with json.loads, normalises the questions with _clef_question and reads max(4096, 16384) tokens)", "state": "the row's `state` string (JSON-encoded object), unchanged",
    "questions": "the row's `questions` dict: {keep_<id>: {type: noul, instructions: ...}}", "model_dir": str(MODEL_DIR), "max_length_used_by_predict": MAX_LEN, "load_in_4bit": True,
    "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                 "unsloth": unsloth.__version__, "bitsandbytes": bitsandbytes.__version__}}, indent=1), encoding="utf-8")


@torch.no_grad()
def score_row(state, questions):
    state = _parsed(state)                                                              # predict(): a JSON-looking string becomes an object, rendered later as sorted compact JSON
    questions = {str(name): _clef_question(q) for name, q in questions.items()}
    processor = tokenizer if hasattr(tokenizer, "tokenizer") else None
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    encoded = encode_record(tok, {"state": state, "questions": questions, "images": None}, max_length=MAX_LEN, processor=processor)
    device = next(model.parameters()).device
    amp = _clef_amp_dtype(model, device)
    batch = collate_records([encoded], tok.pad_token_id, device)
    model.eval()
    with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
        logits, _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], records=batch["records"], **batch["media"])
    rows_ = [row[: len(q.option_ids)] for q, row in zip(encoded.questions, logits.float().cpu())]
    scales = _served_temperatures(model.decision_config, rows_, [{"qtype": QUESTION_TYPES.index("noul")} for _ in encoded.questions])
    out = []
    for q, row, s in zip(encoded.questions, rows_, scales):
        p = dict(zip(q.option_ids, (row / s).softmax(-1).tolist()))
        out.append((q.question_id, row.tolist(), q.option_ids, p["true"]))
    return out, len(encoded.input_ids)


out_path = PRIV / f"reference_scores_run{args.run}.jsonl"
t0, lat = time.time(), []
with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    for i, r in enumerate(rows):
        rec = {"id": r["id"], "unit_ids": [q[len("keep_"):] for q in r["questions"]], "p_yes": None, "logits": None, "option_ids": None, "input_tokens": None, "error": None}
        t1 = time.time()
        try:
            res, n_tok = score_row(r["state"], r["questions"])
            assert [x[0] for x in res] == list(r["questions"]), "question order changed"
            rec.update(p_yes=[x[3] for x in res], logits=[x[1] for x in res], option_ids=res[0][2], input_tokens=n_tok)
            if i % 10 == 0:
                api = FastDecisionModel.predict(model, tokenizer, r["state"], r["questions"])
                rec["p_yes_api"] = [api[q]["noul"] for q in r["questions"]]
        except Exception as exc:                                         # recorded, never skipped
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        lat.append(time.time() - t1)
        f.write(json.dumps(rec) + "\n")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
import numpy as np  # noqa: E402

print(json.dumps({"run": args.run, "rows": len(rows), "errors": sum(1 for _ in open(out_path, encoding="utf-8") if '"error": "' in _), "load_s": round(load_s, 1),
                  "score_s": round(time.time() - t0, 1), "latency_median_s": round(float(np.median(lat)), 3), "latency_p95_s": round(float(np.percentile(lat, 95)), 3),
                  "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "amp_dtype": str(_clef_amp_dtype(model, next(model.parameters()).device)),
                  "output": str(out_path)}))
