"""Train the FINAL cross-encoder on all labelled v2 rows (3 seeds, hyperparameters identical to data/training/train_cross_encoder.py) and
save each seed's weights to models/final/ce_seed{0,1,2}/. Seed-averaging rule: mean of sigmoid(logit) over the 3 seeds (prob_mean).
Also writes results_v2/final/ce_logit_shift.json: the distribution of out-of-fold logits vs final-model logits on the training rows
(the final model has seen those rows, so its logits there are over-confident; this is a caveat, not a bug).
No API calls. Needs a GPU for sensible runtime (about 1 minute per seed on an RTX 3070 Ti)."""
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from transformers import get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parents[1] / "data" / "training"
OUT = TRAIN / "results_v2" / "final"
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SEEDS = [0, 1, 2]
LR, BATCH, EPOCHS, WARMUP, WD, MAX_LEN = 2e-5, 16, 3, 0.10, 0.01, 128     # identical to train_cross_encoder.py
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

oof = pd.read_csv(OUT / "final_oof_scores.csv", dtype={"pair_id": str, "video_id": str})
txt = pd.read_csv(TRAIN / "pairs_v2.csv", dtype=str, keep_default_na=False)[["pair_id", "text_a", "text_b", "start_a", "start_b"]]
df = oof[["pair_id", "label", "ce_prob_mean", "ce_logit"]].merge(txt, on="pair_id", validate="1:1")
assert (pd.to_numeric(df.start_a) <= pd.to_numeric(df.start_b)).all()       # earlier unit first (text_a), as in training
assert len(df) == 3562
first, second, y = df.text_a.tolist(), df.text_b.tolist(), df.label.values.astype(np.float32)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def predict(model, tok, firsts, seconds):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(firsts), 64):
            enc = tok(firsts[i:i + 64], seconds[i:i + 64], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)                                             # raw logits


insample_logits = []
for seed in SEEDS:
    t0 = time.time()
    set_seed(seed)
    ce = CrossEncoder(MODEL_NAME, num_labels=1, device=dev)
    model, tok = ce.model.to(dev), ce.tokenizer
    n_pos = int(y.sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - n_pos) / n_pos, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * int(np.ceil(len(y) / BATCH))
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    g = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        model.train()
        order = g.permutation(len(y))
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            enc = tok([first[j] for j in idx], [second[j] for j in idx], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            loss = loss_fn(model(**enc).logits.squeeze(-1), torch.tensor(y[idx], device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
    d = HERE / f"ce_seed{seed}"
    model.save_pretrained(d)
    tok.save_pretrained(d)
    insample_logits.append(predict(model, tok, first, second))
    print(f"seed {seed}: trained on {len(y)} rows and saved to {d.name} in {time.time() - t0:.0f}s")
    del model, ce, opt
    torch.cuda.empty_cache()

# seed-averaging rule: mean of sigmoid(logit) over seeds, then logit(clip) -- same as the out-of-fold ce_prob_mean pipeline
probs = np.mean([1 / (1 + np.exp(-l)) for l in insample_logits], axis=0)
p = np.clip(probs, 1e-6, 1 - 1e-6)
final_logit = np.log(p / (1 - p))
oof_logit = df.ce_logit.values
stat = lambda a: {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}
res = {"n": len(df), "oof_logit": stat(oof_logit), "final_model_logit_on_training_rows": stat(final_logit),
       "by_class": {c: {"oof": stat(oof_logit[df.label.values == c]), "final": stat(final_logit[df.label.values == c])} for c in (0, 1)},
       "fraction_abs_logit_gt_8": {"oof": float((np.abs(oof_logit) > 8).mean()), "final": float((np.abs(final_logit) > 8).mean())},
       "corr_oof_vs_final": float(np.corrcoef(oof_logit, final_logit)[0, 1]),
       "note": "The final model has seen these rows, so its logits here are over-confident; on new pairs it behaves like the out-of-fold models. The stacker was fit on OOF logits."}
(OUT / "ce_logit_shift.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
pd.DataFrame({"pair_id": df.pair_id, "label": df.label, "oof_logit": oof_logit, "final_logit_insample": final_logit}).to_csv(OUT / "ce_logit_shift_rows.csv", index=False)
print(json.dumps({k: res[k] for k in ("oof_logit", "final_model_logit_on_training_rows", "fraction_abs_logit_gt_8", "corr_oof_vs_final")}, indent=1))
