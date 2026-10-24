"""Fine-tune a cross-encoder with speaker-grouped nested folds; compare with current_stage2.

Real rows only (label != -1), no synthetic data, folds from folds.json. For each outer fold k
and seed in 0,1,2: train on the other folds' rows for a fixed 3 epochs and predict fold k.
Nothing is selected or early-stopped on the held-out fold.
Inputs are never modified. Writes results/ce_scores.csv, results/ce.md and
results/ce_vs_stage2_errors.csv.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from transformers import get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SEEDS = [0, 1, 2]
LR, BATCH, EPOCHS, WARMUP, WD, MAX_LEN = 2e-5, 16, 3, 0.10, 0.01, 128
N_BOOT, BOOT_SEED, STAGE2_T = 1000, 0, 0.65
BANDS = [(0.35, 0.80), (0.25, 0.85), (0.45, 0.75)]
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

# ---------------- load ----------------
df = pd.read_csv(HERE / "pairs_v1.csv", dtype=str, keep_default_na=False)
folds = json.loads((HERE / "folds.json").read_text())["folds"]
df = df[(df.is_synthetic == "0") & (df.label != "-1")].copy()
df["label"] = df.label.astype(int)
df["fold"] = df.video_id.map(folds).astype(int)
for c in ["start_a", "start_b"]:
    df[c] = pd.to_numeric(df[c], errors="raise")
base = pd.read_csv(OUT / "baseline_scores.csv", dtype={"pair_id": str, "video_id": str})
df = df.merge(base[["pair_id", "current_stage2"]], on="pair_id").reset_index(drop=True)
assert len(df) == len(base)

# earlier unit first (raw text, not normalized). The pairs are already ordered; swap if not.
swap = df.start_a > df.start_b
df["first"] = np.where(swap, df.text_b, df.text_a)
df["second"] = np.where(swap, df.text_a, df.text_b)
print(f"rows {len(df)}, positives {df.label.sum()}, swapped for ordering: {int(swap.sum())}")


# ---------------- training ----------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def predict(model, tok, firsts, seconds):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(firsts), 64):
            enc = tok(firsts[i:i + 64], seconds[i:i + 64], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(dev)
            probs.append(torch.sigmoid(model(**enc).logits.squeeze(-1)).cpu().numpy())
    return np.concatenate(probs)


def train_and_predict(train, test, seed):
    set_seed(seed)
    ce = CrossEncoder(MODEL_NAME, num_labels=1, device=dev)
    model, tok = ce.model.to(dev), ce.tokenizer
    n_pos = int(train.label.sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train) - n_pos) / n_pos, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * int(np.ceil(len(train) / BATCH))
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    firsts, seconds, y = train["first"].tolist(), train["second"].tolist(), train.label.values.astype(np.float32)
    g = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        model.train()
        order = g.permutation(len(train))
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            enc = tok([firsts[j] for j in idx], [seconds[j] for j in idx], truncation=True,
                      max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            loss = loss_fn(model(**enc).logits.squeeze(-1), torch.tensor(y[idx], device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
    out = predict(model, tok, test["first"].tolist(), test["second"].tolist())
    del model, ce
    torch.cuda.empty_cache()
    return out


rows = []
for seed in SEEDS:
    for k in sorted(df.fold.unique()):
        t0 = time.time()
        te = df.fold == k
        p = train_and_predict(df[~te], df[te], seed)
        part = df.loc[te, ["pair_id", "video_id", "fold", "label"]].copy()
        part["seed"], part["prob"] = seed, p
        rows.append(part)
        print(f"seed {seed} fold {k}: {time.time() - t0:.0f}s, fold PR-AUC "
              f"{average_precision_score(part.label, p):.3f}")
sc = pd.concat(rows, ignore_index=True)
sc["prob_mean"] = sc.groupby("pair_id").prob.transform("mean")
sc.to_csv(OUT / "ce_scores.csv", index=False)

# seed-averaged probability per row, aligned with df
df["ce"] = df.pair_id.map(sc.drop_duplicates("pair_id").set_index("pair_id").prob_mean)
assert df.ce.notna().all()


# ---------------- metrics ----------------
def recall_at_precision(y, s, p):
    prec, rec, _ = precision_recall_curve(y, s)
    ok = prec >= p
    return float(rec[ok].max()) if ok.any() else 0.0


def rank_metrics(y, s):
    return {"PR-AUC": average_precision_score(y, s), "ROC-AUC": roc_auc_score(y, s),
            "R@P90": recall_at_precision(y, s, 0.90), "R@P80": recall_at_precision(y, s, 0.80)}


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / (y == 1).sum()
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


scopes = [("pooled", df)] + [(f"fold{k}", df[df.fold == k]) for k in sorted(df.fold.unique())]
res = {(n, s): rank_metrics(g.label, g[col]) for s, g in scopes
       for n, col in [("current_stage2", "current_stage2"), ("cross_encoder", "ce")]}

# seed variability of pooled PR-AUC (each seed's out-of-fold predictions over all folds)
seed_pr = [average_precision_score(*(lambda g: (g.label, g.prob))(sc[sc.seed == s])) for s in SEEDS]

# paired video-level bootstrap
rng = np.random.default_rng(BOOT_SEED)
vids = df.video_id.unique()
by_vid = {v: g for v, g in df.groupby("video_id")}
bs = {"current_stage2": [], "ce": []}
for _ in range(N_BOOT):
    samp = pd.concat([by_vid[v] for v in rng.choice(vids, len(vids), replace=True)])
    for col in bs:
        bs[col].append(average_precision_score(samp.label, samp[col]) if samp.label.sum() else np.nan)
bs = {k: np.array(v) for k, v in bs.items()}
ci = {k: np.nanpercentile(v, [2.5, 97.5]) for k, v in bs.items()}
diff = bs["ce"] - bs["current_stage2"]
d_lo, d_hi = np.nanpercentile(diff, [2.5, 97.5])
d_pt = res[("cross_encoder", "pooled")]["PR-AUC"] - res[("current_stage2", "pooled")]["PR-AUC"]

# ---------------- cascade ----------------
y = df.label.values
s2 = df.current_stage2.values
casc = []
p, r, f = prf(y, (s2 >= STAGE2_T).astype(int))
casc.append(("stage2 alone @0.65", p, r, f, 0.0))
p, r, f = prf(y, (df.ce.values >= 0.5).astype(int))
casc.append(("cross-encoder alone @0.5", p, r, f, 1.0))
for lo, hi in BANDS:
    sent = (s2 >= lo) & (s2 <= hi)                     # ambiguous band goes to the cross-encoder
    pred = np.where(sent, df.ce.values >= 0.5, s2 > hi).astype(int)   # outside band: Stage 2 decision
    p, r, f = prf(y, pred)
    casc.append((f"cascade band ({lo}, {hi})", p, r, f, sent.mean()))

# ---------------- error comparison vs Stage 2 ----------------
s2_pred = (s2 >= STAGE2_T).astype(int)
s2_recall = prf(y, s2_pred)[1]
# highest cross-encoder threshold whose pooled recall is >= Stage 2's recall at 0.65
cand = np.sort(np.unique(df.ce.values))[::-1]
ce_thr = next(t for t in cand if prf(y, (df.ce.values >= t).astype(int))[1] >= s2_recall)
ce_pred = (df.ce.values >= ce_thr).astype(int)
ce_p, ce_r, ce_f = prf(y, ce_pred)
df["s2_pred"], df["ce_pred"] = s2_pred, ce_pred
status = np.select(
    [(y == 1) & (s2_pred == 0) & (ce_pred == 1), (y == 0) & (s2_pred == 1) & (ce_pred == 0),
     (y == 1) & (s2_pred == 1) & (ce_pred == 0), (y == 0) & (s2_pred == 0) & (ce_pred == 1)],
    ["fixed_FN", "fixed_FP", "new_FN", "new_FP"], default="")
df["status"] = status
chg = df[df.status != ""][["status", "pair_id", "video_id", "fold", "label", "current_stage2", "ce",
                           "text_a", "text_b"]]
chg.sort_values(["status", "fold"]).to_csv(OUT / "ce_vs_stage2_errors.csv", index=False)
cnt = chg.status.value_counts().to_dict()
n_fn, n_fp = int(((y == 1) & (s2_pred == 0)).sum()), int(((y == 0) & (s2_pred == 1)).sum())
ce_fn, ce_fp = int(((y == 1) & (ce_pred == 0)).sum()), int(((y == 0) & (ce_pred == 1)).sum())

# ---------------- report ----------------
sn = ["pooled", "fold1", "fold2", "fold3", "fold4"]
md = ["# Cross-encoder results", "",
      f"{MODEL_NAME}, num_labels=1, raw text (earlier unit first), max_length {MAX_LEN}, BCE with pos_weight = "
      f"#neg/#pos of the training folds, lr {LR}, batch {BATCH}, {EPOCHS} epochs fixed, {int(WARMUP * 100)}% warmup, "
      f"weight decay {WD}. Real rows only ({len(df)} rows, {int(df.label.sum())} positives), no synthetic data. "
      "Outer folds from folds.json, seeds 0/1/2, probabilities averaged over seeds. Nothing was selected on a "
      "held-out fold.", ""]
for m in ["PR-AUC", "ROC-AUC", "R@P90", "R@P80"]:
    md += [f"### {m}", "", "| model | " + " | ".join(sn) + " |", "|---|" + "---|" * len(sn)]
    md += [f"| {n} | " + " | ".join(f"{res[(n, s)][m]:.3f}" for s in sn) + " |" for n in ["current_stage2", "cross_encoder"]]
    md.append("")
md += [f"Pooled PR-AUC per seed (0/1/2): {', '.join(f'{v:.3f}' for v in seed_pr)}; mean {np.mean(seed_pr):.3f}, "
       f"std {np.std(seed_pr, ddof=1):.3f}.", "",
       f"Pooled PR-AUC 95% CI (1000 video bootstrap resamples, seed 0): current_stage2 [{ci['current_stage2'][0]:.3f}, "
       f"{ci['current_stage2'][1]:.3f}]; cross_encoder [{ci['ce'][0]:.3f}, {ci['ce'][1]:.3f}].", "",
       f"**Paired PR-AUC difference (cross_encoder minus current_stage2): {d_pt:+.3f}, 95% CI [{d_lo:+.3f}, {d_hi:+.3f}]**", "",
       "### Cascade (outside the band keep the Stage 2 decision at 0.65; inside the band use cross-encoder prob >= 0.5)", "",
       "| setup | precision | recall | F1 | fraction sent to cross-encoder |", "|---|---|---|---|---|"]
md += [f"| {n} | {p:.3f} | {r:.3f} | {f:.3f} | {fr:.1%} |" for n, p, r, f, fr in casc]
md += ["", f"### Error comparison at matched recall (cross-encoder threshold {ce_thr:.3f} gives recall {ce_r:.3f} vs "
           f"Stage 2 {s2_recall:.3f}; precision {ce_p:.3f} vs {prf(y, s2_pred)[0]:.3f})", "",
       f"- Stage 2 @0.65: {n_fn} FN, {n_fp} FP.  Cross-encoder: {ce_fn} FN, {ce_fp} FP.",
       f"- Fixed: {cnt.get('fixed_FN', 0)} of {n_fn} FN, {cnt.get('fixed_FP', 0)} of {n_fp} FP.",
       f"- New errors: {cnt.get('new_FN', 0)} FN, {cnt.get('new_FP', 0)} FP.",
       "- Lists with text: results/ce_vs_stage2_errors.csv.", ""]
verdict = "outside the noise (CI excludes 0)" if (d_lo > 0 or d_hi < 0) else "inside the noise (CI includes 0)"
bestc = max(casc[2:], key=lambda c: c[3])
md += ["## Summary", "",
       f"1. Cross-encoder pooled PR-AUC {res[('cross_encoder', 'pooled')]['PR-AUC']:.3f} (seed std {np.std(seed_pr, ddof=1):.3f}) vs "
       f"Stage 2 {res[('current_stage2', 'pooled')]['PR-AUC']:.3f}: diff {d_pt:+.3f}, 95% CI [{d_lo:+.3f}, {d_hi:+.3f}], {verdict}.",
       f"2. Best cascade by F1: {bestc[0]} (P {bestc[1]:.3f}, R {bestc[2]:.3f}, F1 {bestc[3]:.3f}, {bestc[4]:.0%} sent) vs Stage 2 "
       f"alone F1 {casc[0][3]:.3f}.",
       f"3. At matched recall the cross-encoder fixes {cnt.get('fixed_FN', 0)} FN and {cnt.get('fixed_FP', 0)} FP of Stage 2's "
       f"{n_fn}/{n_fp}, and adds {cnt.get('new_FN', 0)} FN and {cnt.get('new_FP', 0)} FP."]
(OUT / "ce.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
