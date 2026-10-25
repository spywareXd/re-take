"""Part B1: neighbor / gap features per pair. Within-video information only; labels are never used.

Inputs (read only): pairs_v1.csv, results/baseline_scores.csv (stage2), processed/<video>/units.json (unit
order, timing, raw text). Embeddings: the pipeline's all-MiniLM-L6-v2 on raw text; raw cosine similarity.
Writes results/neighbor/neighbor_features.csv (pair_id + features + has_<feature> flags) and
results/neighbor/feature_report.md. See preregistration.md for the fixed feature list.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
OUT = HERE / "results" / "neighbor"
OUT.mkdir(parents=True, exist_ok=True)
DUP_WINDOW, DUP_COS = 30, 0.80
CUES = [("sorry",), ("again",), ("let", "me"), ("i", "mean"), ("no", "wait"), ("okay", "so")]   # same cue list as intra_detect.py

from pipeline.step_04_semantic_matching.semantic_matching import get_model  # noqa: E402

# ---------------- load ----------------
pv = pd.read_csv(HERE / "pairs_v1.csv", dtype=str, keep_default_na=False)
pv = pv[pv.is_synthetic == "0"].copy()                      # real rows, including label -1 (the pipeline sees those pairs too)
for c in ["lexical_score", "semantic_score", "containment"]:
    pv[c] = pd.to_numeric(pv[c])
pv["stage2"] = 0.50 * pv.semantic_score + 0.30 * pv.lexical_score + 0.20 * pv.containment
pv["ua"] = pv.pair_id.str.split("__").str[1].str.replace("-", ":", regex=False)
pv["ub"] = pv.pair_id.str.split("__").str[2].str.replace("-", ":", regex=False)
base = pd.read_csv(HERE / "results" / "baseline_scores.csv", dtype={"pair_id": str})
chk = pv.merge(base[["pair_id", "current_stage2"]], on="pair_id")
assert np.allclose(chk.stage2, chk.current_stage2, atol=1e-9), "recomputed stage2 differs from baseline_scores.csv"

model = get_model()
toks = lambda s: re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()
vdata = {}
for v in sorted(pv.video_id.unique()):
    units = json.loads((HERE / "processed" / v / "units.json").read_text())
    emb = model.encode([u["raw_text"] for u in units], batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    vdata[v] = {"units": units, "idx": {u["unit_id"]: i for i, u in enumerate(units)}, "emb": emb,
                "cos": emb @ emb.T, "tok": [toks(u["raw_text"]) for u in units]}


def has_cue(tokens):
    return any(tuple(tokens[i:i + len(c)]) == c for c in CUES for i in range(len(tokens) - len(c) + 1))


# unit-level ranks of pair stage2 (1 = best) among ALL real pairs involving the unit
rank_a, rank_b = {}, {}
long = pd.concat([pv[["pair_id", "video_id", "ua", "stage2"]].rename(columns={"ua": "u"}),
                  pv[["pair_id", "video_id", "ub", "stage2"]].rename(columns={"ub": "u"})])
long["rank"] = long.groupby(["video_id", "u"]).stage2.rank(ascending=False, method="min")
rk = long.set_index(["pair_id", "u"])["rank"]

# ---------------- features ----------------
rows = []
for p in pv.itertuples():
    d = vdata[p.video_id]
    ia, ib = d["idx"][p.ua], d["idx"][p.ub]
    assert ia < ib, f"pair not in chronological order: {p.pair_id}"
    U, T, C = d["units"], d["tok"], d["cos"]
    ta, tb = T[ia], T[ib]
    f = {"pair_id": p.pair_id}
    f["gap_units"] = ib - ia - 1                                  # units strictly between A and B
    f["gap_seconds"] = U[ib]["start"] - U[ia]["end"]
    f["pause_before_b"] = U[ib]["start"] - U[ib - 1]["end"] if ib - 1 >= 0 else np.nan     # silence between previous unit and B
    f["pause_after_a"] = U[ia + 1]["start"] - U[ia]["end"] if ia + 1 < len(U) else np.nan   # silence between A and next unit
    f["a_is_followed_by_b"] = int(ib == ia + 1)
    f["same_open_2"] = int(len(ta) >= 2 and len(tb) >= 2 and ta[:2] == tb[:2])
    f["same_open_3"] = int(len(ta) >= 3 and len(tb) >= 3 and ta[:3] == tb[:3])
    f["len_ratio"] = min(len(ta), len(tb)) / max(len(ta), len(tb), 1)
    f["a_trunc"] = int(not U[ia]["raw_text"].rstrip(" \"')").endswith((".", "!", "?")))
    for name, i in (("a", ia), ("b", ib)):                        # how many other nearby units look like a duplicate of this one
        lo, hi = max(0, i - DUP_WINDOW), min(len(U), i + DUP_WINDOW + 1)
        f[f"dup_degree_{name}"] = int(sum(C[i, j] >= DUP_COS for j in range(lo, hi) if j != i))
    f["rank_a"], f["rank_b"] = rk[(p.pair_id, p.ua)], rk[(p.pair_id, p.ub)]
    f["b_best_for_a"] = int(f["rank_a"] == 1 and f["rank_b"] == 1)
    f["nbr_sim_prev"] = C[ia - 1, ib - 1] if ia - 1 >= 0 else np.nan     # unit before A vs unit before B
    f["nbr_sim_next"] = C[ia + 1, ib + 1] if ib + 1 < len(U) else np.nan  # unit after A vs unit after B
    f["mid_has_restart_cue"] = int(any(has_cue(T[m]) for m in range(ia + 1, ib)))
    rows.append(f)

feat = pd.DataFrame(rows)
NAN_FEATS = ["pause_before_b", "pause_after_a", "nbr_sim_prev", "nbr_sim_next"]
for c in NAN_FEATS:
    feat["has_" + c] = feat[c].notna().astype(int)
feat = feat[feat.pair_id.isin(base.pair_id)]                    # the 3570 rows used everywhere (label != -1)
assert len(feat) == 3570
feat.to_csv(OUT / "neighbor_features.csv", index=False)

# ---------------- report ----------------
num = feat.drop(columns=["pair_id"] + [f"has_{c}" for c in NAN_FEATS])
desc = num.describe().T[["mean", "min", "50%", "max"]]
md = ["# Neighbor features (B1)", "",
      f"{len(feat)} pairs (real, label != -1); labels were not used. Stage 2 for ranks recomputed from pairs_v1.csv and verified equal to "
      "results/baseline_scores.csv. Rank is computed over all real pairs of the video including label -1 pairs (the pipeline sees them). "
      "unit order = order in units.json (checked chronological for every pair).", "",
      "| feature | mean | min | median | max | NaN count |", "|---|---|---|---|---|---|"]
md += [f"| {c} | {r['mean']:.3f} | {r['min']:.3f} | {r['50%']:.3f} | {r['max']:.3f} | {int(feat[c].isna().sum())} |" for c, r in desc.iterrows()]
md += ["", "Features that could not be computed are NaN with a has_<feature> flag: "
       + ", ".join(f"{c} ({int(feat[c].isna().sum())} NaN)" for c in NAN_FEATS) + ".",
       f"Negative gap_seconds (overlapping units): {int((feat.gap_seconds < 0).sum())}."]
(OUT / "feature_report.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
