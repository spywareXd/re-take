"""Repair stale captions in the results_v2 reports whose scripts FIT models (logreg, gbm, cross-encoder, combine), without
retraining: only caption text and the run header change; every table is left byte-for-byte as in the original v2 run.
Counts are computed from pairs_v2.csv / results_v2 score files. Idempotent (skips a file that already has the run header)."""
from datetime import date
from pathlib import Path

import pandas as pd

from report_header import labels_header

HERE = Path(__file__).resolve().parent
R2 = HERE / "results_v2"

# counts computed from the data, not typed in
d = pd.read_csv(HERE / "pairs_v2.csv", dtype=str, keep_default_na=False)
real = d[(d.is_synthetic == "0") & (d.label != "-1")]
n_rows, n_pos = len(real), int((real.label == "1").sum())
b = pd.read_csv(R2 / "baseline_scores.csv")
fn = int(((b.label == 1) & (b.current_stage2 < 0.65)).sum())
fp = int(((b.label == 0) & (b.current_stage2 >= 0.65)).sum())
assert (len(b), int(b.label.sum())) == (n_rows, n_pos), "baseline_scores.csv does not match pairs_v2.csv"
note = f" Report text repaired {date.today().isoformat()} (stale v1 captions); tables are unchanged from the original v2 run, nothing was retrained."
header = labels_header(HERE / "pairs_v2.csv", HERE / "folds.json", n_rows, n_pos, note)

FIXES = {   # file -> [(old caption, new caption)]
    "combine.md": [("(pooled; Stage 2 has 49 FN and 30 FP)", f"(pooled; Stage 2 has {fn} FN and {fp} FP)")],
    "gbm.md": [("(see results/stage2_errors.csv;", "(see results_v2/stage2_errors.csv;")],
    "ce.md": [("- Lists with text: results/ce_vs_stage2_errors.csv.", "- Lists with text: results_v2/ce_vs_stage2_errors.csv.")],
    "logreg.md": [],
}
for name, subs in FIXES.items():
    p = R2 / name
    txt = p.read_text(encoding="utf-8")
    if "**Run header:**" in txt:
        print(name, "already repaired")
        continue
    for old, new in subs:
        assert txt.count(old) == 1, f"{name}: caption not found once: {old}"
        txt = txt.replace(old, new)
    lines = txt.split("\n")
    lines[1:1] = ["", header]                                   # header right after the title line
    p.write_text("\n".join(lines), encoding="utf-8")
    print(name, "repaired:", len(subs), "caption(s) + header")
