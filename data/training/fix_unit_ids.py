"""Repair unit_a_id / unit_b_id in pairs_v1.csv.

A spreadsheet auto-formatted ids like "49:0" into times ("49:00:00", "30:01:00").
pair_id ("<video>__<unit_a>__<unit_b>", ":" written as "-") is intact, so rebuild
the ids from it. Every other column is written back byte-for-byte as text.
A timestamped backup goes to backups/ first.
"""
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
src = HERE / "pairs_v1.csv"

df = pd.read_csv(src, dtype=str, keep_default_na=False)  # all text, no NaN coercion
parts = df.pair_id.str.split("__", expand=True)
new_a = parts[1].str.replace("-", ":", regex=False)
new_b = parts[2].str.replace("-", ":", regex=False)

# sanity: the mangled value must start with the same unit number as pair_id says
for old, new in ((df.unit_a_id, new_a), (df.unit_b_id, new_b)):
    lead = old.str.extract(r"^[a-z]*(\d+)")[0].astype(int)
    want = new.str.extract(r"^[a-z]*(\d+)")[0].astype(int)
    assert (lead == want).all(), "pair_id and unit id disagree"

changed = int((df.unit_a_id != new_a).sum() + (df.unit_b_id != new_b).sum())
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
shutil.copy2(src, HERE / "backups" / f"pairs_v1.pre_idfix_{stamp}.csv")

df["unit_a_id"], df["unit_b_id"] = new_a, new_b
df.to_csv(src, index=False, lineterminator="\n")
print(f"fixed {changed} id cells; backup written")
