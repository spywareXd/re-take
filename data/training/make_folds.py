"""Build the speaker-grouped fold assignment (real videos only) and audit pairs_v1.csv.

Usage: python make_folds.py   (run from anywhere; paths are relative to this file)
Reads pairs_v1.csv only (never modified). Writes folds.json and folds_report.md.
No training or scoring happens here.
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# speaker -> (fold, videos). The CSV speaker column is useless (always speaker_0),
# so the mapping is by video id.
SPEAKERS = {
    "Group A": (1, ["speaker_a_3", "speaker_a_4", "speaker_a_5", "speaker_a_6"]),
    "hawk": (2, ["video_01", "video_02", "video_03"]),
    "Group C": (3, ["video_08", "video_07"]),
    "Group D": (3, ["video_09"]),
    "Shreezy": (4, ["video_04", "video_05"]),
    "Miscellaneous": (4, ["sample_c", "sample_d", "video_06"]),
}
MIN_POSITIVES = 40

df = pd.read_csv(HERE / "pairs_v1.csv", dtype={"unit_a_id": str, "unit_b_id": str})
real = df[df["is_synthetic"] == 0].copy()
synth = df[df["is_synthetic"] == 1]

fold_of = {v: f for _, (f, vids) in SPEAKERS.items() for v in vids}
speaker_of = {v: s for s, (_, vids) in SPEAKERS.items() for v in vids}

# every real video in the data must be mapped, and vice versa
missing = set(real.video_id) - set(fold_of)
unused = set(fold_of) - set(real.video_id)
if missing or unused:
    raise SystemExit(f"Video mismatch. unmapped in data: {missing}; mapped but absent: {unused}")

real["fold"] = real.video_id.map(fold_of)
real["spk"] = real.video_id.map(speaker_of)

# ---- folds.json ----
synthetic_videos = sorted(synth.video_id.unique())
out = {
    "folds": {v: int(f) for v, f in sorted(fold_of.items())},
    "synthetic_videos": synthetic_videos,
    "speaker_of_video": dict(sorted(speaker_of.items())),
    "seed": None,
    "created": date.today().isoformat(),
}
(HERE / "folds.json").write_text(json.dumps(out, indent=2))

# ---- report ----
def counts(g):
    return pd.Series({
        "videos": g.video_id.nunique(), "rows": len(g),
        "pos": int((g.label == 1).sum()), "neg": int((g.label == 0).sum()),
        "unsure": int((g.label == -1).sum()),
    })

L = ["# Fold report", "", f"Created {out['created']}. Source: pairs_v1.csv (real rows only for folds).", ""]

L += ["## 1. Per speaker", "", real.groupby("spk").apply(counts, include_groups=False).to_markdown(), ""]

L += ["## 2. Per fold", ""]
rows = []
for f, g in real.groupby("fold"):
    pos = int((g.label == 1).sum())
    rows.append({"fold": f, "speakers": ", ".join(sorted(g.spk.unique())), "videos": g.video_id.nunique(),
                 "real_rows": len(g), "positives": pos, "pos_rate": round(pos / len(g), 4),
                 "unsure": int((g.label == -1).sum())})
fold_tbl = pd.DataFrame(rows).set_index("fold")
L += [fold_tbl.to_markdown(), ""]

rp, sp = int((real.label == 1).sum()), int((synth.label == 1).sum())
L += ["## 3. Totals", "",
      f"- Real rows: {len(real)} ({rp} positives)",
      f"- Synthetic rows: {len(synth)} ({sp} positives, {synth.video_id.nunique()} videos)",
      f"- All rows: {len(df)} ({rp + sp} positives)",
      f"- Earlier estimates (430 total / ~449 incl. synthetic) do NOT match. "
      f"The correct figure for pairs_v1.csv is {rp + sp} positives total: {rp} real + {sp} synthetic. "
      "synthetic_pairs_v1.csv was deliberately not included.", ""]

L += ["## 4. Leakage checks", ""]
vid_folds = real.groupby("video_id").fold.nunique()
multi = vid_folds[vid_folds > 1]
L.append(f"- video_id in more than one fold: {'NONE' if multi.empty else list(multi.index)}")
# No source-video column exists, so substitute check: does any synthetic text_a/text_b
# equal (case/punctuation-insensitive) the text of a real unit in any real video?
import re
norm = lambda t: re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()
real_text_fold = {}
for v, f in fold_of.items():
    for u in json.loads((HERE / "processed" / v / "units.json").read_text()):
        real_text_fold.setdefault(norm(u["raw_text"]), set()).add(f)
hit = synth[synth.text_a.map(norm).isin(real_text_fold) | synth.text_b.map(norm).isin(real_text_fold)]
L.append("- Synthetic source video: no column recorded. Substitute check (synthetic text equal to a real "
         f"unit's text): {len(hit)} of {len(synth)} synthetic rows match a real unit"
         + ("" if hit.empty else "; per fold: " + str(sorted(f for t in pd.concat([hit.text_a, hit.text_b])
                                                          for f in real_text_fold.get(norm(t), [])))))
# identical (text_a, text_b) in different folds
fold_sets = real.groupby(["text_a", "text_b"]).fold.nunique()
dup = fold_sets[fold_sets > 1]
L.append(f"- Exact duplicate (text_a, text_b) pairs across different folds: {len(dup)}")
if len(dup):
    sub = real.merge(dup.rename("n").reset_index()[["text_a", "text_b"]], on=["text_a", "text_b"])
    pos_dup = sub.groupby(["text_a", "text_b"]).label.agg(lambda x: sorted(set(x)))
    L.append(f"  - of which label-conflicting: {int((pos_dup.map(len) > 1).sum())}")
    L.append("  - examples:")
    for (a, b), g in list(sub.groupby(["text_a", "text_b"]))[:10]:
        L.append(f"    - folds {sorted(g.fold.unique())} | {a[:60]!r} / {b[:60]!r}")
L.append("")

L += ["## 5. Label sanity", ""]
L.append(f"- Unsure (-1) rows in real data: {int((real.label == -1).sum())} "
         "(exclude from training and evaluation)")
L.append(f"- Unsure (-1) rows in synthetic data: {int((synth.label == -1).sum())}")
bad_labels = real[~real.label.isin([-1, 0, 1])]
L.append(f"- Rows with label outside {{-1,0,1}}: {len(bad_labels)}")
# unit_a_id/unit_b_id in the CSV are mangled into time strings (e.g. "49:00:00" for
# unit "49:0"), so take the unit ids from pair_id ("<video>__<a>__<b>", "-" for ":")
# and check that both exist in that video's units.json.
cross = []
units_cache = {}
for r in df.itertuples():
    parts = r.pair_id.split("__")
    if parts[0] != r.video_id:
        cross.append((r.pair_id, "pair_id video != video_id"))
        continue
    if r.is_synthetic == 1:
        continue
    if r.video_id not in units_cache:
        uf = HERE / "processed" / r.video_id / "units.json"
        units_cache[r.video_id] = {u["unit_id"] for u in json.loads(uf.read_text())}
    a, b = (p.replace("-", ":") for p in parts[1:3])
    if a not in units_cache[r.video_id] or b not in units_cache[r.video_id]:
        cross.append((r.pair_id, "unit id not in this video's units.json"))
L.append(f"- Pairs whose units come from different videos / unknown units: "
         f"{'NONE' if not cross else len(cross)}")
for pid, why in cross[:10]:
    L.append(f"  - {pid}: {why}")
L.append("")

L += ["## 6. Warnings", ""]
warn = [f"Fold {f} has only {int(r.positives)} positives (< {MIN_POSITIVES})"
        for f, r in fold_tbl.iterrows() if r.positives < MIN_POSITIVES]
L += [f"- WARNING: {w}" for w in warn] or ["- none"]
L.append("")

report = "\n".join(L)
(HERE / "folds_report.md").write_text(report, encoding="utf-8")
print(report)
