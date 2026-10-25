"""Reconstruct exactly which labels changed in pairs_v1.csv. No model and no API call.

Sources:  current pairs_v1.csv  vs  results/baseline_scores.csv (labels at scoring time)  vs  backups/pairs_v1.pre_idfix_*.csv.
Writes (new files only): label_changes.csv, pairs_v1_original_labels.csv, pairs_v2.csv, results/label_audit_sample.csv,
label_changes.md. pairs_v1.csv itself is never touched.
"""
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SEED = 0
S2_T, CE_T = 0.65, 0.5                       # "wrong" = Stage 2 score >= 0.65 / cross-encoder prob_mean >= 0.5 disagrees with the label
S2_POS_CONF, S2_NEG_CONF = 0.80, 0.35        # audit sample "confident": Stage 2 >= 0.80 for retakes, <= 0.35 for non-retakes
CE_POS_CONF, CE_NEG_CONF = 0.95, 0.05        # and cross-encoder prob_mean >= 0.95 / <= 0.05

for out in ("label_changes.csv", "pairs_v1_original_labels.csv", "pairs_v2.csv", "label_changes.md", "results/label_audit_sample.csv"):
    if (HERE / out).exists():
        raise SystemExit(f"{out} already exists; refusing to overwrite")

rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
cur = rd(HERE / "pairs_v1.csv")
bs = pd.read_csv(RES / "baseline_scores.csv", dtype={"pair_id": str, "video_id": str})
ce = pd.read_csv(RES / "ce_scores.csv", dtype={"pair_id": str}).drop_duplicates("pair_id").set_index("pair_id").prob_mean
bk_files = sorted(glob.glob(str(HERE / "backups" / "pairs_v1.pre_idfix_*.csv")))
bks = [rd(f) for f in bk_files]
folds = json.loads((HERE / "folds.json").read_text())["folds"]
md = ["# Label changes in pairs_v1.csv", ""]

# ---------------- step 1: do the old sources agree? ----------------
drop = ["unit_a_id", "unit_b_id"]                                     # ids are mangled in the backups
bk_equal = all(bks[0].drop(columns=drop).equals(b.drop(columns=drop)) for b in bks[1:])
old = bks[0].set_index("pair_id").label
base_lab = bs.set_index("pair_id").label.astype(str)
base_match = bool((old.reindex(base_lab.index) == base_lab).all())
md += ["## 1. Do the old label sources agree?", "",
       f"- Backups compared: {', '.join(Path(f).name for f in bk_files)}; identical to each other (all columns except the mangled unit ids): **{bk_equal}**.",
       f"- Backup labels vs results/baseline_scores.csv labels on its {len(base_lab)} rows: **{'identical' if base_match else 'DIFFERENT'}**.", ""]
if not (bk_equal and base_match):
    (HERE / "label_changes.md").write_text("\n".join(md + ["STOPPED: the two old sources differ."]), encoding="utf-8")
    raise SystemExit("old sources differ; stopped")

# ---------------- step 2: the changes ----------------
c = cur.set_index("pair_id")
assert set(c.index) == set(old.index) and len(c) == len(old)
chg = c.index[(c.label.values != old.reindex(c.index).values)]
rows = []
s2 = bs.set_index("pair_id").current_stage2
for pid in chg:
    r = c.loc[pid]
    rows.append({"pair_id": pid, "video_id": r.video_id, "fold": folds.get(r.video_id, ""), "text_a": r.text_a, "text_b": r.text_b,
                 "old_label": old[pid], "new_label": r.label, "stage2_score": s2.get(pid, np.nan), "ce_prob_mean": ce.get(pid, np.nan),
                 "pipeline_class": r.pipeline_class, "notes": r.notes, "reason": ""})
ch = pd.DataFrame(rows).sort_values(["fold", "video_id", "pair_id"])
ch.to_csv(HERE / "label_changes.csv", index=False)

ch["direction"] = ch.old_label + "->" + ch.new_label
ch["s2_pred"] = (ch.stage2_score >= S2_T).astype(int)
ch["ce_pred"] = (ch.ce_prob_mean >= CE_T).astype(int)
for lab, name in (("old_label", "old"), ("new_label", "new")):
    num = pd.to_numeric(ch[lab])
    ch[f"s2_wrong_{name}"] = np.where(num == -1, np.nan, (ch.s2_pred != num).astype(float))
    ch[f"ce_wrong_{name}"] = np.where(num == -1, np.nan, (ch.ce_pred != num).astype(float))
cnt = lambda s: ", ".join(f"{k}: {v}" for k, v in s.value_counts().sort_index().items())
md += ["## 2. The changes (label_changes.csv, `reason` column left blank)", "",
       f"- **Total labels changed: {len(ch)}** (of {len(c)} rows; real rows only, none synthetic: {int((c.loc[ch.pair_id].is_synthetic == '1').sum())} synthetic).",
       f"- By fold: {cnt(ch.fold)}", f"- By video: {cnt(ch.video_id)}", f"- By direction (old->new): {cnt(ch.direction)}",
       f"- Changes involving -1: {int(((ch.old_label == '-1') | (ch.new_label == '-1')).sum())}",
       f"- Label totals: before {old.value_counts().to_dict()}, now {c.label.value_counts().to_dict()}", "",
       f"Were these pairs ones the models got wrong? (Stage 2 at {S2_T}, cross-encoder prob_mean at {CE_T}; `n/a` = the label on that side is -1)", "",
       "| against | Stage 2 wrong | cross-encoder wrong | either wrong | both wrong |", "|---|---|---|---|---|"]
for name in ("old", "new"):
    s_, c_ = ch[f"s2_wrong_{name}"], ch[f"ce_wrong_{name}"]
    valid = s_.notna()
    md.append(f"| {name} label (n={int(valid.sum())}) | {int(s_.sum())} | {int(c_.sum())} | {int(((s_ == 1) | (c_ == 1))[valid].sum())} | {int(((s_ == 1) & (c_ == 1))[valid].sum())} |")
md += ["", "| pair_id | video | dir | stage2 | CE | text A | text B |", "|---|---|---|---|---|---|---|"]
for r in ch.itertuples():
    md.append(f"| {r.pair_id} | {r.video_id} | {r.direction} | {r.stage2_score:.2f} | {r.ce_prob_mean:.2f} | {r.text_a[:60]} | {r.text_b[:60]} |")
md.append("")

# ---------------- step 3: preserve both versions ----------------
orig = bks[0].copy()
parts = orig.pair_id.str.split("__", expand=True)
orig["unit_a_id"], orig["unit_b_id"] = parts[1].str.replace("-", ":", regex=False), parts[2].str.replace("-", ":", regex=False)
orig.to_csv(HERE / "pairs_v1_original_labels.csv", index=False, lineterminator="\n")
shutil.copy2(HERE / "pairs_v1.csv", HERE / "pairs_v2.csv")
chk = rd(HERE / "pairs_v1_original_labels.csv")
others = [x for x in chk.columns if x not in ("unit_a_id", "unit_b_id")]
md += ["## 3. Preserved versions", "",
       f"- pairs_v1_original_labels.csv: rebuilt from {Path(bk_files[0]).name}; unit ids rebuilt from pair_id; all other columns unchanged "
       f"(verified identical to the backup: {bool(chk[others].equals(bks[0][others]))}); it differs from the current file in exactly {int((chk.set_index('pair_id').label != c.label.reindex(chk.pair_id).values).sum())} labels.",
       f"- pairs_v2.csv: byte copy of the current pairs_v1.csv (identical: {(HERE / 'pairs_v2.csv').read_bytes() == (HERE / 'pairs_v1.csv').read_bytes()}). pairs_v1.csv was not touched.", ""]

# ---------------- step 4: groups ----------------
groups = rd(HERE / "groups_v1.csv")
uid = lambda pid, i: pid.split("__")[i].replace("-", ":")
lab_now = c.label
lab_old = old
md += ["## 4. Groups (groups_v1.csv)", ""]
found = 0
group_rows = []
for pid in ch.pair_id:
    vid, a, b = c.loc[pid].video_id, uid(pid, 1), uid(pid, 2)
    hit = []
    for g in groups[groups.video_id == vid].itertuples():
        units = g.unit_ids.split("|")
        if a in units and b in units:
            pids = [p for p in c.index if c.loc[p].video_id == vid and uid(p, 1) in units and uid(p, 2) in units]
            labs_now = [lab_now[p] for p in pids]
            labs_old = [lab_old[p] for p in pids]
            n_pairs_possible = len(units) * (len(units) - 1) // 2
            hit.append((g.group_id, len(units), len(pids), n_pairs_possible, labs_now, labs_old, g.keep_none, g.keeper_unit_ids))
    if hit:
        found += 1
    for h in hit:
        ok_now, ok_old = all(x == "1" for x in h[4]), all(x == "1" for x in h[5])
        group_rows.append(f"| {pid} ({ch.set_index('pair_id').loc[pid].direction}) | {h[0]} | {h[1]} | {h[2]}/{h[3]} | "
                          f"{'consistent' if ok_now else 'INCONSISTENT'} (now: {dict(pd.Series(h[4]).value_counts())}) | {'consistent' if ok_old else 'inconsistent'} before | keep_none={h[6]}, keepers={h[7]} |")
    if not hit:
        group_rows.append(f"| {pid} ({ch.set_index('pair_id').loc[pid].direction}) | none | | | | | |")
md += [f"{found} of {len(ch)} changed pairs have both units inside at least one group. Consistent = every candidate pair among the group's units "
       "is labelled 1 (pairs that were never exported as candidates cannot be checked; shown as pairs found / pairs possible).", "",
       "| changed pair | group | units | pairs found/possible | consistency now | before | group keeper info |", "|---|---|---|---|---|---|---|", *group_rows, ""]

# ---------------- step 5: audit sample ----------------
pool = c[(c.is_synthetic == "0") & (c.label != "-1") & (~c.index.isin(ch.pair_id))].copy()
pool = pool.join(s2.rename("stage2_score")).join(ce.rename("ce_prob_mean"), how="left")
pool = pool[pool.stage2_score.notna() & pool.ce_prob_mean.notna()]
pool["label_n"] = pool.label.astype(int)
pool["fold"] = pool.video_id.map(folds)
pos, neg = pool.label_n == 1, pool.label_n == 0
ok = (pos & (pool.stage2_score >= S2_POS_CONF) & (pool.ce_prob_mean >= CE_POS_CONF)) | (neg & (pool.stage2_score <= S2_NEG_CONF) & (pool.ce_prob_mean <= CE_NEG_CONF))
elig = pool[ok]
# stratify by fold (proportional to eligible pairs, largest remainder), half retakes / half non-retakes inside a fold when possible
n_total = len(ch)
share = elig.groupby("fold").size()
alloc = (share / share.sum() * n_total)
take = alloc.astype(int)
for f in (alloc - take).sort_values(ascending=False).index[: n_total - int(take.sum())]:
    take[f] += 1
rng = np.random.default_rng(SEED)
picked = []
for f, n in take.items():
    g = elig[elig.fold == f]
    n_pos = min(n // 2 + n % 2, int((g.label_n == 1).sum())) if n else 0
    n_neg = min(n - n_pos, int((g.label_n == 0).sum()))
    n_pos = n - n_neg if n_pos + n_neg < n else n_pos
    for lab, k in ((1, n_pos), (0, n_neg)):
        gl = g[g.label_n == lab]
        picked.append(gl.iloc[rng.permutation(len(gl))[:k]])
aud = pd.concat(picked).reset_index().rename(columns={"index": "pair_id"})
aud = aud[["pair_id", "video_id", "fold", "text_a", "text_b", "label", "stage2_score", "ce_prob_mean", "pipeline_class", "notes"]]
aud["verdict"], aud["audit_notes"] = "", ""
aud.to_csv(RES / "label_audit_sample.csv", index=False)
md += ["## 5. Random audit sample (results/label_audit_sample.csv, blank `verdict`/`audit_notes` for you)", "",
       f"- Pool: non-changed real pairs with a 0/1 label where Stage 2 and the cross-encoder are both correct and both confident "
       f"(retake: Stage 2 >= {S2_POS_CONF} and CE >= {CE_POS_CONF}; non-retake: Stage 2 <= {S2_NEG_CONF} and CE <= {CE_NEG_CONF}); {len(elig)} eligible pairs "
       f"({int((elig.label_n == 1).sum())} retakes, {int((elig.label_n == 0).sum())} non-retakes).",
       f"- Sample size {len(aud)} (= number of changes), seed {SEED}, allocated to folds in proportion to eligible pairs ({dict(take)}), balanced between retakes and non-retakes within a fold where possible. "
       f"Result: {dict(aud.label.value_counts())} by label, {dict(aud.fold.value_counts().sort_index())} by fold.", "",
       "Reading it: the changed-label rate so far is "
       f"{len(ch)}/{int((c.label != '-1').sum())} = {len(ch) / int((c.label != '-1').sum()):.2%} of labelled pairs. If your audit of these confident pairs finds errors, the labels are noisier than that rate suggests.", ""]
(HERE / "label_changes.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md[:40]))
