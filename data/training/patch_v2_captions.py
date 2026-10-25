"""Fix stale v1 captions IN THE v2 SCRIPTS: replace hard-coded counts/paths in report text with values computed from the
loaded data, and add the run header. Idempotent. Only text printed in reports changes; no metric, threshold, seed or
bootstrap code is touched."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
HDR_IMPORT = "from report_header import insert_header, labels_header  # run header for the report (captions only)\n"

# (file, old, new)
PATCHES = [
    # ---- run header on every script that writes a report: import + replace the final write ----
    ("baselines_v2.py", '(OUT / "baselines.md").write_text("\\n".join(md), encoding="utf-8")',
     '(OUT / "baselines.md").write_text(insert_header(md, labels_header(HERE / "pairs_v2.csv", HERE / "folds.json", len(df), int(df.label.sum()))), encoding="utf-8")'),
    ("logreg_v2.py", '(OUT / "logreg.md").write_text("\\n".join(md), encoding="utf-8")',
     '(OUT / "logreg.md").write_text(insert_header(md, labels_header(HERE / "pairs_v2.csv", HERE / "folds.json", len(real), int(real.label.sum()))), encoding="utf-8")'),
    ("gbm_errors_v2.py", '(OUT / "gbm.md").write_text("\\n".join(md), encoding="utf-8")',
     '(OUT / "gbm.md").write_text(insert_header(md, labels_header(HERE / "pairs_v2.csv", HERE / "folds.json", len(df), int(df.label.sum()))), encoding="utf-8")'),
    ("train_cross_encoder_v2.py", '(OUT / "ce.md").write_text("\\n".join(md), encoding="utf-8")',
     '(OUT / "ce.md").write_text(insert_header(md, labels_header(HERE / "pairs_v2.csv", HERE / "folds.json", len(df), int(df.label.sum()))), encoding="utf-8")'),
    ("combine_v2.py", '(OUT / "combine.md").write_text("\\n".join(md), encoding="utf-8")',
     '(OUT / "combine.md").write_text(insert_header(md, labels_header(HERE / "pairs_v2.csv", HERE / "folds.json", len(df), int(df.label.sum()))), encoding="utf-8")'),
    ("adjudication_compare_v2.py", '(OUT / f"adjudication{SUFFIX}.md").write_text("\\n".join(md), encoding="utf-8")',
     '(OUT / f"adjudication{SUFFIX}.md").write_text(insert_header(md, labels_header(HERE / "pairs_v2.csv", HERE / "folds.json", len(df), int(df.label.sum()))), encoding="utf-8")'),
    ("veto_test_v2.py", '(OUT / "veto.md").write_text("\\n".join(md), encoding="utf-8")',
     '(OUT / "veto.md").write_text(insert_header(md, labels_header(HERE / "pairs_v2.csv", HERE / "folds.json", len(df), int(df.label.sum()))), encoding="utf-8")'),
    # ---- stale counts / paths in report text ----
    ("adjudication_compare_v2.py", 'md += ["### End to end (all 3570 real rows;', 'md += [f"### End to end (all {len(df)} real rows;'),
    ("adjudication_compare_v2.py", 'md += ["### The 79 Stage 2 errors", "",', 'md += [f"### The {len(errors)} Stage 2 errors", "",'),
    ("adjudication_compare_v2.py", 'md += ["### Extra analysis (separate): zero-shot on all 79 Stage 2 errors regardless of band", "",',
     'md += [f"### Extra analysis (separate): zero-shot on all {len(errors)} Stage 2 errors regardless of band", "",'),
    ("adjudication_compare_v2.py", "# the 79 Stage 2 errors", "# the Stage 2 errors (count computed from the data)"),
    ("adjudication_compare_v2.py", "# extra analysis (separate): zero-shot on all 79 Stage 2 errors regardless of band",
     "# extra analysis (separate): zero-shot on all Stage 2 errors regardless of band"),
    ("combine_v2.py", '"### Errors vs Stage 2 @0.65 (pooled; Stage 2 has 49 FN and 30 FP)"',
     'f"### Errors vs Stage 2 @0.65 (pooled; Stage 2 has {int(((y == 1) & (s2_pred == 0)).sum())} FN and {int(((y == 0) & (s2_pred == 1)).sum())} FP)"'),
    ("gbm_errors_v2.py", '"(see results/stage2_errors.csv;', 'f"(see {OUT.name}/stage2_errors.csv;'),
    ("train_cross_encoder_v2.py", '"- Lists with text: results/ce_vs_stage2_errors.csv."', 'f"- Lists with text: {OUT.name}/ce_vs_stage2_errors.csv."'),
    ("veto_test_v2.py", '(all 3,570 rows; everything outside', '(all {len(df):,} rows; everything outside'),
    ("veto_test_v2.py", 'md += ["", "### Per rule: what was vetoed', 'md += ["", f"### Per rule: what was vetoed'),
    ("veto_test_v2.py", '"| rule | FPs removed (of 30) | retakes dropped (of 303) | TP |',
     'f"| rule | FPs removed (of {fp_total}) | retakes dropped (of {tr_total}) | TP |'),
    ("veto_test_v2.py", "of the 30 false positives.\", \"\"]", "of the {fp_total} false positives.\", \"\"]"),
    ("veto_test_v2.py", "of the 30 had an answer under the alias run", "of the {fp_total} had an answer under the alias run"),
    ("veto_test_v2.py", 'md += ["### Exploratory: earlier alias run vs the pinned id on the 30 false positives", "",',
     'md += [f"### Exploratory: earlier alias run vs the pinned id on the {fp_total} false positives", "",'),
    ("veto_test_v2.py", "; 30 FPs and 303 retakes among the 333 pairs.\"", "; {fp_total} FPs and {tr_total} retakes among the {len(SET)} pairs.\""),
    ("veto_test_v2.py", "# consistency with the earlier alias run (read-only, exploratory): the 30 false positives",
     "# consistency with the earlier alias run (read-only, exploratory): the false positives"),
]
IMPORT_FILES = ["baselines_v2.py", "logreg_v2.py", "gbm_errors_v2.py", "train_cross_encoder_v2.py", "combine_v2.py", "adjudication_compare_v2.py", "veto_test_v2.py"]
changed = 0
for name in IMPORT_FILES:
    p = HERE / name
    src = p.read_text(encoding="utf-8")
    if HDR_IMPORT not in src:
        # put the import after the module docstring (first line of the generated file is the docstring header, then the original docstring)
        marker = "\nimport "
        i = src.index(marker) + 1
        src = src[:i] + HDR_IMPORT + src[i:]
        p.write_text(src, encoding="utf-8")
        changed += 1
for name, old, new in PATCHES:
    p = HERE / name
    src = p.read_text(encoding="utf-8")
    if new in src:
        continue
    assert old in src, f"{name}: pattern not found: {old[:60]}"
    p.write_text(src.replace(old, new, 1), encoding="utf-8")
    changed += 1
print("patched", changed, "places")
