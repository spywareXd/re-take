"""Numeric-identity check for the caption fix: compare every regenerated/repaired report in results_v2/ with its backup in
results_v2/_stale_captions_backup/. Table BODY cells (all rows after the header and separator) must have identical numeric
content; header rows and non-table lines are captions and are listed as changes. Also checks that the CSV outputs the
regenerating scripts rewrote are byte-identical. Writes results_v2/caption_fix_check.md. If a numeric cell differs, the
report says so and the file is NOT restored automatically here (see the end of the report)."""
import hashlib
import re
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2 = HERE / "results_v2"
BK = R2 / "_stale_captions_backup"
NUM = re.compile(r"-?\d[\d,]*\.?\d*%?")

REPORTS = ["baselines.md", "logreg.md", "gbm.md", "ce.md", "combine.md", "adjudication_llama3.md", "adjudication_gemini_flashlite.md",
           "veto/veto.md", "v1_vs_v2.md"]
CSVS = ["baselines.csv", "baseline_scores.csv", "adjudication_scores_llama3.csv", "adjudication_scores_gemini_flashlite.csv",
        "adjudication_errors_llama3.csv", "adjudication_errors_gemini_flashlite.csv", "veto/veto_scores.csv", "veto/veto_changed_pairs.csv"]


def tables(text):
    """list of (header_row, body_rows) for each markdown table; rows are lists of cell strings"""
    out, cur = [], []
    for line in text.split("\n") + [""]:
        if line.startswith("|"):
            cur.append([c.strip() for c in line.strip().strip("|").split("|")])
        else:
            if cur:
                out.append(cur)
            cur = []
    res = []
    for t in out:
        if len(t) >= 2 and all(set(c) <= set("-: ") for c in t[1]):
            res.append((t[0], t[2:]))
        else:
            res.append((t[0], t[1:]))
    return res


def nonTable(text):
    return [l for l in text.split("\n") if not l.startswith("|")]


sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
md = ["# Caption-fix check", "", f"Run {date.today().isoformat()}. Old reports: results_v2/_stale_captions_backup/. For every report: table body cells must have identical numbers; "
      "header rows and non-table text are captions and may change.", "",
      "| report | tables compared | numeric cells compared | differing numeric cells | table header captions changed | non-table caption lines changed | run header added |",
      "|---|---|---|---|---|---|---|"]
tot_t = tot_c = tot_d = 0
details = []
for name in REPORTS:
    old, new = (BK / name).read_text(encoding="utf-8"), (R2 / name).read_text(encoding="utf-8")
    to, tn = tables(old), tables(new)
    cells = diffs = hdr = 0
    if len(to) != len(tn):
        diffs += 1
        details.append(f"- {name}: table count differs ({len(to)} vs {len(tn)})")
    for (ho, bo), (hn, bn) in zip(to, tn):
        if ho != hn:
            hdr += 1
            details.append(f"- {name}: table header `{' | '.join(ho)[:110]}` -> `{' | '.join(hn)[:110]}`")
        if len(bo) != len(bn):
            diffs += 1
            details.append(f"- {name}: body row count differs ({len(bo)} vs {len(bn)}) in a table starting `{ho[0]}`")
            continue
        for ro, rn in zip(bo, bn):
            for co, cn in zip(ro, rn):
                nums_o, nums_n = NUM.findall(co), NUM.findall(cn)
                cells += len(nums_o) if nums_o or nums_n else 0
                if nums_o != nums_n:
                    diffs += 1
                    details.append(f"- {name}: numeric cell differs `{co}` vs `{cn}`")
    no_, nn_ = nonTable(old), nonTable(new)
    added = [l for l in nn_ if l.startswith("> **Run header:**")]
    changed_lines = [(a, b) for a, b in zip([l for l in no_], [l for l in nn_ if not l.startswith("> **Run header:**")]) if a != b]
    # lines may shift because of the added header/blank line; compare as multisets instead
    old_set, new_set = [l for l in no_ if l.strip()], [l for l in nn_ if l.strip() and not l.startswith("> **Run header:**")]
    gone = [l for l in old_set if l not in new_set]
    came = [l for l in new_set if l not in old_set]
    for g_, c_ in zip(gone, came):
        details.append(f"- {name}: caption `{g_[:150]}` -> `{c_[:150]}`")
    if name == "v1_vs_v2.md":
        added_note = len(came)
        details.append(f"- {name}: {len(came)} line(s) added, {len(gone)} removed (expected: 1 added review-coverage bullet, 0 removed)")
    md.append(f"| {name} | {len(to)} | {cells} | {diffs} | {hdr} | {max(len(gone), len(came)) if name != 'v1_vs_v2.md' else 0} | {'yes' if added else 'no'} |")
    tot_t += len(to); tot_c += cells; tot_d += diffs
md += ["", f"**Totals: {tot_t} tables compared, {tot_c} numeric cells compared, {tot_d} differing.**", ""]
md += ["### CSV outputs rewritten by the regenerating scripts (must be byte-identical)", "", "| file | identical |", "|---|---|"]
csv_bad = 0
for name in CSVS:
    same = sha(BK / name) == sha(R2 / name)
    csv_bad += 0 if same else 1
    md.append(f"| {name} | {same} |")
md += ["", "### Changes found (captions only)", ""] + (details or ["- none"]) + [""]
md += ["### Result", "", "PASS: no numeric cell differs and every rewritten CSV is identical." if tot_d == 0 and csv_bad == 0 else
       "FAIL: see the differing cells above; the backups in _stale_captions_backup/ are untouched and can be restored.", ""]
(R2 / "caption_fix_check.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
