"""Run header for the results_v2 reports: which labels file, its checksum, folds checksum, date and the counts
computed at run time. Used by the *_v2.py scripts and by fix_captions_v2.py so every report states what it was built from."""
import hashlib
from datetime import date
from pathlib import Path


def sha12(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def labels_header(labels_path, folds_path, n_rows, n_pos, note=""):
    """One blockquote line; n_rows/n_pos must be computed from the loaded data by the caller."""
    return (f"> **Run header:** labels `{Path(labels_path).name}` (sha256 {sha12(labels_path)}), folds `{Path(folds_path).name}` "
            f"(sha256 {sha12(folds_path)}), generated {date.today().isoformat()}; real rows with a 0/1 label: {n_rows}, retakes (label 1): {n_pos}.{note}")


def insert_header(md_lines, header):
    """Join report lines with the run header placed right after the title line."""
    return "\n".join([md_lines[0], "", header] + list(md_lines[1:]))
