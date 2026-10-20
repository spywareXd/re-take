"""
Configuration constants for the training dataset population tool.
"""

from __future__ import annotations

from pathlib import Path

FEATURE_VERSION: str = "v1"

DEFAULT_MIN_LEXICAL: float = 0.10
DEFAULT_WINDOW_SECONDS: float = 300.0

# Exact CSV column specification (29 columns)
CSV_COLUMNS: list[str] = [
    "pair_id",
    "video_id",
    "source_type",
    "speaker",
    "unit_a_id",
    "unit_b_id",
    "text_a",
    "text_b",
    "start_a",
    "end_a",
    "start_b",
    "end_b",
    "label",
    "notes",
    "is_hard_negative",
    "sequence",
    "jaccard",
    "containment",
    "bigram",
    "lexical_score",
    "semantic_score",
    "recency_weight",
    "restart_phrase",
    "disagreement",
    "passes_stage1_default",
    "pipeline_class",
    "split",
    "is_synthetic",
    "feature_version",
]

# Columns recomputed when refreshing an existing video
REFRESH_FEATURE_COLUMNS: list[str] = [
    "sequence",
    "jaccard",
    "containment",
    "bigram",
    "lexical_score",
    "semantic_score",
    "recency_weight",
    "restart_phrase",
    "disagreement",
    "passes_stage1_default",
    "pipeline_class",
    "feature_version",
]

# Groups CSV column specification (16 columns)
GROUPS_CSV_COLUMNS: list[str] = [
    "group_id",
    "video_id",
    "n_units",
    "unit_ids",
    "texts",
    "starts",
    "ends",
    "keeper_unit_id",
    "keep_none",
    "notes",
    "pipeline_keeper_unit_id",
    "heuristic_later",
    "heuristic_earlier",
    "heuristic_confidence",
    "pair_ids",
    "feature_version",
]

# Directory and file defaults
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR: Path = REPO_ROOT / "data" / "training"
DEFAULT_VIDEOS_DIR: Path = DEFAULT_DATA_DIR / "videos"
DEFAULT_PROCESSED_DIR: Path = DEFAULT_DATA_DIR / "processed"
DEFAULT_BACKUPS_DIR: Path = DEFAULT_DATA_DIR / "backups"
DEFAULT_CSV_PATH: Path = DEFAULT_DATA_DIR / "pairs_v1.csv"
DEFAULT_GROUPS_CSV_PATH: Path = DEFAULT_DATA_DIR / "groups_v1.csv"
DEFAULT_MANIFEST_PATH: Path = DEFAULT_DATA_DIR / "manifest.json"

