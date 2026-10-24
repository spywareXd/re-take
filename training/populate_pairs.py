"""
populate_pairs.py

CLI tool to extract candidate pairs from training videos through Stage 2 (semantic matching)
and append them as rows in data/training/pairs_v1.csv for manual labeling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.step_03_token_matching.config import (
    PAUSE_SPLIT_THRESHOLD_SECONDS,
    TokenMatchingConfig,
    default_config as default_token_config,
)
from pipeline.step_04_semantic_matching.config import (
    default_semantic_config,
)
from pipeline.step_05_cutlist.cutlist_builder import select_take
from pipeline.step_05_span_aggregation.span_aggregation import (
    _determine_likely_final_unit_id,
)
from training.config import (
    CSV_COLUMNS,
    DEFAULT_BACKUPS_DIR,
    DEFAULT_CSV_PATH,
    DEFAULT_GROUPS_CSV_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MIN_LEXICAL,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_VIDEOS_DIR,
    DEFAULT_WINDOW_SECONDS,
    FEATURE_VERSION,
    GROUPS_CSV_COLUMNS,
    GROUPS_CSV_COLUMNS_LEGACY,
    REFRESH_FEATURE_COLUMNS,
)

logger = logging.getLogger("populate_pairs")

VALID_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv"}


def compute_sha256(filepath: Path | str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_video_id(filepath: Path | str) -> str:
    """Returns video_id derived from filename stem with spaces replaced by underscores."""
    stem = Path(filepath).stem
    return stem.replace(" ", "_")


def determine_source_type(
    video_id: str,
    filename: str,
    explicit_source: str | None = None,
) -> str:
    """Determines source_type from argument, filename prefix, or 'unknown'."""
    if explicit_source:
        return explicit_source
    lower_name = filename.lower()
    if lower_name.startswith("scripted_"):
        return "scripted"
    if lower_name.startswith("real_"):
        return "real"
    return "unknown"


def load_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Loads manifest.json or returns empty dict."""
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning(f"Failed to read manifest at {manifest_path}: {e}")
    return {}


def atomic_write_manifest(manifest_path: Path, manifest: dict[str, dict[str, Any]]) -> None:
    """Atomically writes manifest to manifest.json via temporary file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_name(f"{manifest_path.name}.tmp.{uuid.uuid4().hex}")
    try:
        temp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temp_path, manifest_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def read_pairs_csv(csv_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Reads pairs_v1.csv with utf-8-sig encoding.
    Verifies that the header matches CSV_COLUMNS exactly.
    Catches PermissionError (e.g. locked in Excel) and exits non-zero.
    """
    if not csv_path.exists():
        return list(CSV_COLUMNS), []

    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return list(CSV_COLUMNS), []
            if header != CSV_COLUMNS:
                raise ValueError(
                    f"Header mismatch in {csv_path}.\n"
                    f"Expected: {CSV_COLUMNS}\n"
                    f"Found:    {header}"
                )
            rows: list[dict[str, Any]] = []
            for row in reader:
                if not row:
                    continue
                rows.append(dict(zip(header, row)))
            return header, rows
    except PermissionError:
        sys.stderr.write("close pairs_v1.csv and rerun\n")
        sys.exit(1)


def backup_pairs_csv(csv_path: Path, backups_dir: Path) -> Path | None:
    """
    Creates a timestamped copy of pairs_v1.csv under backups_dir before write.
    Prunes older backups to keep the 20 most recent.
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None

    backups_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = backups_dir / f"pairs_v1.{timestamp}.csv"

    try:
        shutil.copy2(csv_path, backup_path)
    except PermissionError:
        sys.stderr.write("close pairs_v1.csv and rerun\n")
        sys.exit(1)

    # Prune beyond 20 most recent
    backups = sorted(backups_dir.glob("pairs_v1.*.csv"), key=lambda p: (p.stat().st_mtime, p.name))
    if len(backups) > 20:
        for old in backups[:-20]:
            try:
                old.unlink()
            except Exception:
                pass

    return backup_path


def atomic_write_pairs_csv(
    csv_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] = CSV_COLUMNS,
) -> None:
    """
    Atomically writes rows to pairs_v1.csv with utf-8-sig encoding.
    Catches PermissionError (Excel file locked) and exits non-zero.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = csv_path.with_name(f"{csv_path.name}.tmp.{uuid.uuid4().hex}")
    try:
        with open(temp_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(temp_path, csv_path)
    except PermissionError:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        sys.stderr.write("close pairs_v1.csv and rerun\n")
        sys.exit(1)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def _is_synthetic_video(video_id: str) -> bool:
    """Returns True if a video_id is synthetic or dummy and should be excluded from groups."""
    vid = str(video_id)
    return vid.startswith("dummy_") or vid.startswith("synth_")


def _unit_ids_from_pair_id(pair_id: str, video_id: str) -> tuple[str, str]:
    """
    Derives unit_a_id and unit_b_id from pair_id.
    pair_id format: {video_id}__{A}__{B} where ':' in A and B is replaced by '-'.
    E.g. 'speaker_a_3__11-0__15-0' -> ('11:0', '15:0')
    """
    prefix = video_id + "__"
    if pair_id.startswith(prefix):
        rest = pair_id[len(prefix):]
    else:
        # Fallback: split on __ and take last two parts
        parts = pair_id.split("__")
        if len(parts) >= 3:
            rest = "__".join(parts[1:])
        else:
            rest = pair_id

    # Split on last __ to get A and B
    double_idx = rest.rfind("__")
    if double_idx == -1:
        return rest.replace("-", ":", 1), ""
    raw_a = rest[:double_idx]
    raw_b = rest[double_idx + 2:]

    def _decode(s: str) -> str:
        # Only replace dashes that look like segment:sub separators (digit-digit)
        # e.g. '11-0' -> '11:0', '7-1' -> '7:1'
        # Preserve alpha prefixes: e.g. 'synth-01' stays as-is since these won't have a pure digit:digit pattern
        import re
        # Replace patterns like 'N-M' where N and M are digits with 'N:M'
        return re.sub(r'^(\d+)-(\d+)$', r'\1:\2', s)

    return _decode(raw_a), _decode(raw_b)


def _normalize_unit_id_str(u: str) -> str:
    """
    Normalizes a unit id string, handling Excel time auto-formatting (e.g. '04:00' -> '4:0').
    Preserves synthetic prefixes like 's01-1' or 'dummy'.
    """
    s = str(u).strip()
    if not s or s.lower() == "nan":
        return ""
    m = re.match(r"^0*(\d+):0*(\d+)(?::0+)?$", s)
    if m:
        return f"{int(m.group(1))}:{int(m.group(2))}"
    return s


def _normalize_keeper_unit_ids(keeper_str: str) -> str:
    """Normalizes pipe-delimited keeper_unit_ids, converting any Excel time corruptions."""
    if not keeper_str or str(keeper_str).strip() == "" or str(keeper_str).strip().lower() == "nan":
        return ""
    parts = [str(k).strip() for k in str(keeper_str).split("|") if str(k).strip()]
    return "|".join(_normalize_unit_id_str(k) for k in parts)


def read_groups_csv(groups_csv_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Reads groups_v1.csv with utf-8-sig encoding.
    Detects and migrates legacy headers (keeper_unit_id -> keeper_unit_ids, adds n_pos_pairs/n_exempt_pairs).
    Verifies that the header matches GROUPS_CSV_COLUMNS exactly after migration.
    Catches PermissionError (e.g. locked in Excel) and exits non-zero.
    """
    if not groups_csv_path.exists():
        return list(GROUPS_CSV_COLUMNS), []

    try:
        with open(groups_csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return list(GROUPS_CSV_COLUMNS), []

            # Detect legacy header and migrate
            if header == GROUPS_CSV_COLUMNS_LEGACY:
                print(
                    f"[Migration] {groups_csv_path.name}: legacy header detected. "
                    "Migrating keeper_unit_id -> keeper_unit_ids and adding n_pos_pairs/n_exempt_pairs."
                )
                raw_rows = list(reader)
                rows: list[dict[str, Any]] = []
                for row in raw_rows:
                    if not row:
                        continue
                    d = dict(zip(GROUPS_CSV_COLUMNS_LEGACY, row))
                    # Rename keeper_unit_id -> keeper_unit_ids
                    d["keeper_unit_ids"] = _normalize_keeper_unit_ids(d.pop("keeper_unit_id", ""))
                    d["n_pos_pairs"] = d.get("n_pos_pairs", "")
                    d["n_exempt_pairs"] = d.get("n_exempt_pairs", "")
                    rows.append(d)
                return list(GROUPS_CSV_COLUMNS), rows

            if header != GROUPS_CSV_COLUMNS:
                raise ValueError(
                    f"Header mismatch in {groups_csv_path}.\n"
                    f"Expected: {GROUPS_CSV_COLUMNS}\n"
                    f"Found:    {header}"
                )
            rows = []
            for row in reader:
                if not row:
                    continue
                d = dict(zip(header, row))
                if d.get("keeper_unit_ids"):
                    d["keeper_unit_ids"] = _normalize_keeper_unit_ids(d["keeper_unit_ids"])
                rows.append(d)
            return header, rows
    except PermissionError:
        sys.stderr.write("close groups_v1.csv and rerun\n")
        sys.exit(1)


def backup_groups_csv(groups_csv_path: Path, backups_dir: Path) -> Path | None:
    """
    Creates a timestamped copy of groups_v1.csv under backups_dir before write.
    Prunes older backups to keep the 20 most recent.
    """
    if not groups_csv_path.exists() or groups_csv_path.stat().st_size == 0:
        return None

    backups_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = backups_dir / f"groups_v1.{timestamp}.csv"

    try:
        shutil.copy2(groups_csv_path, backup_path)
    except PermissionError:
        sys.stderr.write("close groups_v1.csv and rerun\n")
        sys.exit(1)

    backups = sorted(backups_dir.glob("groups_v1.*.csv"), key=lambda p: (p.stat().st_mtime, p.name))
    if len(backups) > 20:
        for old in backups[:-20]:
            try:
                old.unlink()
            except Exception:
                pass

    return backup_path


def atomic_write_groups_csv(
    groups_csv_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] = GROUPS_CSV_COLUMNS,
) -> None:
    """
    Atomically writes rows to groups_v1.csv with utf-8-sig encoding.
    Catches PermissionError (Excel file locked) and exits non-zero.
    """
    groups_csv_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = groups_csv_path.with_name(f"{groups_csv_path.name}.tmp.{uuid.uuid4().hex}")
    try:
        with open(temp_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(temp_path, groups_csv_path)
    except PermissionError:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        sys.stderr.write("close groups_v1.csv and rerun\n")
        sys.exit(1)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def _extract_unit_dict(unit: Any) -> dict[str, Any]:
    """Normalizes candidate unit to a plain dictionary."""
    if isinstance(unit, dict):
        return unit
    if hasattr(unit, "to_dict") and callable(unit.to_dict):
        return unit.to_dict()
    if is_dataclass(unit):
        return asdict(unit)
    return {
        "unit_id": getattr(unit, "unit_id", ""),
        "speaker": getattr(unit, "speaker", "speaker_0"),
        "raw_text": getattr(unit, "raw_text", ""),
        "start": getattr(unit, "start", 0.0),
        "end": getattr(unit, "end", 0.0),
    }


def candidate_to_csv_row(
    candidate: dict[str, Any],
    video_id: str,
    source_type: str,
    feature_version: str = FEATURE_VERSION,
) -> dict[str, Any]:
    """
    Transforms an enriched Stage 2 candidate dictionary into an exact CSV row dictionary.
    Guarantees unit_a starts before or at unit_b.
    """
    unit_a = _extract_unit_dict(candidate.get("unit_a", {}))
    unit_b = _extract_unit_dict(candidate.get("unit_b", {}))

    start_a = float(unit_a.get("start", 0.0))
    start_b = float(unit_b.get("start", 0.0))

    # Hard rule: A earlier than B
    if start_a > start_b:
        unit_a, unit_b = unit_b, unit_a
        start_a, start_b = start_b, start_a

    end_a = float(unit_a.get("end", 0.0))
    end_b = float(unit_b.get("end", 0.0))

    uid_a = str(unit_a.get("unit_id", ""))
    uid_b = str(unit_b.get("unit_id", ""))
    pair_id = f"{video_id}__{uid_a.replace(':', '-')}__{uid_b.replace(':', '-')}"

    scores = candidate.get("scores", {})
    if isinstance(scores, dict):
        seq = float(scores.get("sequence", 0.0))
        jac = float(scores.get("jaccard", 0.0))
        cont = float(scores.get("containment", 0.0))
        bg = float(scores.get("bigram", 0.0))
    else:
        seq = float(getattr(scores, "sequence", 0.0))
        jac = float(getattr(scores, "jaccard", 0.0))
        cont = float(getattr(scores, "containment", 0.0))
        bg = float(getattr(scores, "bigram", 0.0))

    lex_score = float(candidate.get("lexical_score", 0.0))
    sem_score = float(candidate.get("semantic_score", 0.0))
    rec_weight = float(candidate.get("recency_weight", 0.0))
    restart_flag = 1 if candidate.get("restart_phrase_detected") else 0
    disagreement = abs(lex_score - sem_score) if ("lexical_score" in candidate or "semantic_score" in candidate) else float(candidate.get("lexical_semantic_disagreement", 0.0))

    # passes_stage1_default calculation based on pipeline defaults
    default_lex_thresh = float(default_token_config.candidate_threshold)
    default_win = float(default_token_config.window_seconds)
    time_gap = abs(start_b - start_a)
    passes_s1 = 1 if (lex_score >= default_lex_thresh and time_gap <= default_win) else 0

    pipeline_class = str(candidate.get("classification", ""))

    return {
        "pair_id": pair_id,
        "video_id": video_id,
        "source_type": source_type,
        "speaker": str(unit_a.get("speaker", "speaker_0")),
        "unit_a_id": uid_a,
        "unit_b_id": uid_b,
        "text_a": str(unit_a.get("raw_text", "")),
        "text_b": str(unit_b.get("raw_text", "")),
        "start_a": round(start_a, 3),
        "end_a": round(end_a, 3),
        "start_b": round(start_b, 3),
        "end_b": round(end_b, 3),
        "label": "",
        "notes": "",
        "is_hard_negative": "",
        "sequence": round(seq, 3),
        "jaccard": round(jac, 3),
        "containment": round(cont, 3),
        "bigram": round(bg, 3),
        "lexical_score": round(lex_score, 3),
        "semantic_score": round(sem_score, 3),
        "recency_weight": round(rec_weight, 3),
        "restart_phrase": restart_flag,
        "disagreement": round(disagreement, 3),
        "passes_stage1_default": passes_s1,
        "pipeline_class": pipeline_class,
        "split": "",
        "is_synthetic": 0,
        "feature_version": feature_version,
    }


def serialize_comparable_units(units: Sequence[Any]) -> list[dict[str, Any]]:
    """Serializes ComparableUnit objects to JSON-compatible dictionaries."""
    res = []
    for u in units:
        res.append(
            {
                "unit_id": getattr(u, "unit_id", ""),
                "source_segment_id": getattr(u, "source_segment_id", 0),
                "start": getattr(u, "start", 0.0),
                "end": getattr(u, "end", 0.0),
                "speaker": getattr(u, "speaker", "speaker_0"),
                "raw_text": getattr(u, "raw_text", ""),
                "normalized_text": getattr(u, "normalized_text", ""),
                "tokens": list(getattr(u, "tokens", [])),
                "confidence": getattr(u, "confidence", 1.0),
                "is_below_min_token_length": getattr(u, "is_below_min_token_length", False),
            }
        )
    return res


def run_pipeline_for_video(
    video_path: Path,
    video_id: str,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    min_lexical: float = DEFAULT_MIN_LEXICAL,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    reuse_transcript: bool = False,
    whisper_provider: str = "whisperx",
    whisper_model: str = "large-v2",
    whisper_device: str | None = None,
    dry_run: bool = False,
) -> tuple[Any, list[Any], list[dict[str, Any]]]:
    """
    Executes existing pipeline transcription, unit splitting, candidate generation, and semantic scoring.
    Intermediate files are written to data/training/processed/<video_id>/ unless dry_run is True.
    """
    from models.transcript import Transcript
    from pipeline.step_01_transcription import TimestampGenerator
    from pipeline.step_03_token_matching import (
        generate_candidate_pairs,
        load_transcript_units,
    )
    from pipeline.step_04_semantic_matching import run_stage2

    video_proc_dir = processed_dir / video_id
    transcript_path = video_proc_dir / "transcript.json"

    transcript: Transcript | None = None
    if reuse_transcript and transcript_path.exists():
        try:
            transcript = Transcript.from_json_file(str(transcript_path))
            logger.info(f"Reusing cached transcript for {video_id} from {transcript_path}")
        except Exception as e:
            logger.warning(f"Failed to load cached transcript at {transcript_path}: {e}")

    if transcript is None:
        if whisper_device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        else:
            device = whisper_device

        compute_type = "float16" if device == "cuda" else "float32"
        generator = TimestampGenerator(
            provider=whisper_provider,
            model_name=whisper_model,
            device=device,
            compute_type=compute_type,
        )

        if dry_run:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp_audio_dir:
                transcript = generator.process(
                    media_path=str(video_path),
                    output_json_path=None,
                    extracted_audio_dir=tmp_audio_dir,
                )
        else:
            video_proc_dir.mkdir(parents=True, exist_ok=True)
            transcript = generator.process(
                media_path=str(video_path),
                output_json_path=str(transcript_path),
                extracted_audio_dir=str(video_proc_dir),
            )


    # Unit Splitting
    units = load_transcript_units(transcript, pause_threshold=PAUSE_SPLIT_THRESHOLD_SECONDS)

    # Token Matching / Candidate generation with user overrides
    token_config = TokenMatchingConfig(
        candidate_threshold=min_lexical,
        window_seconds=window_seconds,
    )
    candidates = generate_candidate_pairs(units, config=token_config)

    # Stage 2 Semantic Matching
    stage2_results = run_stage2(candidates)

    if not dry_run:
        video_proc_dir.mkdir(parents=True, exist_ok=True)
        units_path = video_proc_dir / "units.json"
        candidates_path = video_proc_dir / "candidates.json"

        # Cache units.json
        serialized_units = serialize_comparable_units(units)
        units_path.write_text(json.dumps(serialized_units, indent=2), encoding="utf-8")

        # Cache candidates.json
        candidates_path.write_text(json.dumps(stage2_results, indent=2), encoding="utf-8")

    return transcript, units, stage2_results


def print_stats(
    csv_path: Path = DEFAULT_CSV_PATH,
    groups_csv_path: Path = DEFAULT_GROUPS_CSV_PATH,
) -> None:
    """Prints dataset progress matching the required --stats breakdown for pairs and groups."""
    header, rows = read_pairs_csv(csv_path)

    # Exclude dummy rows
    valid_rows = [r for r in rows if not str(r.get("video_id", "")).startswith("dummy_")]

    by_video: dict[str, list[dict[str, Any]]] = {}
    for r in valid_rows:
        vid = str(r.get("video_id", ""))
        by_video.setdefault(vid, []).append(r)

    print("\nCandidate Pairs Progress (--stats)")
    print("=" * 88)
    print(
        f"{'Video ID':<18} {'Total':<7} {'Labeled':<9} {'Unlabeled':<11} "
        f"{'Label 1':<9} {'Label 0':<9} {'Label -1':<10} {'Passes S1'}"
    )
    print("-" * 88)

    total_rows = len(valid_rows)
    total_labeled = 0
    total_unlabeled = 0
    total_positives = 0
    total_negatives = 0
    total_unsure = 0
    total_passes_s1 = 0

    for vid in sorted(by_video.keys()):
        v_rows = by_video[vid]
        cnt = len(v_rows)
        labeled = sum(1 for r in v_rows if str(r.get("label", "")).strip() != "")
        unlabeled = cnt - labeled
        l_pos = sum(1 for r in v_rows if str(r.get("label", "")).strip() == "1")
        l_neg = sum(1 for r in v_rows if str(r.get("label", "")).strip() == "0")
        l_uns = sum(1 for r in v_rows if str(r.get("label", "")).strip() == "-1")
        passes = sum(1 for r in v_rows if str(r.get("passes_stage1_default", "")).strip() == "1")

        total_labeled += labeled
        total_unlabeled += unlabeled
        total_positives += l_pos
        total_negatives += l_neg
        total_unsure += l_uns
        total_passes_s1 += passes

        print(
            f"{vid:<18} {cnt:<7} {labeled:<9} {unlabeled:<11} "
            f"{l_pos:<9} {l_neg:<9} {l_uns:<10} {passes}"
        )

    print("-" * 88)
    print("Summary Pair Totals (excluding dummy rows):")
    print(f"  Total rows:              {total_rows}")
    print(f"  Labeled rows:            {total_labeled}")
    print(f"  Unlabeled rows:          {total_unlabeled}")
    print(f"  Label 1 (positives):     {total_positives}")
    print(f"  Label 0 (negatives):     {total_negatives}")
    print(f"  Label -1 (unclear):      {total_unsure}")
    print(f"  Passes Stage 1 Default:  {total_passes_s1}")
    print("=" * 88 + "\n")

    # Groups Progress
    print("Retake Groups Progress (--stats)")
    print("=" * 88)
    if not groups_csv_path.exists():
        print(f"No {groups_csv_path.name} found. Run --build-groups to generate groups.")
        print("=" * 88 + "\n")
        return

    groups_header, group_rows = read_groups_csv(groups_csv_path)
    # Exclude synthetic and dummy rows
    valid_groups = [r for r in group_rows if not _is_synthetic_video(str(r.get("video_id", "")))]
    if not valid_groups:
        print(f"No groups found in {groups_csv_path.name}.")
        print("=" * 88 + "\n")
        return

    by_video_groups: dict[str, list[dict[str, Any]]] = {}
    for r in valid_groups:
        vid = str(r.get("video_id", ""))
        by_video_groups.setdefault(vid, []).append(r)

    print(
        f"{'Video ID':<18} {'Groups':<8} {'Keeper':<10} {'Keep None':<11} "
        f"{'Unlabeled':<11} {'Sz2':<5} {'Sz3':<5} {'Sz4-6':<7} {'Sz7+'}"
    )
    print("-" * 88)

    tot_groups = len(valid_groups)
    tot_keeper = 0
    tot_none = 0
    tot_unlabeled = 0
    tot_s2 = 0
    tot_s3 = 0
    tot_s46 = 0
    tot_s7p = 0
    tot_exempt = 0

    for vid in sorted(by_video_groups.keys()):
        v_rows = by_video_groups[vid]
        cnt = len(v_rows)
        keeper = sum(1 for r in v_rows if str(r.get("keeper_unit_ids", "")).strip() != "")
        none_cnt = sum(1 for r in v_rows if str(r.get("keep_none", "")).strip() == "1")
        unlabeled = sum(
            1 for r in v_rows
            if str(r.get("keeper_unit_ids", "")).strip() == "" and str(r.get("keep_none", "")).strip() != "1"
        )
        def _n(r: dict[str, Any]) -> int:
            try:
                return int(r.get("n_units", len(str(r.get("unit_ids", "")).split("|"))  ))
            except (ValueError, TypeError):
                return len(str(r.get("unit_ids", "")).split("|"))
        s2 = sum(1 for r in v_rows if _n(r) == 2)
        s3 = sum(1 for r in v_rows if _n(r) == 3)
        s46 = sum(1 for r in v_rows if 4 <= _n(r) <= 6)
        s7p = sum(1 for r in v_rows if _n(r) >= 7)
        exempt = sum(
            int(r.get("n_exempt_pairs", 0) or 0) for r in v_rows
        )

        tot_keeper += keeper
        tot_none += none_cnt
        tot_unlabeled += unlabeled
        tot_s2 += s2
        tot_s3 += s3
        tot_s46 += s46
        tot_s7p += s7p
        tot_exempt += exempt

        print(
            f"{vid:<18} {cnt:<8} {keeper:<10} {none_cnt:<11} "
            f"{unlabeled:<11} {s2:<5} {s3:<5} {s46:<7} {s7p}"
        )

    print("-" * 88)
    print("Summary Group Totals (excluding synthetic/dummy rows):")
    print(f"  Total groups:            {tot_groups}")
    print(f"  Keeper labeled:          {tot_keeper}")
    print(f"  Keep none:               {tot_none}")
    print(f"  Unlabeled groups:        {tot_unlabeled}")
    print(f"  Groups of size 2:        {tot_s2}")
    print(f"  Groups of size 3:        {tot_s3}")
    print(f"  Groups of size 4-6:      {tot_s46}")
    print(f"  Groups of size 7+:       {tot_s7p}")
    print(f"  Total same_take exempt pairs: {tot_exempt}")
    print("=" * 88 + "\n")


class _GroupsUnionFind:
    """Disjoint set data structure for identifying retake unit groups."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y

    def get_components(self) -> list[set[str]]:
        comp_map: dict[str, set[str]] = {}
        for elem in self.parent:
            root = self.find(elem)
            comp_map.setdefault(root, set()).add(elem)
        return list(comp_map.values())


def _compute_group_heuristics(
    sorted_uids: list[str],
    unit_map: dict[str, dict[str, Any]],
    group_pairs: list[dict[str, Any]],
) -> tuple[str, str, str, str]:
    """
    Computes keeper unit predictions using existing pipeline heuristics and cutlist strategies:
    - pipeline_keeper_unit_id: Step 05 _determine_likely_final_unit_id (or "" if None)
    - heuristic_later: cutlist select_take('keep_later')
    - heuristic_earlier: cutlist select_take('keep_earlier')
    - heuristic_confidence: cutlist select_take('keep_higher_confidence')
    """
    class_summary: dict[str, int] = {}
    for p in group_pairs:
        cls_name = p.get("classification", "")
        if cls_name:
            class_summary[cls_name] = class_summary.get(cls_name, 0) + 1

    # 1. Pipeline keeper heuristic (Step 05)
    pipeline_keeper = _determine_likely_final_unit_id(
        unit_ids=sorted_uids,
        unit_map=unit_map,
        pairs=group_pairs,
        classification_summary=class_summary,
    )
    pipeline_keeper_str = str(pipeline_keeper) if pipeline_keeper else ""

    # Helper to map select_take output ('a', 'b', or direct unit_id) to actual unit_id
    def _map_take(take_res: Any) -> str:
        if isinstance(take_res, tuple):
            take_val = take_res[0]
        else:
            take_val = take_res
        if take_val == "b":
            return sorted_uids[-1]
        elif take_val == "a":
            return sorted_uids[0]
        elif str(take_val) in sorted_uids:
            return str(take_val)
        return sorted_uids[-1]

    # If no candidate pairs exist, construct pair-like dicts between first and last for cutlist select_take
    eval_pairs = group_pairs
    if not eval_pairs:
        u_first = unit_map.get(
            sorted_uids[0],
            {"unit_id": sorted_uids[0], "start": 0.0, "end": 0.0, "confidence": 0.0},
        )
        u_last = unit_map.get(
            sorted_uids[-1],
            {"unit_id": sorted_uids[-1], "start": 1.0, "end": 1.0, "confidence": 0.0},
        )
        eval_pairs = [{"unit_a": u_first, "unit_b": u_last, "classification": "likely_retake"}]

    later_res, _ = select_take(eval_pairs, strategy="keep_later")
    earlier_res, _ = select_take(eval_pairs, strategy="keep_earlier")
    conf_res, _ = select_take(eval_pairs, strategy="keep_higher_confidence")

    heuristic_later = _map_take(later_res)
    heuristic_earlier = _map_take(earlier_res)
    heuristic_confidence = _map_take(conf_res)

    return pipeline_keeper_str, heuristic_later, heuristic_earlier, heuristic_confidence


def build_groups(
    pairs_csv_path: Path = DEFAULT_CSV_PATH,
    groups_csv_path: Path = DEFAULT_GROUPS_CSV_PATH,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    backups_dir: Path = DEFAULT_BACKUPS_DIR,
    video_id: str | None = None,
    dry_run: bool = False,
) -> int:
    """
    Builds or updates data/training/groups_v1.csv from labeled pairs in pairs_v1.csv.
    Groups are only built for natural (non-synthetic) rows.
    Unit IDs are always derived from pair_id to avoid Excel corruption of unit_a_id/unit_b_id.
    Identifies retake groups via connected components of label=1 pairs (union-find).
    Maintains group stability:
    - Same unit set: preserves hand labels, updates pipeline guesses.
    - Changed/split/merged unit set: appends new row, marks old row STALE in notes.
    - Never changes existing group_ids.
    If groups_v1.csv has the legacy header, backs it up and migrates it first.
    """
    header, existing_pairs = read_pairs_csv(pairs_csv_path)

    # Exclude synthetic (is_synthetic==1, synth_, dummy_) rows — groups are for natural data only
    def _is_natural(r: dict[str, Any]) -> bool:
        vid = str(r.get("video_id", ""))
        synth = str(r.get("is_synthetic", "0")).strip()
        return synth != "1" and not _is_synthetic_video(vid)

    valid_pairs = [r for r in existing_pairs if _is_natural(r)]

    # Detect and warn if unit_a_id/unit_b_id differ from pair_id-derived values
    mismatch_count = 0
    for p in valid_pairs:
        vid = str(p.get("video_id", "")).strip()
        pid = str(p.get("pair_id", "")).strip()
        derived_a, derived_b = _unit_ids_from_pair_id(pid, vid)
        raw_a = str(p.get("unit_a_id", "")).strip()
        raw_b = str(p.get("unit_b_id", "")).strip()
        if (derived_a and raw_a and raw_a != derived_a) or (derived_b and raw_b and raw_b != derived_b):
            mismatch_count += 1
    if mismatch_count:
        print(
            f"Warning: {mismatch_count} pair(s) have unit_a_id/unit_b_id that differ from pair_id-derived values "
            "(likely Excel corruption). Using pair_id as the source of truth."
        )

    # Group pairs by video_id
    pairs_by_video: dict[str, list[dict[str, Any]]] = {}
    for p in valid_pairs:
        vid = str(p.get("video_id", "")).strip()
        if vid:
            pairs_by_video.setdefault(vid, []).append(p)

    target_videos = [video_id] if video_id else sorted(pairs_by_video.keys())

    # Read existing groups (migrates legacy header if needed)
    groups_header, existing_groups = read_groups_csv(groups_csv_path)

    # If a migration happened (header was updated in memory), write it back now before proceeding
    if groups_csv_path.exists() and existing_groups:
        with open(groups_csv_path, "r", encoding="utf-8-sig", newline="") as _f:
            _raw_hdr = next(csv.reader(_f), None)
        if _raw_hdr and _raw_hdr == GROUPS_CSV_COLUMNS_LEGACY:
            backup_groups_csv(groups_csv_path, backups_dir)
            atomic_write_groups_csv(groups_csv_path, existing_groups, fieldnames=GROUPS_CSV_COLUMNS)
            print("[Migration] Legacy groups_v1.csv backed up and migrated to new schema.")

    # Video existing groups map
    current_video_existing = {
        vid: [g for g in existing_groups if g.get("video_id") == vid]
        for vid in target_videos
    }

    all_updated_or_new_rows: list[dict[str, Any]] = []
    total_added = 0
    total_updated = 0
    total_stale = 0

    for vid in target_videos:
        video_pairs = pairs_by_video.get(vid, [])
        units_cache_path = processed_dir / vid / "units.json"
        if not units_cache_path.exists():
            print(f"Skipping video '{vid}': no cache found at {units_cache_path}.")
            all_updated_or_new_rows.extend(current_video_existing.get(vid, []))
            continue

        try:
            units_data = json.loads(units_cache_path.read_text(encoding="utf-8"))
            unit_map = {u["unit_id"]: u for u in units_data}
        except Exception as e:
            logger.warning(f"Failed to load units.json for {vid}: {e}")
            all_updated_or_new_rows.extend(current_video_existing.get(vid, []))
            continue

        # Load candidates.json if available
        candidates_cache_path = processed_dir / vid / "candidates.json"
        candidates_data: list[dict[str, Any]] = []
        if candidates_cache_path.exists():
            try:
                candidates_data = json.loads(candidates_cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Build union-find on label=1 pairs, using unit ids derived from pair_id
        uf = _GroupsUnionFind()
        label1_pairs: list[dict[str, Any]] = []  # pairs with label==1
        for p in video_pairs:
            pid = str(p.get("pair_id", "")).strip()
            if str(p.get("label", "")).strip() == "1":
                u_a, u_b = _unit_ids_from_pair_id(pid, vid)
                if u_a and u_b:
                    uf.union(u_a, u_b)
                    # Store derived ids alongside pair for downstream use
                    p_copy = dict(p)
                    p_copy["_derived_a"] = u_a
                    p_copy["_derived_b"] = u_b
                    label1_pairs.append(p_copy)

        components = uf.get_components()
        # Singletons are not groups
        group_components = [c for c in components if len(c) >= 2]

        # Count exempt pairs (label=0, notes contain 'same_take') per component
        exempt_pairs_by_comp: dict[frozenset, int] = {}
        for p in video_pairs:
            if str(p.get("label", "")).strip() == "0" and "same_take" in str(p.get("notes", "")).lower():
                pid = str(p.get("pair_id", "")).strip()
                u_a, u_b = _unit_ids_from_pair_id(pid, vid)
                for comp in group_components:
                    if u_a in comp and u_b in comp:
                        key = frozenset(comp)
                        exempt_pairs_by_comp[key] = exempt_pairs_by_comp.get(key, 0) + 1

        # Prepare new group representations
        new_groups: list[dict[str, Any]] = []
        for comp in group_components:
            sorted_units = sorted(
                [unit_map[u] for u in comp if u in unit_map],
                key=lambda x: (
                    float(x.get("start", 0.0)),
                    float(x.get("end", 0.0)),
                    str(x.get("unit_id", "")),
                ),
            )
            # If any unit in comp was not in unit_map, add fallback
            found_ids = {u.get("unit_id") for u in sorted_units}
            for u in comp:
                if u not in found_ids:
                    sorted_units.append(
                        {"unit_id": u, "start": 0.0, "end": 0.0, "raw_text": "", "confidence": 0.0}
                    )

            sorted_uids = [u["unit_id"] for u in sorted_units]
            n_units = len(sorted_uids)
            unit_ids_str = "|".join(sorted_uids)

            # Build texts from pair rows (text_a/text_b) — no cache needed
            text_lookup: dict[str, str] = {}
            for p in video_pairs:
                pid = str(p.get("pair_id", "")).strip()
                u_a, u_b = _unit_ids_from_pair_id(pid, vid)
                if u_a in comp:
                    text_lookup.setdefault(u_a, str(p.get("text_a", "")))
                if u_b in comp:
                    text_lookup.setdefault(u_b, str(p.get("text_b", "")))
            # Fallback to unit_map raw_text
            for uid in sorted_uids:
                if uid not in text_lookup:
                    text_lookup[uid] = unit_map.get(uid, {}).get("raw_text", "")

            texts_str = " || ".join(
                f"[{i}] {text_lookup.get(uid, '')}"
                for i, uid in enumerate(sorted_uids, 1)
            )
            starts_str = "|".join(
                str(round(float(unit_map.get(uid, {}).get("start", 0.0)), 3))
                for uid in sorted_uids
            )
            ends_str = "|".join(
                str(round(float(unit_map.get(uid, {}).get("end", 0.0)), 3))
                for uid in sorted_uids
            )

            # Relevant label=1 pair_ids (use derived ids for matching)
            comp_pair_ids = sorted(
                str(p.get("pair_id"))
                for p in label1_pairs
                if p.get("_derived_a") in comp and p.get("_derived_b") in comp
            )
            n_pos_pairs = len(comp_pair_ids)
            pair_ids_str = "|".join(comp_pair_ids)
            n_exempt = exempt_pairs_by_comp.get(frozenset(comp), 0)

            # Find matching candidate objects from candidates_data or video_pairs
            group_pairs: list[dict[str, Any]] = []
            comp_set = set(sorted_uids)
            for c in candidates_data:
                c_a = c.get("unit_a", {}).get("unit_id")
                c_b = c.get("unit_b", {}).get("unit_id")
                if c_a in comp_set and c_b in comp_set:
                    group_pairs.append(c)

            if not group_pairs:
                for p in video_pairs:
                    pid = str(p.get("pair_id", "")).strip()
                    p_a, p_b = _unit_ids_from_pair_id(pid, vid)
                    if p_a in comp_set and p_b in comp_set:
                        group_pairs.append({
                            "unit_a": unit_map.get(
                                p_a,
                                {
                                    "unit_id": p_a,
                                    "start": float(p.get("start_a", 0.0)),
                                    "end": float(p.get("end_a", 0.0)),
                                    "confidence": 1.0,
                                },
                            ),
                            "unit_b": unit_map.get(
                                p_b,
                                {
                                    "unit_id": p_b,
                                    "start": float(p.get("start_b", 0.0)),
                                    "end": float(p.get("end_b", 0.0)),
                                    "confidence": 1.0,
                                },
                            ),
                            "classification": p.get("pipeline_class", "likely_retake"),
                        })

            pk, hl, he, hc = _compute_group_heuristics(sorted_uids, unit_map, group_pairs)

            new_groups.append({
                "unit_set": comp_set,
                "n_units": n_units,
                "unit_ids": unit_ids_str,
                "texts": texts_str,
                "starts": starts_str,
                "ends": ends_str,
                "pair_ids": pair_ids_str,
                "n_pos_pairs": n_pos_pairs,
                "n_exempt_pairs": n_exempt,
                "pipeline_keeper_unit_id": pk,
                "heuristic_later": hl,
                "heuristic_earlier": he,
                "heuristic_confidence": hc,
            })

        # Match new groups against existing rows for this video
        vid_existing = current_video_existing.get(vid, [])
        max_n = 0
        for r in vid_existing:
            gid = str(r.get("group_id", ""))
            if f"{vid}__g" in gid:
                suffix = gid.split(f"{vid}__g")[-1]
                if suffix.isdigit():
                    max_n = max(max_n, int(suffix))

        matched_new_indices: set[int] = set()
        video_final_rows: list[dict[str, Any]] = []

        for ex_row in vid_existing:
            ex_uids = set(str(ex_row.get("unit_ids", "")).split("|"))
            match_idx = None
            for idx, ng in enumerate(new_groups):
                if idx not in matched_new_indices and ng["unit_set"] == ex_uids:
                    match_idx = idx
                    break

            if match_idx is not None:
                # Same unit set: preserve existing row & hand labels, update pipeline guesses
                matched_new_indices.add(match_idx)
                ng = new_groups[match_idx]
                ex_row["n_units"] = ng["n_units"]
                ex_row["unit_ids"] = ng["unit_ids"]
                ex_row["texts"] = ng["texts"]
                ex_row["starts"] = ng["starts"]
                ex_row["ends"] = ng["ends"]
                ex_row["pipeline_keeper_unit_id"] = ng["pipeline_keeper_unit_id"]
                ex_row["heuristic_later"] = ng["heuristic_later"]
                ex_row["heuristic_earlier"] = ng["heuristic_earlier"]
                ex_row["heuristic_confidence"] = ng["heuristic_confidence"]
                ex_row["pair_ids"] = ng["pair_ids"]
                ex_row["n_pos_pairs"] = ng["n_pos_pairs"]
                ex_row["n_exempt_pairs"] = ng["n_exempt_pairs"]
                ex_row["feature_version"] = FEATURE_VERSION

                # Check if any keeper_unit_ids are no longer in unit_ids
                keeper_ids_str = _normalize_keeper_unit_ids(str(ex_row.get("keeper_unit_ids", "")).strip())
                ex_row["keeper_unit_ids"] = keeper_ids_str
                if keeper_ids_str:
                    unit_set = ng["unit_set"]
                    keeper_ids = [k.strip() for k in keeper_ids_str.split("|") if k.strip()]
                    bad_keepers = [k for k in keeper_ids if k not in unit_set]
                    curr_notes = str(ex_row.get("notes", "")).strip()
                    if bad_keepers:
                        stale_msg = f"STALE: keeper not in unit_ids: {bad_keepers}"
                        if stale_msg not in curr_notes:
                            ex_row["notes"] = f"{curr_notes}; {stale_msg}" if curr_notes else stale_msg
                            print(f"Warning: Group {ex_row['group_id']}: {stale_msg}")
                            total_stale += 1
                    else:
                        # Clear false stale note if previously added due to unnormalized time format
                        if "STALE: keeper not in unit_ids" in curr_notes:
                            cleaned_notes = "; ".join(
                                p.strip() for p in curr_notes.split(";")
                                if "STALE: keeper not in unit_ids" not in p
                            )
                            ex_row["notes"] = cleaned_notes

                video_final_rows.append(ex_row)
                total_updated += 1
            else:
                # Existing group unit set no longer formed by pair labels
                stale_msg = "STALE: unit set no longer matches pair labels"
                curr_notes = str(ex_row.get("notes", "")).strip()
                if stale_msg not in curr_notes:
                    ex_row["notes"] = f"{curr_notes}; {stale_msg}" if curr_notes else stale_msg
                    print(
                        f"Warning: Group {ex_row['group_id']} marked STALE: unit set no longer matches pair labels"
                    )
                    total_stale += 1
                video_final_rows.append(ex_row)

        # Unmatched new groups get appended with new group_ids
        unmatched_new = [ng for idx, ng in enumerate(new_groups) if idx not in matched_new_indices]
        unmatched_new.sort(
            key=lambda x: float(x["starts"].split("|")[0]) if x["starts"] else 0.0
        )

        for ng in unmatched_new:
            max_n += 1
            new_gid = f"{vid}__g{max_n}"
            new_row = {
                "group_id": new_gid,
                "video_id": vid,
                "n_units": ng["n_units"],
                "unit_ids": ng["unit_ids"],
                "texts": ng["texts"],
                "starts": ng["starts"],
                "ends": ng["ends"],
                "keeper_unit_ids": "",
                "keep_none": "",
                "notes": "",
                "pipeline_keeper_unit_id": ng["pipeline_keeper_unit_id"],
                "heuristic_later": ng["heuristic_later"],
                "heuristic_earlier": ng["heuristic_earlier"],
                "heuristic_confidence": ng["heuristic_confidence"],
                "pair_ids": ng["pair_ids"],
                "n_pos_pairs": ng["n_pos_pairs"],
                "n_exempt_pairs": ng["n_exempt_pairs"],
                "feature_version": FEATURE_VERSION,
            }
            video_final_rows.append(new_row)
            total_added += 1

        all_updated_or_new_rows.extend(video_final_rows)

    # Reassemble final rows maintaining original file positions for existing rows
    updated_by_gid = {r["group_id"]: r for r in all_updated_or_new_rows}
    existing_gids = {r["group_id"] for r in existing_groups}
    brand_new_rows = [r for r in all_updated_or_new_rows if r["group_id"] not in existing_gids]

    final_rows: list[dict[str, Any]] = []
    for ex in existing_groups:
        gid = ex.get("group_id")
        if gid in updated_by_gid:
            final_rows.append(updated_by_gid[gid])
        else:
            final_rows.append(ex)
    final_rows.extend(brand_new_rows)

    if dry_run:
        print(
            f"\n[DRY RUN] Would update groups_v1.csv: {total_added} added, "
            f"{total_updated} updated, {total_stale} stale."
        )
        for r in all_updated_or_new_rows:
            print(
                f"  [DRY RUN SAMPLE] {r['group_id']} (units={r['unit_ids']}, keeper={r['pipeline_keeper_unit_id']})"
            )
        print("[DRY RUN] No files written.")
        return 0

    backup_groups_csv(groups_csv_path, backups_dir)
    atomic_write_groups_csv(groups_csv_path, final_rows, fieldnames=GROUPS_CSV_COLUMNS)

    print(
        f"Groups successfully updated: {total_added} added, "
        f"{total_updated} preserved/updated, {total_stale} marked stale."
    )
    return 0


def _bfs_chain(
    unit_a: str,
    unit_b: str,
    label1_adjacency: dict[str, list[tuple[str, str]]],
) -> list[str] | None:
    """Returns the shortest label-1 chain (list of pair_ids) linking unit_a to unit_b, or None."""
    from collections import deque
    if unit_a == unit_b:
        return []
    visited: set[str] = {unit_a}
    queue: deque[tuple[str, list[str]]] = deque([(unit_a, [])])
    while queue:
        node, path = queue.popleft()
        for neighbor, pair_id in label1_adjacency.get(node, []):
            if neighbor not in visited:
                new_path = path + [pair_id]
                if neighbor == unit_b:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))
    return None


def check_groups(
    groups_csv_path: Path = DEFAULT_GROUPS_CSV_PATH,
    pairs_csv_path: Path = DEFAULT_CSV_PATH,
) -> int:
    """
    Validates hand labels and consistency in groups_v1.csv (read-only):
    - Real contradictions: label=0 pairs inside a group whose notes do NOT contain 'same_take'
      (shows the BFS chain linking them). -1 and same_take 0-pairs are never flagged.
    - keeper_unit_ids containing ids not in unit_ids, or duplicates
    - both keeper_unit_ids and keep_none = 1
    - STALE rows
    - unlabeled pairs (blank label) between two units of the same group
    - groups with more than 8 units
    Exits 0 even if issues found; prints total at end.
    """
    if not groups_csv_path.exists():
        print(f"Error: {groups_csv_path} does not exist. Run --build-groups first.")
        return 0

    header, groups = read_groups_csv(groups_csv_path)
    if not groups:
        print("No groups found in groups_v1.csv.")
        return 0

    _, pairs = read_pairs_csv(pairs_csv_path)
    # Only natural rows
    valid_pairs = [p for p in pairs if not _is_synthetic_video(str(p.get("video_id", ""))) and str(p.get("is_synthetic", "0")).strip() != "1"]

    # Build per-video pair lookups
    pairs_by_video: dict[str, list[dict[str, Any]]] = {}
    for p in valid_pairs:
        vid = str(p.get("video_id", "")).strip()
        if vid:
            pairs_by_video.setdefault(vid, []).append(p)

    # Build adjacency for BFS (label=1 edges)
    label1_adj_by_video: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for vid, vp in pairs_by_video.items():
        adj: dict[str, list[tuple[str, str]]] = {}
        for p in vp:
            if str(p.get("label", "")).strip() == "1":
                pid = str(p.get("pair_id", "")).strip()
                u_a, u_b = _unit_ids_from_pair_id(pid, vid)
                if u_a and u_b:
                    adj.setdefault(u_a, []).append((u_b, pid))
                    adj.setdefault(u_b, []).append((u_a, pid))
        label1_adj_by_video[vid] = adj

    issues: list[str] = []

    for g in groups:
        gid = str(g.get("group_id", "")).strip()
        vid = str(g.get("video_id", "")).strip()
        uids = [u.strip() for u in str(g.get("unit_ids", "")).split("|") if u.strip()]
        u_set = set(uids)
        keeper_ids_str = str(g.get("keeper_unit_ids", "")).strip()
        keep_none = str(g.get("keep_none", "")).strip()
        notes = str(g.get("notes", "")).strip()

        # 1. Group with fewer than 2 units
        if len(uids) < 2:
            issues.append(f"Group '{gid}': has fewer than 2 units ({len(uids)} found).")

        # 2. keeper_unit_ids validation
        keeper_ids_str = _normalize_keeper_unit_ids(keeper_ids_str)
        if keeper_ids_str:
            keeper_ids = [k.strip() for k in keeper_ids_str.split("|") if k.strip()]
            # Duplicates
            seen_k: set[str] = set()
            dups = [k for k in keeper_ids if k in seen_k or seen_k.add(k)]
            if dups:
                issues.append(f"Group '{gid}': duplicate ids in keeper_unit_ids: {dups}")
            # Unknown ids
            bad = [k for k in keeper_ids if k not in u_set]
            if bad:
                issues.append(f"Group '{gid}': keeper_unit_ids {bad} not in unit_ids: {uids}")

        # 3. Both keeper_unit_ids and keep_none = 1
        if keeper_ids_str and keep_none == "1":
            issues.append(f"Group '{gid}': both keeper_unit_ids and keep_none=1 are set.")

        # 4. STALE row
        if "STALE:" in notes or notes.startswith("STALE"):
            issues.append(f"Group '{gid}' is STALE: {notes}")
            continue

        # 5. Large groups
        if len(uids) > 8:
            texts_short = str(g.get("texts", ""))[:80]
            issues.append(f"Group '{gid}': has {len(uids)} units (> 8): {texts_short}...")

        # 6. Check against pairs in this video
        vid_pairs = pairs_by_video.get(vid, [])
        adj = label1_adj_by_video.get(vid, {})
        for p in vid_pairs:
            p_label = str(p.get("label", "")).strip()
            p_notes = str(p.get("notes", "")).lower()
            pid = str(p.get("pair_id", "")).strip()
            u_a, u_b = _unit_ids_from_pair_id(pid, vid)

            # Skip -1 pairs entirely
            if p_label == "-1":
                continue

            # Real contradiction: label=0, both units in group, notes do NOT contain 'same_take'
            if p_label == "0" and u_a in u_set and u_b in u_set:
                if "same_take" not in p_notes:
                    chain = _bfs_chain(u_a, u_b, adj)
                    chain_str = " -> ".join(chain) if chain else "no direct chain found"
                    text_a = str(p.get("text_a", ""))[:60]
                    text_b = str(p.get("text_b", ""))[:60]
                    issues.append(
                        f"Group '{gid}': real contradiction '{pid}' (label=0, not same_take) "
                        f"between '{text_a}' and '{text_b}'. "
                        f"Label-1 chain: [{chain_str}]"
                    )

            # Unlabeled pair (blank) between two group units
            if p_label == "" and u_a in u_set and u_b in u_set:
                issues.append(
                    f"Group '{gid}': unlabeled internal pair '{pid}' ({u_a} ~ {u_b})."
                )

    total = len(issues)
    if issues:
        print("\nGroup Validation Issues (--check-groups):")
        print("=" * 88)
        for issue in issues:
            print(f"  [!] {issue}")
        print("=" * 88)
    else:
        print("No group validation issues found. All groups are valid.")
    print(f"Total issues found: {total}\n")
    return 0


def eval_keepers(groups_csv_path: Path = DEFAULT_GROUPS_CSV_PATH) -> int:
    """
    Evaluates keeper heuristics against hand labels (read-only).
    A strategy is correct if its chosen unit is IN the keeper_unit_ids set.
    Breaks down by: All, Size 2, Size 3+. Prints per-video counts.
    Excludes keep_none=1 groups and synthetic/dummy rows.
    """
    if not groups_csv_path.exists():
        print(f"No groups file found at {groups_csv_path}. Run --build-groups first.")
        return 0

    header, groups = read_groups_csv(groups_csv_path)
    # Exclude synthetic and dummy rows
    valid_groups = [g for g in groups if not _is_synthetic_video(str(g.get("video_id", "")))]

    labeled_groups = [
        g
        for g in valid_groups
        if str(g.get("keeper_unit_ids", "")).strip() != "" or str(g.get("keep_none", "")).strip() == "1"
    ]

    if not labeled_groups:
        print(
            "No labeled groups found in groups_v1.csv. "
            "Label keeper_unit_ids or keep_none in data/training/groups_v1.csv first."
        )
        return 0

    keep_none_groups = [g for g in labeled_groups if str(g.get("keep_none", "")).strip() == "1"]
    eval_groups = [
        g
        for g in labeled_groups
        if str(g.get("keep_none", "")).strip() != "1" and str(g.get("keeper_unit_ids", "")).strip() != ""
    ]

    print("\nKeeper Heuristics Evaluation (--eval-keepers)")
    print("=" * 88)
    print(f"Total hand-labeled groups:              {len(labeled_groups)}")
    print(f"Excluded keep_none=1 groups:            {len(keep_none_groups)}")
    print(f"Groups evaluated for strategy accuracy: {len(eval_groups)}")

    if not eval_groups:
        print("All labeled groups have keep_none=1; no keeper units to evaluate strategy accuracy.")
        print("=" * 88 + "\n")
        return 0

    def _keeper_set(g: dict[str, Any]) -> set[str]:
        return {_normalize_unit_id_str(k) for k in str(g.get("keeper_unit_ids", "")).split("|") if k.strip()}

    def _is_correct(strategy_col: str, g: dict[str, Any]) -> bool:
        chosen = str(g.get(strategy_col, "")).strip()
        return bool(chosen) and chosen in _keeper_set(g)

    # Partition by size
    size_2 = [
        g
        for g in eval_groups
        if int(g.get("n_units", len(str(g.get("unit_ids", "")).split("|")))) == 2
    ]
    size_3p = [
        g
        for g in eval_groups
        if int(g.get("n_units", len(str(g.get("unit_ids", "")).split("|")))) >= 3
    ]

    subsets = [
        ("All", eval_groups),
        ("Size 2", size_2),
        ("Size 3+", size_3p),
    ]

    strategies = [
        ("pipeline_keeper_unit_id", "Pipeline (Step 05)"),
        ("heuristic_later", "Keep Later"),
        ("heuristic_earlier", "Keep Earlier"),
        ("heuristic_confidence", "Keep Higher Conf"),
    ]

    print("-" * 88)
    header_line = f"{'Strategy':<26}"
    for name, subset in subsets:
        header_line += f" | {name + f' (N={len(subset)})':<18}"
    print(header_line)
    print("-" * 88)

    for col_key, strat_label in strategies:
        row_str = f"{strat_label:<26}"
        for _, subset in subsets:
            if not subset:
                row_str += f" | {'N/A (N=0)':<18}"
                continue

            if col_key == "pipeline_keeper_unit_id":
                decided = [
                    g for g in subset if str(g.get("pipeline_keeper_unit_id", "")).strip() != ""
                ]
                deferred = len(subset) - len(decided)
                if not decided:
                    row_str += f" | {f'0/{len(subset)} (0.0%) [def {deferred}]':<18}"
                else:
                    correct = sum(1 for g in decided if _is_correct(col_key, g))
                    acc = (correct / len(decided)) * 100.0
                    row_str += f" | {f'{correct}/{len(decided)} ({acc:.1f}%) [def {deferred}]':<18}"
            else:
                correct = sum(1 for g in subset if _is_correct(col_key, g))
                acc = (correct / len(subset)) * 100.0
                row_str += f" | {f'{correct}/{len(subset)} ({acc:.1f}%)':<18}"
        print(row_str)

    print("=" * 88)

    # Per-video breakdown
    by_vid: dict[str, list[dict[str, Any]]] = {}
    for g in eval_groups:
        vid = str(g.get("video_id", ""))
        by_vid.setdefault(vid, []).append(g)
    if by_vid:
        print("\nPer-video labeled groups:")
        for vid in sorted(by_vid):
            vg = by_vid[vid]
            print(f"  {vid}: {len(vg)} group(s)")

    if keep_none_groups:
        print(
            f"\nNote: {len(keep_none_groups)} group(s) with keep_none=1 were excluded from strategy accuracy."
        )
    print(
        "Pipeline Step 05 heuristic reports accuracy only on groups where it decided; [def X] shows deferred groups.\n"
    )
    return 0


def process_videos(
    videos_dir: Path = DEFAULT_VIDEOS_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    backups_dir: Path = DEFAULT_BACKUPS_DIR,
    csv_path: Path = DEFAULT_CSV_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    single_video: str | None = None,
    min_lexical: float = DEFAULT_MIN_LEXICAL,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    source_type: str | None = None,
    whisper_model: str = "large-v2",
    refresh_video_id: str | None = None,
    dry_run: bool = False,
) -> int:
    """
    Main execution workflow for populate_pairs:
    1. Validates inputs and duplicate stems.
    2. Handles refresh mode or standard ingestion mode.
    3. Guarantees atomic write and backup safety.
    """
    header, existing_rows = read_pairs_csv(csv_path)
    manifest = load_manifest(manifest_path)

    # --------------------------------------------------------------------------
    # Case A: Refresh mode
    # --------------------------------------------------------------------------
    if refresh_video_id:
        if refresh_video_id.startswith("dummy_"):
            sys.stderr.write(
                f"Error: '{refresh_video_id}' is a dummy bootstrap video and cannot be modified.\n"
            )
            return 1

        # Locate cached transcript or video file
        video_proc_dir = processed_dir / refresh_video_id
        transcript_cache = video_proc_dir / "transcript.json"

        target_video_file: Path | None = None
        if not transcript_cache.exists():
            # Search in videos_dir for matching stem
            if videos_dir.exists():
                for vf in videos_dir.iterdir():
                    if vf.is_file() and vf.suffix.lower() in VALID_VIDEO_EXTENSIONS:
                        if get_video_id(vf) == refresh_video_id:
                            target_video_file = vf
                            break
            if target_video_file is None:
                sys.stderr.write(
                    f"Error: Neither transcript cache {transcript_cache} nor video file found for refresh '{refresh_video_id}'.\n"
                )
                return 1
        else:
            target_video_file = transcript_cache

        print(f"Refreshing features for video '{refresh_video_id}'...")
        _, units, stage2_candidates = run_pipeline_for_video(
            video_path=target_video_file,
            video_id=refresh_video_id,
            processed_dir=processed_dir,
            min_lexical=min_lexical,
            window_seconds=window_seconds,
            reuse_transcript=True,
            whisper_model=whisper_model,
            dry_run=dry_run,
        )

        # Determine source_type from existing rows or manifest or filename
        derived_source_type = source_type
        if not derived_source_type:
            for r in existing_rows:
                if r.get("video_id") == refresh_video_id and r.get("source_type"):
                    derived_source_type = r.get("source_type")
                    break
        if not derived_source_type:
            derived_source_type = determine_source_type(refresh_video_id, target_video_file.name)

        new_pairs_by_id: dict[str, dict[str, Any]] = {}
        for c in stage2_candidates:
            row = candidate_to_csv_row(c, refresh_video_id, derived_source_type)
            new_pairs_by_id[row["pair_id"]] = row

        updated_count = 0
        untouched_count = 0
        appended_count = 0
        seen_in_existing: set[str] = set()

        refreshed_rows: list[dict[str, Any]] = []
        for r in existing_rows:
            vid = str(r.get("video_id", ""))
            pid = str(r.get("pair_id", ""))

            if vid != refresh_video_id:
                refreshed_rows.append(r)
                continue

            if pid in new_pairs_by_id:
                new_row = new_pairs_by_id[pid]
                # Recompute only REFRESH_FEATURE_COLUMNS; leave labels, notes, text, etc. untouched
                for col in REFRESH_FEATURE_COLUMNS:
                    r[col] = new_row[col]
                refreshed_rows.append(r)
                updated_count += 1
                seen_in_existing.add(pid)
            else:
                # Pair in file that new run no longer generates is left untouched
                refreshed_rows.append(r)
                untouched_count += 1

        # Pairs that the new run generates but the file lacks are appended
        appended_rows: list[dict[str, Any]] = []
        for pid, new_row in new_pairs_by_id.items():
            if pid not in seen_in_existing:
                appended_rows.append(new_row)
                appended_count += 1

        appended_rows.sort(
            key=lambda x: (
                float(x.get("start_a", 0.0)),
                float(x.get("start_b", 0.0)),
                str(x.get("unit_a_id", "")),
                str(x.get("unit_b_id", "")),
            )
        )
        refreshed_rows.extend(appended_rows)

        print(
            f"Refresh {refresh_video_id}: {updated_count} updated, "
            f"{appended_count} appended, {untouched_count} untouched."
        )

        if dry_run:
            print("[DRY RUN] Would update CSV and manifest. No files written.")
            return 0

        # Backup before write
        backup_pairs_csv(csv_path, backups_dir)

        # Atomic write
        atomic_write_pairs_csv(csv_path, refreshed_rows, fieldnames=CSV_COLUMNS)

        # Update manifest
        existing_manifest_entry = manifest.get(refresh_video_id, {})
        manifest[refresh_video_id] = {
            "video_id": refresh_video_id,
            "filename": existing_manifest_entry.get("filename", target_video_file.name),
            "sha256": existing_manifest_entry.get("sha256", ""),
            "processed_date": datetime.now(timezone.utc).isoformat(),
            "units": len(units),
            "pairs_exported": updated_count + appended_count,
            "min_lexical": min_lexical,
            "window_seconds": window_seconds,
            "feature_version": FEATURE_VERSION,
            "embedding_model_name": default_semantic_config.embedding_model,
        }
        atomic_write_manifest(manifest_path, manifest)
        return 0

    # --------------------------------------------------------------------------
    # Case B: Standard Ingestion mode
    # --------------------------------------------------------------------------
    if not videos_dir.exists():
        videos_dir.mkdir(parents=True, exist_ok=True)

    video_files: list[Path] = []
    if single_video:
        sv_path = Path(single_video)
        if not sv_path.is_absolute():
            candidate_p = videos_dir / single_video
            if candidate_p.exists():
                sv_path = candidate_p
        if not sv_path.exists():
            sys.stderr.write(f"Error: Specified video not found: {single_video}\n")
            return 1
        video_files = [sv_path]
    else:
        for vf in sorted(videos_dir.iterdir()):
            if vf.is_file() and vf.suffix.lower() in VALID_VIDEO_EXTENSIONS:
                video_files.append(vf)

    if not video_files:
        print("No videos found to process.")
        return 0

    # Validate: Two files with the same stem is an error
    stem_map: dict[str, Path] = {}
    for vf in video_files:
        v_id = get_video_id(vf)
        if v_id in stem_map:
            sys.stderr.write(
                f"Error: Two files with the same stem found: '{stem_map[v_id].name}' and '{vf.name}'.\n"
            )
            return 1
        stem_map[v_id] = vf

    known_hashes = {
        entry.get("sha256"): entry.get("video_id")
        for entry in manifest.values()
        if entry.get("sha256")
    }

    # Gather existing pair_ids for deduplication (excluding dummy_ rows)
    existing_pair_ids = {
        str(r.get("pair_id"))
        for r in existing_rows
        if not str(r.get("video_id", "")).startswith("dummy_")
    }

    all_new_rows: list[dict[str, Any]] = []
    manifest_updates: dict[str, dict[str, Any]] = {}

    for vf in video_files:
        vid = get_video_id(vf)

        # Skip if video_id already in manifest
        if vid in manifest:
            print(f"Video '{vid}' already in manifest; skipping.")
            continue

        file_sha256 = compute_sha256(vf)

        # If same sha256 appears under a new filename, warn and skip
        if file_sha256 in known_hashes:
            matched_vid = known_hashes[file_sha256]
            print(
                f"Warning: File '{vf.name}' has identical SHA-256 to already processed video '{matched_vid}'. Skipping."
            )
            continue

        st = determine_source_type(vid, vf.name, explicit_source=source_type)
        print(f"Processing video '{vid}' (source_type: {st})...")

        _, units, stage2_candidates = run_pipeline_for_video(
            video_path=vf,
            video_id=vid,
            processed_dir=processed_dir,
            min_lexical=min_lexical,
            window_seconds=window_seconds,
            reuse_transcript=False,
            whisper_model=whisper_model,
            dry_run=dry_run,
        )

        video_new_rows: list[dict[str, Any]] = []
        for c in stage2_candidates:
            row = candidate_to_csv_row(c, vid, st)
            pid = row["pair_id"]
            if pid in existing_pair_ids:
                # Never duplicate existing rows
                continue
            video_new_rows.append(row)
            existing_pair_ids.add(pid)

        # Order rows by start_a
        video_new_rows.sort(
            key=lambda x: (
                float(x.get("start_a", 0.0)),
                float(x.get("start_b", 0.0)),
                str(x.get("unit_a_id", "")),
                str(x.get("unit_b_id", "")),
            )
        )

        all_new_rows.extend(video_new_rows)
        known_hashes[file_sha256] = vid

        manifest_updates[vid] = {
            "video_id": vid,
            "filename": vf.name,
            "sha256": file_sha256,
            "processed_date": datetime.now(timezone.utc).isoformat(),
            "units": len(units),
            "pairs_exported": len(video_new_rows),
            "min_lexical": min_lexical,
            "window_seconds": window_seconds,
            "feature_version": FEATURE_VERSION,
            "embedding_model_name": default_semantic_config.embedding_model,
        }

        s1_count = sum(1 for r in video_new_rows if r.get("passes_stage1_default") == 1)
        print(
            f"Video '{vid}': {len(units)} units, {len(video_new_rows)} candidate pairs exported "
            f"({s1_count} passes Stage 1 default)."
        )

    if not all_new_rows:
        print("No new candidate pairs to export.")
        if manifest_updates and not dry_run:
            manifest.update(manifest_updates)
            atomic_write_manifest(manifest_path, manifest)
        return 0

    if dry_run:
        print(f"\n[DRY RUN] Would export {len(all_new_rows)} candidate pairs across {len(manifest_updates)} videos.")
        for r in all_new_rows[:5]:
            print(f"  [DRY RUN SAMPLE] {r['pair_id']} (class={r['pipeline_class']}, lexical={r['lexical_score']})")
        print("[DRY RUN] No files written.")
        return 0

    # Write Safety: Backup before write
    backup_pairs_csv(csv_path, backups_dir)

    # Append new rows
    combined_rows = existing_rows + all_new_rows
    atomic_write_pairs_csv(csv_path, combined_rows, fieldnames=CSV_COLUMNS)

    # Update manifest ONLY after CSV write succeeds
    manifest.update(manifest_updates)
    atomic_write_manifest(manifest_path, manifest)

    print(f"\nSuccessfully exported {len(all_new_rows)} new candidate pairs to {csv_path}.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract candidate pairs from training videos and append to labeling CSV."
    )
    parser.add_argument("--video", type=str, default=None, help="Process one specific video file")
    parser.add_argument(
        "--min-lexical",
        type=float,
        default=DEFAULT_MIN_LEXICAL,
        help=f"Minimum lexical score threshold (default: {DEFAULT_MIN_LEXICAL})",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help=f"Window in seconds for candidate pairs (default: {DEFAULT_WINDOW_SECONDS})",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default=None,
        help="Source type tag ('scripted', 'real', etc.)",
    )
    parser.add_argument(
        "--refresh",
        type=str,
        default=None,
        metavar="VIDEO_ID",
        help="Recompute feature columns for an already-processed video",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v2",
        help="Whisper model size (default: large-v2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be added without writing to disk",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset progress statistics and exit",
    )
    parser.add_argument(
        "--build-groups",
        action="store_true",
        help="Build/update groups_v1.csv from labeled pairs",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Target video ID for --build-groups (processes all videos if omitted)",
    )
    parser.add_argument(
        "--check-groups",
        action="store_true",
        help="Validate hand labels in groups_v1.csv (read-only)",
    )
    parser.add_argument(
        "--eval-keepers",
        action="store_true",
        help="Evaluate take selection heuristics against hand labels (read-only)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.stats:
        print_stats(DEFAULT_CSV_PATH, DEFAULT_GROUPS_CSV_PATH)
        sys.exit(0)

    if args.check_groups:
        exit_code = check_groups(DEFAULT_GROUPS_CSV_PATH, DEFAULT_CSV_PATH)
        sys.exit(exit_code)

    if args.eval_keepers:
        exit_code = eval_keepers(DEFAULT_GROUPS_CSV_PATH)
        sys.exit(exit_code)

    if args.build_groups:
        exit_code = build_groups(
            pairs_csv_path=DEFAULT_CSV_PATH,
            groups_csv_path=DEFAULT_GROUPS_CSV_PATH,
            processed_dir=DEFAULT_PROCESSED_DIR,
            backups_dir=DEFAULT_BACKUPS_DIR,
            video_id=args.video_id,
            dry_run=args.dry_run,
        )
        sys.exit(exit_code)

    exit_code = process_videos(
        videos_dir=DEFAULT_VIDEOS_DIR,
        processed_dir=DEFAULT_PROCESSED_DIR,
        backups_dir=DEFAULT_BACKUPS_DIR,
        csv_path=DEFAULT_CSV_PATH,
        manifest_path=DEFAULT_MANIFEST_PATH,
        single_video=args.video,
        min_lexical=args.min_lexical,
        window_seconds=args.window_seconds,
        source_type=args.source_type,
        whisper_model=args.whisper_model,
        refresh_video_id=args.refresh,
        dry_run=args.dry_run,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
