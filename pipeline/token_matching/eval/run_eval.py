from __future__ import annotations

import json
from pathlib import Path

from ..candidate_generator import generate_candidate_pairs
from ..normalize import split_into_units
from ..schemas import TranscriptSegment


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_transcript(path: Path) -> list[TranscriptSegment]:
    raw_items = json.loads(path.read_text(encoding="utf-8"))
    return [TranscriptSegment(**item) for item in raw_items]


def _load_labels(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_eval() -> dict[str, object]:
    root = _repo_root()
    transcript_path = root / "data" / "transcripts" / "test.json"
    labels_path = root / "data" / "transcripts" / "test_labels.json"
    report_path = root / "pipeline" / "token_matching" / "eval" / "report.md"

    segments = _load_transcript(transcript_path)
    units = [unit for segment in segments for unit in split_into_units(segment)]
    candidates = generate_candidate_pairs(units)

    if not labels_path.exists():
        report = (
            "# Token Matching Eval\n\n"
            f"- Transcript: `{transcript_path}`\n"
            "- Status: labels missing\n"
            "- Next step: add labeled ground-truth retake pairs to `data/transcripts/test_labels.json`\n"
            f"- Candidate count: {len(candidates)}\n"
        )
        report_path.write_text(report, encoding="utf-8")
        return {
            "status": "labels_missing",
            "transcript": str(transcript_path),
            "candidates": [candidate.to_dict() for candidate in candidates],
        }

    labels = _load_labels(labels_path)
    report = (
        "# Token Matching Eval\n\n"
        f"- Transcript: `{transcript_path}`\n"
        f"- Labels: `{labels_path}`\n"
        f"- Label count: {len(labels)}\n"
        f"- Candidate count: {len(candidates)}\n"
    )
    report_path.write_text(report, encoding="utf-8")
    return {
        "status": "ok",
        "transcript": str(transcript_path),
        "labels": len(labels),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }


def main() -> None:
    result = run_eval()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
