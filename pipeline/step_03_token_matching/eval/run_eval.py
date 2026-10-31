from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
_repo_root_path = Path(__file__).resolve().parents[3]
if str(_repo_root_path) not in sys.path:
    sys.path.insert(0, str(_repo_root_path))

from pipeline.step_03_token_matching.candidate_generator import generate_candidate_pairs
from pipeline.step_02_preprocessing.normalize import load_transcript_units


def _repo_root() -> Path:
    return _repo_root_path


def run_eval() -> dict[str, object]:
    root = _repo_root()
    output_path = root / "data" / "raw" / "transcripts" / "output_transcript.json"
    test_path = root / "data" / "transcripts" / "test.json"
    transcript_path = output_path if output_path.exists() else test_path

    labels_path = root / "data" / "transcripts" / "test_labels.json"
    report_path = root / "pipeline" / "step_03_token_matching" / "eval" / "report.md"

    units = load_transcript_units(transcript_path)
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
            "units": len(units),
            "candidates": [candidate.to_dict() for candidate in candidates],
        }

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
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
