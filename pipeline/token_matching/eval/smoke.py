from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
_repo_root_path = Path(__file__).resolve().parents[3]
if str(_repo_root_path) not in sys.path:
    sys.path.insert(0, str(_repo_root_path))

try:
    from pipeline.token_matching.candidate_generator import rank_similar_sentence_ids
    from pipeline.token_matching.normalize import load_transcript_units
except ImportError:
    from ..candidate_generator import rank_similar_sentence_ids
    from ..normalize import load_transcript_units


def _repo_root() -> Path:
    return _repo_root_path


def run_smoke() -> dict[str, object]:
    root = _repo_root()
    output_path = root / "data" / "raw" / "transcripts" / "output_transcript.json"
    test_path = root / "data" / "transcripts" / "test.json"
    transcript_path = output_path if output_path.exists() else test_path

    units = load_transcript_units(transcript_path)
    ranked_sentence_ids = rank_similar_sentence_ids(units)

    return {
        "transcript": str(transcript_path),
        "units": len(units),
        "similar_sentence_ids": ranked_sentence_ids,
    }


def main() -> None:
    print(json.dumps(run_smoke(), indent=2))


if __name__ == "__main__":
    main()
