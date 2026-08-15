from __future__ import annotations

import json
from pathlib import Path

from ..candidate_generator import rank_similar_sentence_ids
from ..normalize import split_into_units
from ..schemas import TranscriptSegment


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_transcript(path: Path) -> list[TranscriptSegment]:
    raw_items = json.loads(path.read_text(encoding="utf-8"))
    return [TranscriptSegment(**item) for item in raw_items]


def run_smoke() -> dict[str, object]:
    root = _repo_root()
    transcript_path = root / "data" / "transcripts" / "test.json"

    segments = _load_transcript(transcript_path)
    units = [unit for segment in segments for unit in split_into_units(segment)]
    ranked_sentence_ids = rank_similar_sentence_ids(units)

    return {
        "transcript": str(transcript_path),
        "segments": len(segments),
        "units": len(units),
        "similar_sentence_ids": ranked_sentence_ids,
    }


def main() -> None:
    print(json.dumps(run_smoke(), indent=2))


if __name__ == "__main__":
    main()
