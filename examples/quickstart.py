"""Find the retakes in a small synthetic transcript, step by step (no video, no GPU needed).

    uv run python examples/quickstart.py            # downloads the pair scorer (about 270 MB) the first time
    uv run python examples/quickstart.py --mock     # fully offline: a cheap stand-in scorer, scores are NOT comparable with the real ones

The pipeline only finds and groups sentences that are repeats of each other. It never decides what to keep.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.final.final_scorer import FinalScorer  # noqa: E402
from models.transcript import Transcript  # noqa: E402
from pipeline.step_02_preprocessing import load_transcript_units  # noqa: E402
from pipeline.step_03_token_matching import generate_candidate_pairs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcript", default=str(REPO / "examples" / "sample_transcript.json"))
    ap.add_argument("--mock", action="store_true", help="use the offline stand-in scorer")
    args = ap.parse_args()

    transcript = Transcript.model_validate(json.loads(Path(args.transcript).read_text(encoding="utf-8")))
    units = load_transcript_units(transcript)                                   # step 2: sentence units
    print(f"step 2: {len(units)} sentence units")
    for u in units:
        print(f"  {u.unit_id:>5}  {u.start:6.2f}-{u.end:6.2f}  {u.raw_text}")

    candidates = generate_candidate_pairs(units)                                # step 3: token matching
    print(f"\nstep 3: {len(candidates)} candidate pairs (earlier unit first)")

    scorer = FinalScorer(mock=args.mock)                                        # step 4: the pair scorer
    pairs = [{"text_a": c.unit_a.raw_text, "text_b": c.unit_b.raw_text} for c in candidates]
    detail = scorer.score_detailed(pairs)
    decisions, resolved_by = scorer.resolve(detail["p"], detail["ce_prob"])
    print(f"\nstep 4: scorer mode = {scorer.mode}")
    for c, d, by, p in zip(candidates, decisions, resolved_by, detail["p"]):
        if d != "not_retake":
            print(f"  {d:9} ({by}, p={p:.2f})  {c.unit_a.unit_id} -> {c.unit_b.unit_id}: {c.unit_a.raw_text!r} / {c.unit_b.raw_text!r}")

    spans = [dataclasses.asdict(c) for c, d in zip(candidates, decisions) if d == "retake"]
    print(f"\n{len(spans)} pair(s) decided as retakes. Next steps (span grouping, choosing what to keep, cutting) are in pipeline/ and docs/pipeline.md.")


if __name__ == "__main__":
    main()
