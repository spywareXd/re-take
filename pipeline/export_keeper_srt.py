"""
Export the keeper transcript as an .srt subtitle file. Nothing is cut and no video is written.

    python -m pipeline.export_keeper_srt data/videos/sample_b.mp4 [--keeper-provider finetuned] [--only-groups] [--out runs/srt_sample_b/keepers.srt]

Runs the pipeline up to the keeper step (transcript -> sentences -> retake candidates -> groups -> keeper choice), then writes one subtitle cue per kept
sentence at its ORIGINAL timestamps. The timeline is not shortened: the file spans the whole video, and the places where a discarded take was simply have
no subtitle. By default the cues are every sentence that is not discarded (keepers of retake groups plus all sentences that are not in a group).
--only-groups writes the keepers of retake groups only.

Which sentences are discarded follows the cut rule exactly: only groups whose keeper step is `resolved` discard anything (failed groups discard nothing).
Output folder: runs/srt_<video>_<provider>_<timestamp>/ (never overwritten): keepers.srt, discarded.txt (ids only), processed/ (the pipeline's intermediate files).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[1]


def srt_time(seconds: float) -> str:
    ms = max(0, int(round(float(seconds) * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _u(unit: Any, key: str) -> Any:
    return unit[key] if isinstance(unit, dict) else getattr(unit, key)


def keeper_unit_ids(spans: Sequence[dict]) -> tuple[set[str], set[str]]:
    """(discarded ids, kept ids that belong to a resolved retake group)."""
    from pipeline.step_08_video_edit.cut_generator import resolved_discard_unit_ids

    discarded = set(resolved_discard_unit_ids(spans))
    kept = set()
    for s in spans:
        if s.get("adjudication_status") == "resolved" and s.get("is_genuine_retake_group") is not False:
            kept.update(str(u) for u in (s.get("keeper_unit_ids") or ([s["keeper_unit_id"]] if s.get("keeper_unit_id") else [])))
    return discarded, kept


def build_srt(units: Iterable[Any], spans: Sequence[dict], only_groups: bool = False) -> tuple[str, list[str], int]:
    """(srt text, discarded unit ids in time order, number of cues)."""
    discarded, group_keepers = keeper_unit_ids(spans)
    ordered = sorted(units, key=lambda u: (float(_u(u, "start")), str(_u(u, "unit_id"))))
    chosen = [u for u in ordered if str(_u(u, "unit_id")) not in discarded and (not only_groups or str(_u(u, "unit_id")) in group_keepers)]
    chosen = [u for u in chosen if str(_u(u, "raw_text")).strip()]
    blocks = []
    for i, u in enumerate(chosen):
        start, end = float(_u(u, "start")), float(_u(u, "end"))
        if i + 1 < len(chosen):
            end = min(end, float(_u(chosen[i + 1], "start")))           # cues never overlap
        end = max(end, start + 0.001)
        blocks.append(f"{i + 1}\n{srt_time(start)} --> {srt_time(end)}\n{str(_u(u, 'raw_text')).strip()}\n")
    gone = [str(_u(u, "unit_id")) for u in ordered if str(_u(u, "unit_id")) in discarded]
    return "\n".join(blocks), gone, len(blocks)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_video")
    ap.add_argument("--keeper-provider", default=None, choices=["finetuned", "gpt-oss", "keep-last"])
    ap.add_argument("--only-groups", action="store_true", help="write only the keepers of retake groups")
    ap.add_argument("--out", default=None, help="the .srt path (default: inside the new run folder)")
    ap.add_argument("--model", default="large-v2", help="ASR model size")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--reuse-transcript-from", default=None, help="an earlier run folder whose processed/transcript.json is copied in (same sentence ids)")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(REPO))
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
    except ImportError:
        pass
    from pipeline.run_pipeline import get_video_pipeline_paths, run_full_pipeline

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    video = Path(args.input_video).resolve()
    provider = args.keeper_provider or "default"
    out = REPO / "runs" / f"srt_{video.stem}_{provider.replace('-', '')}_{datetime.now():%Y%m%d_%H%M%S}"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting")
    processed = out / "processed"
    processed.mkdir(parents=True)
    if args.reuse_transcript_from:
        import shutil
        shutil.copyfile(Path(args.reuse_transcript_from) / "processed" / "transcript.json", processed / "transcript.json")
    res = run_full_pipeline(video, processed_dir=processed, model_name=args.model, device=args.device, llm_provider="groq", llm_model="openai/gpt-oss-120b",
                            keeper_provider=args.keeper_provider, reuse_existing_transcript=bool(args.reuse_transcript_from), stop_after="keeper")
    if not res.success:
        print(f"[ERROR] pipeline failed: {res.error}")
        return 1
    paths = get_video_pipeline_paths(video, processed_dir=processed)
    units = json.loads(paths.units.read_text(encoding="utf-8"))["units"]
    data = json.loads(paths.adjudicated_spans.read_text(encoding="utf-8"))
    spans = data.get("spans", data) if isinstance(data, dict) else data
    text, gone, n = build_srt(units, spans, args.only_groups)
    srt_path = Path(args.out) if args.out else out / "keepers.srt"
    if srt_path.exists():
        raise SystemExit(f"{srt_path} exists; not overwriting")
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text(text, encoding="utf-8")
    (out / "discarded.txt").write_text("\n".join(gone) + ("\n" if gone else ""), encoding="utf-8")
    print(f"[OK] {n} subtitle cues, {len(gone)} discarded sentences, nothing cut.\n  srt: {srt_path}\n  folder: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
