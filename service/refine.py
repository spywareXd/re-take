"""/v1/refine: turn the sentences the user removes into the exact file-time ranges to cut, using the chosen cut mode (cut-mode code on this branch)."""
from __future__ import annotations

from pathlib import Path

from pipeline.step_07_review.cache import MediaCache, media_fingerprint

from .jobs import UserError


def refine_available() -> bool:
    try:
        from pipeline.step_08_video_edit import CUT_MODES, build_cutlist  # noqa: F401
        return True
    except Exception:
        return False


def refine_ranges(path: str, remove: list[str], mode: str, cache_dir: str) -> dict:
    from pipeline.step_08_video_edit import CUT_MODES, CutConfig, build_cutlist, load_envelope, probe_media

    if mode not in CUT_MODES:
        raise ValueError(f"mode must be one of {list(CUT_MODES)}")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError("Could not read the file")
    cache = MediaCache(cache_dir, media_fingerprint(p))
    transcript, units = cache.get("transcript"), cache.get("units")
    if transcript is None or units is None:
        raise UserError("This file has not been analysed yet. Run Remove Retakes first.")
    ids = {u["unit_id"] for u in units}
    unknown = [r for r in remove if r not in ids]
    if unknown:
        raise ValueError(f"unknown sentence id: {unknown[0]}")
    duration = (probe_media(p)[0] or float(transcript.get("duration", 0.0)))
    spans = [{"span_id": "refine", "adjudication_status": "resolved", "is_genuine_retake_group": True, "keeper_unit_id": None, "units_to_remove": list(remove)}]
    cutlist, _ = build_cutlist(mode, spans, units, transcript, p.name, max(duration, float(transcript.get("duration", 0.0))),
                               envelope=load_envelope(p), video_path=p, cfg=CutConfig(cut_mode=mode))
    return {"ranges": [{"start": d.start, "end": d.end} for d in cutlist.discard_spans]}
