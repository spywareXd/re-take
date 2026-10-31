"""FastAPI app: the local retake service for the SimpleCut editor (backend-handoff.md section 3, schema simplecut.retakes/2). 127.0.0.1, no auth."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
import asyncio
import json
import re

from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from pipeline.step_07_review import ReviewConfig, analyze_media, build_groups, validate_document
from pipeline.step_07_review.labels import record_answers

from .config import ServiceConfig
from .events import EventLog, read_log, sse_format
from . import export_options as xo
from .jobs import JobContext, JobManager, UserError
from .project_store import ProjectCache, ProjectError, ProjectStore
from .llm import LLMSetup, adjudicate_groups, detect_llm, keeper_advice, record_decision
from .refine import refine_available, refine_ranges

logger = logging.getLogger("service")

SILENCE_MIN_S, SILENCE_MAX_S, SILENCE_STEP_S = 0.1, 3.0, 0.05            # the slider's range; validated again on the server


class BadRequest(Exception):
    pass


class Conflict(Exception):
    """HTTP 409 with a machine-readable body: the request would replace an existing result and needs the user's confirmation first."""

    def __init__(self, message: str, **body: Any):
        super().__init__(message)
        self.body = {"error": message, **body}


@dataclass
class Deps:
    """Injection points (tests pass fakes; production leaves them None)."""
    transcribe_fn: Optional[Callable[[Path], Any]] = None
    scorer: Any = None                      # a FinalScorer; None = build the real one lazily, once
    llm_client: Any = None                  # an LLMClient (e.g. MockLLMClient)
    keeper_backend: Any = None              # decision backend for the 'finetuned' keeper (tests); None = Studio Decision API on localhost


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def create_app(config: Optional[ServiceConfig] = None, deps: Optional[Deps] = None) -> FastAPI:
    cfg = config or ServiceConfig()
    deps = deps or Deps()
    jobs = JobManager(cfg.max_jobs_kept)
    llm: LLMSetup = detect_llm(cfg, deps.llm_client)
    review_cfg = ReviewConfig(cache_dir=cfg.cache_dir, labels_path=cfg.labels_path, skipped_path=cfg.skipped_path, max_questions=cfg.max_questions,
                              unsure_band_enabled=cfg.unsure_band_enabled)
    state: dict[str, Any] = {"scorer": deps.scorer}
    from pipeline.step_06_llm_adjudication.keeper_policy import PROVIDERS, KeeperConfig
    from pipeline.step_06_llm_adjudication.keeper_providers import FinetunedProvider, KeepLastProvider

    keeper_cfg = KeeperConfig.from_env(provider=cfg.keeper_provider)
    if keeper_cfg.provider not in PROVIDERS:
        raise ValueError(f"keeper provider '{keeper_cfg.provider}' is not one of {list(PROVIDERS)}")
    finetuned, keep_last = FinetunedProvider(keeper_cfg, deps.keeper_backend), KeepLastProvider()
    if hasattr(finetuned.backend, "before_load"):
        def _make_room():                                     # the scorer's cross-encoders leave the GPU before the decision model loads
            sc = state["scorer"]
            if sc is not None and getattr(sc, "_ce", None) is not None:
                sc._ce = None
            from pipeline.run_pipeline import _free_cuda_cache
            _free_cuda_cache()
        finetuned.backend.before_load = _make_room
    state["keeper"] = keeper_cfg.provider                     # the service-wide keeper; PUT /v1/keeper switches it while the service runs

    def availability(name: str) -> tuple[bool, str]:
        if name == "finetuned":
            return finetuned.availability()
        if name == "keep-last":
            return keep_last.availability()
        return bool(llm.available), (f"LLM provider '{llm.provider}'" if llm.available else "no LLM key is configured")

    def provider_status() -> dict[str, dict]:
        return {n: dict(zip(("available", "reason"), availability(n))) for n in PROVIDERS}

    def keeper_state() -> dict:
        return {"current": state["keeper"], "default": keeper_cfg.provider, "providers": provider_status(), "finetuned": finetuned.info()}

    def pick_provider(requested: Any) -> str:
        if requested is not None and requested not in PROVIDERS:
            raise BadRequest(f"'keeper_provider' must be one of {list(PROVIDERS)}.")
        return requested or state["keeper"]

    def scorer_provider():
        if state["scorer"] is None:
            from pipeline.step_04_semantic_matching import get_final_scorer
            state["scorer"] = get_final_scorer(unsure_band_enabled=cfg.unsure_band_enabled, max_questions=cfg.max_questions)
        return state["scorer"]

    app = FastAPI(title="ReTake service", version="1", docs_url=None, redoc_url=None)
    app.state.jobs, app.state.config, app.state.llm = jobs, cfg, llm

    # ---- errors: always {"error": "..."} ----
    @app.exception_handler(BadRequest)
    async def _bad(_: Request, exc: BadRequest):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(Conflict)
    async def _conflict(_: Request, exc: Conflict):
        return JSONResponse(exc.body, status_code=409)

    @app.exception_handler(RequestValidationError)
    async def _val(_: Request, exc: RequestValidationError):
        return JSONResponse({"error": "The request is not valid."}, status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        return JSONResponse({"error": "Not found." if exc.status_code == 404 else "The request could not be handled."}, status_code=exc.status_code)

    async def body_of(request: Request) -> dict:
        try:
            body = await request.json()
        except Exception:
            raise BadRequest("The request body must be JSON.")
        if not isinstance(body, dict):
            raise BadRequest("The request body must be a JSON object.")
        return body

    def project_of(body: dict) -> Optional[ProjectStore]:
        """The optional 'project' field: the project folder. With it, everything the request stores goes to <project>/data/; without it the old service/data is used."""
        folder = body.get("project")
        if folder is None:
            return None
        if not isinstance(folder, str) or not folder.strip():
            raise BadRequest("'project' must be the project folder path.")
        try:
            return ProjectStore.create(folder)
        except (OSError, ProjectError) as exc:
            raise BadRequest(f"The project folder cannot be used: {exc}")

    def cfg_for(store: Optional[ProjectStore]) -> ReviewConfig:
        if store is None:
            return review_cfg
        return replace(review_cfg, cache_dir=str(store.cache_dir), labels_path=str(store.labels_path), skipped_path=str(store.skipped_path))

    def pick_model(body: dict, store: Optional[ProjectStore]) -> tuple[str, str]:
        """(offered name, faster-whisper name): the request's 'model', else the project's saved choice, else today's default. Unknown names are a 400."""
        from pipeline.step_01_transcription import model_catalog as mc
        name = body.get("model")
        if name is not None and not isinstance(name, str):
            raise BadRequest("'model' must be a model name.")
        name = name or (store.settings().get("transcription_model") if store else None) or mc.DEFAULT_MODEL
        try:
            return name, mc.resolve(name)
        except mc.UnknownModel as exc:
            raise BadRequest(str(exc))

    def pick_gap(body: dict, store: Optional[ProjectStore]) -> float:
        """The silence gap (seconds) that splits sentence units: the request's 'silence_gap_s', else the project's saved value, else today's 0.7. Range-checked here."""
        from pipeline.step_02_preprocessing import PAUSE_SPLIT_THRESHOLD_SECONDS
        v = body.get("silence_gap_s")
        if v is None:
            v = (store.settings().get("silence_gap_s") if store else None)
        if v is None:
            return PAUSE_SPLIT_THRESHOLD_SECONDS
        if not _num(v) or not (SILENCE_MIN_S <= v <= SILENCE_MAX_S):
            raise BadRequest(f"'silence_gap_s' must be a number of seconds between {SILENCE_MIN_S:g} and {SILENCE_MAX_S:g}.")
        return round(float(v), 4)

    def units_gap(store: ProjectStore) -> Optional[float]:
        """The silence gap the stored units were made with (0.7, the old fixed value, when nothing is recorded); None when there are no units."""
        from pipeline.step_02_preprocessing import PAUSE_SPLIT_THRESHOLD_SECONDS
        return (store.get("units_meta") or {}).get("silence_gap_s", PAUSE_SPLIT_THRESHOLD_SECONDS) if store.has("units") else None

    def revalidate(store: ProjectStore, meta: dict, model_name: str) -> None:
        """The model was set back to the one the stored transcript was made with: the stale marks that the model change caused are lifted (the files never moved).
        A transcript the user set aside on purpose (reset) is not brought back by this."""
        if "transcript" in store.stale() and meta.get("model") == model_name and "transcript" not in store.reset_marks():
            for n in ("transcript", "words", "units", "keepers", "cutlist", *store._names_like("pairs")):
                store.clear_stale(n)

    def guard_replace(store: ProjectStore, body: dict, model_name: str, gap: float) -> None:
        """A changed model never silently overwrites a transcript: the transcript and everything downstream are marked stale, and the request is refused (409)
        until it is repeated with "replace_existing": true. The old results are kept (stale/ copies)."""
        meta = store.get("transcript_meta") or {}
        revalidate(store, meta, model_name)
        if store.has("transcript") and meta.get("model") and meta["model"] != model_name:
            store.invalidate("transcript")
        if "transcript" in store.reset_marks():                                       # the user set the transcription aside: start from scratch, nothing to ask
            store.update_settings(transcription_model=model_name, silence_gap_s=gap)
            return
        old_gap = units_gap(store)
        if old_gap is not None and abs(old_gap - gap) > 1e-9 and "transcript" not in store.stale():
            store.invalidate("units")                                                       # the units (and pairs/keepers/cutlist built on them) no longer match the slider
        if ("transcript" in store.stale() or "units" in store.stale()) and not body.get("replace_existing"):
            if "transcript" not in store.stale():
                raise Conflict(f"The stored sentences were split at a {old_gap:g} s silence gap and the setting is now {gap:g} s. Re-splitting replaces them and the results "
                               "built on them (pairs, keepers, cut list); the old ones are kept as numbered copies. Send the request again with 'replace_existing': true, "
                               "or call /v1/project/resplit (no re-transcription).", code="silence_changed", units_silence_gap_s=old_gap, requested=gap, stale=store.stale())
            raise Conflict("This project already has a transcript made with a different model. Re-transcribing replaces it and the results built on it; "
                           "the old ones are kept as numbered copies. Send the request again with 'replace_existing': true to go ahead.",
                           code="model_changed", transcript_model=meta.get("model"), requested=model_name, stale=store.stale())
        store.update_settings(transcription_model=model_name, silence_gap_s=gap)

    def need_path(body: dict) -> str:
        path = body.get("path")
        if not isinstance(path, str) or not path.strip():
            raise BadRequest("'path' is required.")
        return path

    # ---- capabilities ----
    @app.get("/v1/capabilities")
    def capabilities():
        keeper = keeper_state()
        from pipeline.step_01_transcription.model_catalog import catalog
        from pipeline.step_02_preprocessing import PAUSE_SPLIT_THRESHOLD_SECONDS
        return {"adjudicate": bool(keeper["providers"][keeper["current"]]["available"]), "refine": refine_available(), "keeper": keeper, "transcription": catalog(),
                "silence_gap": {"default": PAUSE_SPLIT_THRESHOLD_SECONDS, "min": SILENCE_MIN_S, "max": SILENCE_MAX_S, "step": SILENCE_STEP_S, "unit": "seconds"}}

    # ---- project settings and status ----
    @app.put("/v1/project/settings")
    async def project_settings(request: Request):
        """Change the project's settings. A transcription model that differs from the one the stored transcript was made with marks it (and what depends on it) stale
        and says so; nothing is recomputed or replaced here."""
        body = await body_of(request)
        store = project_of(body)
        if store is None:
            raise BadRequest("'project' is required.")
        changes: dict = {}
        if "transcription_model" in body:
            name, _ = pick_model({"model": body["transcription_model"]}, None)
            changes["transcription_model"] = name
        if "silence_gap_s" in body:
            changes["silence_gap_s"] = pick_gap({"silence_gap_s": body["silence_gap_s"]}, None)
        if not changes:
            raise BadRequest("Nothing to change: send 'transcription_model' and/or 'silence_gap_s'.")
        store.update_settings(**changes)
        meta = store.get("transcript_meta") or {}
        if "transcription_model" in changes:
            revalidate(store, meta, changes["transcription_model"])
        stale: list[str] = []
        if "transcription_model" in changes and store.has("transcript") and meta.get("model") and meta["model"] != changes["transcription_model"]:
            stale = store.invalidate("transcript")
        ask = "The transcript was made with a different model. Re-transcribe to bring it up to date?" if stale else None
        old_gap = units_gap(store)
        if "silence_gap_s" in changes and old_gap is not None and abs(old_gap - changes["silence_gap_s"]) > 1e-9 and "transcript" not in store.stale():
            stale = sorted(set(stale) | set(store.invalidate("units")))
            ask = ask or "The sentences were split at a different silence gap. Re-split them now (no re-transcription)? Pairs, keepers and the cut list built on them will be recomputed."
        return {"settings": store.settings(), "stale": store.stale(), "newly_stale": stale, "ask": ask}

    @app.get("/v1/project/units-preview")
    def units_preview(project: str, silence_gap_s: float):
        """How many sentences a silence gap would produce, from the saved word timestamps (no re-transcription, nothing stored)."""
        from pipeline.step_07_review.json_export import _load_units
        store = ProjectStore(project)
        transcript = store.get("transcript")
        if transcript is None:
            raise BadRequest("This project has no transcript yet; the preview needs one.")
        gap = pick_gap({"silence_gap_s": silence_gap_s}, None)
        return {"silence_gap_s": gap, "units": len(_load_units(transcript, gap)), "current_silence_gap_s": units_gap(store),
                "current_units": len(store.get("units") or [])}

    @app.post("/v1/project/resplit")
    async def resplit(request: Request):
        """Re-split the saved transcript into sentences at the project's silence gap, WITHOUT re-transcribing. Without "confirm": true it only says what would be
        replaced (409). With it, the old units are kept as stale/units.<n>.json, and pairs, keepers and cutlist are marked stale (kept, recomputed on the next analyze)."""
        from pipeline.step_07_review.json_export import _load_units, unit_dicts
        body = await body_of(request)
        store = project_of(body)
        if store is None:
            raise BadRequest("'project' is required.")
        transcript = store.get("transcript")
        if transcript is None:
            raise BadRequest("This project has no transcript yet.")
        if "transcript" in store.stale():
            raise BadRequest("The transcript is out of date (a different model was chosen). Re-transcribe first.")
        gap = pick_gap(body, store)
        affected = [n for k in ("units", *ProjectStore.DOWNSTREAM["units"]) for n in store._names_like(k)]
        if affected and not body.get("confirm"):
            raise Conflict("Re-splitting replaces the existing sentences and marks everything built on them stale. The old versions are kept as numbered copies. "
                           "Send 'confirm': true to go ahead.", code="resplit_needs_confirmation", will_replace=affected, silence_gap_s=gap)
        units = await run_in_threadpool(lambda: unit_dicts(_load_units(transcript, gap)))
        store.invalidate("units")
        copy = store.save_result("units", units)
        store.put("units_meta", {"silence_gap_s": gap})
        store.update_settings(silence_gap_s=gap)
        return {"units": len(units), "silence_gap_s": gap, "stale": store.stale(), "previous_units_copy": copy.name if copy else None}

    @app.get("/v1/project/status")
    def project_status(project: str):
        store = ProjectStore(project)
        return {"settings": store.settings(), "stale": store.stale(), "source": store.check_source(), "transcript_model": (store.get("transcript_meta") or {}).get("model"),
                "has": {n: store.has(n) for n in ("transcript", "words", "units", "keepers", "cutlist")}, "legacy": legacy_names(store),
                "reset": store.reset_marks()}

    def legacy_names(store: ProjectStore) -> list[str]:
        """What the old per-machine cache (service/data/cache/<fingerprint>) holds for this project's source video and the project does not have yet."""
        ref = (store.get("project", {}) or {}).get("source") or {}
        fp = (ref.get("fingerprint") or {}).get("sha256")
        if not fp or not re.fullmatch(r"[0-9a-f]{64}", fp):
            return []
        have = {"transcript": store.has("transcript"), "units": store.has("units")}
        found = []
        for stem in store.legacy_available(cfg.cache_dir, fp):
            name = stem if stem in ("transcript", "units") else (f"pairs__{stem[len('scores__'):]}" if stem.startswith("scores__") else None)
            if name and not have.get(name) and not store.has(re.sub(r"[^A-Za-z0-9._-]", "_", name)):
                found.append(stem)
        return sorted(found)

    # ---- set work aside (Ctrl+Z of a step, or the Reset buttons) and bring it back (redo) ----
    WORK_KINDS = {"transcription": "transcript", "keepers": "keepers"}

    def legacy_cache_for(path: Any) -> Any:
        from pipeline.step_07_review.cache import MediaCache, media_fingerprint
        if not isinstance(path, str) or not Path(path).is_file():
            raise BadRequest("'path' (the video) is required when the project has no folder.")
        return MediaCache(cfg.cache_dir, media_fingerprint(Path(path)))

    def legacy_names_of(cache: Any, what: str) -> list[str]:
        """The results of the old per-machine cache that belong to `what` (the old cache keeps no keeper groups)."""
        if what != "transcription":
            return []
        names = ["transcript", "units", "units_meta"]
        names += sorted(p.stem for p in cache.dir.glob("scores__*.json")) if cache.dir.is_dir() else []
        return names

    async def work_body(request: Request) -> tuple[dict, str, Optional[ProjectStore]]:
        body = await body_of(request)
        what = body.get("what")
        if what not in WORK_KINDS:
            raise BadRequest(f"'what' must be one of {', '.join(WORK_KINDS)}.")
        return body, what, project_of(body)

    @app.post("/v1/work/reset")
    async def work_reset(request: Request):
        """Set the transcription (transcript, words, sentences, pair scores) or the keeper groups aside. Nothing is deleted and the user's answers are untouched: the
        results stay on disk, the next run computes them again (the old ones become numbered copies), and /v1/work/restore brings them back."""
        body, what, store = await work_body(request)
        if store is not None:
            names = store.reset_work(WORK_KINDS[what])
            return {"reset": names, "where": "project", "kept": True}
        cache = legacy_cache_for(body.get("path"))
        return {"reset": cache.reset(legacy_names_of(cache, what)), "where": "per-machine cache", "kept": True}

    @app.post("/v1/work/restore")
    async def work_restore(request: Request):
        body, what, store = await work_body(request)
        if store is not None:
            return {"restored": store.restore_work(WORK_KINDS[what]), "where": "project"}
        cache = legacy_cache_for(body.get("path"))
        return {"restored": cache.restore(legacy_names_of(cache, what)), "where": "per-machine cache"}

    @app.post("/v1/project/import-legacy")
    async def project_import_legacy(request: Request):
        """COPY the old per-machine cache files of this project's source video into the project. The originals are not touched, and nothing the project
        already has is replaced."""
        body = await body_of(request)
        store = project_of(body)
        if store is None:
            raise BadRequest("'project' is required.")
        path = body.get("path")
        if isinstance(path, str) and path.strip() and Path(path).is_file() and not (store.get("project", {}) or {}).get("source"):
            store.set_source(path)                                                      # a project saved after its first analysis: record the video it belongs to
        ref = (store.get("project", {}) or {}).get("source") or {}
        fp = (ref.get("fingerprint") or {}).get("sha256")
        if not fp or not re.fullmatch(r"[0-9a-f]{64}", fp):
            raise BadRequest("This project has no source video recorded, so there is nothing to match old data to.")
        done = store.import_legacy(cfg.cache_dir, fp)
        return {"imported": done, "originals_untouched": True}

    # ---- export (item 4): options, saved presets, and the export job ----
    @app.get("/v1/export/options")
    def export_options():
        """Everything the export dialog needs. Hardware (NVENC) appears only when ffmpeg lists the encoder AND a test encode worked."""
        from . import export_media as xm
        hw = {k: True for k, v in xm.nvenc_support().items() if v}
        presets = xo.all_presets()
        return {"kinds": list(xo.KINDS), "presets": presets, "preset_order": list(presets),         # JSON objects have no order: built-ins first, then the user's
                "resolutions": {"source": True, "heights": list(xo.RESOLUTION_PRESETS), "canvases": {k: list(v) for k, v in xo.CANVAS_PRESETS.items()}, "custom": True,
                                "fits": list(xo.FITS), "min": xo.MIN_DIM, "max": xo.MAX_DIM, "must_be_even": True},
                "fps": {"source": True, "choices": list(xo.FPS_CHOICES), "custom_min": xo.FPS_MIN, "custom_max": xo.FPS_MAX},
                "video": {"codecs": list(xo.VIDEO_CODECS), "crf_presets": xo.CRF_PRESETS, "crf_range": list(xo.CRF_RANGE), "bitrate_kbps_range": list(xo.BITRATE_RANGE_KBPS),
                          "speeds": list(xo.X264_SPEEDS), "hardware": hw or None},
                "audio": {"codec": "aac", "bitrates_kbps": list(xo.AUDIO_BITRATES), "normalize_loudness_default": False, "fade_ms_default": 15.0, "fade_ms_max": 100.0},
                "audio_only_formats": list(xo.AUDIO_FORMATS), "text_formats": {k: list(v) for k, v in xo.TEXT_FORMATS.items()},
                "encoders": xm.encoders_available()}

    @app.get("/v1/export/presets")
    def export_presets():
        return xo.all_presets()

    @app.put("/v1/export/presets/{name}")
    async def export_preset_save(name: str, request: Request):
        """Save a custom preset in the USER's config folder (not in a project)."""
        body = await body_of(request)
        try:
            return xo.save_user_preset(name, body)
        except xo.OptionError as exc:
            raise BadRequest(str(exc))

    @app.delete("/v1/export/presets/{name}", status_code=204)
    def export_preset_delete(name: str):
        if name in xo.BUILTIN_PRESETS:
            raise BadRequest("Built-in presets cannot be deleted.")
        xo.delete_user_preset(name)
        return Response(status_code=204)

    @app.post("/v1/export", status_code=202)
    async def export_start(request: Request):
        """Export into <project>/data/exports/ under a name that never overwrites an earlier export. Body: project, path (the source video), kind
        (video | audio | subtitles | cutlist | transcript), format (for audio / text kinds), preset + video/resolution/fps/audio overrides, and the cut as `cut_ranges`
        (removed, [{start,end}]) or `keep_spans`; without either the project's saved cut list is used, else the whole video is exported."""
        from . import export_media as xm
        body = await body_of(request)
        store = project_of(body)
        if store is None:
            raise BadRequest("'project' is required: exports are written into the project folder.")
        path = Path(need_path(body))
        kind = body.get("kind", "video")
        if kind not in xo.KINDS:
            raise BadRequest(f"'kind' must be one of {', '.join(xo.KINDS)}.")
        fmt = body.get("format")
        if kind in xo.TEXT_FORMATS and fmt not in xo.TEXT_FORMATS[kind]:
            raise BadRequest(f"'format' for {kind} must be one of {', '.join(xo.TEXT_FORMATS[kind])}.")
        if kind == "audio" and fmt not in xo.AUDIO_FORMATS:
            raise BadRequest(f"'format' for audio must be one of {', '.join(xo.AUDIO_FORMATS)}.")
        if not path.is_file():
            raise BadRequest("Could not read the file")

        def prepare():
            info = xm.probe_video(path)
            if kind == "video" and not info.has_video:
                raise xo.OptionError("The source has no video stream.")
            cuts = body.get("cut_ranges")
            if body.get("keep_spans") is not None:
                keeps = xm.normalise_keeps(body["keep_spans"], info.duration)
            else:
                if cuts is None:
                    saved = store.get("cutlist") or {}
                    cuts = [(r["start"], r["end"]) for r in saved.get("ranges", [])] if isinstance(saved, dict) else []
                keeps = xm.keep_from_cuts([(c["start"], c["end"]) if isinstance(c, dict) else c for c in cuts], info.duration)
            if not keeps:
                raise xo.OptionError("Nothing is left after the cuts: there is nothing to export.")
            opts = None
            if kind == "video":
                opts = xo.apply_preset(body.get("preset"), {k: body.get(k) for k in xo.PRESET_KEYS})
                nv = xm.nvenc_support() if (opts.get("video") or {}).get("hardware") else {"h264": False, "h265": False}
                xo.plan_resolution(info.width, info.height, opts.get("resolution"))          # validate now: a bad option is a 400, not a failed job
                xo.plan_fps(opts.get("fps"), info.vfr, info.fps)
                xo.plan_video(opts.get("video"), nv)
                xo.plan_audio(opts.get("audio"))
            elif kind == "audio":
                if not info.has_audio:
                    raise xo.OptionError("The source has no audio stream.")
                xo.plan_audio(body.get("audio"))
            return info, keeps, opts

        try:
            info, keeps, opts = await run_in_threadpool(prepare)
        except (xo.OptionError, xm.ExportError) as exc:
            raise BadRequest(str(exc))
        destination = None
        if body.get("destination") is not None:
            raw = body["destination"]
            if not isinstance(raw, str) or not raw.strip() or not Path(raw).is_absolute():
                raise BadRequest("'destination' must be the full path of a folder.")
            destination = Path(raw)
            if destination.exists() and not destination.is_dir():
                raise BadRequest("'destination' is a file, not a folder.")
        stem = xm.safe_stem(body.get("name") or path.name) + {"video": "_" + (body.get("preset") or "custom"), "audio": "_audio", "subtitles": "_subtitles",
                                                                "cutlist": "_cutlist", "transcript": "_transcript"}[kind]
        ext = {"video": "mp4", "audio": fmt, "subtitles": fmt, "cutlist": fmt, "transcript": fmt}[kind]

        def run(ctx: JobContext):
            try:
                out = store.unique_export_path(stem, ext, destination)                  # reserved atomically: never an existing file
            except OSError as exc:                                                       # an unwritable destination is said in plain words
                raise UserError(f"Could not write into the destination folder: {exc.strerror or exc}")
            ok = False
            try:
                with ctx.events.stage("export_render", f"{kind} -> {out.name}") as st:
                    def prog(pct: float, msg: str) -> None:
                        st.progress(pct, msg)
                        ctx.progress("Exporting", pct / 100.0)

                    cancelled = ctx.job.cancel.is_set
                    plan: dict = {}
                    if kind == "video":
                        plan = xm.render_video(info, keeps, out, opts, prog, cancelled)
                    elif kind == "audio":
                        plan = xm.render_audio(info, keeps, out, fmt, body.get("audio"), prog, cancelled)
                    else:
                        _write_text_export(store, kind, fmt, out, info, keeps, path)
                    try:
                        shown = out.resolve().relative_to(store.root.resolve()).as_posix()      # inside the project: relative to its folder; elsewhere: the full path
                    except ValueError:
                        shown = str(out)
                    result = {"path": str(out), "relative": shown, "kind": kind, "plan": plan, "notes": plan.get("notes", [])}
                    if kind in ("video", "audio"):
                        result["verify"] = xm.verify_output(out)
                        if kind == "video":
                            result["notes"].append(f"Output frame rate: {result['verify']['frame_rate_mode']}")
                    st.done_message = out.name
                ok = True
                return result
            finally:
                if not ok:
                    try:
                        out.unlink()                                                    # the file THIS job reserved; a failed export leaves no empty file behind
                    except OSError:
                        pass

        return {"job": jobs.submit("export", run, events_path=store.progress_path)}

    def _write_text_export(store: ProjectStore, kind: str, fmt: str, out: Path, info, keeps, source: Path) -> None:
        from . import export_media as xm
        if kind == "subtitles":
            units, words = store.get("units"), store.get("words")
            if units is None or words is None:
                raise UserError("There is no transcript in this project yet; analyze the video first.")
            cues = xm.kept_cues(units, words, keeps)
            text = xm.to_srt(cues) if fmt == "srt" else xm.to_vtt(cues)
        elif kind == "cutlist":
            text = xm.to_edl(keeps, source.name, info.fps, source.stem) if fmt == "edl" else json.dumps(xm.cutlist_json(source.name, info.duration, keeps, info.fps), indent=1)
        else:
            transcript = store.get("transcript")
            if transcript is None:
                raise UserError("There is no transcript in this project yet; analyze the video first.")
            text = json.dumps(transcript, indent=1, ensure_ascii=False) if fmt == "json" else xm.transcript_txt(transcript, None)
        out.write_text(text, encoding="utf-8", newline="\n")

    # ---- keeper provider: read / switch (service-wide, until the service restarts) ----
    @app.get("/v1/keeper")
    def keeper_get():
        return keeper_state()

    @app.put("/v1/keeper")
    async def keeper_put(request: Request):
        body = await body_of(request)
        name = body.get("provider")
        if name not in PROVIDERS:
            raise BadRequest(f"'provider' must be one of {list(PROVIDERS)}.")
        ok, why = availability(name)
        if not ok:
            raise BadRequest(f"Keeper '{name}' is not available: {why}.")
        state["keeper"] = name
        logger.info("keeper provider switched to '%s'", name)
        return keeper_state()

    # ---- analyze ----
    @app.post("/v1/analyze", status_code=202)
    async def analyze(request: Request):
        body = await body_of(request)
        path = need_path(body)
        duration = body.get("duration_seconds")
        if duration is not None and not (_num(duration) and duration > 0):
            raise BadRequest("'duration_seconds' must be a positive number.")
        used = body.get("used")
        if used is not None:
            if not isinstance(used, list) or not all(isinstance(r, (list, tuple)) and len(r) == 2 and _num(r[0]) and _num(r[1]) and r[1] > r[0] for r in used):
                raise BadRequest("'used' must be a list of [start, end] seconds.")
        job_id = jobs.new_id("j")
        store = project_of(body)
        rcfg = cfg_for(store)
        model_name, real_model = pick_model(body, store)
        rcfg = replace(rcfg, transcription={**rcfg.transcription, "model_name": real_model})
        gap = pick_gap(body, store)
        if store is not None:
            guard_replace(store, body, model_name, gap)

        def run(ctx: JobContext):
            cache = None
            if store is not None:
                store.set_source(path)
                cache = ProjectCache(store, meta={"model": model_name}, units_meta={"silence_gap_s": gap})
            doc = analyze_media(path, name=body.get("name"), duration_seconds=duration, used=used, cfg=rcfg, scorer=deps.scorer,
                                transcribe_fn=deps.transcribe_fn, job=job_id, progress=ctx.progress, scorer_provider=scorer_provider, cache=cache,
                                events=ctx.events, audio_dir=store.audio_dir if store else None, silence_gap_s=gap)
            ctx.check()
            if not any(p["class"] == "ambiguous" for p in doc["pairs"]):      # nothing to ask: go straight to the groups
                ctx.progress("Building groups", 0.97)
                doc = build_groups(doc, None, rcfg)
            return _checked(doc, path, doc["media"]["duration_seconds"], job_id)

        jobs.submit("analyze", run, heavy=True, job_id=job_id, events_path=store.progress_path if store else None)
        return {"job": job_id}

    # ---- resolve (stateless: everything comes from the echoed candidates document) ----
    @app.post("/v1/resolve", status_code=202)
    async def resolve(request: Request):
        body = await body_of(request)
        path = need_path(body)
        cand = body.get("candidates")
        if not isinstance(cand, dict) or cand.get("stage") != "candidates" or not isinstance(cand.get("job"), str):
            raise BadRequest("'candidates' must be the candidates document.")
        labels = body.get("labels")
        if labels is not None and not isinstance(labels, dict):
            raise BadRequest("'labels' must be the labels request.")
        rcfg = cfg_for(project_of(body))

        def run(ctx: JobContext):
            ctx.progress("Saving your answers", 0.1)
            record_answers(cand, labels, rcfg)                                   # appended BEFORE the groups are built
            ctx.progress("Building groups", 0.5)
            doc = build_groups(cand, labels, rcfg)
            return _checked(doc, path, cand["media"]["duration_seconds"], cand["job"])

        return {"job": jobs.submit("resolve", run)}

    # ---- adjudicate (optional) ----
    @app.post("/v1/adjudicate", status_code=202)
    async def adjudicate(request: Request):
        body = await body_of(request)
        path = need_path(body)
        name = pick_provider(body.get("keeper_provider"))
        ok, why = availability(name)
        if not ok:
            raise BadRequest(f"AI advice is not available: keeper provider '{name}' cannot run ({why})." if name != "gpt-oss"
                             else "AI advice is not available: no LLM key is configured.")
        groups = body.get("groups")
        if not isinstance(groups, list) or not groups or not all(
            isinstance(g, dict) and isinstance(g.get("id"), str) and isinstance(g.get("sentences"), list) and len(g["sentences"]) >= 2
            and all(isinstance(s, dict) and isinstance(s.get("id"), str) and _num(s.get("start")) and _num(s.get("end")) for s in g["sentences"]) for g in groups):
            raise BadRequest("'groups' must be a non-empty list of groups with at least two sentences each.")

        store = project_of(body)
        cache_where = ProjectCache(store) if store else cfg.cache_dir
        decisions_path = str(store.keeper_decisions_path) if store else cfg.keeper_decisions_path

        def run(ctx: JobContext):
            with ctx.events.stage("keeper_selection", f"Keeper '{name}' on {len(groups)} groups"):
                advice = _advise(ctx)
                if store is not None:
                    store.save_result("keepers", advice)
                return advice

        def _advise(ctx: JobContext):
            if name in ("finetuned", "keep-last"):
                try:
                    return keeper_advice(groups, finetuned if name == "finetuned" else keep_last, ctx, path, cache_where, decisions_path,
                                         body.get("job"))
                finally:
                    if name == "finetuned" and keeper_cfg.unload_after:
                        finetuned.unload()
            advice = adjudicate_groups(groups, llm, ctx)                                # a plain JSON array, never wrapped (unchanged)
            for a in advice:
                record_decision(decisions_path, {"job": body.get("job"), "group": a["group"], "provider": "gpt-oss", "model": llm.config.model if llm.config else llm.provider,
                                                            "decided_by": "gpt-oss", "status": "resolved", "keep": a["keep"], "reason": a["reason"],
                                                            "source": "keeper_advice"})
            return advice

        return {"job": jobs.submit("adjudicate", run, events_path=store.progress_path if store else None)}

    # ---- refine (cut-mode code is on this branch) ----
    @app.post("/v1/refine")
    async def refine(request: Request):
        if not refine_available():
            raise BadRequest("Refine is not available on this service.")
        body = await body_of(request)
        path = need_path(body)
        remove, mode = body.get("remove"), body.get("mode", "anchored")
        if not isinstance(remove, list) or not remove or not all(isinstance(r, str) for r in remove):
            raise BadRequest("'remove' must be a non-empty list of sentence ids.")
        store = project_of(body)
        log = EventLog(jobs.new_id("c"), store.progress_path if store else None)          # refine is synchronous: its events go straight to the project log
        try:
            with log.stage("cutlist_build", f"Mode {mode}, {len(remove)} sentences removed"):
                result = await run_in_threadpool(refine_ranges, path, remove, mode, ProjectCache(store) if store else cfg.cache_dir)
                if store is not None:
                    store.save_result("cutlist", result)
                return result       # ffmpeg work must not block the event loop
        except FileNotFoundError:
            raise BadRequest("Could not read the file")
        except (ValueError, UserError) as exc:
            raise BadRequest(str(exc))

    # ---- jobs ----
    @app.get("/v1/jobs/{job_id}")
    def job_state(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            return JSONResponse({"error": "Unknown job."}, status_code=404)
        return job.snapshot()

    @app.get("/v1/jobs/{job_id}/events")
    def job_events(job_id: str, after: int = 0):
        """The job's stage events after sequence number `after` (poll with the last seq you saw). Not a stream: see /stream."""
        job = jobs.get(job_id)
        if job is None:
            return JSONResponse({"error": "Unknown job."}, status_code=404)
        return {"events": job.events.since(max(0, after)), "finished": job.events.closed}

    @app.get("/v1/jobs/{job_id}/events/stream")
    async def job_events_stream(job_id: str, request: Request, after: int = 0):
        """Server-Sent Events: one `stage` event per pipeline event; resumes after `Last-Event-ID` / ?after=. The loop only sleeps and reads memory, never blocks."""
        job = jobs.get(job_id)
        if job is None:
            return JSONResponse({"error": "Unknown job."}, status_code=404)
        try:
            seq = max(after, int(request.headers.get("last-event-id", 0)))
        except ValueError:
            seq = after

        async def gen():
            nonlocal seq
            idle = 0.0
            while True:
                evs = job.events.since(seq)
                for ev in evs:
                    seq = ev["seq"]
                    yield sse_format(ev)
                if not evs and job.events.closed:
                    yield "event: end\ndata: {}\n\n"
                    return
                idle = 0.0 if evs else idle + 0.15
                if idle >= 15.0:
                    idle = 0.0
                    yield ": keep-alive\n\n"
                if await request.is_disconnected():
                    return
                await asyncio.sleep(0.15)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/v1/project/events")
    def project_events(project: str):
        """The persisted log of a project (data/progress.jsonl): kept after the jobs are gone."""
        store = ProjectStore(project)
        return {"events": read_log(store.progress_path)}

    @app.delete("/v1/jobs/{job_id}", status_code=204)
    def job_cancel(job_id: str):
        jobs.cancel(job_id)
        return Response(status_code=204)

    return app


def _checked(doc: dict, path: str, claimed_duration: float, job: str) -> dict:
    """Every document must pass the editor's validation before it is sent; otherwise the job fails with a plain message."""
    from pipeline.step_08_video_edit import probe_media

    try:
        file_duration = probe_media(path)[0]
    except Exception:
        file_duration = None
    issues = validate_document(doc, file_duration=file_duration or claimed_duration, expected_job=job)
    if issues:
        logger.error("document failed validation, not sent: %s", "; ".join(issues[:10]))
        raise UserError("The retake service produced a result the editor would reject, so it was not sent.")
    return doc
