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

from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from pipeline.step_07_review import ReviewConfig, analyze_media, build_groups, validate_document
from pipeline.step_07_review.labels import record_answers

from .config import ServiceConfig
from .events import EventLog, read_log, sse_format
from .jobs import JobContext, JobManager, UserError
from .project_store import ProjectCache, ProjectError, ProjectStore
from .llm import LLMSetup, adjudicate_groups, detect_llm, keeper_advice, record_decision
from .refine import refine_available, refine_ranges

logger = logging.getLogger("service")


class BadRequest(Exception):
    pass


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

    def need_path(body: dict) -> str:
        path = body.get("path")
        if not isinstance(path, str) or not path.strip():
            raise BadRequest("'path' is required.")
        return path

    # ---- capabilities ----
    @app.get("/v1/capabilities")
    def capabilities():
        keeper = keeper_state()
        return {"adjudicate": bool(keeper["providers"][keeper["current"]]["available"]), "refine": refine_available(), "keeper": keeper}

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

        def run(ctx: JobContext):
            cache = None
            if store is not None:
                store.set_source(path)
                cache = ProjectCache(store)
            doc = analyze_media(path, name=body.get("name"), duration_seconds=duration, used=used, cfg=rcfg, scorer=deps.scorer,
                                transcribe_fn=deps.transcribe_fn, job=job_id, progress=ctx.progress, scorer_provider=scorer_provider, cache=cache,
                                events=ctx.events, audio_dir=store.audio_dir if store else None)
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
                return _advise(ctx)

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
                return await run_in_threadpool(refine_ranges, path, remove, mode, ProjectCache(store) if store else cfg.cache_dir)       # ffmpeg work must not block the event loop
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
