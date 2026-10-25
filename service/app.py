"""FastAPI app: the local retake service for the SimpleCut editor (backend-handoff.md section 3, schema simplecut.retakes/2). 127.0.0.1, no auth."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from pipeline.step_07_review import ReviewConfig, analyze_media, build_groups, validate_document
from pipeline.step_07_review.labels import record_answers

from .config import ServiceConfig
from .jobs import JobContext, JobManager, UserError
from .llm import LLMSetup, adjudicate_groups, detect_llm
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

    def need_path(body: dict) -> str:
        path = body.get("path")
        if not isinstance(path, str) or not path.strip():
            raise BadRequest("'path' is required.")
        return path

    # ---- capabilities ----
    @app.get("/v1/capabilities")
    def capabilities():
        return {"adjudicate": bool(llm.available), "refine": refine_available()}

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

        def run(ctx: JobContext):
            doc = analyze_media(path, name=body.get("name"), duration_seconds=duration, used=used, cfg=review_cfg, scorer=deps.scorer,
                                transcribe_fn=deps.transcribe_fn, job=job_id, progress=ctx.progress, scorer_provider=scorer_provider)
            ctx.check()
            if not any(p["class"] == "ambiguous" for p in doc["pairs"]):      # nothing to ask: go straight to the groups
                ctx.progress("Building groups", 0.97)
                doc = build_groups(doc, None, review_cfg)
            return _checked(doc, path, doc["media"]["duration_seconds"], job_id)

        jobs.submit("analyze", run, heavy=True, job_id=job_id)
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

        def run(ctx: JobContext):
            ctx.progress("Saving your answers", 0.1)
            record_answers(cand, labels, review_cfg)                                   # appended BEFORE the groups are built
            ctx.progress("Building groups", 0.5)
            doc = build_groups(cand, labels, review_cfg)
            return _checked(doc, path, cand["media"]["duration_seconds"], cand["job"])

        return {"job": jobs.submit("resolve", run)}

    # ---- adjudicate (optional) ----
    @app.post("/v1/adjudicate", status_code=202)
    async def adjudicate(request: Request):
        body = await body_of(request)
        need_path(body)
        if not llm.available:
            raise BadRequest("AI advice is not available: no LLM key is configured.")
        groups = body.get("groups")
        if not isinstance(groups, list) or not groups or not all(
            isinstance(g, dict) and isinstance(g.get("id"), str) and isinstance(g.get("sentences"), list) and len(g["sentences"]) >= 2
            and all(isinstance(s, dict) and isinstance(s.get("id"), str) and _num(s.get("start")) and _num(s.get("end")) for s in g["sentences"]) for g in groups):
            raise BadRequest("'groups' must be a non-empty list of groups with at least two sentences each.")

        def run(ctx: JobContext):
            return adjudicate_groups(groups, llm, ctx)                                  # a plain JSON array, never wrapped

        return {"job": jobs.submit("adjudicate", run)}

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
        try:
            return await run_in_threadpool(refine_ranges, path, remove, mode, cfg.cache_dir)       # ffmpeg work must not block the event loop
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
