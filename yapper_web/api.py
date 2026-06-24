"""FastAPI service layer.

No sign-in: a signed, HttpOnly session cookie (1-year TTL) is the identity. Everything
is namespaced by ``session_id``; every query filters by it. The browser uploads the
movie straight to S3 with a presigned PUT (the API never touches the large file), then
the front half is enqueued; a per-language run kicks off the back half after a pre-flight
budget check. Progress streams over SSE; the finished commentary video is fetched via a presigned GET.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Cookie, Depends, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, URLSafeSerializer
from pydantic import BaseModel
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from yapper.config import _slugify
from yapper.pipeline import BACK_HALF, FRONT_HALF, STAGE_COMPUTE, SUPPORTED_LANGS

from . import logging_config, metrics
from .budget import Budget
from .db import (
    Movie,
    MovieStage,
    MovieStatus,
    Run,
    RunStage,
    RunStatus,
    init_db,
    session_scope,
    stage_avg_seconds,
    touch_session,
)
from .settings import get_settings

log = logging.getLogger("yapper_web.api")
S = get_settings()
_signer = URLSafeSerializer(S.session_secret, salt="jf-session")
_STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging_config.configure()
    init_db()
    metrics.register_api_collector()
    yield


app = FastAPI(title="Yapper 短片解說", docs_url="/api/docs", lifespan=lifespan)


# ---------------------------------------------------------------------------
# session: signed cookie, minted on first request, valid 1 year
# ---------------------------------------------------------------------------
def get_session(
    response: Response, jf_session: str | None = Cookie(default=None)
) -> str:
    sid: str | None = None
    if jf_session:
        try:
            sid = _signer.loads(jf_session)
        except BadSignature:
            sid = None
    if not sid:
        import uuid

        sid = str(uuid.uuid4())
        response.set_cookie(
            S.session_cookie,
            _signer.dumps(sid),
            max_age=S.session_max_age,
            httponly=True,
            samesite="lax",
            secure=S.cookie_secure,
            path="/",
        )
    with session_scope() as db:
        touch_session(db, sid)
    return sid


SessionId = Depends(get_session)


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------
# Browsers must PUT with EXACTLY this Content-Type so it matches what we sign into the presigned
# URL (deterministic match — avoids the iOS empty-file.type signature mismatch). The real format is
# detected later by ffmpeg, so a constant here is fine.
UPLOAD_CONTENT_TYPE = "application/octet-stream"


class CreateMovie(BaseModel):
    filename: str
    content_type: str = (
        "video/mp4"  # advisory only; the upload is signed as UPLOAD_CONTENT_TYPE
    )
    lang: str = "zh"  # output language chosen at upload; back half auto-runs in it


class CreateRun(BaseModel):
    movie_id: str
    lang: str = "zh"
    force: bool = (
        False  # regenerate a fresh recap even if one already exists for this language
    )


class RetryMovie(BaseModel):
    # "resume": re-run from the failed stage, reusing every cached artifact before it.
    # "restart": clear all artifacts (front + back) and re-run from the first stage (ingest),
    #            keeping only the uploaded source so the user never re-uploads.
    mode: str = "resume"
    lang: str | None = None  # which language run to resume (back half); defaults to the movie's


def _stage_rows(stages: list | None) -> list[dict]:
    # ``at`` is the row's last-update epoch; for the currently-running stage that's its
    # start time, so the browser can show a live elapsed timer (now - at).
    return [
        {
            "stage": s.stage,
            "status": s.status,
            "seconds": s.seconds,
            "at": s.recorded_at.timestamp() if s.recorded_at else None,
        }
        for s in (stages or [])
    ]


# Cold-start ETA seeds (seconds) — rough per-stage guesses used until real history exists; once
# runs accumulate, stage_avg_seconds() replaces them. Keys are the FRONT_HALF/BACK_HALF stages.
_STAGE_SEED = {
    "ingest": 5,
    "audio": 15,
    "asr": 180,
    "shots": 40,
    "scenes": 50,
    "understand": 80,
    "script": 120,
    "budget": 1,
    "tts": 150,
    "edl": 5,
    "subs": 3,
    "render": 200,
}
_ETA_CACHE: dict[str, tuple[float, dict]] = (
    {}
)  # scope -> (expires_epoch, {stage: avg_sec})
_ETA_TTL = 60.0


def _stage_avgs(scope: str) -> dict:
    """Per-stage historical averages, cached briefly so list/stream requests don't re-query."""
    cached = _ETA_CACHE.get(scope)
    now = time.time()
    if cached and cached[0] > now:
        return cached[1]
    try:
        with session_scope() as db:
            avgs = stage_avg_seconds(db, scope)
    except Exception:  # noqa: BLE001 — ETA is best-effort; never break a view
        avgs = {}
    _ETA_CACHE[scope] = (now + _ETA_TTL, avgs)
    return avgs


def _eta(scope: str, all_stages: list[str], stages: list | None) -> dict:
    """Estimate total/elapsed/remaining seconds for a movie (front half) or run (back half).
    Remaining = sum of historical averages (or seeds) for stages not yet finished; total tightens
    toward reality as real stage durations replace estimates. Approximate — a ballpark expectation.
    """
    avgs = _stage_avgs(scope)

    def est(stage: str) -> float:
        return float(avgs.get(stage) or _STAGE_SEED.get(stage, 30))

    done = {
        s.stage: s for s in (stages or []) if s.status in ("ran", "cached", "error")
    }
    elapsed = sum(float(s.seconds or 0.0) for s in done.values())
    remaining = sum(est(s) for s in all_stages if s not in done)
    return {
        "estimated_total_sec": round(elapsed + remaining),
        "elapsed_sec": round(elapsed),
        "estimated_remaining_sec": round(remaining),
    }


def _movie_view(m: Movie, stages: list | None = None) -> dict:
    return {
        "id": m.id,
        "filename": m.original_filename,
        "slug": m.slug,
        "status": m.status.value,
        "duration_sec": m.duration_sec,
        "error": m.error,
        "has_source": bool(m.source_key) and m.status != MovieStatus.registered,
        "now": time.time(),  # server clock, for skew-free elapsed in the UI
        "all_stages": list(
            FRONT_HALF
        ),  # ordered front-half pipeline (ingest -> scenes)
        "stage_kinds": STAGE_COMPUTE,  # stage -> "llm"|"gpu"|"cpu" for UI styling
        "stages": _stage_rows(stages),  # per-stage timing/state recorded so far
        **_eta("movie", FRONT_HALF, stages),  # estimated_total/elapsed/remaining_sec
    }


def _run_view(r: Run, stages: list | None = None) -> dict:
    return {
        "id": r.id,
        "movie_id": r.movie_id,
        "lang": r.lang,
        "status": r.status.value,
        "llm_tokens_in": r.llm_tokens_in,
        "llm_tokens_out": r.llm_tokens_out,
        "llm_cost_usd": round(r.llm_cost_usd, 5),
        "output_duration_sec": r.output_duration_sec,
        "has_output": bool(r.output_key),
        "error": r.error,
        "now": time.time(),  # server clock, for skew-free elapsed in the UI
        "all_stages": list(
            BACK_HALF
        ),  # ordered back-half pipeline (understand -> render)
        "stage_kinds": STAGE_COMPUTE,  # stage -> "llm"|"gpu"|"cpu" for UI styling
        "stages": _stage_rows(stages),
        **_eta("run", BACK_HALF, stages),  # estimated_total/elapsed/remaining_sec
    }


# ---------------------------------------------------------------------------
# movies
# ---------------------------------------------------------------------------
@app.post("/api/movies")
def create_movie(body: CreateMovie, sid: str = SessionId):
    from .storage import storage_for_web

    slug = _slugify(Path(body.filename).stem)
    ext = Path(body.filename).suffix or ".mp4"
    # Output language picked at upload time; the front half auto-starts the recap in it (one
    # step). Coerce anything unsupported to the default so an upload is never blocked on it.
    lang = body.lang.lower()
    if lang not in SUPPORTED_LANGS:
        lang = SUPPORTED_LANGS[0]
    with session_scope() as db:
        # MAX_MOVIES_PER_SESSION <= 0 disables the per-session cap (unlimited uploads).
        if S.max_movies_per_session > 0:
            n = db.query(Movie).filter(Movie.session_id == sid).count()
            if n >= S.max_movies_per_session:
                raise HTTPException(
                    429, f"movie limit reached ({S.max_movies_per_session} per session)"
                )
        movie = Movie(
            session_id=sid,
            original_filename=body.filename,
            slug=slug,
            s3_prefix="",
            source_key="",
            status=MovieStatus.registered,
            default_lang=lang,
        )
        db.add(movie)
        db.flush()
        mid = movie.id
        # scope by the unique movie_id so two same-named uploads in one session don't collide
        movie.s3_prefix = f"{sid}/{mid}/{slug}"
        source_key = f"sources/{sid}/{mid}/{slug}{ext}"
        movie.source_key = source_key
    store = storage_for_web(S)
    # Sign a FIXED Content-Type and make the browser send EXACTLY that (see UPLOAD_CONTENT_TYPE).
    # The old bug: we signed the file's MIME type but the PUT only sent it `if (f.type)`; iOS often
    # reports an empty/quirky file.type, so signed != sent -> 403 SignatureDoesNotMatch on every
    # mobile upload. A constant on both ends always matches (MinIO is strict about this; AWS too).
    # The stored Content-Type is irrelevant — the worker re-downloads and ffmpeg sniffs the format.
    upload_url = store.presign_put(
        source_key, expires=S.upload_url_ttl, content_type=UPLOAD_CONTENT_TYPE
    )
    return {
        "movie_id": mid,
        "upload_url": upload_url,
        "method": "PUT",
        "source_key": source_key,
        "upload_content_type": UPLOAD_CONTENT_TYPE,
    }


@app.post("/api/movies/{movie_id}/complete")
def complete_movie(movie_id: str, sid: str = SessionId):
    from .storage import storage_for_web

    from .tasks import start_front_half

    with session_scope() as db:
        movie = db.get(Movie, movie_id)
        if movie is None or movie.session_id != sid:
            raise HTTPException(404, "movie not found")
        source_key = movie.source_key
        already = movie.status not in (MovieStatus.registered, MovieStatus.error)
    store = storage_for_web(S)
    if not store.exists(source_key):
        raise HTTPException(
            400, "upload not found in storage; PUT the file to upload_url first"
        )
    if not already:
        with session_scope() as db:
            db.get(Movie, movie_id).status = MovieStatus.uploaded
        start_front_half(movie_id)
    return {"ok": True, "movie_id": movie_id}


@app.get("/api/movies")
def list_movies(sid: str = SessionId):
    with session_scope() as db:
        movies = db.scalars(
            select(Movie)
            .where(Movie.session_id == sid)
            .order_by(Movie.uploaded_at.desc())
        ).all()
        return {"movies": [_movie_view(m, m.stages) for m in movies]}


@app.get("/api/movies/{movie_id}/source-url")
def movie_source_url(movie_id: str, sid: str = SessionId):
    """Presigned GET for the original uploaded movie, so the browser can stream it in a
    player (S3/MinIO honour range requests, so large files seek without downloading)."""
    from .storage import storage_for_web

    with session_scope() as db:
        movie = db.get(Movie, movie_id)
        if movie is None or movie.session_id != sid:
            raise HTTPException(404, "movie not found")
        if not movie.source_key or movie.status == MovieStatus.registered:
            raise HTTPException(409, "source not uploaded yet")
        key, filename = movie.source_key, movie.original_filename
    store = storage_for_web(S)
    return JSONResponse(
        {"url": store.presign_get(key, expires=S.result_url_ttl), "filename": filename}
    )


@app.get("/api/movies/{movie_id}/runs")
def list_movie_runs(movie_id: str, sid: str = SessionId):
    with session_scope() as db:
        movie = db.get(Movie, movie_id)
        if movie is None or movie.session_id != sid:
            raise HTTPException(404, "movie not found")
        runs = db.scalars(
            select(Run).where(Run.movie_id == movie_id).order_by(Run.created_at)
        ).all()
        return {"runs": [_run_view(r, r.stages) for r in runs]}


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------
@app.post("/api/runs")
def create_run(body: CreateRun, sid: str = SessionId):
    from .tasks import clear_back_half_artifacts, start_back_half

    lang = body.lang.lower()
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(
            400, f"unsupported lang {lang!r}; expected {SUPPORTED_LANGS}"
        )
    slug = None
    with session_scope() as db:
        movie = db.get(Movie, body.movie_id)
        if movie is None or movie.session_id != sid:
            raise HTTPException(404, "movie not found")
        if movie.status != MovieStatus.ready:
            raise HTTPException(409, f"movie not ready (status={movie.status.value})")
        slug = movie.slug
        # Idempotent per (movie, lang) — return the existing recap — UNLESS force=True, which
        # regenerates a fresh one (any language, even one that's already done).
        existing = db.scalar(
            select(Run).where(Run.movie_id == body.movie_id, Run.lang == lang)
        )
        if (
            existing is not None
            and existing.status != RunStatus.error
            and not body.force
        ):
            return _run_view(existing)
        # per-session concurrency cap (a finished run being regenerated doesn't count as active)
        active = (
            db.query(Run)
            .filter(
                Run.session_id == sid,
                Run.status.in_([RunStatus.queued, RunStatus.running]),
            )
            .count()
        )
        if active >= S.max_concurrent_runs_per_session:
            raise HTTPException(
                429, f"too many concurrent runs ({S.max_concurrent_runs_per_session})"
            )

        # No money budget: enqueue straight away (token/cost usage is still recorded per
        # run for display). The concurrency cap above protects the GPU box, not a wallet.
        run = existing
        if run is None:
            run = Run(
                movie_id=body.movie_id,
                session_id=sid,
                lang=lang,
                status=RunStatus.queued,
            )
            db.add(run)
        else:  # re-queue: clear the prior result so the UI resets
            run.status = RunStatus.queued
            run.error = None
            run.output_key = None
            run.output_duration_sec = None
        db.flush()
        rid = run.id
    # Regenerate: drop the cached back-half artifacts for this (movie, lang) so the chain
    # recomputes a FRESH recap rather than returning the old one. The shared front-half cache is
    # left intact (ASR/scenes aren't redone). Must happen before the chain re-materializes from S3.
    if body.force:
        clear_back_half_artifacts(sid, body.movie_id, slug, lang)
    start_back_half(rid)
    with session_scope() as db:
        return _run_view(db.get(Run, rid))


@app.post("/api/movies/{movie_id}/retry")
def retry_movie(movie_id: str, body: RetryMovie, sid: str = SessionId):
    """Recover a pipeline that stopped on a stage error — without re-uploading the video.

    - ``mode="resume"``: continue from the failed stage. A failed front half (movie error) re-runs
      the front-half chain (cached stages are skipped, so only the failed step onward recomputes),
      then auto-starts the back half; a failed back half re-runs that chain for the language.
    - ``mode="restart"``: wipe every cached artifact (front + back) and re-run from the first stage
      (ingest). The uploaded source is kept, so it's a clean slate without a re-upload.
    """
    from .tasks import clear_movie_artifacts, start_back_half, start_front_half

    mode = body.mode.lower()
    if mode not in ("resume", "restart"):
        raise HTTPException(400, f"unknown mode {body.mode!r}; expected 'resume' or 'restart'")

    with session_scope() as db:
        movie = db.get(Movie, movie_id)
        if movie is None or movie.session_id != sid:
            raise HTTPException(404, "movie not found")
        if movie.status == MovieStatus.registered:
            raise HTTPException(409, "nothing to retry — the upload hasn't been completed")
        slug = movie.slug
        lang = (body.lang or movie.default_lang or SUPPORTED_LANGS[0]).lower()
        front_failed = movie.status == MovieStatus.error
        target_run = db.scalar(
            select(Run).where(Run.movie_id == movie_id, Run.lang == lang)
        )
        back_failed = target_run is not None and target_run.status == RunStatus.error
        if mode == "resume" and not front_failed and not back_failed:
            raise HTTPException(409, "nothing to resume — no stage is in an error state")
        # Protect the GPU box: same per-session concurrency cap as a fresh run (the failed
        # run we're reviving isn't counted as active, so a recovery is never blocked by itself).
        active = (
            db.query(Run)
            .filter(
                Run.session_id == sid,
                Run.status.in_([RunStatus.queued, RunStatus.running]),
            )
            .count()
        )
        if active >= S.max_concurrent_runs_per_session:
            raise HTTPException(
                429, f"too many concurrent runs ({S.max_concurrent_runs_per_session})"
            )

    # --- restart: clear everything, re-run from ingest (keep the upload) -----------------
    if mode == "restart":
        clear_movie_artifacts(sid, movie_id, slug)
        with session_scope() as db:
            movie = db.get(Movie, movie_id)
            movie.status = MovieStatus.uploaded
            movie.error = None
            movie.duration_sec = None
            db.query(MovieStage).filter(MovieStage.movie_id == movie_id).delete()
            # Drop the runs (and their stage rows): the front half re-runs and ``mark_ready``
            # auto-starts a fresh back-half run for the movie's default language. Leaving an old
            # queued/done row would make ``mark_ready`` skip the auto-start (it only revives an
            # *errored* run), and a stale 'done' row would point at an artifact we just deleted.
            run_ids = [r.id for r in db.query(Run).filter(Run.movie_id == movie_id).all()]
            for rid in run_ids:
                db.query(RunStage).filter(RunStage.run_id == rid).delete()
            db.query(Run).filter(Run.movie_id == movie_id).delete()
        start_front_half(movie_id)
        return {"ok": True, "mode": "restart", "movie_id": movie_id}

    # --- resume: front half failed -> re-run the front chain (skips cached stages) -------
    if front_failed:
        with session_scope() as db:
            movie = db.get(Movie, movie_id)
            movie.status = MovieStatus.uploaded
            movie.error = None
            db.query(MovieStage).filter(
                MovieStage.movie_id == movie_id, MovieStage.status == "error"
            ).delete()
        start_front_half(movie_id)  # cached front stages skip; failed stage onward re-runs, then auto back half
        return {"ok": True, "mode": "resume", "scope": "front", "movie_id": movie_id}

    # --- resume: back half failed -> re-run the back chain for this language -------------
    with session_scope() as db:
        run = db.scalar(select(Run).where(Run.movie_id == movie_id, Run.lang == lang))
        if run is None or run.status != RunStatus.error:
            raise HTTPException(409, f"nothing to resume for language {lang!r}")
        run.status = RunStatus.queued
        run.error = None
        run.output_key = None
        run.output_duration_sec = None
        db.query(RunStage).filter(
            RunStage.run_id == run.id, RunStage.status == "error"
        ).delete()
        rid = run.id
    start_back_half(rid)  # no artifact clearing -> the pipeline resumes at the failed back-half stage
    return {"ok": True, "mode": "resume", "scope": "back", "run_id": rid}


def _fetch_run(run_id: str, sid: str) -> dict:
    with session_scope() as db:
        run = db.get(Run, run_id)
        if run is None or run.session_id != sid:
            raise HTTPException(404, "run not found")
        return _run_view(run, run.stages)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, sid: str = SessionId):
    return _fetch_run(run_id, sid)


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str, sid: str = SessionId):
    """Server-sent events: poll the run row until it reaches a terminal state."""

    async def stream():
        last = None
        for _ in range(3600):  # safety bound (~2h at 2s)
            try:
                view = await run_in_threadpool(_fetch_run, run_id, sid)
            except HTTPException:
                yield f"event: error\ndata: {json.dumps({'error': 'not found'})}\n\n"
                return
            # Dedup on real state, ignoring the volatile ``now``: the browser ticks the
            # current stage's elapsed locally, so we only push on an actual change.
            sig = json.dumps(
                {k: v for k, v in view.items() if k != "now"}, ensure_ascii=False
            )
            if sig != last:
                yield f"data: {json.dumps(view, ensure_ascii=False)}\n\n"
                last = sig
            if view["status"] in ("done", "error"):
                return
            await asyncio.sleep(2)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/runs/{run_id}/result")
def run_result(run_id: str, sid: str = SessionId):
    from .storage import storage_for_web

    with session_scope() as db:
        run = db.get(Run, run_id)
        if run is None or run.session_id != sid:
            raise HTTPException(404, "run not found")
        if not run.output_key:
            raise HTTPException(409, "result not ready")
        key = run.output_key
    store = storage_for_web(S)
    return JSONResponse({"url": store.presign_get(key, expires=S.result_url_ttl)})


# ---------------------------------------------------------------------------
# budget + ops
# ---------------------------------------------------------------------------
@app.get("/api/budget")
def budget_status(sid: str = SessionId):
    b = Budget()
    return {
        "remaining_usd": round(b.remaining_usd(), 4),
        "spent_usd": round(b.spent_usd(), 4),
        "cap_usd": S.llm_max_cost_usd,
        "per_run_estimate_usd": round(S.run_cost_estimate_usd(), 4),
    }


@app.get("/metrics")
def prometheus_metrics():
    body, content_type = metrics.render_latest()
    return Response(content=body, media_type=content_type)


@app.get("/healthz")
def healthz():
    return {"ok": True}


# static frontend (mounted last so /api/* and /metrics win)
if _STATIC.exists():

    @app.get("/")
    def index():
        return FileResponse(_STATIC / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        # Browsers (and the /api/docs Swagger page) probe /favicon.ico at the root; serve the
        # one SVG icon for all of them (modern browsers render SVG favicons fine).
        return FileResponse(_STATIC / "favicon.svg", media_type="image/svg+xml")

    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
