"""Celery orchestration: resource-routed tasks that wrap the pipeline.

The pipeline functions are coarse (``run_front_half`` / ``run_back_half``) but both
support ``until`` + artifact caching, so we chain them into per-resource phase tasks
*without changing any stage logic* — each phase re-enters the function, skips cached
stages, and runs only its slice:

    front half:  fh_local (cpu)  -> fh_asr (gpu) -> fh_scenes (cpu) -> mark_ready
    back half:   bh_script (llm) -> bh_tts (gpu) -> bh_render (render) -> finalize

Queue concurrency enforces the physical limits (gpu=1, llm small, render≈n_cpu); the
GPU services are additionally guarded by a Redis lock so correctness survives scaling
the gpu worker past one. Every task stages the run's working dir in from S3, runs its
slice, and stages new artifacts back out — cheap on a warm single-host cache, correct
on a cold multi-host one.
"""

from __future__ import annotations

import json
import logging
import shutil
from contextlib import contextmanager
from pathlib import Path

import redis
from celery import Celery, chain
from celery.signals import task_prerun

from yapper import pipeline
from yapper.config import Config, load_config
from yapper.storage import Storage
from yapper.timing import RunTimer

from . import gpu_supervisor, logging_config
from .budget import Budget, BudgetGuard, StageUsageLedger
from .storage import storage_for_web
from .db import (
    Movie,
    MovieStatus,
    Run,
    RunStage,
    RunStatus,
    init_db,
    record_stage_event,
    record_timings,
    session_scope,
)
from .metrics import push_run_metrics
from .settings import get_settings

log = logging.getLogger("yapper_web.tasks")
S = get_settings()

app = Celery("yapper", broker=S.broker_url, backend=S.result_backend)
app.conf.update(
    task_acks_late=True,                  # re-deliver if a worker dies mid-task
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,         # long tasks: don't hoard the queue
    task_track_started=True,
    task_default_queue="cpu",
    # Long stages (TTS especially) can run for many minutes. Redis' DEFAULT broker
    # visibility_timeout is 1h: with acks_late on, a task still running after that is assumed
    # dead and REDELIVERED to another worker — so it executes twice, the duplicate corrupts the
    # stage rows / breaks the chain, and the run gets stuck "running" even though the audio
    # finished. Widen the window well past any real task so a slow TTS is never redelivered,
    # and add a hard time limit BELOW it so a genuinely-stuck task is killed (not duplicated).
    broker_transport_options={"visibility_timeout": 21600},          # 6h
    result_backend_transport_options={"visibility_timeout": 21600},
    broker_connection_retry_on_startup=True,
    task_soft_time_limit=18000,           # 5h: catchable SoftTimeLimitExceeded for cleanup
    task_time_limit=18600,                # 5h10m: hard SIGKILL backstop (< visibility_timeout)
    task_routes={
        "yapper_web.tasks.fh_local": {"queue": "cpu"},
        # ASR and TTS live on DIFFERENT physical GPUs, so they get separate queues and
        # can run concurrently; each queue is still serialized within itself (one model
        # per GPU) by a single-concurrency worker + the per-service Redis lock.
        "yapper_web.tasks.fh_asr": {"queue": "asr"},
        "yapper_web.tasks.fh_scenes": {"queue": "cpu"},
        "yapper_web.tasks.mark_ready": {"queue": "cpu"},
        "yapper_web.tasks.bh_script": {"queue": "llm"},
        "yapper_web.tasks.bh_tts": {"queue": "tts"},
        "yapper_web.tasks.bh_render": {"queue": "render"},
        "yapper_web.tasks.finalize_run": {"queue": "render"},
    },
)

_redis = redis.Redis.from_url(S.redis_url)


@task_prerun.connect
def _bind_log_ctx(task_id=None, task=None, args=None, **_):
    logging_config.configure()


# ---------------------------------------------------------------------------
# storage + working-dir helpers
# ---------------------------------------------------------------------------
def storage() -> Storage:
    return storage_for_web(S)


def _movie_root(session_id: str, movie_id: str) -> Path:
    """Per-movie working cache, scoped by the unique movie_id (so same-named uploads in
    one session never share a dir)."""
    return Path(S.work_root) / session_id / movie_id


def _local_movie_path(session_id: str, movie_id: str, slug: str, source_key: str) -> Path:
    """Local source path whose stem slugifies back to ``slug`` (so Config.movie_dir
    resolves to <movie_root>/<slug>). Kept outside movie_dir so it isn't persisted."""
    ext = Path(source_key).suffix or ".mp4"
    p = _movie_root(session_id, movie_id) / "sources" / f"{slug}{ext}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _build_cfg(session_id: str, movie_id: str) -> Config:
    """Load pipeline config and point artifacts + scratch at this movie's working cache,
    so the pipeline writes there and Config.movie_dir == <movie_root>/<slug>."""
    cfg = load_config(S.pipeline_config)
    root = _movie_root(session_id, movie_id)
    cfg.raw.setdefault("paths", {})["artifacts_dir"] = str(root)
    cfg.raw["paths"]["scratch_dir"] = str(root / "_scratch")
    return cfg


def _artifact_prefix(session_id: str, movie_id: str, slug: str) -> str:
    return f"{session_id}/{movie_id}/{slug}"


def _prepare(session_id: str, movie_id: str, slug: str, source_key: str) -> tuple[Path, Config, Storage, str]:
    """Stage in: download the source movie + materialize existing artifacts from S3.
    Takes primitives (not a live ORM row) so it runs outside any DB transaction."""
    store = storage()
    movie_path = _local_movie_path(session_id, movie_id, slug, source_key)
    if not movie_path.exists() or movie_path.stat().st_size == 0:
        store.download(source_key, movie_path)
    cfg = _build_cfg(session_id, movie_id)
    prefix = _artifact_prefix(session_id, movie_id, slug)
    store.materialize(prefix, cfg.movie_dir(str(movie_path)))
    return movie_path, cfg, store, prefix


def _movie_fields(movie_id: str) -> tuple[str, str, str]:
    """(session_id, slug, source_key) for a movie, read in a short transaction."""
    with session_scope() as db:
        m = db.get(Movie, movie_id)
        return m.session_id, m.slug, m.source_key


def _persist(store: Storage, cfg: Config, movie_path: Path, prefix: str) -> None:
    store.persist(cfg.movie_dir(str(movie_path)), prefix)


@contextmanager
def _gpu_lock(name: str, ttl: int = 3600):
    """Serialize access to the single always-on GPU server (flag-off path). One Redis
    lock => one in-flight request, matching one model per GPU."""
    lock = _redis.lock(f"lock:{name}", timeout=ttl, blocking_timeout=ttl)
    acquired = lock.acquire()
    try:
        yield acquired
    finally:
        if acquired:
            try:
                lock.release()
            except redis.exceptions.LockError:
                pass


@contextmanager
def _gpu_slot(service: str):
    """Yield the ASR/TTS target the pipeline should dial.

    With gpud (GPU_SUPERVISOR_TARGET set): lease an on-demand instance — admission and
    serialization are handled by the pool, so up to MAX run concurrently. Without it:
    serialize on the single always-on server via the Redis lock and yield ``None`` (the
    pipeline falls back to the static *_GRPC_TARGET env)."""
    with gpu_supervisor.lease(service) as target:
        if target is None:
            with _gpu_lock(service):
                yield None
        else:
            yield target


@contextmanager
def _gpu_slots(service: str, count: int):
    """Like :func:`_gpu_slot` but lease up to ``count`` instances for parallel work (TTS).

    With gpud: lease up to ``count`` on-demand instances (degrades to whatever's free, >=1)
    and yield their targets. Without gpud: serialize on the single always-on server via the
    Redis lock and yield ``[None]`` (the pipeline falls back to the static env target)."""
    with gpu_supervisor.lease_many(service, count) as targets:
        if targets == [None]:
            with _gpu_lock(service):
                yield [None]
        else:
            yield targets


def _stage_recorder(scope: str, obj_id: str):
    """Build a RunTimer ``on_event`` callback that mirrors live stage transitions into
    the DB (``movie_stages`` for the front half, ``run_stages`` for the back half) so
    the UI can show the current step and per-stage timing as it happens."""
    def _on_event(stage: str, seconds: float, status: str) -> None:
        try:
            with session_scope() as db:
                record_stage_event(db, scope, obj_id, stage, seconds, status)
        except Exception:  # noqa: BLE001 — progress tracking must never break a task
            log.debug("stage event not recorded (%s %s/%s)", scope, obj_id, stage, exc_info=True)
    return _on_event


def _llm_stage_tracking(scope: str, obj_id: str, guard: BudgetGuard):
    """Wiring for the LLM back half so spend/tokens are attributed to the *current* stage
    (understand vs script). Returns ``(on_event, on_usage)``:

    - ``on_event`` mirrors stage transitions to the DB (as :func:`_stage_recorder`) AND
      tracks which stage is running. The back half is single-threaded within the task, so
      by the time an LLM call fires the holder names the stage that issued it.
    - ``on_usage`` does the authoritative budget accounting first (``guard.post``), then
      records the per-stage tally — guarded so a metrics failure never disturbs the run.
    """
    record = _stage_recorder(scope, obj_id)
    current: dict[str, str | None] = {"stage": None}
    ledger = StageUsageLedger()

    def on_event(stage: str, seconds: float, status: str) -> None:
        if status == "running":
            current["stage"] = stage
        record(stage, seconds, status)

    def on_usage(usage: object) -> None:
        guard.post(usage)
        try:
            ledger.record(current["stage"] or "unknown", usage)
        except Exception:  # noqa: BLE001 — per-stage metrics must never break a run
            log.debug("per-stage llm usage not recorded (%s)", obj_id, exc_info=True)

    return on_event, on_usage


def _timer(cfg: Config, movie_path: Path, lang: str | None = None, *, on_event=None) -> RunTimer:
    bdir = pipeline.back_half_dir(cfg, str(movie_path), lang) if lang else cfg.movie_dir(str(movie_path))
    bdir.mkdir(parents=True, exist_ok=True)
    return RunTimer(bdir / "timings.json", on_event=on_event)


def _load_timings(path: Path) -> list[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("stages", [])
    except (OSError, json.JSONDecodeError):
        return []


def clear_back_half_artifacts(session_id: str, movie_id: str, slug: str, lang: str) -> int:
    """Wipe one (movie, lang) back-half output set from S3 + local so a regenerate recomputes
    it fresh. The shared front-half artifacts live in the parent dir and are left intact, so
    ASR/scene detection are NOT redone — only understand→render for this language. Returns the
    number of S3 objects deleted. Must run BEFORE the chain is enqueued (the tasks re-materialize
    from S3, so a stale cached copy there would otherwise be pulled back down)."""
    prefix = f"{_artifact_prefix(session_id, movie_id, slug)}/{lang}/"
    deleted = 0
    try:
        deleted = storage().delete_prefix(prefix)
    except Exception:  # noqa: BLE001 — best-effort; a leftover object just gets overwritten
        log.warning("regenerate: could not clear S3 prefix %s", prefix, exc_info=True)
    shutil.rmtree(_movie_root(session_id, movie_id) / slug / lang, ignore_errors=True)
    log.info("regenerate: cleared back-half cache %s (%d S3 objects)", prefix, deleted)
    return deleted


def clear_movie_artifacts(session_id: str, movie_id: str, slug: str) -> int:
    """Wipe ALL cached artifacts for a movie — the shared front half (probe/transcript/shots/
    scenes/clip-index) AND every language's back half — from S3 + the local working dir, so a
    full "start over" recomputes from the first stage (ingest). The SOURCE upload lives under a
    separate ``sources/…`` prefix and is left untouched, so the user never re-uploads. Returns the
    number of S3 objects deleted. Must run BEFORE the chain is re-enqueued (tasks re-materialize
    from S3, so a stale cached copy there would otherwise be pulled back down)."""
    prefix = f"{_artifact_prefix(session_id, movie_id, slug)}/"
    deleted = 0
    try:
        deleted = storage().delete_prefix(prefix)
    except Exception:  # noqa: BLE001 — best-effort; a leftover object just gets overwritten
        log.warning("start-over: could not clear S3 prefix %s", prefix, exc_info=True)
    # drop the whole local working dir too; the source re-downloads from S3 on the next _prepare
    _prune_movie_scratch(session_id, movie_id)
    log.info("start-over: cleared all artifacts %s (%d S3 objects)", prefix, deleted)
    return deleted


def _prune_movie_scratch(session_id: str, movie_id: str) -> None:
    """Best-effort: drop the local working dir (an ``emptyDir`` on the node's root disk) once
    artifacts are safely in S3. S3 is the source of truth, so a later phase re-materializes on
    a miss. Keeps the small prod box from filling up over time. Never raises."""
    try:
        root = _movie_root(session_id, movie_id)
        shutil.rmtree(root, ignore_errors=True)
        log.info("pruned local scratch %s", root)
    except Exception:  # noqa: BLE001 — cleanup must never fail a task
        log.debug("scratch prune failed for %s/%s", session_id, movie_id, exc_info=True)


def _fail(model_cls, obj_id: str, exc: Exception) -> None:
    with session_scope() as db:
        obj = db.get(model_cls, obj_id)
        if obj is not None:
            obj.status = (MovieStatus.error if model_cls is Movie else RunStatus.error)
            obj.error = f"{type(exc).__name__}: {exc}"[:2000]


# ===========================================================================
# FRONT HALF — fh_local (cpu) -> fh_asr (gpu) -> fh_scenes (cpu) -> mark_ready
# ===========================================================================
def start_front_half(movie_id: str) -> str:
    """Build + dispatch the front-half chain. Returns the chain's async id."""
    init_db()
    res = chain(
        fh_local.si(movie_id),
        fh_asr.si(movie_id),
        fh_scenes.si(movie_id),
        mark_ready.si(movie_id),
    ).apply_async()
    return res.id


@app.task(bind=True, max_retries=2, default_retry_delay=30)
def fh_local(self, movie_id: str) -> str:
    """ingest + audio extraction (local, cheap)."""
    sid, slug, source_key = _movie_fields(movie_id)
    logging_config.bind(movie_id=movie_id, movie_slug=slug, session_id=sid, stage="fh_local")
    try:
        with session_scope() as db:
            db.get(Movie, movie_id).status = MovieStatus.processing
        movie_path, cfg, store, prefix = _prepare(sid, movie_id, slug, source_key)
        rec = _stage_recorder("movie", movie_id)
        pipeline.run_front_half(movie_path, cfg, until="audio", timer=_timer(cfg, movie_path, on_event=rec))
        _persist(store, cfg, movie_path, prefix)
    except Exception as exc:  # noqa: BLE001
        _fail(Movie, movie_id, exc)
        raise
    return movie_id


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def fh_asr(self, movie_id: str) -> str:
    """ASR via the GPU box — serialized on the gpu queue + Redis lock."""
    sid, slug, source_key = _movie_fields(movie_id)
    logging_config.bind(movie_id=movie_id, movie_slug=slug, session_id=sid, stage="fh_asr")
    try:
        movie_path, cfg, store, prefix = _prepare(sid, movie_id, slug, source_key)
        rec = _stage_recorder("movie", movie_id)
        with _gpu_slot("asr") as asr_target:
            pipeline.run_front_half(movie_path, cfg, until="asr",
                                    timer=_timer(cfg, movie_path, on_event=rec), asr_target=asr_target)
        _persist(store, cfg, movie_path, prefix)
    except Exception as exc:  # noqa: BLE001
        _fail(Movie, movie_id, exc)
        raise
    return movie_id


@app.task(bind=True, max_retries=2, default_retry_delay=30)
def fh_scenes(self, movie_id: str) -> str:
    """shots + scenes + keyframes -> clip_index (local, heavier CPU)."""
    sid, slug, source_key = _movie_fields(movie_id)
    logging_config.bind(movie_id=movie_id, movie_slug=slug, session_id=sid, stage="fh_scenes")
    try:
        movie_path, cfg, store, prefix = _prepare(sid, movie_id, slug, source_key)
        rec = _stage_recorder("movie", movie_id)
        pipeline.run_front_half(movie_path, cfg, timer=_timer(cfg, movie_path, on_event=rec))  # rest, cached upstream
        _persist(store, cfg, movie_path, prefix)
    except Exception as exc:  # noqa: BLE001
        _fail(Movie, movie_id, exc)
        raise
    return movie_id


def _ensure_run(movie_id: str, session_id: str, lang: str) -> str | None:
    """Get-or-create a queued Run for (movie, lang); return the run_id to START, or None if a
    non-error run already exists (so we never double-start). Honors the (movie_id, lang) unique
    constraint by selecting first."""
    lang = lang.lower()
    with session_scope() as db:
        existing = db.query(Run).filter(Run.movie_id == movie_id, Run.lang == lang).one_or_none()
        if existing is not None:
            if existing.status != RunStatus.error:
                return None                      # already queued/running/done
            existing.status = RunStatus.queued   # retry a previously-failed run
            existing.error = None
            return existing.id
        run = Run(movie_id=movie_id, session_id=session_id, lang=lang, status=RunStatus.queued)
        db.add(run)
        db.flush()
        return run.id


@app.task
def mark_ready(movie_id: str) -> str:
    sid = None
    auto_lang = None
    with session_scope() as db:
        movie = db.get(Movie, movie_id)
        if movie is not None and movie.status != MovieStatus.error:
            sid = movie.session_id
            auto_lang = movie.default_lang
            movie.status = MovieStatus.ready
            # best-effort: record probed duration if the front half wrote it
            try:
                from yapper.schemas import ProbeManifest

                pm_path = _movie_root(movie.session_id, movie_id) / movie.slug / "probe.json"
                if pm_path.exists():
                    movie.duration_sec = ProbeManifest.load(pm_path).duration_sec
            except Exception:  # noqa: BLE001
                pass
    # One-step UX: a language was chosen at upload, so kick the commentary off automatically as soon
    # as the front half is ready — no separate "generate" click. (Idempotent per movie+lang.)
    run_id = None
    if sid is not None and auto_lang:
        try:
            run_id = _ensure_run(movie_id, sid, auto_lang)
        except Exception:  # noqa: BLE001 — never fail readiness on the auto-run; it can be started manually
            log.warning("auto-run not started for %s (%s)", movie_id, auto_lang, exc_info=True)
    if run_id is not None:
        start_back_half(run_id)              # back half re-materializes the working dir from S3
    elif sid is not None:
        # No auto-run: front-half artifacts are safe in S3, so free the local scratch now (the
        # back half would re-materialize from S3 on demand if started manually later).
        _prune_movie_scratch(sid, movie_id)
    return movie_id


# ===========================================================================
# BACK HALF — bh_script (llm) -> bh_tts (gpu) -> bh_render (render) -> finalize
# ===========================================================================
def start_back_half(run_id: str) -> str:
    init_db()
    res = chain(
        bh_script.si(run_id),
        bh_tts.si(run_id),
        bh_render.si(run_id),
        finalize_run.si(run_id),
    ).apply_async()
    return res.id


def _run_ctx(run_id: str):
    """Fetch (movie_id, lang, session_id, slug, source_key) for a run."""
    with session_scope() as db:
        run = db.get(Run, run_id)
        movie = db.get(Movie, run.movie_id)
        return run.movie_id, run.lang, movie.session_id, movie.slug, movie.source_key


@app.task(bind=True, max_retries=1, default_retry_delay=30)
def bh_script(self, run_id: str) -> str:
    """MAP + REDUCE + runtime budget (LLM-bound). Records actual token/cost usage per run
    (for display) — there is no money cap, so a run is never blocked or aborted on spend."""
    movie_id, lang, sid, slug, source_key = _run_ctx(run_id)
    logging_config.bind(run_id=run_id, movie_slug=slug, session_id=sid, lang=lang, stage="bh_script")
    with session_scope() as db:
        db.get(Run, run_id).status = RunStatus.running
    guard = BudgetGuard(Budget())  # token/cost accounting only (no cap enforcement)
    try:
        movie_path, cfg, store, prefix = _prepare(sid, movie_id, slug, source_key)
        on_event, on_usage = _llm_stage_tracking("run", run_id, guard)
        client = _build_guarded_client(cfg, guard, on_usage=on_usage)
        pipeline.run_back_half(
            movie_path, cfg, lang=lang, llm_client=client, until="budget",
            timer=_timer(cfg, movie_path, lang, on_event=on_event),
        )
        _persist(store, cfg, movie_path, prefix)
        with session_scope() as db:
            run = db.get(Run, run_id)
            run.llm_tokens_in = guard.tokens_in
            run.llm_tokens_out = guard.tokens_out
            run.llm_cost_usd = guard.cost_usd
    except Exception as exc:  # noqa: BLE001
        _fail(Run, run_id, exc)
        raise
    return run_id


def _build_guarded_client(cfg: Config, guard: BudgetGuard, *, on_usage=None):
    """Construct the LLMClient the worker injects, with budget hooks attached.

    ``on_usage`` defaults to ``guard.post`` (per-run token/cost accounting); the back half
    passes a composite that also records the per-stage tally (see :func:`_llm_stage_tracking`).
    """
    from yapper.llm.client import LLMClient

    lcfg = cfg.section("llm")
    return LLMClient(
        api_key=cfg.require_env("LLM_API_KEY"),
        base_url=cfg.env("LLM_BASE_URL") or lcfg.get("base_url", "https://api.minimax.io/v1"),
        model=cfg.env("LLM_MODEL") or lcfg.get("model", "MiniMax-M3"),
        temperature=float(lcfg.get("temperature", 0.7)),
        max_output_tokens=int(lcfg.get("max_output_tokens", 32000)),
        vision=bool(lcfg.get("vision", True)),
        max_images=int(lcfg.get("max_images", 180)),
        timeout=float(lcfg.get("request_timeout_sec", 1800)),
        default_pre_request_hook=None,   # no spend cap: never abort a run mid-flight
        default_on_usage=on_usage or guard.post,  # record token/cost usage per run (+ per stage)
    )


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def bh_tts(self, run_id: str) -> str:
    """Voiceover synthesis on the GPU box. Leases up to TTS_INSTANCES on-demand instances
    and synthesizes the independent narration lines across them in parallel (degrades to
    however many the shared pool can give, >=1)."""
    movie_id, lang, sid, slug, source_key = _run_ctx(run_id)
    logging_config.bind(run_id=run_id, movie_slug=slug, session_id=sid, lang=lang, stage="bh_tts")
    try:
        movie_path, cfg, store, prefix = _prepare(sid, movie_id, slug, source_key)
        with _gpu_slots("tts", S.tts_instances) as tts_targets:
            pipeline.run_back_half(
                movie_path, cfg, lang=lang, until="tts",
                timer=_timer(cfg, movie_path, lang, on_event=_stage_recorder("run", run_id)),
                tts_targets=tts_targets,
            )
        _persist(store, cfg, movie_path, prefix)
    except Exception as exc:  # noqa: BLE001
        _fail(Run, run_id, exc)
        raise
    return run_id


@app.task(bind=True, max_retries=1, default_retry_delay=30)
def bh_render(self, run_id: str) -> str:
    """EDL + subtitles + ffmpeg render (CPU). Overlaps other movies' TTS/LLM."""
    movie_id, lang, sid, slug, source_key = _run_ctx(run_id)
    logging_config.bind(run_id=run_id, movie_slug=slug, session_id=sid, lang=lang, stage="bh_render")
    try:
        movie_path, cfg, store, prefix = _prepare(sid, movie_id, slug, source_key)
        out = pipeline.run_back_half(
            movie_path, cfg, lang=lang,
            timer=_timer(cfg, movie_path, lang, on_event=_stage_recorder("run", run_id)),
        )
        _persist(store, cfg, movie_path, prefix)
        # upload the final commentary video to a stable key for presigned download
        out_key = f"{prefix}/{lang}/recap_final.mp4"
        store.upload(out, out_key)
        with session_scope() as db:
            db.get(Run, run_id).output_key = out_key
    except Exception as exc:  # noqa: BLE001
        _fail(Run, run_id, exc)
        raise
    return run_id


@app.task
def finalize_run(run_id: str) -> str:
    """Mark done, mirror timings.json -> run_stages, push run metrics."""
    movie_id, lang, sid, slug, _source_key = _run_ctx(run_id)
    bdir = _movie_root(sid, movie_id) / slug / lang
    stages = _load_timings(bdir / "timings.json")
    nar = pb = None
    out_dur = None
    try:
        from yapper.schemas import Edl, Script

        sf = bdir / "script_final.json"
        if sf.exists():
            lines = Script.load(sf).lines
            nar = sum(1 for ln in lines if getattr(ln, "kind", "narration") == "narration")
            pb = sum(1 for ln in lines if getattr(ln, "kind", "narration") == "playback")
        edl_p = bdir / "edl.json"
        if edl_p.exists():
            edl = Edl.load(edl_p)
            out_dur = sum(getattr(seg, "vo_duration", 0.0) or (seg.src_out - seg.src_in) for seg in edl.segments)
    except Exception:  # noqa: BLE001
        pass
    with session_scope() as db:
        run = db.get(Run, run_id)
        if run.status != RunStatus.error:
            run.status = RunStatus.done
        from datetime import datetime, timezone
        run.finished_at = datetime.now(timezone.utc)
        run.output_duration_sec = out_dur
        # Reconcile (don't replace) the live rows with the final timings.json, then read
        # them back so metrics use the accurate per-stage durations rather than the
        # last task's cached:0 view.
        record_timings(db, run_id, stages)
        db.flush()
        final_stages = [
            {"stage": s.stage, "status": s.status, "seconds": s.seconds}
            for s in db.query(RunStage).filter(RunStage.run_id == run_id).order_by(RunStage.id).all()
        ]
        tokens_in, tokens_out, cost = run.llm_tokens_in, run.llm_tokens_out, run.llm_cost_usd
    push_run_metrics(
        run_id, lang, stages=final_stages, tokens_in=tokens_in, tokens_out=tokens_out,
        cost_usd=cost, output_duration_sec=out_dur, narration_lines=nar, playback_lines=pb,
    )
    # Free the movie's local scratch once no other run for it is still active (another
    # language run shares the front-half artifacts + source video, so only prune when idle).
    with session_scope() as db:
        active = db.query(Run).filter(
            Run.movie_id == movie_id, Run.status.in_([RunStatus.queued, RunStatus.running])
        ).count()
    if active == 0:
        _prune_movie_scratch(sid, movie_id)
    return run_id


# convenience for `celery -A yapper_web.tasks worker`
celery_app = app
