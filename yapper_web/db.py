"""Postgres data model: sessions, movies, runs, run_stages.

A *session* (no sign-in, a signed 1-year cookie) owns *movies*; each movie has one
or more *runs* (one per narration language) that reuse the shared front-half
artifacts. *run_stages* mirrors the pipeline's ``timings.json`` for the run
drill-down dashboard. All ownership flows from ``session_id`` — every query in the
API filters by it.
"""

from __future__ import annotations

import enum
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from .settings import get_settings


def _uuid() -> str:
    """Time-sortable, human-readable id: Pacific-time stamp + short random suffix
    (e.g. ``20260617-143522-PDT-a1b2c3d4``). Lexicographically sorts by creation time and
    tells you at a glance which upload an S3 prefix / working dir belongs to — far easier to
    track than a bare UUID. Existing bare-UUID ids stay valid; this only affects ids minted
    from now on. Always <=36 chars (fits the String(36) primary-key columns)."""
    suffix = uuid.uuid4().hex[:8]
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("America/Los_Angeles"))
        return f"{now:%Y%m%d-%H%M%S-%Z}-{suffix}"          # ...-PDT-/-PST- per season
    except Exception:  # noqa: BLE001 — tz database missing: degrade to a UTC-stamped id
        return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S-UTC}-{suffix}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class MovieStatus(str, enum.Enum):
    registered = "registered"   # upload URL issued, file not yet confirmed
    uploaded = "uploaded"       # in S3, front half not started
    processing = "processing"   # front half running
    ready = "ready"             # front half done, can spawn language runs
    error = "error"


class RunStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


class Session_(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    cost_spent_usd: Mapped[float] = mapped_column(Float, default=0.0)

    movies: Mapped[list["Movie"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Movie(Base):
    __tablename__ = "movies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    original_filename: Mapped[str] = mapped_column(String(512))
    slug: Mapped[str] = mapped_column(String(256))
    s3_prefix: Mapped[str] = mapped_column(String(512))      # {session_id}/{slug}
    source_key: Mapped[str] = mapped_column(String(512))     # full S3 key of the upload
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[MovieStatus] = mapped_column(Enum(MovieStatus), default=MovieStatus.registered)
    # Output language chosen at UPLOAD time; the front half auto-starts a back-half run in this
    # language when it finishes (one-step upload). Nullable so pre-existing movies fall back to
    # the manual "pick language + generate" flow.
    default_lang: Mapped[str | None] = mapped_column(String(8), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    session: Mapped[Session_] = relationship(back_populates="movies")
    runs: Mapped[list["Run"]] = relationship(back_populates="movie", cascade="all, delete-orphan")
    stages: Mapped[list["MovieStage"]] = relationship(
        back_populates="movie", cascade="all, delete-orphan", order_by="MovieStage.id"
    )


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (UniqueConstraint("movie_id", "lang", name="uq_run_movie_lang"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    movie_id: Mapped[str] = mapped_column(ForeignKey("movies.id"), index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    lang: Mapped[str] = mapped_column(String(8))
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus), default=RunStatus.queued)
    celery_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    target_sec: Mapped[int | None] = mapped_column(Integer, nullable=True)

    llm_tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    llm_tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    llm_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    output_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    output_duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    movie: Mapped[Movie] = relationship(back_populates="runs")
    stages: Mapped[list["RunStage"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="RunStage.id"
    )


class RunStage(Base):
    __tablename__ = "run_stages"
    # One row per (run, stage): record_stage_event upserts via one_or_none(), so a duplicate
    # would both break that read and double-count in the metrics join. (create_all adds this
    # on fresh DBs; an existing prod table needs a manual migration — the collector's render
    # query also de-dupes per run, so the metric is correct regardless.)
    __table_args__ = (UniqueConstraint("run_id", "stage", name="uq_run_stage"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    stage: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))   # running | ran | cached | error
    seconds: Mapped[float] = mapped_column(Float, default=0.0)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped[Run] = relationship(back_populates="stages")


class MovieStage(Base):
    """Per-stage timing for a movie's front half (ingest -> scenes), the symmetric twin
    of ``RunStage`` for the per-language back half. Written live by the worker so the UI
    can show the current front-half step and how long each took."""

    __tablename__ = "movie_stages"
    __table_args__ = (UniqueConstraint("movie_id", "stage", name="uq_movie_stage"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    movie_id: Mapped[str] = mapped_column(ForeignKey("movies.id"), index=True)
    stage: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))   # running | ran | cached | error
    seconds: Mapped[float] = mapped_column(Float, default=0.0)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    movie: Mapped[Movie] = relationship(back_populates="stages")


# --- engine / session factory ------------------------------------------------
_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def _ensure_engine() -> sessionmaker[Session]:
    global _engine, _SessionLocal
    if _SessionLocal is None:
        _engine = create_engine(get_settings().database_url, pool_pre_ping=True, future=True)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _SessionLocal


def init_db() -> None:
    """Create tables if absent (idempotent). For real migrations, swap in Alembic."""
    _ensure_engine()
    Base.metadata.create_all(_engine)
    _add_missing_columns(_engine)


# Tiny hand-rolled migrations: create_all() never ALTERs an existing table, so new NULLABLE
# columns added to a model won't appear on a DB created by an earlier version. Add them here,
# idempotently (skipped on a fresh DB where create_all already made the column). (table, column,
# DDL type) — keep additive + nullable so this stays safe without Alembic.
_ADDED_COLUMNS = [("movies", "default_lang", "VARCHAR(8)")]


def _add_missing_columns(engine) -> None:
    from sqlalchemy import inspect as _inspect
    from sqlalchemy import text as _text

    insp = _inspect(engine)
    for table, column, ddl in _ADDED_COLUMNS:
        try:
            present = {c["name"] for c in insp.get_columns(table)}
        except Exception:  # noqa: BLE001 — table absent (shouldn't happen post create_all)
            continue
        if column not in present:
            with engine.begin() as conn:
                conn.execute(_text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional DB session: commit on success, rollback on error."""
    factory = _ensure_engine()
    db = factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# --- small helpers used across api/tasks -------------------------------------
def touch_session(db: Session, session_id: str) -> Session_:
    """Get-or-create a session row and bump last_seen."""
    s = db.get(Session_, session_id)
    if s is None:
        s = Session_(id=session_id)
        db.add(s)
    else:
        s.last_seen = _now()
    return s


_TERMINAL_STATUS = {"ran", "error"}


def record_stage_event(
    db: Session, scope: str, obj_id: str, stage: str, seconds: float, status: str
) -> None:
    """Upsert one stage row, never downgrading a stage that already finished.

    ``scope`` is ``"run"`` (``RunStage`` keyed by ``run_id``) or ``"movie"``
    (``MovieStage`` keyed by ``movie_id``). A ``"running"`` marker is written when a
    stage starts and replaced by ``"ran"``/``"error"`` when it ends. Crucially, a
    ``"cached"`` (skipped) event from a *later* task that re-enters the pipeline must
    NOT overwrite the real duration measured by whichever task actually ran the stage —
    so the headline-expensive steps (understand/script/tts) keep their true timing
    across the resumed multi-task chain. ``max`` on the duration makes re-runs of an
    always-recomputed stage (budget/edl/subs) keep their largest observed time."""
    if scope == "run":
        model: type[RunStage] | type[MovieStage] = RunStage
        fk = "run_id"
        existing = db.query(RunStage).filter(
            RunStage.run_id == obj_id, RunStage.stage == stage
        ).one_or_none()
    else:
        model = MovieStage
        fk = "movie_id"
        existing = db.query(MovieStage).filter(
            MovieStage.movie_id == obj_id, MovieStage.stage == stage
        ).one_or_none()

    seconds = float(seconds or 0.0)
    if existing is None:
        db.add(model(**{fk: obj_id, "stage": stage, "status": status, "seconds": seconds}))
        return
    if status in _TERMINAL_STATUS:
        existing.status = status
        existing.seconds = max(float(existing.seconds or 0.0), seconds)
        existing.recorded_at = _now()
    elif existing.status not in _TERMINAL_STATUS:
        # advance a not-yet-finished row (queued -> running, or running -> cached);
        # 'cached' keeps any seconds already noted, 'running' carries none.
        existing.status = status
        if status != "cached":
            existing.seconds = seconds
        existing.recorded_at = _now()
    # else: 'running'/'cached' arriving on an already-terminal row -> ignore (no downgrade).


def record_timings(db: Session, run_id: str, stages: list[dict]) -> None:
    """Reconcile a run's stage rows from a parsed timings.json ``stages`` list.

    Upserts (never deletes) via :func:`record_stage_event`, so the accurate live rows
    written during the run are preserved — the final timings.json from the last task in
    the chain marks earlier stages as ``cached:0`` and must not clobber them."""
    for s in stages:
        record_stage_event(
            db, "run", run_id,
            s.get("stage", "?"), float(s.get("seconds", 0.0)), s.get("status", "ran"),
        )


def stage_avg_seconds(db: Session, scope: str = "run", *, cutoff_days: int = 30) -> dict[str, float]:
    """Mean wall-time per stage over recently FINISHED stages (status 'ran'/'cached'), keyed by
    stage name — the basis for the pipeline ETA. ``scope`` is 'run' (RunStage) or 'movie'
    (MovieStage). Returns {} when there's no history yet (caller falls back to static seeds)."""
    from datetime import timedelta

    model = RunStage if scope == "run" else MovieStage
    cutoff = _now() - timedelta(days=cutoff_days)
    rows = (
        db.query(model.stage, func.avg(model.seconds))
        .filter(model.recorded_at >= cutoff, model.status.in_(("ran", "cached")), model.seconds > 0)
        .group_by(model.stage)
        .all()
    )
    return {stage: float(avg or 0.0) for stage, avg in rows}


__all__ = [
    "Base", "Session_", "Movie", "Run", "RunStage", "MovieStage", "MovieStatus", "RunStatus",
    "init_db", "session_scope", "touch_session", "record_timings", "record_stage_event",
    "stage_avg_seconds", "func",
]
