"""Prometheus metrics.

Two surfaces:

1. **API `/metrics`** — the authoritative platform scrape target. A custom collector
   computes gauges *at scrape time* from the sources of truth (Postgres for run/movie
   state, Redis for the budget ledger, the Celery broker for queue depth). This avoids
   the multiprocess-registry headache of trying to aggregate counters across short-lived
   worker processes.

2. **Worker push** — per-run histograms/counters (stage durations, LLM tokens & cost,
   render timing, output-quality proxies) pushed to a Pushgateway *if* ``PUSHGATEWAY_URL``
   is set, grouped by ``run_id``. No gateway configured -> the push helpers are no-ops,
   and the DB-derived gauges from surface (1) still cover the high-value signals.

Quality proxies (output duration, line counts, loudness, dupe count) turn the user's
manual watch-test into tracked series — see the Grafana "run drill-down" dashboard.
"""

from __future__ import annotations

import logging
import os

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

from .settings import get_settings

log = logging.getLogger("yapper_web.metrics")

CELERY_QUEUES = ("cpu", "asr", "tts", "llm", "render")

# ---------------------------------------------------------------------------
# (1) API-side custom collector: snapshot gauges from DB + Redis + broker.
# ---------------------------------------------------------------------------
class PlatformCollector(Collector):
    """Yielded on every scrape of the API's /metrics. Defensive: a failing source
    degrades to fewer series rather than a 500."""

    def collect(self):
        # --- runs / movies (Postgres) -------------------------------------
        try:
            from sqlalchemy import func

            from .db import Movie, Run, RunStatus, session_scope

            runs = GaugeMetricFamily(
                "yapper_runs", "Run count by status (snapshot)", labels=["status"]
            )
            inprog = 0
            with session_scope() as db:
                rows = dict(
                    db.query(Run.status, func.count(Run.id)).group_by(Run.status).all()
                )
                for status in RunStatus:
                    n = int(rows.get(status, 0))
                    runs.add_metric([status.value], n)
                    if status in (RunStatus.queued, RunStatus.running):
                        inprog += n
                movies_total = db.query(func.count(Movie.id)).scalar() or 0
                sessions_total = db.query(func.count(func.distinct(Run.session_id))).scalar() or 0
            yield runs
            g = GaugeMetricFamily("yapper_runs_in_progress", "Queued + running runs")
            g.add_metric([], inprog)
            yield g
            mt = GaugeMetricFamily("yapper_movies_total", "Registered movies")
            mt.add_metric([], movies_total)
            yield mt
            st = GaugeMetricFamily("yapper_active_sessions_total", "Sessions with runs")
            st.add_metric([], sessions_total)
            yield st
        except Exception as e:  # noqa: BLE001
            log.warning("metrics: DB collect failed: %s", e)

        # --- budget ledger (Redis) ----------------------------------------
        try:
            from .budget import Budget

            b = Budget()
            for name, val, doc in (
                ("yapper_llm_budget_remaining_usd", b.remaining_usd(), "USD of LLM credit left"),
                ("yapper_llm_cost_usd_total", b.spent_usd(), "USD of LLM credit spent"),
                ("yapper_llm_budget_reserved_usd", b.reserved_usd(), "USD held by in-flight runs"),
                ("yapper_llm_budget_cap_usd", get_settings().llm_max_cost_usd, "Configured spend cap"),
            ):
                g = GaugeMetricFamily(name, doc)
                g.add_metric([], float(val))
                yield g
        except Exception as e:  # noqa: BLE001
            log.warning("metrics: budget collect failed: %s", e)

        # --- queue depth (Celery broker = Redis lists) --------------------
        try:
            import redis

            r = redis.Redis.from_url(get_settings().redis_url)
            qd = GaugeMetricFamily("yapper_queue_depth", "Pending tasks per queue", labels=["queue"])
            for q in CELERY_QUEUES:
                qd.add_metric([q], int(r.llen(q) or 0))
            yield qd
        except Exception as e:  # noqa: BLE001
            log.warning("metrics: queue collect failed: %s", e)

        # --- per-stage timing + render efficiency (Postgres) --------------
        # The resource-usage signals for the long stages (asr/understand/script/tts/
        # render). run_stages/movie_stages are written live by the worker's stage
        # recorder; we expose avg/max over a recent window (also bounds scrape cost on
        # the unindexed stage column) plus the count currently running.
        try:
            from datetime import datetime, timedelta, timezone

            from sqlalchemy import func

            from .db import MovieStage, Run, RunStage, RunStatus, session_scope

            cutoff = datetime.now(timezone.utc) - timedelta(days=30)
            running_cutoff = datetime.now(timezone.utc) - timedelta(hours=6)  # ignore orphaned rows
            savg = GaugeMetricFamily(
                "yapper_stage_seconds_avg", "Mean per-stage wall time (recent)",
                labels=["scope", "stage", "status"],
            )
            smax = GaugeMetricFamily(
                "yapper_stage_seconds_max", "Max per-stage wall time (recent)",
                labels=["scope", "stage", "status"],
            )
            srun = GaugeMetricFamily(
                "yapper_stage_running", "Stages currently in the running state",
                labels=["scope", "stage"],
            )
            rtf = GaugeMetricFamily(
                "yapper_render_rtf",
                "Render realtime-factor = output_sec / render_sec, mean over recent done runs",
                labels=["lang"],
            )
            with session_scope() as db:
                for scope, model in (("run", RunStage), ("movie", MovieStage)):
                    for stage, status, avg_s, max_s in db.query(
                        model.stage, model.status, func.avg(model.seconds), func.max(model.seconds)
                    ).filter(model.recorded_at >= cutoff).group_by(model.stage, model.status).all():
                        savg.add_metric([scope, stage, status], float(avg_s or 0.0))
                        smax.add_metric([scope, stage, status], float(max_s or 0.0))
                    for stage, n in db.query(model.stage, func.count()).filter(
                        model.status == "running", model.recorded_at >= running_cutoff
                    ).group_by(model.stage).all():
                        srun.add_metric([scope, stage], int(n))
                # Encode efficiency over recent *done* runs. Collapse render seconds to one
                # value per run first (robust if a race produced duplicate render rows), then
                # average output/render by language.
                render_sec = (
                    db.query(RunStage.run_id.label("run_id"), func.max(RunStage.seconds).label("sec"))
                    .filter(
                        RunStage.stage == "render", RunStage.status == "ran",
                        RunStage.seconds > 0, RunStage.recorded_at >= cutoff,
                    )
                    .group_by(RunStage.run_id)
                    .subquery()
                )
                for lang, val in (
                    db.query(Run.lang, func.avg(Run.output_duration_sec / render_sec.c.sec))
                    .join(render_sec, render_sec.c.run_id == Run.id)
                    .filter(Run.status == RunStatus.done, Run.output_duration_sec.isnot(None))
                    .group_by(Run.lang)
                    .all()
                ):
                    if val is not None:
                        rtf.add_metric([lang], float(val))
            yield savg
            yield smax
            yield srun
            yield rtf
        except Exception as e:  # noqa: BLE001
            log.warning("metrics: stage timing collect failed: %s", e)

        # --- per-stage LLM spend/tokens (Redis tallies) -------------------
        try:
            from .budget import StageUsageLedger

            cost = GaugeMetricFamily(
                "yapper_llm_stage_cost_usd_total", "Cumulative LLM USD by pipeline stage",
                labels=["stage"],
            )
            toks = GaugeMetricFamily(
                "yapper_llm_stage_tokens_total", "Cumulative LLM tokens by pipeline stage",
                labels=["stage", "direction"],
            )
            for stage, v in StageUsageLedger().snapshot().items():
                cost.add_metric([stage], float(v["cost_usd"]))
                toks.add_metric([stage, "in"], float(v["tokens_in"]))
                toks.add_metric([stage, "out"], float(v["tokens_out"]))
            yield cost
            yield toks
        except Exception as e:  # noqa: BLE001
            log.warning("metrics: stage llm collect failed: %s", e)


_api_registered = False


def register_api_collector() -> None:
    """Register the platform collector on the default registry (call once on API start)."""
    global _api_registered
    if not _api_registered:
        from prometheus_client import REGISTRY

        REGISTRY.register(PlatformCollector())
        _api_registered = True


def render_latest() -> tuple[bytes, str]:
    """(body, content_type) for the /metrics response."""
    return generate_latest(), CONTENT_TYPE_LATEST


# ---------------------------------------------------------------------------
# (2) Worker-side per-run metrics, pushed to a Pushgateway when configured.
# ---------------------------------------------------------------------------
def _gateway() -> str | None:
    return os.environ.get("PUSHGATEWAY_URL") or None


def push_run_metrics(
    run_id: str,
    lang: str,
    *,
    stages: list[dict] | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    output_duration_sec: float | None = None,
    narration_lines: int | None = None,
    playback_lines: int | None = None,
    json_retries: int = 0,
    failed_stage: str | None = None,
) -> None:
    """Push one run's metrics to the Pushgateway grouped by run_id. No-op if no
    PUSHGATEWAY_URL. Everything here is also persisted to Postgres, so this is the
    fast-path for Grafana, not the source of truth."""
    gw = _gateway()
    if not gw:
        return
    try:
        from prometheus_client import push_to_gateway

        reg = CollectorRegistry()
        dur = Histogram(
            "yapper_stage_duration_seconds", "Per-stage wall time",
            ["stage", "lang", "status"],
            buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1200, 3600), registry=reg,
        )
        for s in stages or []:
            dur.labels(s.get("stage", "?"), lang, s.get("status", "ran")).observe(float(s.get("seconds", 0.0)))

        Counter("yapper_llm_tokens", "LLM tokens this run", ["lang", "direction"], registry=reg)\
            .labels(lang, "in").inc(tokens_in)
        Counter("yapper_llm_tokens_out_run", "", ["lang"], registry=reg).labels(lang).inc(tokens_out)
        Gauge("yapper_run_llm_cost_usd", "LLM USD this run", ["lang"], registry=reg).labels(lang).set(cost_usd)
        if json_retries:
            Counter("yapper_llm_json_retries", "Non-JSON LLM retries", ["lang"], registry=reg)\
                .labels(lang).inc(json_retries)

        # output-quality proxies
        if output_duration_sec is not None:
            Gauge("yapper_output_duration_seconds", "Final commentary length", ["lang"], registry=reg)\
                .labels(lang).set(output_duration_sec)
        if narration_lines is not None:
            Gauge("yapper_script_lines", "Script lines by kind", ["lang", "kind"], registry=reg)\
                .labels(lang, "narration").set(narration_lines)
        if playback_lines is not None:
            Gauge("yapper_script_lines", "Script lines by kind", ["lang", "kind"], registry=reg)\
                .labels(lang, "playback").set(playback_lines)
        if failed_stage:
            Counter("yapper_stage_failures", "Stage failures", ["stage", "lang"], registry=reg)\
                .labels(failed_stage, lang).inc()

        push_to_gateway(gw, job="yapper_run", grouping_key={"run_id": run_id}, registry=reg)
    except Exception as e:  # noqa: BLE001 — telemetry must never fail a run
        log.warning("push_run_metrics(%s) failed: %s", run_id, e)
