"""yapper_web — the platform shell around the yapper CLI pipeline.

A thin orchestration + transport + observability layer:

- ``settings``  env-driven configuration for the web layer
- ``db``        Postgres data model (sessions, movies, runs, run_stages)
- ``budget``    the $-cap guard (Redis counter + pre-flight gate + accounting)
- ``metrics``   Prometheus metric definitions + timings.json emitter
- ``tasks``     Celery app + resource-queued tasks wrapping the pipeline
- ``api``       FastAPI app (no-signin sessions, uploads, runs, SSE, /metrics)

The core ``yapper`` package never imports anything from here, so the CLI
keeps working with zero web dependencies installed.
"""

__all__ = ["settings"]
