"""Env-driven configuration for the web platform.

Everything here is read from the environment (injected by docker-compose / the EC2
instance) so secrets stay out of the repo. Defaults are dev-friendly so the stack
runs locally with `docker compose up`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _csv(val: str) -> list[str]:
    return [x.strip() for x in val.split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    # --- infra ---------------------------------------------------------------
    database_url: str = os.environ.get(
        "DATABASE_URL", "postgresql+psycopg://jieshuo:jieshuo@postgres:5432/jieshuo"
    )
    redis_url: str = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    celery_broker_url: str = os.environ.get("CELERY_BROKER_URL", "")
    celery_result_backend: str = os.environ.get("CELERY_RESULT_BACKEND", "")

    # --- storage (S3 / MinIO) ------------------------------------------------
    s3_bucket: str = os.environ.get("S3_BUCKET", "jieshuo-artifacts")
    s3_region: str = os.environ.get("AWS_REGION", "us-east-1")
    # Internal endpoint used by api/workers (e.g. http://minio:9000 locally; unset -> AWS).
    s3_endpoint_url: str | None = os.environ.get("S3_ENDPOINT_URL") or None
    # Browser-reachable endpoint used ONLY for presigned URLs (e.g. http://localhost:9000
    # locally). Unset -> presign against the internal endpoint (AWS prod).
    s3_public_endpoint_url: str | None = os.environ.get("S3_PUBLIC_ENDPOINT_URL") or None
    # Where the worker materializes the per-run working cache before running stages.
    work_root: str = os.environ.get("WORK_ROOT", "/scratch/work")

    # --- sessions (no sign-in) ----------------------------------------------
    session_secret: str = os.environ.get("SESSION_SECRET", "dev-insecure-change-me")
    session_cookie: str = os.environ.get("SESSION_COOKIE", "jf_session")
    session_max_age: int = int(os.environ.get("SESSION_MAX_AGE", str(365 * 24 * 3600)))
    cookie_secure: bool = os.environ.get("COOKIE_SECURE", "true").lower() == "true"

    # --- cost guard ($3 MiniMax credit) -------------------------------------
    llm_max_cost_usd: float = float(os.environ.get("LLM_MAX_COST_USD", "3.0"))
    # MiniMax-M3 USD per 1K tokens (override to track real pricing).
    llm_price_in_per_1k: float = float(os.environ.get("LLM_PRICE_IN_PER_1K", "0.0003"))
    llm_price_out_per_1k: float = float(os.environ.get("LLM_PRICE_OUT_PER_1K", "0.0011"))
    # Conservative worst-case token estimate per run for the pre-flight gate.
    llm_est_in_tokens: int = int(os.environ.get("LLM_EST_IN_TOKENS", "150000"))
    llm_est_out_tokens: int = int(os.environ.get("LLM_EST_OUT_TOKENS", "32000"))

    # --- per-session quotas (protect the budget) ----------------------------
    # <= 0 disables the per-session movie cap (unlimited uploads).
    max_movies_per_session: int = int(os.environ.get("MAX_MOVIES_PER_SESSION", "0"))
    max_concurrent_runs_per_session: int = int(os.environ.get("MAX_CONCURRENT_RUNS", "2"))
    upload_url_ttl: int = int(os.environ.get("UPLOAD_URL_TTL", "14400"))  # 4h: large videos over cellular
    result_url_ttl: int = int(os.environ.get("RESULT_URL_TTL", "3600"))

    # --- on-demand GPU supervisor (gpud) ------------------------------------
    # Empty -> disabled: workers use the static ASR_GRPC_TARGET / TTS_GRPC_TARGET
    # always-on servers. Set (e.g. "localhost:50050") -> lease instances from gpud.
    gpu_supervisor_target: str = os.environ.get("GPU_SUPERVISOR_TARGET", "")
    gpud_heartbeat_s: float = float(os.environ.get("GPUD_HEARTBEAT_SEC", "30"))
    gpud_acquire_timeout_s: float = float(os.environ.get("GPUD_ACQUIRE_TIMEOUT_SEC", "180"))
    # How many TTS instances one run leases to synthesize lines in parallel (lines are
    # independent). Capped by gpud's GPUD_TTS_MAX and shared across runs, so it degrades
    # gracefully to however many it can get (>=1). 1 = the old single-instance behaviour.
    tts_instances: int = max(1, int(os.environ.get("TTS_INSTANCES", "3")))

    # --- misc ----------------------------------------------------------------
    pipeline_config: str | None = os.environ.get("PIPELINE_CONFIG") or None
    cors_origins: list[str] = field(default_factory=lambda: _csv(os.environ.get("CORS_ORIGINS", "")))

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    def run_cost_estimate_usd(self) -> float:
        """Worst-case USD a single back-half run can spend (MAP + REDUCE)."""
        return (
            self.llm_est_in_tokens / 1000 * self.llm_price_in_per_1k
            + self.llm_est_out_tokens / 1000 * self.llm_price_out_per_1k
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
