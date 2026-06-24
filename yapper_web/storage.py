"""Single construction point for the web layer's object store.

Local and production differ only by env (MinIO endpoint vs AWS S3) — same boto3 code
path — so both the API and the Celery workers build their `S3Storage` here, threading
the internal endpoint (server-side ops) and the public endpoint (browser presigned
URLs) from settings. `LocalStorage` is intentionally NOT used in the web layer; it's a
CLI-only shim.
"""

from __future__ import annotations

from yapper.storage import S3Storage

from .settings import Settings, get_settings


def storage_for_web(settings: Settings | None = None) -> S3Storage:
    s = settings or get_settings()
    return S3Storage(
        s.s3_bucket,
        endpoint_url=s.s3_endpoint_url,
        public_endpoint_url=s.s3_public_endpoint_url,
        region=s.s3_region,
    )
