"""Storage abstraction: a durable artifact store with a local working cache.

The CLI runs entirely on local disk (``LocalStorage`` — a no-op shim, so behaviour
is byte-identical to before). The platform runs the same pipeline against ``S3Storage``:
S3 is the durable, per-session store; a per-run *local working directory* is the cache
the pipeline actually reads/writes.

The integration seam is the **run boundary**, never deep inside ffmpeg/sqlite (those
need real local files). A worker task does:

    store.materialize(prefix, work_dir)     # pull existing artifacts down (resume)
    cfg.raw["paths"]["artifacts_dir"] = str(work_dir.parent)  # point Config at the cache
    run_front_half(movie, cfg, ...)          # unchanged; sees cached artifacts, skips
    store.persist(work_dir, prefix)          # push new/changed artifacts up

Because every path in the pipeline flows through ``Config`` (``artifacts_dir`` /
``scratch_dir`` / ``movie_dir``), pointing those at the working dir is the only wiring
needed. On a single host the working dir is warm, so materialize/persist are cheap;
on a second host they refill from S3 — horizontal scale becomes config, not a rewrite.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Protocol, runtime_checkable

log = logging.getLogger("yapper.storage")


@runtime_checkable
class Storage(Protocol):
    """Durable artifact store. Keys are POSIX-style relative paths (``a/b/c.json``)."""

    def exists(self, key: str) -> bool: ...

    def upload(self, local_path: str | Path, key: str) -> None: ...

    def download(self, key: str, local_path: str | Path) -> None: ...

    def materialize(self, prefix: str, local_dir: str | Path) -> int:
        """Pull every object under ``prefix`` into ``local_dir`` (mirrors layout).
        Returns the number of files fetched. Resumability: a stage whose artifact is
        already present locally (or pulled here) is skipped by the pipeline."""
        ...

    def persist(self, local_dir: str | Path, prefix: str) -> int:
        """Push new/changed files from ``local_dir`` up under ``prefix``.
        Returns the number of files uploaded."""
        ...

    def delete_prefix(self, prefix: str) -> int:
        """Delete every object under ``prefix`` (e.g. to clear one stage's cache so a
        regenerate recomputes it fresh). Returns the number of objects deleted."""
        ...

    def presign_put(self, key: str, expires: int = 3600, content_type: str | None = None) -> str: ...

    def presign_get(self, key: str, expires: int = 3600) -> str: ...


def _iter_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class LocalStorage:
    """No-op shim: the pipeline already works on local disk, so materialize/persist
    do nothing and there is no durable copy to sync. Used by the CLI so its behaviour
    is unchanged. ``upload``/``download`` copy within the filesystem (rooted at
    ``root``) for the rare caller that asks for a key explicitly."""

    def __init__(self, root: str | Path = "."):
        self.root = Path(root)

    def _abs(self, key: str) -> Path:
        return self.root / key

    def exists(self, key: str) -> bool:
        return self._abs(key).exists()

    def upload(self, local_path: str | Path, key: str) -> None:
        dst = self._abs(key)
        if Path(local_path).resolve() == dst.resolve():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)

    def download(self, key: str, local_path: str | Path) -> None:
        src = self._abs(key)
        dst = Path(local_path)
        if src.resolve() == dst.resolve():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def materialize(self, prefix: str, local_dir: str | Path) -> int:  # noqa: ARG002
        return 0  # already local — nothing to fetch

    def persist(self, local_dir: str | Path, prefix: str) -> int:  # noqa: ARG002
        return 0  # already local — nothing durable to sync to

    def delete_prefix(self, prefix: str) -> int:
        target = self._abs(prefix)
        if not target.exists():
            return 0
        n = sum(1 for _ in _iter_files(target)) if target.is_dir() else 1
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
        return n

    def presign_put(self, key: str, expires: int = 3600, content_type: str | None = None) -> str:
        return f"file://{self._abs(key)}"

    def presign_get(self, key: str, expires: int = 3600) -> str:
        return f"file://{self._abs(key)}"


class S3Storage:
    """S3-backed durable store. Keys map 1:1 to object keys under an optional
    ``base_prefix``. ``materialize``/``persist`` mirror a local working dir to/from a
    key prefix, skipping unchanged files (size + md5 against the stored ETag) so a
    warm cache and resumed runs stay cheap.

    boto3 is imported lazily so the core package has no hard dependency on it (the CLI
    never constructs this class)."""

    def __init__(
        self,
        bucket: str,
        *,
        base_prefix: str = "",
        endpoint_url: str | None = None,
        public_endpoint_url: str | None = None,
        region: str | None = None,
        client=None,
    ):
        self.bucket = bucket
        self.base_prefix = base_prefix.strip("/")
        if client is not None:
            self.s3 = client
            # tests/injected clients: presign against the same client
            self._presign = client
        else:
            import boto3  # lazy: optional dependency, only on the server

            # MinIO and other non-AWS S3 need path-style addressing (no bucket-in-host)
            # for presigned URLs to validate; harmless for AWS.
            cfg = None
            if endpoint_url or public_endpoint_url:
                from botocore.client import Config

                cfg = Config(s3={"addressing_style": "path"})
            self.s3 = boto3.client("s3", endpoint_url=endpoint_url, region_name=region, config=cfg)
            # Presigned URLs embed the endpoint host. Inside the cluster we reach storage
            # at `endpoint_url` (e.g. http://minio:9000), but the browser must use a
            # reachable address — sign those URLs against `public_endpoint_url` when set
            # (local MinIO). Prod/AWS leaves both unset -> one default client.
            self._presign = (
                boto3.client("s3", endpoint_url=public_endpoint_url, region_name=region, config=cfg)
                if public_endpoint_url else self.s3
            )

    # -- key helpers --------------------------------------------------------
    def _key(self, key: str) -> str:
        key = key.lstrip("/")
        return f"{self.base_prefix}/{key}" if self.base_prefix else key

    # -- single-object ops --------------------------------------------------
    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def upload(self, local_path: str | Path, key: str) -> None:
        self.s3.upload_file(str(local_path), self.bucket, self._key(key))

    def download(self, key: str, local_path: str | Path) -> None:
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, self._key(key), str(dst))

    # -- directory <-> prefix mirroring ------------------------------------
    def _list(self, prefix: str) -> dict[str, dict]:
        """Map relative-key -> {size, etag} for everything under ``prefix``."""
        out: dict[str, dict] = {}
        full = self._key(prefix).rstrip("/") + "/"
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(full):]
                if rel:
                    out[rel] = {"size": obj["Size"], "etag": obj["ETag"].strip('"')}
        return out

    def materialize(self, prefix: str, local_dir: str | Path) -> int:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        remote = self._list(prefix)
        fetched = 0
        for rel, meta in remote.items():
            dst = local_dir / rel
            if dst.exists() and dst.stat().st_size == meta["size"]:
                continue  # warm cache hit (size match; etag check is best-effort below)
            dst.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, self._key(f"{prefix}/{rel}"), str(dst))
            fetched += 1
        if fetched:
            log.info("materialize %s -> %s: %d files", prefix, local_dir, fetched)
        return fetched

    def persist(self, local_dir: str | Path, prefix: str) -> int:
        local_dir = Path(local_dir)
        if not local_dir.exists():
            return 0
        remote = self._list(prefix)
        uploaded = 0
        for path in _iter_files(local_dir):
            rel = path.relative_to(local_dir).as_posix()
            meta = remote.get(rel)
            # Skip when size matches and (for non-multipart objects) the md5 matches the ETag.
            if meta and meta["size"] == path.stat().st_size:
                if "-" in meta["etag"] or _md5(path) == meta["etag"]:
                    continue
            self.s3.upload_file(str(path), self.bucket, self._key(f"{prefix}/{rel}"))
            uploaded += 1
        if uploaded:
            log.info("persist %s -> %s: %d files", local_dir, prefix, uploaded)
        return uploaded

    def delete_prefix(self, prefix: str) -> int:
        full = self._key(prefix).rstrip("/") + "/"
        paginator = self.s3.get_paginator("list_objects_v2")
        keys = [
            obj["Key"]
            for page in paginator.paginate(Bucket=self.bucket, Prefix=full)
            for obj in page.get("Contents", [])
        ]
        deleted = 0
        for i in range(0, len(keys), 1000):     # delete_objects caps at 1000 keys/call
            batch = keys[i:i + 1000]
            self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": [{"Key": k} for k in batch]})
            deleted += len(batch)
        if deleted:
            log.info("delete_prefix %s: %d objects", full, deleted)
        return deleted

    # -- presigned URLs (browser direct upload/download) -------------------
    def presign_put(self, key: str, expires: int = 3600, content_type: str | None = None) -> str:
        params = {"Bucket": self.bucket, "Key": self._key(key)}
        if content_type:
            params["ContentType"] = content_type
        return self._presign.generate_presigned_url("put_object", Params=params, ExpiresIn=expires)

    def presign_get(self, key: str, expires: int = 3600) -> str:
        return self._presign.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": self._key(key)}, ExpiresIn=expires
        )


def from_env() -> Storage:
    """Pick a backend from the environment. ``STORAGE_BACKEND=s3`` -> S3 (needs
    ``S3_BUCKET``); anything else -> LocalStorage rooted at the CWD (CLI default)."""
    backend = os.environ.get("STORAGE_BACKEND", "local").lower()
    if backend == "s3":
        bucket = os.environ.get("S3_BUCKET")
        if not bucket:
            raise RuntimeError("STORAGE_BACKEND=s3 requires S3_BUCKET")
        return S3Storage(
            bucket,
            base_prefix=os.environ.get("S3_PREFIX", ""),
            endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
            public_endpoint_url=os.environ.get("S3_PUBLIC_ENDPOINT_URL"),
            region=os.environ.get("AWS_REGION"),
        )
    return LocalStorage(os.environ.get("ARTIFACTS_ROOT", "."))
