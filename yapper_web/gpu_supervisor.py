"""Client side of the on-demand GPU model: talk to `gpud` to lease an ASR/TTS instance
for the duration of a Celery task.

`lease(service)` is the one entry point the worker uses. When `GPU_SUPERVISOR_TARGET` is
set it: Acquires an instance (gpud launches/reuses one on a GPU with free vRAM), spawns a
background heartbeat so the lease isn't reclaimed mid-task, yields the instance's
**target** (host:port) for the worker to dial, and Releases in `finally`. When the flag
is empty it yields `None` — the caller then falls back to the static `*_GRPC_TARGET` env
(the always-on path), so the supervisor is fully opt-in.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import grpc

from yapper_rpc import gpud_pb2, gpud_pb2_grpc

from .settings import Settings, get_settings

log = logging.getLogger("yapper_web.gpu_supervisor")


class GpudClient:
    def __init__(self, target: str, channel: grpc.Channel | None = None):
        self.target = target
        self._channel = channel or grpc.insecure_channel(target)
        self._stub = gpud_pb2_grpc.GpudStub(self._channel)

    def acquire(self, service: str, timeout: float) -> gpud_pb2.Lease:
        # gpud may block server-side until the pool frees up; give the RPC room beyond that.
        return self._stub.Acquire(gpud_pb2.AcquireRequest(service=service), timeout=timeout + 30)

    def heartbeat(self, lease_id: str) -> bool:
        return self._stub.Heartbeat(gpud_pb2.LeaseRef(lease_id=lease_id), timeout=10).ok

    def release(self, lease_id: str) -> bool:
        return self._stub.Release(gpud_pb2.LeaseRef(lease_id=lease_id), timeout=10).ok

    def close(self) -> None:
        self._channel.close()


def _acquire_one(service: str, s: Settings) -> tuple[str, Callable[[], None]]:
    """Acquire one gpud instance. Returns ``(dial_target, release)`` — ``release`` stops the
    heartbeat and frees the lease. Raises if the pool can't satisfy within the timeout."""
    client = GpudClient(s.gpu_supervisor_target)
    lease_obj = client.acquire(service, s.gpud_acquire_timeout_s)

    stop = threading.Event()

    def release() -> None:
        stop.set()
        try:
            client.release(lease_obj.lease_id)
        except Exception as e:  # noqa: BLE001 — gpud reaps via TTL even if release is lost
            log.warning("gpud release failed (%s): %s", lease_obj.lease_id, e)
        client.close()

    # Once the lease exists server-side, ANY failure before we hand back `release` would
    # orphan it (and leak the channel) — so release+close on a mid-setup raise.
    try:
        # gpud reports the instance as localhost:<port> (box-local). The worker reaches the box
        # at the SAME host it reaches gpud (tunnel host / host.docker.internal / Service DNS), so
        # dial that host with the leased port rather than gpud's literal "localhost".
        host = s.gpu_supervisor_target.rsplit(":", 1)[0]
        port = lease_obj.target.rsplit(":", 1)[-1]
        dial_target = f"{host}:{port}"
        log.info("gpud: leased %s -> %s (%s)", service, dial_target, lease_obj.lease_id)

        def _heartbeat() -> None:
            while not stop.wait(s.gpud_heartbeat_s):
                try:
                    client.heartbeat(lease_obj.lease_id)
                except Exception as e:  # noqa: BLE001 — transient; gpud TTL covers a real outage
                    log.warning("gpud heartbeat failed (%s): %s", lease_obj.lease_id, e)

        threading.Thread(target=_heartbeat, name=f"gpud-hb-{service}", daemon=True).start()
    except BaseException:
        release()
        raise

    return dial_target, release


@contextmanager
def lease(service: str, settings: Settings | None = None) -> Iterator[str | None]:
    """Lease a GPU instance for the wrapped work. Yields the target host:port, or ``None``
    when the supervisor is disabled (caller falls back to the static env target)."""
    s = settings or get_settings()
    if not s.gpu_supervisor_target:
        yield None
        return
    target, release = _acquire_one(service, s)
    try:
        yield target
    finally:
        release()


@contextmanager
def lease_many(service: str, count: int, settings: Settings | None = None) -> Iterator[list[str | None]]:
    """Lease up to ``count`` instances to run independent work in parallel (e.g. TTS lines).

    Acquires CONCURRENTLY so cold-start launches overlap (one cold-start of latency, not
    ``count``). Requires at least one — raises if even that fails. Extra instances are
    best-effort: when the pool (capped by GPUD_TTS_MAX, shared across runs) can't satisfy
    all of them, the surplus ``Acquire`` calls raise and are dropped, so a run degrades to
    whatever it could get (>=1). Every acquired lease is heartbeated and released, so there
    is no leak. Yields ``[None]`` when the supervisor is disabled (static-target fallback).

    Caveat: under genuine cross-run contention the surplus acquires block until the gpud
    acquire timeout before being dropped (gpud has no non-blocking try-acquire), so the
    stage start can be delayed in that rare case; lower ``count`` or add a try-acquire RPC
    if that becomes real.
    """
    s = settings or get_settings()
    if not s.gpu_supervisor_target or count <= 1:
        with lease(service, s) as t:
            yield [t]
        return

    import concurrent.futures

    acquired: list[tuple[str, Callable[[], None]]] = []
    errors: list[Exception] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as ex:
        for fut in [ex.submit(_acquire_one, service, s) for _ in range(count)]:
            try:
                acquired.append(fut.result())
            except Exception as e:  # noqa: BLE001 — surplus instance unavailable; degrade
                errors.append(e)

    if not acquired:
        raise errors[0] if errors else RuntimeError(f"no {service} instance available")
    if errors:
        log.info("gpud: leased %d/%d %s instances (%d unavailable)", len(acquired), count, service, len(errors))
    targets = [t for t, _ in acquired]
    try:
        yield targets
    finally:
        for _, release in acquired:
            release()
