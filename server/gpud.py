"""gpud — on-demand GPU supervisor (runs always-on on the GPU box, no model loaded).

Owns a bounded pool of ASR/TTS model-server subprocesses. A worker leases an instance
for one Celery task: gpud reuses a warm idle instance or launches a new one on a GPU
that has free vRAM (picking an instance port from a reserved range), waits for it to be
SERVING, and hands back its address. After the lease is released the instance goes idle
and is reaped once it stays idle past the grace period — freeing the GPU. Model traffic
goes worker -> instance directly (gpud is control-plane only).

The pool/placement/lease logic (`Supervisor`) is pure and unit-testable: NVML
(`GpuProbe`), process launch (`Launcher`), and the clock are injected adapters with
production defaults (`NvmlProbe`, `SubprocessLauncher`, `time.monotonic`).

Run (in the ASR uv env, which also carries NVML):
    cd ~/jieshuo && PYTHONPATH=~/jieshuo \
      uv run --no-sync --project server/asr python server/gpud.py
"""

from __future__ import annotations

import itertools
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import grpc

from yapper_rpc import gpud_pb2, gpud_pb2_grpc

# ---------------------------------------------------------------------------
# adapters (injected; production defaults below)
# ---------------------------------------------------------------------------
class GpuProbe(Protocol):
    def gpus(self) -> list[int]: ...
    def free_mb(self, index: int) -> int: ...
    def total_mb(self, index: int) -> int: ...


class Launcher(Protocol):
    def launch(self, service: str, gpu_index: int, port: int) -> object:
        """Start the model server (CUDA_VISIBLE_DEVICES=gpu, *_GRPC_PORT=port); return a handle."""
        ...

    def wait_ready(self, handle: object, port: int, timeout: float) -> None:
        """Block until the server is gRPC-health SERVING; raise on timeout/failure."""
        ...

    def terminate(self, handle: object) -> None: ...


@dataclass
class ServiceSpec:
    name: str
    max_instances: int
    vram_mb: int


@dataclass
class Config:
    services: dict[str, ServiceSpec]
    port_range: tuple[int, int]            # inclusive
    headroom_mb: int = 1024
    idle_timeout_s: float = 60.0
    lease_ttl_s: float = 120.0
    acquire_timeout_s: float = 180.0
    launch_timeout_s: float = 120.0
    visible_gpus: list[int] | None = None  # None = all the probe reports


# ---------------------------------------------------------------------------
# pool state
# ---------------------------------------------------------------------------
@dataclass
class Instance:
    service: str
    gpu_index: int
    port: int
    state: str = "starting"          # starting -> ready
    leased: bool = False
    lease_id: str | None = None
    released_at: float | None = None  # monotonic; when it last went idle
    handle: object | None = None

    @property
    def target(self) -> str:
        # host:port the worker dials. localhost because the worker reaches it through the
        # SSH tunnel (gpud + the port range are forwarded to the box).
        return f"localhost:{self.port}"


def instance_state(inst: Instance) -> str:
    """Lifecycle bucket for the metrics breakdown of pooled processes: a leased instance
    is busy serving a task; an unleased one is either still loading its model (``starting``
    — a cold start) or warm and ``idle``."""
    if inst.leased:
        return "leased"
    return "idle" if inst.state == "ready" else inst.state


@dataclass
class _Lease:
    lease_id: str
    instance: Instance
    expires_at: float


class Unavailable(RuntimeError):
    """Pool full / no GPU fits within the acquire timeout."""


_ids = itertools.count(1)


class Supervisor:
    """Thread-safe on-demand pool. Slow launches happen OUTSIDE the lock; a Condition
    coordinates waiters so releases/launch-completions wake blocked Acquires."""

    def __init__(self, cfg: Config, probe: GpuProbe, launcher: Launcher,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.probe = probe
        self.launcher = launcher
        self.clock = clock
        self._cond = threading.Condition()
        self._instances: list[Instance] = []
        self._leases: dict[str, _Lease] = {}

    # -- public API ---------------------------------------------------------
    def acquire(self, service: str) -> _Lease:
        if service not in self.cfg.services:
            raise ValueError(f"unknown service {service!r}")
        deadline = self.clock() + self.cfg.acquire_timeout_s
        while True:
            reserved: Instance | None = None
            with self._cond:
                self._reclaim_expired_locked()
                # 1) reuse a warm idle instance
                inst = self._take_idle_locked(service)
                if inst is not None:
                    return self._lease_locked(inst)
                # 2) launch a new instance if under MAX and a GPU fits
                if self._count_locked(service) < self.cfg.services[service].max_instances:
                    gpu = self._select_gpu_locked(service)
                    if gpu is not None:
                        reserved = self._reserve_starting_locked(service, gpu)
                # 3) otherwise wait below
            if reserved is not None:
                if self._launch(reserved):           # slow work, no lock held
                    with self._cond:
                        reserved.state = "ready"
                        lease = self._lease_locked(reserved)
                        self._cond.notify_all()
                        return lease
                # launch failed: drop the reservation and retry placement
                with self._cond:
                    self._drop_locked(reserved)
                    self._cond.notify_all()
                continue
            # 3) block until something frees up or we time out
            with self._cond:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise Unavailable(f"no capacity for {service} within {self.cfg.acquire_timeout_s}s")
                self._cond.wait(timeout=min(remaining, 1.0))

    def heartbeat(self, lease_id: str) -> bool:
        with self._cond:
            lease = self._leases.get(lease_id)
            if lease is None:
                return False
            lease.expires_at = self.clock() + self.cfg.lease_ttl_s
            return True

    def release(self, lease_id: str) -> bool:
        with self._cond:
            lease = self._leases.pop(lease_id, None)
            if lease is None:
                return False
            self._mark_idle_locked(lease.instance)
            self._cond.notify_all()
            return True

    def reap_once(self) -> int:
        """Reclaim TTL-expired leases and SIGTERM instances idle past the grace. Returns
        the number of instances reaped. Called by the background thread (and tests)."""
        reaped = 0
        with self._cond:
            self._reclaim_expired_locked()
            now = self.clock()
            for inst in list(self._instances):
                if (not inst.leased and inst.state == "ready"
                        and inst.released_at is not None
                        and now - inst.released_at >= self.cfg.idle_timeout_s):
                    self.launcher.terminate(inst.handle)
                    self._instances.remove(inst)
                    reaped += 1
            if reaped:
                self._cond.notify_all()
        return reaped

    def status(self) -> tuple[list[Instance], dict[int, tuple[int, int]]]:
        with self._cond:
            gpus = {g: (self.probe.free_mb(g), self.probe.total_mb(g)) for g in self._visible_gpus()}
            return list(self._instances), gpus

    def shutdown(self) -> int:
        """Terminate every launched instance so model servers don't orphan and pin vRAM when
        gpud stops (a bare kill of gpud would otherwise leave them running). Returns the count
        terminated. Terminate outside the lock — each terminate() waits on the process."""
        with self._cond:
            insts = list(self._instances)
            self._instances.clear()
            self._leases.clear()
        for inst in insts:
            self.launcher.terminate(inst.handle)
        return len(insts)

    # -- internals (call with lock held unless noted) -----------------------
    def _visible_gpus(self) -> list[int]:
        all_g = self.probe.gpus()
        if self.cfg.visible_gpus is None:
            return all_g
        return [g for g in all_g if g in self.cfg.visible_gpus]

    def _count_locked(self, service: str) -> int:
        return sum(1 for i in self._instances if i.service == service)

    def _take_idle_locked(self, service: str) -> Instance | None:
        for inst in self._instances:
            if inst.service == service and inst.state == "ready" and not inst.leased:
                return inst
        return None

    def _pending_mb_locked(self, gpu: int) -> int:
        # vRAM of instances we've placed on this GPU but that haven't loaded yet — NVML's
        # free_mb doesn't reflect them, so subtract to avoid double-booking during launch.
        return sum(self.cfg.services[i.service].vram_mb
                   for i in self._instances if i.gpu_index == gpu and i.state == "starting")

    def _select_gpu_locked(self, service: str) -> int | None:
        need = self.cfg.services[service].vram_mb + self.cfg.headroom_mb
        for gpu in self._visible_gpus():
            effective_free = self.probe.free_mb(gpu) - self._pending_mb_locked(gpu)
            if effective_free >= need:
                return gpu
        return None

    def _free_port_locked(self) -> int:
        used = {i.port for i in self._instances}
        lo, hi = self.cfg.port_range
        for p in range(lo, hi + 1):
            if p not in used:
                return p
        raise Unavailable(f"no free port in range {lo}-{hi}")

    def _reserve_starting_locked(self, service: str, gpu: int) -> Instance:
        inst = Instance(service=service, gpu_index=gpu, port=self._free_port_locked())
        self._instances.append(inst)
        return inst

    def _drop_locked(self, inst: Instance) -> None:
        if inst in self._instances:
            self._instances.remove(inst)

    def _lease_locked(self, inst: Instance) -> _Lease:
        lease = _Lease(lease_id=f"lease-{next(_ids)}", instance=inst,
                       expires_at=self.clock() + self.cfg.lease_ttl_s)
        inst.leased = True
        inst.lease_id = lease.lease_id
        inst.released_at = None
        self._leases[lease.lease_id] = lease
        return lease

    def _mark_idle_locked(self, inst: Instance) -> None:
        inst.leased = False
        inst.lease_id = None
        inst.released_at = self.clock()

    def _reclaim_expired_locked(self) -> None:
        now = self.clock()
        for lease_id, lease in list(self._leases.items()):
            if lease.expires_at < now:          # holder died (stopped heartbeating)
                del self._leases[lease_id]
                self._mark_idle_locked(lease.instance)

    def _launch(self, inst: Instance) -> bool:
        try:
            inst.handle = self.launcher.launch(inst.service, inst.gpu_index, inst.port)
            self.launcher.wait_ready(inst.handle, inst.port, self.cfg.launch_timeout_s)
            return True
        except Exception:  # noqa: BLE001 — caller drops the reservation + retries
            if inst.handle is not None:
                try:
                    self.launcher.terminate(inst.handle)
                except Exception:  # noqa: BLE001
                    pass
            return False


# ---------------------------------------------------------------------------
# production adapters
# ---------------------------------------------------------------------------
class NvmlProbe:
    def __init__(self):
        import pynvml  # lazy: only needed on the box

        self._n = pynvml
        self._n.nvmlInit()
        self._handles = {i: self._n.nvmlDeviceGetHandleByIndex(i)
                         for i in range(self._n.nvmlDeviceGetCount())}

    def gpus(self) -> list[int]:
        return sorted(self._handles)

    def free_mb(self, index: int) -> int:
        return int(self._n.nvmlDeviceGetMemoryInfo(self._handles[index]).free // (1024 * 1024))

    def total_mb(self, index: int) -> int:
        return int(self._n.nvmlDeviceGetMemoryInfo(self._handles[index]).total // (1024 * 1024))


class SubprocessLauncher:
    """Launches the existing asr_service.py / tts_service.py as subprocesses. The command
    per service comes from env (GPUD_ASR_CMD / GPUD_TTS_CMD); the child inherits gpud's
    env (HF_TOKEN, model dirs, ref wav, ...) plus CUDA_VISIBLE_DEVICES and the port var."""

    PORT_ENV = {"asr": "ASR_GRPC_PORT", "tts": "TTS_GRPC_PORT"}

    def __init__(self, commands: dict[str, list[str]]):
        self.commands = commands  # {"asr": [...argv...], "tts": [...]}

    def launch(self, service: str, gpu_index: int, port: int):
        import subprocess

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        env[self.PORT_ENV[service]] = str(port)
        # metrics side port stays off by default for pooled instances (avoid port sprawl)
        env.setdefault(f"{service.upper()}_METRICS_PORT", "0")
        return subprocess.Popen(self.commands[service], env=env)

    def wait_ready(self, handle, port: int, timeout: float) -> None:
        from grpc_health.v1 import health_pb2, health_pb2_grpc

        deadline = time.monotonic() + timeout
        channel = grpc.insecure_channel(f"localhost:{port}")
        stub = health_pb2_grpc.HealthStub(channel)
        last = None
        while time.monotonic() < deadline:
            if handle.poll() is not None:
                raise RuntimeError(f"server exited early (code {handle.returncode})")
            try:
                resp = stub.Check(health_pb2.HealthCheckRequest(), timeout=5)
                if resp.status == health_pb2.HealthCheckResponse.SERVING:
                    channel.close()
                    return
            except grpc.RpcError as e:  # not up yet
                last = e
            time.sleep(2)
        channel.close()
        raise TimeoutError(f"server on :{port} not SERVING within {timeout}s ({last})")

    def terminate(self, handle) -> None:
        if handle is None:
            return
        handle.terminate()
        try:
            handle.wait(timeout=10)
        except Exception:  # noqa: BLE001
            handle.kill()


# ---------------------------------------------------------------------------
# gRPC servicer + main
# ---------------------------------------------------------------------------
class GpudServicer(gpud_pb2_grpc.GpudServicer):
    def __init__(self, sup: Supervisor):
        self.sup = sup

    def Acquire(self, request, context):
        try:
            lease = self.sup.acquire(request.service)
        except Unavailable as e:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(e))
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        return gpud_pb2.Lease(lease_id=lease.lease_id, target=lease.instance.target,
                              service=request.service, ready=True)

    def Heartbeat(self, request, context):
        return gpud_pb2.HeartbeatReply(ok=self.sup.heartbeat(request.lease_id))

    def Release(self, request, context):
        return gpud_pb2.ReleaseReply(ok=self.sup.release(request.lease_id))

    def Status(self, request, context):
        instances, gpus = self.sup.status()
        return gpud_pb2.StatusReply(
            instances=[gpud_pb2.InstanceStatus(
                service=i.service, target=i.target, gpu_index=i.gpu_index,
                leased=i.leased, lease_id=i.lease_id or "") for i in instances],
            gpus=[gpud_pb2.GpuStatus(index=g, free_mb=f, total_mb=t) for g, (f, t) in gpus.items()],
        )


def _config_from_env() -> Config:
    lo, hi = os.environ.get("GPUD_PORT_RANGE", "50060-50099").split("-")
    visible = os.environ.get("GPUD_VISIBLE_GPUS")
    return Config(
        services={
            "asr": ServiceSpec("asr", int(os.environ.get("GPUD_ASR_MAX", "3")),
                               int(os.environ.get("GPUD_ASR_VRAM_MB", "7600"))),
            "tts": ServiceSpec("tts", int(os.environ.get("GPUD_TTS_MAX", "3")),
                               int(os.environ.get("GPUD_TTS_VRAM_MB", "4500"))),
        },
        port_range=(int(lo), int(hi)),
        headroom_mb=int(os.environ.get("GPUD_VRAM_HEADROOM_MB", "1024")),
        idle_timeout_s=float(os.environ.get("GPUD_IDLE_TIMEOUT_SEC", "60")),
        lease_ttl_s=float(os.environ.get("GPUD_LEASE_TTL_SEC", "120")),
        acquire_timeout_s=float(os.environ.get("GPUD_ACQUIRE_TIMEOUT_SEC", "180")),
        visible_gpus=[int(x) for x in visible.split(",")] if visible else None,
    )


def serve() -> None:
    from concurrent import futures

    from grpc_health.v1 import health, health_pb2, health_pb2_grpc

    cfg = _config_from_env()
    # Launch each service in its own uv env (CWD is ~/jieshuo). --no-sync skips the per-launch
    # resolve (envs are synced once at setup); children inherit gpud's env (PYTHONPATH,
    # HF_TOKEN, model dirs, ...) plus CUDA_VISIBLE_DEVICES + the assigned port.
    commands = {
        "asr": os.environ.get(
            "GPUD_ASR_CMD", "uv run --no-sync --project server/asr python server/asr_service.py").split(),
        "tts": os.environ.get(
            "GPUD_TTS_CMD", "uv run --no-sync --project server/tts python server/tts_service.py").split(),
    }
    sup = Supervisor(cfg, NvmlProbe(), SubprocessLauncher(commands))

    # background reaper
    stop = threading.Event()

    def _reaper():
        while not stop.wait(5.0):
            try:
                sup.reap_once()
            except Exception:  # noqa: BLE001 — reaping must never crash the daemon
                pass

    threading.Thread(target=_reaper, daemon=True).start()

    _start_metrics(sup)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    gpud_pb2_grpc.add_GpudServicer_to_server(GpudServicer(sup), server)
    hs = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(hs, server)
    hs.set("yapper_rpc.Gpud", health_pb2.HealthCheckResponse.SERVING)
    hs.set("", health_pb2.HealthCheckResponse.SERVING)
    port = int(os.environ.get("GPUD_PORT", "50050"))
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"gpud on :{port}  (port range {cfg.port_range}, asr_max={cfg.services['asr'].max_instances}, "
          f"tts_max={cfg.services['tts'].max_instances})")

    # Graceful shutdown: on SIGTERM/SIGINT, terminate every launched instance so the model
    # servers don't orphan and keep pinning vRAM (a bare kill of gpud would strand them).
    import signal
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    signal.signal(signal.SIGINT, lambda *_: done.set())
    done.wait()
    stop.set()                                  # halt the reaper thread
    n = sup.shutdown()
    print(f"gpud: terminating {n} instance(s) and stopping")
    server.stop(grace=5).wait()


def _start_metrics(sup: Supervisor) -> None:
    try:
        from prometheus_client import REGISTRY, start_http_server
        from prometheus_client.core import GaugeMetricFamily

        class _Collector:
            def collect(self):
                instances, gpus = sup.status()
                up = GaugeMetricFamily(
                    "yapper_gpud_instances", "Live model-server processes by service and lifecycle",
                    labels=["service", "state"],
                )
                counts: dict[tuple[str, str], int] = {}
                for i in instances:
                    key = (i.service, instance_state(i))
                    counts[key] = counts.get(key, 0) + 1
                for (svc, st), n in counts.items():
                    up.add_metric([svc, st], n)
                yield up
                free = GaugeMetricFamily("yapper_gpud_gpu_free_mb", "Free vRAM per GPU", labels=["gpu"])
                total = GaugeMetricFamily("yapper_gpud_gpu_total_mb", "Total vRAM per GPU", labels=["gpu"])
                for g, (f, t) in gpus.items():
                    free.add_metric([str(g)], f)
                    total.add_metric([str(g)], t)
                yield free
                yield total

        REGISTRY.register(_Collector())
        start_http_server(int(os.environ.get("GPUD_METRICS_PORT", "9050")))
    except Exception:  # noqa: BLE001 — metrics optional
        pass


if __name__ == "__main__":
    serve()
