"""Per-stage wall-clock timing for a single pipeline run.

Wraps each stage so a run records how long every step took — and which were
served from cache — into ``artifacts/<slug>/timings.json``. The file is rewritten
after every stage so a partial/interrupted run still leaves a useful record.

Usage::

    timer = RunTimer(mdir / "timings.json")
    if needs_to_run:
        with timer.stage("shots"):
            ...do work...
    else:
        timer.skipped("shots")
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("yapper.timing")

# (stage_name, seconds, status) — status is "running" | "ran" | "cached" | "error".
StageEvent = Callable[[str, float, str], None]


class RunTimer:
    """Accumulates per-stage durations and flushes them to a JSON file.

    ``on_event`` (optional) is invoked as ``on_event(stage, seconds, status)`` the
    moment a stage starts (``status="running"``, ``seconds=0``) and again when it ends
    (``"ran"``/``"error"`` with the measured duration) or is skipped (``"cached"``). The
    platform passes a callback that mirrors these into the DB so progress is live *and*
    the real duration of each stage survives a chain that re-enters the pipeline per
    resource (the JSON file only keeps the last writer's view)."""

    def __init__(self, path: str | Path, *, on_event: StageEvent | None = None):
        self.path = Path(path)
        self.stages: list[dict] = []
        self._on_event = on_event
        self._wall_start = time.perf_counter()
        self._started_at = datetime.now(timezone.utc)

    @contextmanager
    def stage(self, name: str):
        """Time a stage that is actually running; records 'error' if it raises."""
        self._emit(name, 0.0, "running")
        start = time.perf_counter()
        try:
            yield
        except BaseException:
            self._record(name, time.perf_counter() - start, "error")
            raise
        else:
            self._record(name, time.perf_counter() - start, "ran")

    def skipped(self, name: str) -> None:
        """Record a stage that was served from cache (0s of compute)."""
        self._record(name, 0.0, "cached")

    # -- internals ----------------------------------------------------------
    def _record(self, name: str, seconds: float, status: str) -> None:
        self.stages.append({"stage": name, "seconds": round(seconds, 3), "status": status})
        log.info("[time] %-10s %8.2fs  (%s)", name, seconds, status)
        self._emit(name, round(seconds, 3), status)
        self._flush()

    def _emit(self, name: str, seconds: float, status: str) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(name, seconds, status)
        except Exception as e:  # noqa: BLE001 — progress tracking must never break a run
            log.warning("stage event callback failed for %s (%s): %s", name, status, e)

    def _flush(self) -> None:
        computed = round(sum(s["seconds"] for s in self.stages), 3)
        data = {
            "started_at": self._started_at.isoformat(timespec="seconds"),
            "wall_seconds": round(time.perf_counter() - self._wall_start, 3),
            "computed_seconds": computed,
            "stages": self.stages,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as e:  # timing must never break a render
            log.warning("could not write timings to %s: %s", self.path, e)
