"""Cost control — the $3 MiniMax-credit guard.

The whole platform shares ONE small pool of LLM credit, so spend is tracked globally
in Redis as an authoritative counter (atomic, survives restarts, shared across
workers). Two enforcement points:

1. **Pre-flight gate** (`reserve`): before a back-half run is enqueued/started, check
   that the remaining budget covers a worst-case run estimate. If not, the run is
   refused *before* any tokens are spent.
2. **Post-call accounting** (`record_usage`): after each LLM call returns, convert
   ``usage`` to USD and decrement the pool. A `BudgetGuard` wired into
   ``LLMClient.complete_structured`` also re-checks *inside* the worker, so a run that
   slips past the gate still aborts cleanly mid-flight instead of overspending.

A reservation is a soft hold released when the run records its actual spend, so two
concurrent runs can't both pass the gate against the same headroom.
"""

from __future__ import annotations

import logging

import redis

from .settings import Settings, get_settings

log = logging.getLogger("yapper_web.budget")

# Redis keys
_SPENT = "jf:llm:spent_usd_milli"        # integer milli-USD actually spent
_RESERVED = "jf:llm:reserved_usd_milli"  # integer milli-USD held by in-flight runs

# Per-stage attribution tallies (metrics only; NOT authoritative for the cap). Hashes
# keyed by pipeline stage ("understand" | "script" | ...), surfaced by the API collector
# as yapper_llm_stage_* series so spend/tokens split by the MAP (understand) vs REDUCE
# (script) pass. Kept separate from _SPENT so a metrics failure can't affect enforcement.
_STAGE_COST_MILLI = "jf:llm:stage_cost_usd_milli"   # hash: stage -> milli-USD
_STAGE_TOKENS_IN = "jf:llm:stage_tokens_in"         # hash: stage -> prompt tokens
_STAGE_TOKENS_OUT = "jf:llm:stage_tokens_out"       # hash: stage -> completion tokens


def _milli(usd: float) -> int:
    return int(round(usd * 1000))


class BudgetExceeded(RuntimeError):
    """Raised when a request would exceed the configured spend cap."""


def usage_cost_usd(usage: object, s: Settings | None = None) -> float:
    """USD for one LLM call from its OpenAI-style ``usage`` object."""
    s = s or get_settings()
    pin = int(getattr(usage, "prompt_tokens", 0) or 0)
    pout = int(getattr(usage, "completion_tokens", 0) or 0)
    return pin / 1000 * s.llm_price_in_per_1k + pout / 1000 * s.llm_price_out_per_1k


class Budget:
    """Global spend ledger backed by Redis."""

    def __init__(self, client: "redis.Redis | None" = None, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.r = client or redis.Redis.from_url(self.s.redis_url)

    # -- introspection ------------------------------------------------------
    def spent_usd(self) -> float:
        return int(self.r.get(_SPENT) or 0) / 1000

    def reserved_usd(self) -> float:
        return int(self.r.get(_RESERVED) or 0) / 1000

    def remaining_usd(self) -> float:
        """Headroom after both actual spend and outstanding reservations."""
        return max(0.0, self.s.llm_max_cost_usd - self.spent_usd() - self.reserved_usd())

    # -- pre-flight gate ----------------------------------------------------
    def reserve(self, estimate_usd: float | None = None) -> float:
        """Atomically hold ``estimate_usd`` (default: worst-case run estimate) against
        the pool. Returns the amount reserved. Raises ``BudgetExceeded`` if there isn't
        enough headroom — caller must NOT proceed. Always pair with ``release``."""
        est = self.s.run_cost_estimate_usd() if estimate_usd is None else estimate_usd
        est_m = _milli(est)
        cap_m = _milli(self.s.llm_max_cost_usd)
        # Atomic check-and-increment so concurrent runs can't both pass.
        new_reserved = self.r.incrby(_RESERVED, est_m)
        spent_m = int(self.r.get(_SPENT) or 0)
        if spent_m + new_reserved > cap_m:
            self.r.decrby(_RESERVED, est_m)  # roll back the hold
            raise BudgetExceeded(
                f"insufficient LLM budget: need ${est:.4f}, "
                f"only ${self.remaining_usd():.4f} of ${self.s.llm_max_cost_usd:.2f} left"
            )
        log.info("budget: reserved $%.4f (remaining $%.4f)", est, self.remaining_usd())
        return est

    def release(self, estimate_usd: float) -> None:
        """Release a previously-held reservation (after recording actual spend)."""
        if estimate_usd <= 0:
            return
        # Don't let the counter go negative under races.
        if self.r.decrby(_RESERVED, _milli(estimate_usd)) < 0:
            self.r.set(_RESERVED, 0)

    # -- accounting ---------------------------------------------------------
    def record_usage(self, usage: object) -> float:
        """Add one call's actual USD to the spent ledger. Returns the call's cost."""
        cost = usage_cost_usd(usage, self.s)
        self.r.incrby(_SPENT, _milli(cost))
        return cost

    def check_can_spend(self, est_usd: float = 0.0) -> None:
        """Cheap hard-cap check used mid-run (inside the LLM client). Raises if the
        pool is already exhausted; ``est_usd`` adds headroom for the upcoming call."""
        if self.spent_usd() + est_usd >= self.s.llm_max_cost_usd:
            raise BudgetExceeded(
                f"LLM spend cap reached (${self.spent_usd():.4f} / ${self.s.llm_max_cost_usd:.2f})"
            )


class BudgetGuard:
    """Adapter passed into ``LLMClient.complete_structured`` as the two hooks.

    ``guard.pre`` is the pre-request hook (hard-cap check); ``guard.post`` records the
    call's usage to the ledger and into the per-run accumulator for DB persistence.
    """

    def __init__(self, budget: Budget):
        self.budget = budget
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_usd = 0.0

    def pre(self) -> None:
        self.budget.check_can_spend()

    def post(self, usage: object) -> None:
        self.tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)
        self.cost_usd += self.budget.record_usage(usage)


class StageUsageLedger:
    """Per-stage LLM token/cost tallies in Redis (``HINCRBY``), surfaced by the API
    collector as ``yapper_llm_stage_*``. Purely for attribution metrics — separate from
    the authoritative spend ledger (``_SPENT``) so a failure here never affects budget
    enforcement or a run. Cumulative + Redis-persisted, so it reads like a counter for
    ``rate()`` in Grafana. Like :class:`Budget`, the Redis client is injectable for tests."""

    def __init__(self, client: "redis.Redis | None" = None, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.r = client or redis.Redis.from_url(self.s.redis_url)

    def record(self, stage: str, usage: object) -> None:
        """Add one call's tokens + USD to the given stage's tallies."""
        stage = stage or "unknown"
        pin = int(getattr(usage, "prompt_tokens", 0) or 0)
        pout = int(getattr(usage, "completion_tokens", 0) or 0)
        cost_milli = _milli(usage_cost_usd(usage, self.s))
        if pin:
            self.r.hincrby(_STAGE_TOKENS_IN, stage, pin)
        if pout:
            self.r.hincrby(_STAGE_TOKENS_OUT, stage, pout)
        if cost_milli:
            self.r.hincrby(_STAGE_COST_MILLI, stage, cost_milli)

    def snapshot(self) -> dict[str, dict[str, float]]:
        """``{stage: {"cost_usd", "tokens_in", "tokens_out"}}`` across all recorded stages."""
        def _h(key: str) -> dict[str, int]:
            raw = self.r.hgetall(key) or {}
            out: dict[str, int] = {}
            for k, v in raw.items():
                ks = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
                out[ks] = int(v)
            return out

        cost, tin, tout = _h(_STAGE_COST_MILLI), _h(_STAGE_TOKENS_IN), _h(_STAGE_TOKENS_OUT)
        return {
            s: {
                "cost_usd": cost.get(s, 0) / 1000,
                "tokens_in": tin.get(s, 0),
                "tokens_out": tout.get(s, 0),
            }
            for s in (set(cost) | set(tin) | set(tout))
        }
