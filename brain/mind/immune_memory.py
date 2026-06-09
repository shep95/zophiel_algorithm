"""
immune_memory.py — Adaptive Immunity for the Sovereign Organism
================================================================
Where nomad_security.py provides *innate* immunity (always-on barriers:
rate limits, SSRF guard, body caps — the skin and mucous membranes), this
module provides *adaptive* immunity: the body's memory of past attackers.

Biological mapping
------------------
  Antigen        →  an offending request (replay, SSRF, rate abuse, bad auth)
  Antibody titer →  a per-client threat score that RISES on each offense and
                    DECAYS exponentially over time (immune memory fades)
  Quarantine     →  when titer crosses a threshold the client is isolated,
                    exactly as a high antibody response neutralises a pathogen,
                    and is released once the titer decays back down
  Inflammation   →  a *graded, system-wide* tightening of defences when many
                    offenses happen at once — a fever, not a hard lockdown
  Immunological  →  repeat offenders escalate FASTER each time (the secondary
   memory           response is stronger and quicker than the primary one)

Doctrine: the organism does not merely block one bad request — it *remembers*
the attacker, escalates against repeat offenders, and raises a body-wide fever
under sustained assault, then heals back to baseline when the threat passes.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable


# Severity weights — how strong an antigen each offense type is.
OFFENSE_WEIGHTS: dict[str, float] = {
    "invalid_token":            3.0,   # active credential probing
    "replay_detected":          4.0,   # captured-traffic replay
    "ssrf_blocked":             4.0,   # trying to pivot to internal targets
    "tampering_injection":      4.0,   # SQLi / XSS / template injection in body
    "client_not_in_allowlist":  2.0,
    "rate_limit_exceeded":      1.0,   # could be a noisy-but-legit client
    "global_rate_limit":        0.5,
    "permission_denied":        1.5,
    "authentication_required":  1.0,
    "_default":                 1.0,
}


@dataclass
class _Antibodies:
    titer: float = 0.0          # current antibody level
    last_update: float = 0.0    # epoch seconds of last decay calc
    exposures: int = 0          # lifetime offense count (drives memory boost)
    quarantined_until: float = 0.0


class ThreatMemory:
    """
    Adaptive-immunity store. Per-client antibody titers that rise on offense
    and decay with a configurable half-life. Crossing `quarantine_threshold`
    isolates the client until the titer decays below `release_threshold`.
    """

    def __init__(
        self,
        half_life_s: float = 300.0,          # antibody titer halves every 5 min
        quarantine_threshold: float = 8.0,
        release_threshold: float = 4.0,
        max_clients: int = 50_000,
        clock: Callable[[], float] = time.time,
    ):
        self._half_life = max(1.0, half_life_s)
        self._decay_k = math.log(2) / self._half_life
        self._quar_threshold = quarantine_threshold
        self._release_threshold = release_threshold
        self._max_clients = max_clients
        self._clock = clock
        self._cells: dict[str, _Antibodies] = {}

    def _current_titer(self, ab: _Antibodies, now: float) -> float:
        elapsed = max(0.0, now - ab.last_update)
        ab.titer *= math.exp(-self._decay_k * elapsed)
        ab.last_update = now
        if ab.titer < 1e-6:
            ab.titer = 0.0
        return ab.titer

    def record_offense(self, client_id: str, offense: str) -> float:
        """Register an antigen. Returns the client's new antibody titer."""
        now = self._clock()
        ab = self._cells.get(client_id)
        if ab is None:
            if len(self._cells) >= self._max_clients:
                self._evict(now)
            ab = _Antibodies(last_update=now)
            self._cells[client_id] = ab

        self._current_titer(ab, now)
        weight = OFFENSE_WEIGHTS.get(offense, OFFENSE_WEIGHTS["_default"])
        # Secondary-response boost: repeat offenders escalate faster (immune
        # memory). +15% per prior exposure, capped at 3x the primary response.
        memory_boost = min(3.0, 1.0 + 0.15 * ab.exposures)
        ab.titer += weight * memory_boost
        ab.exposures += 1

        if ab.titer >= self._quar_threshold:
            # Quarantine length scales with how far over threshold we are,
            # bounded so a single client cannot be locked out forever.
            over = ab.titer - self._quar_threshold
            ab.quarantined_until = now + min(3600.0, self._half_life * (1.0 + over))
        return ab.titer

    def is_quarantined(self, client_id: str) -> bool:
        ab = self._cells.get(client_id)
        if ab is None:
            return False
        now = self._clock()
        if ab.quarantined_until and now < ab.quarantined_until:
            return True
        # Past the quarantine window: only release once titer has truly decayed.
        if ab.quarantined_until and now >= ab.quarantined_until:
            if self._current_titer(ab, now) >= self._release_threshold:
                # still hot — extend a short cooldown
                ab.quarantined_until = now + self._half_life
                return True
            ab.quarantined_until = 0.0
        return False

    def titer(self, client_id: str) -> float:
        ab = self._cells.get(client_id)
        return 0.0 if ab is None else self._current_titer(ab, self._clock())

    def vaccinate(self, client_ids, titer: float | None = None) -> int:
        """
        Pre-seed immunity from a known-bad feed (IOC list). Like a vaccine, this
        primes the antibody response BEFORE first contact, so a known-hostile
        client is quarantined on its very first request. Returns count seeded.
        """
        now = self._clock()
        level = self._quar_threshold + 1.0 if titer is None else titer
        n = 0
        for cid in client_ids:
            cid = str(cid).strip()
            if not cid:
                continue
            ab = self._cells.get(cid) or _Antibodies(last_update=now)
            ab.titer = max(ab.titer, level)
            ab.last_update = now
            ab.exposures += 1  # treated as prior exposure -> faster re-escalation
            if ab.titer >= self._quar_threshold:
                over = ab.titer - self._quar_threshold
                ab.quarantined_until = now + min(3600.0, self._half_life * (1.0 + over))
            self._cells[cid] = ab
            n += 1
        return n

    def save(self, path) -> None:
        """Persist antibody memory so immunity survives a restart (no amnesia)."""
        import json
        from pathlib import Path
        now = self._clock()
        data = {
            "saved_at": now,
            "half_life_s": self._half_life,
            "cells": {
                cid: {
                    "titer": self._current_titer(ab, now),
                    "exposures": ab.exposures,
                    "quarantined_until": ab.quarantined_until,
                }
                for cid, ab in self._cells.items()
                if self._current_titer(ab, now) > 0 or ab.quarantined_until > now
            },
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(p)  # atomic

    def load(self, path) -> int:
        """Restore antibody memory written by save(). Returns cells restored."""
        import json
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return 0
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return 0
        now = self._clock()
        saved_at = float(data.get("saved_at", now))
        elapsed = max(0.0, now - saved_at)
        decay = math.exp(-self._decay_k * elapsed)  # age the memory while it was offline
        restored = 0
        for cid, c in data.get("cells", {}).items():
            titer = float(c.get("titer", 0.0)) * decay
            if titer < 1e-6 and float(c.get("quarantined_until", 0)) <= now:
                continue
            self._cells[cid] = _Antibodies(
                titer=titer, last_update=now,
                exposures=int(c.get("exposures", 0)),
                quarantined_until=float(c.get("quarantined_until", 0.0)),
            )
            restored += 1
        return restored

    def _evict(self, now: float) -> None:
        # Drop the coldest 10% of clients (lowest current titer) to bound memory.
        scored = sorted(
            self._cells.items(),
            key=lambda kv: self._current_titer(kv[1], now),
        )
        for cid, _ in scored[: max(1, len(scored) // 10)]:
            del self._cells[cid]

    def stats(self) -> dict:
        now = self._clock()
        active = sum(1 for ab in self._cells.values() if self._current_titer(ab, now) > 0)
        quarantined = sum(1 for cid in list(self._cells) if self.is_quarantined(cid))
        return {
            "tracked_clients": len(self._cells),
            "active_titers": active,
            "quarantined": quarantined,
            "half_life_s": self._half_life,
        }


class InflammationController:
    """
    Innate, graded, system-wide response. Counts offenses in a sliding window;
    a high rate raises an 'inflammation level' (0..1) that tightens defences
    body-wide — a fever — then subsides as the assault passes. This is distinct
    from the binary organism lockdown: the body runs hot, it does not flatline.
    """

    def __init__(
        self,
        window_s: float = 30.0,
        onset: int = 20,          # offenses/window where fever begins
        peak: int = 100,          # offenses/window = full inflammation
        clock: Callable[[], float] = time.time,
    ):
        self._window = window_s
        self._onset = onset
        self._peak = max(onset + 1, peak)
        self._clock = clock
        self._events: list[float] = []

    def record_denial(self) -> None:
        now = self._clock()
        self._events.append(now)
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        if self._events and self._events[0] < cutoff:
            self._events = [t for t in self._events if t >= cutoff]

    def level(self) -> float:
        now = self._clock()
        self._prune(now)
        n = len(self._events)
        if n <= self._onset:
            return 0.0
        return min(1.0, (n - self._onset) / (self._peak - self._onset))

    def rate_multiplier(self) -> float:
        """
        Allowed-throughput multiplier. At rest = 1.0 (full capacity); at peak
        inflammation = 0.25 (defences clamp to a quarter to shed load).
        """
        return 1.0 - 0.75 * self.level()

    def stats(self) -> dict:
        return {
            "inflammation_level": round(self.level(), 3),
            "rate_multiplier": round(self.rate_multiplier(), 3),
            "recent_offenses": len(self._events),
            "window_s": self._window,
        }
