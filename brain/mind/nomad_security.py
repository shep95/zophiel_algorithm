"""
NOMAD CYBER SECURITY — Defensive Layer of the Zophiel/Aureon System
Python port of the NOMAD security stack from zophiel_engine/src/security/nomad.ts

Implements:
  - AuditLog         — HMAC-chained tamper-evident audit trail
  - RbacPolicy       — Role-based access control
  - ApiKeyRegistry   — SHA-256 hashed API key management
  - RateLimiter      — Connection + RPM rate limiting
  - DistributedRateLimiter — Per-client rate limiting
  - ReplayGuard      — Nonce + timestamp replay attack prevention
  - SSRFGuard        — SSRF/private-IP request blocking
  - VitalGuard       — Organism vitals — partial compromise = full lockdown
  - ClientAllowlist  — Explicit client allowlist enforcement
  - NomadSecurityStack — Full assembled stack

Doctrine: "All security organs must be vital simultaneously.
           Partial compromise = total shutdown."
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .immune_memory import ThreatMemory, InflammationController
from .organ_registry import SovereignOrganism, OrganState


# ── Types ─────────────────────────────────────────────────────────────────────

Role = Literal["viewer", "operator", "admin", "sovereign"]
_ROLE_RANK: dict[str, int] = {"viewer": 1, "operator": 2, "admin": 3, "sovereign": 4}

AuditEventType = Literal[
    "job_started","job_completed","job_failed","api_request","api_denied",
    "rate_limit_exceeded","replay_detected","client_rejected_allowlist",
    "ssrf_blocked","organism_lockdown","audit_chain_breach","query_received","query_answered",
]


# ── AuditLog — HMAC-chained tamper-evident log ───────────────────────────────

@dataclass
class AuditEvent:
    id: str
    ts: str
    type: str
    correlation_id: str | None
    peer: str | None
    detail: str | None
    prev_entry_id: str
    entry_mac: str


class AuditLog:
    """
    Append-only audit log where each entry is HMAC-signed and chains
    to the previous entry. Any tampering breaks the chain.
    """

    def __init__(self, log_dir: str | None = None, chain_key_hex: str | None = None):
        self._entries: list[AuditEvent] = []
        self._file_path: Path | None = None

        if log_dir:
            p = Path(log_dir)
            p.mkdir(parents=True, exist_ok=True)
            self._file_path = p / "zophiel-audit.jsonl"
            # Persist the HMAC chain key alongside the log. Without this, a fresh
            # random key on every restart would invalidate every previously
            # written entry, breaking verify_chain() and tripping a permanent
            # lockdown the moment the log is reloaded in non-dev mode.
            self._chain_key = self._resolve_persistent_key(p, chain_key_hex)
            self._load()
        else:
            # Ephemeral, in-memory log (e.g. dev / tests): a per-process key is fine.
            self._chain_key = bytes.fromhex(chain_key_hex) if chain_key_hex else secrets.token_bytes(32)

    @staticmethod
    def _resolve_persistent_key(log_path: Path, chain_key_hex: str | None) -> bytes:
        # Explicit key always wins and is written so future loads stay consistent.
        key_file = log_path / "chain.key"
        if chain_key_hex:
            key = bytes.fromhex(chain_key_hex)
            key_file.write_text(chain_key_hex, encoding="utf-8")
            return key
        if key_file.exists():
            try:
                return bytes.fromhex(key_file.read_text(encoding="utf-8").strip())
            except Exception:
                pass  # corrupt key file — regenerate below
        key = secrets.token_bytes(32)
        key_file.write_text(key.hex(), encoding="utf-8")
        try:
            os.chmod(key_file, 0o600)  # restrict to owner where supported
        except OSError:
            pass
        return key

    def _sign(self, ev_id, ts, ev_type, prev_id, detail) -> str:
        prev = prev_id or "GENESIS"
        payload = f"{ev_id}|{ts}|{ev_type}|{prev}|{detail or ''}"
        return hmac.new(self._chain_key, payload.encode(), hashlib.sha256).hexdigest()

    def _load(self) -> None:
        if not self._file_path or not self._file_path.exists():
            return
        for line in self._file_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                self._entries.append(AuditEvent(**d))
            except Exception:
                pass

    def record(self, ev_type: str, correlation_id: str | None = None,
               peer: str | None = None, detail: str | None = None) -> AuditEvent:
        prev = self._entries[-1] if self._entries else None
        ev_id = f"{int(time.time()*1000)}-{secrets.token_hex(4)}"
        ts    = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())
        prev_id = prev.id if prev else ""
        mac   = self._sign(ev_id, ts, ev_type, prev_id, detail)
        event = AuditEvent(
            id=ev_id, ts=ts, type=ev_type,
            correlation_id=correlation_id, peer=peer,
            detail=detail, prev_entry_id=prev_id, entry_mac=mac,
        )
        self._entries.append(event)
        if self._file_path:
            with self._file_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(event.__dict__) + "\n")
        return event

    def verify_chain(self) -> tuple[bool, list[str]]:
        errors: list[str] = []
        prev_id = ""
        for ev in self._entries:
            expected = self._sign(ev.id, ev.ts, ev.type, ev.prev_entry_id, ev.detail)
            if not hmac.compare_digest(ev.entry_mac, expected):
                errors.append(f"Entry {ev.id}: HMAC mismatch")
            if ev.prev_entry_id != prev_id:
                errors.append(f"Entry {ev.id}: chain broken")
            prev_id = ev.id
        return len(errors) == 0, errors

    def query(self, limit: int = 100) -> list[AuditEvent]:
        return self._entries[-limit:]

    def fingerprint(self) -> str:
        return self._chain_key[:8].hex()


# ── RbacPolicy ───────────────────────────────────────────────────────────────

class RbacPolicy:
    _ROUTES: dict[str, Role] = {
        "GET /health":               "viewer",
        "GET /organism/vitals":      "viewer",
        "GET /v1/engines":           "viewer",
        "GET /v1/jobs":              "operator",
        "GET /v1/jobs/:id":          "operator",
        "GET /v1/jobs/:id/pages":    "operator",
        "POST /v1/jobs":             "operator",
        "POST /v1/lookup":           "operator",
        "POST /v1/query":            "operator",
        "GET /v1/audit":             "admin",
        "DELETE /v1/jobs/:id":       "admin",
        "POST /v1/admin/keys":       "sovereign",
    }

    def authorize(self, principal: dict | None, method: str, path: str) -> bool:
        if not principal:
            return False
        required = self._match_route(method, path)
        need = _ROLE_RANK[required]
        roles = principal.get("roles", [])
        return any(_ROLE_RANK.get(r, 0) >= need for r in roles)

    def _match_route(self, method: str, path: str) -> Role:
        key = f"{method.upper()} {path}"
        if key in self._ROUTES:
            return self._ROUTES[key]
        for pattern, role in self._ROUTES.items():
            if ":id" in pattern:
                base = pattern.split(":id")[0].rstrip("/")
                if path.startswith(base):
                    return role
        return "admin"


# ── ApiKeyRegistry ───────────────────────────────────────────────────────────

class ApiKeyRegistry:
    def __init__(self, api_key_entries: list[str], require_auth: bool = False):
        self.require_auth = require_auth and bool(api_key_entries)
        self._keys: dict[str, dict] = {}
        for entry in api_key_entries:
            parts = entry.split(":", 1)
            if len(parts) == 2:
                h, role = parts
                self._keys[h.strip()] = {"subject": f"key:{role}", "roles": [role.strip().lower()]}

    @staticmethod
    def hash_key(raw: str) -> str:
        return hashlib.sha256(raw.encode()).hexdigest()

    def verify_token(self, token: str) -> dict | None:
        h = self.hash_key(token)
        for key_hash, principal in self._keys.items():
            if hmac.compare_digest(key_hash, h):
                return principal
        return None

    @staticmethod
    def generate_key(role: Role = "operator") -> tuple[str, str]:
        raw = secrets.token_urlsafe(24)
        config_entry = f"{ApiKeyRegistry.hash_key(raw)}:{role}"
        return raw, config_entry


# ── RateLimiter ───────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, max_connections: int = 50, max_rpm: int = 120):
        self._max_conn = max_connections
        self._max_rpm  = max_rpm
        self._active   = 0
        self._timestamps: list[float] = []

    def try_acquire(self) -> bool:
        now = time.time()
        self._timestamps = [t for t in self._timestamps if t >= now - 60]
        if len(self._timestamps) >= self._max_rpm or self._active >= self._max_conn:
            return False
        self._active += 1
        self._timestamps.append(now)
        return True

    def release(self) -> None:
        self._active = max(0, self._active - 1)


class DistributedRateLimiter:
    def __init__(self, max_per_minute: int = 60):
        self._max = max_per_minute
        self._buckets: dict[str, list[float]] = {}

    def try_acquire(self, client_id: str, limit_override: int | None = None) -> bool:
        now = time.time()
        limit = self._max if limit_override is None else max(1, limit_override)
        hits = [t for t in self._buckets.get(client_id, []) if t >= now - 60]
        if len(hits) >= limit:
            return False
        hits.append(now)
        self._buckets[client_id] = hits
        return True


# ── ReplayGuard ───────────────────────────────────────────────────────────────

class ReplayGuard:
    def __init__(self, max_clock_skew_ms: int = 60000,
                 nonce_ttl_ms: int = 120000, max_entries: int = 10000):
        self._skew  = max_clock_skew_ms
        self._ttl   = nonce_ttl_ms
        self._max   = max_entries
        self._seen: dict[str, float] = {}

    def validate(self, nonce: str, timestamp_ms: int, correlation_id: str) -> None:
        self._purge()
        now_ms = int(time.time() * 1000)
        if timestamp_ms <= 0 or abs(now_ms - timestamp_ms) > self._skew:
            raise ValueError("Message timestamp outside allowed clock skew window.")
        key = f"{correlation_id}:{nonce}"
        if key in self._seen:
            raise ValueError("Replay detected: duplicate nonce.")
        self._seen[key] = now_ms + self._ttl
        if len(self._seen) > self._max:
            oldest = next(iter(self._seen))
            del self._seen[oldest]

    def _purge(self) -> None:
        now_ms = int(time.time() * 1000)
        self._seen = {k: v for k, v in self._seen.items() if v > now_ms}


# ── SSRFGuard ─────────────────────────────────────────────────────────────────

_BLOCKED_HOSTS = frozenset(["localhost", "127.0.0.1", "::1", "0.0.0.0"])
_BLOCKED_TLDS  = frozenset([".local", ".internal", ".corp", ".lan"])

class SSRFGuard:
    def __init__(self, block_private: bool = True, block_link_local: bool = True,
                 allowed_schemes: list[str] | None = None):
        self._block_private    = block_private
        self._block_link_local = block_link_local
        self._schemes          = set(allowed_schemes or ["http", "https"])

    def validate_url(self, url: str) -> tuple[bool, str | None]:
        from urllib.parse import urlparse
        try:
            p = urlparse(url.strip())
        except Exception:
            return False, "invalid_url"
        scheme = p.scheme.lower()
        if scheme not in self._schemes:
            return False, "scheme_not_allowed"
        host = (p.hostname or "").lower()
        if host in _BLOCKED_HOSTS:
            return False, "blocked_host"
        for tld in _BLOCKED_TLDS:
            if host.endswith(tld):
                return False, "blocked_tld"

        # IP-literal check. ipaddress understands decimal ("2130706433"),
        # hex ("0x7f000001"), octal, IPv6 ("::1", "fc00::"), and IPv4-mapped
        # forms — closing the encoded-IP SSRF bypasses a regex misses.
        stripped = host.strip("[]")  # IPv6 literals arrive bracketed
        literal_ip = self._as_ip(stripped)
        if literal_ip is not None:
            reason = self._classify_ip(literal_ip)
            if reason:
                return False, reason
            return True, None

        # Hostname: resolve and screen every A/AAAA record to defeat DNS
        # rebinding / names that point at internal ranges.
        try:
            import socket
            infos = socket.getaddrinfo(stripped, None)
        except Exception:
            return True, None  # unresolvable here; caller's fetch will fail safely
        for info in infos:
            ip = self._as_ip(info[4][0])
            if ip is not None:
                reason = self._classify_ip(ip)
                if reason:
                    return False, reason
        return True, None

    @staticmethod
    def _as_ip(value: str) -> "ipaddress._BaseAddress | None":
        try:
            return ipaddress.ip_address(value)
        except ValueError:
            # Accept legacy integer/hex encodings of IPv4 (e.g. 2130706433).
            try:
                if value.lower().startswith("0x"):
                    return ipaddress.ip_address(int(value, 16))
                if value.isdigit():
                    return ipaddress.ip_address(int(value))
            except (ValueError, ipaddress.AddressValueError):
                return None
            return None

    def _classify_ip(self, ip: "ipaddress._BaseAddress") -> str | None:
        if ip.is_loopback:
            return "loopback_ip"
        if ip.is_link_local:
            return "link_local_ip"
        if self._block_private and (ip.is_private or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return "private_ip"
        return None

    def validate_many(self, urls: list[str]) -> list[tuple[str, str]]:
        blocked = []
        for url in urls:
            ok, reason = self.validate_url(url)
            if not ok:
                blocked.append((url, reason or "blocked"))
        return blocked


# ── VitalGuard ───────────────────────────────────────────────────────────────

class VitalGuard:
    """
    Organism vitals monitor.
    Doctrine: ALL security organs must be vital simultaneously.
    One failure = total lockdown.
    """

    def __init__(self, audit: AuditLog, dev_mode: bool = False):
        self._audit      = audit
        self._dev_mode   = dev_mode
        self._pulse      = 1
        self._lockdown_reason: str | None = None

    def pulse_check(self) -> None:
        self._pulse += 1
        if self._dev_mode:
            self._lockdown_reason = None
            return
        valid, errors = self._audit.verify_chain()
        self._lockdown_reason = None if valid else (errors[0] if errors else "audit_chain_invalid")
        if self._lockdown_reason:
            self._audit.record("audit_chain_breach", detail=self._lockdown_reason)

    def is_vital(self) -> bool:
        return self._dev_mode or self._lockdown_reason is None

    def require_vital(self, operation: str) -> None:
        if not self.is_vital():
            raise PermissionError(f"ORGANISM_LOCKDOWN: {operation} blocked — {self._lockdown_reason}")

    def vitals_report(self) -> dict[str, Any]:
        valid, _ = self._audit.verify_chain()
        return {
            "vital": self.is_vital(),
            "pulse_generation": self._pulse,
            "organism_fingerprint": self._audit.fingerprint(),
            "lockdown_reason": self._lockdown_reason,
            "doctrine": "All security organs must be vital simultaneously. Partial compromise = total shutdown.",
            "organs": [
                {"id": "audit_immune",  "name": "Audit Immune System", "state": "vital" if valid else "critical"},
                {"id": "gateway_skin",  "name": "Gateway Skin",        "state": "vital" if self.is_vital() else "critical"},
                {"id": "ssrf_lungs",    "name": "SSRF Lungs",          "state": "vital"},
                {"id": "replay_shield", "name": "Replay Shield",       "state": "vital"},
                {"id": "rate_membrane", "name": "Rate Membrane",       "state": "vital"},
            ],
        }


# ── ClientAllowlist ───────────────────────────────────────────────────────────

class ClientAllowlist:
    def __init__(self, allowed: list[str], require: bool = False):
        self._allowed = set(s.strip() for s in allowed if s.strip())
        self._require = require

    def is_allowed(self, client_id: str) -> bool:
        if self._require and not self._allowed:
            return False
        if not self._allowed:
            return True
        return client_id in self._allowed


# ── Full NOMAD Security Stack ─────────────────────────────────────────────────

@dataclass
class NomadSecurityStack:
    audit:       AuditLog
    rbac:        RbacPolicy
    auth:        ApiKeyRegistry
    allowlist:   ClientAllowlist
    rate_limiter: RateLimiter
    distributed: DistributedRateLimiter
    replay_guard: ReplayGuard
    vital_guard: VitalGuard
    ssrf_guard:  SSRFGuard
    threat_memory: ThreatMemory
    inflammation:  InflammationController
    organism:      SovereignOrganism
    memory_path:   str | None = None


def build_security_stack(
    log_dir: str | None = None,
    api_key_entries: list[str] | None = None,
    require_auth: bool = False,
    client_allowlist: list[str] | None = None,
    dev_mode: bool = True,
    chain_key_hex: str | None = None,
    max_connections: int = 50,
    max_rpm: int = 120,
    max_rpm_per_client: int = 60,
) -> NomadSecurityStack:
    audit       = AuditLog(log_dir=log_dir, chain_key_hex=chain_key_hex)
    vital       = VitalGuard(audit, dev_mode=dev_mode)
    vital.pulse_check()

    threat_memory = ThreatMemory()
    inflammation  = InflammationController()

    # The 17-organ body, with live probes for organs we can actually check.
    def _audit_probe() -> tuple[OrganState, str]:
        ok, errs = audit.verify_chain()
        return ("vital", "chain intact") if ok else ("critical", errs[0] if errs else "chain broken")

    organism = SovereignOrganism(
        probes={
            "audit_immune": _audit_probe,
            "antibody_memory": lambda: ("vital", f"{threat_memory.stats()['quarantined']} quarantined"),
            "inflammation": lambda: (
                ("vital" if inflammation.level() == 0 else "vital"),
                f"level {inflammation.level():.2f}"),
        },
        dev_mode=dev_mode,
    )
    organism.pulse()

    # Adaptive immunity persistence: restore antibody memory from prior runs so
    # immunity survives restarts, then vaccinate from a known-bad IOC feed.
    memory_path = os.path.join(log_dir, "antibody_memory.json") if log_dir else None
    if memory_path:
        restored = threat_memory.load(memory_path)
        if restored:
            audit.record("job_started", detail=f"antibody memory restored: {restored} cells")
    ioc_file = os.environ.get("ZOPHIEL_IOC_FILE")
    if ioc_file and os.path.exists(ioc_file):
        try:
            with open(ioc_file, encoding="utf-8") as f:
                iocs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
            seeded = threat_memory.vaccinate(iocs)
            audit.record("job_started", detail=f"vaccinated against {seeded} IOCs")
        except Exception as e:
            audit.record("job_failed", detail=f"IOC vaccination failed: {e}")

    stack = NomadSecurityStack(
        audit        = audit,
        rbac         = RbacPolicy(),
        auth         = ApiKeyRegistry(api_key_entries or [], require_auth),
        allowlist    = ClientAllowlist(client_allowlist or []),
        rate_limiter = RateLimiter(max_connections, max_rpm),
        distributed  = DistributedRateLimiter(max_rpm_per_client),
        replay_guard = ReplayGuard(),
        vital_guard  = vital,
        ssrf_guard   = SSRFGuard(),
        threat_memory = threat_memory,
        inflammation  = inflammation,
        organism      = organism,
        memory_path   = memory_path,
    )
    audit.record("job_started", detail="NOMAD security stack initialised")
    return stack


def validate_request(stack: NomadSecurityStack, method: str, path: str,
                     token: str | None = None, client_id: str = "anonymous",
                     correlation_id: str | None = None) -> tuple[bool, str]:
    """
    Full request validation pipeline.
    Returns (allowed: bool, reason: str).
    """
    def _reject(reason: str, event: str, *, peer: str | None = None,
                detail: str | None = None) -> tuple[bool, str]:
        """Record an offense, build adaptive-immunity memory, raise inflammation."""
        stack.audit.record(event, correlation_id=correlation_id,
                           peer=peer if peer is not None else client_id, detail=detail)
        stack.threat_memory.record_offense(client_id, reason)
        stack.inflammation.record_denial()
        return False, reason

    # Vitals check
    if not stack.vital_guard.is_vital():
        stack.audit.record("organism_lockdown", correlation_id=correlation_id, detail=path)
        return False, "organism_lockdown"

    # Adaptive immunity: a client whose antibody titer is high is quarantined
    # before any other work — the organism remembers and isolates repeat threats.
    if stack.threat_memory.is_quarantined(client_id):
        stack.audit.record("client_rejected_allowlist", correlation_id=correlation_id,
                           peer=client_id, detail="quarantined")
        return False, "quarantined"

    # Client allowlist
    if not stack.allowlist.is_allowed(client_id):
        return _reject("client_not_in_allowlist", "client_rejected_allowlist", peer=client_id)

    # Rate limiting — tightened body-wide when the organism is inflamed (fever).
    effective = max(1, int(stack.distributed._max * stack.inflammation.rate_multiplier()))
    if not stack.distributed.try_acquire(client_id, limit_override=effective):
        return _reject("rate_limit_exceeded", "rate_limit_exceeded", peer=client_id)

    if not stack.rate_limiter.try_acquire():
        return _reject("global_rate_limit", "rate_limit_exceeded", detail="global")

    # Auth
    principal: dict | None = None
    if stack.auth.require_auth:
        if not token:
            stack.rate_limiter.release()
            return _reject("authentication_required", "api_denied", detail="no_token")
        principal = stack.auth.verify_token(token)
        if not principal:
            stack.rate_limiter.release()
            return _reject("invalid_token", "api_denied", detail="invalid_token")
    else:
        principal = {"subject": "anonymous", "roles": ["operator"]}

    # RBAC
    if not stack.rbac.authorize(principal, method, path):
        stack.rate_limiter.release()
        return _reject("permission_denied", "api_denied", detail=f"rbac_denied {method} {path}")

    # validate_request is a synchronous gate, not a long-lived connection.
    # Release the global connection slot we acquired above so _active does not
    # leak (otherwise the limiter permanently blocks after max_connections hits).
    # The per-minute RPM window (_timestamps) remains the real throttle.
    stack.rate_limiter.release()

    stack.audit.record("api_request", correlation_id=correlation_id,
                       peer=client_id, detail=f"{method} {path}")
    return True, "ok"


# ── Singleton stack ───────────────────────────────────────────────────────────
_AUDIT_DIR = os.environ.get("ZOPHIEL_AUDIT_DIR")
_DEV_MODE  = os.environ.get("ZOPHIEL_DEV_MODE", "true").lower() in ("true","1","yes")

_stack: NomadSecurityStack | None = None

def get_stack() -> NomadSecurityStack:
    global _stack
    if _stack is None:
        _stack = build_security_stack(log_dir=_AUDIT_DIR, dev_mode=_DEV_MODE)
    return _stack

def ssrf_check(url: str) -> tuple[bool, str | None]:
    """Quick SSRF check on a single URL."""
    return get_stack().ssrf_guard.validate_url(url)

def vitals() -> dict:
    stack = get_stack()
    stack.organism.pulse()                       # refresh the 17-organ body
    report = stack.organism.report()
    report["adaptive_immunity"] = stack.threat_memory.stats()
    report["inflammation"] = stack.inflammation.stats()
    return report


def scan_and_record(client_id: str, payload, correlation_id: str | None = None) -> list[str]:
    """
    Toll-like-receptor scan of a request payload (innate pattern recognition).
    Any injection-shaped content becomes an antigen the adaptive system
    remembers. Returns the list of attack labels detected (empty = clean).
    """
    from .pattern_immunity import scan_payload, worst_offense
    stack = get_stack()
    hits = scan_payload(payload)
    if not hits:
        return []
    offense = worst_offense(hits) or "tampering_injection"
    stack.threat_memory.record_offense(client_id, offense)
    stack.inflammation.record_denial()
    stack.audit.record("api_denied", correlation_id=correlation_id, peer=client_id,
                       detail="pattern_immunity: " + ",".join(h.label for h in hits))
    return [h.label for h in hits]


def persist_memory() -> None:
    """Flush antibody memory to disk if a persistence path is configured."""
    stack = get_stack()
    if stack.memory_path:
        try:
            stack.threat_memory.save(stack.memory_path)
        except Exception:
            pass


def is_quarantined(client_id: str) -> bool:
    return get_stack().threat_memory.is_quarantined(client_id)
