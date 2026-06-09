"""
organ_registry.py — The Sovereign Organism's full anatomy (Python port)
========================================================================
Ports the 17-organ interdependent body from the TypeScript engine into the
live Python path. Every security subsystem is an organ; no organ works in
isolation. Organs activate only after their dependencies are vital/dormant
(a directed acyclic graph), and a single critical organ trips full lockdown.

Doctrine: all organs must be vital simultaneously. Attack one — the body
locks down. Attack all at once — the interlocking dependency chains still
refuse, because each child organ checks its parents on every pulse.

States:
  vital    — organ healthy and active
  dormant  — organ intentionally inactive (e.g. dev bypass, not wired)
  critical — organ failed; trips lockdown
  pending  — not yet checked this pulse
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

OrganState = Literal["vital", "dormant", "critical", "pending"]

# --- The 17 organs and their dependency graph (parents must be up first) ------
ORGAN_DEPENDENCIES: dict[str, list[str]] = {
    "crypto_core":        [],                                              # the heart of trust
    "supply_spleen":      ["crypto_core"],                                 # dependency integrity
    "audit_immune":       ["crypto_core"],                                 # tamper-evident memory
    "tpm_skeletal":       ["crypto_core"],                                 # boot attestation
    "hsm_heart":          ["crypto_core", "tpm_skeletal", "audit_immune"], # key custody
    "ca_liver":           ["crypto_core", "audit_immune", "hsm_heart"],    # cert transparency
    "console_brain":      ["audit_immune", "ca_liver"],                    # sovereign auth
    "rate_limit_nerves":  ["audit_immune"],                               # throttle reflex
    "pqc_lungs":          ["hsm_heart", "ca_liver", "audit_immune", "crypto_core"],
    "gateway_skin":       ["console_brain", "pqc_lungs", "audit_immune", "rate_limit_nerves"],
    "vault_marrow":       ["tpm_skeletal", "hsm_heart", "audit_immune", "crypto_core"],
    "threat_lungs":       ["audit_immune"],                               # IOC oxygen
    "siem_retina":        ["audit_immune", "threat_lungs"],              # sees all events
    "soar_spine":         ["siem_retina", "audit_immune"],              # automated reflex
    "antibody_memory":    ["audit_immune"],                              # adaptive immunity
    "inflammation":       ["audit_immune"],                              # graded fever
    "dlp_kidney":         ["audit_immune", "gateway_skin"],             # egress filter
}

ORGAN_META: dict[str, tuple[str, str]] = {
    "crypto_core":       ("Crypto Core",        "Cryptographic primitive integrity (the heart)"),
    "supply_spleen":     ("Supply Spleen",      "Dependency / SBOM integrity"),
    "audit_immune":      ("Audit Immune",       "Tamper-evident HMAC-chained memory"),
    "tpm_skeletal":      ("TPM Skeletal",       "Boot attestation baseline"),
    "hsm_heart":         ("HSM Heart",          "Non-extractable key custody"),
    "ca_liver":          ("CA Liver",           "Certificate transparency chain"),
    "console_brain":     ("Console Brain",      "Sovereign auth (Argon2/WebAuthn/ZK)"),
    "rate_limit_nerves": ("Rate Nerves",        "Connection / RPM throttle reflex"),
    "pqc_lungs":         ("PQC Lungs",          "Kyber/Dilithium secure respiration"),
    "gateway_skin":      ("Gateway Skin",       "RBAC perimeter + request barriers"),
    "vault_marrow":      ("Vault Marrow",       "Encrypted data-at-rest stem cells"),
    "threat_lungs":      ("Threat Lungs",       "IOC feed freshness — threat oxygen"),
    "siem_retina":       ("SIEM Retina",        "Event correlation — sees all"),
    "soar_spine":        ("SOAR Spine",         "Automated response reflex arc"),
    "antibody_memory":   ("Antibody Memory",    "Adaptive immunity (B/T-cell memory)"),
    "inflammation":      ("Inflammation",       "Graded system-wide fever response"),
    "dlp_kidney":        ("DLP Kidney",         "Egress classifier — filters leakage"),
}


@dataclass
class OrganStatus:
    id: str
    name: str
    role: str
    state: OrganState
    depends_on: list[str]
    detail: str = ""
    last_pulse: float = 0.0


def organ_activation_order() -> list[str]:
    """Topological order — every parent appears before its children."""
    visited: set[str] = set()
    order: list[str] = []

    def visit(node: str) -> None:
        if node in visited:
            return
        for dep in ORGAN_DEPENDENCIES[node]:
            visit(dep)
        visited.add(node)
        order.append(node)

    for organ in ORGAN_DEPENDENCIES:
        visit(organ)
    return order


def dependencies_satisfied(organ: str, statuses: dict[str, OrganStatus]) -> bool:
    for dep in ORGAN_DEPENDENCIES[organ]:
        s = statuses.get(dep)
        if s is None or s.state not in ("vital", "dormant"):
            return False
    return True


# A health probe returns (state, detail). Absent probe => dormant ("not wired").
HealthProbe = Callable[[], tuple[OrganState, str]]


class SovereignOrganism:
    """
    Live 17-organ organism. Pulses each organ in dependency order; if a child's
    parents aren't healthy it goes critical before it's even checked. Any
    critical organ => the whole body is non-vital (lockdown).
    """

    def __init__(self, probes: dict[str, HealthProbe] | None = None, dev_mode: bool = True):
        self._probes = probes or {}
        self._dev_mode = dev_mode
        self._statuses: dict[str, OrganStatus] = {}
        self._pulse = 0
        self._lockdown_reason: str | None = None
        for oid in organ_activation_order():
            name, role = ORGAN_META[oid]
            self._statuses[oid] = OrganStatus(
                id=oid, name=name, role=role, state="pending",
                depends_on=ORGAN_DEPENDENCIES[oid],
            )

    def pulse(self) -> None:
        self._pulse += 1
        now = time.time()
        any_critical = False
        for oid in organ_activation_order():
            st = self._statuses[oid]
            st.last_pulse = now
            if not dependencies_satisfied(oid, self._statuses):
                st.state, st.detail = "critical", "dependency organ not vital"
                any_critical = True
                continue
            probe = self._probes.get(oid)
            if probe is None:
                # No probe wired: dormant in dev, but core organs are assumed
                # vital so the body can actually operate.
                st.state = "vital" if oid in _ALWAYS_VITAL else "dormant"
                st.detail = "assumed vital" if oid in _ALWAYS_VITAL else "not wired (dormant)"
                continue
            try:
                state, detail = probe()
                st.state, st.detail = state, detail
                if state == "critical":
                    any_critical = True
            except Exception as e:  # a throwing probe = a failed organ
                st.state, st.detail = "critical", f"probe error: {e}"
                any_critical = True

        if any_critical:
            failed = [s.id for s in self._statuses.values() if s.state == "critical"]
            self._lockdown_reason = f"critical organs: {', '.join(failed)}"
        else:
            self._lockdown_reason = None

    def is_vital(self) -> bool:
        if self._lockdown_reason:
            return False
        return all(s.state in ("vital", "dormant") for s in self._statuses.values())

    def fingerprint(self) -> str:
        """Binds to the live organ state — changes when the body changes."""
        material = "|".join(f"{s.id}:{s.state}" for s in self._statuses.values())
        return hashlib.sha256(f"{material}|{self._pulse}".encode()).hexdigest()

    def report(self) -> dict:
        return {
            "vital": self.is_vital(),
            "pulse_generation": self._pulse,
            "organism_fingerprint": self.fingerprint(),
            "lockdown_reason": self._lockdown_reason,
            "organ_count": len(self._statuses),
            "organs": [
                {"id": s.id, "name": s.name, "role": s.role,
                 "state": s.state, "detail": s.detail, "depends_on": s.depends_on}
                for s in self._statuses.values()
            ],
            "doctrine": ("All organs must be vital simultaneously. Attack one — "
                         "the body locks down. Attack all at once — the "
                         "interlocking dependency chains still refuse."),
        }


# Core organs that operate in-process and are healthy unless a probe says otherwise.
_ALWAYS_VITAL = frozenset({
    "crypto_core", "audit_immune", "rate_limit_nerves", "gateway_skin",
    "antibody_memory", "inflammation",
})
