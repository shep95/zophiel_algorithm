"""
pattern_immunity.py — Toll-Like Receptors (innate pattern recognition)
=======================================================================
The innate immune system recognises broad "patterns of pathogenicity"
(PAMPs) via toll-like receptors — it doesn't need to have seen the exact
microbe before, just its shape. This module is that organ: it signature-scans
request payloads for the structural shapes of common attacks (SQLi, XSS,
path traversal, command injection, SSRF, template injection) and reports
each hit as an antigen the adaptive system can then remember.

It is deliberately conservative — these are high-signal structural patterns,
not a full WAF — so legitimate prose rarely trips them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# (label, severity, compiled pattern). Severity maps onto immune_memory weights.
_SIGNATURES: list[tuple[str, str, re.Pattern]] = [
    # SQL injection
    ("sqli", "tampering", re.compile(r"(?i)\b(union\s+select|select\s+.*\s+from\s+|insert\s+into\s+|drop\s+table|or\s+1\s*=\s*1|';\s*--|/\*.*\*/)\b")),
    ("sqli", "tampering", re.compile(r"(?i)(\bor\b|\band\b)\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+")),
    # Cross-site scripting
    ("xss", "tampering", re.compile(r"(?i)(<script[\s>]|javascript:|onerror\s*=|onload\s*=|<img[^>]+src\s*=\s*['\"]?\s*x)")),
    # Path traversal
    ("path_traversal", "info_disclosure", re.compile(r"(\.\./|\.\.\\|%2e%2e[%2f%5c]|/etc/passwd|\\windows\\system32)")),
    # OS command injection
    ("cmd_injection", "elevation", re.compile(r"(?i)(;\s*(cat|ls|rm|wget|curl|nc|bash|sh|powershell)\b|\|\s*(cat|nc|bash|sh)\b|\$\(.*\)|`.*`)")),
    # Server-side template injection
    ("ssti", "elevation", re.compile(r"(\{\{.*\}\}|\$\{.*\}|<%=?.*%>)")),
    # SSRF / internal target probing
    ("ssrf", "info_disclosure", re.compile(r"(?i)(https?://(127\.|10\.|192\.168\.|169\.254\.|localhost|0x7f|metadata\.google|169\.254\.169\.254))")),
    # Null byte / control smuggling
    ("null_byte", "tampering", re.compile(r"(%00|\x00)")),
]

# Toll-like-receptor hit -> the offense type the adaptive system records.
_OFFENSE_FOR = {
    "sqli": "tampering_injection",
    "xss": "tampering_injection",
    "path_traversal": "ssrf_blocked",
    "cmd_injection": "invalid_token",     # treat as high-severity active attack
    "ssti": "invalid_token",
    "ssrf": "ssrf_blocked",
    "null_byte": "replay_detected",
}


@dataclass
class PatternHit:
    label: str
    category: str
    snippet: str


def scan(text: str, max_len: int = 20000) -> list[PatternHit]:
    """Return all toll-like-receptor matches in `text` (truncated for safety)."""
    if not text:
        return []
    sample = text[:max_len]
    hits: list[PatternHit] = []
    seen: set[str] = set()
    for label, category, pat in _SIGNATURES:
        m = pat.search(sample)
        if m and label not in seen:
            seen.add(label)
            hits.append(PatternHit(label=label, category=category,
                                   snippet=m.group(0)[:60]))
    return hits


def scan_payload(payload) -> list[PatternHit]:
    """Recursively flatten a JSON-ish payload to text and scan it."""
    def flatten(obj) -> str:
        if isinstance(obj, dict):
            return " ".join(f"{k} {flatten(v)}" for k, v in obj.items())
        if isinstance(obj, (list, tuple)):
            return " ".join(flatten(v) for v in obj)
        return str(obj)
    return scan(flatten(payload))


def worst_offense(hits: list[PatternHit]) -> str | None:
    """Map the most severe hit onto an adaptive-immunity offense type."""
    if not hits:
        return None
    # Order by severity proxy: active attacks first.
    priority = ["cmd_injection", "ssti", "ssrf", "sqli", "path_traversal", "xss", "null_byte"]
    for label in priority:
        for h in hits:
            if h.label == label:
                return _OFFENSE_FOR[label]
    return _OFFENSE_FOR[hits[0].label]
