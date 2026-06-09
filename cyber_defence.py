"""
cyber_defence.py — SOLIA Cyber Defence Brain Module
=====================================================
Integrates Nomad Cyber Algorithm knowledge into the SOLIA/Aureon pipeline.

Capabilities:
  - Threat classification (STRIDE: Spoofing/Tampering/Repudiation/
    Information Disclosure/DoS/Elevation of Privilege)
  - Post-quantum cryptography awareness (Kyber1024, Dilithium5)
  - Nomad Sovereign Organism architecture knowledge
  - Chaos cipher and Imperial stack explanations
  - Incident response routing
  - Defence synthesis in Asher's voice

Used by: think() in aureon_test_runner.py and the main SOLIA orchestrator
"""

from __future__ import annotations
import re
from typing import Optional


# ── Nomad Knowledge Base ──────────────────────────────────────────────────────

NOMAD_FACTS = [
    # PQC
    "Kyber1024 (ML-KEM) is a post-quantum key encapsulation mechanism resistant to quantum Grover and Shor attacks.",
    "Dilithium5 is a lattice-based digital signature algorithm; NIST selected it as a post-quantum standard (FIPS 204).",
    "Classical RSA and ECDH are broken by Shor's algorithm on a sufficiently powerful quantum computer.",
    "HKDF (HMAC-based Key Derivation Function) derives session keys from Kyber-shared secrets with domain separation.",
    "AES-256-GCM provides authenticated encryption; the 256-bit key provides 128-bit post-quantum security.",
    "QS-CA (Quantum-Safe Certificate Authority) issues Dilithium-signed certificates and logs them to a CT log.",

    # Chaos Cipher
    "Chaos mode applies per-message random layer ordering, 16–272 byte CSPRNG padding, and 0–40ms timing jitter.",
    "Chaos fingerprint is an 8-byte HMAC tag on each message for tamper detection without revealing pattern.",
    "Per-message keys are derived from sequence number and timestamp — identical plaintext never produces identical ciphertext.",
    "Traffic analysis attacks are defeated by timing jitter and chaotic padding that eliminates wire-level patterns.",

    # Imperial Cipher Stack
    "Imperial cipher stack applies 7 historical cipher layers (Hieroglyph, Augustan, Scytale, etc.) before AES-GCM.",
    "Aureon occult veil applies a planetary epoch key derived from astronomical time — adds a non-repeating temporal layer.",
    "Layer shuffle in chaos mode derives the Imperial layer order from a per-message key — no predictable sequence.",

    # Sovereign Organism
    "Nomad Sovereign Organism has 11 interlocking organs: Crypto Core, Audit Immune, TPM Skeletal, HSM Heart, CA Liver, Console Brain, Rate Nerves, PQC Lungs, Gateway Skin, Vault Marrow, Supply Spleen.",
    "Partial compromise equals total lockdown: breach the audit chain OR lose TPM attestation OR HSM disconnect triggers system-wide lockdown.",
    "Organism pulses every 30 seconds re-verifying all 11 organs — attacker must breach all organs simultaneously.",
    "Vault encryption binds to the organism fingerprint — data sealed under one pulse cannot be read under another.",

    # Gateway / Auth
    "API Gateway enforces RBAC, body size limits, security headers, and session auth on every request.",
    "Console uses Argon2id password hashing, RFC 6238 TOTP MFA, and FIDO2/WebAuthn AAL3 authentication.",
    "Zero-knowledge sovereign proof layer prevents credential replay — proof of knowledge without revealing the credential.",
    "Session tickets are AES-GCM encrypted, HMAC-sealed, and single-use — replay attacks are impossible.",

    # Vault
    "DB field vault uses per-field AES-256-GCM with tenant-bound AAD — field values are opaque to the database layer.",
    "File vault uses AES-256-GCM at rest with object ID validation and owner authorization checks.",
    "HSM (Hardware Security Module) holds non-extractable private keys — keys never appear in process memory.",
    "TPM boot attestation verifies PCR values at startup — tampered OS or firmware blocks key access.",

    # STRIDE Threats
    "Spoofing threat: forged session tokens. Mitigation: timing-safe validation + audit chain + HMAC-sealed tickets.",
    "Tampering threat: audit log modification. Mitigation: chained HMAC entries — any edit breaks the chain.",
    "Information disclosure threat: heap dump exposing keys. Mitigation: HSM non-extractable keys + Worker thread sandbox.",
    "DoS threat: handshake flood. Mitigation: distributed Redis rate limiter + per-IP connection caps.",
    "Elevation of privilege threat: TOTP phishing. Mitigation: FIDO2/WebAuthn + ZK sovereign proof (AAL3).",
    "Repudiation threat: operator denying privileged actions. Mitigation: immutable append-only audit log + CT log.",

    # Supply Chain
    "SBOM (Software Bill of Materials) hash verification runs at startup — dependency tampering triggers lockdown.",
    "SAST/DAST/fuzz CI gates run on every commit — static analysis, dynamic testing, and random input fuzzing.",
    "Supply chain attack mitigation: allowlist of dependency hashes + Supply Spleen organ verification.",

    # Deployment
    "nginx WAF rules block SQLi, path traversal, and DDoS at the edge before traffic reaches the application.",
    "Kubernetes Helm chart deploys the full sovereign stack with resource limits and liveness probes.",
    "Sidecar mode creates per-connection PQC tunnels — each service connection gets an isolated session.",

    # Principles
    "Defence in depth: 11 independent security layers — defeating one layer does not compromise others.",
    "Zero-trust architecture: every request is authenticated and authorized regardless of network origin.",
    "Fail-closed design: when in doubt, deny. Lockdown is the default response to anomaly detection.",
    "Principle of least privilege: each organ has only the permissions required for its specific function.",
]

# ── STRIDE Classifier ─────────────────────────────────────────────────────────

STRIDE_PATTERNS = {
    "SPOOFING":                ["spoof","impersonat","fake","forged","identity","session hijack","token","credential"],
    "TAMPERING":               ["tamper","modif","alter","corrupt","inject","sql injection","xss","cross-site"],
    "REPUDIATION":             ["deni","repudiat","audit","log","trace","non-repudiation","accountability"],
    "INFORMATION_DISCLOSURE":  ["leak","exfiltrat","disclosure","sniff","intercept","eavesdrop","expose","dump","plaintext"],
    "DENIAL_OF_SERVICE":       ["ddos","dos","flood","exhaust","overload","rate limit","bandwidth","availability","crash"],
    "ELEVATION_OF_PRIVILEGE":  ["privilege","escalat","unauthoriz","bypass","admin","root","sudo","lateral movement"],
}

NOMAD_COMPONENT_PATTERNS = {
    "pqc":        ["kyber","dilithium","post.quantum","lattice","kem","ml-kem","pqc","quantum.safe","quantum.resistant"],
    "chaos":      ["chaos","unpredictabl","pattern","timing","jitter","padding","shuffle"],
    "imperial":   ["imperial","cipher","stack","layer","historical","hieroglyph","augustan","scytale","veil","occult"],
    "organism":   ["organism","organ","lockdown","pulse","vital","hsm","tpm","audit chain","sovereign"],
    "gateway":    ["gateway","rbac","rate limit","waf","firewall","perimeter","ddos","nginx","cloudflare"],
    "vault":      ["vault","encrypt.*rest","field.*encrypt","db.*encrypt","file.*vault","at.rest"],
    "auth":       ["webauthn","fido","totp","mfa","argon2","zero.knowledge","zk.proof","console"],
    "general":    ["cyber","security","attack","defence","defense","malware","phishing","ransomware","vulnerability","exploit","threat"],
}


class CyberAnalysis:
    def __init__(self, is_cyber, stride_category, nomad_components, threat_level, relevant_facts, defence_response):
        self.is_cyber = is_cyber
        self.stride_category = stride_category
        self.nomad_components = nomad_components
        self.threat_level = threat_level
        self.relevant_facts = relevant_facts
        self.defence_response = defence_response


def classify(query: str) -> CyberAnalysis:
    q = query.lower()

    # Check if cyber-related
    is_cyber = any(
        pat in q
        for pats in NOMAD_COMPONENT_PATTERNS.values()
        for pat in pats
    )

    # STRIDE classification
    stride_cat = None
    for cat, patterns in STRIDE_PATTERNS.items():
        if any(p in q for p in patterns):
            stride_cat = cat
            break

    # Nomad component matching
    matched_components = []
    for comp, patterns in NOMAD_COMPONENT_PATTERNS.items():
        for pat in patterns:
            if pat in q:
                matched_components.append(comp)
                break

    # Threat level heuristic
    if stride_cat in ("DENIAL_OF_SERVICE", "ELEVATION_OF_PRIVILEGE"):
        threat_level = "critical"
    elif stride_cat in ("SPOOFING", "INFORMATION_DISCLOSURE"):
        threat_level = "high"
    elif stride_cat in ("TAMPERING", "REPUDIATION"):
        threat_level = "medium"
    elif is_cyber:
        threat_level = "low"
    else:
        threat_level = "none"

    # Pull relevant facts
    relevant = _get_relevant_facts(q, matched_components, stride_cat)

    # Build defence response
    defence = _build_defence_response(stride_cat, matched_components, relevant) if is_cyber else None

    return CyberAnalysis(
        is_cyber=is_cyber,
        stride_category=stride_cat,
        nomad_components=matched_components,
        threat_level=threat_level,
        relevant_facts=relevant,
        defence_response=defence,
    )


def _get_relevant_facts(query: str, components: list[str], stride_cat: Optional[str]) -> list[str]:
    """Return the top facts most relevant to this query."""
    q_tokens = set(re.findall(r"[a-z]{3,}", query))
    scored: list[tuple[float, str]] = []
    for fact in NOMAD_FACTS:
        f_tokens = set(re.findall(r"[a-z]{3,}", fact.lower()))
        score = len(q_tokens & f_tokens)
        # Boost by component match
        for comp in components:
            comp_pats = NOMAD_COMPONENT_PATTERNS.get(comp, [])
            for pat in comp_pats:
                if pat in fact.lower():
                    score += 2
                    break
        # Boost by STRIDE match
        if stride_cat:
            stride_pats = STRIDE_PATTERNS.get(stride_cat, [])
            if any(p in fact.lower() for p in stride_pats):
                score += 3
        scored.append((score, fact))
    scored.sort(key=lambda x: -x[0])
    return [f for _, f in scored[:5] if _ > 0]


def _build_defence_response(
    stride_cat: Optional[str],
    components: list[str],
    facts: list[str],
) -> str:
    """Synthesize a defence response in Asher/SOLIA voice."""
    parts = []

    if stride_cat:
        stride_labels = {
            "SPOOFING":               "Spoofing",
            "TAMPERING":              "Tampering",
            "REPUDIATION":            "Repudiation",
            "INFORMATION_DISCLOSURE": "Information Disclosure",
            "DENIAL_OF_SERVICE":      "Denial of Service",
            "ELEVATION_OF_PRIVILEGE": "Elevation of Privilege",
        }
        parts.append(f"STRIDE classification: {stride_labels[stride_cat]}.")

    if "pqc" in components:
        parts.append("Post-quantum defence active: Kyber1024 KEM + Dilithium5 signatures resist Shor's algorithm.")
    if "organism" in components:
        parts.append("Sovereign Organism protocol: partial compromise triggers full lockdown across all 11 organs.")
    if "chaos" in components:
        parts.append("Chaos mode eliminates wire patterns — identical plaintext never produces the same ciphertext twice.")
    if "vault" in components:
        parts.append("Vault Marrow: AES-256-GCM field encryption with HSM-bound keys — non-extractable by design.")
    if "gateway" in components:
        parts.append("Gateway Skin: RBAC perimeter, Redis rate limits, WAF rules block edge-level attacks before they reach the core.")

    if facts:
        parts.append(facts[0])

    if not parts:
        parts.append("Defence in depth: no single breach compromises the system. Fail-closed on anomaly.")

    return " ".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

def get_cyber_facts() -> list[str]:
    """Return all Nomad facts for injection into the RAG corpus."""
    return NOMAD_FACTS


def analyse_threat(query: str) -> Optional[str]:
    """
    Entry point for the SOLIA think() pipeline.
    Returns a cyber-defence synthesis string, or None if not cyber-related.

    Only surfaces a defence overlay for GENUINE security questions — a single
    incidental keyword ("log", "token", "quantum", "pattern") is not enough.
    This prevents the STRIDE/Kyber overlay from polluting unrelated answers.
    """
    result = classify(query)
    if not result.is_cyber:
        return None
    # Require real security intent: a STRIDE category, a meaningful threat level,
    # or at least two distinct Nomad components matched. Otherwise stay quiet.
    strong = (
        result.stride_category is not None
        or result.threat_level in ("medium", "high", "critical")
        or len(set(result.nomad_components) - {"general"}) >= 2
    )
    if not strong:
        return None
    return result.defence_response


def full_analysis(query: str) -> dict:
    """Full structured analysis — used in tests and the audit pipeline."""
    result = classify(query)
    return {
        "is_cyber": result.is_cyber,
        "stride": result.stride_category,
        "components": result.nomad_components,
        "threat_level": result.threat_level,
        "facts_count": len(result.relevant_facts),
        "defence": result.defence_response,
    }
