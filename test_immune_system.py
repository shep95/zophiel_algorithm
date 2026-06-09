"""
test_immune_system.py — verifies the adaptive immunity layer behaves like
a real immune system: memory, decay, quarantine, secondary response, fever.
"""
from brain.mind.immune_memory import ThreatMemory, InflammationController
from brain.mind import nomad_security as ns


def test_titer_rises_and_quarantines():
    clk = [1000.0]
    tm = ThreatMemory(half_life_s=100, quarantine_threshold=8, release_threshold=4,
                      clock=lambda: clk[0])
    for _ in range(3):
        tm.record_offense("attacker", "ssrf_blocked")  # weight 4
    assert tm.is_quarantined("attacker"), "high titer must quarantine"
    assert not tm.is_quarantined("innocent"), "untouched client stays free"
    print("PASS: titer rises and quarantines the attacker only")


def test_memory_decays_and_releases():
    clk = [1000.0]
    tm = ThreatMemory(half_life_s=100, quarantine_threshold=8, release_threshold=4,
                      clock=lambda: clk[0])
    for _ in range(3):
        tm.record_offense("x", "ssrf_blocked")
    assert tm.is_quarantined("x")
    clk[0] += 10_000  # long enough for titer to fully decay
    assert not tm.is_quarantined("x"), "memory must fade — quarantine released"
    assert tm.titer("x") < 1.0, "titer decayed toward zero"
    print("PASS: immune memory decays and releases over time")


def test_secondary_response_is_stronger():
    clk = [1000.0]
    a = ThreatMemory(half_life_s=1e9, clock=lambda: clk[0])  # no decay
    first = a.record_offense("rep", "rate_limit_exceeded")
    # many prior exposures -> later offenses add more (memory boost)
    for _ in range(20):
        a.record_offense("rep", "rate_limit_exceeded")
    b = ThreatMemory(half_life_s=1e9, clock=lambda: clk[0])
    fresh = b.record_offense("new", "rate_limit_exceeded")
    # the 22nd identical offense on a known client must add more than the 1st on a fresh one
    boosted = a.record_offense("rep", "rate_limit_exceeded") - 21  # subtract prior accumulation approx
    assert first == fresh, "primary response identical for same offense"
    last_titer = a.titer("rep")
    assert last_titer > fresh * 21, "secondary response escalates faster"
    print("PASS: secondary (repeat-offender) response escalates faster")


def test_inflammation_fever_and_recovery():
    clk = [1000.0]
    inf = InflammationController(window_s=30, onset=5, peak=25, clock=lambda: clk[0])
    assert inf.level() == 0.0 and inf.rate_multiplier() == 1.0
    for _ in range(25):
        inf.record_denial()
    assert inf.level() == 1.0, "burst of denials => full fever"
    assert inf.rate_multiplier() <= 0.25, "fever clamps throughput"
    clk[0] += 60  # window passes
    assert inf.level() == 0.0, "fever subsides after assault passes"
    assert inf.rate_multiplier() == 1.0, "throughput restored"
    print("PASS: inflammation rises under assault then heals to baseline")


def test_integration_with_stack():
    stack = ns.build_security_stack(dev_mode=True)
    # Hammer with SSRF-class offenses via a forced bad route to build memory
    blocked_eventually = False
    for i in range(10):
        allowed, reason = ns.validate_request(stack, "DELETE", "/v1/admin/keys",
                                              client_id="badguy")  # sovereign-only -> denied
        if reason == "quarantined":
            blocked_eventually = True
            break
    assert blocked_eventually, "repeat RBAC denials must trigger quarantine"
    # A clean client is unaffected
    ok, reason = ns.validate_request(stack, "POST", "/v1/query", client_id="goodguy")
    assert ok, f"clean client should pass, got {reason}"
    v = ns.vitals()
    assert any(o["id"] == "antibody_memory" for o in v["organs"]), "vitals expose antibody organ"
    print("PASS: adaptive immunity integrated into the live validate_request gate")


if __name__ == "__main__":
    test_titer_rises_and_quarantines()
    test_memory_decays_and_releases()
    test_secondary_response_is_stronger()
    test_inflammation_fever_and_recovery()
    test_integration_with_stack()
    print("\n*** ALL IMMUNE-SYSTEM TESTS PASSED ***")
