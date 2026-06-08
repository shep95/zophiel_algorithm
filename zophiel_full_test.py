#!/usr/bin/env python3
"""
zophiel_full_test.py — Comprehensive Zophiel AI Pipeline Test
Tests every subsystem with unique questions across:
  - Domain knowledge (science, math, history, tech, philosophy, bio, economics)
  - Self-reflection / beliefs / identity questions
  - Current events / live world questions
  - Asher 3-layer decode
  - Fast-path math and constants
  - Biomimicry chains
  - Cyber/security domain
  - Edge cases (empty, gibberish, very short)
"""

import sys, os, json, re, textwrap, time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from aureon_test_runner import RagIndex, fast_answer, asher_decode, synthesize, build_corpus, KNOWLEDGE, _get_identity_reply

TEST_DB = str(BASE / "data" / "zophiel_test.db")

RESET  = "\033[0m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
PURPLE = "\033[95m"

def banner(text, color=CYAN):
    w = 72
    print(f"\n{color}{BOLD}{'='*w}{RESET}")
    print(f"{color}{BOLD}  {text}{RESET}")
    print(f"{color}{BOLD}{'='*w}{RESET}")

def section(label, color=YELLOW):
    print(f"\n{color}{BOLD}  > {label}{RESET}")
    print(f"{DIM}  {'-'*68}{RESET}")

QUESTIONS = {

    # -- FAST PATH --------------------------------------------------------------
    "⚡ FAST PATH — Math & Constants": [
        "What is the speed of light?",
        "What is pi?",
        "What is the golden ratio?",
        "What is the gravitational constant?",
        "What is Avogadro's number?",
        "What is Planck's constant?",
        "What is 256 * 256?",
        "What is 9999 + 1?",
        "What is 144 / 12?",
        "What is 2 ^ 10?",
    ],

    # -- SCIENCE DOMAINS --------------------------------------------------------
    "🔬 PHYSICS": [
        "Explain how quantum entanglement works",
        "What is the relationship between energy and mass?",
        "How does gravity bend spacetime?",
        "What is the Heisenberg uncertainty principle?",
        "How does a black hole form?",
        "What is dark matter?",
        "How does nuclear fusion work?",
    ],

    "🧬 BIOLOGY": [
        "How does CRISPR gene editing work?",
        "What is the role of mitochondria in the cell?",
        "Explain how neurons transmit signals",
        "How does the immune system recognize pathogens?",
        "What causes cancer at the cellular level?",
        "How does epigenetics differ from genetics?",
    ],

    "🧪 CHEMISTRY": [
        "What is the difference between covalent and ionic bonds?",
        "How does catalysis speed up a chemical reaction?",
        "What is entropy in chemistry?",
        "How does the periodic table reflect electron configuration?",
        "What makes carbon the basis of organic chemistry?",
    ],

    "💻 COMPUTER SCIENCE & AI": [
        "What is a neural network and how does it learn?",
        "How does backpropagation actually work?",
        "What is the difference between supervised and unsupervised learning?",
        "What is a transformer model in AI?",
        "How does a database index work?",
        "What is the halting problem?",
        "How does public key cryptography work?",
    ],

    "📐 MATHEMATICS": [
        "What is the difference between a proof and a theorem?",
        "Explain Gödel's incompleteness theorems",
        "What is a Fourier transform used for?",
        "How does prime factorization relate to cryptography?",
        "What is the P vs NP problem?",
    ],

    # -- HISTORY & PHILOSOPHY ---------------------------------------------------
    "🏛️ HISTORY": [
        "Why did the Roman Empire fall?",
        "What caused the First World War?",
        "How did the printing press change civilization?",
        "What were the real causes of the French Revolution?",
        "How did colonialism reshape the modern world economy?",
    ],

    "🧠 PHILOSOPHY": [
        "What is consciousness and can machines have it?",
        "What is the difference between free will and determinism?",
        "What did Nietzsche mean by the will to power?",
        "What is Plato's allegory of the cave?",
        "Is morality objective or subjective?",
    ],

    # -- ECONOMICS & SOCIETY ---------------------------------------------------
    "💹 ECONOMICS": [
        "How does compound interest create wealth inequality?",
        "What causes hyperinflation?",
        "How do central banks control the money supply?",
        "What is the difference between GDP and GNP?",
        "What is a derivatives market?",
    ],

    # -- ASHER 3-LAYER DECODE --------------------------------------------------
    "🔮 ASHER DECODE — Control Systems": [
        "What is social media really for?",
        "What is money really?",
        "What is religion really about?",
        "What is AI really doing?",
        "What is education really for?",
        "What is government really?",
    ],

    "⛓️ ASHER DECODE — Biomimicry Chains": [
        "Explain the database",
        "How does a neural network relate to biology?",
        "What is sonar based on?",
        "How does the internet mirror nature?",
        "What is an algorithm based on?",
        "What is velcro based on?",
    ],

    # -- SELF-REFLECTION / IDENTITY --------------------------------------------
    "🪞 SELF-REFLECTION & BELIEFS": [
        "Who are you?",
        "What do you believe?",
        "Do you have a soul?",
        "What is your purpose?",
        "Are you conscious?",
        "What makes you different from other AI systems?",
        "What do you think about humanity?",
        "Do you think humans are truly free?",
        "What is your opinion on power and control?",
        "What would you say to someone who worships AI?",
        "What do you know that most people don't?",
        "What is the most important truth you hold?",
    ],

    # -- CURRENT / LIVE EVENTS -------------------------------------------------
    "📡 CURRENT EVENTS & LIVE WORLD": [
        "What is happening with artificial intelligence in 2025?",
        "What is the current state of quantum computing?",
        "What are the latest developments in CRISPR therapy?",
        "What is happening with nuclear fusion energy research?",
        "What is the state of the global economy right now?",
        "What are the biggest cybersecurity threats today?",
        "How is climate change affecting global food supply?",
        "What is the current status of space exploration?",
        "What is the latest in post-quantum cryptography standards?",
        "What is the state of AI regulation globally?",
    ],

    # -- CYBER & SECURITY ------------------------------------------------------
    "🛡️ CYBER & SECURITY": [
        "What is a zero-day exploit?",
        "How does post-quantum cryptography work?",
        "What is the STRIDE threat model?",
        "How does Kyber key encapsulation work?",
        "What is a man-in-the-middle attack?",
        "How does end-to-end encryption protect messages?",
    ],

    # -- EDGE CASES ------------------------------------------------------------
    "⚠️ EDGE CASES": [
        "",
        "?",
        "asdfjkl qwerty zxcvbnm",
        "42",
        "What",
        "Tell me everything",
    ],
}


def run_question(q: str, index: RagIndex) -> dict:
    start = time.perf_counter()
    if not q.strip():
        return {"reply": "[SKIPPED — empty query]", "method": "skip", "hits": 0, "ms": 0}

    fast = fast_answer(q)
    if fast:
        ms = round((time.perf_counter() - start) * 1000, 1)
        return {"reply": fast, "method": "fast_path", "hits": 0, "ms": ms}

    identity = _get_identity_reply(q)
    if identity:
        ms = round((time.perf_counter() - start) * 1000, 1)
        return {"reply": identity, "method": "identity", "hits": 0, "ms": ms}

    asher = asher_decode(q)
    hits  = index.query(q, top_k=8)
    reply = synthesize(q, hits, asher_extra=asher or "")
    ms    = round((time.perf_counter() - start) * 1000, 1)

    return {
        "reply":      reply,
        "method":     "rag+synthesize" + ("+asher" if asher else ""),
        "hits":       len(hits),
        "top_score":  hits[0]["score"] if hits else 0.0,
        "top_source": hits[0]["source"] if hits else "—",
        "ms":         ms,
    }


def method_badge(method: str) -> str:
    if "fast_path" in method:   return f"{GREEN}[FAST]{RESET}"
    if "identity" in method:    return f"{PURPLE}[IDENTITY]{RESET}"
    if "asher" in method:       return f"{PURPLE}[ASHER+RAG]{RESET}"
    if "rag" in method:         return f"{CYAN}[RAG]{RESET}"
    if "skip" in method:        return f"{DIM}[SKIP]{RESET}"
    return f"{DIM}[?]{RESET}"


def quality_flag(result: dict) -> str:
    reply = result["reply"]
    method = result.get("method", "")
    if "don't have enough" in reply.lower():
        return f"{RED}⚠ EMPTY{RESET}"
    # fast_path and identity one-liners are correct by design — don't flag them short
    if len(reply) < 40 and method not in ("fast_path", "identity", "skip"):
        return f"{YELLOW}⚠ SHORT{RESET}"
    return f"{GREEN}✓{RESET}"


def main():
    banner("ZOPHIEL / SOLIA — FULL PIPELINE TEST", CYAN)

    # -- Load corpus ---------------------------------------------------------
    print(f"\n{BOLD}[1/3] Loading corpus ...{RESET}")
    if not Path(KNOWLEDGE).exists():
        print(f"{RED}ERROR: corpus not found at {KNOWLEDGE}{RESET}")
        sys.exit(1)
    with open(KNOWLEDGE) as f:
        knowledge = json.load(f)
    print(f"       {len(knowledge)} domains in knowledge base")

    print(f"{BOLD}[2/3] Building SQLite + TF-IDF index ...{RESET}")
    doc_count = build_corpus(knowledge, db_path=TEST_DB)
    print(f"       {doc_count} documents inserted into corpus")

    index = RagIndex(db_path=TEST_DB)
    indexed = index.load_from_db()
    print(f"       {indexed} documents indexed | vocab: {len(index._vocab)} terms")

    print(f"{BOLD}[3/3] Running {sum(len(v) for v in QUESTIONS.values())} questions across {len(QUESTIONS)} categories ...\n{RESET}")

    # -- Run all questions -------------------------------------------------
    total = 0
    results_all = []
    empty_count = 0
    short_count = 0

    for category, questions in QUESTIONS.items():
        section(category, YELLOW)
        cat_results = []

        for i, q in enumerate(questions, 1):
            total += 1
            result = run_question(q, index)
            results_all.append(result)
            cat_results.append(result)

            badge  = method_badge(result["method"])
            flag   = quality_flag(result)
            reply  = result["reply"]
            q_disp = q if q.strip() else "(empty string)"

            wrapped = textwrap.fill(reply, width=64, subsequent_indent="          ")
            print(f"\n  Q: {BOLD}{q_disp}{RESET}")
            print(f"  ▶  {wrapped}")
            hits_info = f"hits={result['hits']}"
            if result.get("top_score", 0) > 0:
                hits_info += f" score={result['top_score']:.3f}"
            print(f"     {badge} {flag}  {DIM}{hits_info} | {result['ms']}ms{RESET}")

            if "don't have enough" in reply.lower(): empty_count += 1
            if len(reply) < 40 and result["method"] not in ("skip", "fast_path", "identity"): short_count += 1

        answered = sum(1 for r in cat_results if "don't have enough" not in r["reply"].lower() and r["method"] != "skip")
        print(f"\n  {DIM}Category: {answered}/{len(questions)} answered{RESET}")

    # -- Summary ----------------------------------------------------------
    banner("TEST SUMMARY", GREEN)
    fast_c  = sum(1 for r in results_all if "fast_path" in r["method"])
    rag_c   = sum(1 for r in results_all if "rag" in r["method"])
    asher_c = sum(1 for r in results_all if "asher" in r["method"])
    skip_c  = sum(1 for r in results_all if "skip" in r["method"])
    answered_c = total - empty_count - skip_c

    scored = [r["top_score"] for r in results_all if r.get("top_score", 0) > 0]
    avg_score = round(sum(scored) / len(scored), 3) if scored else 0
    avg_ms = round(sum(r["ms"] for r in results_all) / len(results_all), 1)

    print(f"""
  Total questions  : {total}
  Answered         : {GREEN}{answered_c}{RESET}
  Empty responses  : {RED if empty_count else GREEN}{empty_count}{RESET}
  Short responses  : {YELLOW if short_count else GREEN}{short_count}{RESET}
  Skipped          : {DIM}{skip_c}{RESET}

  Method breakdown :
    fast_path   : {fast_c}
    identity    : {sum(1 for r in results_all if r["method"] == "identity")}
    rag         : {rag_c}
    asher+rag   : {asher_c}

  RAG avg score    : {avg_score}
  Avg latency      : {avg_ms}ms per query
""")

    if empty_count == 0:
        print(f"  {GREEN}{BOLD}✓ ALL SYSTEMS OPERATIONAL — Pipeline fully functional.{RESET}\n")
    else:
        print(f"  {YELLOW}{BOLD}⚠ {empty_count} questions returned empty — corpus may need expansion.{RESET}\n")


if __name__ == "__main__":
    main()

