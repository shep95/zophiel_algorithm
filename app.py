"""
app.py — Zophiel / SOLIA  •  Railway entry point
=================================================
Runs a Flask REST API wrapping the full Zophiel pipeline:
  POST /ask      { "query": "..." }  -> { "reply": "...", "method": "..." }
  GET  /health   -> { "status": "ok", "docs": <corpus size> }
  GET  /         -> API info

Web search fallback system:
  - RAG confidence < 0.35  → auto Wikipedia + web search
  - News/current-events query → Google News RSS headlines
  - AUREON_WEB_SEARCH=1    → force web search on all queries

Startup: Flask binds immediately; corpus is built in a background thread
so the Railway healthcheck passes within seconds.
"""

import os, sys, json, sqlite3, re, threading
from pathlib import Path
from flask import Flask, request, jsonify

BASE = Path(__file__).resolve().parent
DB   = os.environ.get("DB_PATH",        str(BASE / "data" / "aureon.db"))
KNOW = os.environ.get("KNOWLEDGE_PATH", str(BASE / "data" / "corpus_knowledge.json"))
RAG_MIN_CONFIDENCE = float(os.environ.get("RAG_CONFIDENCE", "0.35"))

# API key auth — set ZOPHIEL_API_KEY env var on Railway to enable.
# /health and / are always public (Railway healthcheck needs them).
# If not set, the API is open (backwards-compatible with existing deploys).
_API_KEY: str | None = os.environ.get("ZOPHIEL_API_KEY") or None

sys.path.insert(0, str(BASE))

from aureon_test_runner import RagIndex, fast_answer, asher_decode, synthesize, build_corpus
from cyber_defence import analyse_threat

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state — populated by background loader
# ---------------------------------------------------------------------------
INDEX: RagIndex | None = None
_READY = False          # True once corpus is loaded and index is built
_STATUS = "loading"     # human-readable status for /health

def _boot():
    """Build corpus and load TF-IDF index. Runs in a background thread."""
    global INDEX, _READY, _STATUS
    try:
        _STATUS = "building corpus"
        # Always start from a clean DB — removes any stale/malformed file
        if os.path.exists(DB):
            os.remove(DB)
            print(f"[Zophiel] Removed old DB: {DB}")
        print(f"[Zophiel] Building corpus from {KNOW} ...")
        with open(KNOW, encoding="utf-8") as f:
            knowledge = json.load(f)
        doc_count = build_corpus(knowledge, db_path=DB)
        print(f"[Zophiel] Corpus built — {doc_count} documents")

        _STATUS = "loading index"
        print(f"[Zophiel] Loading TF-IDF index ...")
        idx = RagIndex(DB)
        idx.load_from_db()
        INDEX = idx
        _READY = True
        _STATUS = "ok"
        print(f"[Zophiel] Ready — {len(INDEX._docs)} documents indexed")
    except Exception as e:
        _STATUS = f"boot error: {e}"
        print(f"[Zophiel] BOOT ERROR: {e}")

threading.Thread(target=_boot, daemon=True).start()

# ---------------------------------------------------------------------------
# Query classifiers
# ---------------------------------------------------------------------------
_NEWS_RE = re.compile(
    r"\b(today|right now|currently|latest|recent|news|happening|2024|2025|2026|"
    r"this week|this year|just announced|new release|just dropped|trending|"
    r"what.s going on|what is going on|whats happening)\b",
    re.I,
)

def _is_news_query(q: str) -> bool:
    return bool(_NEWS_RE.search(q))

def _corpus_confidence(hits: list) -> float:
    if not hits:
        return 0.0
    return max(h.get("score", 0.0) if isinstance(h, dict) else 0.0 for h in hits)

# ---------------------------------------------------------------------------
# Web search functions
# ---------------------------------------------------------------------------
import urllib.request, urllib.parse as _urlparse

def _extract_subject(query: str) -> list[str]:
    """Return candidate Wikipedia titles to try, most specific first."""
    q = re.sub(
        r"^(when did|who is|who was|what is|what was|where did|where is|"
        r"how did|how does|tell me about|explain|describe|what happened to)\s+",
        "", query.strip(), flags=re.I,
    )
    # Cut at connectors
    q = re.split(r"\s+(and\s+what|and\s+|,|because|but)\s*", q, maxsplit=1)[0]
    # Strip trailing verbs / common stop words
    q = re.sub(r"\s+(die|died|death|born|live|famous|young|old|known|called)\s*$", "", q, flags=re.I).strip()
    candidates = [q]
    # Also try without leading "Queen" / "King" / "President" honorific
    no_title = re.sub(r"^(queen|king|president|prime minister|prince|princess)\s+", "", q, flags=re.I).strip()
    if no_title and no_title != q:
        candidates.append(no_title)
    return candidates


def _fetch_wiki_pages(titles: str, query_words: set) -> list[dict]:
    """Fetch Wikipedia intro extracts for pipe-separated titles, score sentences."""
    params = _urlparse.urlencode({
        "action": "query", "titles": titles,
        "prop": "extracts", "exintro": "1", "explaintext": "1", "format": "json",
    })
    req = urllib.request.Request(
        f"https://en.wikipedia.org/w/api.php?{params}",
        headers={"User-Agent": "Zophiel/SOLIA 1.0 (educational)"},
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode())
    candidates: list[tuple[float, str, str]] = []
    for page in data.get("query", {}).get("pages", {}).values():
        if int(page.get("pageid", -1)) < 0:
            continue
        title = page.get("title", "wikipedia")
        extract = page.get("extract", "").strip()
        if not extract:
            continue
        for sent in re.split(r"(?<=[.!?])\s+", extract)[:20]:
            sent = sent.strip()
            if len(sent) < 40:
                continue
            sent_words = set(re.findall(r"[a-z]{3,}", sent.lower()))
            overlap = len(query_words & sent_words) / max(1, len(query_words))
            candidates.append((overlap, sent, title))
    candidates.sort(key=lambda x: -x[0])
    return [{"text": s, "score": 0.85 + sc * 0.1, "source": f"wikipedia:{t}"}
            for sc, s, t in candidates[:8]]


def _wikipedia_search(query: str) -> list[dict]:
    """Direct subject lookup first; full-text search fallback."""
    query_words = set(re.findall(r"[a-z]{3,}", query.lower()))
    hits: list[dict] = []
    try:
        # 1. Direct title lookup — try each candidate subject until we get hits
        for subject in _extract_subject(query):
            if subject:
                hits = _fetch_wiki_pages(subject, query_words)
            if hits:
                break

        # 2. If direct lookup empty, fall back to full-text search
        if not hits:
            params = _urlparse.urlencode({
                "action": "query", "list": "search",
                "srsearch": query, "format": "json", "srlimit": 3,
            })
            req = urllib.request.Request(
                f"https://en.wikipedia.org/w/api.php?{params}",
                headers={"User-Agent": "Zophiel/SOLIA 1.0 (educational)"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())
            results = data.get("query", {}).get("search", [])
            if results:
                titles = "|".join(r["title"] for r in results)
                hits = _fetch_wiki_pages(titles, query_words)
    except Exception as ex:
        print(f"[Zophiel] wikipedia_search error: {ex}")
    return hits


def _news_search(query: str) -> list[dict]:
    """Google News RSS — returns live headlines."""
    hits = []
    try:
        params = _urlparse.urlencode({
            "q": query, "hl": "en-US", "gl": "US", "ceid": "US:en",
        })
        req = urllib.request.Request(
            f"https://news.google.com/rss/search?{params}",
            headers={"User-Agent": "Mozilla/5.0 (Zophiel/SOLIA 1.0)"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        for item in re.findall(r"<item>(.*?)</item>", raw, re.DOTALL)[:8]:
            tm = re.search(r"<title>(.*?)</title>", item)
            if not tm:
                continue
            title = tm.group(1).strip()
            title = re.sub(r"&amp;", "&", title)
            title = re.sub(r"&lt;", "<", title)
            title = re.sub(r"&gt;", ">", title)
            title = re.sub(r"&#\d+;", "", title)
            title = re.sub(r"\s+-\s+[^-]{3,50}$", "", title).strip()
            if len(title) > 20:
                hits.append({"text": title, "score": 0.85, "source": "google-news"})
    except Exception as ex:
        print(f"[Zophiel] news_search error: {ex}")
    return hits


def _web_fallback(query: str) -> tuple[list[dict], str]:
    """Wikipedia first, Google News as backup."""
    wiki_hits = _wikipedia_search(query)
    if wiki_hits:
        return wiki_hits, "wikipedia"
    news_hits = _news_search(query)
    return news_hits, "google-news"

# ---------------------------------------------------------------------------
# Reply builders
# ---------------------------------------------------------------------------
def _build_news_reply(query: str, live_hits: list, corpus_hits: list, decode: str) -> str:
    headlines = [h["text"] for h in live_hits[:6]]
    reply = "Here is what is happening right now:\n"
    reply += "\n".join(f"  • {h}" for h in headlines)
    if decode:
        reply += f"\n\nAsher Pattern: {decode}"
    ctx = synthesize(query, corpus_hits, asher_extra="")
    if ctx:
        reply += f"\n\nContext from corpus: {ctx}"
    return reply


def _build_web_reply(query: str, web_hits: list, decode: str) -> str:
    reply = synthesize(query, web_hits, asher_extra=decode or "")
    if not reply:
        texts = [h["text"] for h in web_hits[:4] if isinstance(h, dict)]
        reply = " ".join(texts)
    return reply

# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------
def _check_auth() -> bool:
    """Return True if request is authorised.

    Authorised when:
      - ZOPHIEL_API_KEY is not set (open mode, backwards-compatible)
      - Authorization: Bearer <key> header matches ZOPHIEL_API_KEY
      - x-api-key: <key> header matches (alternative header for frontends)
    """
    if _API_KEY is None:
        return True
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer ") and auth_header[7:] == _API_KEY:
        return True
    if request.headers.get("x-api-key", "") == _API_KEY:
        return True
    return False

# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route("/")
def root():
    return jsonify({
        "name":       "Zophiel / SOLIA AI Engine",
        "version":    "1.1.0",
        "author":     "Aureon Software",
        "status":     _STATUS,
        "routes":     {"POST /ask": "query the AI", "GET /health": "status check"},
        "docs":       len(INDEX._docs) if INDEX else 0,
        "web_search": True,
        "rag_confidence_threshold": RAG_MIN_CONFIDENCE,
    })


@app.route("/health")
def health():
    # Always return 200 — Railway uses this to confirm the process is alive,
    # not to confirm the corpus is loaded. Include ready flag for monitoring.
    docs = 0
    try:
        conn = sqlite3.connect(DB)
        docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
    except Exception:
        pass
    return jsonify({
        "status":  _STATUS,
        "ready":   _READY,
        "docs":    docs,
        "indexed": len(INDEX._docs) if INDEX else 0,
    }), 200


@app.route("/ask", methods=["POST"])
def ask():
    if not _check_auth():
        return jsonify({"error": "Unauthorized — provide Authorization: Bearer <key>"}), 401

    body  = request.get_json(force=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query field required"}), 400

    # If corpus is still loading, go straight to web search
    if not _READY or INDEX is None:
        web_hits, web_source = _web_fallback(query)
        reply = _build_web_reply(query, web_hits, decode="")
        if not reply:
            reply = "I am still loading my knowledge base. Please ask again in a few seconds."
        return jsonify({"reply": reply, "method": f"web_fallback({web_source})",
                        "ready": False})

    # 1. Fast path (math / constants)
    fast = fast_answer(query)
    if fast:
        return jsonify({"reply": fast, "method": "fast_path"})

    # 2. Code generation
    try:
        from brain.code_engine import generate_code
        code_reply = generate_code(query)
        if code_reply:
            return jsonify({"reply": code_reply, "method": "code_engine"})
    except Exception:
        pass

    # 3. Cyber defence overlay
    cyber = analyse_threat(query)

    # 3. Asher decode
    decode = asher_decode(query)

    # 4. RAG retrieval
    corpus_hits = INDEX.query(query, top_k=8)
    confidence  = _corpus_confidence(corpus_hits)

    web_hits   = []
    web_source = ""
    method_parts = []

    # 5. Route: news query → Google News headlines
    if _is_news_query(query) or os.environ.get("AUREON_WEB_SEARCH") == "1":
        web_hits = _news_search(query)
        web_source = "google-news"
        reply = _build_news_reply(query, web_hits, corpus_hits, decode)
        method_parts = ["live_search", "rag", "synthesize"]

    # 6. Route: low-confidence corpus → Wikipedia + web fallback
    elif confidence < RAG_MIN_CONFIDENCE:
        web_hits, web_source = _web_fallback(query)
        if web_hits:
            reply = _build_web_reply(query, web_hits, decode)
            method_parts = [f"web_fallback({web_source})", "synthesize"]
            if decode:
                method_parts.insert(1, "asher_decode")
        else:
            reply = synthesize(query, corpus_hits, asher_extra=decode or "")
            method_parts = ["rag", "synthesize"]

    # 7. Route: confident corpus answer
    else:
        reply = synthesize(query, corpus_hits, asher_extra=decode or "")
        method_parts = ["rag", "synthesize"]
        if decode:
            method_parts.insert(1, "asher_decode")

    if cyber:
        reply = cyber + "  " + reply
        method_parts.append("cyber_defence")

    return jsonify({
        "reply":       reply,
        "method":      "+".join(method_parts),
        "confidence":  round(confidence, 3),
        "corpus_hits": len(corpus_hits),
        "web_hits":    len(web_hits),
        "web_source":  web_source,
        "asher":       decode,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"[Zophiel] Starting on port {port} ...")
    app.run(host="0.0.0.0", port=port)
