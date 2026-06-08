"""
app.py — Zophiel / SOLIA  •  Railway entry point
=================================================
Runs a Flask REST API wrapping the full Zophiel pipeline:
  POST /ask      { "query": "..." }  -> { "reply": "...", "method": "..." }
  GET  /health   -> { "status": "ok", "docs": <corpus size> }
  GET  /         -> API info

Environment variables (all optional):
  PORT                 HTTP port  (default: 8080)
  DB_PATH              Path to SQLite corpus  (default: data/aureon.db)
  KNOWLEDGE_PATH       Path to corpus_knowledge.json  (default: data/corpus_knowledge.json)
  AUREON_WEB_SEARCH    Set to "1" to enable live DuckDuckGo search (auto-on for news queries)
"""

import os, sys, json, sqlite3, re
from pathlib import Path
from flask import Flask, request, jsonify

BASE = Path(__file__).resolve().parent
DB   = os.environ.get("DB_PATH",        str(BASE / "data" / "aureon.db"))
KNOW = os.environ.get("KNOWLEDGE_PATH", str(BASE / "data" / "corpus_knowledge.json"))
sys.path.insert(0, str(BASE))

from aureon_test_runner import RagIndex, fast_answer, asher_decode, synthesize, build_corpus
from cyber_defence import analyse_threat

# --- Startup: always rebuild corpus from JSON so db is never stale/malformed ---
print(f"[Zophiel] Building corpus from {KNOW} ...")
try:
    with open(KNOW, encoding="utf-8") as f:
        _knowledge = json.load(f)
    _doc_count = build_corpus(_knowledge, db_path=DB)
    print(f"[Zophiel] Corpus built — {_doc_count} documents written to {DB}")
except Exception as e:
    print(f"[Zophiel] WARNING: corpus build failed: {e}")
    _doc_count = 0

print(f"[Zophiel] Loading TF-IDF index ...")
INDEX = RagIndex(DB)
INDEX.load_from_db()
print(f"[Zophiel] Ready — {len(INDEX._docs)} documents indexed")

app = Flask(__name__)

# --- Current-events / news query detector ---
_NEWS_SIGNALS = re.compile(
    r"\b(today|right now|currently|latest|recent|news|happening|2024|2025|2026|"
    r"this week|this year|just announced|new release|just dropped|trending|"
    r"what.s going on|what is going on|whats happening)\b",
    re.I,
)

def _is_news_query(q: str) -> bool:
    return bool(_NEWS_SIGNALS.search(q))


def _live_search(query: str) -> list[dict]:
    """Google News RSS fetch — real headlines, no API key required."""
    import urllib.request, urllib.parse, re as _re
    hits = []
    try:
        params = urllib.parse.urlencode({
            "q": query, "hl": "en-US", "gl": "US", "ceid": "US:en",
        })
        req = urllib.request.Request(
            f"https://news.google.com/rss/search?{params}",
            headers={"User-Agent": "Mozilla/5.0 (Zophiel/SOLIA 1.0)"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        for item in _re.findall(r"<item>(.*?)</item>", raw, _re.DOTALL)[:8]:
            title_m = _re.search(r"<title>(.*?)</title>", item)
            if not title_m:
                continue
            title = title_m.group(1).strip()
            title = _re.sub(r"&amp;", "&", title)
            title = _re.sub(r"&lt;", "<", title)
            title = _re.sub(r"&gt;", ">", title)
            title = _re.sub(r"&#\d+;", "", title)
            # strip "- Source Name" suffix for cleaner synthesis
            title = _re.sub(r"\s+-\s+[^-]{3,50}$", "", title).strip()
            if len(title) > 20:
                hits.append({"text": title, "score": 0.85, "source": "google-news"})
    except Exception as ex:
        print(f"[Zophiel] live_search error: {ex}")
    return hits


@app.route("/")
def root():
    return jsonify({
        "name":       "Zophiel / SOLIA AI Engine",
        "version":    "1.0.0",
        "author":     "Aureon Software",
        "routes":     {"POST /ask": "query the AI", "GET /health": "status check"},
        "docs":       len(INDEX._docs),
        "web_search": True,
    })


@app.route("/health")
def health():
    try:
        conn = sqlite3.connect(DB)
        docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        return jsonify({"status": "ok", "docs": docs, "indexed": len(INDEX._docs)})
    except Exception as e:
        return jsonify({"status": f"db error: {e}", "docs": 0, "indexed": len(INDEX._docs)}), 200


@app.route("/ask", methods=["POST"])
def ask():
    body  = request.get_json(force=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query field required"}), 400

    # 1. Fast path (math / constants)
    fast = fast_answer(query)
    if fast:
        return jsonify({"reply": fast, "method": "fast_path"})

    # 2. Cyber defence overlay
    cyber = analyse_threat(query)

    # 3. Live web search — auto-on for news queries OR if AUREON_WEB_SEARCH=1
    use_web = _is_news_query(query) or os.environ.get("AUREON_WEB_SEARCH") == "1"
    live_hits = _live_search(query) if use_web else []

    # 4. RAG retrieval (blend live + corpus)
    corpus_hits = INDEX.query(query, top_k=8)
    hits = (live_hits + corpus_hits) if live_hits else corpus_hits

    # 5. Asher decode
    decode = asher_decode(query)

    # 6. Synthesize
    if live_hits:
        # News-mode: lead with live headlines, append corpus context
        headlines = [h["text"] for h in live_hits[:6]]
        corpus_texts = [h["text"] if isinstance(h, dict) else h
                        for h in corpus_hits[:3]]
        reply = "Here is what is happening right now:\n"
        reply += "\n".join(f"  • {h}" for h in headlines)
        if decode:
            reply += f"\n\nAsher Pattern: {decode}"
        if corpus_texts:
            ctx = synthesize(query, corpus_hits, asher_extra="")
            if ctx:
                reply += f"\n\nContext from corpus: {ctx}"
    else:
        reply = synthesize(query, hits, asher_extra=decode or "")
    if cyber:
        reply = cyber + "  " + reply

    method_parts = []
    if live_hits:  method_parts.append("live_search")
    method_parts.append("rag")
    if decode:     method_parts.append("asher_decode")
    if cyber:      method_parts.append("cyber_defence")
    method_parts.append("synthesize")

    return jsonify({
        "reply":       reply,
        "method":      "+".join(method_parts),
        "hits":        len(hits),
        "live_hits":   len(live_hits),
        "cyber":       cyber,
        "asher":       decode,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"[Zophiel] Starting on port {port}")
    app.run(host="0.0.0.0", port=port)
