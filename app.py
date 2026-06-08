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
  AUREON_WEB_SEARCH    Set to "1" to enable Zophiel live DuckDuckGo search
"""

import os, sys, json, sqlite3, math, re
from pathlib import Path
from flask import Flask, request, jsonify

BASE = Path(__file__).resolve().parent
DB   = os.environ.get("DB_PATH",        str(BASE / "data" / "aureon.db"))
KNOW = os.environ.get("KNOWLEDGE_PATH", str(BASE / "data" / "corpus_knowledge.json"))
sys.path.insert(0, str(BASE))

from aureon_test_runner import RagIndex, fast_answer, asher_decode, synthesize
from cyber_defence import analyse_threat

print(f"[Zophiel] Loading corpus from {DB} ...")
INDEX = RagIndex(DB)
print(f"[Zophiel] Ready — {len(INDEX.docs)} documents")

app = Flask(__name__)

@app.route("/")
def root():
    return jsonify({
        "name":       "Zophiel / SOLIA AI Engine",
        "version":    "1.0.0",
        "author":     "Aureon Software",
        "routes":     {"POST /ask": "query the AI", "GET /health": "status check"},
        "web_search": os.environ.get("AUREON_WEB_SEARCH") == "1",
    })

@app.route("/health")
def health():
    try:
        conn = sqlite3.connect(DB)
        docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        return jsonify({"status": "ok", "docs": docs})
    except Exception as e:
        return jsonify({"status": f"db error: {e}", "docs": 0}), 500

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

    # 3. Zophiel live search (optional — requires AUREON_WEB_SEARCH=1)
    live_hits = []
    if os.environ.get("AUREON_WEB_SEARCH") == "1":
        try:
            import urllib.request, urllib.parse
            params = urllib.parse.urlencode({"q": query, "format": "json",
                                             "no_redirect": "1", "no_html": "1"})
            req = urllib.request.Request(
                f"https://api.duckduckgo.com/?{params}",
                headers={"User-Agent": "SOLIA/Zophiel 1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())
            if data.get("AbstractText"):
                live_hits.append({"text": data["AbstractText"], "score": 0.9,
                                  "source": data.get("AbstractURL", "ddg")})
            for t in data.get("RelatedTopics", [])[:3]:
                if isinstance(t, dict) and t.get("Text"):
                    live_hits.append({"text": t["Text"], "score": 0.7,
                                      "source": t.get("FirstURL", "ddg")})
        except Exception:
            pass

    # 4. RAG retrieval
    hits = live_hits if live_hits else INDEX.query(query, top_k=8)

    # 5. Asher decode
    decode = asher_decode(query)

    # 6. Synthesize
    reply = synthesize(query, hits, asher_extra=decode or "")
    if cyber:
        reply = cyber + "  " + reply

    method_parts = []
    if live_hits:   method_parts.append("zophiel_search")
    method_parts.append("rag")
    if decode:      method_parts.append("asher_decode")
    if cyber:       method_parts.append("cyber_defence")
    method_parts.append("synthesize")

    return jsonify({
        "reply":  reply,
        "method": "+".join(method_parts),
        "hits":   len(hits),
        "cyber":  cyber,
        "asher":  decode,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"[Zophiel] Starting on port {port}")
    app.run(host="0.0.0.0", port=port)
