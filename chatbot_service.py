"""
chatbot_service.py  ─  AI Chatbot Microservice (RAG Edition)
=============================================================
Architecture   : Multimodal RAG + Memory
LLM            : meta-llama/Llama-3-8B-Instruct  via Ollama  (local EC2)
RAG retrieval  : calls rag_service (/retrieve) → hybrid search + BGE-Reranker
Memory         : per-conversation sliding window stored in Redis DB 6
                 via redis_service (HTTP port 6390)
Port           : 7600

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAILOVER STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every incoming message checks whether rag_service is reachable.

  rag_service  UP   →  Full RAG pipeline
                         retrieve_context() → hybrid search + reranker
                         → build prompt with retrieved org-doc chunks
                         → call Llama-3 via Ollama
                         → cite source filenames in response

  rag_service  DOWN →  Direct Llama-3 fallback
                         skip retrieval entirely
                         → call Llama-3 via Ollama (LLM-only, no docs)
                         → note in response that org docs are unavailable
                         → retry RAG on next message automatically

Health check is a lightweight GET /health to rag_service with a
3-second timeout.  The result is cached for RAG_HEALTH_TTL seconds
(default 10 s) to avoid hammering a down service on every message.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# RAG service
RAG_SERVICE_URL  = os.getenv("RAG_SERVICE_URL",  "http://127.0.0.1:7900").rstrip("/")
RAG_TOP_K        = int(os.getenv("RAG_TOP_K",    "5"))
RAG_ALPHA        = float(os.getenv("RAG_ALPHA",  "0.6"))        # hybrid weight (1=dense, 0=BM25)
RAG_MIN_SCORE    = float(os.getenv("RAG_MIN_SCORE", "0.25"))    # discard chunks below this rerank score
RAG_HEALTH_TTL   = int(os.getenv("RAG_HEALTH_TTL", "10"))       # seconds to cache health-check result

# Ollama / Llama-3-8B-Instruct (local EC2 install)
OLLAMA_API   = os.getenv("OLLAMA_API",   "http://127.0.0.1:11434/api/generate")
MODEL_NAME   = os.getenv("OLLAMA_MODEL", "llama3")   # ollama tag; adjust if you pulled a different variant

# redis_service (HTTP gateway to Redis)
REDIS_SERVICE_URL = os.getenv("REDIS_SERVICE_URL", "http://127.0.0.1:6390").rstrip("/")

# Redis DB allocation (shared with other services — do not change)
CHATBOT_REDIS_DB = 6   # chat history + response cache

# Conversation memory
HISTORY_WINDOW = int(os.getenv("HISTORY_WINDOW", "8"))    # messages sent to LLM
HISTORY_LIMIT  = int(os.getenv("HISTORY_LIMIT",  "50"))   # max messages stored per conv

# Identity
BOT_NAME = "ChakoraBot"
ORG_NAME = "ChakoraHub"

# ═══════════════════════════════════════════════════════════════
# RAG HEALTH CACHE
# Lightweight in-process cache so health checks don't fire on
# every single message when rag_service is down.
# ═══════════════════════════════════════════════════════════════

_rag_health_cache: Dict[str, Any] = {
    "is_up":      None,    # True / False / None (unknown)
    "checked_at": 0.0,     # time.time() of last check
}


def _is_rag_up() -> bool:
    """
    Returns True if rag_service /health responds OK within 3 s.
    Result is cached for RAG_HEALTH_TTL seconds.
    """
    now = time.time()
    if (now - _rag_health_cache["checked_at"]) < RAG_HEALTH_TTL and \
            _rag_health_cache["is_up"] is not None:
        return _rag_health_cache["is_up"]

    try:
        req = urllib.request.Request(
            f"{RAG_SERVICE_URL}/health",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data   = json.loads(resp.read().decode())
            is_up  = data.get("status") in ("healthy", "degraded")   # degraded = partial, still usable
    except Exception as e:
        print(f"[RAG-health] rag_service unreachable: {e}")
        is_up = False

    _rag_health_cache["is_up"]      = is_up
    _rag_health_cache["checked_at"] = now
    status_str = "UP ✅" if is_up else "DOWN ❌"
    print(f"[RAG-health] rag_service is {status_str}")
    return is_up


def _invalidate_rag_health_cache() -> None:
    """Force re-check on next call."""
    _rag_health_cache["checked_at"] = 0.0


# ═══════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════

app = FastAPI(title=f"{ORG_NAME} AI Chatbot Service (RAG + Fallback)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# REDIS SERVICE HTTP HELPERS
# ═══════════════════════════════════════════════════════════════

def _redis_service_candidates() -> List[str]:
    urls = [REDIS_SERVICE_URL]
    if REDIS_SERVICE_URL.endswith(":6390"):
        urls.append(REDIS_SERVICE_URL[:-5] + ":6380")
    elif REDIS_SERVICE_URL.endswith(":6380"):
        urls.append(REDIS_SERVICE_URL[:-5] + ":6390")
    return list(dict.fromkeys(urls))


def _rs(method: str, path: str, payload=None, query: Optional[dict] = None) -> dict:
    """
    HTTP call to redis_service.  Timeout = 5 s.
    Returns {} on any error — cache failures never block LLM responses.
    """
    body    = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}

    for base_url in _redis_service_candidates():
        url = f"{base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        req = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw  = resp.read().decode()
                data = json.loads(raw) if raw else {}
                return data if isinstance(data, dict) else {"success": True, "data": data}
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode()
            except Exception:
                detail = str(e)
            print(f"[redis_service] HTTP {e.code} [{method} {path}] @ {base_url}: {detail}")
            break
        except Exception as exc:
            print(f"[redis_service] request failed [{method} {path}] @ {base_url}: {exc}")

    return {"success": False}


# ── Response cache ─────────────────────────────────────────────

def cache_response_get(response_hash: str) -> Optional[str]:
    res = _rs("GET", "/chatbot/response/get", query={"response_hash": response_hash})
    if res.get("success") and res.get("found"):
        print(f"[cache] HIT  hash={response_hash[:12]}…")
        return res.get("response_text")
    return None


def cache_response_set(response_hash: str, text: str) -> None:
    _rs("POST", "/chatbot/response/set", payload={
        "response_hash": response_hash, "response_text": text,
    })


# ── Conversation history ───────────────────────────────────────

def history_append(conv_id: str, role: str, content: str,
                   user_id: str = "anonymous") -> int:
    res = _rs("POST", "/chatbot/history/append", payload={
        "conversation_id": conv_id, "role": role,
        "content": content, "user_id": user_id,
    })
    return int(res.get("message_count", 0))


def history_get(conv_id: str, limit: int = HISTORY_LIMIT) -> List[Dict]:
    res = _rs("GET", "/chatbot/history/get",
              query={"conversation_id": conv_id, "limit": limit})
    if res.get("success") and res.get("found"):
        return res.get("messages", [])
    return []


def conversation_clear(conv_id: str) -> int:
    res = _rs("DELETE", "/chatbot/conversation/clear",
              query={"conversation_id": conv_id})
    return int(res.get("deleted", 0))


# ═══════════════════════════════════════════════════════════════
# RAG RETRIEVAL  (called only when rag_service is UP)
# ═══════════════════════════════════════════════════════════════

def retrieve_context(query: str, top_k: int = RAG_TOP_K) -> List[Dict]:
    """
    POST to rag_service /retrieve → hybrid search + BGE-Reranker-v2-m3.
    Returns filtered chunk list [{text, filename, modality, rerank_score, …}].
    Filters out chunks whose rerank_score < RAG_MIN_SCORE.
    Returns [] on any network/parse error.
    """
    try:
        payload = json.dumps({
            "query": query, "top_k": top_k, "alpha": RAG_ALPHA,
        }).encode()
        req = urllib.request.Request(
            f"{RAG_SERVICE_URL}/retrieve",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        results  = data.get("results", [])
        filtered = [r for r in results
                    if r.get("rerank_score", 1.0) >= RAG_MIN_SCORE]
        print(f"[RAG] '{query[:60]}…' → {len(results)} hits, "
              f"{len(filtered)} above threshold ({RAG_MIN_SCORE})")
        return filtered

    except Exception as e:
        print(f"[RAG] retrieve_context failed: {e}")
        # Invalidate health cache so the next message re-checks immediately
        _invalidate_rag_health_cache()
        return []


def format_rag_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a context block for the LLM prompt."""
    if not chunks:
        return ""

    text_chunks  = [c for c in chunks if c.get("modality", "text") == "text"]
    image_chunks = [c for c in chunks if c.get("modality") == "image"]

    parts = []
    if text_chunks:
        parts.append("### Relevant Organisation Documents\n")
        for i, chunk in enumerate(text_chunks, 1):
            src   = chunk.get("filename", "unknown")
            score = chunk.get("rerank_score", 0.0)
            parts.append(f"[Doc {i} — {src}  score={score:.3f}]\n{chunk['text']}\n")

    if image_chunks:
        parts.append("\n### Referenced Images / Diagrams\n")
        for chunk in image_chunks:
            src = chunk.get("filename", "unknown")
            parts.append(
                f"- {chunk.get('image_caption', chunk['text'])}  (source: {src})"
            )

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# CONVERSATION CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_conversation_context(conv_id: str) -> str:
    """Return the last HISTORY_WINDOW messages formatted for the LLM."""
    messages = history_get(conv_id, limit=HISTORY_WINDOW)
    if not messages:
        return ""
    lines = ["### Conversation History"]
    for msg in messages[-HISTORY_WINDOW:]:
        role = "User" if msg["role"] == "user" else BOT_NAME
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# Two variants: with RAG context, and fallback (LLM-only).
# ═══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT_RAG = f"""You are {BOT_NAME}, the official AI assistant for {ORG_NAME}.

{ORG_NAME} is an educational platform offering courses and internship programs in:
Data Engineering, Python, Web Development, Machine Learning, Cloud, DevOps, and more.

Your primary responsibility is to answer questions using the provided Organisation Documents.
Rules:
1. Base your answer PRIMARILY on the retrieved document context below.
2. If the context contains a direct answer, cite the source document name.
3. If the context is partial, combine it with your general knowledge and say so explicitly.
4. If the context is irrelevant or absent, answer from general knowledge but clearly state the org docs didn't cover this.
5. Be concise and professional. Keep responses under 5 sentences unless the question demands more detail.
6. Never fabricate facts about {ORG_NAME}; only state what's in the documents.
7. If you see [Image …] references, acknowledge them but don't invent image contents.
"""

_SYSTEM_PROMPT_FALLBACK = f"""You are {BOT_NAME}, the official AI assistant for {ORG_NAME}.

{ORG_NAME} is an educational platform offering courses and internship programs in:
Data Engineering, Python, Web Development, Machine Learning, Cloud, DevOps, and more.

⚠️  Note: The organisation document retrieval system is temporarily unavailable.
You are answering from your general training knowledge only.
If the question is specific to {ORG_NAME} internal documents or data, please let the user know
they should try again shortly when the document system is back online.

Rules:
1. Answer helpfully from general knowledge.
2. Be transparent when you are not certain.
3. Keep responses concise and professional.
"""


# ═══════════════════════════════════════════════════════════════
# LLM CALL  (Ollama — Llama-3-8B-Instruct, local EC2)
# ═══════════════════════════════════════════════════════════════

async def call_llm(
    user_message:  str,
    rag_context:   str,
    conv_context:  str,
    rag_available: bool = True,
) -> str:
    """
    Call Ollama with:
      - System prompt  (RAG variant or fallback variant based on rag_available)
      - RAG context    (retrieved org-doc chunks, empty string if RAG is down)
      - Conversation history
      - User question

    Uses a 30-min response cache keyed on hash(user_message + rag_context).
    The cache key includes rag_available so RAG and non-RAG responses are
    stored separately.
    """
    cache_key = hashlib.sha256(
        f"{user_message}||{rag_context}||{rag_available}".encode()
    ).hexdigest()

    # ── Cache HIT ─────────────────────────────────────────────
    cached = cache_response_get(cache_key)
    if cached is not None:
        return cached

    # ── Build full prompt ──────────────────────────────────────
    system = _SYSTEM_PROMPT_RAG if rag_available else _SYSTEM_PROMPT_FALLBACK
    sections = [system]
    if rag_context:
        sections.append(f"\n{rag_context}")
    if conv_context:
        sections.append(f"\n{conv_context}")
    sections.append(f"\n### User Question\n{user_message}")
    sections.append(f"\n### {BOT_NAME}'s Response")
    full_prompt = "\n".join(sections)

    # ── Call Ollama ────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                OLLAMA_API,
                json={
                    "model":  MODEL_NAME,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,    # lower for factual RAG answers
                        "top_p":       0.9,
                        "max_tokens":  512,
                        "stop":        ["### User Question", "User:"],
                    },
                },
            )

        if resp.status_code == 200:
            ai_text = resp.json().get("response", "").strip()
            if not ai_text:
                ai_text = ("I couldn't generate a response. "
                           "Please try again.")
            cache_response_set(cache_key, ai_text)
            return ai_text
        else:
            print(f"[Ollama] error {resp.status_code}: {resp.text[:200]}")
            return "I'm having technical difficulties. Please try again in a moment."

    except httpx.TimeoutException:
        return ("My response is taking too long. "
                "Please try a shorter question or retry in a moment.")
    except Exception as e:
        print(f"[Ollama] exception: {e}")
        traceback.print_exc()
        return "I encountered an error generating a response. Please try again."


# ═══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    message:         str
    conversation_id: Optional[str] = None
    user_id:         Optional[str] = "anonymous"
    user_name:       Optional[str] = "User"
    user_type:       Optional[str] = "anonymous"


class ChatResponse(BaseModel):
    success:         bool
    response:        str
    conversation_id: str
    sources:         List[str]    # source filenames (empty when RAG is down)
    rag_used:        bool         # True = RAG pipeline used, False = LLM-only fallback
    timestamp:       str


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def generate_conversation_id() -> str:
    ts   = str(int(time.time() * 1000))
    salt = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    return f"conv_{ts}_{salt}"


# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    rag_up = _is_rag_up()
    cache_stats_res = _rs("GET", "/chatbot/stats")
    cache_stats     = (cache_stats_res.get("stats", {})
                       if cache_stats_res.get("success") else {})
    return {
        "service":   f"{ORG_NAME} AI Chatbot (RAG + Fallback)",
        "status":    "running",
        "model":     MODEL_NAME,
        "rag": {
            "url":       RAG_SERVICE_URL,
            "reachable": rag_up,
            "top_k":     RAG_TOP_K,
            "alpha":     RAG_ALPHA,
            "mode":      "RAG pipeline" if rag_up else "LLM-only fallback",
        },
        "memory":    f"redis_service DB{CHATBOT_REDIS_DB}",
        "timestamp": datetime.now().isoformat(),
        "cache":     cache_stats,
    }


@app.post("/chatbot/message", response_model=ChatResponse)
async def handle_message(chat: ChatMessage):
    """
    Full chatbot pipeline with automatic RAG ↔ LLM-only failover:

    RAG service UP:
      1. Persist user message
      2. retrieve_context()  → hybrid search + BGE-Reranker
      3. build conversation context (sliding window)
      4. call_llm(rag_context=<chunks>)  → Llama-3 via Ollama
      5. Persist AI response
      6. Return response + source filenames

    RAG service DOWN:
      1. Persist user message
      2. Skip retrieval (rag_context = "")
      3. build conversation context
      4. call_llm(rag_available=False)   → Llama-3 via Ollama, fallback system prompt
      5. Persist AI response
      6. Return response with rag_used=False so the client can show a notice
    """
    try:
        conv_id = chat.conversation_id or generate_conversation_id()

        # Persist user message
        history_append(conv_id, "user", chat.message, chat.user_id or "anonymous")

        # ── Decide: RAG or LLM-only fallback ──────────────────
        rag_available = _is_rag_up()

        rag_chunks  = []
        rag_context = ""
        sources     = []

        if rag_available:
            # Full RAG pipeline
            rag_chunks  = retrieve_context(chat.message, top_k=RAG_TOP_K)
            rag_context = format_rag_context(rag_chunks)
            sources     = list({
                c.get("filename", "") for c in rag_chunks if c.get("filename")
            })
            if not rag_chunks:
                # retrieve_context returned nothing (low confidence / empty index)
                print("[RAG] No chunks above threshold — using LLM-only for this message")
                rag_available = False
        else:
            print("[chatbot] RAG service is DOWN — using LLM-only fallback")

        # Build conversation context (sliding window from memory)
        conv_context = build_conversation_context(conv_id)

        # Generate AI response
        ai_response = await call_llm(
            user_message  = chat.message,
            rag_context   = rag_context,
            conv_context  = conv_context,
            rag_available = rag_available,
        )

        # Persist AI response
        history_append(conv_id, "assistant", ai_response, BOT_NAME)

        return ChatResponse(
            success         = True,
            response        = ai_response,
            conversation_id = conv_id,
            sources         = sources,
            rag_used        = rag_available,
            timestamp       = datetime.now().isoformat(),
        )

    except Exception as e:
        print(f"[chatbot] handle_message error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chatbot/history")
def get_history(conversation_id: str):
    try:
        messages = history_get(conversation_id, limit=HISTORY_LIMIT)
        return {
            "success":         True,
            "conversation_id": conversation_id,
            "messages":        messages,
            "count":           len(messages),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chatbot/clear")
def clear_conversation(data: dict):
    conv_id = data.get("conversation_id")
    if not conv_id:
        return {"success": False, "message": "No conversation_id provided"}
    deleted = conversation_clear(conv_id)
    return {
        "success":         True,
        "conversation_id": conv_id,
        "keys_deleted":    deleted,
    }


@app.get("/chatbot/stats")
def get_stats():
    try:
        rag_up      = _is_rag_up()
        cache_res   = _rs("GET", "/chatbot/stats")
        cache_stats = (cache_res.get("stats", {})
                       if cache_res.get("success") else {})

        rag_index: dict = {}
        if rag_up:
            try:
                req = urllib.request.Request(f"{RAG_SERVICE_URL}/stats")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    rag_index = json.loads(resp.read().decode())
            except Exception as e:
                rag_index = {"error": str(e)}

        return {
            "success": True,
            "stats": {
                "model":           MODEL_NAME,
                "rag_service":     RAG_SERVICE_URL,
                "rag_status":      "up" if rag_up else "down (LLM fallback active)",
                "rag_index":       rag_index,
                "active_convs":    cache_stats.get("active_conversations", 0),
                "cached_llm_resp": cache_stats.get("cached_llm_responses", 0),
                "history_window":  HISTORY_WINDOW,
                "rag_top_k":       RAG_TOP_K,
                "rag_alpha":       RAG_ALPHA,
                "rag_min_score":   RAG_MIN_SCORE,
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/chatbot/rag-status")
def rag_status():
    """Quick endpoint to check whether RAG service is reachable."""
    _invalidate_rag_health_cache()   # force fresh check
    up = _is_rag_up()
    return {
        "rag_service": RAG_SERVICE_URL,
        "is_up":       up,
        "mode":        "RAG pipeline" if up else "LLM-only fallback",
        "checked_at":  datetime.now().isoformat(),
    }


@app.get("/chatbot/rag-test")
async def rag_test(q: str = "What courses does ChakoraHub offer?"):
    """Debug endpoint: run retrieval only (no LLM) for a given query."""
    if not _is_rag_up():
        return {
            "query":   q,
            "error":   "rag_service is currently unavailable",
            "chunks":  [],
            "context": "",
        }
    chunks = retrieve_context(q, top_k=RAG_TOP_K)
    return {
        "query":   q,
        "chunks":  chunks,
        "context": format_rag_context(chunks),
    }


# ═══════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN
# ═══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    print("=" * 65)
    print(f"🚀 {ORG_NAME} Chatbot Service  (RAG + LLM-only Fallback)")
    print(f"   LLM          : {MODEL_NAME}  via Ollama ({OLLAMA_API})")
    print(f"   RAG service  : {RAG_SERVICE_URL}  top_k={RAG_TOP_K}  alpha={RAG_ALPHA}")
    print(f"   Memory       : redis_service {REDIS_SERVICE_URL} DB{CHATBOT_REDIS_DB}")
    print(f"   History      : {HISTORY_WINDOW}-message sliding window")
    print(f"   Failover     : RAG DOWN → LLM-only (Llama-3 direct)")
    print("=" * 65)

    # redis_service health
    redis_health = _rs("GET", "/health")
    if redis_health.get("success"):
        print("✓ redis_service reachable")
    else:
        print("✗ redis_service unreachable — conversation memory degraded")

    # RAG service health
    if _is_rag_up():
        print(f"✓ rag_service reachable at {RAG_SERVICE_URL}")
        print("  Mode: FULL RAG pipeline (hybrid search + BGE-Reranker)")
    else:
        print(f"✗ rag_service UNREACHABLE at {RAG_SERVICE_URL}")
        print("  Mode: LLM-only fallback (Llama-3 direct, no org docs)")
        print("  The chatbot will automatically switch back to RAG when the service recovers.")

    # Ollama health
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get("http://127.0.0.1:11434/api/tags")
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"✓ Ollama reachable — models: {models}")
            if MODEL_NAME not in " ".join(models):
                print(f"  ⚠️  '{MODEL_NAME}' not found. Run: ollama pull {MODEL_NAME}")
        else:
            print(f"✗ Ollama returned status {resp.status_code}")
    except Exception as e:
        print(f"✗ Ollama unreachable: {e}")
        print("  ⚠️  The LLM itself is down — chatbot cannot respond until Ollama is running.")

    print("✓ Chatbot service ready")


# ═══════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7600)
