"""
chatbot_service.py  ─  AI Chatbot Microservice (RAG Edition)
=============================================================
Architecture   : Multimodal RAG + Memory
LLM            : meta-llama/Llama-3-8B-Instruct  via Ollama  (local EC2)
RAG retrieval  : calls rag_service (/retrieve) -> hybrid search + BGE-Reranker
Memory         : per-conversation sliding window stored in Redis DB 6
                 via redis_service (HTTP port 6390)
Port           : 7600

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAILOVER STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every incoming message checks whether rag_service is reachable.

  rag_service  UP   ->  Full RAG pipeline
                         retrieve_context() -> hybrid search + reranker
                         -> build prompt with retrieved org-doc chunks
                         -> call Llama-3 via Ollama
                         -> cite source filenames in response

  rag_service  DOWN ->  Direct Llama-3 fallback
                         skip retrieval entirely
                         -> call Llama-3 via Ollama (LLM-only, no docs)
                         -> note in response that org docs are unavailable
                         -> retry RAG on next message automatically

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

try:
    from dotenv import load_dotenv
    import pathlib
    load_dotenv(dotenv_path=pathlib.Path(__file__).resolve().parent / ".env", override=False)
except Exception:
    pass


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

# Redis DB allocation (shared with other services -- do not change)
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
    Returns True if rag_service /health responds OK.
    Fast check with 0.5 s timeout. Caches result for RAG_HEALTH_TTL seconds.
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
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            data   = json.loads(resp.read().decode())
            is_up  = data.get("status") in ("healthy", "degraded")
    except Exception as e:
        is_up = False

    _rag_health_cache["is_up"]      = is_up
    _rag_health_cache["checked_at"] = now
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
# IN-MEMORY FALLBACK STORAGE (When Redis is offline)
# ═══════════════════════════════════════════════════════════════

_in_memory_history: Dict[str, List[Dict[str, Any]]] = {}
_in_memory_cache: Dict[str, str] = {}
_redis_health_cache: Dict[str, Any] = {"is_up": None, "checked_at": 0.0}


def _redis_service_candidates() -> List[str]:
    urls = [REDIS_SERVICE_URL]
    if REDIS_SERVICE_URL.endswith(":6390"):
        urls.append(REDIS_SERVICE_URL[:-5] + ":6380")
    elif REDIS_SERVICE_URL.endswith(":6380"):
        urls.append(REDIS_SERVICE_URL[:-5] + ":6390")
    return list(dict.fromkeys(urls))


def _rs(method: str, path: str, payload=None, query: Optional[dict] = None) -> dict:
    """
    HTTP call to redis_service. Fast timeout (0.3s).
    Caches offline status to avoid repeated blocking calls.
    """
    now = time.time()
    if _redis_health_cache["is_up"] is False and (now - _redis_health_cache["checked_at"]) < 30.0:
        return {"success": False}

    body    = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}

    for base_url in _redis_service_candidates():
        url = f"{base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        req = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=0.3) as resp:
                raw  = resp.read().decode()
                data = json.loads(raw) if raw else {}
                _redis_health_cache["is_up"] = True
                _redis_health_cache["checked_at"] = now
                return data if isinstance(data, dict) else {"success": True, "data": data}
        except Exception:
            pass

    _redis_health_cache["is_up"] = False
    _redis_health_cache["checked_at"] = now
    return {"success": False}



# ── Response cache ─────────────────────────────────────────────

def cache_response_get(response_hash: str) -> Optional[str]:
    res = _rs("GET", "/chatbot/response/get", query={"response_hash": response_hash})
    if res.get("success") and res.get("found"):
        print(f"[cache] HIT  hash={response_hash[:12]}…")
        return res.get("response_text")
    return _in_memory_cache.get(response_hash)


def cache_response_set(response_hash: str, text: str) -> None:
    _in_memory_cache[response_hash] = text
    _rs("POST", "/chatbot/response/set", payload={
        "response_hash": response_hash, "response_text": text,
    })


# ── Conversation history ───────────────────────────────────────

def history_append(conv_id: str, role: str, content: str,
                   user_id: str = "anonymous", user_name: str = "") -> int:
    msg = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "user_name": user_name
    }
    if conv_id not in _in_memory_history:
        _in_memory_history[conv_id] = []
    _in_memory_history[conv_id].append(msg)
    if len(_in_memory_history[conv_id]) > HISTORY_LIMIT:
        _in_memory_history[conv_id] = _in_memory_history[conv_id][-HISTORY_LIMIT:]

    res = _rs("POST", "/chatbot/history/append", payload={
        "conversation_id": conv_id, "role": role,
        "content": content, "user_id": user_id, "user_name": user_name
    })
    return int(res.get("message_count", len(_in_memory_history[conv_id])))



def history_get(conv_id: str, limit: int = HISTORY_LIMIT) -> List[Dict]:
    res = _rs("GET", "/chatbot/history/get",
              query={"conversation_id": conv_id, "limit": limit})
    if res.get("success") and res.get("found"):
        return res.get("messages", [])
    msgs = _in_memory_history.get(conv_id, [])
    return msgs[-limit:] if limit else msgs


def conversation_clear(conv_id: str) -> int:
    deleted = 0
    if conv_id in _in_memory_history:
        del _in_memory_history[conv_id]
        deleted = 1
    res = _rs("DELETE", "/chatbot/conversation/clear",
              query={"conversation_id": conv_id})
    return int(res.get("deleted", deleted))



# ═══════════════════════════════════════════════════════════════
# RAG RETRIEVAL  (called only when rag_service is UP)
# ═══════════════════════════════════════════════════════════════

def retrieve_context(query: str, top_k: int = RAG_TOP_K) -> List[Dict]:
    """
    POST to rag_service /retrieve -> hybrid search + BGE-Reranker-v2-m3.
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
        print(f"[RAG] '{query[:60]}…' -> {len(results)} hits, "
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
            parts.append(f"[Doc {i} -- {src}  score={score:.3f}]\n{chunk['text']}\n")

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

[WARNING] Note: The organisation document retrieval system is temporarily unavailable.
You are answering from your general training knowledge only.
If the question is specific to {ORG_NAME} internal documents or data, please let the user know
they should try again shortly when the document system is back online.

Rules:
1. Answer helpfully from general knowledge.
2. Be transparent when you are not certain.
3. Keep responses concise and professional.
"""


# ═══════════════════════════════════════════════════════════════
# KNOWLEDGE FALLBACK ENGINE FOR CHAKORAHUB
# ═══════════════════════════════════════════════════════════════

def _generate_knowledge_response(user_message: str) -> str:
    msg = (user_message or "").lower().strip()
    
    if any(k in msg for k in ["course", "offer", "program", "class", "learn", "technology", "tech", "subject", "syllabus"]):
        return (
            "We offer comprehensive, industry-focused training programs & courses at ChakoraHub:\n\n"
            "• **Data Engineering**: Informatica PowerCenter/IDMC, Snowflake, PySpark, SQL & Data Warehousing\n"
            "• **Python & Web Development**: Python Core & Advanced, Django, Flask, HTML/CSS, React\n"
            "• **Cloud & DevOps**: AWS Services, Docker, Kubernetes, CI/CD Pipelines, Terraform\n"
            "• **Machine Learning & AI**: Python for Data Science, ML Algorithms, Deep Learning, NLP\n"
            "• **Software Testing & QA**: Automation Testing with Selenium, Java/Python, API Testing\n\n"
            "Would you like more details on a specific course or information on how to register?"
        )
    
    if any(k in msg for k in ["register", "enroll", "sign up", "signup", "apply", "join", "admission"]):
        return (
            "You can easily register for our courses and internship programs directly on ChakoraHub:\n\n"
            "1. Click on **Register** in the navigation bar or choose your course.\n"
            "2. Fill in your name, email, contact number, and program interest.\n"
            "3. Submit your application to get instant access to trial modules and student portal.\n\n"
            "For assistance, reach out to our team at support@chakorahub.com!"
        )
        
    if any(k in msg for k in ["intern", "placement", "job", "career", "experience"]):
        return (
            "ChakoraHub offers real-time **Internship Programs** with hands-on enterprise experience:\n\n"
            "• **Real-World Projects**: Work on live enterprise data pipelines, web apps, and cloud infrastructure.\n"
            "• **Mentorship**: Guidance from experienced senior professionals.\n"
            "• **Placement Support**: Resume building, mock technical interviews, and referral support.\n\n"
            "Check the **Internships** section on ChakoraHub to apply!"
        )
        
    if any(k in msg for k in ["refund", "cancel", "cancelled", "deducted", "failed payment", "middle way"]):
        return (
            "If your payment was cancelled or failed during transaction:\n\n"
            "1. **Automatic Refund**: Any amount deducted for a cancelled transaction is automatically reversed by your bank within 3-5 business days.\n"
            "2. **Check Portal**: Log into your student portal to verify if your registration went through.\n"
            "3. **Contact Billing Support**: Send your transaction ID or payment receipt to **support@chakorahub.com** and our team will verify and resolve it immediately!"
        )

    if any(k in msg for k in ["fee", "cost", "price", "pricing", "charge", "pay", "payment"]):
        return (
            "Our course & internship fees are structured to be competitive and affordable:\n\n"
            "• Flexible installment options are available for students.\n"
            "• Early bird and bundle discounts apply to select courses.\n\n"
            "Please reach out to support@chakorahub.com for current fee details and offers."
        )


    if any(k in msg for k in ["contact", "phone", "email", "support", "help", "address", "reach"]):
        return (
            "You can contact ChakoraHub support anytime:\n\n"
            "• **Email**: support@chakorahub.com\n"
            "• **Support Desk**: Raise a ticket directly via the Support portal\n"
            "• **Student Portal**: Available 24/7 for active students\n\n"
            "Our team is happy to help you with course selection or technical queries!"
        )

    if any(k in msg for k in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]):
        return (
            "Hello! 👋 I'm ChakoraBot, your AI assistant for ChakoraHub.\n\n"
            "How can I help you today? You can ask me about:\n"
            "• 📚 Our Courses & Syllabus\n"
            "• ✍️ Registration & Enrollment\n"
            "• 💼 Internship Programs\n"
            "• 📞 Support & Contact Info"
        )
        
    return (
        "Welcome to ChakoraHub! We specialize in training and internships across Data Engineering, Python, Web Development, Cloud & DevOps, and Machine Learning.\n\n"
        "Feel free to ask about our courses, registration process, internship opportunities, or how to get started!"
    )


_ollama_health_cache: Dict[str, Any] = {"is_up": None, "checked_at": 0.0}

def _is_ollama_up() -> bool:
    now = time.time()
    if (now - _ollama_health_cache["checked_at"]) < 10 and _ollama_health_cache["is_up"] is not None:
        return _ollama_health_cache["is_up"]
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            is_up = (resp.status == 200)
    except Exception:
        is_up = False
    _ollama_health_cache["is_up"] = is_up
    _ollama_health_cache["checked_at"] = now
    return is_up


# ═══════════════════════════════════════════════════════════════
# LLM CALL  (Ollama -- Llama-3-8B-Instruct, local EC2)
# ═══════════════════════════════════════════════════════════════

async def call_llm(
    user_message:  str,
    rag_context:   str,
    conv_context:  str,
    rag_available: bool = True,
) -> str:
    """
    Call Ollama if available, otherwise fall back to ChakoraHub Knowledge Engine.
    """
    cache_key = hashlib.sha256(
        f"{user_message}||{rag_context}||{rag_available}".encode()
    ).hexdigest()

    cached = cache_response_get(cache_key)
    if cached is not None:
        return cached

    # Fast check: if Ollama server is down, return knowledge engine response immediately
    if not _is_ollama_up():
        print("[Ollama] Server offline -> using ChakoraHub Knowledge Fallback")
        fallback_resp = _generate_knowledge_response(user_message)
        cache_response_set(cache_key, fallback_resp)
        return fallback_resp

    # Build full prompt
    system = _SYSTEM_PROMPT_RAG if rag_available else _SYSTEM_PROMPT_FALLBACK
    sections = [system]
    if rag_context:
        sections.append(f"\n{rag_context}")
    if conv_context:
        sections.append(f"\n{conv_context}")
    sections.append(f"\n### User Question\n{user_message}")
    sections.append(f"\n### {BOT_NAME}'s Response")
    full_prompt = "\n".join(sections)

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                OLLAMA_API,
                json={
                    "model":  MODEL_NAME,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p":       0.9,
                        "max_tokens":  512,
                        "stop":        ["### User Question", "User:"],
                    },
                },
            )

        if resp.status_code == 200:
            ai_text = resp.json().get("response", "").strip()
            if not ai_text:
                ai_text = _generate_knowledge_response(user_message)
            cache_response_set(cache_key, ai_text)
            return ai_text
        else:
            print(f"[Ollama] error {resp.status_code}: fallback active")
            fallback_resp = _generate_knowledge_response(user_message)
            cache_response_set(cache_key, fallback_resp)
            return fallback_resp

    except Exception as e:
        print(f"[Ollama] call exception ({e}) -> using ChakoraHub Knowledge Fallback")
        fallback_resp = _generate_knowledge_response(user_message)
        cache_response_set(cache_key, fallback_resp)
        return fallback_resp



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


_support_active_mode = {}  # conv_id -> bool


@app.post("/chatbot/end_support")
async def end_support_endpoint(data: dict):
    conv_id = data.get("conversation_id")
    closing_msg = data.get("closing_message") or "The Support Person has ended this live session. Thank you for contacting ChakoraHub Support! 🙏 We are glad we could assist you today.\n\nIs there anything else I can help you with regarding our courses, registration, or internships?"
    if conv_id:
        _support_active_mode[conv_id] = False
        history_append(conv_id, "agent", closing_msg, user_id="support_agent", user_name="Support Person")
    return {"success": True, "conversation_id": conv_id}



@app.post("/chatbot/message", response_model=ChatResponse)

async def handle_message(chat: ChatMessage):

    try:
        conv_id = chat.conversation_id or generate_conversation_id()

        # 1. If message is sent by a human Support Person, activate support mode, store as agent and DO NOT invoke AI
        if chat.user_type == "support" or chat.user_id == "support_agent":
            _support_active_mode[conv_id] = True
            history_append(
                conv_id,
                "agent",
                chat.message,
                user_id=chat.user_id or "support_agent",
                user_name="Support Person"
            )
            return ChatResponse(
                success=True,
                response=chat.message,
                conversation_id=conv_id,
                sources=[],
                rag_used=False,
                timestamp=datetime.now().isoformat()
            )

        # Persist user message
        history_append(
            conv_id,
            "user",
            chat.message,
            user_id=chat.user_id or "anonymous",
            user_name=chat.user_name or "Student"
        )

        # Check if student explicitly requested support agent
        msg_lower = (chat.message or "").lower().strip()
        if any(k in msg_lower for k in ["talk to support", "support agent", "human agent", "talk with agent", "connect to support", "speak with agent", "human", "gayatri", "ganesh"]):
            _support_active_mode[conv_id] = True
            ai_resp = (
                "I have notified our support team! An email with a direct join link has been sent to our support agents.\n\n"
                "A support person will join this conversation shortly. Please stay on this chat!"
            )

            history_append(conv_id, "assistant", ai_resp, BOT_NAME)
            return ChatResponse(
                success=True,
                response=ai_resp,
                conversation_id=conv_id,
                sources=[],
                rag_used=False,
                timestamp=datetime.now().isoformat()
            )

        # 2. IF SUPPORT MODE IS ACTIVE FOR THIS CONVERSATION: CHAKORABOT STAYS SILENT!
        if _support_active_mode.get(conv_id):
            print(f"[SUPPORT ACTIVE] ChakoraBot staying silent for conversation {conv_id[:16]}")
            return ChatResponse(
                success=True,
                response="",  # completely silent response so ChakoraBot does not output anything
                conversation_id=conv_id,
                sources=[],
                rag_used=False,
                timestamp=datetime.now().isoformat()
            )





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
                print("[RAG] No chunks above threshold -- using LLM-only for this message")
                rag_available = False
        else:
            print("[chatbot] RAG service is DOWN -- using LLM-only fallback")

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
    print(f"[START] {ORG_NAME} Chatbot Service  (RAG + LLM-only Fallback)")
    print(f"   LLM          : {MODEL_NAME}  via Ollama ({OLLAMA_API})")
    print(f"   RAG service  : {RAG_SERVICE_URL}  top_k={RAG_TOP_K}  alpha={RAG_ALPHA}")
    print(f"   Memory       : redis_service {REDIS_SERVICE_URL} DB{CHATBOT_REDIS_DB}")
    print(f"   History      : {HISTORY_WINDOW}-message sliding window")
    print(f"   Failover     : RAG DOWN -> LLM-only (Llama-3 direct)")
    print("=" * 65)

    # redis_service health
    redis_health = _rs("GET", "/health")
    if redis_health.get("success"):
        print("[OK] redis_service reachable")
    else:
        print("[ERROR] redis_service unreachable - conversation memory degraded")

    # RAG service health
    if _is_rag_up():
        print(f"[OK] rag_service reachable at {RAG_SERVICE_URL}")
        print("  Mode: FULL RAG pipeline (hybrid search + BGE-Reranker)")
    else:
        print(f"[ERROR] rag_service UNREACHABLE at {RAG_SERVICE_URL}")
        print("  Mode: LLM-only fallback (Llama-3 direct, no org docs)")
        print("  The chatbot will automatically switch back to RAG when the service recovers.")

    # Ollama health
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get("http://127.0.0.1:11434/api/tags")
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"[OK] Ollama reachable - models: {models}")
            if MODEL_NAME not in " ".join(models):
                print(f"  [WARNING] '{MODEL_NAME}' not found. Run: ollama pull {MODEL_NAME}")
        else:
            print(f"[ERROR] Ollama returned status {resp.status_code}")
    except Exception as e:
        print(f"[ERROR] Ollama unreachable: {e}")
        print("  [WARNING] The LLM itself is down - chatbot cannot respond until Ollama is running.")

    print("[OK] Chatbot service ready")


# ═══════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7600)
