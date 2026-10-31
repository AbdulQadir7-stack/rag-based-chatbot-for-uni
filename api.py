# api.py - FastAPI backend with JWT auth, chat, feedback, admin

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
import jwt
import bcrypt
import datetime
import uuid
import json
import os
from typing import Optional
from dotenv import load_dotenv

from agents import (
    process_query,
    prepare_query,
    stream_answer,
    semantic_route,
    is_valid_query,
    get_quick_questions,
    clear_retrieval_caches,
    LOW_CONFIDENCE_MESSAGE,
    MIN_CONFIDENCE_SCORE,
)
from database import (
    init_db,
    add_log,
    get_user,
    get_logs_for_admin,
    save_feedback,
    create_user,
    get_all_users,
    get_recent_memory,
    save_memory,
    get_admin_analytics,
)

try:
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi import _rate_limit_exceeded_handler
except ImportError:
    Limiter = None
    RateLimitExceeded = None
    get_remote_address = None
    SlowAPIMiddleware = None
    _rate_limit_exceeded_handler = None

# ─── App setup ─────────────────────────────────────────────────────────────────
load_dotenv()

app = FastAPI(title="AJKU AI Assistant", version="3.2")

if Limiter:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
else:
    class NoopLimiter:
        def limit(self, _rule):
            def decorator(func):
                return func
            return decorator

    limiter = NoopLimiter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is missing. Add it to your .env file.")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 8

security = HTTPBearer(auto_error=False)


# ─── Token helpers ─────────────────────────────────────────────────────────────
def create_token(user_id: int, role: str, username: str) -> str:
    payload = {
        "user_id": user_id,
        "role": role,
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired. Please login again.")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token.")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    if not credentials:
        raise HTTPException(401, "Authorization header missing.")
    return decode_token(credentials.credentials)


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required.")
    return user


# ─── Request / Response models ─────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    token: str
    role: str
    user_id: int
    username: str

class ChatRequest(BaseModel):
    question: str
    history: Optional[str] = ""

class ChatResponse(BaseModel):
    answer: str
    route: str
    collection: str
    query_id: str
    chunks_found: int = 0
    confidence_score: int = 0
    corrected_query: str = ""
    was_corrected: bool = False
    sources: list[dict] = Field(default_factory=list)
    follow_up_suggestions: list[str] = Field(default_factory=list)
    debug_info: Optional[dict] = None

class FeedbackRequest(BaseModel):
    query_id: str
    thumbs_up: bool

class RegisterRequest(BaseModel):
    username: str
    password: str
    role: Optional[str] = "student"


def build_debug_info(result: dict, user: dict) -> dict | None:
    if user.get("role") != "admin":
        return None

    docs = result.get("retrieved_docs") or result.get("docs") or []
    return {
        "collection": result.get("collection", "none"),
        "confidence_score": result.get("confidence_score", 0),
        "top_rerank_score": result.get("top_rerank_score", 0.0),
        "rerank_method": result.get("rerank_method", ""),
        "top_documents": [
            {
                "doc_id": doc.get("id", ""),
                "collection": (doc.get("metadata") or {}).get("collection", ""),
                "rerank_score": round(float(doc.get("rerank_score", 0.0)), 3),
                "rerank_method": doc.get("rerank_method", ""),
                "question": (doc.get("metadata") or {}).get("question", ""),
            }
            for doc in docs[:5]
        ],
    }


# ─── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()


# ─── Auth endpoints ─────────────────────────────────────────────────────────────
@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    user = get_user(req.username)
    if not user:
        raise HTTPException(401, "User not found.")
    if not bcrypt.checkpw(req.password.encode(), user["password"]):
        raise HTTPException(401, "Wrong password.")
    token = create_token(user["id"], user["role"], user["username"])
    return LoginResponse(token=token, role=user["role"], user_id=user["id"], username=user["username"])


@app.post("/register")
def register(req: RegisterRequest, user: dict = Depends(require_admin)):
    """Admin only: create new user"""
    success = create_user(req.username, req.password, req.role)
    if not success:
        raise HTTPException(400, "Username already exists.")
    return {"message": f"User '{req.username}' created with role '{req.role}'."}


# ─── Chat endpoint ─────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(request: Request, req: ChatRequest, user: dict = Depends(get_current_user)):
    query_id = uuid.uuid4().hex
    allowed, reason = is_valid_query(req.question)
    if not allowed:
        add_log(
            user["user_id"],
            user["username"],
            req.question,
            reason,
            "blocked",
            0,
            query_id=query_id,
            collection="none",
        )
        return ChatResponse(
            answer="Please ask AJKU-related questions only.",
            route="blocked",
            collection="none",
            query_id=query_id,
            chunks_found=0,
            confidence_score=0,
            corrected_query=req.question,
            was_corrected=False,
            sources=[],
            follow_up_suggestions=get_quick_questions(3),
        )

    stored_history = get_recent_memory(user["user_id"])
    combined_history = "\n".join(
        part for part in [stored_history, req.history or ""] if part.strip()
    )

    result = process_query(req.question, user["user_id"], combined_history)
    route = semantic_route(result.get("corrected_query", req.question))

    add_log(
        user["user_id"],
        user["username"],
        req.question,
        result["answer"],
        route,
        1,
        query_id=query_id,
        collection=result["collection"],
        chunks_found=result.get("chunks_found", 0),
        top_rerank_score=result.get("top_rerank_score", 0.0),
        rerank_method=result.get("rerank_method", ""),
    )
    save_memory(user["user_id"], "user", req.question)
    save_memory(user["user_id"], "assistant", result["answer"])

    return ChatResponse(
        answer=result["answer"],
        route=route,
        collection=result["collection"],
        query_id=query_id,
        chunks_found=result.get("chunks_found", 0),
        confidence_score=result.get("confidence_score", 0),
        corrected_query=result.get("corrected_query", req.question),
        was_corrected=result.get("was_corrected", False),
        sources=result.get("sources", []),
        follow_up_suggestions=result.get("follow_up_suggestions", []),
        debug_info=build_debug_info(result, user),
    )


@app.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(request: Request, req: ChatRequest, user: dict = Depends(get_current_user)):
    query_id = uuid.uuid4().hex
    allowed, reason = is_valid_query(req.question)

    def event(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False) + "\n"

    if not allowed:
        add_log(
            user["user_id"], user["username"], req.question, reason, "blocked", 0,
            query_id=query_id, collection="none"
        )

        def blocked_stream():
            yield event({
                "type": "final",
                "answer": "Please ask AJKU-related questions only.",
                "route": "blocked",
                "collection": "none",
                "query_id": query_id,
                "chunks_found": 0,
                "confidence_score": 0,
                "corrected_query": req.question,
                "was_corrected": False,
                "sources": [],
                "follow_up_suggestions": get_quick_questions(3),
            })

        return StreamingResponse(blocked_stream(), media_type="application/x-ndjson")

    stored_history = get_recent_memory(user["user_id"])
    combined_history = "\n".join(
        part for part in [stored_history, req.history or ""] if part.strip()
    )
    prepared = prepare_query(req.question, combined_history)
    route = semantic_route(prepared["corrected_query"])

    def token_stream():
        debug_info = build_debug_info(
            {
                "collection": prepared["collection"],
                "confidence_score": prepared["confidence_score"],
                "top_rerank_score": prepared["docs"][0].get("rerank_score", 0.0) if prepared["docs"] else 0.0,
                "rerank_method": prepared["docs"][0].get("rerank_method", "") if prepared["docs"] else "",
                "docs": prepared["docs"],
            },
            user,
        )
        yield event({
            "type": "meta",
            "route": route,
            "collection": prepared["collection"],
            "query_id": query_id,
            "chunks_found": len(prepared["chunks"]),
            "confidence_score": prepared["confidence_score"],
            "corrected_query": prepared["corrected_query"],
            "was_corrected": prepared["was_corrected"],
            "sources": prepared["sources"],
            "follow_up_suggestions": prepared["follow_up_suggestions"],
            "debug_info": debug_info,
        })

        answer = ""
        if prepared["confidence_score"] < MIN_CONFIDENCE_SCORE:
            answer = LOW_CONFIDENCE_MESSAGE
            yield event({"type": "token", "text": answer})
        else:
            answer_parts = []
            for token in stream_answer(
                prepared["corrected_query"],
                prepared["chunks"],
                combined_history,
            ):
                answer_parts.append(token)
                yield event({"type": "token", "text": token})
            answer = "".join(answer_parts)

        add_log(
            user["user_id"],
            user["username"],
            req.question,
            answer,
            route,
            1,
            query_id=query_id,
            collection=prepared["collection"],
            chunks_found=len(prepared["chunks"]),
            top_rerank_score=prepared["docs"][0].get("rerank_score", 0.0) if prepared["docs"] else 0.0,
            rerank_method=prepared["docs"][0].get("rerank_method", "") if prepared["docs"] else "",
        )
        save_memory(user["user_id"], "user", req.question)
        save_memory(user["user_id"], "assistant", answer)

        yield event({
            "type": "final",
            "answer": answer,
            "route": route,
            "collection": prepared["collection"],
            "query_id": query_id,
            "chunks_found": len(prepared["chunks"]),
            "confidence_score": prepared["confidence_score"],
            "corrected_query": prepared["corrected_query"],
            "was_corrected": prepared["was_corrected"],
            "sources": prepared["sources"],
            "follow_up_suggestions": prepared["follow_up_suggestions"],
            "debug_info": debug_info,
        })

    return StreamingResponse(token_stream(), media_type="application/x-ndjson")


@app.get("/quick-questions")
def quick_questions(user: dict = Depends(get_current_user)):
    return {"questions": get_quick_questions(3)}


# ─── Feedback endpoint ─────────────────────────────────────────────────────────
@app.post("/feedback")
def feedback(req: FeedbackRequest, user: dict = Depends(get_current_user)):
    save_feedback(user["user_id"], req.query_id, req.thumbs_up)
    return {"status": "ok"}


# ─── Admin endpoints ───────────────────────────────────────────────────────────
@app.get("/admin/stats")
def admin_stats(user: dict = Depends(require_admin)):
    return get_admin_analytics()


@app.get("/admin/analytics")
def admin_analytics(user: dict = Depends(require_admin)):
    return get_admin_analytics()


@app.get("/admin/logs")
def admin_logs(limit: int = 100, user: dict = Depends(require_admin)):
    return get_logs_for_admin(limit)


@app.get("/admin/users")
def admin_users(user: dict = Depends(require_admin)):
    return get_all_users()


@app.post("/admin/reload-data")
def admin_reload_data(user: dict = Depends(require_admin)):
    clear_retrieval_caches()
    return {"status": "ok", "message": "Retrieval caches cleared. Re-run populate_multicol.py if source files changed."}


# ─── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "version": "3.2"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
