import logging
import time
import uuid

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.auth import get_current_user, require_role
from app.agent import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Support Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    latency_ms: float


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: dict = Depends(get_current_user),
):
    require_role(user, ["agent", "admin"])
    thread_id = request.thread_id or str(uuid.uuid4())

    start = time.monotonic()
    try:
        reply = await run_agent(request.message, thread_id)
    except Exception as e:
        logger.exception("Agent error for thread %s", thread_id)
        raise HTTPException(status_code=500, detail="Agent failed to process request")
    elapsed_ms = (time.monotonic() - start) * 1000

    logger.info("POST /chat latency=%.1fms thread=%s user=%s", elapsed_ms, thread_id, user.get("preferred_username"))
    return ChatResponse(reply=reply, thread_id=thread_id, latency_ms=round(elapsed_ms, 1))
