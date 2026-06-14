from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agent.clinic_agent import run_agent

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Clinic AI Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[list[Message]] = []
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str]
    timestamp: str

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "Clinic AI Assistant"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        history = [
            {"role": m.role, "content": m.content}
            for m in request.history
        ]

        reply = run_agent(
            message=request.message,
            history=history
        )

        return ChatResponse(
            reply=reply,
            session_id=request.session_id,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))