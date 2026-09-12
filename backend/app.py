import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .harness.types import (
    Phase, SOPState, ChatRequest, ChatResponse, ClaimRecord, PolicyHolder
)
from .harness.state_machine import SOPStateMachine
from .engine.llm_engine import LLMEngine
from .engine.mock_engine import MockEngine

app = FastAPI(title="Insurance Claims SOP Agent", version="1.0.0")

# In-memory store for sessions
active_sessions: Dict[str, SOPStateMachine] = {}
session_histories: Dict[str, List[Dict[str, str]]] = {}

# Global Engine
engine = LLMEngine()
use_mock_only = False

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

class ConfigRequest(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    use_mock: Optional[bool] = None

class ResetRequest(BaseModel):
    session_id: Optional[str] = None

@app.post("/api/config")
async def update_engine_config(req: ConfigRequest):
    global use_mock_only
    if req.use_mock is not None:
        use_mock_only = req.use_mock
    engine.update_config(api_key=req.api_key, base_url=req.base_url, model=req.model)
    return {
        "status": "success",
        "model": engine.model,
        "base_url": engine.base_url,
        "has_api_key": bool(engine.api_key),
        "use_mock_only": use_mock_only
    }

@app.get("/api/config")
async def get_engine_config():
    return {
        "model": engine.model,
        "base_url": engine.base_url,
        "has_api_key": bool(engine.api_key),
        "use_mock_only": use_mock_only
    }

@app.post("/api/reset")
async def reset_session(req: ResetRequest):
    sid = req.session_id or str(uuid.uuid4())
    active_sessions[sid] = SOPStateMachine(sid)
    session_histories[sid] = []
    return {"session_id": sid, "status": "reset", "phase": Phase.VERIFY_ID}

@app.get("/api/scenarios")
async def get_demo_scenarios():
    return [
        {
            "id": "margaret_chen_demo",
            "title": "Insurance Demo Test Case (Margaret Chen)",
            "description": "Caller provides identity info and asks about denied January healthcare claim in one shot.",
            "messages": [
                "I’m the policyholder. My name is Margaret Chen, policy POL-9921. I’m calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."
            ]
        },
        {
            "id": "emotional_refusal",
            "title": "Emotional Caller & SOP Recovery",
            "description": "Caller is frustrated and refuses verification. Agent de-escalates with empathy, explains privacy, offers alternative ID fields.",
            "messages": [
                "I already told you who I am. This is ridiculous. Just tell me why my claim was denied."
            ]
        },
        {
            "id": "out_of_scope_test",
            "title": "Out-of-Scope Guardrail",
            "description": "Caller asks irrelevant questions ('What is RL?'). Agent rejects politely and offers human escalation on repeat.",
            "messages": [
                "What is RL?",
                "Can you explain Q-learning to me?"
            ]
        },
        {
            "id": "step_by_step_pii",
            "title": "Progressive Partial Verification",
            "description": "Caller supplies name first, asks why, then supplies phone and email as alternate fields.",
            "messages": [
                "Hi, my name is Ava Lopez.",
                "Why do you need my info? Fine, my phone is +16503882920 and email is ava.lopez@email.com."
            ]
        }
    ]

@app.get("/api/state/{session_id}")
async def get_session_state(session_id: str):
    sm = active_sessions.get(session_id)
    if not sm:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "phase": sm.state.phase,
        "accumulated_pii": sm.state.accumulated_pii,
        "verified_fields": sm.state.verified_fields,
        "cross_phase_memory": sm.state.cross_phase_memory,
        "active_case_id": sm.state.active_case_id,
        "trace_log": sm.state.trace_log,
        "post_process": sm.state.post_process
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    sid = req.session_id
    if sid not in active_sessions:
        active_sessions[sid] = SOPStateMachine(sid)
        session_histories[sid] = []

    sm = active_sessions[sid]
    history = session_histories[sid]

    # Evaluate state machine turn
    state_result = sm.evaluate_turn(req.message)

    # Choose generation engine
    active_engine = MockEngine() if use_mock_only else engine
    reply = active_engine.generate_response(req.message, state_result, history)

    # Record history
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": reply})

    return ChatResponse(
        session_id=sid,
        reply=reply,
        current_phase=sm.state.phase,
        sop_state=sm.state,
        trace=sm.state.trace_log,
        active_case=sm.get_active_claim(),
        verified_policyholder=sm.get_verified_policyholder()
    )

# Mount frontend files if directory exists
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")
