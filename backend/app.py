"""Local demo API. Each opaque session owns its state, model settings and turn lock."""
import asyncio
import ipaddress
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .harness.types import Phase, SOPState
from .harness.state_machine import SOPStateMachine
from .harness.grounded_data import grounded_data
from .engine.llm_engine import LLMEngine
from .engine.mock_engine import MockEngine

app = FastAPI(title='Insurance Claims SOP Harness', version='2.0.0')
FRONTEND_DIR = Path(__file__).resolve().parent.parent / 'frontend'
MAX_SESSIONS, MAX_TURNS, SESSION_TTL = 256, 120, 7200


@dataclass
class Session:
    machine: SOPStateMachine
    engine: LLMEngine = field(default_factory=LLMEngine)
    use_mock: bool = False
    history: list = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    touched: float = field(default_factory=time.monotonic)


sessions: dict[str, Session] = {}


class Payload(BaseModel):
    model_config = ConfigDict(extra='forbid')


class SessionRequest(Payload):
    session_id: str = Field(min_length=32, max_length=80)


class ConfigRequest(SessionRequest):
    api_key: Optional[str] = Field(default=None, max_length=1024)
    base_url: Optional[str] = Field(default=None, max_length=500)
    model: Optional[str] = Field(default=None, min_length=1, max_length=150)
    use_mock: Optional[bool] = None


class ResetRequest(Payload):
    session_id: Optional[str] = None


class RestoreRequest(SessionRequest):
    turn_index: int = Field(ge=0)


class ChatRequest(SessionRequest):
    message: str = Field(min_length=1, max_length=6000)

    @field_validator('message')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('Message cannot be blank')
        return value.strip()


class VerifyCardRequest(SessionRequest):
    name: Optional[str] = Field(default=None, max_length=200)
    dob: Optional[str] = Field(default=None, max_length=100)
    phone: Optional[str] = Field(default=None, max_length=50)
    email: Optional[str] = Field(default=None, max_length=200)
    id_last4: Optional[str] = Field(default=None, max_length=10)
    id_type: Optional[str] = Field(default=None, max_length=50)




def get_session(sid):
    item = sessions.get(sid)
    if not item or time.monotonic() - item.touched > SESSION_TTL:
        if item and not item.lock.locked():
            sessions.pop(sid, None)
        raise HTTPException(404, 'Session expired or not found. Start a new call.')
    item.touched = time.monotonic()
    return item


def config_view(item):
    return {'model': item.engine.model, 'base_url': item.engine.base_url,
            'has_api_key': bool(item.engine.api_key), 'use_mock_only': item.use_mock,
            'engine_mode': 'mock' if item.use_mock or not item.engine.api_key else 'live'}


def redact_text(value):
    if not isinstance(value, str):
        return value
    for holder in grounded_data.policyholders:
        for field_name in ['name', 'dob', 'email', 'phone', 'id_last4', 'policy_number']:
            value = re.sub(re.escape(getattr(holder, field_name)), '[redacted]', value, flags=re.I)
    value = re.sub(r'\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b', '[email redacted]', value)
    value = re.sub(r'(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)', '[phone redacted]', value)
    value = re.sub(r'\b(?:my name is|i am|this is)\s+[A-Za-z]+(?:\s+[A-Za-z]+){1,2}', '[name redacted]', value, flags=re.I)
    value = re.sub(r'\b(?:dob|date of birth|ssn|social security|national id)\b\s*(?:is|:)?\s*\d[\d/ -]*', '[identity redacted]', value, flags=re.I)
    return value


def redact_tree(data):
    if isinstance(data, str):
        return redact_text(data)
    if isinstance(data, list):
        return [redact_tree(v) for v in data]
    if isinstance(data, dict):
        return {k: redact_tree(v) for k, v in data.items()}
    return data


def public_state(sm):
    state = sm.state
    authorized = sm.get_verified_policyholder() is not None
    output = state.model_dump(mode='json', exclude={'accumulated_pii', 'history_snapshots', 'mock_outbox'})
    output['identity_verified'] = authorized
    output['data_shield_active'] = not authorized
    output['collected_fields'] = [k for k in ['name', 'dob', 'phone', 'email', 'id_last4'] if getattr(state.accumulated_pii, k)]
    if not authorized:
        output['active_case_id'] = None
        output['post_process'] = {'email_offered': False, 'user_decision': None, 'delivery_status': None}
    output['post_process'].pop('sent_to', None)
    output['history_snapshots'] = []
    output['outbox_count'] = len(getattr(state, 'mock_outbox', [])) if authorized else 0
    return redact_tree(output)


def reply_payload(sid, item, reply=None):
    public = public_state(item.machine)
    claim = item.machine.get_active_claim()
    return {'session_id': sid, 'reply': reply, 'current_phase': public['phase'],
            'sop_state': public, 'trace': public['trace_log'],
            'active_case': claim.model_dump(mode='json', exclude={'party_id'}) if claim else None,
            'verified_policyholder': None, 'engine_mode': item.engine.last_mode,
            'fallback_reason': item.engine.fallback_reason}


@app.middleware('http')
async def response_headers(request: Request, call_next):
    # Same-origin browser calls; the UI never stores secrets in browser persistence.
    if request.method == 'POST' and request.headers.get('origin'):
        origin = urlparse(request.headers['origin'])
        if origin.netloc != request.headers.get('host'):
            return JSONResponse({'detail': 'Cross-origin writes are not allowed'}, status_code=403)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Pydantic's default response includes rejected input, which may be a token.
    return JSONResponse({'detail': [{'loc': error['loc'], 'msg': error['msg'], 'type': error['type']}
                                     for error in exc.errors()]}, status_code=422)


@app.get('/api/health')
async def health():
    return {'status': 'ok', 'data': 'synthetic fixtures', 'email_delivery': 'simulated', 'human_transfer': 'simulated'}


@app.post('/api/reset')
async def reset_session(req: ResetRequest):
    if req.session_id:
        item = get_session(req.session_id)
        async with item.lock:
            item.machine = SOPStateMachine(req.session_id)
            item.history.clear()
        sid = req.session_id
    else:
        expired = [k for k, v in sessions.items() if time.monotonic() - v.touched > SESSION_TTL and not v.lock.locked()]
        for key in expired:
            del sessions[key]
        if len(sessions) >= MAX_SESSIONS:
            raise HTTPException(503, 'Demo session capacity reached; try again later.')
        sid = str(uuid.uuid4())
        sessions[sid] = Session(SOPStateMachine(sid))
    return {'session_id': sid, 'status': 'reset', 'phase': Phase.VERIFY_ID, 'sop_state': public_state(sessions[sid].machine)}


def validate_endpoint(value):
    parsed = urlparse(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
        raise HTTPException(422, 'Use an API base URL without credentials, query or fragment.')
    allow_local = os.getenv('AI_ALLOW_LOCAL_ENDPOINT') == '1'
    if parsed.scheme != 'https' and not (allow_local and parsed.scheme == 'http' and parsed.hostname in ['localhost', '127.0.0.1', '::1']):
        raise HTTPException(422, 'API endpoint must use HTTPS. Local HTTP requires AI_ALLOW_LOCAL_ENDPOINT=1.')
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80))
        if not allow_local and any(not ipaddress.ip_address(info[4][0]).is_global for info in addresses):
            raise HTTPException(422, 'Private network API endpoints require AI_ALLOW_LOCAL_ENDPOINT=1.')
    except socket.gaierror:
        raise HTTPException(422, 'API endpoint hostname could not be resolved.')


@app.post('/api/config')
async def update_config(req: ConfigRequest):
    item = get_session(req.session_id)
    async with item.lock:
        if req.base_url and req.base_url.rstrip('/') != item.engine.base_url:
            # Never redirect an existing secret to a newly chosen host.
            if not req.api_key:
                raise HTTPException(422, 'Enter the API token for the new endpoint when changing its URL.')
            await asyncio.to_thread(validate_endpoint, req.base_url)
        item.engine.update_config(req.api_key, req.base_url, req.model)
        if req.use_mock is not None:
            item.use_mock = req.use_mock
    return config_view(item)


@app.get('/api/config')
async def get_config(session_id: str):
    return config_view(get_session(session_id))


@app.get('/api/state/{session_id}')
async def state_endpoint(session_id: str):
    item = get_session(session_id)
    async with item.lock:
        return reply_payload(session_id, item)


def process_turn(item, message):
    sm = item.machine
    if item.use_mock or sm.state.phase in [Phase.CONCLUDED, Phase.ESCALATED]:
        semantic = {}
        item.engine.last_mode = 'mock' if item.use_mock or not item.engine.api_key else 'live'
        item.engine.fallback_reason = None
    else:
        semantic = item.engine.interpret(message, sm.state, item.history)
    result = sm.evaluate_turn(message, semantic=semantic)
    result['response_topics'] = semantic.get('response_topics')
    result['response_style'] = semantic.get('response_style', 'concise')
    reply = item.engine.generate_response(message, result, item.history)
    if hasattr(sm, 'record_grounded_topics'):
        sm.record_grounded_topics(result.get('grounded_topics', []))
    if sm.state.phase == Phase.POST_PROCESS:
        sm.state.post_process.draft_summary = sm.generate_draft_email(sm.get_verified_policyholder(), sm.get_active_claim())
    item.history.extend([{'role': 'user', 'content': message}, {'role': 'assistant', 'content': reply}])
    sm.record_snapshot(message, reply)
    return reply


@app.post('/api/chat')
async def chat(req: ChatRequest):
    item = get_session(req.session_id)
    async with item.lock:
        if len(item.history) // 2 >= MAX_TURNS:
            raise HTTPException(429, 'Demo turn limit reached. Start a new call.')
        # Network I/O and model processing do not block other sessions' event loop.
        reply = await asyncio.to_thread(process_turn, item, req.message)
        return reply_payload(req.session_id, item, reply)


@app.post('/api/verify-card')
async def verify_card(req: VerifyCardRequest):
    item = get_session(req.session_id)
    async with item.lock:
        if len(item.history) // 2 >= MAX_TURNS:
            raise HTTPException(429, 'Demo turn limit reached. Start a new call.')
        form_data = {
            "name": req.name or "",
            "dob": req.dob or "",
            "phone": req.phone or "",
            "email": req.email or "",
            "id_last4": req.id_last4 or "",
            "id_type": req.id_type or "",
        }
        res = item.machine.verify_card_data(form_data)
        reply = res["agent_reply"]
        item.history.extend([
            {'role': 'user', 'content': '[Submitted Security Verification Card]'},
            {'role': 'assistant', 'content': reply}
        ])
        return reply_payload(req.session_id, item, reply)



@app.post('/api/restore')
async def restore(req: RestoreRequest):
    item = get_session(req.session_id)
    async with item.lock:
        if not item.machine.restore_to_turn(req.turn_index):
            raise HTTPException(400, 'Invalid turn index')
        item.history = item.history[:(req.turn_index + 1) * 2]
        return {'status': 'restored', **reply_payload(req.session_id, item)}


@app.get('/api/trajectory/{session_id}')
async def trajectory(session_id: str):
    item = get_session(session_id)
    async with item.lock:
        snapshots = []
        for snapshot in item.machine.state.history_snapshots:
            temporary = SOPStateMachine(session_id)
            temporary.state = SOPState.model_validate(snapshot.state_dump)
            snapshots.append({'turn_index': snapshot.turn_index, 'phase': snapshot.phase,
                              'state_dump': public_state(temporary),
                              'user_message': '[caller text omitted; may contain PII]',
                              'agent_reply': redact_text(snapshot.agent_reply), 'timestamp': snapshot.timestamp})
        return {'session_id': session_id, 'turns_count': len(snapshots), 'snapshots': snapshots}


@app.get('/api/scenarios')
async def scenarios():
    return [
        {'id': 'margaret_chen_demo', 'title': 'Margaret’s January denial',
         'description': 'The requested one-turn identity and remembered-intent example.',
         'messages': ['I’m the policyholder. My name is Margaret Chen, policy POL-9921. I’m calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.']},
        {'id': 'emotional_refusal', 'title': 'Frustrated caller', 'description': 'Empathy while verification stays locked.',
         'messages': ['I already told you who I am. This is ridiculous. Just tell me why my claim was denied.']},
        {'id': 'step_by_step_pii', 'title': 'Partial identity', 'description': 'Name, then alternate phone and email.',
         'messages': ['Hi, my name is Ava Lopez.', 'My phone is +16503882920 and my email is ava.lopez@email.com.']},
        {'id': 'out_of_scope_test', 'title': 'Scope recovery', 'description': 'Repeated unrelated requests reach a simulated handoff.',
         'messages': ['What is RL?', 'Can you explain Q-learning to me?']},
        {'id': 'proxy_david_chen', 'title': 'Representative request', 'description': 'Representative identity and consent require human review.',
         'messages': ['I am David Chen, calling on behalf of my mother Margaret Chen, policy POL-9921. DOB is 1985-03-15, SSN last four is 4472. I am calling about her denied healthcare claim from January.']},
    ]


app.mount('/static', StaticFiles(directory=str(FRONTEND_DIR)), name='static')


@app.get('/')
async def index():
    return FileResponse(FRONTEND_DIR / 'index.html')
