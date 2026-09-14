"""Local demo API. Each opaque session owns its state, model settings and turn lock."""
import asyncio
import copy
import ipaddress
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
from dotenv import load_dotenv

load_dotenv()

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
from .harness.conversation_context import apply_model_reference, prepare_conversation_turn
from .harness.customer_policy import decide, record_execution
from .harness.customer_service import render_policy_reply
from .harness.extractor import UtteranceExtractor
from .lab import router as lab_router

app = FastAPI(title='Insurance Claims SOP Harness', version='2.0.0')
app.include_router(lab_router)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / 'frontend'
MAX_SESSIONS, MAX_TURNS, SESSION_TTL = 256, 120, 7200


@dataclass
class Session:
    machine: SOPStateMachine
    engine: LLMEngine = field(default_factory=LLMEngine)
    use_mock: bool = False
    controller: str = 'ppo42'
    policy_state: dict = field(default_factory=dict)
    policy_decision: Optional[dict] = None
    policy_snapshots: list = field(default_factory=list)
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
    controller: Optional[Literal['rule', 'ppo42', 'ppo7']] = None


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
            'engine_mode': 'mock' if item.use_mock or not item.engine.api_key else 'live',
            'controller': item.controller}


def public_policy_decision(item):
    return public_decision(item.policy_decision)


def public_decision(decision):
    if not decision:
        return None
    # Observation/raw speech and internal runtime memory are never API fields.
    fields = ('selected_action', 'probabilities', 'allowed_actions', 'action_mask',
              'source', 'checkpoint_sha256', 'checkpoint', 'fallback_reason',
              'reason', 'forced', 'grounded_answered', 'response_generation')
    return {key: decision[key] for key in fields if key in decision}


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


def public_state(sm, *, force_shield=False):
    state = sm.state
    authorized = not force_shield and sm.get_verified_policyholder() is not None
    output = state.model_dump(mode='json', exclude={'accumulated_pii', 'history_snapshots', 'mock_outbox'})
    output['identity_verified'] = authorized
    output['data_shield_active'] = not authorized
    output['collected_fields'] = [k for k in ['name', 'dob', 'phone', 'email', 'id_last4'] if getattr(state.accumulated_pii, k)]
    if not authorized:
        output['active_case_id'] = None
        output['verified_party_id'] = None
        output['post_process'] = {'email_offered': False, 'user_decision': None, 'delivery_status': None}
        for event in output['trace_log']:
            if not event.get('data_shield_active', True):
                event['details'] = 'Protected historical decision; claim details are currently locked.'
    output['post_process'].pop('sent_to', None)
    output['history_snapshots'] = []
    output['outbox_count'] = len(getattr(state, 'mock_outbox', [])) if authorized else 0
    return redact_tree(output)


def reply_payload(sid, item, reply=None):
    public = public_state(item.machine)
    claim = item.machine.get_active_claim()
    choices = []
    owner = item.machine.get_verified_policyholder()
    if owner and item.machine.state.phase == Phase.RESOLVE_INTENT:
        _, candidates = grounded_data.find_claim(owner.party_id, item.machine.state.cross_phase_memory)
        choices = [{'case_id': c.case_id, 'case_type': c.case_type, 'created_at': c.created_at}
                   for c in candidates[:6]]
    return {'session_id': sid, 'reply': reply, 'current_phase': public['phase'],
            'controller': item.controller, 'policy_decision': public_policy_decision(item),
            'claim_choices': choices,
            'sop_state': public, 'trace': public['trace_log'],
            'active_case': claim.model_dump(mode='json', exclude={'party_id'}) if claim else None,
            'verified_policyholder': None, 'engine_mode': item.engine.last_mode,
            'model_activity': dict(item.engine.model_activity),
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
            item.policy_state.clear()
            item.policy_decision = None
            item.policy_snapshots.clear()
            item.engine.reset_turn_activity()
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
        if req.controller is not None:
            item.controller = req.controller
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
    item.engine.reset_turn_activity()
    context = prepare_conversation_turn(sm, message, item.history)
    if item.use_mock or sm.state.phase in [Phase.CONCLUDED, Phase.ESCALATED]:
        semantic = {}
        item.engine.last_mode = 'mock' if item.use_mock or not item.engine.api_key else 'live'
        item.engine.fallback_reason = None
    else:
        semantic = item.engine.interpret(message, sm.state, item.history, conversation_context=context)
    context = apply_model_reference(sm, message, context, semantic)
    context['public_help'] = UtteranceExtractor.support_orientation(message)
    result = sm.evaluate_turn(message, semantic=semantic, conversation_context=context)
    return complete_customer_response(item, message, result, context, semantic)


def complete_customer_response(item, message, result, context, semantic=None):
    """One execution path for text chat and a submitted identity form."""
    sm = item.machine
    semantic = semantic or {}
    # A bounded emotion proposal may influence conversational action choice,
    # never identity evidence, claim permissions or consent.
    context = {**context, 'emotion': result.get('emotion')}
    proposed = [t for t in semantic.get('response_topics', []) or [] if t != 'unknown']
    contextual_topics = context.get('response_topics', [])
    result['response_topics'] = (contextual_topics if context.get('is_contextual_followup') or context.get('document_focus')
                                 else proposed or contextual_topics)
    result['response_style'] = (context.get('response_style', 'concise') if context.get('is_contextual_followup')
                                else semantic.get('response_style', 'concise'))
    decision = decide(sm, message, item.history, item.controller, item.policy_state, conversation_context=context)
    # Offline mode must not call a provider even if this session has a token.
    engine = MockEngine() if item.use_mock else item.engine
    reply = render_policy_reply(sm, message, result, context, decision,
                                engine=engine, history=item.history,
                                enable_response_planning=bool(not item.use_mock and item.engine.api_key and item.engine.last_mode == 'live'))
    if message == '[Submitted Security Verification Card]' and sm.get_verified_policyholder():
        reply = f'Thank you, {sm.get_verified_policyholder().name}. ' + reply
    if hasattr(sm, 'record_grounded_topics'):
        sm.record_grounded_topics(result.get('grounded_topics', []))
    if sm.state.phase == Phase.POST_PROCESS:
        sm.state.post_process.draft_summary = sm.generate_draft_email(sm.get_verified_policyholder(), sm.get_active_claim())
    item.history.extend([{'role': 'user', 'content': message}, {'role': 'assistant', 'content': reply}])
    answered = bool(result.get('grounded_answer_delivered'))
    decision['grounded_answered'] = answered
    decision['response_generation'] = result.get('response_generation', 'deterministic')
    record_execution(item.policy_state, decision, grounded_answered=answered)
    item.policy_decision = decision
    sm.record_snapshot(message, reply)
    item.policy_snapshots.append({'state': copy.deepcopy(item.policy_state), 'decision': copy.deepcopy(decision)})
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
        if item.machine.state.phase != Phase.VERIFY_ID:
            raise HTTPException(409, 'Identity verification is already closed for this call. Start a new call to use a different identity.')
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
        sm = item.machine
        item.engine.reset_turn_activity()
        previous_snapshots = len(sm.state.history_snapshots)
        res = sm.verify_card_data(form_data)
        if res.get('field_errors'):
            return {**reply_payload(req.session_id, item, res['agent_reply']), 'field_errors': res['field_errors']}
        # The state-machine card helper is also used on its own; replace its
        # provisional snapshot with the response actually chosen and displayed.
        del sm.state.history_snapshots[previous_snapshots:]
        message = '[Submitted Security Verification Card]'
        context = prepare_conversation_turn(sm, '', item.history)
        reply = complete_customer_response(item, message, res, context)
        return reply_payload(req.session_id, item, reply)



@app.post('/api/restore')
async def restore(req: RestoreRequest):
    item = get_session(req.session_id)
    async with item.lock:
        if not item.machine.restore_to_turn(req.turn_index):
            raise HTTPException(400, 'Invalid turn index')
        item.engine.reset_turn_activity()
        item.history = item.history[:(req.turn_index + 1) * 2]
        item.policy_snapshots = item.policy_snapshots[:req.turn_index + 1]
        if item.policy_snapshots:
            snapshot = item.policy_snapshots[-1]
            item.policy_state = copy.deepcopy(snapshot['state'])
            item.policy_decision = copy.deepcopy(snapshot['decision'])
        else:
            item.policy_state, item.policy_decision = {}, None
        return {'status': 'restored', **reply_payload(req.session_id, item)}


@app.get('/api/trajectory/{session_id}')
async def trajectory(session_id: str):
    item = get_session(session_id)
    async with item.lock:
        snapshots = []
        current_owner = item.machine.get_verified_policyholder()
        for index, snapshot in enumerate(item.machine.state.history_snapshots):
            temporary = SOPStateMachine(session_id)
            temporary.state = SOPState.model_validate(snapshot.state_dump)
            past_owner = temporary.get_verified_policyholder()
            can_read_claim = bool(current_owner and past_owner and current_owner.party_id == past_owner.party_id)
            policy = item.policy_snapshots[index]['decision'] if index < len(item.policy_snapshots) else None
            snapshots.append({'turn_index': snapshot.turn_index, 'phase': snapshot.phase,
                              'state_dump': public_state(temporary, force_shield=not can_read_claim),
                              'policy_decision': public_decision(policy),
                              'user_message': '[caller text omitted; may contain PII]',
                              'agent_reply': (redact_text(snapshot.agent_reply) if can_read_claim or not past_owner
                                              else '[Protected historical reply omitted; claim access is currently locked.]'),
                              'timestamp': snapshot.timestamp})
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
