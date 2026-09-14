"""Public API boundaries and the live adapter's mocked HTTP contract."""
import json
import httpx
import pytest
from fastapi.testclient import TestClient
from backend.app import app, sessions
from backend.engine.llm_engine import LLMEngine
from backend.harness.state_machine import SOPStateMachine

DEMO = 'I’m the policyholder. My name is Margaret Chen, policy POL-9921. I’m calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.'


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv('AI_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    sessions.clear()
    with TestClient(app) as client:
        yield client
    sessions.clear()


def start(client):
    result = client.post('/api/reset', json={})
    assert result.status_code == 200
    return result.json()['session_id']


def chat(client, sid, text):
    response = client.post('/api/chat', json={'session_id': sid, 'message': text})
    assert response.status_code == 200, response.text
    return response.json()


def test_public_api_complete_flow_and_single_outbox(client):
    sid = start(client)
    result = chat(client, sid, DEMO)
    assert result['current_phase'] == 'PROCESS_CASE'
    assert result['active_case']['case_id'] == 'CL-2048'
    assert set(result['sop_state']['verified_fields']) == {'name', 'dob', 'id_last4'}
    assert result['sop_state']['identity_verified']
    assert result['verified_policyholder'] is None
    chat(client, sid, 'How long does review take after I submit?')
    result = chat(client, sid, "That's all")
    assert result['current_phase'] == 'POST_PROCESS'
    assert 'review timing' in result['sop_state']['post_process']['draft_summary']
    result = chat(client, sid, 'Yes, please send the email summary.')
    assert result['current_phase'] == 'CONCLUDED'
    assert result['sop_state']['outbox_count'] == 1
    assert result['sop_state']['post_process']['delivery_status'] == 'simulated'
    result = chat(client, sid, 'Please send it again')
    assert result['sop_state']['outbox_count'] == 1


def test_unverified_metadata_trajectory_and_escalation_do_not_expose_records(client):
    sid = start(client)
    result = chat(client, sid, 'My name is Margaret Chen. My denied healthcare claim was in January.')
    assert result['sop_state']['collected_fields'] == ['name']
    assert not result['sop_state']['identity_verified']
    assert result['active_case'] is None
    assert result['verified_policyholder'] is None
    trajectory = client.get('/api/trajectory/' + sid).text
    state = client.get('/api/state/' + sid).text
    for secret in ['margaret@email.com', '1985-03-15', '4472', 'pathology report', 'CL-2048']:
        assert secret not in trajectory
        assert secret not in state
    result = chat(client, sid, 'Please connect me with a human representative')
    assert result['current_phase'] == 'ESCALATED'
    assert result['sop_state']['data_shield_active']
    assert result['active_case'] is None


def test_config_and_state_are_session_scoped_and_token_never_returned(client):
    one, two = start(client), start(client)
    response = client.post('/api/config', json={'session_id': one, 'api_key': 'test-private-token', 'use_mock': True})
    assert response.status_code == 200
    assert response.json()['has_api_key']
    assert 'test-private-token' not in response.text
    assert not client.get('/api/config', params={'session_id': two}).json()['has_api_key']
    assert chat(client, one, DEMO)['current_phase'] == 'PROCESS_CASE'
    assert chat(client, two, 'Hello')['current_phase'] == 'VERIFY_ID'
    changed = client.post('/api/config', json={'session_id': one, 'base_url': 'https://example.com/v1'})
    assert changed.status_code == 422  # Cannot redirect stored credentials.
    assert 'test-private-token' not in client.get('/api/state/' + one).text


def test_owner_managed_server_key_is_reused_but_cannot_be_redirected(monkeypatch):
    monkeypatch.setenv('AI_API_KEY', 'owner-managed-test-key')
    monkeypatch.setenv('AI_BASE_URL', 'https://api.example.test/v1')
    monkeypatch.setenv('AI_MODEL', 'owner-managed-model')
    sessions.clear()
    with TestClient(app) as managed_client:
        sid = start(managed_client)
        config = managed_client.get('/api/config', params={'session_id': sid}).json()
        assert config['has_api_key'] and config['api_key_source'] == 'server'
        assert config['base_url'] == 'https://api.example.test/v1'
        assert config['model'] == 'owner-managed-model'
        assert 'owner-managed-test-key' not in json.dumps(config)
        blocked = managed_client.post('/api/config', json={
            'session_id': sid, 'base_url': 'https://attacker.example/v1', 'model': 'expensive-model'})
        assert blocked.status_code == 403
        allowed = managed_client.post('/api/config', json={'session_id': sid, 'use_mock': True, 'controller': 'rule'})
        assert allowed.status_code == 200 and allowed.json()['controller'] == 'rule'
    sessions.clear()


def test_skip_decision_cannot_be_reversed_or_false_send_claimed(client):
    sid = start(client)
    chat(client, sid, DEMO)
    chat(client, sid, "That's all")
    result = chat(client, sid, "Yes, but don't send the email.")
    assert result['sop_state']['post_process']['user_decision'] == 'declined'
    assert 'skip' in result['reply']
    assert result['sop_state']['outbox_count'] == 0
    result = chat(client, sid, 'Yes send it')
    assert result['sop_state']['post_process']['user_decision'] == 'declined'
    assert result['sop_state']['outbox_count'] == 0


def test_post_process_followup_answers_without_consent_and_refreshes_summary(client):
    sid = start(client)
    chat(client, sid, DEMO)
    chat(client, sid, "That's all")
    result = chat(client, sid, 'What if I cannot get the pathology report?')
    assert 'replacement' in result['reply']
    assert result['current_phase'] == 'POST_PROCESS'
    assert result['sop_state']['outbox_count'] == 0
    assert 'replacement' in result['sop_state']['post_process']['draft_summary']


def test_deadline_followup_does_not_change_remembered_claim_date(client):
    sid = start(client)
    chat(client, sid, DEMO)
    result = chat(client, sid, 'Is the appeal deadline in March 2026?')
    assert result['current_phase'] == 'PROCESS_CASE'
    assert result['sop_state']['cross_phase_memory']['date_hint'] == 'January'
    assert '2026-03-18' in result['reply']
    assert 'has passed' in result['reply']


def test_restore_recovers_full_state_and_trims_trajectory(client):
    sid = start(client)
    chat(client, sid, 'My name is Margaret Chen.')
    chat(client, sid, 'DOB 1985-03-15, SSN 4472. Denied healthcare claim in January.')
    result = client.post('/api/restore', json={'session_id': sid, 'turn_index': 0}).json()
    assert result['current_phase'] == 'VERIFY_ID'
    assert result['active_case'] is None
    assert len(result['trace']) == 1
    assert client.get('/api/trajectory/' + sid).json()['turns_count'] == 1


@pytest.mark.parametrize('message', ['', '   ', 'x' * 6001])
def test_request_validation_rejects_unusable_turns(client, message):
    sid = start(client)
    assert client.post('/api/chat', json={'session_id': sid, 'message': message}).status_code == 422


def test_unknown_session_and_cross_origin_write_rejected(client):
    assert client.post('/api/chat', json={'session_id': 'x' * 36, 'message': 'Hello'}).status_code == 404
    assert client.post('/api/reset', json={}, headers={'origin': 'https://attacker.example'}).status_code == 403


def proposal(**updates):
    data = {'is_out_of_scope': False, 'emotion': 'neutral', 'is_refusal': False,
            'demands_human': False, 'is_proxy': False, 'wrap_up': False,
            'topics': ['document_alternatives'], 'style': 'step_by_step', 'slots': []}
    data.update(updates)
    return data


def fake_provider(monkeypatch, content, status=200, capture=None):
    original = httpx.Client
    def handler(request):
        if capture is not None:
            capture.append(json.loads(request.content))
        return httpx.Response(status, json={'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]})
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr('backend.engine.llm_engine.httpx.Client', lambda **kwargs: original(transport=transport, **kwargs))


def test_live_model_selects_bounded_topic_from_messy_language(monkeypatch):
    requests = []
    fake_provider(monkeypatch, json.dumps(proposal()), capture=requests)
    engine = LLMEngine(api_key='test-secret')
    machine = SOPStateMachine('live')
    machine.evaluate_turn(DEMO)
    text = 'The lab shut down and all I have is a fuzzy photocopy. Any other way?'
    semantic = engine.interpret(text, machine.state)
    result = machine.evaluate_turn(text, semantic)
    result['response_topics'] = semantic['response_topics']
    response = engine.generate_response(text, result, [])
    assert 'replacement copy' in response
    assert engine.last_mode == 'live'
    assert requests[0]['response_format']['type'] == 'json_schema'
    payload = json.dumps(requests)
    for secret in ['test-secret', 'pathology report', 'margaret@email.com', '1985-03-15', '1450.00']:
        assert secret not in payload


@pytest.mark.parametrize('malformed', ['not JSON', '{}', json.dumps(proposal(phase='CONCLUDED', reply='Your claim is approved'))])
def test_invalid_or_injected_model_output_falls_back_without_free_text(monkeypatch, malformed):
    fake_provider(monkeypatch, malformed)
    engine = LLMEngine(api_key='test-secret')
    machine = SOPStateMachine('invalid')
    semantic = engine.interpret('Please help with my claim', machine.state)
    assert semantic == {}
    assert engine.last_mode == 'fallback'
    response = engine.generate_response('Please help with my claim', machine.evaluate_turn('Please help with my claim', semantic), [])
    assert 'approved' not in response
    assert machine.get_active_claim() is None


def test_hallucinated_identity_without_evidence_is_discarded(monkeypatch):
    slots = [{'field': field, 'value': value, 'evidence': 'fabricated'} for field, value in [('name', 'Margaret Chen'), ('dob', '1985-03-15'), ('id_last4', '4472')]]
    fake_provider(monkeypatch, json.dumps(proposal(slots=slots)))
    engine, machine = LLMEngine(api_key='test-secret'), SOPStateMachine('hallucination')
    semantic = engine.interpret('Tell me why the claim was denied', machine.state)
    assert semantic['pii'] == {}
    machine.evaluate_turn('Tell me why the claim was denied', semantic)
    assert machine.state.phase.value == 'VERIFY_ID'


def test_provider_error_reports_fallback_without_body_or_token(monkeypatch, caplog):
    fake_provider(monkeypatch, 'sensitive-provider-body', status=401)
    engine = LLMEngine(api_key='test-private-token')
    assert engine.interpret('Hello', SOPStateMachine('error').state) == {}
    assert '401' in engine.fallback_reason
    assert 'test-private-token' not in caplog.text
    assert 'sensitive-provider-body' not in caplog.text


def test_invalid_token_is_not_echoed_in_validation_response(client):
    sid = start(client)
    token = 'private-token-' * 100
    response = client.post('/api/config', json={'session_id': sid, 'api_key': token})
    assert response.status_code == 422
    assert token not in response.text


def test_explicit_correction_recovers_conflicting_identity():
    machine = SOPStateMachine('correction')
    machine.evaluate_turn('My name is Margaret Chen.')
    machine.evaluate_turn('My name is Ava Lopez.')
    assert machine.state.pii_conflicts == ['name']
    machine.evaluate_turn('Correction, my name is Ava Lopez. Phone +16503882920, email ava.lopez@email.com.')
    assert machine.state.pii_conflicts == []
    assert machine.get_verified_policyholder().party_id == 'P7'
