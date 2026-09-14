"""HTTP integration of human input, masked policy, grounded speech and SOP gates."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from backend.app import app, sessions
from backend.harness import customer_policy
from backend.harness.types import AgentAction

OPENING = 'My name is Margaret Chen. DOB 1985-03-15, SSN last four 4472. My denied healthcare claim from January.'


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv('AI_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    sessions.clear()
    with TestClient(app) as value:
        yield value
    sessions.clear()


def start(client, controller='ppo42'):
    sid = client.post('/api/reset', json={}).json()['session_id']
    response = client.post('/api/config', json={'session_id': sid, 'controller': controller, 'use_mock': True})
    assert response.status_code == 200
    assert response.json()['controller'] == controller
    return sid


def say(client, sid, message):
    response = client.post('/api/chat', json={'session_id': sid, 'message': message})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize('controller', ['rule', 'ppo42', 'ppo7'])
def test_real_multiturn_reference_format_end_and_consent(client, controller):
    sid = start(client, controller)
    for message, matches in [('My name is Margaret Chen. My denied healthcare claim from January.', 1),
                             ('DOB 1985-03-15', 2)]:
        data = say(client, sid, message)
        assert data['current_phase'] == 'VERIFY_ID'
        assert len(data['sop_state']['verified_fields']) == matches
        assert data['active_case'] is None
        assert data['policy_decision']['source'] == controller
        assert 'ANSWER_GROUNDED' not in data['policy_decision']['allowed_actions']
    data = say(client, sid, 'SSN last four 4472')
    assert data['active_case']['case_id'] == 'CL-2048'
    assert data['policy_decision']['grounded_answered']
    say(client, sid, 'What documents do I need?')
    data = say(client, sid, 'What is the second one?')
    assert 'office note' in data['reply'].lower() and 'assessment' in data['reply'].lower()
    assert 'pathology report' not in data['reply'].lower()
    data = say(client, sid, "I can't get it.")
    assert 'visit summary' in data['reply'] and 'human review' in data['reply']
    data = say(client, sid, 'Can I send photos?')
    assert any(word in data['reply'].lower() for word in ['photo', 'scan', 'readable', 'copy'])
    data = say(client, sid, 'And the deadline?')
    assert '2026-03-18' in data['reply']
    data = say(client, sid, 'That answers my questions.')
    assert data['current_phase'] == 'POST_PROCESS'
    assert data['policy_decision']['selected_action'] == 'OFFER_EMAIL_SUMMARY'
    assert data['sop_state']['outbox_count'] == 0
    data = say(client, sid, 'No, skip the email summary.')
    assert data['current_phase'] == 'CONCLUDED'
    assert data['policy_decision']['source'] == 'sop'
    assert data['policy_decision']['selected_action'] is None
    assert data['policy_decision']['checkpoint_sha256'] is None
    assert data['sop_state']['outbox_count'] == 0
    assert not any(data['policy_decision']['action_mask'])


def test_action_changes_actual_speech_without_inventing_caller_input(client, monkeypatch):
    # A controlled causal probe, not an assertion that either trained seed picks
    # this action. Same observation and text, different legal decision.
    responses = []
    for action in [AgentAction.ACK_EMOTION, AgentAction.ANSWER_GROUNDED]:
        sid = start(client, 'rule')
        say(client, sid, OPENING)
        raw = 'I am angry. Why was my claim denied?'
        with monkeypatch.context() as scoped:
            scoped.setattr(customer_policy.RuleBasedPolicy, 'select_action', lambda self, obs, a=action: a)
            data = say(client, sid, raw)
        responses.append(data['reply'])
        assert data['policy_decision']['selected_action'] == action.value
        assert sessions[sid].history[-2]['content'] == raw
        assert sessions[sid].machine.state.history_snapshots[-1].user_message == raw
        assert data['active_case']['case_id'] == 'CL-2048'
        assert data['current_phase'] == 'PROCESS_CASE'
    assert responses[0] != responses[1]
    assert 'upsetting' in responses[0].lower()
    assert 'upsetting' not in responses[1].lower()
    for reply in responses:
        assert 'did not include the pathology report and the treating provider office note' in reply.lower()


def test_offline_with_stored_token_never_calls_provider(client, monkeypatch):
    sid = start(client)
    client.post('/api/config', json={'session_id': sid, 'api_key': 'test-never-export', 'use_mock': True})
    monkeypatch.setattr(sessions[sid].engine, 'interpret', lambda *a, **kw: pytest.fail('Offline called interpreter'))
    monkeypatch.setattr(sessions[sid].engine, 'generate_response', lambda *a, **kw: pytest.fail('Offline called provider composer'))
    data = say(client, sid, OPENING)
    assert data['engine_mode'] == 'mock'
    assert 'test-never-export' not in json.dumps(data)
    assert data['policy_decision']['source'] == 'ppo42'


def test_policy_restore_restores_its_memory_and_keeps_original_human_history(client):
    sid = start(client)
    say(client, sid, 'My name is Margaret Chen. I prefer phone instead of SSN.')
    runtime = copy.deepcopy(sessions[sid].policy_state)
    decision = copy.deepcopy(sessions[sid].policy_decision)
    say(client, sid, 'DOB 1985-03-15. My denied healthcare claim from January.')
    say(client, sid, 'SSN last four 4472')
    assert sessions[sid].policy_state['turn_count'] == 3
    restored = client.post('/api/restore', json={'session_id': sid, 'turn_index': 0}).json()
    assert restored['current_phase'] == 'VERIFY_ID'
    assert sessions[sid].policy_state == runtime
    assert sessions[sid].policy_decision == decision
    assert len(sessions[sid].history) == 2
    assert len(sessions[sid].policy_snapshots) == 1
    data = say(client, sid, 'DOB 1985-03-15')
    assert data['active_case'] is None
    assert sessions[sid].policy_state['turn_count'] == 2
    assert 'id_last4' in sessions[sid].policy_state['avoided_identity_fields']


def test_public_help_and_scope_do_not_claim_ppo_execution(client):
    sid = start(client)
    data = say(client, sid, 'What is a claim?')
    assert data['policy_decision']['source'] == 'sop'
    assert data['policy_decision']['selected_action'] is None
    assert 'insur' in data['reply'].lower()
    assert 'Please share your full name.' not in data['reply']
    data = say(client, sid, 'What is RL?')
    assert data['policy_decision']['source'] == 'sop'
    assert data['active_case'] is None
    data = say(client, sid, 'What is Q-learning?')
    assert data['current_phase'] == 'ESCALATED'


def test_corrupt_checkpoint_fallback_is_honest_in_http_payload(client, monkeypatch):
    sid = start(client)
    def corrupt(*args):
        raise ValueError('Incompatible feature version')
    monkeypatch.setattr(customer_policy, '_load_policy', corrupt)
    data = say(client, sid, 'My name is Margaret Chen')
    assert data['controller'] == 'ppo42'
    assert data['policy_decision']['source'] == 'rule'
    assert 'unavailable' in data['policy_decision']['fallback_reason']
    assert data['policy_decision']['checkpoint_sha256'] is None
    assert 'Margaret' not in json.dumps(data['policy_decision'])


def test_export_preserves_executed_policy_without_raw_identity(client):
    sid = start(client)
    data = say(client, sid, 'My name is Margaret Chen. DOB 1985-03-15.')
    record = client.get('/api/trajectory/' + sid).json()['snapshots'][0]
    assert record['policy_decision'] == data['policy_decision']
    assert 'observation' not in record['policy_decision']
    assert '1985-03-15' not in json.dumps(record)
    assert 'Margaret Chen' not in json.dumps(record)
    assert record['user_message'].startswith('[caller text omitted')


def test_client_cannot_inject_controller_actions_or_policy_state(client):
    sid = start(client)
    for field, value in [('selected_action', 'SEND_EMAIL'), ('policy_state', {'grounded_answered': True}),
                         ('conversation_context', {'context_claim_id': 'CL-2048'})]:
        response = client.post('/api/chat', json={'session_id': sid, 'message': 'hello', field: value})
        assert response.status_code == 422
    state = client.get('/api/state/' + sid).json()
    assert state['current_phase'] == 'VERIFY_ID'
    assert state['active_case'] is None
    assert state['sop_state']['outbox_count'] == 0


def test_revoked_identity_locks_protected_trace_and_historical_answers(client):
    sid = start(client)
    say(client, sid, OPENING)
    say(client, sid, 'What documents do I need?')
    data = say(client, sid, 'Actually I am calling on behalf of my mother Margaret Chen.')
    assert data['current_phase'] == 'ESCALATED'
    assert not data['sop_state']['identity_verified']
    assert data['active_case'] is None
    state = client.get('/api/state/' + sid).text
    history = client.get('/api/trajectory/' + sid).text
    for secret in ['CL-2048', 'pathology report', 'office note', '1450.00']:
        assert secret not in state
        assert secret not in history


def test_restoring_state_is_not_reported_as_a_new_model_call(client):
    sid = start(client)
    say(client, sid, 'My name is Margaret Chen')
    sessions[sid].engine.model_activity = {'interpretation_requests': 1, 'interpretation_succeeded': 1,
                                           'planning_requests': 1, 'planning_succeeded': 1}
    data = client.post('/api/restore', json={'session_id': sid, 'turn_index': 0}).json()
    assert not any(data['model_activity'].values())
