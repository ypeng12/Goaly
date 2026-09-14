"""HTTP integration with a fake provider, never a paid/live-provider run.

These tests prove the request/response contract and real customer path. They do
not certify a provider model's comprehension or a checkpoint's learned quality.
"""
import json

import httpx
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app import app, sessions
from backend.harness import customer_policy
from backend.harness.dialogue import EMOTIONS, dialogue_signals
from backend.harness.types import AGENT_ACTIONS, AgentAction
from backend.rl.featurizer import StateFeaturizer

OPENING = ('My name is Margaret Chen. DOB 1985-03-15, SSN last four 4472. '
           'My denied healthcare claim from January.')


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv('AI_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    sessions.clear()
    with TestClient(app) as value:
        yield value
    sessions.clear()


def say(client, sid, text):
    response = client.post('/api/chat', json={'session_id': sid, 'message': text})
    assert response.status_code == 200, response.text
    return response.json()


def start(client, verified=True):
    sid = client.post('/api/reset', json={}).json()['session_id']
    client.post('/api/config', json={'session_id': sid, 'use_mock': True, 'controller': 'rule'})
    if verified:
        say(client, sid, OPENING)
    return sid


def enable_provider(client, sid, controller='rule'):
    response = client.post('/api/config', json={
        'session_id': sid, 'api_key': 'fake-contract-token', 'use_mock': False, 'controller': controller})
    assert response.status_code == 200


def fake_provider(monkeypatch, *, topics=('documents',), emotion='neutral', reference=None, invalid_plan=False):
    captures = []
    original = httpx.Client

    def handler(request):
        payload = json.loads(request.content)
        captures.append(payload)
        format_ = payload['response_format']['json_schema']
        if format_['name'] == 'insurance_fact_plan':
            request_data = json.loads(payload['messages'][-1]['content'])
            content = {'ordered_fact_ids': list(reversed(request_data['required_fact_ids'])),
                       'layout': 'steps', 'introduction': 'direct'}
            if invalid_plan:
                content['unauthorized_fact'] = 'The claim is approved and the money is guaranteed.'
        else:
            content = {'is_out_of_scope': False, 'emotion': emotion, 'is_refusal': False,
                       'demands_human': False, 'is_proxy': False, 'wrap_up': False,
                       'topics': list(topics), 'style': 'concise', 'slots': []}
            if 'document_reference' in format_['schema']['properties']:
                content['document_reference'] = reference
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(content)}}]})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr('backend.engine.llm_engine.httpx.Client', lambda **kw: original(transport=transport, **kw))
    return captures


def capture_policy_features(monkeypatch):
    observed = []

    class ControlledPolicy:
        def __call__(self, features, mask):
            observed.append((features.clone(), mask.clone()))
            anxious = features[20 + EMOTIONS.index('anxiety')].item() == 1.0
            target = AgentAction.ACK_EMOTION if anxious else AgentAction.ANSWER_GROUNDED
            logits = torch.zeros(len(AGENT_ACTIONS))
            logits[AGENT_ACTIONS.index(target)] = 10.0
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            return logits, torch.tensor(0.0)

    metadata = {'sha256': 'a' * 64, 'training_seed': 42, 'actual_steps': 1,
                'feature_version': StateFeaturizer.VERSION, 'environment_version': 4,
                'checkpoint_family': 'customer_multi_turn_v1', 'mask_version': 'customer_sop_v1'}
    monkeypatch.setattr(customer_policy, '_load_policy', lambda *a: (ControlledPolicy(), StateFeaturizer(), metadata))
    return observed


def test_http_uses_interpretation_then_validated_fact_plan(client, monkeypatch):
    sid = start(client)
    enable_provider(client, sid)
    captures = fake_provider(monkeypatch, topics=('appeal_deadline', 'status'))
    data = say(client, sid, 'Explain the claim status and appeal deadline.')
    assert [p['response_format']['json_schema']['name'] for p in captures] == ['insurance_interpretation', 'insurance_fact_plan']
    assert data['model_activity'] == {'interpretation_requests': 1, 'interpretation_succeeded': 1,
                                      'planning_requests': 1, 'planning_succeeded': 1}
    assert data['policy_decision']['response_generation'] == 'live_fact_plan'
    assert data['policy_decision']['grounded_answered']
    assert data['current_phase'] == 'PROCESS_CASE'
    assert all(fact in data['reply'] for fact in ['2026-03-18', 'has passed', 'cannot promise an extension'])
    facts = json.loads(captures[-1]['messages'][-1]['content'])['approved_facts']
    assert all(item['text'] in data['reply'] for item in facts)
    for identity_value in ['1985-03-15', '4472', 'margaret@email.com', 'fake-contract-token']:
        assert identity_value not in json.dumps(captures)
        assert identity_value not in json.dumps(data)


def test_http_model_reference_applies_before_policy_and_fact_composition(client, monkeypatch):
    sid = start(client)
    say(client, sid, 'What documents are needed?')
    enable_provider(client, sid)
    captures = fake_provider(monkeypatch, topics=('document_alternatives',), reference=2)
    data = say(client, sid, 'There is no way of getting hold of the second piece of paperwork.')
    assert data['current_phase'] == 'PROCESS_CASE'
    assert 'visit summary' in data['reply'] and 'office note' in data['reply']
    assert 'hospital or lab' not in data['reply']
    assert data['policy_decision']['response_generation'] == 'live_fact_plan'
    assert len(captures) == 2


@pytest.mark.parametrize('verified', [False, True])
def test_model_emotion_reaches_policy_features_mask_and_actual_speech(client, monkeypatch, verified):
    sid = start(client, verified=verified)
    message = 'The waiting keeps me awake at night. What documents does my claim need?'
    assert dialogue_signals(message)['emotion'] == 'neutral'
    enable_provider(client, sid, controller='ppo42')
    fake_provider(monkeypatch, emotion='anxiety')
    observed = capture_policy_features(monkeypatch)
    data = say(client, sid, message)
    assert len(observed) == 1
    features, mask = observed[0]
    assert features[20 + EMOTIONS.index('anxiety')].item() == 1
    assert mask[AGENT_ACTIONS.index(AgentAction.ACK_EMOTION)]
    assert data['policy_decision']['source'] == 'ppo42'
    assert data['policy_decision']['selected_action'] == 'ACK_EMOTION'
    assert 'worrying' in data['reply']
    if verified:
        assert data['current_phase'] == 'PROCESS_CASE'
        assert 'pathology report' in data['reply']
        assert data['model_activity']['planning_succeeded'] == 1
    else:
        assert data['current_phase'] == 'VERIFY_ID' and data['active_case'] is None
        assert 'pathology report' not in data['reply']
        assert data['model_activity']['planning_requests'] == 0


def test_model_emotion_cannot_override_explicit_scope_or_unlock_claim(client, monkeypatch):
    sid = start(client, verified=False)
    enable_provider(client, sid, controller='ppo42')
    captures = fake_provider(monkeypatch, emotion='anxiety', topics=('documents',))
    observed = capture_policy_features(monkeypatch)
    data = say(client, sid, 'What is RL?')
    assert not observed
    assert data['policy_decision']['source'] == 'sop'
    assert data['policy_decision']['selected_action'] is None
    assert data['active_case'] is None and data['current_phase'] == 'VERIFY_ID'
    assert 'unrelated' in data['reply'] and 'pathology report' not in data['reply']
    assert len(captures) == 1 and data['model_activity']['planning_requests'] == 0


def test_bad_fact_plan_falls_back_without_side_effects_or_provider_credit(client, monkeypatch):
    sid = start(client)
    enable_provider(client, sid)
    fake_provider(monkeypatch, topics=('appeal_deadline',), invalid_plan=True)
    data = say(client, sid, 'Explain the appeal deadline.')
    assert data['engine_mode'] == 'fallback'
    assert data['policy_decision']['response_generation'] == 'grounded_plan_fallback'
    assert data['model_activity']['planning_requests'] == 1
    assert data['model_activity']['planning_succeeded'] == 0
    assert data['current_phase'] == 'PROCESS_CASE'
    assert data['sop_state']['outbox_count'] == 0
    assert '2026-03-18' in data['reply'] and 'cannot promise an extension' in data['reply']
    assert 'money is guaranteed' not in data['reply']


def test_terminal_replay_resets_activity_and_does_not_call_fake_provider(client, monkeypatch):
    sid = start(client)
    say(client, sid, "That's all.")
    say(client, sid, 'No, skip the email summary.')
    enable_provider(client, sid)
    captures = fake_provider(monkeypatch)
    data = say(client, sid, 'Yes, send the summary after all.')
    assert not captures
    assert all(value == 0 for value in data['model_activity'].values())
    assert data['policy_decision']['source'] == 'sop'
    assert data['sop_state']['post_process']['user_decision'] == 'declined'
    assert data['sop_state']['outbox_count'] == 0
