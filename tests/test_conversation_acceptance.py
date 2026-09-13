"""Customer-language acceptance: finishing is separate from email consent."""
import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend.rl.featurizer import StateFeaturizer
import numpy as np

DEMO = 'My name is Margaret Chen, policy POL-9921. DOB 1985-03-15, SSN last four 4472. Denied healthcare claim from January.'


def begin(client, verified=True):
    sid = client.post('/api/reset', json={}).json()['session_id']
    client.post('/api/config', json={'session_id': sid, 'use_mock': True})
    if verified:
        turn(client, sid, DEMO)
    return sid


def turn(client, sid, message):
    r = client.post('/api/chat', json={'session_id': sid, 'message': message})
    assert r.status_code == 200
    return r.json()


@pytest.mark.parametrize('message', [
    'That answers my question.', 'This answered all my questions.',
    "You've answered my question, thanks.", 'All my questions are answered.',
    'I have everything I need.', 'Thanks, that helps.',
    'That answers my question. I’m ready to wrap up.',
])
def test_natural_finish_offers_summary_without_sending(message):
    with TestClient(app) as client:
        sid = begin(client)
        r = turn(client, sid, message)
        assert r['current_phase'] == 'POST_PROCESS'
        assert r['sop_state']['post_process']['user_decision'] == 'pending'
        assert r['sop_state']['outbox_count'] == 0
        assert turn(client, sid, 'No thanks, skip it.')['current_phase'] == 'CONCLUDED'


@pytest.mark.parametrize('message', [
    'That does not answer my question.', "That didn't answer my question.",
    'That answers my question, but I have another question.',
    'That answers my question. Where should I upload the files',
    'That answers my question. Explain the appeal deadline',
    'Thanks, that helps. I still need the submission instructions.',
])
def test_unfinished_followup_does_not_wrap_up(message):
    with TestClient(app) as client:
        sid = begin(client)
        r = turn(client, sid, message)
        assert r['current_phase'] == 'PROCESS_CASE'
        assert not r['sop_state']['post_process']['email_offered']


def test_finish_phrase_does_not_bypass_identity():
    with TestClient(app) as client:
        sid = begin(client, verified=False)
        r = turn(client, sid, 'That answers my question.')
        assert r['current_phase'] == 'VERIFY_ID' and r['active_case'] is None


def test_model_finish_proposal_cannot_override_an_explicit_followup():
    from backend.harness.state_machine import SOPStateMachine
    sm = SOPStateMachine('finish-proposal')
    sm.evaluate_turn(DEMO)
    sm.evaluate_turn('That answers my question. Explain the deadline',
                     {'wrap_up': True, 'response_topics': ['unknown']})
    assert sm.state.phase.value == 'PROCESS_CASE'


def test_concise_answers_preserve_conditions_and_expand_on_request():
    with TestClient(app) as client:
        sid = begin(client)
        short = turn(client, sid, 'Can I use a scan of the pathology report?')['reply']
        assert 'fully readable' in short and 'original' in short and 'confirm' in short
        brief = turn(client, sid, 'What if I cannot get the pathology report?')['reply']
        detailed = turn(client, sid, 'More details on alternatives for the missing pathology report please.')['reply']
        assert 'replacement copy' in brief and 'human review' in brief
        assert len(detailed) > len(brief)
        r = turn(client, sid, 'That answers my question.')
        summary = r['sop_state']['post_process']['draft_summary']
        assert len(summary.split()) < 300
        for fact in ['CL-2048', 'DENIED', 'pathology report', 'office note', '2026-03-18']:
            assert fact in summary


@pytest.mark.parametrize('message', [
    "The lab shut down and all I've got is a fuzzy photocopy. Any other way?",
    'The clinic closed permanently. How can I obtain the report now?',
])
def test_unavailable_document_gets_grounded_alternatives(message):
    with TestClient(app) as client:
        sid = begin(client)
        r = turn(client, sid, message)
        assert r['current_phase'] == 'PROCESS_CASE'
        assert 'replacement copy' in r['reply'] and 'human review' in r['reply']


def test_verification_alternative_is_not_a_document_question():
    from backend.harness.extractor import UtteranceExtractor
    assert 'document_alternatives' not in UtteranceExtractor.extract_topics(
        'Is there another way to verify my identity?')


def test_emotion_ablation_changes_only_emotion_inputs():
    obs = {'emotion': 'frustration', 'privacy_concern': True, 'phase': 'VERIFY_ID'}
    full = StateFeaturizer().featurize(obs)
    ablated = StateFeaturizer(use_emotion_features=False).featurize(obs)
    assert full[20:25].sum() == 1 and ablated[20:25].sum() == 0
    assert np.array_equal(full[:20], ablated[:20])
    assert np.array_equal(full[25:], ablated[25:])
