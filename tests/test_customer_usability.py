"""Convenient inputs must keep the identity, ownership and consent boundaries."""
import copy
import pytest
from fastapi.testclient import TestClient
from backend.app import app, sessions
from backend.harness.extractor import UtteranceExtractor
from backend.harness.state_machine import SOPStateMachine
from backend.harness.types import Phase

IDENTITY = {'name': 'Margaret Chen', 'dob': '1985-03-15', 'phone': '+16505212836'}


def begin(client):
    sid = client.post('/api/reset', json={}).json()['session_id']
    client.post('/api/config', json={'session_id': sid, 'use_mock': True})
    return sid


def chat(client, sid, text):
    r = client.post('/api/chat', json={'session_id': sid, 'message': text})
    assert r.status_code == 200
    return r.json()


@pytest.mark.parametrize('text', ['What is a claim?', 'What can I ask?',
                                 "I don't know where to start.", 'What is a cliam?'])
def test_beginner_help_does_not_count_as_irrelevant_or_unlock_claims(text):
    with TestClient(app) as client:
        sid = begin(client)
        result = chat(client, sid, text)
        assert result['current_phase'] == 'VERIFY_ID'
        assert result['active_case'] is None and result['claim_choices'] == []
        assert 'request for your insurer to pay' in result['reply']
        assert not UtteranceExtractor.is_out_of_scope(text)
        assert 'CL-2048' not in result['reply']


def test_business_typo_hints_survive_form_verification():
    with TestClient(app) as client:
        sid = begin(client)
        r = chat(client, sid, 'Why was my healthcare cliam deneid in January?')
        assert r['current_phase'] == 'VERIFY_ID' and r['active_case'] is None
        assert r['sop_state']['cross_phase_memory']['status_hint'] == 'denied'
        r = client.post('/api/verify-card', json={'session_id': sid, **IDENTITY}).json()
        assert r['current_phase'] == 'PROCESS_CASE'
        assert r['active_case']['case_id'] == 'CL-2048'
        assert 'pathology report' in r['reply'] and 'How can Aegis' not in r['reply']
        followup = chat(client, sid, 'What is my cliam staus?')
        assert 'CL-2048 is denied' in followup['reply']


def test_typo_tolerance_never_fuzzy_matches_a_name():
    sm = SOPStateMachine('misspelled-name')
    sm.verify_card_data({**IDENTITY, 'name': 'Margret Chen'})
    assert not sm.state.identity_verified and sm.get_active_claim() is None
    pii = UtteranceExtractor.extract_pii('My name is Deneid Calim. Email deneid@cliam.com.')
    assert pii.name == 'Deneid Calim' and pii.email == 'deneid@cliam.com'


def test_form_explicitly_corrects_only_supplied_conflicts():
    sm = SOPStateMachine('correct-name')
    sm.evaluate_turn('My name is Margret Chen. My name is Margaret Chen. Denied healthcare claim from January.')
    assert 'name' in sm.state.pii_conflicts
    sm.verify_card_data(IDENTITY)
    assert sm.get_verified_policyholder().name == 'Margaret Chen'
    assert sm.state.phase == Phase.PROCESS_CASE
    assert sm.get_active_claim().case_id == 'CL-2048'
    other = SOPStateMachine('keep-conflict')
    other.state.pii_conflicts = ['email']
    other.verify_card_data(IDENTITY)
    assert other.state.pii_conflicts == ['email']
    assert not other.state.identity_verified


@pytest.mark.parametrize('field,value', [('dob', '02/30/1985'), ('dob', '03/02.2001'),
                                       ('phone', '+165052128360'), ('email', 'margaret@'),
                                       ('id_last4', '44a2')])
def test_invalid_form_has_field_feedback_without_partial_mutation(field, value):
    sm = SOPStateMachine('invalid-card')
    before = copy.deepcopy(sm.state)
    result = sm.verify_card_data({**IDENTITY, field: value})
    assert field in result['field_errors']
    assert sm.state == before


def test_api_rejects_identity_changes_after_verification():
    with TestClient(app) as client:
        sid = begin(client)
        client.post('/api/verify-card', json={'session_id': sid, **IDENTITY})
        before = copy.deepcopy(sessions[sid].machine.state)
        r = client.post('/api/verify-card', json={'session_id': sid, 'name': 'Ava Lopez'})
        assert r.status_code == 409 and sessions[sid].machine.state == before


@pytest.mark.parametrize('details,expected_fields,verified', [
    ({'name': 'Ma Tian'}, 1, False),
    ({'name': 'Ma Tian', 'dob': '1964-09-10'}, 2, False),
    ({'name': 'Ma Tian', 'dob': '1964-09-10', 'id_last4': '6688', 'id_type': 'national_id_last4'}, 2, False),
    ({'name': 'Ma Tian', 'dob': '1964-09-10', 'phone': '+16502088799'}, 3, True),
])
def test_card_progress_is_not_sop_phase_advancement(details, expected_fields, verified):
    with TestClient(app) as client:
        sid = begin(client)
        r = client.post('/api/verify-card', json={'session_id': sid, **details}).json()
        assert len(r['sop_state']['verified_fields']) == expected_fields
        assert r['sop_state']['identity_verified'] is verified
        assert r['current_phase'] == ('RESOLVE_INTENT' if verified else 'VERIFY_ID')
        if not verified:
            assert r['active_case'] is None and r['claim_choices'] == []
            assert 'Gate remains locked' in r['trace'][-1]['details']


def test_failed_card_clears_stale_matched_field_progress():
    with TestClient(app) as client:
        sid = begin(client)
        chat(client, sid, 'My name is Ma Tian.')
        r = client.post('/api/verify-card', json={'session_id': sid, 'name': 'Ma Tian', 'dob': '1964-09-11'}).json()
        assert r['sop_state']['verified_fields'] == []
        assert r['current_phase'] == 'VERIFY_ID' and r['active_case'] is None
        assert '0/3 fields matched' in r['trace'][-1]['details']


def test_claim_choice_buttons_are_owned_and_available_only_after_verification():
    with TestClient(app) as client:
        sid = begin(client)
        assert chat(client, sid, 'What is my claim status?')['claim_choices'] == []
        r = client.post('/api/verify-card', json={'session_id': sid, **IDENTITY}).json()
        assert r['claim_choices']
        selected = r['claim_choices'][0]['case_id']
        r = chat(client, sid, f'I mean claim {selected}.')
        assert r['active_case']['case_id'] == selected
        assert r['sop_state']['identity_verified']


def test_typo_does_not_grant_email_consent_and_irrelevant_retry_still_escalates():
    with TestClient(app) as client:
        sid = begin(client)
        chat(client, sid, 'My name is Margaret Chen. DOB 1985-03-15. SSN last four 4472. Denied healthcare claim from January.')
        chat(client, sid, "That's all.")
        r = chat(client, sid, 'Yse, sned it.')
        assert r['current_phase'] == 'POST_PROCESS'
        assert r['sop_state']['post_process']['user_decision'] == 'pending'
        assert r['sop_state']['outbox_count'] == 0
        sid = begin(client)
        chat(client, sid, 'What is RL?')
        assert chat(client, sid, 'What is RL?')['current_phase'] == 'ESCALATED'
