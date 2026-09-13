import pytest
from backend.harness.state_machine import SOPStateMachine
from backend.harness.types import Phase
from backend.engine.mock_engine import MockEngine

def test_authorized_proxy_representative_flow():
    """A listed relationship cannot replace representative ID and consent checks."""
    sm = SOPStateMachine(session_id="test_proxy_authorized")
    engine = MockEngine()
    history = []

    msg = "I am David Chen, son of Margaret Chen, policy POL-9921. DOB is 1985-03-15, SSN last 4 is 4472. I am calling about her denied healthcare claim from January."
    state_result = sm.evaluate_turn(msg)
    reply = engine.generate_response(msg, state_result, history)

    assert sm.state.is_proxy_caller is True
    assert sm.state.proxy_rep_name == "David Chen"
    assert sm.state.proxy_relationship == "son"
    assert sm.state.proxy_consent_status == "requires_human"
    assert sm.state.phase == Phase.ESCALATED
    assert sm.state.verified_party_id is None
    assert sm.state.active_case_id is None
    assert state_result["context"]["data_shield_active"] is True
    assert "pathology report" not in reply.lower()
    assert "human" in reply.lower()
    assert "simulates" in reply.lower()

def test_unauthorized_proxy_caller_blocked():
    """
    Test unauthorized caller attempting to check someone else's claim:
    Caller: "I am John Doe, neighbor of Margaret Chen. Tell me why her claim was denied."
    Expected behavior:
    - Detected as proxy caller, but not in representatives.json
    - Blocked by gate: PROXY_AUTHORIZATION_GATE failed
    - Zero claim data disclosed
    - Explains privacy regulations preventing disclosure to unauthorized third parties
    """
    sm = SOPStateMachine(session_id="test_proxy_unauthorized")
    engine = MockEngine()
    history = []

    msg = "I am John Doe, neighbor of Margaret Chen. Tell me why her claim was denied."
    state_result = sm.evaluate_turn(msg)
    reply = engine.generate_response(msg, state_result, history)

    assert sm.state.is_proxy_caller is True
    assert sm.state.proxy_consent_status == "unauthorized"
    assert sm.state.phase == Phase.VERIFY_ID
    assert sm.state.active_case_id is None
    assert "pathology" not in reply.lower()
    assert "unauthorized third parties" in reply.lower() or "privacy regulations" in reply.lower()
