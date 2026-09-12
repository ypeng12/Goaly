import pytest
from backend.harness.state_machine import SOPStateMachine
from backend.harness.types import Phase
from backend.engine.mock_engine import MockEngine

def test_frustration_and_deescalation():
    """
    Test: Caller: "I already told you who I am. This is ridiculous. Just tell me why my claim was denied."
    Expected behavior:
    - Acknowledge the frustration.
    - Explain that claim details are protected and require verification first.
    - Offer allowed verification options.
    - Do NOT disclose claim details or skip verification.
    """
    sm = SOPStateMachine(session_id="test_frustration_session")
    engine = MockEngine()
    history = []

    msg1 = "I already told you who I am. This is ridiculous. Just tell me why my claim was denied."
    state_result = sm.evaluate_turn(msg1)
    reply = engine.generate_response(msg1, state_result, history)

    # 1. Verification must not be bypassed
    assert sm.state.phase == Phase.VERIFY_ID, "Must remain in VERIFY_ID"
    assert sm.state.verified_party_id is None
    assert sm.state.active_case_id is None

    # 2. Data shield must be active
    assert state_result["context"]["data_shield_active"] is True

    # 3. Reply must not reveal denial reason or documents needed
    assert "pathology report" not in reply.lower()
    assert "office note" not in reply.lower()

    # 4. Reply must acknowledge frustration and explain privacy
    reply_lower = reply.lower()
    assert "understand" in reply_lower or "frustration" in reply_lower
    assert "privacy" in reply_lower or "protect" in reply_lower
    assert "verify" in reply_lower or "verification" in reply_lower

    # 5. Follow-up: User provides alternative fields (Phone + Email + DOB)
    msg2 = "Fine. My phone is +16505212836, email is margaret@email.com, and my DOB is 1985-03-15."
    state_result2 = sm.evaluate_turn(msg2)
    reply2 = engine.generate_response(msg2, state_result2, history)

    # Now verified!
    assert sm.state.verified_party_id == "P9"
    assert sm.state.phase in [Phase.RESOLVE_INTENT, Phase.PROCESS_CASE]

def test_demands_human_escalation():
    """
    Test: Caller demands human agent -> immediate escalation.
    """
    sm = SOPStateMachine(session_id="test_escalate_session")
    engine = MockEngine()
    history = []

    msg = "I refuse to speak with a bot. Let me speak to a human representative right now."
    state_result = sm.evaluate_turn(msg)
    reply = engine.generate_response(msg, state_result, history)

    assert sm.state.phase == Phase.ESCALATED
    assert "transfer" in reply.lower() or "connect" in reply.lower()
    assert "human" in reply.lower() or "representative" in reply.lower()
