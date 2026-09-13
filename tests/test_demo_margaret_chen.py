import pytest
from backend.harness.state_machine import SOPStateMachine
from backend.harness.types import Phase
from backend.engine.mock_engine import MockEngine

def test_margaret_chen_single_utterance_flow():
    """
    Test the complete Insurance Demo Test Case:
    Caller says: I’m the policyholder. My name is Margaret Chen, policy POL-9921.
    I’m calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.
    """
    sm = SOPStateMachine(session_id="test_demo_session")
    engine = MockEngine()
    history = []

    msg1 = "I’m the policyholder. My name is Margaret Chen, policy POL-9921. I’m calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."
    
    # 1. Evaluate turn
    state_result = sm.evaluate_turn(msg1)
    reply1 = engine.generate_response(msg1, state_result, history)
    history.append({"role": "user", "content": msg1})
    history.append({"role": "assistant", "content": reply1})

    # Assertions on verification
    assert sm.state.verified_party_id == "P9", "Must identify party P9 (Margaret Chen)"
    assert len(sm.state.verified_fields) >= 3, "Must verify at least 3 PII fields"
    assert "name" in sm.state.verified_fields
    assert "dob" in sm.state.verified_fields
    assert "policy_number" not in sm.state.verified_fields
    assert "id_last4" in sm.state.verified_fields

    # Assertions on cross-phase memory
    assert sm.state.cross_phase_memory.case_type_hint == "healthcare"
    assert sm.state.cross_phase_memory.status_hint == "denied"
    assert "January" in sm.state.cross_phase_memory.date_hint

    # Assertions on auto-resolution of active case
    assert sm.state.active_case_id == "CL-2048", "Must resolve to January denied healthcare claim CL-2048"
    assert sm.state.phase == Phase.PROCESS_CASE

    # Assertions on grounded reply
    assert "CL-2048" in reply1
    assert "pathology report" in reply1.lower()
    assert "office note" in reply1.lower()
    assert "2026-03-18" in reply1

    # 2. Turn 2: Follow-up question on submission
    msg2 = "How soon do I need to submit these documents and how do I send them?"
    state_result2 = sm.evaluate_turn(msg2)
    reply2 = engine.generate_response(msg2, state_result2, history)
    history.append({"role": "user", "content": msg2})
    history.append({"role": "assistant", "content": reply2})

    assert sm.state.phase == Phase.PROCESS_CASE
    assert "within a week" in reply2.lower()
    assert "portal" in reply2.lower() or "upload" in reply2.lower()

    # 3. Turn 3: User indicates done
    msg3 = "That is all the questions I have. Thank you!"
    state_result3 = sm.evaluate_turn(msg3)
    reply3 = engine.generate_response(msg3, state_result3, history)
    history.append({"role": "user", "content": msg3})
    history.append({"role": "assistant", "content": reply3})

    # Must transition to POST_PROCESS and offer email summary
    assert sm.state.phase == Phase.POST_PROCESS
    assert sm.state.post_process.email_offered is True
    assert "email summary" in reply3.lower()
    assert "skip" in reply3.lower()

    # 4. Turn 4: User accepts email summary
    msg4 = "Yes, please send the summary to my email."
    state_result4 = sm.evaluate_turn(msg4)
    reply4 = engine.generate_response(msg4, state_result4, history)

    assert sm.state.phase == Phase.CONCLUDED
    assert sm.state.post_process.user_decision == "accepted"
    assert sm.state.post_process.sent_to == "margaret@email.com"
    assert "margaret@email.com" in reply4.lower()
