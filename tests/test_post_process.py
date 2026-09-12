import pytest
from backend.harness.state_machine import SOPStateMachine
from backend.harness.types import Phase
from backend.engine.mock_engine import MockEngine

def test_post_process_skip_email():
    """
    Test user explicitly declining the email summary.
    """
    sm = SOPStateMachine(session_id="test_skip_email")
    engine = MockEngine()
    history = []

    # Fast forward: verify and resolve
    msg1 = "I am Margaret Chen, POL-9921, DOB 1985-03-15, SSN last four 4472. Denied healthcare claim from January."
    sm.evaluate_turn(msg1)
    assert sm.state.phase == Phase.PROCESS_CASE

    # Signal done
    msg2 = "That is all, thank you."
    state_result2 = sm.evaluate_turn(msg2)
    reply2 = engine.generate_response(msg2, state_result2, history)
    assert sm.state.phase == Phase.POST_PROCESS

    # User declines
    msg3 = "No thanks, skip it."
    state_result3 = sm.evaluate_turn(msg3)
    reply3 = engine.generate_response(msg3, state_result3, history)

    assert sm.state.phase == Phase.CONCLUDED
    assert sm.state.post_process.user_decision == "declined"
    assert "skip" in reply3.lower()

def test_post_process_draft_contents():
    """
    Verify the draft email contains all 3 required elements:
    1) what was discussed
    2) claim status/outcome
    3) major follow-up items/next steps
    """
    sm = SOPStateMachine(session_id="test_email_contents")
    msg1 = "I am Margaret Chen, POL-9921, DOB 1985-03-15, SSN last four 4472. Denied healthcare claim from January."
    sm.evaluate_turn(msg1)
    
    ph = sm.get_verified_policyholder()
    ac = sm.get_active_claim()
    draft = sm.generate_draft_email(ph, ac)

    assert "What Was Discussed" in draft
    assert "Claim Status & Outcome" in draft
    assert "Major Follow-up Items & Next Steps" in draft
    assert "CL-2048" in draft
    assert "DENIED" in draft
    assert "pathology report" in draft.lower()
