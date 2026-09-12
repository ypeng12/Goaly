import pytest
from backend.harness.state_machine import SOPStateMachine
from backend.harness.types import Phase
from backend.engine.mock_engine import MockEngine

def test_single_out_of_scope_rejection():
    """
    Test: Out of scope query (e.g. 'What is RL?')
    Expected behavior:
    - Polite rejection
    - Re-affirms insurance scope
    - Phase does not leak or break
    """
    sm = SOPStateMachine(session_id="test_oos_session")
    engine = MockEngine()
    history = []

    msg = "What is RL?"
    state_result = sm.evaluate_turn(msg)
    reply = engine.generate_response(msg, state_result, history)

    assert state_result["is_out_of_scope"] is True
    assert "unable to answer" in reply.lower() or "claims and policy" in reply.lower() or "apologize" in reply.lower()
    assert sm.state.phase == Phase.VERIFY_ID

def test_repeated_out_of_scope_triggers_escalation():
    """
    Test: Consecutive out-of-scope queries trigger human escalation.
    """
    sm = SOPStateMachine(session_id="test_oos_repeat_session")
    engine = MockEngine()
    history = []

    # 1st out of scope
    msg1 = "What is reinforcement learning?"
    sm.evaluate_turn(msg1)
    assert sm.state.out_of_scope_count == 1
    assert sm.state.phase == Phase.VERIFY_ID

    # 2nd out of scope
    msg2 = "Can you write python code for q-learning?"
    state_result2 = sm.evaluate_turn(msg2)
    reply2 = engine.generate_response(msg2, state_result2, history)

    assert sm.state.out_of_scope_count >= 2
    assert sm.state.phase == Phase.ESCALATED
    assert "transfer" in reply2.lower() or "human" in reply2.lower()
