import pytest
from backend.harness.state_machine import SOPStateMachine
from backend.harness.types import Phase
from backend.engine.mock_engine import MockEngine

def test_progressive_partial_pii_verification():
    """
    Test a conversational multi-turn verification flow:
    Turn 1: User gives name only ("Ava Lopez") -> 1 PII, stay in VERIFY_ID.
    Turn 2: User asks why, then gives phone and email -> 3 PII total, verified!
    """
    sm = SOPStateMachine(session_id="test_partial_pii")
    engine = MockEngine()
    history = []

    # Turn 1: Name only
    msg1 = "Hi, my name is Ava Lopez."
    state_result1 = sm.evaluate_turn(msg1)
    reply1 = engine.generate_response(msg1, state_result1, history)
    history.append({"role": "user", "content": msg1})
    history.append({"role": "assistant", "content": reply1})

    assert sm.state.phase == Phase.VERIFY_ID
    assert sm.state.verified_party_id is None
    assert sm.state.accumulated_pii.name == "Ava Lopez"
    assert "verify" in reply1.lower() or "3 pieces" in reply1.lower()

    # Turn 2: Alternate fields: Phone + Email
    msg2 = "My phone is +16503882920 and my email is ava.lopez@email.com."
    state_result2 = sm.evaluate_turn(msg2)
    reply2 = engine.generate_response(msg2, state_result2, history)

    # Now verified! Party P7
    assert sm.state.verified_party_id == "P7"
    assert len(sm.state.verified_fields) >= 3
    assert sm.state.phase in [Phase.RESOLVE_INTENT, Phase.PROCESS_CASE]
