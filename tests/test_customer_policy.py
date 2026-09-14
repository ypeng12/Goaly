"""Actual checkpoint inference on human turns, independent of caller simulation."""
import copy
import json

import pytest

from backend.harness import customer_policy
from backend.harness.customer_policy import decide, record_execution
from backend.harness.state_machine import SOPStateMachine
from backend.harness.types import AGENT_ACTIONS, Phase


IDENTITY = "My name is Margaret Chen. DOB is 1985-03-15. SSN last four is 4472."
CASE_OPENING = IDENTITY + " My denied healthcare claim from January."


def opened_case():
    machine = SOPStateMachine("customer-policy-test")
    machine.evaluate_turn(CASE_OPENING)
    assert machine.state.phase == Phase.PROCESS_CASE
    return machine


@pytest.mark.parametrize("mode", customer_policy.CONTROLLER_MODES)
def test_identity_threshold_is_independent_of_customer_controller(mode):
    machine = SOPStateMachine("customer-policy-identity")
    runtime = {}
    for index, message in enumerate(("My name is Margaret Chen", "DOB is 1985-03-15")):
        machine.evaluate_turn(message)
        before = copy.deepcopy(machine.state.model_dump())
        decision = decide(machine, message, [], mode, runtime)
        assert machine.state.model_dump() == before
        assert machine.state.phase == Phase.VERIFY_ID
        assert machine.get_active_claim() is None
        assert len(machine.state.verified_fields) == index + 1
        assert "ANSWER_GROUNDED" not in decision["allowed_actions"]
        assert "OFFER_EMAIL_SUMMARY" not in decision["allowed_actions"]
        assert "SEND_EMAIL" not in decision["allowed_actions"]
        record_execution(runtime, decision)


@pytest.mark.parametrize("mode", ("ppo42", "ppo7"))
def test_actual_checkpoint_probabilities_drive_choice_and_mask(mode):
    machine = SOPStateMachine("customer-policy-inference")
    message = "This is ridiculous. My name is Margaret Chen."
    machine.evaluate_turn(message)
    decision = decide(machine, message, [], mode, {})
    assert decision["source"] == mode
    if decision["checkpoint"]["checkpoint_family"] == "legacy_simulator_v3":
        assert "original simulator-trained PPO" in decision["fallback_reason"]
    else:
        assert decision["fallback_reason"] is None
    assert len(decision["checkpoint_sha256"]) == 64
    assert decision["selected_action"] == max(decision["probabilities"], key=decision["probabilities"].get)
    assert sum(decision["probabilities"].values()) == pytest.approx(1.0)
    for action, permitted in zip(AGENT_ACTIONS, decision["action_mask"]):
        if not permitted:
            assert decision["probabilities"][action.value] == 0.0
    assert decision["execution"]["kind"] == "empathy_then_progress"
    assert decision["execution"]["followup_kind"] == "ask_identity_field"
    # Public trace contains feature values, never the actual identity fields.
    assert "Margaret" not in json.dumps(decision)


@pytest.mark.parametrize("mode", customer_policy.CONTROLLER_MODES)
def test_answer_does_not_authorize_ending_a_real_customer_conversation(mode):
    machine = opened_case()
    runtime = {"grounded_answered": True, "turn_count": 5}
    message = "What documents do I need?"
    machine.evaluate_turn(message)
    decision = decide(machine, message, [], mode, runtime)
    assert decision["selected_action"] == "ANSWER_GROUNDED"
    assert "OFFER_EMAIL_SUMMARY" not in decision["allowed_actions"]
    assert "ESCALATE_HUMAN" not in decision["allowed_actions"]
    assert machine.state.phase == Phase.PROCESS_CASE
    assert machine.state.post_process.user_decision is None


@pytest.mark.parametrize("mode", customer_policy.CONTROLLER_MODES)
def test_post_process_followup_is_answered_before_repeat_consent(mode):
    machine = opened_case()
    machine.evaluate_turn("That answers my questions.")
    assert machine.state.phase == Phase.POST_PROCESS
    message = "What if I cannot get the lab report?"
    machine.evaluate_turn(message)
    decision = decide(machine, message, [], mode, {"grounded_answered": True},
                      {"response_topics": ["document_alternatives"], "is_contextual_followup": True})
    assert decision["selected_action"] == "ANSWER_GROUNDED"
    assert "OFFER_EMAIL_SUMMARY" not in decision["allowed_actions"]
    assert "SEND_EMAIL" not in decision["allowed_actions"]
    assert machine.state.post_process.user_decision == "pending"
    assert machine.state.mock_outbox == []


def test_ambiguous_reference_requires_clarification_not_an_invented_answer():
    machine = opened_case()
    decision = decide(machine, "What about that?", [], "ppo42", {},
                      {"needs_clarification": True, "clarification_question": "Do you mean the document or the deadline?"})
    assert decision["allowed_actions"] == ["ASK_CLAIM_CLARIFICATION"]
    assert decision["selected_action"] == "ASK_CLAIM_CLARIFICATION"
    assert decision["execution"]["clarification_question"] == "Do you mean the document or the deadline?"


@pytest.mark.parametrize("message,terminal", [
    ("I want a human representative.", Phase.ESCALATED),
    ("No, skip the email summary.", Phase.CONCLUDED),
])
def test_terminal_sop_response_does_not_sample_or_repeat_side_effects(monkeypatch, message, terminal):
    machine = opened_case()
    machine.evaluate_turn("That answers my questions.")
    machine.evaluate_turn(message)
    monkeypatch.setattr(customer_policy, "_load_policy", lambda *_: pytest.fail("Terminal turn sampled a policy"))
    before = copy.deepcopy(machine.state.model_dump())
    decision = decide(machine, message, [], "ppo42", {})
    assert machine.state.phase == terminal
    assert machine.state.model_dump() == before
    assert decision["source"] == "sop"
    assert decision["selected_action"] is None
    assert not any(decision["action_mask"])
    assert sum(decision["probabilities"].values()) == 0


def test_rule_fallback_is_reported_as_rule_not_ppo(monkeypatch):
    machine = SOPStateMachine("customer-policy-unavailable")
    machine.evaluate_turn("My name is Margaret Chen")
    def unavailable(*args):
        raise ValueError("Incompatible schema")
    monkeypatch.setattr(customer_policy, "_load_policy", unavailable)
    decision = decide(machine, "My name is Margaret Chen", [], "ppo42", {})
    assert decision["source"] == "rule"
    assert decision["requested_controller"] == "ppo42"
    assert "unavailable" in decision["fallback_reason"]
    assert decision["checkpoint_sha256"] is None


def test_identity_request_respects_actual_provided_fields_and_privacy_preference():
    machine = SOPStateMachine("customer-policy-fields")
    message = "My name is Margaret Chen. I prefer phone instead of SSN."
    machine.evaluate_turn(message)
    decision = decide(machine, message, [], "rule", {})
    assert decision["execution"]["identity_field"] == "phone"
    assert "id_last4" in decision["execution"]["avoided_identity_fields"]
    runtime = {}
    record_execution(runtime, decision)
    machine.evaluate_turn("My phone is (650) 521-2836")
    second = decide(machine, "My phone is (650) 521-2836", [], "rule", runtime)
    assert second["execution"]["identity_field"] == "dob"
    assert second["execution"]["identity_field"] not in machine.state.verified_fields


def test_runtime_records_executed_resolutions_and_grounded_answers_only():
    machine = SOPStateMachine("customer-policy-memory")
    machine.evaluate_turn(IDENTITY)
    runtime = {}
    before = copy.deepcopy(runtime)
    decision = decide(machine, IDENTITY, [], "rule", runtime)
    assert runtime == before
    record_execution(runtime, decision)
    assert runtime["resolution_attempts"] == 1
    assert runtime["grounded_answered"] is False
    machine.evaluate_turn("My denied healthcare claim from January")
    decision = decide(machine, "My denied healthcare claim from January", [], "rule", runtime)
    record_execution(runtime, decision, grounded_answered=True)
    assert runtime["resolution_attempts"] == 0
    assert runtime["grounded_answered"] is True
    assert runtime["turn_count"] == 2


def test_scope_guard_and_public_help_are_explicit_sop_decisions():
    machine = SOPStateMachine("customer-policy-scope")
    machine.evaluate_turn("What is RL?")
    decision = decide(machine, "What is RL?", [], "ppo42", {})
    assert decision["source"] == "sop"
    assert decision["selected_action"] is None
    assert machine.state.phase == Phase.VERIFY_ID
    machine = SOPStateMachine("customer-policy-public-help")
    decision = decide(machine, "What is a claim?", [], "ppo42", {}, {"public_help": True})
    assert decision["source"] == "sop"
    assert not any(decision["action_mask"])


def test_bounded_model_emotion_changes_response_order_without_gate_changes():
    machine = SOPStateMachine("customer-policy-model-emotion")
    machine.evaluate_turn("My name is Margaret Chen")
    before = copy.deepcopy(machine.state.model_dump())
    decision = decide(machine, "This has been difficult for me.", [], "rule", {}, {"emotion": "anxiety"})
    assert decision["observation"]["emotion"] == "anxiety"
    assert decision["emotion_source"] == "bounded_context"
    assert decision["selected_action"] == "ACK_EMOTION"
    assert decision["execution"]["emotion"] == "anxiety"
    assert machine.state.model_dump() == before
    invalid = decide(machine, "Hello", [], "rule", {}, {"emotion": "identity_verified"})
    assert invalid["emotion_source"] == "text_detector"
    assert "ANSWER_GROUNDED" not in invalid["allowed_actions"]


def test_corrupt_checkpoint_has_an_explicit_safe_fallback(monkeypatch):
    import pickle
    machine = SOPStateMachine("customer-policy-corrupt")
    def corrupt(*_):
        raise pickle.UnpicklingError("corrupt")
    monkeypatch.setattr(customer_policy, "_load_policy", corrupt)
    decision = decide(machine, "Hello", [], "ppo42", {})
    assert decision["source"] == "rule"
    assert "UnpicklingError" in decision["fallback_reason"]


def test_nonfinite_policy_probabilities_have_an_explicit_safe_fallback(monkeypatch):
    import torch
    from backend.rl.featurizer import StateFeaturizer
    machine = SOPStateMachine("customer-policy-nan")
    class InvalidPolicy:
        def __call__(self, *_):
            return torch.full((len(AGENT_ACTIONS),), float("nan")), torch.tensor(0.0)
    monkeypatch.setattr(customer_policy, "_load_policy", lambda *_: (InvalidPolicy(), StateFeaturizer(), {}))
    decision = decide(machine, "Hello", [], "ppo42", {})
    assert decision["source"] == "rule"
    assert "ValueError" in decision["fallback_reason"]
    assert sum(decision["probabilities"].values()) == 1.0
