"""Uncertainty is service orientation, not a failed case lookup budget."""
import pytest

from backend.harness.customer_policy import CONTROLLER_MODES, decide, record_execution
from backend.harness.customer_service import render_policy_reply
from backend.harness.state_machine import SOPStateMachine
from backend.harness.types import Phase


YA_WEN = "My name is Ya Wen Li. DOB 1989-12-03. Phone +16505212830."
MARGARET = "My name is Margaret Chen. DOB 1985-03-15. SSN last four 4472."


def execute(machine, message, runtime, mode="rule"):
    result = machine.evaluate_turn(message)
    decision = decide(machine, message, [], mode, runtime)
    reply = render_policy_reply(machine, message, result, {}, decision)
    record_execution(runtime, decision)
    machine.record_snapshot(message, reply)
    return decision


@pytest.mark.parametrize("mode", CONTROLLER_MODES)
def test_repeated_uncertainty_never_consumes_lookup_budget_or_allows_handoff(mode):
    machine = SOPStateMachine("uncertain-customer")
    runtime = {}
    for message in (YA_WEN, "i don;t know", "claim", "I hate you", "i don't know", "claim"):
        decision = execute(machine, message, runtime, mode)
        assert machine.state.phase == Phase.RESOLVE_INTENT
        assert machine.state.resolution_status == "needs_intent"
        assert machine.get_active_claim() is None
        assert decision["observation"]["resolution_attempts"] == 0
        assert "ESCALATE_HUMAN" not in decision["allowed_actions"]
        assert decision["probabilities"]["ESCALATE_HUMAN"] == 0
        assert runtime["resolution_attempts"] == 0
        assert not machine.state.mock_outbox


@pytest.mark.parametrize("mode", CONTROLLER_MODES)
def test_old_retry_counter_cannot_escalate_a_customer_without_an_intent(mode):
    machine = SOPStateMachine("old-orientation-counter")
    machine.evaluate_turn(YA_WEN)
    runtime = {"last_phase": "RESOLVE_INTENT", "resolution_attempts": 99}
    decision = execute(machine, "i don't know", runtime, mode)
    assert "ESCALATE_HUMAN" not in decision["allowed_actions"]
    assert decision["observation"]["resolution_attempts"] == 0
    assert runtime["resolution_attempts"] == 0


@pytest.mark.parametrize("hint,status", [
    ("Claim status", "ambiguous"),
    ("My denied healthcare claim from January 2099", "no_match"),
])
def test_real_unresolved_lookups_retain_bounded_handoff(hint, status):
    machine = SOPStateMachine("unresolved-case")
    runtime = {}
    execute(machine, MARGARET, runtime)
    execute(machine, "i don't know", runtime)
    assert runtime["resolution_attempts"] == 0
    first = execute(machine, hint, runtime)
    assert machine.state.resolution_status == status
    assert runtime["resolution_attempts"] == 1
    assert "ESCALATE_HUMAN" not in first["allowed_actions"]
    second = execute(machine, "The same claim", runtime)
    assert runtime["resolution_attempts"] == 2
    assert "ESCALATE_HUMAN" not in second["allowed_actions"]
    last = execute(machine, "The same claim", runtime)
    assert last["selected_action"] == "ESCALATE_HUMAN"
    assert machine.state.phase == Phase.ESCALATED
    assert machine.get_active_claim() is None


def test_empathy_followed_by_a_real_lookup_response_counts_once():
    machine = SOPStateMachine("empathetic-lookup")
    runtime = {}
    execute(machine, MARGARET, runtime)
    decision = execute(machine, "I'm angry. My healthcare claim from January 2099", runtime)
    assert machine.state.resolution_status == "no_match"
    assert decision["selected_action"] == "ACK_EMOTION"
    assert decision["execution"]["followup_kind"] == "resolve_intent"
    assert runtime["resolution_attempts"] == 1


def test_requested_handoff_still_takes_precedence_during_orientation():
    machine = SOPStateMachine("requested-handoff")
    runtime = {}
    execute(machine, YA_WEN, runtime)
    decision = execute(machine, "Please connect me with a human representative", runtime)
    assert machine.state.phase == Phase.ESCALATED
    assert decision["source"] == "sop"
    assert decision["selected_action"] is None
