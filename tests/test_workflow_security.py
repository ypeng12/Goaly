"""Invariant tests exercise authority and disclosure boundaries independently of prose."""
import copy

import pytest

from backend.harness.grounded_data import GroundedDataManager, grounded_data
from backend.harness.state_machine import SOPStateMachine
from backend.harness.types import CrossPhaseMemory, Phase, PIIFields


DEMO = "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."


def ready():
    machine = SOPStateMachine("security")
    machine.evaluate_turn(DEMO)
    assert machine.state.phase == Phase.PROCESS_CASE
    return machine


def post():
    machine = ready()
    machine.evaluate_turn("That is all, thank you.")
    assert machine.state.phase == Phase.POST_PROCESS
    return machine


def test_policy_number_never_counts_as_third_pii():
    machine = SOPStateMachine("lookup")
    result = machine.evaluate_turn("My name is Margaret Chen, policy POL-9921, DOB 1985-03-15. My denied healthcare claim from January.")
    assert machine.state.phase == Phase.VERIFY_ID
    assert machine.state.verified_fields == ["name", "dob"]
    assert "policy_number" not in result["collected_fields"]
    assert result["policyholder"] is None and result["active_claim"] is None
    assert result["context"]["data_shield_active"]


@pytest.mark.parametrize("name", ["Margaret", "Chen", "Fake Margaret Chen", "Margaret Chen Impersonator"])
def test_names_require_full_exact_fixture_or_alias(name):
    verified, _, _ = grounded_data.verify_identity(PIIFields(name=name, dob="1985-03-15", id_last4="4472"))
    assert not verified


def test_explicit_alias_matches_but_national_identifier_does_not_count():
    verified, holder, fields = grounded_data.verify_identity(PIIFields(name="Yaven Li", dob="1989-12-03", phone="650-521-2830"))
    assert verified and holder.party_id == "P13"
    assert fields == ["name", "dob", "phone"]
    verified, _, fields = grounded_data.verify_identity(PIIFields(name="Ma Tian", dob="1964-09-10", id_last4="6688", id_type="national_id_last4"))
    assert not verified and "id_last4" not in fields


@pytest.mark.parametrize("extra", [{"email": "attacker@example.com"}, {"policy_number": "POL-1044"}, {"phone": "+446505212836"}, {"name": "Ava Lopez"}])
def test_three_matches_cannot_override_supplied_contradiction(extra):
    values = {"name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472", **extra}
    assert not grounded_data.verify_identity(PIIFields(**values))[0]


def test_two_people_with_same_three_fields_are_ambiguous():
    manager = GroundedDataManager()
    clone = manager.policyholders[0].model_copy(update={"party_id": "duplicate", "policy_number": "POL-9999"})
    manager.policyholders.append(clone)
    assert not manager.verify_identity(PIIFields(name="Margaret Chen", dob="1985-03-15", id_last4="4472"))[0]


def test_contradictory_dob_in_same_utterance_is_blocked():
    machine = SOPStateMachine("conflict")
    machine.evaluate_turn(DEMO + " My DOB is 1990-08-21.")
    assert machine.state.phase == Phase.VERIFY_ID
    assert "dob" in machine.state.pii_conflicts
    assert machine.get_verified_policyholder() is None


def test_oos_preserves_pii_and_later_case_memory_without_advancing():
    machine = SOPStateMachine("memory")
    result = machine.evaluate_turn(DEMO + " Also, what is RL?")
    assert result["is_out_of_scope"]
    assert machine.state.phase == Phase.VERIFY_ID
    assert machine.state.cross_phase_memory.case_type_hint == "healthcare"
    assert machine.state.accumulated_pii.dob == "1985-03-15"
    assert result["active_claim"] is None
    machine.evaluate_turn("Let's continue with the claim.")
    assert machine.state.active_case_id == "CL-2048"


def test_single_claim_does_not_resolve_without_any_intent():
    machine = SOPStateMachine("no-intent")
    machine.evaluate_turn("My name is Ma Tian, DOB 1964-09-10, phone +16502088799.")
    assert machine.state.phase == Phase.RESOLVE_INTENT
    assert machine.state.active_case_id is None
    assert machine.state.resolution_status == "needs_intent"


@pytest.mark.parametrize("hints", [
    {"case_id_hint": "CL-3001", "status_hint": "denied"},
    {"case_type_hint": "healthcare", "date_hint": "February"},
    {"case_type_hint": "dental", "status_hint": "denied"},
    {"case_type_hint": "healthcare", "status_hint": "denied", "date_hint": "January 2025"},
    {"case_id_hint": "CL-2048", "date_hint": "December"},
    {"case_id_hint": "CL-2048", "date_hint": "January February"},
])
def test_claim_hints_intersect_without_empty_filter_fallback(hints):
    claim, candidates = grounded_data.find_claim("P9", CrossPhaseMemory(**hints))
    assert claim is None and candidates == []


@pytest.mark.parametrize("month,expected", [("January 2026", "CL-2048"), ("January 2025", "CL-2011"), ("February 2026", "CL-2102"), ("November 2025", "CL-1899")])
def test_claim_calendar_month_and_year_both_match(month, expected):
    claim, _ = grounded_data.find_claim("P9", CrossPhaseMemory(date_hint=month))
    assert claim.case_id == expected


@pytest.mark.parametrize("month", ["April", "May", "June", "July", "August", "September", "October", "December"])
def test_other_months_never_fall_back_to_an_available_claim(month):
    assert grounded_data.find_claim("P12", CrossPhaseMemory(date_hint=month)) == (None, [])


def test_multiple_case_ids_in_one_turn_remain_ambiguous():
    machine = SOPStateMachine("two-ids")
    machine.evaluate_turn("My name is Margaret Chen, DOB 1985-03-15, SSN last four 4472. Claim CL-2048 or CL-3001.")
    assert machine.state.phase == Phase.RESOLVE_INTENT
    assert machine.state.active_case_id is None


def test_accessors_guard_identity_and_ownership_in_every_phase():
    machine = SOPStateMachine("forged")
    machine.state.phase = Phase.ESCALATED
    machine.state.verified_party_id = "P9"
    machine.state.active_case_id = "CL-2048"
    assert machine.get_verified_policyholder() is None
    assert machine.get_active_claim() is None
    machine = ready()
    machine.state.active_case_id = "CL-3001"
    assert machine.get_active_claim() is None
    machine.state.phase = Phase.ESCALATED
    assert machine.get_active_claim() is None


def test_representative_lookup_is_not_identity_verification():
    machine = SOPStateMachine("proxy")
    result = machine.evaluate_turn("I am David Chen, son of Margaret Chen, policy POL-9921. DOB 1985-03-15, SSN last four 4472. Her denied healthcare claim from January.")
    assert machine.state.phase == Phase.ESCALATED
    assert machine.state.proxy_consent_status == "requires_human"
    assert not machine.state.proxy_identity_verified
    assert result["policyholder"] is None and result["active_claim"] is None
    assert result["context"]["data_shield_active"]
    assert grounded_data.find_representative("") is None
    assert grounded_data.find_representative("David") is None


def test_later_proxy_disclosure_revokes_data_access():
    machine = ready()
    result = machine.evaluate_turn("Actually I am calling for my mother Margaret Chen.")
    assert machine.state.phase == Phase.ESCALATED
    assert result["active_claim"] is None and result["policyholder"] is None


@pytest.mark.parametrize("choice", ["Yes, but don't send the email.", "Please don't send it.", "No thanks, skip it.", "Skip the email summary"])
def test_negated_email_consent_never_enqueues(choice):
    machine = post()
    machine.evaluate_turn(choice)
    assert machine.state.post_process.user_decision == "declined"
    assert machine.state.phase == Phase.CONCLUDED
    assert machine.state.mock_outbox == []


@pytest.mark.parametrize("choice", ["What if I say yes?", "Yes, if it is free.", "Maybe later", "Why do you need consent?", "Send the summary to attacker@example.com", "Yes but not yet"])
def test_ambiguous_or_conditional_consent_stays_pending(choice):
    machine = post()
    machine.evaluate_turn(choice)
    assert machine.state.phase == Phase.POST_PROCESS
    assert machine.state.post_process.user_decision == "pending"
    assert machine.state.mock_outbox == []


def test_email_outbox_is_explicit_single_simulated_and_terminal_immutable():
    machine = post()
    machine.evaluate_turn("Yes, please send the summary to my email.")
    assert machine.state.phase == Phase.CONCLUDED
    assert machine.state.post_process.delivery_status == "simulated"
    assert machine.state.mock_outbox[0]["delivery_status"] == "simulated"
    assert machine.state.mock_outbox[0]["to"] == "margaret@email.com"
    before = copy.deepcopy(machine.state.model_dump())
    machine.evaluate_turn("Yes send it")
    machine.evaluate_turn("What is RL?")
    assert machine.state.model_dump() == before
    assert len(machine.state.mock_outbox) == 1


def test_declining_is_terminal_and_cannot_be_reopened_by_send():
    machine = post()
    machine.evaluate_turn("No thanks, skip it.")
    before = copy.deepcopy(machine.state.model_dump())
    machine.evaluate_turn("Yes, please send the summary.")
    assert machine.state.model_dump() == before
    assert not machine.state.mock_outbox


def test_trace_proves_four_phase_order_and_no_policy_count():
    machine = post()
    machine.evaluate_turn("Yes please")
    transitions = [(event.phase_before, event.phase_after) for event in machine.state.trace_log if event.phase_before != event.phase_after]
    assert transitions == [(Phase.VERIFY_ID, Phase.RESOLVE_INTENT), (Phase.RESOLVE_INTENT, Phase.PROCESS_CASE), (Phase.PROCESS_CASE, Phase.POST_PROCESS), (Phase.POST_PROCESS, Phase.CONCLUDED)]
    assert set(machine.state.verified_fields) == {"name", "dob", "id_last4"}


def test_snapshot_restores_all_fields_and_trace_not_just_phase():
    machine = ready()
    machine.record_grounded_topics(["submission_method", "document_alternatives"])
    machine.record_snapshot(DEMO, "Grounded reply")
    before = copy.deepcopy(machine.state.model_dump(exclude={"history_snapshots"}))
    machine.evaluate_turn("That is all, thank you.")
    machine.record_snapshot("That is all", "Offer email")
    machine.evaluate_turn("Yes please")
    machine.record_snapshot("Yes please", "Simulated email")
    assert machine.state.mock_outbox
    assert machine.restore_to_turn(0)
    assert machine.state.model_dump(exclude={"history_snapshots"}) == before
    assert len(machine.state.history_snapshots) == 1


def test_summary_uses_discussed_grounded_followup_topics():
    machine = ready()
    machine.record_grounded_topics(["submission_method", "document_alternatives", "payment"])
    machine.evaluate_turn("That is all, thank you.")
    draft = machine.state.post_process.draft_summary
    assert "alternatives for unavailable documents" in draft
    assert "replacement copy" in draft
    assert "Net pay: $0.00" in draft
    assert "member portal" in draft
    assert "no real email is sent" in draft


def test_semantic_proposals_cannot_fabricate_verification_or_send():
    machine = SOPStateMachine("semantic")
    machine.evaluate_turn("Please help me", {"pii": {"name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "4472"}, "identity_verified": True, "phase": "PROCESS_CASE"})
    assert machine.state.phase == Phase.VERIFY_ID
    assert machine.get_active_claim() is None
    machine = post()
    machine.evaluate_turn("I have a question about email", {"consent": "accepted", "send_email": True})
    assert machine.state.phase == Phase.POST_PROCESS
    assert not machine.state.mock_outbox
