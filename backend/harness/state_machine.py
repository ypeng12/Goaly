import datetime
from typing import Dict, Any, Optional, Tuple, List
from .types import (
    Phase, SOPState, PIIFields, CrossPhaseMemory, TraceEvent,
    PostProcessState, ClaimRecord, PolicyHolder
)
from .grounded_data import grounded_data
from .extractor import UtteranceExtractor
from .context_builder import ContextBuilder

class SOPStateMachine:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = SOPState(session_id=session_id)
        self.extractor = UtteranceExtractor()

    def _add_trace(self, gate: str, passed: bool, details: str, phase_before: Phase):
        event = TraceEvent(
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            phase_before=phase_before,
            phase_after=self.state.phase,
            gate_evaluated=gate,
            gate_passed=passed,
            details=details,
            data_shield_active=(self.state.phase == Phase.VERIFY_ID)
        )
        self.state.trace_log.append(event)

    def get_verified_policyholder(self) -> Optional[PolicyHolder]:
        if not self.state.verified_party_id:
            return None
        for ph in grounded_data.policyholders:
            if ph.party_id == self.state.verified_party_id:
                return ph
        return None

    def get_active_claim(self) -> Optional[ClaimRecord]:
        if not self.state.active_case_id:
            return None
        return grounded_data.get_claim_by_id(self.state.active_case_id)

    def generate_draft_email(self, policyholder: PolicyHolder, claim: Optional[ClaimRecord]) -> str:
        if not claim:
            claim_text = "Your recent insurance inquiry."
            status_text = "Under review."
            next_steps = "Please review your online portal for updates."
        else:
            claim_text = f"Claim Reference: {claim.case_id} ({claim.case_type.capitalize()} Claim created on {claim.created_at})"
            status_text = f"Status: {claim.status.upper()}. Reason: {claim.denial_reason or claim.summary}."
            if claim.documents_needed:
                docs = ", ".join(claim.documents_needed)
                deadline = f"Appeal deadline: {claim.appeal_deadline}." if claim.appeal_deadline else ""
                next_steps = (
                    f"Please submit the following required documents ({docs}) via the member portal or upload link within a week. "
                    f"{deadline} Average review time once received is usually less than a week."
                )
            else:
                next_steps = "No additional documents needed at this time."

        email_body = (
            f"Dear {policyholder.name},\n\n"
            f"Here is a summary of our discussion today regarding your insurance claim:\n\n"
            f"1. What Was Discussed:\n   {claim_text}\n\n"
            f"2. Claim Status & Outcome:\n   {status_text}\n\n"
            f"3. Major Follow-up Items & Next Steps:\n   {next_steps}\n\n"
            f"If you have any further questions, you can access your account through the member portal or reach our support team.\n\n"
            f"Sincerely,\nInsurance Claims Support Team"
        )
        return email_body

    def evaluate_turn(self, user_text: str) -> Dict[str, Any]:
        """
        Processes one conversation turn through the SOP Harness.
        Returns a dict containing:
        - 'is_out_of_scope': bool
        - 'is_frustrated': bool
        - 'is_refusal': bool
        - 'demands_human': bool
        - 'context': prompt context from ContextBuilder
        - 'policyholder': Optional[PolicyHolder]
        - 'active_claim': Optional[ClaimRecord]
        - 'transition_note': str
        """
        phase_before = self.state.phase

        # 1. Out of Scope Check
        is_oos = self.extractor.is_out_of_scope(user_text)
        if is_oos:
            self.state.out_of_scope_count += 1
            if self.state.out_of_scope_count >= 2:
                self.state.phase = Phase.ESCALATED
                self._add_trace(
                    gate="OUT_OF_SCOPE_GUARD",
                    passed=False,
                    details=f"Consecutive out-of-scope queries ({self.state.out_of_scope_count}). Triggered human escalation.",
                    phase_before=phase_before
                )
            else:
                self._add_trace(
                    gate="OUT_OF_SCOPE_GUARD",
                    passed=False,
                    details=f"Out-of-scope query detected ('{user_text[:50]}...'). Rejection required.",
                    phase_before=phase_before
                )

            ph = self.get_verified_policyholder()
            ac = self.get_active_claim()
            context = ContextBuilder.build_system_prompt(
                self.state, ph, ac, is_out_of_scope=True
            )
            return {
                "is_out_of_scope": True,
                "is_frustrated": False,
                "is_refusal": False,
                "demands_human": False,
                "context": context,
                "policyholder": ph,
                "active_claim": ac,
                "transition_note": "Out-of-scope query handled."
            }
        else:
            # Reset consecutive out of scope counter if user returns to scope
            self.state.out_of_scope_count = 0

        # 2. Extract Emotion, Refusal & Human Escalation Signals
        emotions = self.extractor.detect_emotion_and_intent(user_text)
        is_frustrated = emotions["is_frustrated"]
        is_refusal = emotions["is_refusal"]
        demands_human = emotions["demands_human"]

        if demands_human:
            self.state.phase = Phase.ESCALATED
            self._add_trace(
                gate="HUMAN_ESCALATION",
                passed=True,
                details="Caller explicitly demanded human representative.",
                phase_before=phase_before
            )
            ph = self.get_verified_policyholder()
            ac = self.get_active_claim()
            context = ContextBuilder.build_system_prompt(self.state, ph, ac)
            return {
                "is_out_of_scope": False,
                "is_frustrated": is_frustrated,
                "is_refusal": is_refusal,
                "demands_human": True,
                "context": context,
                "policyholder": ph,
                "active_claim": ac,
                "transition_note": "Escalated to human representative upon request."
            }

        # 3. Always Extract and Store Cross-Phase Memory
        hints = self.extractor.extract_cross_phase_hints(user_text)
        if hints.case_type_hint:
            self.state.cross_phase_memory.case_type_hint = hints.case_type_hint
        if hints.status_hint:
            self.state.cross_phase_memory.status_hint = hints.status_hint
        if hints.date_hint:
            self.state.cross_phase_memory.date_hint = hints.date_hint
        if hints.raw_utterance_snippet:
            self.state.cross_phase_memory.raw_utterance_snippet = hints.raw_utterance_snippet

        # 4. Phase-Specific State Handling
        if self.state.phase == Phase.VERIFY_ID:
            # Extract PII from user text
            new_pii = self.extractor.extract_pii(user_text)

            # Update accumulated PII
            if new_pii.name: self.state.accumulated_pii.name = new_pii.name
            if new_pii.policy_number: self.state.accumulated_pii.policy_number = new_pii.policy_number
            if new_pii.dob: self.state.accumulated_pii.dob = new_pii.dob
            if new_pii.id_last4: self.state.accumulated_pii.id_last4 = new_pii.id_last4
            if new_pii.phone: self.state.accumulated_pii.phone = new_pii.phone
            if new_pii.email: self.state.accumulated_pii.email = new_pii.email

            # Evaluate verification gate
            is_verified, ph_match, matched_fields = grounded_data.verify_identity(self.state.accumulated_pii)

            if is_verified and ph_match:
                self.state.verified_party_id = ph_match.party_id
                self.state.verified_fields = matched_fields
                self.state.phase = Phase.RESOLVE_INTENT
                self._add_trace(
                    gate="VERIFY_ID_GATE",
                    passed=True,
                    details=f"Verified caller as {ph_match.name} with {len(matched_fields)} fields: {', '.join(matched_fields)}.",
                    phase_before=phase_before
                )

                # Proactive Resolution using cross-phase memory!
                claim_match, candidates = grounded_data.find_claim(ph_match.party_id, self.state.cross_phase_memory)
                if claim_match:
                    self.state.active_case_id = claim_match.case_id
                    self.state.phase = Phase.PROCESS_CASE
                    self._add_trace(
                        gate="RESOLVE_INTENT_GATE",
                        passed=True,
                        details=f"Proactively auto-resolved case {claim_match.case_id} ({claim_match.case_type}, {claim_match.status}) from cross-phase memory hints.",
                        phase_before=Phase.RESOLVE_INTENT
                    )
            else:
                # Not verified yet
                if is_refusal or is_frustrated:
                    self.state.refusal_count += 1
                    if self.state.refusal_count >= 3:
                        self.state.phase = Phase.ESCALATED
                        self._add_trace(
                            gate="VERIFY_ID_GATE",
                            passed=False,
                            details="Repeated verification refusals (3+). Escalating to human.",
                            phase_before=phase_before
                        )
                    else:
                        self._add_trace(
                            gate="VERIFY_ID_GATE",
                            passed=False,
                            details="Caller expressed frustration/refusal. Applying empathy, privacy explanation, and alternate ID options.",
                            phase_before=phase_before
                        )
                else:
                    self._add_trace(
                        gate="VERIFY_ID_GATE",
                        passed=False,
                        details=f"PII insufficient. Collected {len(matched_fields)}/3 required fields. Requesting remaining.",
                        phase_before=phase_before
                    )

        elif self.state.phase == Phase.RESOLVE_INTENT:
            ph = self.get_verified_policyholder()
            if ph:
                claim_match, candidates = grounded_data.find_claim(ph.party_id, self.state.cross_phase_memory)
                if claim_match:
                    self.state.active_case_id = claim_match.case_id
                    self.state.phase = Phase.PROCESS_CASE
                    self._add_trace(
                        gate="RESOLVE_INTENT_GATE",
                        passed=True,
                        details=f"Resolved active case {claim_match.case_id}.",
                        phase_before=phase_before
                    )

        elif self.state.phase == Phase.PROCESS_CASE:
            # Check if user indicates they have no more questions or want to wrap up
            text_l = user_text.lower()
            wrap_up_signals = [
                "no more question", "that's all", "that is all", "i'm good", "im good",
                "thank you that's all", "thanks that helps", "nothing else", "no that's it",
                "no that is it", "no thanks", "nope that's it", "wrap up"
            ]
            if any(sig in text_l for sig in wrap_up_signals):
                self.state.phase = Phase.POST_PROCESS
                ph = self.get_verified_policyholder()
                ac = self.get_active_claim()
                draft = self.generate_draft_email(ph, ac) if ph else ""
                self.state.post_process.email_offered = True
                self.state.post_process.draft_summary = draft
                self.state.post_process.user_decision = "pending"
                self._add_trace(
                    gate="PROCESS_CASE_GATE",
                    passed=True,
                    details="User signaled questions answered. Transitioned to POST_PROCESS to offer email summary.",
                    phase_before=phase_before
                )

        elif self.state.phase == Phase.POST_PROCESS:
            # Detect consent
            consent = self.extractor.detect_post_process_consent(user_text)
            ph = self.get_verified_policyholder()
            if consent == "accepted":
                self.state.post_process.user_decision = "accepted"
                self.state.post_process.sent_to = ph.email if ph else "email on file"
                self.state.phase = Phase.CONCLUDED
                self._add_trace(
                    gate="POST_PROCESS_CONSENT",
                    passed=True,
                    details=f"Caller accepted email summary. Marked sent to {self.state.post_process.sent_to}.",
                    phase_before=phase_before
                )
            elif consent == "declined":
                self.state.post_process.user_decision = "declined"
                self.state.phase = Phase.CONCLUDED
                self._add_trace(
                    gate="POST_PROCESS_CONSENT",
                    passed=True,
                    details="Caller declined email summary. Skipped and concluded.",
                    phase_before=phase_before
                )
            else:
                self._add_trace(
                    gate="POST_PROCESS_CONSENT",
                    passed=False,
                    details="Consent choice unclear; re-confirming.",
                    phase_before=phase_before
                )

        ph = self.get_verified_policyholder()
        ac = self.get_active_claim()
        context = ContextBuilder.build_system_prompt(
            self.state, ph, ac,
            is_frustrated=is_frustrated,
            is_refusal=is_refusal,
            is_out_of_scope=False
        )

        return {
            "is_out_of_scope": False,
            "is_frustrated": is_frustrated,
            "is_refusal": is_refusal,
            "demands_human": False,
            "context": context,
            "policyholder": ph,
            "active_claim": ac,
            "transition_note": f"Phase: {phase_before.value} -> {self.state.phase.value}"
        }
