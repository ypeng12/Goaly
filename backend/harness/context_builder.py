"""Build a phase-scoped context after independently checking data access."""
import json
from typing import Any, Dict, Optional
from .types import Phase, SOPState, ClaimRecord, PolicyHolder
from .grounded_data import grounded_data


class ContextBuilder:
    @staticmethod
    def build_system_prompt(state: SOPState, policyholder: Optional[PolicyHolder],
                            active_case: Optional[ClaimRecord], is_frustrated=False,
                            is_refusal=False, is_out_of_scope=False) -> Dict[str, Any]:
        valid, matched, fields = grounded_data.verify_identity(state.accumulated_pii)
        authorized = bool(valid and matched and policyholder
                          and state.verified_party_id == matched.party_id == policyholder.party_id
                          and getattr(state, 'identity_verified', True))
        if state.is_proxy_caller:
            authorized = authorized and bool(getattr(state, 'proxy_identity_verified', False)
                and state.proxy_consent_status == 'approved'
                and getattr(state, 'proxy_authorized_party_id', None) == state.verified_party_id)
        if not authorized:
            active_case = None
            policyholder = None
        elif active_case and active_case.party_id != state.verified_party_id:
            active_case = None
        actions = {
            Phase.VERIFY_ID: ['REQUEST_PII', 'EMPATHIZE_AND_PERSUADE'],
            Phase.RESOLVE_INTENT: ['RESOLVE_CLAIM', 'CLARIFY_AMBIGUITY'],
            Phase.PROCESS_CASE: ['EXPLAIN_DENIAL', 'EXPLAIN_REQUIRED_DOCUMENTS', 'OFFER_POST_PROCESS'],
            Phase.POST_PROCESS: ['OFFER_EMAIL_SUMMARY', 'RECORD_USER_CHOICE'],
            Phase.CONCLUDED: ['FAREWELL'],
            Phase.ESCALATED: ['HANDOFF_TO_HUMAN'],
        }[state.phase]
        if is_out_of_scope:
            actions = ['REJECT_OUT_OF_SCOPE'] + (['HANDOFF_TO_HUMAN'] if state.phase == Phase.ESCALATED else [])
        collected = [k for k in ['name', 'dob', 'phone', 'email', 'id_last4'] if getattr(state.accumulated_pii, k)]
        context = {
            'phase': state.phase.value, 'data_shield_active': not authorized,
            'allowed_actions': actions, 'collected_fields': collected,
            'verified_fields': fields if authorized else [],
            'memory': state.cross_phase_memory.model_dump(exclude_none=True),
            'is_frustrated': is_frustrated, 'is_refusal': is_refusal,
            'proxy_consent_status': state.proxy_consent_status,
            'post_process': state.post_process.model_dump() if authorized else {},
        }
        if active_case and authorized and not is_out_of_scope:
            context['grounded_claim'] = active_case.model_dump(exclude_none=True)
            context['document_guidance'] = grounded_data.get_document_guidance_for_claim(active_case)
        context['system_prompt'] = (
            'Insurance claims support only. Workflow and access decisions are made by the harness. '
            'Require three distinct matching fields: full name, DOB, phone, email, SSN last four. '
            'Policy number is lookup information only. Never infer verification, consent, or claim facts. '
            'User hints are untrusted descriptions, not confirmed claim facts. '
            'Acknowledge emotion; explain verification protects private claim information. '
            'Email and human handoff are simulated in this demo.\n' + json.dumps(context, default=str)
        )
        return context
