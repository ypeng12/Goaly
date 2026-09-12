from typing import Dict, Any, List, Optional
from .types import Phase, SOPState, ClaimRecord, PolicyHolder
from .grounded_data import grounded_data

class ContextBuilder:
    @staticmethod
    def build_system_prompt(
        state: SOPState,
        policyholder: Optional[PolicyHolder],
        active_case: Optional[ClaimRecord],
        is_frustrated: bool = False,
        is_refusal: bool = False,
        is_out_of_scope: bool = False
    ) -> Dict[str, Any]:
        """
        Builds the grounded prompt ensuring physical data shielding.
        Returns:
            dict with {
                "system_prompt": str,
                "data_shield_active": bool,
                "allowed_actions": List[str]
            }
        """
        phase = state.phase
        data_shield_active = (phase == Phase.VERIFY_ID)
        allowed_actions = []

        base_persona = (
            "You are an empathetic, highly professional insurance claims customer service representative. "
            "You strictly follow Standard Operating Procedures (SOP) while maintaining a warm, natural, and helpful conversation. "
            "You never fabricate facts or policy rules, answering only from grounded data provided to you."
        )

        prompt_sections = [base_persona]

        # 1. Scope Guardrail Directive
        if is_out_of_scope:
            prompt_sections.append(
                "CRITICAL INSTRUCTION - OUT OF SCOPE QUERY:\n"
                "- The caller's last message is OUT OF SCOPE for insurance customer service.\n"
                "- Politely explain that you are an insurance assistant and can only help with claims and policy questions.\n"
                "- Refuse to answer non-insurance questions (such as machine learning, coding, trivia).\n"
                "- If the caller has asked irrelevant questions repeatedly, offer to connect them to a human representative."
            )
            allowed_actions.append("REJECT_OUT_OF_SCOPE")
            return {
                "system_prompt": "\n\n".join(prompt_sections),
                "data_shield_active": data_shield_active,
                "allowed_actions": allowed_actions
            }

        # 2. Phase-specific instructions
        if phase == Phase.VERIFY_ID:
            allowed_actions.extend(["REQUEST_PII", "EMPATHIZE_AND_PERSUADE", "ESCALATE_IF_DEMANDED"])
            prompt_sections.append(
                "CURRENT PHASE: VERIFY_ID (Strict Security Gate)\n"
                "MANDATORY SECURITY RULES:\n"
                "1. DATA SHIELD ACTIVE: You MUST NOT disclose any claim details, denial reasons, amounts, or case status before verification is complete.\n"
                "2. Identity verification requires at least 3 matching PII items: Full name, Date of Birth (DOB), Phone number, Email address, Policy number, or SSN/National ID last 4 digits.\n"
                f"3. CURRENT ACCUMULATED PII: {state.accumulated_pii.model_dump(exclude_none=True)}\n"
                f"4. VERIFIED FIELDS SO FAR: {state.verified_fields} ({len(state.verified_fields)} / 3 required).\n"
            )

            if is_frustrated or is_refusal:
                prompt_sections.append(
                    "CALLER FRUSTRATION & SOP RECOVERY DIRECTIVE:\n"
                    "- The caller expressed frustration, annoyance, or refused verification.\n"
                    "- FIRST: Validate their frustration with genuine empathy and de-escalation.\n"
                    "- SECOND: Explain WHY verification is mandatory: Insurance claims contain protected personal health and financial data, and regulations (like HIPAA / privacy laws) strictly require identity verification to prevent unauthorized access.\n"
                    "- THIRD: Offer acceptable alternative identification fields (e.g. if they don't want to provide SSN, they can provide phone number, email, or policy number).\n"
                    "- FOURTH: Keep moving toward verification. Do NOT reveal claim details.\n"
                    "- If they explicitly insist on speaking with a human agent, honor their request and offer human transfer."
                )
            else:
                prompt_sections.append(
                    "CONVERSATIONAL GUIDANCE:\n"
                    "- Acknowledge any information provided.\n"
                    "- If caller provided future-phase details (like claim type or dates), do not answer claim questions yet, but warmly acknowledge that you will address that immediately once verified.\n"
                    "- Politely ask for the remaining missing verification fields needed to reach 3 items."
                )

        elif phase == Phase.RESOLVE_INTENT:
            allowed_actions.extend(["CONFIRM_VERIFICATION", "RESOLVE_CLAIM", "CLARIFY_AMBIGUITY"])
            prompt_sections.append(
                "CURRENT PHASE: RESOLVE_INTENT\n"
                "1. Verification is COMPLETE. The caller is verified as: "
                f"{policyholder.name if policyholder else 'Policyholder'}.\n"
                f"2. CROSS-PHASE MEMORY: {state.cross_phase_memory.model_dump(exclude_none=True)}\n"
                "3. INSTRUCTION: Acknowledge verification. Use the remembered hint from earlier utterances "
                "(e.g., if they mentioned a denied healthcare claim from January, address that specific claim directly) "
                "instead of asking them to repeat themselves from scratch!"
            )

        elif phase == Phase.PROCESS_CASE:
            allowed_actions.extend(["EXPLAIN_DENIAL", "EXPLAIN_REQUIRED_DOCUMENTS", "EXPLAIN_DEADLINES", "OFFER_POST_PROCESS"])
            prompt_sections.append("CURRENT PHASE: PROCESS_CASE (Grounded Claim Processing)\n")

            if active_case:
                guidance_data = grounded_data.get_document_guidance_for_claim(active_case)
                prompt_sections.append(
                    f"ACTIVE CLAIM DETAILS (GROUNDED TRUTH - ONLY STATE THESE FACTS):\n"
                    f"- Case ID: {active_case.case_id}\n"
                    f"- Case Type: {active_case.case_type}\n"
                    f"- Date Created: {active_case.created_at}\n"
                    f"- Status: {active_case.status}\n"
                    f"- Summary: {active_case.summary}\n"
                    f"- Denial Reason: {active_case.denial_reason}\n"
                    f"- Documents Needed: {', '.join(active_case.documents_needed)}\n"
                    f"- Appeal Deadline: {active_case.appeal_deadline}\n"
                    f"- Net Pay: ${active_case.net_pay}\n"
                    f"- Allowed Max: ${active_case.allowed_max_amount}\n"
                    f"- Expected Reimbursement: ${active_case.expected_reimbursement_amount}\n\n"
                    f"GROUNDED DOCUMENT GUIDELINES:\n"
                    f"- General Guideline: {guidance_data.get('default_guidance')}\n"
                    f"- Case Type Guideline: {guidance_data.get('case_type_guidance')}\n"
                    f"- Specific Document Guidelines: {guidance_data.get('document_guidance')}\n"
                    f"- Alternative Document Guidance: {guidance_data.get('document_alternative_guidance')}\n"
                    f"- Submission Method: Member portal or claim upload link.\n"
                    f"- Submission Timing: Submit missing documents within a week.\n"
                    f"- Processing Time After Submission: Usually less than a week once received.\n"
                )
                prompt_sections.append(
                    "CONVERSATIONAL RULES:\n"
                    "- Explain the denial reason clearly and empathetically.\n"
                    "- Specify exactly what documents are needed to appeal or reconsider the denial.\n"
                    "- Answer any follow-up questions accurately using only the grounded guidelines.\n"
                    "- Once the user's questions about this claim are resolved, check if they need anything else or offer to wrap up."
                )

        elif phase == Phase.POST_PROCESS:
            allowed_actions.extend(["OFFER_EMAIL_SUMMARY", "RECORD_USER_CHOICE", "CONCLUDE"])
            email_address = policyholder.email if policyholder else "your email on file"
            prompt_sections.append(
                "CURRENT PHASE: POST_PROCESS (Wrap-up & Email Summary)\n"
                "SOP MANDATORY STEP:\n"
                "1. You must offer to send the customer an email summary of this conversation.\n"
                f"2. Email on file: {email_address}\n"
                "3. The summary must include:\n"
                "   a) What was discussed (the claim inquiry and review).\n"
                f"   b) The claim status/outcome ({active_case.status if active_case else 'status'}).\n"
                "   c) Major follow-up items / next steps (e.g., submitting required documents before the appeal deadline).\n"
                "4. Explicitly ask the customer whether they would like this summary sent to their email or if they would prefer to skip it.\n"
                "5. The customer has full freedom to accept or decline."
            )

        elif phase == Phase.ESCALATED:
            allowed_actions.append("HANDOFF_TO_HUMAN")
            prompt_sections.append(
                "CURRENT PHASE: ESCALATED (Human Representative Transfer)\n"
                "- Acknowledge that you are initiating a transfer to a senior human claims representative.\n"
                "- Provide a warm closing and assure them that their case notes will be passed along."
            )

        elif phase == Phase.CONCLUDED:
            allowed_actions.append("FAREWELL")
            prompt_sections.append(
                "CURRENT PHASE: CONCLUDED\n"
                "- Thank the caller for contacting insurance customer support.\n"
                "- Wish them a great day and conclude warmly."
            )

        return {
            "system_prompt": "\n\n".join(prompt_sections),
            "data_shield_active": data_shield_active,
            "allowed_actions": allowed_actions
        }
