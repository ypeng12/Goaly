import re
from typing import Dict, Any, List
from .base import BaseEngine
from ..harness.types import Phase, PolicyHolder, ClaimRecord
from ..harness.grounded_data import grounded_data

class MockEngine(BaseEngine):
    """
    Deterministic domain reasoning engine that produces grounded, natural, and compliant
    responses for testing and zero-API-key operation.
    """

    def generate_response(
        self,
        user_text: str,
        state_result: Dict[str, Any],
        history: List[Dict[str, str]]
    ) -> str:
        context = state_result.get("context", {})
        allowed_actions = context.get("allowed_actions", [])
        policyholder: PolicyHolder = state_result.get("policyholder")
        active_claim: ClaimRecord = state_result.get("active_claim")
        is_oos = state_result.get("is_out_of_scope", False)
        is_frustrated = state_result.get("is_frustrated", False)
        is_refusal = state_result.get("is_refusal", False)
        demands_human = state_result.get("demands_human", False)

        user_lower = user_text.lower()

        # 1. Out of Scope Rejection
        if "REJECT_OUT_OF_SCOPE" in allowed_actions or is_oos:
            # Check if this triggered escalation
            if state_result.get("demands_human") or (state_result.get("transition_note") and "human escalation" in state_result["transition_note"].lower()):
                return (
                    "I am an insurance claims assistant and can only assist with claims and policy questions. "
                    "Since you have further questions outside of my scope, I am transferring you to a human customer service representative who can assist you directly. Please hold on."
                )
            return (
                "I apologize, but as an insurance claims support assistant, I can only assist with policy, coverage, and claim inquiries. "
                "I am unable to answer general or technical questions such as machine learning or reinforcement learning. "
                "If you have any questions regarding your insurance claims or policy, I would be happy to help, or I can connect you with a human representative."
            )

        # 2. Human Escalation Demand
        if demands_human or "HANDOFF_TO_HUMAN" in allowed_actions:
            name_greet = f", {policyholder.name}" if policyholder else ""
            return (
                f"I completely understand{name_greet}. I am transferring your call right now to a senior human claims representative who will be able to assist you directly. "
                "Please stay on the line while I connect you."
            )

        # 3. VERIFY_ID Phase
        if "Unauthorized proxy caller" in state_result.get("transition_note", ""):
            return (
                "I understand you are calling regarding a policyholder's claim. However, because claim files contain protected personal health and financial data, "
                "privacy regulations strictly require us to verify your authorization before disclosing any information. "
                "We cannot disclose any claim information to unauthorized third parties without verified consent on file. "
                "If you are an authorized representative, please have the policyholder contact us directly or submit an authorization designation form."
            )

        if "REQUEST_PII" in allowed_actions:
            if is_frustrated or is_refusal:
                return (
                    "I completely understand your frustration, and I apologize for any inconvenience. "
                    "Because claim files contain protected personal health and sensitive financial information, "
                    "insurance privacy regulations strictly require us to verify your identity before we can disclose any claim details. "
                    "We require 3 verification items to safeguard your account. If you prefer, instead of SSN, you may verify using your "
                    "phone number, email address, or policy number. Could you please share one of those so I can safely pull up your claim?"
                )
            else:
                # Normal PII request
                name = policyholder.name if policyholder else "there"
                return (
                    f"Hello! Thank you for contacting claims support. To protect your personal health information under our privacy policy, "
                    "I need to verify your identity with at least 3 pieces of information (such as your full name, policy number, date of birth, phone number, email, or the last four digits of your SSN). "
                    "Could you please share your verification details?"
                )

        # 4. RESOLVE_INTENT & PROCESS_CASE Transition
        # This handles Margaret Chen's demo test case where identity is verified and cross-phase memory had "denied healthcare claim from January"
        if active_claim and ("EXPLAIN_DENIAL" in allowed_actions or "RESOLVE_CLAIM" in allowed_actions):
            # Check if proxy caller
            is_proxy = state_result.get("is_proxy_caller", False)
            rep_name = state_result.get("proxy_rep_name", "David")
            caller_name = rep_name if is_proxy else (policyholder.name.split()[0] if policyholder else "there")
            greeting_prefix = (
                f"Thank you for verifying, {caller_name}. As an authorized representative for {policyholder.name if policyholder else 'the policyholder'}, I have opened the file.\n\n"
                if is_proxy else
                f"Thank you for verifying your details, {caller_name}. I have your account open.\n\n"
            )

            # If user asks specific follow-up questions about documents or submission
            has_timing = any(q in user_lower for q in ["how soon", "when do i need", "when should", "deadline to submit"])
            has_method = any(q in user_lower for q in ["how do i submit", "where do i submit", "how to submit", "how do i send", "how to send", "upload"])
            
            if has_timing and has_method:
                return (
                    f"For claim {active_claim.case_id}, please submit the missing documents ({', '.join(active_claim.documents_needed)}) within a week. "
                    f"The best method is to upload them directly via the member portal or the claim upload link so they attach directly to your file. "
                    f"Keep in mind that the final appeal deadline is {active_claim.appeal_deadline}."
                )
            elif has_timing:
                return (
                    f"For claim {active_claim.case_id}, please submit the missing documents ({', '.join(active_claim.documents_needed)}) within a week. "
                    f"Keep in mind that the final appeal deadline for this claim is {active_claim.appeal_deadline}."
                )
            elif has_method:
                return (
                    f"For claim {active_claim.case_id}, the best starting point is to upload them directly via the member portal or claim upload link. "
                    f"Each file should be clear and legible. If online upload is not available, we can help arrange fax or mail submission."
                )
            elif any(q in user_lower for q in ["alternative", "don't have", "cannot get", "substitute", "missing report", "what if we cannot"]):
                return (
                    f"If the original pathology report is not immediately available, you can request a replacement copy from the hospital or treating lab. "
                    f"A complete, readable scan is acceptable. For the office note, ask the clinic for a visit summary or have them fax the chart directly. "
                    f"If none of those can be obtained, a human claims representative can review manual alternatives with you."
                )
            elif any(q in user_lower for q in ["how much", "amount", "net pay", "reimbursement", "cost", "fee"]):
                return (
                    f"For claim {active_claim.case_id}, the allowed maximum amount is ${active_claim.allowed_max_amount}, and the net fee is ${active_claim.net_fee}. "
                    f"Because the claim was denied due to missing documents, the net pay and expected reimbursement are currently ${active_claim.net_pay}."
                )

            # Default / Opening response for active claim
            docs_needed = " and the ".join(active_claim.documents_needed)
            return (
                f"{greeting_prefix}"
                f"Regarding your {active_claim.case_type} claim ({active_claim.case_id}) from {active_claim.created_at}: "
                f"the claim was denied because {active_claim.denial_reason}. Specifically, we still need the {docs_needed}.\n\n"
                f"You have until {active_claim.appeal_deadline} to submit these documents for an appeal. "
                f"Would you like guidance on how to submit these files, or do you have any other questions regarding this claim?"
            )

        # 5. POST_PROCESS Phase (Offer email summary)
        if "OFFER_EMAIL_SUMMARY" in allowed_actions:
            caller_name = policyholder.name.split()[0] if policyholder else "there"
            user_email = policyholder.email if policyholder else "your email address on file"
            claim_id = active_claim.case_id if active_claim else "your claim"
            return (
                f"Before we conclude today, {caller_name}, I'd like to offer to send an email summary of our conversation to {user_email}. "
                f"This will summarize what was discussed, the status of claim {claim_id} (denied), and your next steps (uploading the pathology report and office note before the March 18, 2026 appeal deadline). "
                f"Would you like me to send this email summary, or would you prefer to skip it?"
            )

        # 6. CONCLUDED Phase
        if "FAREWELL" in allowed_actions:
            caller_name = policyholder.name.split()[0] if policyholder else ""
            user_decision = state_result.get("post_process", {}).get("user_decision") if isinstance(state_result.get("post_process"), dict) else None
            # Check decision
            if any(w in user_lower for w in ["yes", "sure", "please", "send", "yep"]):
                email_dest = policyholder.email if policyholder else "your email on file"
                return (
                    f"Great! I have sent the summary to {email_dest}. "
                    f"Thank you for contacting claims support today{f', {caller_name}' if caller_name else ''}. Best of luck with your document submission, and have a wonderful day!"
                )
            else:
                return (
                    f"Understood, I will skip sending the email summary. "
                    f"Thank you for contacting claims support today{f', {caller_name}' if caller_name else ''}. Have a wonderful day!"
                )

        # Fallback polite response
        return "I am here to assist with your insurance claim. How may I assist you further?"
