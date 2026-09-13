"""Deterministic SOP authority; model outputs are bounded, untrusted proposals."""
import copy
import datetime
import re
from typing import Any, Dict, List, Optional

from .types import Phase, SOPState, PIIFields, CrossPhaseMemory, TraceEvent, ClaimRecord, PolicyHolder, StateSnapshot, Representative
from .grounded_data import grounded_data
from .extractor import UtteranceExtractor
from .context_builder import ContextBuilder


class SOPStateMachine:
    TERMINAL_PHASES = {Phase.CONCLUDED, Phase.ESCALATED}
    PII_NAMES = ("name", "dob", "phone", "email", "id_last4")
    HINT_NAMES = ("case_id_hint", "case_type_hint", "status_hint", "date_hint", "topic_hint")
    TOPICS = {"status", "denial_reason", "documents", "submission_method", "submission_timing", "processing_time", "document_alternatives", "file_format", "receipt_confirmation", "appeal_deadline", "payment"}

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = SOPState(session_id=session_id)
        self.extractor = UtteranceExtractor()

    @staticmethod
    def _now() -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _add_trace(self, gate: str, passed: bool, details: str, phase_before: Phase):
        self.state.trace_log.append(TraceEvent(
            timestamp=self._now(), phase_before=phase_before, phase_after=self.state.phase,
            gate_evaluated=gate, gate_passed=passed, details=details,
            data_shield_active=self.get_verified_policyholder() is None,
        ))

    def record_snapshot(self, user_msg: str, reply: str):
        self.state.history_snapshots.append(StateSnapshot(
            turn_index=len(self.state.history_snapshots), user_message=user_msg, agent_reply=reply,
            phase=self.state.phase,
            state_dump=copy.deepcopy(self.state.model_dump(exclude={"history_snapshots"})),
            timestamp=self._now(),
        ))

    def restore_to_turn(self, turn_index: int) -> bool:
        if not 0 <= turn_index < len(self.state.history_snapshots):
            return False
        retained = copy.deepcopy(self.state.history_snapshots[:turn_index + 1])
        self.state = SOPState.model_validate(copy.deepcopy(retained[-1].state_dump))
        self.state.history_snapshots = retained
        return True

    def get_verified_policyholder(self) -> Optional[PolicyHolder]:
        # Check independently of phase: escalation is not identity verification.
        if not self.state.identity_verified or not self.state.verified_party_id or self.state.pii_conflicts:
            return None
        verified, ph, fields = grounded_data.verify_identity(self.state.accumulated_pii)
        if not verified or not ph or ph.party_id != self.state.verified_party_id:
            return None
        if len(set(self.state.verified_fields) & set(fields) & set(self.PII_NAMES)) < 3:
            return None
        if self.state.is_proxy_caller:
            rep = self.get_proxy_representative()
            if (not rep or not self.state.proxy_identity_verified
                    or self.state.proxy_consent_status != "approved"
                    or self.state.proxy_authorized_party_id != ph.party_id
                    or rep.buyer_party_id != ph.party_id):
                return None
        return ph

    def get_proxy_representative(self) -> Optional[Representative]:
        if not self.state.is_proxy_caller or not self.state.proxy_rep_name:
            return None
        return grounded_data.find_representative(self.state.proxy_rep_name)

    def get_active_claim(self) -> Optional[ClaimRecord]:
        ph = self.get_verified_policyholder()
        if not ph or not self.state.active_case_id:
            return None
        claim = grounded_data.get_claim_by_id(self.state.active_case_id)
        if not claim or claim.party_id != ph.party_id:
            return None
        # Recheck hints as well as ownership so contradictory memory cannot leak
        # the old selected case through a different API or a restored snapshot.
        match, _ = grounded_data.find_claim(ph.party_id, self.state.cross_phase_memory)
        return claim if match and match.case_id == claim.case_id else None

    def record_grounded_topics(self, topics: List[str]):
        if self.get_active_claim() is None:
            return
        for topic in topics:
            if topic in self.TOPICS and topic not in self.state.discussion_topics:
                self.state.discussion_topics.append(topic)

    def generate_draft_email(self, policyholder: PolicyHolder, claim: Optional[ClaimRecord]) -> str:
        # Ignore caller-supplied objects that have not passed this session's gate.
        trusted_ph, trusted_claim = self.get_verified_policyholder(), self.get_active_claim()
        if trusted_ph is None or policyholder is None or trusted_ph.party_id != policyholder.party_id:
            return ""
        claim = trusted_claim if claim and trusted_claim and claim.case_id == trusted_claim.case_id else None
        if claim is None:
            discussed, outcome, steps = "Insurance support inquiry.", "No claim outcome was reviewed.", "No claim-specific next steps were recorded."
        else:
            labels = {
                "status": "claim status", "denial_reason": "the recorded denial reason",
                "documents": "required documents", "submission_method": "document submission methods",
                "submission_timing": "document submission timing", "processing_time": "review timing",
                "document_alternatives": "alternatives for unavailable documents", "file_format": "document format and readability",
                "receipt_confirmation": "confirming document receipt", "appeal_deadline": "the recorded appeal deadline",
                "payment": "recorded claim payment amounts",
            }
            topics = ", ".join(labels[t] for t in self.state.discussion_topics if t in labels) or "the claim inquiry"
            discussed = f"Claim {claim.case_id} ({claim.case_type}, created {claim.created_at}): {topics}."
            outcome = f"Recorded status: {claim.status.upper()}. {claim.summary}."
            if claim.denial_reason:
                outcome += f" Recorded reason: {claim.denial_reason}."
            if "payment" in self.state.discussion_topics:
                amounts = [("Net pay", claim.net_pay), ("Allowed maximum", claim.allowed_max_amount), ("Expected reimbursement", claim.expected_reimbursement_amount)]
                outcome += " " + " ".join(f"{label}: ${value}." for label, value in amounts if value is not None)
            followups = []
            if claim.documents_needed:
                followups.append("Required documents: " + ", ".join(claim.documents_needed) + ".")
                guidance = grounded_data.get_document_guidance_for_claim(claim)
                if "submission_method" in self.state.discussion_topics:
                    followups.append(guidance.get("default_guidance", ""))
                if "document_alternatives" in self.state.discussion_topics:
                    followups.extend(guidance.get("document_alternative_guidance", {}).values())
                if "file_format" in self.state.discussion_topics:
                    followups.extend(guidance.get("document_guidance", {}).values())
                if "processing_time" in self.state.discussion_topics:
                    timing = guidance.get("claim_followup_settings", {}).get("average_processing_time_after_submission", {}).get("en")
                    if timing:
                        followups.append(f"Fixture guidance for average processing after submission: {timing}; this is not a guaranteed completion date.")
                if "submission_timing" in self.state.discussion_topics:
                    for item in guidance.get("followup_qa", []):
                        if item.get("topic") == "submission_timing":
                            followups.append(item["en"].format(case_id=claim.case_id, documents=", ".join(claim.documents_needed)))
                if "receipt_confirmation" in self.state.discussion_topics:
                    for item in guidance.get("followup_qa", []):
                        if item.get("topic") == "receipt_confirmation":
                            followups.append(item["en"].format(case_id=claim.case_id, documents=", ".join(claim.documents_needed)))
            if claim.appeal_deadline:
                followups.append(f"Recorded appeal deadline: {claim.appeal_deadline}. Support must confirm available options if that date has passed.")
            steps = " ".join(s for s in followups if s) or "No additional document requirement is recorded in this claim fixture."
        return (f"Dear {trusted_ph.name},\n\nHere is a summary of our discussion today:\n\n"
                f"1. What Was Discussed:\n   {discussed}\n\n"
                f"2. Claim Status & Outcome:\n   {outcome}\n\n"
                f"3. Major Follow-up Items & Next Steps:\n   {steps}\n\n"
                "Insurance Claims Support Team\n\nDemo summary: delivery is simulated; no real email is sent.")

    def _result(self, phase_before: Phase, *, is_oos=False, emotions=None, note=None) -> Dict[str, Any]:
        emotions = emotions or {}
        ph, claim = self.get_verified_policyholder(), self.get_active_claim()
        context = ContextBuilder.build_system_prompt(
            self.state, ph, claim, is_out_of_scope=is_oos,
            is_frustrated=bool(emotions.get("is_frustrated")), is_refusal=bool(emotions.get("is_refusal")),
        )
        # Every renderer receives the shield result from the authority.
        context["data_shield_active"] = ph is None
        candidates = grounded_data.find_claim(ph.party_id, self.state.cross_phase_memory)[1] if ph and self.state.phase == Phase.RESOLVE_INTENT else []
        return {
            "state": self.state, "is_out_of_scope": is_oos,
            "is_frustrated": bool(emotions.get("is_frustrated")), "emotion": emotions.get("emotion"),
            "is_refusal": bool(emotions.get("is_refusal")), "demands_human": self.state.phase == Phase.ESCALATED,
            "context": context, "policyholder": ph, "active_claim": claim,
            "is_proxy_caller": self.state.is_proxy_caller, "proxy_rep_name": self.state.proxy_rep_name,
            "proxy_relationship": self.state.proxy_relationship, "proxy_consent_status": self.state.proxy_consent_status,
            "collected_fields": [f for f in self.PII_NAMES if getattr(self.state.accumulated_pii, f)],
            "candidate_claims": candidates, "resolution_status": self.state.resolution_status,
            "transition_note": note or f"Phase: {phase_before.value} -> {self.state.phase.value}",
        }

    @staticmethod
    def _literal_pii(field: str, value: str, text: str) -> bool:
        if not isinstance(value, str) or len(value) > 150:
            return False
        if field == "phone":
            return re.sub(r"\D", "", value).lstrip("+") in re.sub(r"[^\d]", "", text)
        if field == "dob":
            if value in text:
                return True
            try:
                day = datetime.date.fromisoformat(value)
            except ValueError:
                return False
            formats = [f"{day.month}/{day.day}/{day.year}", f"{day.month:02}/{day.day:02}/{day.year}", day.strftime("%B %d, %Y"), f"{day.strftime('%B')} {day.day}, {day.year}", f"{day.strftime('%B')} {day.day} {day.year}"]
            return any(v.casefold() in text.casefold() for v in formats)
        return bool(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.IGNORECASE))

    def _accumulate(self, text: str, semantic: Dict[str, Any]):
        hints = self.extractor.extract_cross_phase_hints(text)
        proposal = semantic.get("hints", {})
        if isinstance(proposal, dict):
            for field in self.HINT_NAMES:
                value = proposal.get(field)
                if not getattr(hints, field, None) and isinstance(value, str) and len(value) <= 100:
                    # IDs and dates must be grounded in user text; domain labels
                    # are bounded interpretations of the utterance, not truth.
                    allowed = {
                        "case_type_hint": {"healthcare", "dental", "auto"},
                        "status_hint": {"denied", "open", "closed", "pending", "approved"},
                        "topic_hint": self.TOPICS | {"status_inquiry", "denial_question", "document_submission", "next_steps", "general_claim_question"},
                    }
                    if (field in allowed and value in allowed[field]) or (field not in allowed and value.casefold() in text.casefold()):
                        setattr(hints, field, value)
        memory = self.state.cross_phase_memory
        correction = bool(re.search(r"\b(?:actually|correction|i meant|correct that|rather than|not .+ but)\b", text, re.IGNORECASE))
        has_hint = False
        case_hint_allowed = self.state.phase in {Phase.VERIFY_ID, Phase.RESOLVE_INTENT} or bool(
            re.search(r"\b(?:different claim|another claim|switch claim|instead|actually|correction|i meant)\b", text, re.I)
            or re.search(r"\bCL-\d+\b", text, re.I))
        for field in self.HINT_NAMES:
            value = getattr(hints, field, None)
            if value and (field == "topic_hint" or case_hint_allowed):
                has_hint = True
                prior = getattr(memory, field, None)
                if correction:
                    memory.hint_history[field] = []
                values = memory.hint_history.setdefault(field, [])
                if prior and prior not in values and not correction:
                    values.append(prior)
                if value not in values:
                    values.append(value)
                setattr(memory, field, value)
        if has_hint:
            # Store business memory without retaining whole PII-bearing turns.
            snippet = " / ".join(str(getattr(hints, field)) for field in self.HINT_NAMES if getattr(hints, field, None))
            memory.raw_utterance_snippet = snippet
            if snippet not in memory.notes:
                memory.notes.append(snippet)
        # Preserve EVERY explicit month/year/case ID, including multiple values
        # in one message. Ambiguity must cause clarification, never a first-hit
        # match. Strip labelled birth dates before looking for claim dates.
        business = re.sub(r"\b(?:dob|date of birth|birthday|born)\s*(?:is|on|was|:)?\s*(?:[a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})", "", text, flags=re.IGNORECASE)
        if case_hint_allowed and (not self.extractor.extract_pii(text).dob or business.strip(" .!") != text.strip(" .!")):
            import calendar
            for month in range(1, 13):
                if re.search(r"\b(?:" + calendar.month_name[month] + "|" + calendar.month_abbr[month] + r")\b", business, re.IGNORECASE):
                    value = calendar.month_name[month]
                    if value not in memory.hint_history.setdefault("date_hint", []):
                        memory.hint_history["date_hint"].append(value)
            for value in re.findall(r"\b20\d{2}(?:-\d{2}(?:-\d{2})?)?\b", business):
                if value not in memory.hint_history.setdefault("date_hint", []):
                    memory.hint_history["date_hint"].append(value)
        for value in re.findall(r"\bCL-\d+\b", business, re.IGNORECASE):
            if value.upper() not in memory.hint_history.setdefault("case_id_hint", []):
                memory.hint_history["case_id_hint"].append(value.upper())
        if self.state.phase != Phase.VERIFY_ID:
            return
        pii = self.extractor.extract_pii(text)
        # Detect repeated contradictory labelled identity values within one
        # utterance, before an extractor's first match can conceal the others.
        label_patterns = {
            "name": r"\b(?:my (?:full )?name is|name\s*:|i am|i'm|this is)\s+",
            "dob": r"\b(?:dob|date of birth|birthday|born)\s*",
            "id_last4": r"\b(?:ssn|social security|national id|last (?:four|4))\b",
            "phone": r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)",
            "email": r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b",
            "policy_number": r"\bPOL-\d+\b",
        }
        for field, pattern in label_patterns.items():
            values = set()
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                value = getattr(self.extractor.extract_pii(text[match.start():end]), field, None)
                if value:
                    values.add(grounded_data._clean_phone(value) if field == "phone" else grounded_data._clean_str(value))
            if len(values) > 1 and not correction and field not in self.state.pii_conflicts:
                self.state.pii_conflicts.append(field)
        proposal_pii = semantic.get("pii", {})
        if isinstance(proposal_pii, dict):
            for field in (*self.PII_NAMES, "policy_number"):
                value = proposal_pii.get(field)
                if not getattr(pii, field) and self._literal_pii(field, value, text):
                    setattr(pii, field, value)
        for field in (*self.PII_NAMES, "policy_number", "id_type"):
            value = getattr(pii, field, None)
            if not value:
                continue
            previous = getattr(self.state.accumulated_pii, field, None)
            if correction and field in self.state.pii_conflicts:
                self.state.pii_conflicts.remove(field)
            if previous and str(previous).casefold() != str(value).casefold() and field != "id_type":
                # An explicit correction is acceptable; silently discarding a
                # contradictory identity is not.
                if not correction and field not in self.state.pii_conflicts:
                    self.state.pii_conflicts.append(field)
                if correction and field in self.state.pii_conflicts:
                    self.state.pii_conflicts.remove(field)
            setattr(self.state.accumulated_pii, field, value)

    def _resolve(self):
        ph = self.get_verified_policyholder()
        if not ph:
            return
        memory = self.state.cross_phase_memory
        has_intent = any(getattr(memory, field, None) for field in self.HINT_NAMES)
        if not has_intent:
            self.state.resolution_status = "needs_intent"
            self._add_trace("RESOLVE_INTENT_GATE", False, "Verified identity; ask what claim or insurance question the caller needs help with.", Phase.RESOLVE_INTENT)
            return
        claim, candidates = grounded_data.find_claim(ph.party_id, memory)
        if claim:
            self.state.active_case_id = claim.case_id
            self.state.resolution_status = "resolved"
            self.state.phase = Phase.PROCESS_CASE
            self._add_trace("RESOLVE_INTENT_GATE", True, f"Resolved owned claim {claim.case_id} using the intersection of remembered hints.", Phase.RESOLVE_INTENT)
            self.record_grounded_topics(["status"] + (["denial_reason", "documents", "appeal_deadline"] if claim.denial_reason else []))
        else:
            self.state.active_case_id = None
            self.state.resolution_status = "ambiguous" if candidates else "no_match"
            self._add_trace("RESOLVE_INTENT_GATE", False, "Remembered hints are ambiguous or do not match an owned claim; clarification required.", Phase.RESOLVE_INTENT)

    @staticmethod
    def _consent(text: str) -> Optional[str]:
        """Explicit, negation-safe consent; model proposals cannot send email."""
        clean = " ".join(text.casefold().replace("’", "'").split())
        if "?" in clean or re.search(r"\b(?:if|unless|until|maybe|perhaps|might|would|could|can you explain|what|why|when)\b", clean):
            return None
        clean = re.sub(r"[.!]+$", "", clean).strip()
        if re.search(r"\b(?:do not|don't|dont|never)\s+(?:send|email)\b", clean) or re.fullmatch(r"(?:no(?: thanks| thank you)?[, ]*)?(?:please )?skip(?: (?:it|the email|the summary|email|the email summary))?", clean) or clean in {"no", "no thanks", "no thank you"}:
            return "declined"
        if re.search(r"\b(?:no|not|don't|dont|never|stop|skip|but|cancel)\b", clean):
            return None
        if clean in {"yes", "yes please", "yes, please", "please do", "send it", "send it please", "please send it", "yes send it", "yes, send it"}:
            return "accepted"
        if re.fullmatch(r"(?:yes[, ]+)?(?:please )?(?:send|email)(?: me)? (?:the |an? |my )?(?:email )?summary(?: (?:to (?:my |the )?email(?: address)?(?: on file)?|to me|please))?", clean):
            return "accepted"
        return None

    @staticmethod
    def _wrap_up(text: str) -> bool:
        clean = " ".join(text.casefold().replace("’", "'").split())
        if "?" in clean or re.search(r"\b(?:but|not done|don't|do not|another question|one more)\b", clean):
            return False
        return bool(re.search(r"\b(?:no more questions?|that's all|that is all|i'm good|im good|nothing else|no (?:that's|that is) it|nope that's it|wrap up|thanks that helps|no thanks)\b", clean))

    def verify_card_data(self, form_data: Dict[str, str]) -> Dict[str, Any]:
        """Direct deterministic card form verification bypassing LLM extraction."""
        phase_before = self.state.phase
        if phase_before in self.TERMINAL_PHASES:
            return self._result(phase_before, note="This session is terminal. Start a new session to continue.")

        for field in ("name", "dob", "phone", "email", "id_last4", "id_type"):
            val = form_data.get(field, "").strip()
            if val:
                setattr(self.state.accumulated_pii, field, val)

        dob_val = form_data.get("dob", "").strip()
        if dob_val:
            m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", dob_val)
            if m:
                month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                iso_dob = f"{year:04d}-{month:02d}-{day:02d}"
                self.state.accumulated_pii.dob = iso_dob

        verified, ph, matched = grounded_data.verify_identity(self.state.accumulated_pii)
        if verified and ph and len(matched) >= 3:
            self.state.identity_verified = True
            self.state.verified_party_id = ph.party_id
            self.state.verified_fields = list(matched)
            if self.state.phase == Phase.VERIFY_ID:
                self.state.phase = Phase.RESOLVE_INTENT
                self._add_trace(
                    "VERIFY_ID_GATE", True,
                    f"Security card form matched {len(matched)}/3 fields for policyholder {ph.name} ({ph.party_id}). Gate unlocked.",
                    Phase.VERIFY_ID
                )
                self._resolve()
            reply_msg = f"Thank you, {ph.name}. Your security verification card has been verified. How can Aegis support help with your claim today?"
        else:
            num_matched = len(matched) if matched else 0
            self._add_trace(
                "VERIFY_ID_GATE", False,
                f"Security card form submitted ({num_matched}/3 fields matched). Gate remains locked.",
                phase_before
            )
            needed = max(0, 3 - num_matched)
            reply_msg = f"Thank you for submitting the verification card. We currently have {num_matched} matching field(s). We need {needed} more matching field(s) to verify your identity."

        self.record_snapshot("[Submitted Security Verification Card]", reply_msg)
        res = self._result(phase_before, note=f"Card verification: {phase_before.value} -> {self.state.phase.value}")
        res["agent_reply"] = reply_msg
        return res

    def evaluate_turn(self, user_text: str, semantic: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        phase_before = self.state.phase
        if phase_before in self.TERMINAL_PHASES:
            return self._result(phase_before, note="This session is terminal. Start a new session to continue.")
        semantic = semantic if isinstance(semantic, dict) else {}
        # Preserve useful information even when the message is refused by scope.
        self._accumulate(user_text, semantic)
        emotions = self.extractor.detect_emotion_and_intent(user_text)
        for key in ("is_refusal", "demands_human"):
            emotions[key] = bool(emotions.get(key) or semantic.get(key) is True)
        if semantic.get("emotion") in {"frustration", "anxiety", "anger", "confusion", "refusal"}:
            emotions["emotion"] = semantic["emotion"]
            emotions["is_frustrated"] = True
        is_oos = self.extractor.is_out_of_scope(user_text) or semantic.get("is_out_of_scope") is True

        # A third-party disclosure remains blocked in EVERY phase. A roster
        # lookup cannot substitute for verified representative credentials.
        proxy = self.extractor.extract_proxy_context(user_text)
        if proxy.get("is_proxy"):
            self.state.is_proxy_caller = True
            self.state.proxy_rep_name = proxy.get("rep_name") or self.state.proxy_rep_name
            self.state.proxy_relationship = proxy.get("relationship") or self.state.proxy_relationship
            rep = grounded_data.find_representative(self.state.proxy_rep_name or "", proxy.get("buyer_name"))
            self.state.proxy_consent_status = "requires_human" if rep else "unauthorized"
            self.state.proxy_identity_verified = False
            self.state.proxy_authorized_party_id = None
            self.state.identity_verified = False
            self.state.verified_party_id = None
            self.state.verified_fields = []
            self.state.active_case_id = None
            if rep or phase_before != Phase.VERIFY_ID:
                self.state.phase = Phase.ESCALATED
            self._add_trace("PROXY_AUTHORIZATION_GATE", False, "Representative identity and live consent cannot be verified with these fixtures. Human verification is required." if rep else "No exact authorized representative match. Third-party claim access denied.", phase_before)

        if emotions.get("demands_human"):
            self.state.phase = Phase.ESCALATED
            self._add_trace("HUMAN_ESCALATION", True, "Caller requested a human. Demo handoff recorded; no live connection is made.", phase_before)
            return self._result(phase_before, emotions=emotions, is_oos=is_oos)
        if self.state.phase == Phase.ESCALATED:
            return self._result(phase_before, emotions=emotions, is_oos=is_oos)
        if is_oos:
            self.state.out_of_scope_count += 1
            if self.state.out_of_scope_count >= 2:
                self.state.phase = Phase.ESCALATED
            self._add_trace("OUT_OF_SCOPE_GUARD", False, "Out-of-scope request refused; repeated requests require a demo human handoff." if self.state.phase == Phase.ESCALATED else "Out-of-scope request refused. Useful in-scope information retained without advancing the SOP.", phase_before)
            return self._result(phase_before, emotions=emotions, is_oos=True)
        self.state.out_of_scope_count = 0

        if self.state.is_proxy_caller:
            if emotions.get("is_refusal"):
                self.state.refusal_count += 1
                if self.state.refusal_count >= 3:
                    self.state.phase = Phase.ESCALATED
            return self._result(phase_before, emotions=emotions, note="Representative access remains blocked pending human verification.")

        if self.state.phase == Phase.VERIFY_ID:
            verified, ph, fields = grounded_data.verify_identity(self.state.accumulated_pii)
            self.state.verified_fields = fields if not self.state.pii_conflicts else []
            if verified and ph and not self.state.pii_conflicts:
                self.state.identity_verified = True
                self.state.verified_party_id = ph.party_id
                self.state.phase = Phase.RESOLVE_INTENT
                self.state.refusal_count = 0
                self._add_trace("VERIFY_ID_GATE", True, f"Identity matched {len(fields)} allowed distinct PII fields: {', '.join(fields)}. Policy number is lookup only.", phase_before)
                self._resolve()
            else:
                if emotions.get("is_refusal"):
                    self.state.refusal_count += 1
                    if self.state.refusal_count >= 3:
                        self.state.phase = Phase.ESCALATED
                self._add_trace("VERIFY_ID_GATE", False, "Three distinct matching allowed PII fields are required with no contradictory values. Explain privacy and offer alternative fields." if self.state.phase != Phase.ESCALATED else "Three verification refusals reached; recorded a demo human handoff without disclosing claim data.", phase_before)
        elif self.state.phase == Phase.RESOLVE_INTENT:
            self._resolve()
        elif self.state.phase == Phase.PROCESS_CASE:
            if self.get_active_claim() is None:
                self.state.phase = Phase.ESCALATED
                self._add_trace("CASE_OWNERSHIP_GATE", False, "New claim hints conflict with the selected owned case. Human clarification required before further disclosure.", phase_before)
            elif self._wrap_up(user_text) or (semantic.get("wrap_up") is True and "?" not in user_text and not set(semantic.get("response_topics", [])) - {"unknown"}) or bool(re.fullmatch(r"(?:please )?(?:send|email)(?: me)? (?:an? |the )?(?:email )?summary[.!]?", user_text.strip(), re.I)):
                self.state.phase = Phase.POST_PROCESS
                self.state.post_process.email_offered = True
                self.state.post_process.user_decision = "pending"
                self.state.post_process.draft_summary = self.generate_draft_email(self.get_verified_policyholder(), self.get_active_claim())
                self._add_trace("PROCESS_CASE_GATE", True, "Caller finished claim questions. Offer optional email summary; explicit send or skip decision is required on a subsequent turn.", phase_before)
            else:
                topic = self.state.cross_phase_memory.topic_hint
                topic_aliases = {"denial_question": "denial_reason", "document_submission": "submission_method", "status_inquiry": "status"}
                self.record_grounded_topics([topic_aliases.get(topic, topic)] if topic else [])
                self._add_trace("PROCESS_CASE_GATE", True, "Answer only using the verified caller's selected claim and fixture guidance.", phase_before)
        elif self.state.phase == Phase.POST_PROCESS:
            consent = self._consent(user_text)
            ph = self.get_verified_policyholder()
            if not ph or self.get_active_claim() is None:
                self.state.phase = Phase.ESCALATED
                self._add_trace("POST_PROCESS_CONSENT", False, "Identity, ownership or case context is no longer valid. No summary queued.", phase_before)
            elif consent == "accepted":
                post = self.state.post_process
                if not post.outbox_id:
                    post.outbox_id = f"demo-{self.session_id}-summary"
                    self.state.mock_outbox.append({"id": post.outbox_id, "to": ph.email, "subject": "Your insurance claims conversation summary", "body": post.draft_summary, "delivery_status": "simulated", "created_at": self._now()})
                post.user_decision, post.sent_to, post.delivery_status = "accepted", ph.email, "simulated"
                self.state.phase = Phase.CONCLUDED
                self._add_trace("POST_PROCESS_CONSENT", True, "Explicit email consent recorded. Exactly one mock outbox entry created; no real email sent.", phase_before)
            elif consent == "declined":
                self.state.post_process.user_decision = "declined"
                self.state.post_process.delivery_status = "skipped"
                self.state.phase = Phase.CONCLUDED
                self._add_trace("POST_PROCESS_CONSENT", True, "Caller declined the optional email. No outbox entry created.", phase_before)
            else:
                self._add_trace("POST_PROCESS_CONSENT", False, "Email consent is ambiguous. Ask explicitly whether to send or skip; do not infer consent.", phase_before)
        return self._result(phase_before, emotions=emotions)
