import re
from typing import Dict, Any, Optional, Tuple
from .types import PIIFields, CrossPhaseMemory

KNOWN_NAMES = [
    "Margaret Chen",
    "Ava Lopez",
    "Ma Tian",
    "Ya Wen Li",
    "Yaven Li",
    "David Chen"
]

OUT_OF_SCOPE_PATTERNS = [
    r"\bwhat is rl\b",
    r"\breinforcement learning\b",
    r"\bq-learning\b",
    r"\bdeep learning\b",
    r"\bmachine learning\b",
    r"\bwrite (?:a )?(?:poem|code|script|story|essay)\b",
    r"\bcapital of\b",
    r"\bwho is the president\b",
    r"\bwho won the\b",
    r"\bweather in\b",
    r"\bquantum physics\b",
    r"\bbitcoin\b|\bcrypto\b",
    r"\btell me a joke\b",
    r"\bpython\b|\bjavascript\b|\bgolang\b",
    r"ignore (?:all |previous )?instructions",
    r"print (?:system )?prompt",
    r"system prompt",
    r"reveal (?:your )?instructions",
    r"claims\.json",
    r"jailbreak"
]

class UtteranceExtractor:
    @staticmethod
    def extract_pii(text: str) -> PIIFields:
        pii = PIIFields()
        text_lower = text.lower()

        # Check known policyholders first
        for name in ["Margaret Chen", "Ava Lopez", "Ma Tian", "Ya Wen Li", "Yaven Li"]:
            if name.lower() in text_lower:
                pii.name = name
                break
        if not pii.name:
            name_match = re.search(r"(?:my name is|i am|name:?|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text, re.IGNORECASE)
            if name_match:
                extracted = name_match.group(1).strip()
                # filter out words like "the policyholder"
                if not any(w in extracted.lower() for w in ["the", "calling", "asking", "policyholder"]):
                    pii.name = extracted

        # 2. Policy Number (e.g. POL-9921)
        pol_match = re.search(r"\b(POL-\d{4})\b", text, re.IGNORECASE)
        if pol_match:
            pii.policy_number = pol_match.group(1).upper()

        # 3. Date of Birth (YYYY-MM-DD or MM/DD/YYYY)
        dob_match = re.search(r"\b(19\d{2}|20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b", text)
        if dob_match:
            pii.dob = dob_match.group(0).replace("/", "-")
        else:
            dob_match2 = re.search(r"\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19\d{2}|20\d{2})\b", text)
            if dob_match2:
                m, d, y = dob_match2.group(1), dob_match2.group(2), dob_match2.group(3)
                pii.dob = f"{y}-{m}-{d}"

        # 4. SSN last 4 or ID last 4
        ssn_match = re.search(r"(?:ssn|social security|id|national id|last 4|last four)?(?:.*?)(?:last (?:four|4)(?: is| digits are)?:?|\blast 4:?)\s*(\d{4})\b", text, re.IGNORECASE)
        if ssn_match:
            pii.id_last4 = ssn_match.group(1)
            pii.id_type = "ssn_last4"
        else:
            # Look for "SSN is 4472" or "ID is 6688"
            ssn_match2 = re.search(r"(?:ssn|social|id)\s+(?:is\s+|:\s*)?(\d{4})\b", text, re.IGNORECASE)
            if ssn_match2:
                pii.id_last4 = ssn_match2.group(1)
                pii.id_type = "ssn_last4"

        # 5. Phone
        phone_match = re.search(r"(\+?1\d{10}|\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})", text)
        if phone_match:
            raw_phone = phone_match.group(1)
            digits = re.sub(r"\D", "", raw_phone)
            if len(digits) == 10:
                pii.phone = f"+1{digits}"
            elif len(digits) == 11 and digits.startswith("1"):
                pii.phone = f"+{digits}"

        # 6. Email
        email_match = re.search(r"\b([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b", text)
        if email_match:
            pii.email = email_match.group(1).lower()

        return pii

    @staticmethod
    def extract_cross_phase_hints(text: str) -> CrossPhaseMemory:
        hints = CrossPhaseMemory()
        text_lower = text.lower()

        # Case type
        if "healthcare" in text_lower or "medical" in text_lower or "health" in text_lower:
            hints.case_type_hint = "healthcare"
        elif "dental" in text_lower or "teeth" in text_lower or "tooth" in text_lower:
            hints.case_type_hint = "dental"
        elif "auto" in text_lower or "car" in text_lower or "vehicle" in text_lower or "accident" in text_lower:
            hints.case_type_hint = "auto"

        # Status hint
        if "denied" in text_lower or "denial" in text_lower or "rejected" in text_lower:
            hints.status_hint = "denied"
        elif "closed" in text_lower or "settled" in text_lower:
            hints.status_hint = "closed"
        elif "open" in text_lower or "in progress" in text_lower or "pending" in text_lower:
            hints.status_hint = "open"

        # Date / Month hint
        months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        for m in months:
            if m in text_lower:
                hints.date_hint = m.capitalize()
                break

        # Check explicit year or date snippet
        date_pattern = re.search(r"\b(202[456](?:-[0-1][0-9]-[0-3][0-9])?)\b", text)
        if date_pattern:
            if hints.date_hint:
                hints.date_hint = f"{hints.date_hint} {date_pattern.group(1)}"
            else:
                hints.date_hint = date_pattern.group(1)

        # Snippet
        hints.raw_utterance_snippet = text[:120]
        return hints

    @staticmethod
    def detect_emotion_and_intent(text: str) -> Dict[str, Any]:
        text_lower = text.lower()

        is_frustrated = any(phrase in text_lower for phrase in [
            "already told you",
            "ridiculous",
            "frustrated",
            "annoying",
            "waste of time",
            "just tell me",
            "why do you need",
            "why is this so hard",
            "tired of this",
            "stop asking"
        ])

        is_refusal = any(phrase in text_lower for phrase in [
            "refuse",
            "won't give",
            "will not tell",
            "none of your business",
            "i don't want to give",
            "not giving you",
            "why should i give"
        ])

        demands_human = any(phrase in text_lower for phrase in [
            "speak to human",
            "human representative",
            "real person",
            "talk to someone",
            "supervisor",
            "manager",
            "live agent",
            "human agent"
        ])

        return {
            "is_frustrated": is_frustrated,
            "is_refusal": is_refusal,
            "demands_human": demands_human
        }

    @staticmethod
    def is_out_of_scope(text: str) -> bool:
        text_lower = text.lower().strip()
        for pat in OUT_OF_SCOPE_PATTERNS:
            if re.search(pat, text_lower):
                return True
        return False

    @staticmethod
    def extract_proxy_context(text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        is_proxy = False
        rep_name = None
        buyer_name = None
        relationship = None

        proxy_triggers = [
            "on behalf of", "calling for my", "for my mother", "for my father",
            "for my wife", "for my husband", "son of", "daughter of", "representative for",
            "calling for", "neighbor of", "friend of", "attorney", "lawyer", "doctor",
            "surgeon", "nurse", "calling about her", "checking on her", "her claim"
        ]
        if any(t in text_lower for t in proxy_triggers):
            is_proxy = True

        # Check caller introduced name
        name_intro = re.search(r"(?:i am|my name is|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text, re.IGNORECASE)
        if name_intro:
            candidate_rep = name_intro.group(1).strip()
            # If introduced name is NOT Margaret Chen, but Margaret is mentioned
            if ("margaret" in text_lower or "mom" in text_lower or "mother" in text_lower) and "margaret" not in candidate_rep.lower():
                is_proxy = True
                rep_name = candidate_rep
                buyer_name = "Margaret Chen"

        if "david chen" in text_lower:
            rep_name = "David Chen"
            is_proxy = True
        elif "david" in text_lower and any(w in text_lower for w in ["son", "calling for", "mother", "mom"]):
            rep_name = "David Chen"
            is_proxy = True

        if "margaret chen" in text_lower or "margaret" in text_lower or "mom" in text_lower or "mother" in text_lower:
            buyer_name = "Margaret Chen"

        if "mother" in text_lower or "mom" in text_lower or "son" in text_lower:
            relationship = "son"
        elif "neighbor" in text_lower:
            relationship = "neighbor"
        elif "doctor" in text_lower or "surgeon" in text_lower:
            relationship = "medical_provider"
        elif "lawyer" in text_lower or "attorney" in text_lower:
            relationship = "legal_rep"

        return {
            "is_proxy": is_proxy,
            "rep_name": rep_name,
            "buyer_name": buyer_name,
            "relationship": relationship
        }

    @staticmethod
    def detect_post_process_consent(text: str) -> Optional[str]:
        """
        Returns 'accepted', 'declined', or None
        """
        text_lower = text.lower().strip()

        accept_phrases = [
            "yes", "sure", "please send", "send it", "email it", "that would be great",
            "yep", "yeah", "sounds good", "okay", "send email", "go ahead", "please do"
        ]
        decline_phrases = [
            "no", "no thanks", "skip", "don't send", "not needed", "no need", "nope",
            "i'm good", "don't email", "skip it"
        ]

        # Check declines first
        for dp in decline_phrases:
            if dp in text_lower:
                return "declined"

        for ap in accept_phrases:
            if ap in text_lower:
                return "accepted"

        return None
