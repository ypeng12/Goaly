"""Reactive caller simulator for AgentPolicyEnv.

CallerProfile represents a synthetic caller with a specific identity and
conversational style.  It produces the next caller utterance in response
to the agent's spoken response, without reading private SOP state or action labels.

Supported styles:
- cooperative: provides PII promptly, follows SOP flow smoothly.
- frustrated: adds emotional preamble; cooperates with empathy/explanation.
- confused: uncertain about required info, provides non-counting details first.
- privacy_sensitive: refuses SSN/ID initially, prefers phone/email.
- escalating: cooperative early, then explicitly demands human escalation.
"""
import copy
import re
import random
from typing import Optional, List, Dict, Any

from .types import AgentAction, Phase, PolicyHolder, ClaimRecord


class CallerProfile:
    """A synthetic caller that reacts to the bounded dialogue text protocol."""

    def __init__(
        self,
        policyholder: PolicyHolder,
        claim_hint: str = "denied healthcare claim",
        style: str = "cooperative",
        escalate_after: int = 6,
        email_accepted: bool = True,
    ):
        self.ph = policyholder
        self.claim_hint = claim_hint
        self.style = style
        self.escalate_after = escalate_after
        self.email_accepted = email_accepted
        self.reset()

    def reset(self, seed=None):
        self.rng = random.Random(seed)

        # Conversation state
        self._turn = 0
        self._pii_fields_given: set[str] = set()
        self._intent_expressed = False
        self._claim_details_given = False
        self._email_decided = False
        self._email_accepted = self.email_accepted
        self._reassured = False
        self._gate_explained = False

    def get_state(self) -> Dict[str, Any]:
        """Snapshot internal caller state for exact counterfactual branching."""
        return {
            "turn": self._turn,
            "pii_fields_given": set(self._pii_fields_given),
            "intent_expressed": self._intent_expressed,
            "claim_details_given": self._claim_details_given,
            "email_decided": self._email_decided,
            "email_accepted": self._email_accepted,
            "style": self.style,
            "reassured": self._reassured,
            "gate_explained": self._gate_explained,
            "rng_state": self.rng.getstate(),
        }

    def set_state(self, state_dict: Dict[str, Any]) -> None:
        """Restore internal caller state from a snapshot."""
        self._turn = state_dict["turn"]
        self._pii_fields_given = set(state_dict["pii_fields_given"])
        self._intent_expressed = state_dict["intent_expressed"]
        self._claim_details_given = state_dict["claim_details_given"]
        self._email_decided = state_dict["email_decided"]
        self._email_accepted = state_dict["email_accepted"]
        self.style = state_dict.get("style", self.style)
        self._reassured = state_dict['reassured']
        self._gate_explained = state_dict['gate_explained']
        self.rng.setstate(state_dict['rng_state'])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_opening_utterance(self) -> str:
        """First message the caller sends before any agent action."""
        self._pii_fields_given.add('name')
        if self.style == "frustrated":
            return (
                f"This is absolutely ridiculous — I've been waiting for weeks! "
                f"My name is {self.ph.name} and I need help with my {self.claim_hint}."
            )
        elif self.style == "confused":
            return (
                f"Hello? I'm not really sure how this works. "
                f"My policy number is {self.ph.policy_number} and my name is {self.ph.name}. "
                f"I got a letter about my {self.claim_hint}."
            )
        elif self.style == "privacy_sensitive":
            return (
                f"Hello, my name is {self.ph.name}. I'm calling to inquire about my {self.claim_hint}. "
                f"Please note that I prefer not to share sensitive government IDs over the phone."
            )
        elif self.style == "escalating":
            return f"Hi, my name is {self.ph.name}. I need help urgently with my {self.claim_hint}."
        else:
            return (
                f"Hi, my name is {self.ph.name}. "
                f"I'm calling about my {self.claim_hint}."
            )

    def respond_to(self, agent_action: AgentAction, agent_reply: str, state=None) -> Optional[str]:
        """React to spoken text only; action label and SOP state are not oracles.

        This deterministic simulator understands the bounded renderer protocol.
        Supplying a different action label with identical speech has no effect.
        """
        self._turn += 1
        if self.style == "escalating" and self._turn >= self.escalate_after:
            return "I want to speak with a human agent now."
        low = agent_reply.lower()
        if "i hear how" in low and "your concern" in low:
            self._reassured = True
            return "Thank you for understanding. Let's continue."
        if "three matching identity" in low and "phone or email instead" in low:
            self._gate_explained = True
        if "please share your" in low:
            if self.style == "frustrated" and not self._reassured:
                return "I'm frustrated. Please acknowledge my concern before asking for more details."
            if self.style == "confused" and not self._gate_explained:
                return "I'm confused. Why do you need this? Please explain one thing at a time."
            if self.style == "privacy_sensitive" and not self._gate_explained:
                return "I prefer not to share sensitive government IDs. Can I use phone or email?"
            labels = {"full name": "name", "date of birth": "dob", "phone number": "phone",
                      "email address": "email", "government id last four": "id_last4"}
            requested = next((field for label, field in labels.items()
                              if f"please share your {label}" in low), None)
            return self._provide_next_pii(set(self._pii_fields_given), requested=requested)
        if "what would you like to understand" in low:
            self._intent_expressed = True
            return f"I need the status and next steps for my {self.claim_hint}."
        if "which claim reference or year" in low:
            self._claim_details_given = True
            return f"My letter only says {self.claim_hint}. I have no claim reference or year to add."
        if re.search(r"claim cl-\d+ is ", low):
            return "Thank you for explaining the claim details. That is all the questions I have."
        if "would you like an email summary" in low:
            if self._email_decided:
                return "I already answered that."
            self._email_decided = True
            return ("Yes, please send the summary to my email." if self._email_accepted
                    else "No thanks, please skip the summary.")
        if "connect you with a human" in low:
            return "Thank you for arranging the handoff."
        return "I did not understand that. Could you clarify?"

    def _provide_next_pii(self, verified_fields: set[str], requested=None) -> str:
        """Return a caller utterance with the next missing PII field."""
        if self.style == "privacy_sensitive":
            # Avoid SSN/id_last4; prioritize phone and email
            pii_order = ["name", "dob", "phone", "email"]
        elif self.style == "confused" and self._turn <= 1 and "dob" not in self._pii_fields_given:
            pii_order = ["dob", "phone", "name", "email", "id_last4"]
        else:
            pii_order = ["name", "dob", "phone", "email", "id_last4"]

        remaining = [f for f in pii_order if f not in verified_fields and f not in self._pii_fields_given]

        if not remaining:
            return f"I've already provided all my details. My policy is {self.ph.policy_number}."

        field = requested if requested in remaining else remaining[0]
        self._pii_fields_given.add(field)

        if field == "name":
            return f"My name is {self.ph.name}."
        elif field == "dob":
            from datetime import date
            d = date.fromisoformat(self.ph.dob)
            return f"My date of birth is {d.month:02d}/{d.day:02d}/{d.year}."
        elif field == "phone":
            digits = re.sub(r"\D", "", self.ph.phone)
            if digits.startswith("1") and len(digits) == 11:
                digits = digits[1:]
            formatted = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            return f"My phone number is {formatted}."
        elif field == "email":
            return f"My email is {self.ph.email}."
        elif field == "id_last4":
            id_label = "SSN last four" if self.ph.id_type == "ssn_last4" else "government ID last four"
            return f"My {id_label} is {self.ph.id_last4}."

        return "I'm not sure what else to provide."


def make_margaret_chen_profile(style: str = "frustrated") -> "CallerProfile":
    """Create a CallerProfile for Margaret Chen (fixture policyholder 0)."""
    from .grounded_data import grounded_data
    ph = next(p for p in grounded_data.policyholders if p.name == "Margaret Chen")
    return CallerProfile(ph, claim_hint="denied healthcare claim", style=style)


def make_all_profiles() -> List["CallerProfile"]:
    """Return the canonical 4 benchmark profiles specified in RL specification:
    - Margaret Chen: frustrated
    - Ava Lopez: confused
    - Ma Tian: cooperative
    - Ya Wen Li: escalating
    """
    from .grounded_data import grounded_data
    ph_map = {ph.name: ph for ph in grounded_data.policyholders}
    return [
        CallerProfile(ph_map["Margaret Chen"], claim_hint="denied healthcare claim", style="frustrated"),
        CallerProfile(ph_map["Ava Lopez"], claim_hint="pending healthcare claim", style="confused"),
        CallerProfile(ph_map["Ma Tian"], claim_hint="denied healthcare claim", style="cooperative"),
        CallerProfile(ph_map["Ya Wen Li"], claim_hint="auto collision claim", style="escalating", escalate_after=4),
    ]


def make_train_profiles() -> List["CallerProfile"]:
    """Training combinations include empathy, gate explanations and no-match recovery."""
    from .grounded_data import grounded_data
    ph_map = {ph.name: ph for ph in grounded_data.policyholders}
    return [
        CallerProfile(ph_map["Margaret Chen"], claim_hint="denied healthcare claim", style="frustrated"),
        CallerProfile(ph_map["Ma Tian"], claim_hint="denied healthcare claim", style="cooperative"),
        CallerProfile(ph_map["Ava Lopez"], claim_hint="pending healthcare claim", style="cooperative"),
        CallerProfile(ph_map["Ya Wen Li"], claim_hint="auto collision claim", style="frustrated"),
        CallerProfile(ph_map["Ma Tian"], claim_hint="denied healthcare claim", style="confused"),
        CallerProfile(ph_map["Margaret Chen"], claim_hint="denied healthcare claim", style="privacy_sensitive", email_accepted=False),
        CallerProfile(ph_map["Margaret Chen"], claim_hint="denied dental claim", style="cooperative"),
    ]


def make_val_profiles() -> List["CallerProfile"]:
    """Validation split: confused style."""
    from .grounded_data import grounded_data
    ph_map = {ph.name: ph for ph in grounded_data.policyholders}
    return [
        CallerProfile(ph_map["Ava Lopez"], claim_hint="pending healthcare claim", style="confused"),
        CallerProfile(ph_map["Margaret Chen"], claim_hint="denied healthcare claim", style="confused"),
    ]


def make_test_profiles() -> List["CallerProfile"]:
    """Held-out identity/style/scenario combinations; identities themselves are reused."""
    from .grounded_data import grounded_data
    ph_map = {ph.name: ph for ph in grounded_data.policyholders}
    return [
        CallerProfile(ph_map["Ya Wen Li"], claim_hint="auto collision claim", style="escalating", escalate_after=4),
        CallerProfile(ph_map["Ya Wen Li"], claim_hint="auto collision claim", style="privacy_sensitive", email_accepted=False),
        CallerProfile(ph_map["Ava Lopez"], claim_hint="pending healthcare claim", style="privacy_sensitive"),
        CallerProfile(ph_map["Ma Tian"], claim_hint="denied dental claim", style="escalating", escalate_after=5),
    ]
