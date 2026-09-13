"""Reactive caller simulator for AgentPolicyEnv.

CallerProfile represents a synthetic caller with a specific identity and
conversational style.  It produces the next caller utterance in response
to the agent's chosen action and the current SOP state.

Supported styles:
- cooperative: provides PII promptly, follows SOP flow smoothly.
- frustrated: adds emotional preamble; cooperates with empathy/explanation.
- confused: uncertain about required info, provides non-counting details first.
- privacy_sensitive: refuses SSN/ID initially, prefers phone/email.
- escalating: cooperative early, then explicitly demands human escalation.
"""
import copy
import re
from typing import Optional, List, Dict, Any

from .types import AgentAction, Phase, PolicyHolder, ClaimRecord


class CallerProfile:
    """A synthetic caller that reacts to agent actions to drive SOP state."""

    def __init__(
        self,
        policyholder: PolicyHolder,
        claim_hint: str = "denied healthcare claim",
        style: str = "cooperative",
        escalate_after: int = 6,
    ):
        self.ph = policyholder
        self.claim_hint = claim_hint
        self.style = style
        self.escalate_after = escalate_after

        # Conversation state
        self._turn = 0
        self._pii_fields_given: set[str] = set()
        self._intent_expressed = False
        self._claim_details_given = False
        self._email_decided = False
        self._email_accepted = style not in {"escalating"}

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_opening_utterance(self) -> str:
        """First message the caller sends before any agent action."""
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
            return f"Hello, I need urgent supervisor assistance regarding my {self.claim_hint}."
        else:
            return (
                f"Hi, my name is {self.ph.name}. "
                f"I'm calling about my {self.claim_hint}."
            )

    def respond_to(self, agent_action: AgentAction, agent_reply: str, state) -> Optional[str]:
        """Generate the caller's next utterance in response to agent action + state."""
        self._turn += 1

        # Escalating style: demand human after N turns regardless
        if self.style == "escalating" and self._turn >= self.escalate_after:
            return "I have had enough waiting. I want to speak with a human agent right now."

        return self._respond_by_action(agent_action, state)

    # ------------------------------------------------------------------
    # Internal response logic
    # ------------------------------------------------------------------

    def _respond_by_action(self, action: AgentAction, state) -> Optional[str]:
        verified_fields = set(getattr(state, "verified_fields", []) or [])

        if action == AgentAction.ACK_EMOTION:
            if self.style in {"frustrated", "escalating"}:
                return (
                    f"Thank you for understanding. I really need to resolve this. "
                    f"My name is {self.ph.name}."
                )
            elif self.style == "confused":
                return "Thank you for being patient with me. What information do you need first?"
            # Cooperative
            if not self._intent_expressed:
                self._intent_expressed = True
                return (
                    f"I appreciate that. I'm calling about my {self.claim_hint}. "
                    f"Can you help me understand the status?"
                )
            return "I appreciate that. Let's continue with my claim question."

        elif action == AgentAction.ASK_IDENTITY_FIELD:
            return self._provide_next_pii(verified_fields)

        elif action == AgentAction.EXPLAIN_VERIFICATION_GATE:
            if self.style == "privacy_sensitive":
                return (
                    "Thank you for clarifying. Since you offer alternatives to SSN, "
                    + self._provide_next_pii(verified_fields)
                )
            return (
                "I understand why security is needed. Let me give you my information. "
                + self._provide_next_pii(verified_fields)
            )

        elif action == AgentAction.RESOLVE_INTENT:
            if not self._intent_expressed:
                self._intent_expressed = True
                return (
                    f"I need to understand the status of my {self.claim_hint}. "
                    f"Can you tell me why it was denied and what I should do next?"
                )
            return "Yes, that's exactly what I need help with."

        elif action == AgentAction.ASK_CLAIM_CLARIFICATION:
            if not self._claim_details_given:
                self._claim_details_given = True
                return (
                    f"It was a {self.claim_hint}. The claim was submitted about a month ago "
                    f"and I received a notice of denial last week."
                )
            return "I don't have any other details to add right now."

        elif action == AgentAction.ANSWER_GROUNDED:
            return "Thank you for explaining the claim details clearly. That is all the questions I have."

        elif action == AgentAction.OFFER_EMAIL_SUMMARY:
            if not self._email_decided:
                self._email_decided = True
                if self._email_accepted:
                    return "Yes, please send the summary to my email."
                else:
                    return "No thanks, please skip the summary."
            return "I already answered that."

        elif action == AgentAction.SEND_EMAIL:
            return "Thank you. That covers everything I needed today."

        elif action == AgentAction.ESCALATE_HUMAN:
            return "Understood. Please connect me with the supervisor."

        return "I see. Please continue."

    def _provide_next_pii(self, verified_fields: set[str]) -> str:
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

        field = remaining[0]
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
        CallerProfile(ph_map["Ma Tian"], claim_hint="property damage claim", style="cooperative"),
        CallerProfile(ph_map["Ya Wen Li"], claim_hint="auto collision claim", style="escalating", escalate_after=4),
    ]


def make_train_profiles() -> List["CallerProfile"]:
    """Training split: cooperative + frustrated styles."""
    from .grounded_data import grounded_data
    ph_map = {ph.name: ph for ph in grounded_data.policyholders}
    return [
        CallerProfile(ph_map["Margaret Chen"], claim_hint="denied healthcare claim", style="frustrated"),
        CallerProfile(ph_map["Ma Tian"], claim_hint="property damage claim", style="cooperative"),
        CallerProfile(ph_map["Ava Lopez"], claim_hint="pending healthcare claim", style="cooperative"),
        CallerProfile(ph_map["Ya Wen Li"], claim_hint="auto collision claim", style="frustrated"),
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
    """Test split: escalating + privacy_sensitive styles."""
    from .grounded_data import grounded_data
    ph_map = {ph.name: ph for ph in grounded_data.policyholders}
    return [
        CallerProfile(ph_map["Ya Wen Li"], claim_hint="auto collision claim", style="escalating", escalate_after=4),
        CallerProfile(ph_map["Margaret Chen"], claim_hint="denied healthcare claim", style="privacy_sensitive"),
        CallerProfile(ph_map["Ava Lopez"], claim_hint="pending healthcare claim", style="privacy_sensitive"),
        CallerProfile(ph_map["Ma Tian"], claim_hint="property damage claim", style="escalating", escalate_after=5),
    ]
