"""Reactive caller simulator for AgentPolicyEnv.

CallerProfile represents a synthetic caller with a specific identity and
conversational style.  It produces the next caller utterance in response
to the agent's chosen action and the current SOP state.

Correct pairing with AgentPolicyEnv:

    CallerProfile provides caller utterances
            ↓
    SOP harness updates state (evaluate_turn)
            ↓
    AgentPolicy selects action from action_mask
            ↓
    Agent action → agent reply + reward shaping
            ↓
    CallerProfile reacts to agent reply → next utterance

The CallerProfile is *reactive* to agent actions, not just a pre-written script.
This allows the same profile to work with any agent policy, making RuleBasedPolicy
vs RandomPolicy comparisons meaningful.
"""
import re
from typing import Optional

from .types import AgentAction, Phase, PolicyHolder, ClaimRecord


class CallerProfile:
    """A synthetic caller that reacts to agent actions to drive SOP state.

    The caller provides real PII from the fixture (the same data the state
    machine's UtteranceExtractor knows how to parse), so identity verification
    can actually complete.  Conversational style (cooperative / frustrated /
    escalating) modulates when the caller volunteers information.

    Args:
        policyholder: The fixture PolicyHolder this caller represents.
        claim_hint:   A short description of the claim topic to mention.
        style:        "cooperative" | "frustrated" | "escalating"
                      cooperative: provides PII promptly when asked
                      frustrated:  adds emotional preamble; still cooperates
                      escalating:  cooperative but escalates after N failed turns
        escalate_after: For escalating style, how many turns before demanding human.
    """

    def __init__(
        self,
        policyholder: PolicyHolder,
        claim_hint: str = "denied healthcare claim",
        style: str = "cooperative",
        escalate_after: int = 8,
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
        self._email_accepted = style != "escalating"

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
        elif self.style == "escalating":
            return f"Hello, I need help with my {self.claim_hint}."
        else:
            return (
                f"Hi, my name is {self.ph.name}. "
                f"I'm calling about my {self.claim_hint}."
            )

    def respond_to(self, agent_action: AgentAction, agent_reply: str, state) -> Optional[str]:
        """Generate the caller's next utterance in response to agent action + state.

        Returns None when the caller has nothing more to say (episode should end
        via the agent selecting SEND_EMAIL or ESCALATE_HUMAN).
        """
        self._turn += 1

        # Escalating style: demand human after N turns regardless
        if self.style == "escalating" and self._turn >= self.escalate_after:
            return "I want to speak with a human agent right now."

        return self._respond_by_action(agent_action, state)

    # ------------------------------------------------------------------
    # Internal response logic
    # ------------------------------------------------------------------

    def _respond_by_action(self, action: AgentAction, state) -> Optional[str]:
        verified_fields = set(getattr(state, "verified_fields", []) or [])

        if action == AgentAction.ACK_EMOTION:
            if self.style == "frustrated":
                return (
                    f"Thank you for understanding. "
                    f"I really need to resolve this. My name is {self.ph.name}."
                )
            # Cooperative: express the intent to help the phase advance
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
            return (
                "I understand. Let me give you my information. "
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
                    f"and I got a denial notice last week."
                )
            return "I don't have any other details to add right now."

        elif action == AgentAction.ANSWER_GROUNDED:
            return "Thank you for explaining. What are my next steps?"

        elif action == AgentAction.OFFER_EMAIL_SUMMARY:
            if not self._email_decided:
                self._email_decided = True
                if self._email_accepted:
                    return f"Yes, please send it to {self.ph.email}."
                else:
                    return "No thanks, I don't need an email."
            return "I already answered that."

        elif action == AgentAction.SEND_EMAIL:
            return "Thank you. That's all I needed."

        elif action == AgentAction.ESCALATE_HUMAN:
            return "OK, please transfer me."

        return "I see. Please continue."

    def _provide_next_pii(self, verified_fields: set[str]) -> str:
        """Return a caller utterance with the next missing PII field.

        We provide fields in order: name → dob → phone → email → id_last4.
        We skip fields already verified.  Each field is phrased naturally so
        the UtteranceExtractor's patterns can parse it.
        """
        pii_order = ["name", "dob", "phone", "email", "id_last4"]
        remaining = [f for f in pii_order if f not in verified_fields and f not in self._pii_fields_given]

        if not remaining:
            # All PII given; provide policy number as a fallback
            return f"I've already provided all my details. My policy is {self.ph.policy_number}."

        field = remaining[0]
        self._pii_fields_given.add(field)

        if field == "name":
            return f"My name is {self.ph.name}."
        elif field == "dob":
            # Format as MM/DD/YYYY so extractor's pattern matches
            from datetime import date
            d = date.fromisoformat(self.ph.dob)
            return f"My date of birth is {d.month:02d}/{d.day:02d}/{d.year}."
        elif field == "phone":
            # Format as digits in a natural phrase
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

        return f"I'm not sure what else to provide."


def make_margaret_chen_profile(style: str = "cooperative") -> "CallerProfile":
    """Create a CallerProfile for Margaret Chen (fixture policyholder 0)."""
    from .grounded_data import grounded_data
    ph = next(p for p in grounded_data.policyholders if p.name == "Margaret Chen")
    return CallerProfile(ph, claim_hint="denied healthcare claim", style=style)


def make_all_profiles() -> list["CallerProfile"]:
    """Return one cooperative CallerProfile per fixture policyholder."""
    from .grounded_data import grounded_data
    profiles = []
    for ph in grounded_data.policyholders:
        claim_hint = "denied claim"
        for claim in grounded_data.claims:
            if claim.party_id == ph.party_id:
                claim_hint = f"{claim.case_type} claim"
                break
        profiles.append(CallerProfile(ph, claim_hint=claim_hint, style="cooperative"))
    return profiles
