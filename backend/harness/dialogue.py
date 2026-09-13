"""Observable dialogue signals and bounded action rendering for the RL harness.

No caller profile labels or private simulator state are exposed to the policy.
The simulator is deliberately a finite text protocol, not a human language model.
"""
import re

from .extractor import UtteranceExtractor
from .types import AgentAction
from ..engine.mock_engine import MockEngine

EMOTIONS = ('neutral', 'frustration', 'anger', 'anxiety', 'confusion')


def dialogue_signals(text):
    signals = UtteranceExtractor.detect_emotion_and_intent(text)
    low = text.lower()
    if "not really sure" in low or "one thing at a time" in low:
        signals['emotion'] = 'confusion'
    signals['privacy_concern'] = bool(re.search(
        r"prefer not to share|sensitive government|rather use phone|avoid.*(?:ssn|id)", low
    ))
    return signals


def render_action(action, machine, caller_text):
    """Render the selected act; protected facts come only from gated getters."""
    state = machine.state
    signals = dialogue_signals(caller_text)
    if action == AgentAction.ACK_EMOTION:
        label = {'frustration': 'frustrating', 'anger': 'upsetting',
                 'anxiety': 'worrying', 'confusion': 'confusing'}.get(signals['emotion'], 'difficult')
        return f"I hear how {label} this is. Thank you for explaining your concern. We can take this at your pace."
    if action in {AgentAction.ASK_IDENTITY_FIELD, AgentAction.EXPLAIN_VERIFICATION_GATE}:
        labels = {'name': 'full name', 'dob': 'date of birth', 'phone': 'phone number',
                  'email': 'email address', 'id_last4': 'government ID last four'}
        field = next((f for f in labels if f not in state.verified_fields), 'email')
        question = f"Please share your {labels[field]}."
        if action == AgentAction.ASK_IDENTITY_FIELD:
            return question
        return ("To protect your claim information, we need three matching identity details. "
                "You can use phone or email instead of a government ID. Let's take one thing at a time. " + question)
    if action == AgentAction.RESOLVE_INTENT:
        hints = state.cross_phase_memory
        remembered = ' / '.join(str(getattr(hints, k)) for k in
                               ('case_type_hint', 'status_hint', 'date_hint') if getattr(hints, k))
        return f"I saved your earlier claim question{': ' + remembered if remembered else ''}. What would you like to understand about it?"
    if action == AgentAction.ASK_CLAIM_CLARIFICATION:
        return "Which claim reference or year appears on your letter? I want to match the right case."
    if action == AgentAction.ANSWER_GROUNDED:
        claim = machine.get_active_claim()
        if claim is None:
            return "I cannot open claim details until your identity and case are verified."
        # Reuse the runtime's grounded composer, with the caller's real question.
        result = machine._result(state.phase)
        result['response_topics'] = ['status', 'denial_reason', 'documents', 'appeal_deadline']
        reply = MockEngine().generate_response(caller_text, result, [])
        machine.record_grounded_topics(result.get('grounded_topics', []))
        return reply
    if action == AgentAction.OFFER_EMAIL_SUMMARY:
        return "Would you like an email summary of what we discussed, the claim outcome and next steps, or would you prefer to skip it?"
    if action == AgentAction.SEND_EMAIL:
        return "Your consent is recorded. The summary is in the demo outbox; delivery is simulated."
    if action == AgentAction.ESCALATE_HUMAN:
        return "I will connect you with a human claims representative. This demo simulates the handoff; verification is still required for protected details."
    raise ValueError(f'Unknown dialogue action: {action}')
