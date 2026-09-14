"""Execute a selected dialogue action for either a human or training caller.

No caller text is fabricated here. The caller's original message has already
been consumed by the SOP. Policy actions choose how to respond; the machine
still owns claim access, phase transitions and side effects.
"""
from ..engine.mock_engine import EMPATHY, MockEngine
from .types import AgentAction, Phase


def render_policy_reply(machine, message, result, conversation_context, decision,
                        *, engine=None, history=None, enable_response_planning=False):
    engine = engine or MockEngine()
    history = history or []
    context = conversation_context or {}
    action_name = decision.get('selected_action')
    action = AgentAction(action_name) if action_name else None
    execution = decision.get('execution', {})
    result['conversation_context'] = context
    result['policy_action'] = action_name
    if action is not None:
        result['policy_empathy'] = action == AgentAction.ACK_EMOTION
    result['enable_response_planning'] = enable_response_planning
    result['grounded_answer_delivered'] = False

    def compose():
        reply = engine.generate_response(message, result, history)
        used = set(result.get('grounded_topics', [])) & machine.TOPICS
        result['grounded_answer_delivered'] = bool(
            used and machine.get_active_claim() and not context.get('needs_clarification'))
        return reply

    # Mandatory decisions are executed as SOP responses, including terminal
    # consent acknowledgements. No checkpoint receives credit for those turns.
    if action is None or decision.get('source') == 'sop':
        return compose()
    if action.value not in decision.get('allowed_actions', []):
        raise ValueError('Cannot execute an action outside the customer SOP mask')

    def identity_question(explain=False):
        if machine.state.phase != Phase.VERIFY_ID:
            raise ValueError('Identity requests are only permitted during verification')
        label = execution.get('identity_label', 'one identity detail')
        if label == 'full name':
            question = ('Please use Verify using a form to enter your full name exactly as it appears '
                        'on your policy. Include a middle name or suffix if it appears there.')
        elif execution.get('correction_needed'):
            question = f'Please correct your {label}, or use the verification form to review the details you provided.'
        elif label == 'date of birth':
            question = ('Please share your date of birth. You can type 1985 3 15, '
                        '1985-03-15, or 03/15/1985. You can also use the verification form.')
        else:
            question = f'Please share your {label}. You can also use the verification form.'
        remembered = 'I’ve kept your claim question for after verification. ' if any(
            getattr(machine.state.cross_phase_memory, key) for key in machine.HINT_NAMES) else ''
        if explain:
            return (remembered + 'To protect your privacy, claim details stay locked until 3 identity details match. '
                    'Phone or email can replace SSN.\n\n' + question)
        return remembered + question

    def clarify():
        if machine.state.phase == Phase.RESOLVE_INTENT:
            if action == AgentAction.ASK_CLAIM_CLARIFICATION:
                candidates = result.get('candidate_claims', [])
                if not candidates:
                    return 'I could not match the claim yet. What claim reference, type or year appears on your letter? You can also ask for a human representative.'
            return compose()
        claim = machine.get_active_claim()
        if not claim:
            return compose()
        docs = ', '.join(claim.documents_needed)
        if docs and context.get('needs_clarification'):
            return f'Which item do you mean: {docs}? I want to give you the right instructions.'
        return 'Which part would you like me to explain: the claim status, requested documents, or next steps?'

    if action == AgentAction.ASK_IDENTITY_FIELD:
        return identity_question()
    if action == AgentAction.EXPLAIN_VERIFICATION_GATE:
        return identity_question(explain=True)
    if action == AgentAction.ESCALATE_HUMAN:
        # The mask permits this only after resolution attempts are exhausted;
        # recheck the state as a second boundary before applying the side effect.
        if machine.state.phase != Phase.RESOLVE_INTENT or execution.get('escalation_reason') != 'case_resolution_exhausted':
            raise ValueError('This policy handoff is not authorized at this phase')
        before = machine.state.phase
        machine.state.phase = Phase.ESCALATED
        machine._add_trace('CUSTOMER_POLICY_HANDOFF', True, 'Claim clarification attempts exhausted; requested a simulated human handoff.', before)
        result.update(machine._result(before))
        return compose()
    if action in {AgentAction.RESOLVE_INTENT, AgentAction.ASK_CLAIM_CLARIFICATION}:
        return clarify()
    if action == AgentAction.ACK_EMOTION:
        empathy = EMPATHY.get(execution.get('emotion'), 'Thank you for explaining. We can take this one step at a time. ')
        followup = execution.get('followup_kind')
        if followup == 'ask_identity_field':
            return empathy + '\n\n' + identity_question()
        if followup == 'ask_claim_clarification':
            return empathy + '\n\n' + clarify()
        body = compose()
        return body if body.startswith(empathy) else empathy + '\n\n' + body
    if action == AgentAction.ANSWER_GROUNDED:
        if not machine.get_active_claim():
            raise ValueError('A verified owned claim is required to answer')
        return compose()
    if action == AgentAction.OFFER_EMAIL_SUMMARY:
        if machine.state.phase != Phase.POST_PROCESS:
            raise ValueError('The caller has not finished claim questions')
        return compose()
    # Consent is processed once by the SOP, not resubmitted as invented speech.
    raise ValueError('Unsupported customer action; consent belongs to the SOP')
