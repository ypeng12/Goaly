"""Conversational repair within RESOLVE_INTENT, shared by chat and simulation.

Remember the question actually asked, not just the workflow phase. Model acts
may inform phrasing; only an explicit current reply to a fresh server-issued
offer can select a previously displayed, still-owned claim. No synthetic user
utterance is injected, and this module cannot verify identity or send email.
"""
import datetime
import re

from .extractor import OUT_OF_SCOPE_PATTERNS, business_text, UtteranceExtractor
from .grounded_data import grounded_data
from .types import IntentDialogueState, Phase

REPAIR_ACTS = {'unsure', 'generic_claim', 'service_feedback', 'ask_help', 'decline'}


def is_short_acknowledgement(message):
    text = ' '.join(business_text(message).split()).strip(' .!')
    return text in {'yes', 'yes please', 'yes, please', 'sure', 'okay', 'ok',
                    'please do', 'go ahead', 'that one', 'yes, that one',
                    'no', 'no thanks', 'something else', 'not that'}


def local_act(message):
    text = ' '.join(business_text(message).replace('don;t', "don't").replace('dont', "don't").split()).strip(' .!?')
    if any(re.search(p, text) for p in OUT_OF_SCOPE_PATTERNS):
        return 'none'
    if re.fullmatch(r"(?:i )?(?:don't know|do not know|(?:am |i'm )?not sure|have no idea|(?:am |i'm )?confused)(?: (?:what you mean|what to (?:say|choose|ask)|where to start))?", text):
        return 'unsure'
    if re.fullmatch(r"(?:my |a |the |insurance )?claims?(?: please)?", text):
        return 'generic_claim'
    if re.fullmatch(r"(?:i (?:hate you|am frustrated|am angry)|this is (?:ridiculous|frustrating|annoying)|you (?:keep (?:repeating yourself|asking the same question)|are not listening|aren't listening)|stop (?:repeating yourself|asking the same question))", text):
        return 'service_feedback'
    if re.fullmatch(r"(?:(?:can|could) you )?(?:help me understand what you (?:need|mean)|what do you (?:need|mean)|what should i (?:say|tell you)|how do i answer(?: you| that)?|help me(?: please)?)", text):
        return 'ask_help'
    if text in {'no', 'no thanks', 'something else', 'not that', "that's not what i mean", 'not what i meant'}:
        return 'decline'
    return 'none'


def dialogue_context(machine):
    """Only trust a question from this owner's immediately preceding turn."""
    owner = machine.get_verified_policyholder()
    saved = machine.state.intent_dialogue
    if (machine.state.phase != Phase.RESOLVE_INTENT or not owner
            or saved.party_id != owner.party_id or not saved.question
            or saved.issued_turn != len(machine.state.history_snapshots) - 1):
        return {}
    snapshots = machine.state.history_snapshots
    if (not snapshots or snapshots[-1].state_dump.get('intent_dialogue') != saved.model_dump()
            or not snapshots[-1].state_dump.get('identity_verified')
            or snapshots[-1].state_dump.get('verified_party_id') != owner.party_id):
        return {}
    return {'pending_question': saved.question, 'last_question': saved.prompt,
            'recent_caller_acts': saved.recent_acts[-4:]}


def is_service_repair(machine, message):
    # An anchored grammar, not an LLM assertion, grants this local scope repair.
    return bool(dialogue_context(machine) and local_act(message) in REPAIR_ACTS)


def confirmed_status_case(machine, message):
    """Return a fresh owned referent; neither raw history nor model says 'yes'."""
    context = dialogue_context(machine)
    if context.get('pending_question') != 'offer_status':
        return None
    text = ' '.join(business_text(message).split()).strip(' .!')
    if not re.fullmatch(r"(?:yes|yes please|yes, please|sure|okay|ok|please do|go ahead|check its progress|yes, check its progress|yes check its progress|that one|yes, that one)", text):
        return None
    owner = machine.get_verified_policyholder()
    _, candidates = grounded_data.find_claim(owner.party_id, machine.state.cross_phase_memory)
    return next((c for c in candidates if c.case_id == machine.state.intent_dialogue.offered_case_id
                 and c.party_id == owner.party_id), None)


def accept_status_offer(machine, message):
    claim = confirmed_status_case(machine, message)
    if not claim:
        return False
    memory = machine.state.cross_phase_memory
    # Merge, do not erase contrary evidence. The ordinary resolver rechecks all
    # accumulated hints and ownership before the phase can advance.
    memory.case_id_hint = claim.case_id
    ids = memory.hint_history.setdefault('case_id_hint', [])
    if claim.case_id not in ids:
        ids.append(claim.case_id)
    memory.topic_hint = 'status'
    machine.state.intent_dialogue = IntentDialogueState()
    machine._add_trace('CONVERSATIONAL_SELECTION', True,
                       'Caller explicitly accepted the preceding owned-claim status offer. No identity or email consent inferred.',
                       Phase.RESOLVE_INTENT)
    return True


def render_intent_reply(machine, message, context):
    if machine.state.phase != Phase.RESOLVE_INTENT or not machine.get_verified_policyholder():
        machine.state.intent_dialogue = IntentDialogueState()
        return None
    if machine.state.out_of_scope_count:
        machine.state.intent_dialogue = IntentDialogueState()
        return None
    # Existing grounded case ambiguity/no-match responses keep their evidence.
    if machine.state.resolution_status != 'needs_intent':
        machine.state.intent_dialogue = IntentDialogueState()
        return None
    owner = machine.get_verified_policyholder()
    previous = dialogue_context(machine)
    saved = machine.state.intent_dialogue
    act = local_act(message)
    proposed = context.get('dialogue_act')
    if act == 'none' and previous and proposed in REPAIR_ACTS:
        act = proposed  # Response selection only; never a scope/claim/consent grant.
    if act == 'none' and context.get('emotion') in {'anger', 'frustration'}:
        act = 'service_feedback'
    elif act == 'none' and context.get('emotion') == 'confusion':
        act = 'unsure'
    recent = saved.recent_acts[-3:] if previous else []
    question, offered = 'ask_reason', None
    prefix = ''
    _, candidates = grounded_data.find_claim(owner.party_id, machine.state.cross_phase_memory)

    if act == 'service_feedback':
        prefix = ("I'm sorry—I've been asking you to choose without helping you work it out. "
                  if previous else "I'm sorry this has been frustrating. ")
    elif act == 'generic_claim':
        prefix = 'Okay, let’s start with your claim. '
    elif act in {'unsure', 'ask_help'}:
        prefix = ("I'm asking what made you contact us today. " if act == 'ask_help'
                  else "That's okay. You don't need the insurance terms. ")

    if act == 'decline':
        question = 'ask_goal'
        text = ("Okay, I won't assume you want a status check. " if previous.get('pending_question') == 'offer_status' else 'Okay. ')
        text += 'What would you like to understand about your claim? You can describe it in your own words.'
    elif previous.get('pending_question') == 'offer_status' and act == 'service_feedback':
        question, offered = 'offer_status', saved.offered_case_id
        text = prefix + 'I can start by checking where that claim stands, so you don’t have to explain it all. Shall I do that?'
    elif (act in {'generic_claim', 'service_feedback'}
          or act in {'unsure', 'ask_help'} and previous.get('pending_question') in {'describe_problem', 'ask_goal'}):
        if len(candidates) == 1:
            claim = candidates[0]
            question, offered = 'offer_status', claim.case_id
            date = datetime.date.fromisoformat(claim.created_at)
            kind = {'auto': 'car', 'healthcare': 'medical', 'dental': 'dental'}.get(claim.case_type, claim.case_type)
            text = (prefix + f'I can see a {kind} claim dated {date.strftime("%B")} {date.day}, {date.year}. '
                    'Would you like me to check its progress?')
        elif candidates:
            question = 'identify_claim'
            kinds = list(dict.fromkeys({'auto': 'car', 'healthcare': 'medical'}.get(c.case_type, c.case_type) for c in candidates))
            text = prefix + 'Which was it about: ' + ', '.join(kinds) + '? An approximate date also helps; you don’t need a claim number.'
        else:
            question = 'describe_problem'
            text = prefix + 'Tell me what happened or what a letter or bill says, and I’ll help work out the next step.'
    elif act in {'unsure', 'ask_help'}:
        question = 'describe_problem'
        text = prefix + 'What brought you here—a letter you received, a bill, or waiting for an update?'
    elif previous:
        question = 'describe_problem'
        text = 'You can tell me what happened in your own words. What part has been worrying you or seems unclear?'
    else:
        text = 'Your identity is verified. What happened that you’d like help with?'

    if UtteranceExtractor.support_orientation(message) and re.search(r'what (?:is|does)', business_text(message)):
        text = 'A claim is a request for your insurer to pay for a loss or expense. ' + text
    machine.state.intent_dialogue = IntentDialogueState(
        party_id=owner.party_id, question=question, prompt=text, offered_case_id=offered,
        issued_turn=len(machine.state.history_snapshots), recent_acts=recent + [act])
    return text


def public_guidance(machine):
    context = dialogue_context(machine)
    question = context.get('pending_question')
    choices = []
    if question == 'offer_status' and confirmed_status_case(machine, 'yes'):
        choices = [{'label': 'Check its progress', 'message': 'Yes, check its progress.'},
                   {'label': 'Something else', 'message': 'Something else.'}]
    return {'question': question, 'choices': choices}
