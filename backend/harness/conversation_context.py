"""Resolve follow-ups against a verified conversation, without replaying old input.

This is a routing aid, never an authorization, PII or consent source. The current
message still goes unchanged to the SOP. Historical prose is not a fact source:
only document names in the currently authorized claim can become a referent.
"""
import re

from .extractor import OUT_OF_SCOPE_PATTERNS, UtteranceExtractor, business_text
from .types import Phase

DOCUMENT_TOPICS = {'documents', 'document_alternatives', 'file_format',
                   'submission_method', 'submission_timing', 'receipt_confirmation'}
ORDINALS = {'first': 0, '1st': 0, 'second': 1, '2nd': 1, 'third': 2, '3rd': 2,
            'fourth': 3, '4th': 3, 'last': -1}
FOLLOWUP_PATTERNS = [
    r"(?:what(?:'s| is| does| about)?|can you explain|explain|tell me about)?\s*(?:the |that )?(?:first|second|third|fourth|last|1st|2nd|3rd|4th)(?: one| item| document| report| requirement)?(?: mean)?",
    r"(?:what(?:'s| is| does)?|can you explain|explain|tell me about)\s+(?:it|that|this|that document|that report|that note)(?: mean)?",
    r"(?:can you |could you |please )?(?:explain|say|put|make)(?: it| that| this)?(?: more)? (?:simply|simple|clearly|briefly|shorter|in plain english|in simple terms|in simple words)",
    r"(?:more |less )?(?:details?|simply|briefly)(?: please)?",
    r"(?:i |but i |what if i )?(?:can't|cannot|can not|couldn't|don't know how to|am unable to) (?:get|obtain|find|provide)(?: it| that| this| one| them| those)?(?: anywhere| anymore| now)?",
    r"(?:would|will|can|could|are|is) (?:a |my |phone |clear )?(?:photos?|pictures?|scans?|copies|a copy|pdfs?)(?: of (?:it|that|them))? (?:work|help|do|be enough|be okay|be ok|acceptable|okay|ok)",
    r"(?:what about|how about|can i use) (?:a |my |phone |clear )?(?:photos?|pictures?|scans?|copies|a copy|pdfs?)(?: instead)?",
    r"(?:can|could|may) i (?:send|submit|upload) (?:a |my |phone |clear )?(?:photos?|pictures?|scans?|copies|a copy|pdfs?)(?: instead| of (?:it|that|them))?",
    r"(?:how|where) (?:do|can|should) i (?:get|find|send|submit|upload) (?:it|that|this|them|those)",
    r"(?:when is|what(?:'s| is)|and|what about)?\s*(?:the |that |my )?(?:deadline|due date)",
    r"(?:is it|am i|is that) (?:already )?too late",
    r"(?:what if|but|and) (?:none|neither|both) (?:is|are) (?:available|possible)",
]


def is_bounded_followup(message):
    """A small syntactic allowance, not a generic scope override from an LLM."""
    text = business_text(message).rstrip(' .?!')
    if any(re.search(pattern, text) for pattern in OUT_OF_SCOPE_PATTERNS):
        return False
    # The original utterance still supplies emotion, PII and consent to the
    # SOP. Remove only an anchored emotion statement for this grammar check.
    text = re.sub(r"^(?:i am|i'm|i feel|this is)(?: really| very| so)? "
                  r"(?:angry|worried|anxious|frustrated|frustrating|scared|upset|confused|ridiculous)\s*[.!,:]\s*", '', text)
    return any(re.fullmatch(pattern, text) for pattern in FOLLOWUP_PATTERNS)


def is_safe_contextual_followup(machine, message, context):
    """Recheck the server-produced allowance at the gate that consumes it."""
    claim = machine.get_active_claim()
    return bool(isinstance(context, dict) and claim
                and machine.state.phase in {Phase.PROCESS_CASE, Phase.POST_PROCESS}
                and context.get('context_claim_id') == claim.case_id
                and context.get('is_contextual_followup')
                and is_bounded_followup(message))


def _mentioned_documents(text, documents):
    low = text.casefold()
    found = []
    for document in documents:
        short = re.sub(r'^(?:original|supplemental|treating provider)\s+', '', document.casefold())
        positions = [low.find(term) for term in {document.casefold(), short} if term and term in low]
        if positions:
            found.append((min(positions), document))
    return [name for _, name in sorted(found)]


def prepare_conversation_turn(machine, message, history=None):
    """Return bounded routing context; never change machine state or user input.

    ``history`` is accepted for the public interface, but grounding comes from
    server-recorded snapshots whose identity and case binding can be checked.
    A supplied/fabricated chat transcript cannot unlock or select a claim.
    """
    result = {'interpreted_text': message, 'response_topics': [],
              'document_focus': [], 'needs_clarification': False,
              'response_style': 'concise', 'context_claim_id': None,
              'is_contextual_followup': False, 'document_explanation': False,
              'displayed_documents': []}
    claim = machine.get_active_claim()
    if not claim or machine.state.phase not in {Phase.PROCESS_CASE, Phase.POST_PROCESS}:
        return result
    result['context_claim_id'] = claim.case_id
    # Closure and email choices are current-turn SOP intents, never document
    # pronouns. In particular "that answers my questions" must not resurrect
    # the preceding material list and mask out the summary offer.
    if machine._wrap_up(message) or (machine.state.phase == Phase.POST_PROCESS and machine._consent(message)):
        return result
    text = business_text(message)
    if any(re.search(pattern, text) for pattern in OUT_OF_SCOPE_PATTERNS):
        return result
    result['is_contextual_followup'] = is_bounded_followup(message)
    topics = UtteranceExtractor.extract_topics(message)
    recent = []
    for snapshot in reversed(machine.state.history_snapshots[-12:]):
        saved = snapshot.state_dump
        if (not saved.get('identity_verified') or saved.get('active_case_id') != claim.case_id
                or saved.get('phase') not in {Phase.PROCESS_CASE, Phase.POST_PROCESS}):
            break
        recent.append(snapshot)
    latest = recent[0] if recent else None
    last_documents = _mentioned_documents(latest.agent_reply, claim.documents_needed) if latest else []
    # A sentence mentioning an office note inside alternative guidance for the
    # pathology report is not an independently displayed list. Prefer the
    # explicit first line when it names a single requested document.
    if latest:
        first_line_docs = _mentioned_documents(latest.agent_reply.split('\n')[0], claim.documents_needed)
        numbered_list = len(re.findall(r'(?:^|\n)\s*\d+[.)]\s+', latest.agent_reply)) >= 2
        if len(first_line_docs) == 1 and not numbered_list:
            last_documents = first_line_docs
    result['displayed_documents'] = last_documents
    explicit_documents = _mentioned_documents(message, claim.documents_needed)
    focus = explicit_documents
    ordinal = re.search(r'\b(first|second|third|fourth|last|1st|2nd|3rd|4th)\b', text)
    if ordinal and result['is_contextual_followup']:
        index = ORDINALS[ordinal.group(1)]
        if last_documents and -len(last_documents) <= index < len(last_documents):
            focus = [last_documents[index]]
            result['document_explanation'] = True
            topics = ['documents']
        else:
            result['needs_clarification'] = True
    refers_back = bool(re.search(r'\b(?:it|that|this|them|those|one)\b', text))
    simplify = bool(re.search(r'\b(?:simply|simpler|simple|briefly|shorter|plain english|less detail)\b', text))
    expand = bool(re.search(r'\b(?:more detail|elaborate|step.by.step)\b', text))
    if not focus and (refers_back or simplify or expand or result['is_contextual_followup']):
        if len(last_documents) == 1:
            focus = last_documents
        elif refers_back and not topics and len(last_documents) > 1:
            result['needs_clarification'] = True
    if not focus and len(last_documents) == 1 and set(topics) & {'file_format', 'document_alternatives'}:
        focus = last_documents
    if result['is_contextual_followup'] and re.search(r"(?:can't|cannot|can not|couldn't|unable to)\s+(?:get|obtain|find|provide)|none|neither", text):
        topics = ['document_alternatives']
        if not focus and len(last_documents) != 1:
            result['needs_clarification'] = True
    if result['is_contextual_followup'] and re.search(r'\b(?:photos?|pictures?|scans?|copies|pdfs?)\b', text):
        topics = ['file_format']
    if result['is_contextual_followup'] and re.search(r'\b(?:get|find)\b', text) and not topics:
        topics = ['document_alternatives']
    if simplify or expand:
        if latest:
            prior_topics = UtteranceExtractor.extract_topics(latest.user_message)
            if is_bounded_followup(latest.user_message) and re.search(
                    r'\b(?:first|second|third|fourth|last|1st|2nd|3rd|4th)\b', business_text(latest.user_message)):
                prior_topics = ['documents']
            if not prior_topics:
                prior_topics = UtteranceExtractor.extract_topics(latest.agent_reply)
            topics = prior_topics or topics
            if focus and not topics:
                topics = ['documents']
            result['document_explanation'] = bool(focus and 'documents' in topics)
        result['response_style'] = 'step_by_step' if expand else 'plain_language'
    if not topics and focus:
        topics = ['documents']
        result['document_explanation'] = True
    if result['is_contextual_followup'] and not topics and not result['needs_clarification']:
        result['needs_clarification'] = True
    result['response_topics'] = topics
    result['document_focus'] = focus
    # Used only as the composer's phrasing input, never as new caller evidence.
    if focus:
        result['interpreted_text'] = message + '\nReferenced requested document: ' + ', '.join(focus)
    return result


def apply_model_reference(machine, message, context, semantic):
    """Accept an LLM's index only within the freshly rechecked displayed list.

    Broader paraphrases can resolve a referent without inventing a document or
    replaying text as identity/consent. This does *not* grant a scope allowance:
    the ordinary scope guard still evaluates the original current utterance.
    """
    fresh = prepare_conversation_turn(machine, message)
    if not isinstance(semantic, dict) or semantic.get('is_out_of_scope') is True:
        return fresh
    index = semantic.get('document_reference')
    documents = fresh['displayed_documents']
    if (type(index) is not int or not 1 <= index <= len(documents)
            or not fresh['context_claim_id']
            or semantic.get('reference_claim_id') != fresh['context_claim_id']):
        return fresh
    topics = [topic for topic in semantic.get('response_topics', []) if topic in DOCUMENT_TOPICS]
    if not topics:
        return fresh
    document = documents[index - 1]
    if fresh['document_focus'] and document not in fresh['document_focus']:
        # Literal/current ordinal evidence outranks a contrary model guess.
        return fresh
    fresh.update({'document_focus': [document], 'needs_clarification': False,
                  'response_topics': topics, 'document_explanation': 'documents' in topics,
                  'interpreted_text': message + '\nReferenced requested document: ' + document,
                  'model_reference_resolved': True})
    return fresh
