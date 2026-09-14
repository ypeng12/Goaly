"""Grounded response composer, shared by deterministic and model-assisted routing.

An LLM may select bounded topics and phrasing style. Only this composer inserts
claim facts and workflow actions, so free-form model text cannot leak or invent them.
"""
import datetime
import re
from typing import Any, Dict, List
from .base import BaseEngine
from ..harness.types import Phase
from ..harness.grounded_data import grounded_data
from ..harness.extractor import UtteranceExtractor, normalize

LABELS = {'name': 'full name', 'dob': 'date of birth', 'phone': 'phone number', 'email': 'email address', 'id_last4': 'last four digits of your SSN'}
EMPATHY = {
    'frustration': "I understand this is frustrating, especially when you need an answer. ",
    'anger': "I hear how upsetting this has been. Let's take it one step at a time. ",
    'anxiety': "I'm sorry this is worrying you. We can take this one step at a time. ",
    'confusion': "I understand this is confusing. Let me make the next step clearer. ",
}


class MockEngine(BaseEngine):
    def generate_response(self, user_text: str, state_result: Dict[str, Any], history: List[Dict[str, str]]) -> str:
        ctx = state_result.get('context', {})
        actions = ctx.get('allowed_actions', [])
        state = state_result.get('state')
        phase = state.phase if state else Phase(ctx.get('phase', 'VERIFY_ID'))
        emotion = state_result.get('emotion') or UtteranceExtractor.detect_emotion_and_intent(user_text)['emotion']
        allow_empathy = state_result.get('policy_empathy', True)
        empathy = EMPATHY.get(emotion, '') if allow_empathy else ''
        if allow_empathy and not empathy and state_result.get('is_refusal'):
            empathy = "You can choose which identity details you feel comfortable sharing. "
        # Terminal behavior is determined by state, never by keywords in the last message.
        if phase == Phase.CONCLUDED:
            pp = state.post_process.model_dump() if state else ctx.get('post_process', {})
            if pp.get('user_decision') == 'accepted':
                return f"The email summary was added to the demo outbox for {pp.get('sent_to', 'your email on file')}. Delivery is simulated; no real email was sent. Thank you for contacting claims support."
            return 'I will skip the email summary as requested. Thank you for contacting claims support.'
        if phase == Phase.ESCALATED or 'HANDOFF_TO_HUMAN' in actions:
            reason = 'A human representative needs to verify your identity and authorization before any claim details can be shared. ' if state_result.get('proxy_consent_status') == 'requires_human' else ''
            return empathy + reason + 'I have recorded a request to connect you with a human claims representative. This demo simulates the handoff; no live transfer takes place. Verification is still required before protected details can be shared.'
        if state_result.get('is_out_of_scope') or 'REJECT_OUT_OF_SCOPE' in actions:
            return empathy + 'I can help with insurance claims, policy questions, and verification, but I am unable to answer that unrelated request. We can continue with your claim, or you can ask for a human representative.'
        if UtteranceExtractor.support_orientation(user_text):
            next_step = ('Before we open your claim, we need 3 matching identity details. You can use the verification form.'
                         if ctx.get('data_shield_active', True) else
                         'Choose a topic below, or describe what happened in your own words.')
            return (empathy + 'A claim is a request for your insurer to pay for a loss or expense. '
                    'I can help you check its progress, understand a rejection, or find out which documents are needed.\n\n' + next_step)
        if ctx.get('data_shield_active', True):
            if state_result.get('proxy_consent_status') == 'unauthorized':
                return empathy + 'To protect privacy, I cannot disclose claim information to unauthorized third parties. We must verify a representative’s own identity and the policyholder’s authorization. Please ask the policyholder to contact support, or request a human representative.'
            fields = ctx.get('collected_fields', [])
            memory = ctx.get('memory', {})
            remembered = 'I’ve saved your claim question for after verification. ' if any(memory.get(k) for k in ['case_type_hint', 'status_hint', 'case_id_hint', 'topic_hint']) else ''
            missing = [LABELS[k] for k in LABELS if k not in fields]
            if len(fields) >= 3:
                request = 'Those details did not match together. Use the verification form to correct them, or share an explicit correction in chat. '
            else:
                request = f"Please share {max(1, 3 - len(fields))} more item{'s' if 3 - len(fields) != 1 else ''}, choosing from " + ', '.join(missing) + '. '
            return (empathy + remembered + 'To protect your privacy, we must verify 3 matching identity details.\n\n'
                    + request + 'Phone or email can replace SSN. A human representative can also help.')
        if phase == Phase.RESOLVE_INTENT:
            candidates = state_result.get('candidate_claims', [])
            resolution = state_result.get('resolution_status', '')
            memory = ctx.get('memory', {})
            if resolution == 'needs_intent':
                return empathy + 'Your identity is verified. What happened that you’d like help with?'
            if candidates and len(candidates) > 1:
                choices = '; '.join(f'{c.case_id}: {c.case_type}, {c.created_at}, {c.status}' for c in candidates)
                return empathy + 'Your identity is verified. I kept the details you mentioned, but more than one claim matches: ' + choices + '. Which claim reference or year did you mean?'
            if resolution == 'no_match' or any(memory.get(k) for k in ['case_type_hint', 'status_hint', 'date_hint', 'case_id_hint']):
                return empathy + 'Your identity is verified. I kept your earlier claim details, but I could not find a matching claim on this account. Please confirm the claim reference, type, or year, or ask for a human representative.'
            return empathy + 'Your identity is verified. What would you like help with—claim status, a denial, payment, or submitting documents? You can also give a claim reference.'
        claim = state_result.get('active_claim')
        # A second check prevents accidentally rendering a raw ungated record.
        if not claim or 'grounded_claim' not in ctx:
            return empathy + 'I do not have a verified claim record for that question. A human claims representative can help review it.'
        conversation = state_result.get('conversation_context') or {}
        if conversation.get('context_claim_id') != claim.case_id:
            conversation = {}
        # Re-intersect even server-produced referents with the current record.
        document_focus = [d for d in conversation.get('document_focus', []) if d in claim.documents_needed]
        focused_text = user_text + ('\nReferenced requested document: ' + ', '.join(document_focus) if document_focus else '')
        if conversation.get('needs_clarification'):
            choices = '; '.join(f'{i}. {d}' for i, d in enumerate(claim.documents_needed, 1))
            question = ('Which document do you mean? ' + choices + '.' if choices else
                        'Which part would you like me to explain: the status, payment, or next step?')
            if phase == Phase.POST_PROCESS:
                question += ' Your email choice is still pending; nothing has been sent.'
            return empathy + question
        if phase == Phase.POST_PROCESS:
            pp = state.post_process if state else None
            text = (f'Would you like an email summary of our conversation for claim {claim.case_id}, or would you prefer to skip it? '
                    'It includes what we discussed, the claim status and outcome, and the major follow-up items. Email delivery is simulated in this demo.')
            if pp and pp.user_decision == 'pending' and ('?' in user_text or conversation.get('is_contextual_followup')):
                topics = conversation.get('response_topics') or state_result.get('response_topics') or UtteranceExtractor.extract_topics(user_text)
                guidance = grounded_data.get_document_guidance_for_claim(claim)
                pieces = [self._focused_fact(topic, claim, guidance, focused_text, conversation, document_focus) for topic in topics]
                if pieces:
                    state_result['grounded_topics'] = topics
                    if state and hasattr(state, 'discussion_topics'):
                        state.discussion_topics = list(dict.fromkeys(state.discussion_topics + topics))
                    text = '\n\n'.join(p for p in pieces if p) + '\n\n' + text
                else:
                    text = 'You can choose freely; nothing is sent without your explicit agreement. ' + text
            return empathy + text
        topics = conversation.get('response_topics') or state_result.get('response_topics') or UtteranceExtractor.extract_topics(user_text)
        # Preserve first-turn follow-up topics provided during verification.
        if not topics and ctx.get('memory', {}).get('topic_hint'):
            topics = [ctx['memory']['topic_hint']]
        newly_opened = any(t.gate_evaluated == 'RESOLVE_INTENT_GATE' and t.gate_passed for t in (state.trace_log[-1:] if state else []))
        accepted_offer = newly_opened and any(t.gate_evaluated == 'CONVERSATIONAL_SELECTION'
                                              for t in (state.trace_log[-2:] if state else []))
        if not topics:
            topics = ['status'] if newly_opened else ['unknown']
        if accepted_offer:
            topics = ['status']  # Answer the status check we offered, not an unsolicited full dossier.
        elif newly_opened:
            topics = list(dict.fromkeys(['status', 'denial_reason', 'documents', 'appeal_deadline'] + topics))
        guidance = grounded_data.get_document_guidance_for_claim(claim)
        pieces = []
        used = []
        for topic in dict.fromkeys(topics):
            piece = self._focused_fact(topic, claim, guidance, focused_text, conversation, document_focus)
            if piece and piece not in pieces:
                pieces.append(piece)
                used.append(topic)
        if accepted_offer and claim.summary:
            pieces.append('The record says: ' + claim.summary.rstrip('.') + '.')
        state_result['grounded_topics'] = used
        if state and hasattr(state, 'discussion_topics'):
            state.discussion_topics = list(dict.fromkeys(state.discussion_topics + used))
        intro = 'Identity verified. I found the claim using your earlier details.\n\n' if newly_opened and not accepted_offer else ''
        style = (conversation['response_style'] if conversation.get('response_style') in {'plain_language', 'step_by_step'}
                 else state_result.get('response_style', 'concise'))
        if style == 'step_by_step' and len(pieces) > 1:
            body = '\n'.join(f'{i}. {p}' for i, p in enumerate(pieces, 1))
        else:
            body = '\n\n'.join(pieces)
        # Keep the next step available without repeating the same invitation on
        # every follow-up. Contextual turns answer the current question directly.
        ending = "\n\nChoose a next step below, or ask in your own words." if newly_opened and not accepted_offer else ''
        state_result['_response_envelope'] = {'prefix': empathy + intro, 'facts': pieces,
                                               'topics': used, 'ending': ending}
        return empathy + intro + body + ending

    @staticmethod
    def _focused_fact(topic, claim, guidance, user_text, conversation, focus):
        if topic == 'documents' and focus and conversation.get('document_explanation'):
            if conversation.get('response_style') == 'plain_language':
                # Keep the complete approved guidance, but unpack its sentences
                # into small steps rather than inventing a policy explanation.
                return '\n\n'.join(d.capitalize() + ' — in plain language:\n' +
                                   '\n'.join('• ' + sentence for sentence in re.split(r'(?<=\.)\s+', guidance['concise_documents'][d]))
                                   for d in focus)
            return '\n\n'.join(d.capitalize() + ': ' + guidance['document_guidance'][d] for d in focus)
        if topic == 'file_format' and focus:
            detailed = bool(re.search(r'\b(?:detail|details|elaborate|expand|step.by.step)\b', user_text, re.I))
            source = guidance['document_guidance'] if detailed else guidance['concise_documents']
            answer = '\n\n'.join(d.capitalize() + ': ' + source[d] for d in focus)
            if (re.search(r'\b(?:photo|photos|picture|pictures)\b', user_text, re.I)
                    and not any('photo' in document.casefold() for document in focus)):
                # The fixture gives readability requirements and some scan
                # guidance, but does not promise that every photo file type
                # is accepted. Answer that distinction explicitly.
                answer = ("The record does not confirm that a photo is an accepted file type. Check the portal's format instructions or ask support. "
                          'Every page must be legible and complete.\n\n' + answer)
            return answer
        return MockEngine._fact(topic, claim, guidance, user_text)

    @staticmethod
    def _fact(topic, claim, guidance, user_text):
        docs = ', '.join(claim.documents_needed)
        if topic == 'status':
            status = {'in_review': 'under review'}.get(claim.status, claim.status.replace('_', ' '))
            article = 'an' if claim.case_type[:1].lower() in 'aeiou' else 'a'
            return f'Claim {claim.case_id} is {status}. It is {article} {claim.case_type} claim created on {claim.created_at}.'
        if topic == 'denial_reason':
            return f'The recorded denial reason is that {claim.denial_reason}.' if claim.denial_reason else ''
        if topic == 'documents':
            return f'The requested documents are: {docs}.' if docs else 'The record does not list any outstanding documents.'
        if topic == 'payment':
            labels = [('allowed_max_amount', 'Allowed maximum'), ('net_fee', 'Net fee'), ('net_pay', 'Finalized insurer payment'), ('expected_reimbursement_amount', 'Expected reimbursement')]
            amounts = '; '.join(f'{label}: ${getattr(claim, k)}' for k, label in labels if getattr(claim, k) is not None)
            return f'For claim {claim.case_id}: {amounts}. The allowed maximum is not a promise of reimbursement.' if amounts else 'The record does not include payment amounts.'
        if topic == 'appeal_deadline':
            if not claim.appeal_deadline:
                return ''
            past = datetime.date.fromisoformat(claim.appeal_deadline) < datetime.datetime.now(datetime.timezone.utc).date()
            return f'The recorded appeal deadline is {claim.appeal_deadline}. ' + ('That date has passed. A human claims representative must review whether any options remain; I cannot promise an extension.' if past else 'Submitting documents does not guarantee approval.')
        if topic == 'document_alternatives':
            if not docs:
                return 'The claim record does not request additional documents, so I cannot identify a required substitute from this file.'
            if re.search(r'none|no (?:readable )?copy|cannot reissue|tried.*(?:hospital|provider)|alternatives.*(?:exhausted|unavailable)', normalize(user_text), re.I):
                return guidance['claim_followup_settings']['human_review_after_document_alternatives_exhausted']['en']
            specific = [d for d in claim.documents_needed if d.lower() in user_text.lower()]
            detailed = bool(re.search(r'\b(?:detail|details|elaborate|expand|step.by.step)\b', user_text, re.I))
            source = guidance['document_alternative_guidance'] if detailed else guidance['concise_alternatives']
            return '\n'.join(source[d] for d in (specific or claim.documents_needed))
        qa_topic = {'file_format': 'file_format_requirements', 'processing_time': 'processing_time_after_submission'}.get(topic, topic)
        if topic == 'submission_method':
            return guidance['default_guidance']
        if topic in ['file_format', 'submission_timing', 'processing_time', 'receipt_confirmation']:
            if not docs:
                return 'The record lists no outstanding documents or document review estimate for this claim.'
            item = next((q for q in guidance['followup_qa'] if q['topic'] == qa_topic), None)
            if item:
                text = item['en'].format(case_id=claim.case_id, documents=docs, average_processing_time_after_submission=guidance['claim_followup_settings']['average_processing_time_after_submission']['en'])
                if topic == 'submission_timing' and claim.appeal_deadline and datetime.date.fromisoformat(claim.appeal_deadline) < datetime.datetime.now(datetime.timezone.utc).date():
                    text = 'The general document guidance says to submit within a week, but ' + MockEngine._fact('appeal_deadline', claim, guidance, user_text)
                if topic == 'file_format':
                    detailed = bool(re.search(r'\b(?:detail|details|elaborate|expand|step.by.step)\b', user_text, re.I))
                    source = guidance['document_guidance'] if detailed else guidance['concise_documents']
                    text = '\n'.join(source.values())
                return text
        return 'I do not have a grounded rule or claim detail that answers that question. A human claims representative can review it; I cannot change a claim decision or promise coverage.'
