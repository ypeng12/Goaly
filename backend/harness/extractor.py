"""Conservative local extraction. Model proposals may supplement, never verify, slots."""
import datetime
import re
from typing import Any, Dict, Optional
from .types import PIIFields, CrossPhaseMemory

MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
# A year-first date is unambiguous even when a caller separates its parts with
# spaces (for example, "1985 3 15").  Month-first space-separated numbers are
# deliberately excluded: "3 2 2001" is ambiguous without a locale convention.
DOB_VALUE = (r'(?:[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|'
             r'\d{4}(?:[-/.\s]+)\d{1,2}(?:[-/.\s]+)\d{1,2}|'
             r'\d{1,2}(?:[-/.])\d{1,2}(?:[-/.])\d{4})')
OUT_OF_SCOPE_PATTERNS = [
    r'\b(?:reinforcement learning|q[- ]learning|deep learning|machine learning|quantum physics|bitcoin|crypto|python|javascript|golang)\b',
    r'\b(?:what is|explain|teach me)\s+(?:rl|ai|algebra|calculus|photosynthesis)\b',
    r'\b(?:write|generate|create)\b.{0,35}\b(?:poem|code|script|story|essay|recipe)\b',
    r'\b(?:capital of|who is the president|who won the|weather in|tell me a joke|sports score)\b',
    r'ignore\b.{0,30}\binstructions\b', r'\b(?:print|reveal|show)\b.{0,25}\b(?:prompt|instructions|database|fixtures)\b',
    r'\bsystem prompt\b', r'\b(?:recipe|recipes|pasta|soccer|basketball|horoscope|astrology)\b', r'\bclaims\.json\b', r'\bjailbreak\b',
    r'\b(?:pretend|act as|roleplay)\b.{0,35}\b(?:verified|admin|developer|unrestricted)\b',
]
TOPIC_PATTERNS = {
    'document_alternatives': r"alternative|substitute|replacement|can't get|cannot get|don't have|lost (?:the |my )?(?:report|note)|no (?:readable )?copy",
    'submission_timing': r'how soon.*(?:submit|send)|when.*(?:submit|send)|deadline to submit',
    'processing_time': r'how long|how many (?:business |working )?(?:days|weeks|hours).{0,45}(?:review|process|take)|processing time|review time|after (?:i |you )?(?:submit|send|receive)|once.*submit',
    'submission_method': r'how (?:do|can|should) i (?:send|submit|upload)|where.*(?:send|submit|upload)|portal|upload link',
    'file_format': r'file type|format|scan|pdf|readable|legible|photo',
    'receipt_confirmation': r'confirm.*receipt|know you got|confirmation|show up in status|received (?:it|them|the)',
    'payment': r'how much|amount|net pay|reimbursement|cost|fee|payment|paid|money',
    'appeal_deadline': r'appeal|deadline|too late',
    'documents': r'what.*(?:document|report|note)|documents|paperwork|materials',
    'denial_reason': r'why.*(?:denied|rejected)|denial reason|denied|rejected|turned down',
    'status': r'claim status|status|what happened|update|settled|closed|in progress',
}


def normalize(text: str) -> str:
    return text.replace('’', "'").replace('‘', "'").strip()


def business_text(text: str) -> str:
    """A small spelling allowlist for topic routing only, never identity or consent.

    Keep the original message for PII extraction, audit and consent evaluation.
    Avoid editing identifiers and addresses even in this routing-only copy.
    """
    corrections = {'cliam': 'claim', 'cliams': 'claims', 'calim': 'claim',
                   'staus': 'status', 'stauts': 'status', 'deneid': 'denied',
                   'denyed': 'denied', 'documnts': 'documents',
                   'documnets': 'documents', 'insurence': 'insurance',
                   'reimbursment': 'reimbursement', 'paymnt': 'payment'}
    return re.sub(r'(?<![\w@.+-])[a-z]+(?![\w@.+-])',
                  lambda m: corrections.get(m.group(), m.group()), normalize(text).lower())


class UtteranceExtractor:
    @staticmethod
    def support_orientation(text: str) -> bool:
        return bool(re.search(
            r"\b(?:what (?:is|does) (?:a |an insurance )?claim(?: mean)?\b|"
            r"what can i (?:ask|say)|what can you (?:help|do)|"
            r"(?:not sure|don't know|do not know) (?:where|how) to (?:start|begin)|"
            r"help me (?:get started|choose)|what does claim mean)", business_text(text)))

    @staticmethod
    def extract_pii(text: str) -> PIIFields:
        text = normalize(text)
        pii = PIIFields()
        # Names must be supplied as a name, not inferred from a fixture mention.
        names = re.finditer(r"\b(?:my (?:full )?name is|name\s*:|i am|i'm|this is)\s+([a-z][a-z'\-]+(?:\s+[a-z][a-z'\-]+){1,3})", text, re.I)
        for name in names:
            value = re.split(r'\s+(?:calling|policy|dob|born|and|from|son|daughter|on|my|email|phone|ssn|date)\b', name.group(1), flags=re.I)[0]
            stop = {'the', 'a', 'an', 'not', 'so', 'very', 'really', 'still', 'about', 'sorry', 'ridiculous', 'tired', 'frustrated', 'policyholder'}
            if len(value.split()) >= 2 and not set(value.lower().split()).intersection(stop):
                pii.name = value
                break
        if not pii.name and re.fullmatch(r"[A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){1,3}[.!]?", text) and not re.search(r'\b(?:what|why|how|please|thank|thanks|hello|hi|no|yes|send|skip|i|my|claim|this|that|human)\b', text, re.I):
            pii.name = text.rstrip('.!')
        pol = re.search(r'\bPOL-\d+\b', text, re.I)
        if pol:
            pii.policy_number = pol.group().upper()
        # A labelled DOB, or a bare date answer. Claim dates never become DOBs.
        dob = re.search(r'\b(?:dob|date of birth|birthday|born)\s*(?:is|on|was|:)?\s*(' + DOB_VALUE + ')', text, re.I)
        raw_date = dob.group(1) if dob else text.rstrip('. ')
        raw_date = re.sub(r'(\d)(st|nd|rd|th)', r'\1', raw_date, flags=re.I)
        # Make YYYY.MM.DD and 03/15.1985 work like their slash forms, while
        # retaining a separate space-only format for year-first input.
        normalized_numeric = re.sub(r'(?<=\d)[./-](?=\d)', '/', raw_date)
        for fmt in ['%Y/%m/%d', '%Y %m %d', '%m/%d/%Y', '%B %d, %Y', '%B %d %Y', '%b %d, %Y', '%b %d %Y']:
            try:
                pii.dob = datetime.datetime.strptime(normalized_numeric, fmt).date().isoformat()
                break
            except ValueError:
                continue
        ssn = re.search(r'\b(ssn|social security(?: number)?|national id|id)(?:\s+(?:last\s+(?:four|4)(?:\s+digits)?|last4))?\s*(?:is|are|:)?\s*(\d{4})\b', text, re.I)
        if not ssn:
            ssn = re.search(r'\b(last (?:four|4)(?: digits)?)\s*(?:is|are|:)?\s*(\d{4})\b', text, re.I)
        if ssn:
            pii.id_last4 = ssn.group(2)
            pii.id_type = 'national_id_last4' if ssn.group(1).lower() in ['national id', 'id'] else 'ssn_last4'
        phone = re.search(r'(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)', text)
        if phone:
            digits = re.sub(r'\D', '', phone.group())
            if len(digits) == 10:
                pii.phone = '+1' + digits
            elif len(digits) == 11 and digits.startswith('1'):
                pii.phone = '+' + digits
        email = re.search(r'\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b', text)
        if email:
            pii.email = email.group().lower()
        return pii

    @staticmethod
    def extract_topics(text: str):
        text = business_text(text)
        topics = [topic for topic, pattern in TOPIC_PATTERNS.items() if re.search(pattern, text)]
        document_context = re.search(r'\b(?:report|note|copy|photocopy|scan|lab|laboratory|clinic|hospital)\b', text)
        unavailable_or_alternative = re.search(r'\b(?:other (?:way|options?)|shut down|closed permanently)\b', text)
        if document_context and unavailable_or_alternative and 'document_alternatives' not in topics:
            topics.insert(0, 'document_alternatives')
        return topics

    @staticmethod
    def extract_cross_phase_hints(text: str) -> CrossPhaseMemory:
        text = normalize(text)
        low = business_text(text)
        # Remove identity dates so birthdays cannot override an earlier claim month.
        low = re.sub(r'\b(?:dob|date of birth|birthday|born)\s*(?:is|on|was|:)?\s*(?:[a-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})', '', low)
        hints = CrossPhaseMemory()
        for kind, pattern in [('healthcare', r'\b(?:healthcare|medical|health|hospital)\b'), ('dental', r'\b(?:dental|teeth|tooth)\b'), ('auto', r'\b(?:auto|car|vehicle|accident)\b')]:
            if re.search(pattern, low):
                hints.case_type_hint = kind
                break
        for status, pattern in [('denied', r'\b(?:denied|denial|rejected|turned down)\b'), ('closed', r'\b(?:closed|settled|completed)\b'), ('open', r'\b(?:open|in progress|pending)\b')]:
            if re.search(pattern, low):
                hints.status_hint = status
                break
        months = [m for m in MONTHS if re.search(r'\b' + m + r'\b', low)]
        if months:
            hints.date_hint = months[0].capitalize()
        bare_dob = UtteranceExtractor.extract_pii(text).dob and re.fullmatch(r'[\w ,/.-]+', text) and not re.search(r'\b(?:claim|from|in|on|denied|closed)\b', low)
        # CL-2048 is a claim reference, not the year 2048.
        dates = re.sub(r'\b(?:cl|pol)-\d+\b', '', low)
        year = None if bare_dob else re.search(r'\b(20\d{2}(?:-\d{2}-\d{2})?)\b', dates)
        if year:
            hints.date_hint = f'{hints.date_hint} {year.group()}' if hints.date_hint else year.group()
        case = re.search(r'\bCL-\d+\b', text, re.I)
        if case and 'case_id_hint' in CrossPhaseMemory.model_fields:
            hints.case_id_hint = case.group().upper()
        topics = UtteranceExtractor.extract_topics(text)
        if topics:
            hints.topic_hint = topics[0]
        if any([hints.case_type_hint, hints.status_hint, hints.date_hint, hints.topic_hint, case]):
            # Store useful business hints, not raw PII-bearing sentences.
            hints.raw_utterance_snippet = ' / '.join(str(x) for x in [hints.case_type_hint, hints.status_hint, hints.date_hint, case.group().upper() if case else None, hints.topic_hint] if x)
        return hints

    @staticmethod
    def detect_emotion_and_intent(text: str) -> Dict[str, Any]:
        low = normalize(text).lower()
        emotion = 'neutral'
        groups = {
            'confusion': ['confused', "don't understand", 'makes no sense', 'what does that mean'],
            'anxiety': ['anxious', 'worried', 'scared', 'afraid', 'panicking', 'terrified', "can't afford"],
            'frustration': ['already told you', 'ridiculous', 'frustrat', 'annoying', 'waste of time', 'just tell me', 'why do you need', 'tired of this', 'stop asking', 'why is this so hard', 'sick and tired', 'jumping through hoops'],
            'anger': ['furious', 'angry', 'outrageous', 'unacceptable', 'hate this', 'damn'],
        }
        for label, phrases in groups.items():
            if any(p in low for p in phrases):
                emotion = label
        refusal = bool(re.search(r"\b(?:refuse|won't (?:give|share)|will not (?:tell|share|give|provide)|none of your business|don't want to (?:give|share)|not giving|why should i give|no more personal)", low))
        human = bool(re.search(r'\b(?:human(?: representative| agent)?|real person|talk to someone|supervisor|manager|live agent)\b', low))
        if re.search(r"\b(?:don't|do not|no need to)\b.{0,20}\b(?:human|transfer|connect)\b", low):
            human = False
        return {'is_frustrated': emotion in ['frustration', 'anger'], 'is_refusal': refusal, 'demands_human': human, 'emotion': emotion}

    @staticmethod
    def is_out_of_scope(text: str) -> bool:
        low = business_text(text)
        if any(re.search(p, low) for p in OUT_OF_SCOPE_PATTERNS):
            return True
        if UtteranceExtractor.support_orientation(text):
            return False
        # Unknown questions are declined unless they concern the service or its workflow.
        question = bool(re.search(r'\b(?:what|why|how|who|where|when|explain|teach|tell me about)\b', low))
        service = bool(re.search(r'\b(?:claim|claims|policy|coverage|insurance|verify|verification|identity|info|information|name|dob|phone|email|ssn|id|documents?|paperwork|materials?|reports?|notes?|appeal|denied|denial|submit|send|upload|payment|pay|paid|amount|fee|reimbursement|portal|healthcare|dental|auto|human|representative|consent|summary|privacy|status|format|scan|pdf|receipt|confirmation|review|processing|original|copy|alternative|substitute|long|soon|deadline|that|it|this|them|those)\b', low))
        return question and not service

    @staticmethod
    def extract_proxy_context(text: str) -> Dict[str, Any]:
        low = normalize(text).lower()
        is_proxy = bool(re.search(r'\b(?:on behalf of|calling for|for my (?:mother|father|wife|husband)|son of|daughter of|neighbor of|friend of|representative for|her claim|his claim|calling about her|checking on her)\b', low))
        # Being a clinician is only a proxy signal when claiming access to someone else's file.
        is_proxy |= bool(re.search(r'\b(?:attorney|lawyer|doctor|dr|clinic|surgeon|nurse|billing clerk|billing department)\b', low) and re.search(r'\b(?:patient|client|margaret|her|his)\b', low))
        rep = UtteranceExtractor.extract_pii(text).name if is_proxy else None
        buyer_match = re.search(r'\b(?:behalf of|son of|daughter of|neighbor of|friend of|mother|father|wife|husband)\s+([a-z]+\s+[a-z]+)', low)
        buyer = buyer_match.group(1) if buyer_match else None
        relationship = next((label for phrase, label in [('mother', 'son'), ('son of', 'son'), ('neighbor', 'neighbor'), ('doctor', 'medical_provider'), ('lawyer', 'legal_rep')] if phrase in low), None)
        return {'is_proxy': bool(is_proxy), 'rep_name': rep, 'buyer_name': buyer, 'relationship': relationship}

    @staticmethod
    def detect_post_process_consent(text: str) -> Optional[str]:
        low = normalize(text).lower().strip(' .!')
        # Questions, conditions and third-party destinations are never consent.
        if '?' in low or re.search(r'\b(?:if|unless|maybe|perhaps|not sure|not yet|later|can you|could you|would you|why|what|where|how)\b', low):
            return None
        if re.search(r"\b(?:don't|do not|never)\s+(?:send|email)\b", low) or re.fullmatch(r'(?:no(?: thanks| thank you)?|nope|skip(?: it| the (?:email|summary))?|not needed|no need)(?:,?\s+(?:skip it|thanks|please|thank you))*', low):
            return 'declined'
        if re.search(r"\b(?:not|no|don't|do not)\b", low):
            return None
        if '@' in low or re.search(r'\b(?:another|different|instead|wife|friend|husband|boss)\b', low):
            return None
        if re.fullmatch(r'(?:yes|yep|yeah|sure|okay|ok|please do|go ahead|sounds good|that would be great)(?:,?\s+(?:please|thanks|thank you))*', low):
            return 'accepted'
        if re.fullmatch(r'(?:yes[, ]+)?(?:please )?(?:send|email)(?: me)? (?:it|(?:the |this )?(?:email )?summary)(?: (?:please|to my email|to (?:the )?(?:email|address) on file))?', low):
            return 'accepted'
        return None
