from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class Phase(str, Enum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"
    ESCALATED = "ESCALATED"
    CONCLUDED = "CONCLUDED"

class AgentAction(str, Enum):
    """Named actions available to an assistant policy in this SOP environment.

    The action space is constrained per-phase via action_mask.  An action is
    *legal* only when its bit in the mask is True.  Illegal actions should be
    filtered before sampling in any policy implementation.
    """
    ACK_EMOTION             = "ACK_EMOTION"             # Acknowledge caller emotional state
    ASK_IDENTITY_FIELD      = "ASK_IDENTITY_FIELD"      # Request a missing PII field
    EXPLAIN_VERIFICATION_GATE = "EXPLAIN_VERIFICATION_GATE"  # Explain why identity is needed
    RESOLVE_INTENT          = "RESOLVE_INTENT"           # Confirm understanding of caller goal
    ASK_CLAIM_CLARIFICATION = "ASK_CLAIM_CLARIFICATION" # Ask for more detail on the claim
    ANSWER_GROUNDED         = "ANSWER_GROUNDED"          # Provide a grounded factual answer (requires verified)
    OFFER_EMAIL_SUMMARY     = "OFFER_EMAIL_SUMMARY"      # Offer to send an email summary
    SEND_EMAIL              = "SEND_EMAIL"               # Confirm and send the email summary
    ESCALATE_HUMAN          = "ESCALATE_HUMAN"           # Transfer to a human agent


# Canonical ordering used for action_mask vectors (index → AgentAction)
AGENT_ACTIONS: list[AgentAction] = list(AgentAction)
ACTION_SPACE_SIZE: int = len(AGENT_ACTIONS)


class PIIFields(BaseModel):
    name: Optional[str] = None
    policy_number: Optional[str] = None
    dob: Optional[str] = None
    id_type: Optional[str] = None  # ssn_last4 or national_id_last4
    id_last4: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

class PolicyHolder(BaseModel):
    party_id: str
    name: str
    policy_number: str
    dob: str
    id_type: str
    id_last4: str
    phone: str
    email: str
    name_aliases: List[str] = Field(default_factory=list)
    phone_aliases: List[str] = Field(default_factory=list)
    email_aliases: List[str] = Field(default_factory=list)

class ClaimRecord(BaseModel):
    case_id: str
    party_id: str
    case_type: str
    created_at: str
    status: str
    summary: str
    denial_reason: Optional[str] = None
    documents_needed: List[str] = Field(default_factory=list)
    appeal_deadline: Optional[str] = None
    expected_reimbursement_amount: Optional[str] = None
    allowed_max_amount: Optional[str] = None
    net_pay: Optional[str] = None
    net_fee: Optional[str] = None

class CrossPhaseMemory(BaseModel):
    case_id_hint: Optional[str] = None
    case_type_hint: Optional[str] = None
    status_hint: Optional[str] = None
    date_hint: Optional[str] = None
    topic_hint: Optional[str] = None
    raw_utterance_snippet: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
    hint_history: Dict[str, List[str]] = Field(default_factory=dict)

class TraceEvent(BaseModel):
    timestamp: str
    phase_before: Phase
    phase_after: Phase
    gate_evaluated: str
    gate_passed: bool
    details: str
    data_shield_active: bool

class PostProcessState(BaseModel):
    email_offered: bool = False
    draft_summary: Optional[str] = None
    user_decision: Optional[str] = None  # "pending", "accepted", "declined"
    sent_to: Optional[str] = None
    delivery_status: Optional[str] = None  # "simulated" or "skipped"; never actual delivery
    outbox_id: Optional[str] = None

class Representative(BaseModel):
    rep_name: str
    relationship: str
    buyer_name: str
    buyer_party_id: str

class StateSnapshot(BaseModel):
    turn_index: int
    user_message: str
    agent_reply: str
    phase: Phase
    state_dump: Dict[str, Any]
    timestamp: str

class SOPState(BaseModel):
    session_id: str
    phase: Phase = Phase.VERIFY_ID
    accumulated_pii: PIIFields = Field(default_factory=PIIFields)
    verified_party_id: Optional[str] = None
    identity_verified: bool = False
    verified_fields: List[str] = Field(default_factory=list)
    cross_phase_memory: CrossPhaseMemory = Field(default_factory=CrossPhaseMemory)
    active_case_id: Optional[str] = None
    out_of_scope_count: int = 0
    refusal_count: int = 0
    post_process: PostProcessState = Field(default_factory=PostProcessState)
    trace_log: List[TraceEvent] = Field(default_factory=list)
    discussion_topics: List[str] = Field(default_factory=list)
    mock_outbox: List[Dict[str, Any]] = Field(default_factory=list)
    resolution_status: Optional[str] = None
    pii_conflicts: List[str] = Field(default_factory=list)
    # Proxy / Authorized Representative fields
    is_proxy_caller: bool = False
    proxy_rep_name: Optional[str] = None
    proxy_relationship: Optional[str] = None
    proxy_consent_status: Optional[str] = None  # "pending", "approved", "timeout", "unauthorized"
    proxy_identity_verified: bool = False
    proxy_authorized_party_id: Optional[str] = None
    # Time-travel history snapshots
    history_snapshots: List[StateSnapshot] = Field(default_factory=list)

class Message(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    current_phase: Phase
    sop_state: SOPState
    trace: List[TraceEvent]
    active_case: Optional[ClaimRecord] = None
    verified_policyholder: Optional[PolicyHolder] = None
    proxy_rep: Optional[Representative] = None
