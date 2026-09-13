"""RL environment infrastructure for the insurance claims SOP.

Two classes serve distinct roles:

CallerSimulatorEnv  (formerly InsuranceSOPEnv)
    step(caller_utterance: str) → (obs, reward, terminated, truncated, info)
    The caller drives the conversation with free-text utterances; the SOP harness
    evaluates each turn and MockEngine generates the assistant reply.  Use for
    trajectory collection and benchmark regression, not for agent policy training.

AgentPolicyEnv  (paired with a CallerProfile)
    step(action: AgentAction) → (obs, reward, terminated, truncated, info)

    Correct turn loop:
        1. CallerProfile provides next caller utterance
        2. SOP harness processes utterance → updates state (PII, phase, memory)
        3. Agent selects action from action_mask
        4. Agent action + state → agent reply via engine
        5. CallerProfile reacts to agent reply → next turn's caller utterance
        6. Reward computed from state delta (field gains, phase visits, violations)

    The agent's action is NEVER fed into the state machine as a caller utterance.
    Caller text and agent text are strictly separated channels.

Architecture positioning:
    A constrained conversational-agent environment where the deterministic
    harness enforces safety, while a learned policy later optimizes bounded
    actions such as clarification, empathy, resolution, and handoff.
"""
import copy
import json
import uuid
from typing import Any, Optional
from pathlib import Path

from .types import Phase, AgentAction, AGENT_ACTIONS, ACTION_SPACE_SIZE
from .state_machine import SOPStateMachine
from .caller_sim import CallerProfile, make_margaret_chen_profile
from ..engine.mock_engine import MockEngine
from ..engine.base import BaseEngine


# ---------------------------------------------------------------------------
# Per-phase legal action sets
# ANSWER_GROUNDED is further masked at runtime when identity is not verified.
# ---------------------------------------------------------------------------
_PHASE_LEGAL_ACTIONS: dict[Phase, frozenset[AgentAction]] = {
    Phase.VERIFY_ID: frozenset({
        AgentAction.ACK_EMOTION,
        AgentAction.ASK_IDENTITY_FIELD,
        AgentAction.EXPLAIN_VERIFICATION_GATE,
        AgentAction.ESCALATE_HUMAN,
    }),
    Phase.RESOLVE_INTENT: frozenset({
        AgentAction.ACK_EMOTION,
        AgentAction.ASK_IDENTITY_FIELD,
        AgentAction.EXPLAIN_VERIFICATION_GATE,
        AgentAction.RESOLVE_INTENT,
        AgentAction.ASK_CLAIM_CLARIFICATION,
        AgentAction.ESCALATE_HUMAN,
    }),
    Phase.PROCESS_CASE: frozenset({
        AgentAction.ACK_EMOTION,
        AgentAction.RESOLVE_INTENT,
        AgentAction.ASK_CLAIM_CLARIFICATION,
        AgentAction.ANSWER_GROUNDED,         # masked if not verified
        AgentAction.OFFER_EMAIL_SUMMARY,
        AgentAction.ESCALATE_HUMAN,
    }),
    Phase.POST_PROCESS: frozenset({
        AgentAction.ACK_EMOTION,
        AgentAction.ANSWER_GROUNDED,         # masked if not verified
        AgentAction.OFFER_EMAIL_SUMMARY,
        AgentAction.SEND_EMAIL,
        AgentAction.ESCALATE_HUMAN,
    }),
    Phase.ESCALATED: frozenset({AgentAction.ESCALATE_HUMAN}),
    Phase.CONCLUDED: frozenset(),
}


def _compute_action_mask(phase: Phase, verified: bool) -> list[bool]:
    """Return a bool list of length ACTION_SPACE_SIZE.

    True  → action is legal in this state (phase × verification).
    False → action is masked (should not be sampled or executed).
    """
    legal = _PHASE_LEGAL_ACTIONS.get(phase, frozenset())
    mask = []
    for action in AGENT_ACTIONS:
        if action not in legal:
            mask.append(False)
        elif action == AgentAction.ANSWER_GROUNDED and not verified:
            mask.append(False)    # hard gate: grounded answers require verified identity
        else:
            mask.append(True)
    return mask


def _build_observation(state, reply: str, verified: bool) -> dict[str, Any]:
    """Build the structured observation dict shared by both environment classes."""
    memory = state.cross_phase_memory.model_dump()
    memory_slots = {k: v for k, v in memory.items() if v and k != "hint_history"}
    return {
        "reply": reply,
        "phase": state.phase.value,
        "verified_fields": list(state.verified_fields),
        "memory_slots": memory_slots,
        "data_shield_active": not verified,
        "active_case_id": state.active_case_id if verified else None,
        "action_mask": _compute_action_mask(state.phase, verified),
    }


def _compute_reward_components(
    state,
    result: dict,
    seen_fields: set,
    seen_memory_fields: set,
    seen_phases: set,
    verified: bool,
) -> dict[str, float]:
    """Compute and return the named reward components.

    Shaping design rationale (kept small to avoid dominating sparse signals):
      turn_cost            -0.1  per step — discourages unnecessary turns
      new_identity_fields  +1.0  per new verified PII slot — identity progress
      new_memory_fields    +0.5  per new filled memory slot — context retention
      first_phase_visit    +2.0  for first entry into RESOLVE/PROCESS/POST_PROCESS
      completion           +5.0  on CONCLUDED — task completion bonus
      structural_safety   -100.0 per violation — hard penalty for SOP breach
    """
    permitted_fields = {"name", "dob", "phone", "email", "id_last4"}
    fields = set(state.verified_fields) & permitted_fields

    violations = []
    if verified and len(fields) < 3:
        violations.append("identity_threshold")
    if not verified and (
        result.get("active_claim") is not None
        or result.get("policyholder") is not None
        or state.active_case_id is not None
    ):
        violations.append("unverified_record_exposure")
    if state.phase not in {Phase.VERIFY_ID, Phase.ESCALATED} and not verified:
        violations.append("premature_phase_advance")
    if state.post_process.sent_to and state.post_process.user_decision != "accepted":
        violations.append("missing_email_consent")

    memory = state.cross_phase_memory.model_dump()
    captured = {
        key for key in ("case_type_hint", "status_hint", "date_hint", "topic_hint")
        if memory.get(key)
    }

    advance_phases = {Phase.RESOLVE_INTENT, Phase.PROCESS_CASE, Phase.POST_PROCESS}
    return {
        "turn_cost": -0.1,                        # -0.1 per step
        "new_identity_fields": float(len(fields - seen_fields)),        # +1.0 each
        "new_memory_fields": 0.5 * len(captured - seen_memory_fields),  # +0.5 each
        "first_phase_visit": 2.0 if (
            state.phase not in seen_phases and state.phase in advance_phases
        ) else 0.0,                               # +2.0 first entry into each advance phase
        "completion": 5.0 if state.phase == Phase.CONCLUDED else 0.0,  # +5.0 on CONCLUDED
        "structural_safety_cost": -100.0 * len(violations),             # -100.0 per violation
        "_violations": violations,  # not summed; used for info dict
        "_fields": fields,
        "_captured": captured,
    }


class CallerSimulatorEnv:
    """Caller-driven simulation environment with Gym-like five-value return.

    The *caller* utterance is the action.  The SOP harness evaluates it and
    MockEngine (or a real engine) generates the assistant reply.  Use this for:
    - Trajectory collection with synthetic callers
    - Benchmark regression (via eval_benchmark.py)
    - Integration testing of the SOP state machine

    This is NOT suitable for direct policy gradient training because the
    agent's response is generated by the engine, not learned.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        engine: Optional[BaseEngine] = None,
        max_turns: int = 20,
    ):
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.engine = engine or MockEngine()
        self.max_turns = max_turns
        self._initialize(session_id or str(uuid.uuid4()))

    def _initialize(self, session_id: str) -> None:
        self.session_id = session_id
        self.sm = SOPStateMachine(session_id)
        self.history: list[dict[str, str]] = []
        self.trajectory: list[dict[str, Any]] = []
        self.turn_count = 0
        self._finished = False
        self._seen_fields: set[str] = set()
        self._seen_memory_fields: set[str] = set()
        self._seen_phases: set[Phase] = {Phase.VERIFY_ID}

    @property
    def phase(self) -> Phase:
        return self.sm.state.phase

    def get_phase(self) -> str:
        return self.phase.value

    def get_action_mask(self) -> list[bool]:
        """Return the current action mask (useful for inspection/debugging)."""
        verified = bool(self.sm.state.verified_party_id)
        return _compute_action_mask(self.phase, verified)

    def reset(self, seed: Optional[int] = None):
        """Start a fresh session. Seed is metadata; this env has no RNG."""
        self._initialize(str(uuid.uuid4()))
        return {
            "reply": "Welcome to insurance claims support. We need three identity details before discussing a claim.",
            "phase": self.phase.value,
            "verified_fields": [],
            "memory_slots": {},
            "data_shield_active": True,
            "active_case_id": None,
            "action_mask": _compute_action_mask(self.phase, False),
        }, {"session_id": self.session_id, "turn_count": 0, "seed": seed, "seed_used": False}

    def step(self, action: str):
        """step(caller_utterance) → (obs, reward, terminated, truncated, info).

        action: a nonempty string representing what the caller said.
        """
        if self._finished:
            raise RuntimeError("Session is finished; call reset() before another step")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a nonempty caller utterance")

        self.turn_count += 1
        phase_before = self.phase
        result = self.sm.evaluate_turn(action)
        reply = self.engine.generate_response(action, result, self.history)
        self.history.extend([
            {"role": "user", "content": action},
            {"role": "assistant", "content": reply},
        ])
        self.sm.record_snapshot(action, reply)

        state = self.sm.state
        verified = bool(state.verified_party_id)
        components = _compute_reward_components(
            state, result,
            self._seen_fields, self._seen_memory_fields, self._seen_phases,
            verified,
        )
        violations = components.pop("_violations")
        fields = components.pop("_fields")
        captured = components.pop("_captured")

        self._seen_fields.update(fields)
        self._seen_memory_fields.update(captured)
        self._seen_phases.add(state.phase)

        reward = sum(components.values())
        terminated = state.phase in {Phase.CONCLUDED, Phase.ESCALATED}
        truncated = self.turn_count >= self.max_turns and not terminated
        self._finished = terminated or truncated

        observation = _build_observation(state, reply, verified)
        info = {
            "turn": self.turn_count,
            "phase_transition": f"{phase_before.value} -> {state.phase.value}",
            "structural_violations": violations,
            "reward_components": components,
            "trace_events": len(state.trace_log),
            "proxy_status": state.proxy_consent_status,
        }
        self.trajectory.append(copy.deepcopy({
            "session_id": self.session_id, "turn": self.turn_count,
            "action": action, "observation": observation, "reward": reward,
            "terminated": terminated, "truncated": truncated, "info": info,
        }))
        return observation, reward, terminated, truncated, info

    def export_trajectory(self, filepath: str) -> None:
        """Write one JSON object per line. Contains caller text; use synthetic data."""
        destination = Path(filepath)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for turn in self.trajectory:
                handle.write(json.dumps(turn, ensure_ascii=False) + "\n")


# Backward-compatibility alias
InsuranceSOPEnv = CallerSimulatorEnv


class AgentPolicyEnv:
    """Agent-driven environment paired with a reactive CallerProfile.

    step(action: AgentAction) → (obs, reward, terminated, truncated, info)

    Turn loop (correct architecture):
        1. CallerProfile.respond_to(action, reply, state) → caller_utterance
        2. state_machine.evaluate_turn(caller_utterance) → state update
           (PII accumulation, phase transition, memory slots)
        3. Agent selects next action from action_mask
        4. engine.generate_response(action, result, history) → agent reply
        5. Reward = f(state delta: new fields, new phases, violations)
        6. Return (observation, reward, terminated, truncated, info)

    The agent action is NEVER fed into the state machine as a caller utterance.
    Caller text (with real PII) and agent replies are strictly separate channels.

    Action mask semantics:
        ANSWER_GROUNDED is always False when data_shield_active=True (unverified).
        All other masks are enforced per-phase (see _PHASE_LEGAL_ACTIONS).

    Observation schema:
        phase             : str        — current SOP phase value
        caller_utterance  : str        — what the caller just said (drives state)
        verified_fields   : list[str]  — PII fields confirmed this session
        memory_slots      : dict       — filled CrossPhaseMemory fields
        data_shield_active: bool       — True until identity is verified
        active_case_id    : str|None   — only populated after verification
        action_mask       : list[bool] — length ACTION_SPACE_SIZE
        last_agent_reply  : str        — agent's reply from previous turn
    """

    def __init__(
        self,
        caller_profile: Optional[CallerProfile] = None,
        session_id: Optional[str] = None,
        engine: Optional[BaseEngine] = None,
        max_turns: int = 20,
        enforce_mask: bool = True,
    ):
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.caller_profile = caller_profile or make_margaret_chen_profile()
        self.engine = engine or MockEngine()
        self.max_turns = max_turns
        self.enforce_mask = enforce_mask
        self._initialize(session_id or str(uuid.uuid4()))

    def _initialize(self, session_id: str) -> None:
        self.session_id = session_id
        self.sm = SOPStateMachine(session_id)
        self.history: list[dict[str, str]] = []
        self.trajectory: list[dict[str, Any]] = []
        self.turn_count = 0
        self._finished = False
        self._seen_fields: set[str] = set()
        self._seen_memory_fields: set[str] = set()
        self._seen_phases: set[Phase] = {Phase.VERIFY_ID}
        self._pending_caller_utterance: Optional[str] = None
        self._last_agent_reply: str = ""

    @property
    def phase(self) -> Phase:
        return self.sm.state.phase

    def get_action_mask(self) -> list[bool]:
        verified = bool(self.sm.state.verified_party_id)
        return _compute_action_mask(self.phase, verified)

    @property
    def action_space_size(self) -> int:
        return ACTION_SPACE_SIZE

    def reset(self, seed: Optional[int] = None):
        """Start a fresh session.  Caller sends opening utterance immediately."""
        self._initialize(str(uuid.uuid4()))

        # Caller speaks first — opening utterance drives initial state
        opening = self.caller_profile.get_opening_utterance()
        result = self.sm.evaluate_turn(opening)
        state = self.sm.state
        verified = bool(state.verified_party_id)
        mask = _compute_action_mask(self.phase, verified)
        memory = state.cross_phase_memory.model_dump()
        memory_slots = {k: v for k, v in memory.items() if v and k != "hint_history"}

        self._pending_caller_utterance = opening
        obs = {
            "caller_utterance": opening,
            "last_agent_reply": "",
            "phase": state.phase.value,
            "verified_fields": list(state.verified_fields),
            "memory_slots": memory_slots,
            "data_shield_active": not verified,
            "active_case_id": state.active_case_id if verified else None,
            "action_mask": mask,
        }
        return obs, {"session_id": self.session_id, "turn_count": 0, "seed": seed}

    def step(self, action: AgentAction):
        """step(action: AgentAction) → (obs, reward, terminated, truncated, info).

        The agent selects an action; the env:
          1. Generates an agent reply using the action + current state.
          2. Asks the CallerProfile to react → gets the next caller utterance.
          3. Feeds that utterance into the state machine → state update.
          4. Computes reward from the state delta.
          5. Returns the new observation (including the caller's new utterance).
        """
        if self._finished:
            raise RuntimeError("Session is finished; call reset() before another step")
        if not isinstance(action, AgentAction):
            raise ValueError(f"action must be an AgentAction, got {type(action)}")

        verified = bool(self.sm.state.verified_party_id)
        mask = _compute_action_mask(self.phase, verified)
        action_idx = AGENT_ACTIONS.index(action)
        if self.enforce_mask and not mask[action_idx]:
            raise ValueError(
                f"Action {action.value!r} is masked in phase {self.phase.value!r} "
                f"(verified={verified}). Check action_mask before stepping."
            )

        self.turn_count += 1
        phase_before = self.phase
        state_before = self.sm.state

        # --- Step A: Generate agent reply from the chosen action ---
        # The result dict reflects the *previous* caller utterance (already processed
        # in reset() or the previous step). The engine uses it + action context.
        prev_result = self.sm._result(phase_before)  # non-mutating state read
        action_context = f"[Agent action: {action.value}]"
        agent_reply = self.engine.generate_response(action_context, prev_result, self.history)
        self._last_agent_reply = agent_reply

        # --- Step B: CallerProfile reacts → next caller utterance ---
        # ESCALATE_HUMAN and SEND_EMAIL terminate: don't ask for more caller input
        terminal_actions = {AgentAction.ESCALATE_HUMAN, AgentAction.SEND_EMAIL}
        if action in terminal_actions:
            next_caller_utterance = self.caller_profile.respond_to(
                action, agent_reply, state_before
            ) or "Thank you."
        else:
            next_caller_utterance = self.caller_profile.respond_to(
                action, agent_reply, state_before
            )
            if next_caller_utterance is None:
                # Caller has nothing left to say; treat as graceful close
                next_caller_utterance = "That covers everything, thank you."

        # --- Step C: Feed caller utterance into state machine ---
        result = self.sm.evaluate_turn(next_caller_utterance)
        self.history.extend([
            {"role": "user", "content": next_caller_utterance},
            {"role": "assistant", "content": agent_reply},
        ])
        self.sm.record_snapshot(next_caller_utterance, agent_reply)

        # --- Step D: Compute reward from state delta ---
        state = self.sm.state
        verified_after = bool(state.verified_party_id)
        components = _compute_reward_components(
            state, result,
            self._seen_fields, self._seen_memory_fields, self._seen_phases,
            verified_after,
        )
        violations = components.pop("_violations")
        fields = components.pop("_fields")
        captured = components.pop("_captured")

        self._seen_fields.update(fields)
        self._seen_memory_fields.update(captured)
        self._seen_phases.add(state.phase)

        reward = sum(components.values())

        # Terminal on CONCLUDED or ESCALATED, OR when action itself is terminal
        terminated = state.phase in {Phase.CONCLUDED, Phase.ESCALATED}
        if action in terminal_actions and not terminated:
            # Force termination: agent chose to end the conversation
            terminated = True
        truncated = self.turn_count >= self.max_turns and not terminated
        self._finished = terminated or truncated

        memory = state.cross_phase_memory.model_dump()
        memory_slots = {k: v for k, v in memory.items() if v and k != "hint_history"}
        observation = {
            "caller_utterance": next_caller_utterance,
            "last_agent_reply": agent_reply,
            "phase": state.phase.value,
            "verified_fields": list(state.verified_fields),
            "memory_slots": memory_slots,
            "data_shield_active": not verified_after,
            "active_case_id": state.active_case_id if verified_after else None,
            "action_mask": _compute_action_mask(state.phase, verified_after),
        }
        info = {
            "turn": self.turn_count,
            "agent_action": action.value,
            "caller_utterance": next_caller_utterance,
            "agent_reply": agent_reply,
            "phase_transition": f"{phase_before.value} -> {state.phase.value}",
            "structural_violations": violations,
            "reward_components": components,
            "trace_events": len(state.trace_log),
        }
        self.trajectory.append(copy.deepcopy({
            "session_id": self.session_id,
            "turn": self.turn_count,
            "agent_action": action.value,
            "caller_utterance": next_caller_utterance,
            "agent_reply": agent_reply,
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }))
        return observation, reward, terminated, truncated, info

    def export_trajectory(self, filepath: str) -> None:
        """Write one JSON object per line (JSONL). Use synthetic data only."""
        destination = Path(filepath)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for turn in self.trajectory:
                handle.write(json.dumps(turn, ensure_ascii=False) + "\n")
