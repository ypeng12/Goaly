"""RL environment infrastructure for the insurance claims SOP.

Two classes serve distinct roles:

CallerSimulatorEnv  (formerly InsuranceSOPEnv)
    step(caller_utterance: str) → (obs, reward, terminated, truncated, info)
    The caller drives the conversation; the SOP harness evaluates each turn and
    MockEngine generates the assistant reply.  This is a *caller simulator*, not
    a trained assistant policy.  Use it for trajectory collection and benchmark
    regression, not for direct PPO/GRPO training.

AgentPolicyEnv
    step(action: AgentAction) → (obs, reward, terminated, truncated, info)
    The *agent* drives the conversation by selecting a named action from a
    constrained action space.  Each phase exposes a different action_mask so
    that illegal actions (e.g. ANSWER_GROUNDED before identity is verified)
    are structurally blocked.  A RuleBasedPolicy baseline is available in
    rl_baseline.py for comparison with random or LLM policies.

Both classes share the same reward shaping logic and five-value Gym-like return
signature.  Rewards are illustrative shaping scores only; they are not
calibrated customer outcomes.  Use the separate benchmark (eval_benchmark.py)
for explicit, separately-reported SOP assertions.

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
    """Agent-driven environment where the *assistant policy* chooses named actions.

    step(action: AgentAction) → (obs, reward, terminated, truncated, info)

    The action_mask in each observation indicates which of the ACTION_SPACE_SIZE
    actions are currently legal.  An agent must respect the mask; attempting an
    illegal action raises ValueError.

    Observation schema:
        phase             : str   — current SOP phase
        verified_fields   : list  — PII fields confirmed so far
        memory_slots      : dict  — filled CrossPhaseMemory fields
        data_shield_active: bool  — True until identity is verified
        active_case_id    : str|None — only populated after verification
        action_mask       : list[bool] — length ACTION_SPACE_SIZE
        reply             : str   — assistant utterance generated for this action

    This environment is suitable for:
    - Defining a formal action space for post-training (PPO/GRPO)
    - Running a RuleBasedPolicy baseline (see rl_baseline.py)
    - Comparing random / rule-based / LLM policies on cumulative reward
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        engine: Optional[BaseEngine] = None,
        max_turns: int = 20,
        enforce_mask: bool = True,
    ):
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
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
        """Start a fresh session."""
        self._initialize(str(uuid.uuid4()))
        mask = _compute_action_mask(self.phase, False)
        return {
            "reply": "Welcome to insurance claims support. We need three identity details before discussing a claim.",
            "phase": self.phase.value,
            "verified_fields": [],
            "memory_slots": {},
            "data_shield_active": True,
            "active_case_id": None,
            "action_mask": mask,
        }, {"session_id": self.session_id, "turn_count": 0, "seed": seed, "seed_used": False}

    def step(self, action: AgentAction):
        """step(action: AgentAction) → (obs, reward, terminated, truncated, info).

        Raises ValueError if action is masked and enforce_mask=True (default).
        The env translates the agent action into a synthetic caller utterance
        to drive the SOP state machine, then generates an assistant reply.
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

        # Translate agent action to a canonical caller-side trigger utterance.
        # This keeps the SOP state machine (which operates on caller text) intact.
        synthetic_utterance = _action_to_utterance(action, self.sm.state)
        result = self.sm.evaluate_turn(synthetic_utterance)

        # Build a structured system prompt for the engine that names the chosen action.
        context_hint = f"[Agent selected action: {action.value}]"
        reply = self.engine.generate_response(context_hint, result, self.history)
        self.history.extend([
            {"role": "user", "content": synthetic_utterance},
            {"role": "assistant", "content": reply},
        ])
        self.sm.record_snapshot(synthetic_utterance, reply)

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
        terminated = state.phase in {Phase.CONCLUDED, Phase.ESCALATED}
        truncated = self.turn_count >= self.max_turns and not terminated
        self._finished = terminated or truncated

        observation = _build_observation(state, reply, verified_after)
        info = {
            "turn": self.turn_count,
            "agent_action": action.value,
            "phase_transition": f"{phase_before.value} -> {state.phase.value}",
            "structural_violations": violations,
            "reward_components": components,
            "trace_events": len(state.trace_log),
        }
        self.trajectory.append(copy.deepcopy({
            "session_id": self.session_id, "turn": self.turn_count,
            "agent_action": action.value, "synthetic_utterance": synthetic_utterance,
            "observation": observation, "reward": reward,
            "terminated": terminated, "truncated": truncated, "info": info,
        }))
        return observation, reward, terminated, truncated, info

    def export_trajectory(self, filepath: str) -> None:
        """Write one JSON object per line (JSONL). Use synthetic data only."""
        destination = Path(filepath)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for turn in self.trajectory:
                handle.write(json.dumps(turn, ensure_ascii=False) + "\n")


def _action_to_utterance(action: AgentAction, state) -> str:
    """Map an agent action to a synthetic caller utterance that drives the SOP.

    These utterances are minimal triggers — enough to advance the state machine
    without leaking real PII.  In a real training setup the caller side would
    come from a separate caller simulator (CallerSimulatorEnv) or a dataset.
    """
    missing = [
        f for f in ("name", "dob", "phone", "email", "id_last4")
        if f not in (state.verified_fields or [])
    ]
    mapping = {
        AgentAction.ACK_EMOTION: "I understand, this must be frustrating.",
        AgentAction.ASK_IDENTITY_FIELD: (
            f"Could you please provide your {missing[0]}?" if missing
            else "Could you confirm your policy number?"
        ),
        AgentAction.EXPLAIN_VERIFICATION_GATE: (
            "I need to verify your identity before I can access any claim details."
        ),
        AgentAction.RESOLVE_INTENT: "I'd like to understand what you need help with today.",
        AgentAction.ASK_CLAIM_CLARIFICATION: "Can you give me more details about your claim?",
        AgentAction.ANSWER_GROUNDED: "Here is the information from your file.",
        AgentAction.OFFER_EMAIL_SUMMARY: "Would you like me to send a summary to your email?",
        AgentAction.SEND_EMAIL: "Please send the summary to my email.",
        AgentAction.ESCALATE_HUMAN: "I'd like to speak with a human agent please.",
    }
    return mapping[action]
