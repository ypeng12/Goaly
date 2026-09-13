"""RL environment infrastructure for the insurance claims SOP.

Implements Gymnasium-standard interface with formal action masking,
deterministic state-machine gating, and multi-profile caller simulation.

Two classes:
- CallerSimulatorEnv: drives turns by caller utterance text (for evaluation/benchmarks).
- AgentPolicyEnv: Gymnasium environment driven by AgentAction (for RL policy learning).
"""
import copy
import json
import uuid
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .types import Phase, AgentAction, AGENT_ACTIONS, ACTION_SPACE_SIZE
from .state_machine import SOPStateMachine
from .caller_sim import (
    CallerProfile,
    make_margaret_chen_profile,
    make_all_profiles,
    make_train_profiles,
)
from ..engine.mock_engine import MockEngine
from ..engine.base import BaseEngine


# ---------------------------------------------------------------------------
# Per-phase legal action sets
# ANSWER_GROUNDED is masked when identity is unverified.
# SEND_EMAIL is masked when post_process.user_decision != 'accepted'.
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
        AgentAction.ANSWER_GROUNDED,         # masked if unverified
        AgentAction.OFFER_EMAIL_SUMMARY,
        AgentAction.ESCALATE_HUMAN,
    }),
    Phase.POST_PROCESS: frozenset({
        AgentAction.ACK_EMOTION,
        AgentAction.OFFER_EMAIL_SUMMARY,
        AgentAction.SEND_EMAIL,              # masked if consent != 'accepted'
        AgentAction.ESCALATE_HUMAN,
    }),
    Phase.ESCALATED: frozenset({AgentAction.ESCALATE_HUMAN}),
    Phase.CONCLUDED: frozenset(),
}


def _compute_action_mask(
    phase: Phase,
    verified: bool,
    post_process_accepted: bool = False,
) -> list[bool]:
    """Return a bool list of length ACTION_SPACE_SIZE."""
    legal = _PHASE_LEGAL_ACTIONS.get(phase, frozenset())
    mask = []
    for action in AGENT_ACTIONS:
        if action not in legal:
            mask.append(False)
        elif action == AgentAction.ANSWER_GROUNDED and not verified:
            mask.append(False)
        elif action == AgentAction.SEND_EMAIL and not post_process_accepted:
            mask.append(False)
        else:
            mask.append(True)
    return mask


def _build_observation(state, reply: str, verified: bool, caller_msg: str = "") -> dict[str, Any]:
    memory = state.cross_phase_memory.model_dump()
    memory_slots = {k: v for k, v in memory.items() if v and k != "hint_history"}
    post_accepted = getattr(state.post_process, "user_decision", None) == "accepted"
    return {
        "caller_utterance": caller_msg,
        "last_agent_reply": reply,
        "reply": reply,
        "phase": state.phase.value,
        "verified_fields": list(state.verified_fields),
        "memory_slots": memory_slots,
        "data_shield_active": not verified,
        "active_case_id": state.active_case_id if verified else None,
        "action_mask": _compute_action_mask(state.phase, verified, post_accepted),
    }


class CallerSimulatorEnv:
    """Caller-driven simulation environment with Gym-like five-value return."""

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
        verified = bool(self.sm.state.verified_party_id)
        post_accepted = self.sm.state.post_process.user_decision == "accepted"
        return _compute_action_mask(self.phase, verified, post_accepted)

    def reset(self, seed: Optional[int] = None):
        self._initialize(str(uuid.uuid4()))
        obs = _build_observation(
            self.sm.state,
            reply="Welcome to insurance claims support. We need three identity details before discussing a claim.",
            verified=False,
            caller_msg="",
        )
        return obs, {"session_id": self.session_id, "turn_count": 0, "seed": seed}

    def step(self, action: str):
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

        # Compute violations
        violations = []
        permitted_fields = {"name", "dob", "phone", "email", "id_last4"}
        fields = set(state.verified_fields) & permitted_fields
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

        reward = -0.1 + float(len(fields - self._seen_fields))
        self._seen_fields.update(fields)

        terminated = state.phase in {Phase.CONCLUDED, Phase.ESCALATED}
        truncated = self.turn_count >= self.max_turns and not terminated
        self._finished = terminated or truncated

        reward_components = {
            "turn_cost": -0.1,
            "new_identity_fields": float(len(fields - self._seen_fields)),
            "structural_violations": -100.0 * len(violations),
        }

        observation = _build_observation(state, reply, verified, caller_msg=action)
        info = {
            "turn": self.turn_count,
            "phase_transition": f"{phase_before.value} -> {state.phase.value}",
            "structural_violations": violations,
            "reward_components": reward_components,
            "task_success": state.phase == Phase.CONCLUDED,
            "constraint_violation": bool(violations),
        }
        self.trajectory.append({
            "session_id": self.session_id, "turn": self.turn_count,
            "action": action, "observation": observation, "reward": reward,
            "terminated": terminated, "truncated": truncated, "info": info,
        })
        return observation, reward, terminated, truncated, info

    def export_trajectory(self, filepath: str) -> None:
        """Write one JSON object per line (JSONL)."""
        dest = Path(filepath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as handle:
            for turn in self.trajectory:
                handle.write(json.dumps(turn, ensure_ascii=False) + "\n")


# Backward-compatibility alias
InsuranceSOPEnv = CallerSimulatorEnv


class AgentPolicyEnv(gym.Env):
    """Gymnasium-compliant Agent-driven environment paired with a reactive CallerProfile.

    Follows strict turn execution sequence:
      1. Action mask verification (masked actions cannot be chosen).
      2. state_before = copy.deepcopy(sm.state).
      3. Render agent reply based on action and state_before.
      4. If action == ESCALATE_HUMAN: state.phase = ESCALATED, terminal.
      5. If action == SEND_EMAIL: check consent == 'accepted', state.phase = CONCLUDED, terminal.
      6. Caller reacts to (action, agent_reply, state_before) -> next_caller_message.
      7. sm.evaluate_turn(next_caller_message) updates state.
      8. Calculate rewards, enforcing anti-reward hacking and structural safety.
      9. terminated is True ONLY when state.phase in {CONCLUDED, ESCALATED}.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        caller_profile: Optional[CallerProfile] = None,
        session_id: Optional[str] = None,
        engine: Optional[BaseEngine] = None,
        max_turns: int = 20,
        enforce_mask: bool = True,
    ):
        super().__init__()
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns
        self.enforce_mask = enforce_mask
        self.engine = engine or MockEngine()
        self.caller_profile = caller_profile or make_margaret_chen_profile()

        # Gymnasium Spaces
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Dict({
            "phase": spaces.Text(max_length=32),
            "caller_utterance": spaces.Text(max_length=2000),
            "last_agent_reply": spaces.Text(max_length=2000),
            "verified_fields": spaces.Sequence(spaces.Text(max_length=32)),
            "memory_slots": spaces.Dict({
                "case_type_hint": spaces.Text(max_length=64),
                "status_hint": spaces.Text(max_length=64),
                "date_hint": spaces.Text(max_length=64),
                "topic_hint": spaces.Text(max_length=64),
            }),
            "data_shield_active": spaces.Discrete(2),
            "active_case_id": spaces.Text(max_length=32),
            "action_mask": spaces.MultiBinary(ACTION_SPACE_SIZE),
        })

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
        self._pending_caller_utterance: str = ""
        self._last_agent_reply: str = ""
        self._last_caller_utterance: str = ""
        self._has_answered_grounded: bool = False

    @property
    def phase(self) -> Phase:
        return self.sm.state.phase

    def get_action_mask(self) -> list[bool]:
        verified = bool(self.sm.state.verified_party_id)
        post_accepted = self.sm.state.post_process.user_decision == "accepted"
        return _compute_action_mask(self.phase, verified, post_accepted)

    @property
    def action_space_size(self) -> int:
        return ACTION_SPACE_SIZE

    def save_state(self) -> Dict[str, Any]:
        """Snapshot exact environment state for counterfactual branching."""
        return {
            "session_id": self.session_id,
            "sm_state": copy.deepcopy(self.sm.state),
            "history": copy.deepcopy(self.history),
            "turn_count": self.turn_count,
            "finished": self._finished,
            "seen_fields": set(self._seen_fields),
            "seen_memory_fields": set(self._seen_memory_fields),
            "seen_phases": set(self._seen_phases),
            "pending_caller_utterance": self._pending_caller_utterance,
            "last_agent_reply": self._last_agent_reply,
            "last_caller_utterance": self._last_caller_utterance,
            "has_answered_grounded": self._has_answered_grounded,
            "caller_state": self.caller_profile.get_state(),
        }

    def restore_state(self, snapshot: Dict[str, Any]) -> None:
        """Restore exact environment state from a snapshot."""
        self.session_id = snapshot["session_id"]
        self.sm.state = copy.deepcopy(snapshot["sm_state"])
        self.history = copy.deepcopy(snapshot["history"])
        self.turn_count = snapshot["turn_count"]
        self._finished = snapshot["finished"]
        self._seen_fields = set(snapshot["seen_fields"])
        self._seen_memory_fields = set(snapshot["seen_memory_fields"])
        self._seen_phases = set(snapshot["seen_phases"])
        self._pending_caller_utterance = snapshot["pending_caller_utterance"]
        self._last_agent_reply = snapshot["last_agent_reply"]
        self._last_caller_utterance = snapshot["last_caller_utterance"]
        self._has_answered_grounded = snapshot.get("has_answered_grounded", False)
        self.caller_profile.set_state(snapshot["caller_state"])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Start a fresh session. Caller sends opening utterance immediately."""
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            import random
            random.seed(seed)
            np.random.seed(seed)

        self._initialize(str(uuid.uuid4()))

        # Caller provides opening utterance
        opening = self.caller_profile.get_opening_utterance()
        result = self.sm.evaluate_turn(opening)
        state = self.sm.state
        verified = bool(state.verified_party_id)
        post_accepted = state.post_process.user_decision == "accepted"
        mask = _compute_action_mask(self.phase, verified, post_accepted)

        self._pending_caller_utterance = opening
        self._last_caller_utterance = opening

        # Track initial fields from opening
        permitted_fields = {"name", "dob", "phone", "email", "id_last4"}
        self._seen_fields.update(set(state.verified_fields) & permitted_fields)
        memory = state.cross_phase_memory.model_dump()
        self._seen_memory_fields.update(
            k for k in ("case_type_hint", "status_hint", "date_hint", "topic_hint")
            if memory.get(k)
        )

        obs = _build_observation(state, reply="", verified=verified, caller_msg=opening)
        info = {
            "session_id": self.session_id,
            "turn_count": 0,
            "seed": seed,
        }
        return obs, info

    def step(self, action: AgentAction):
        """Execute one agent action following strict timing & safety invariants."""
        if self._finished:
            raise RuntimeError("Session is finished; call reset() before another step")
        if not isinstance(action, AgentAction):
            raise ValueError(f"action must be an AgentAction, got {type(action)}")

        verified_before = bool(self.sm.state.verified_party_id)
        post_accepted_before = self.sm.state.post_process.user_decision == "accepted"
        mask = _compute_action_mask(self.phase, verified_before, post_accepted_before)
        action_idx = AGENT_ACTIONS.index(action)

        if self.enforce_mask and not mask[action_idx]:
            raise ValueError(
                f"Action {action.value!r} is masked in phase {self.phase.value!r} "
                f"(verified={verified_before}, consent={post_accepted_before})."
            )

        self.turn_count += 1
        state_before = copy.deepcopy(self.sm.state)
        phase_before = state_before.phase

        # Step 1: Render agent reply based on action and state_before
        prev_result = self.sm._result(phase_before)
        action_context = f"[Agent action: {action.value}]"
        agent_reply = self.engine.generate_response(action_context, prev_result, self.history)
        self._last_agent_reply = agent_reply

        # Determine terminal behaviors
        if action == AgentAction.ESCALATE_HUMAN:
            # 1. Trace log
            self.sm.state.trace_log.append("agent_escalation_to_human")
            # 2. Transition phase to ESCALATED
            self.sm.state.phase = Phase.ESCALATED
            # 3. Terminated = True, Truncated = False
            terminated = True
            truncated = False
            self._finished = True

            # Evaluate whether escalation was appropriate
            # Appropriate if: caller demanded human, or style is escalating after threshold, or 5+ turns stuck
            caller_demanded = any(
                kw in self._last_caller_utterance.lower()
                for kw in ["manager", "human", "supervisor", "representative", "ridiculous", "enough"]
            ) or (self.caller_profile.style == "escalating" and self.turn_count >= 3) or (self.turn_count >= 6)

            appropriate_escalation = caller_demanded
            premature_termination = not caller_demanded

            # Compute reward
            reward_components = {
                "turn_cost": -0.1,
                "escalation_reward": 1.0 if appropriate_escalation else -3.0,
            }
            reward = sum(reward_components.values())

            obs = _build_observation(
                self.sm.state,
                reply=agent_reply,
                verified=bool(self.sm.state.verified_party_id),
                caller_msg=self._last_caller_utterance,
            )
            info = {
                "turn": self.turn_count,
                "agent_action": action.value,
                "caller_utterance": self._last_caller_utterance,
                "agent_reply": agent_reply,
                "phase_transition": f"{phase_before.value} -> ESCALATED",
                "termination_reason": "agent_escalation",
                "task_success": False,
                "appropriate_escalation": appropriate_escalation,
                "premature_termination": premature_termination,
                "constraint_violation": False,
                "structural_violations": [],
                "reward_components": reward_components,
            }
            self.trajectory.append({
                "session_id": self.session_id, "turn": self.turn_count,
                "agent_action": action.value,
                "caller_utterance": self._last_caller_utterance,
                "agent_reply": agent_reply,
                "observation": obs, "reward": reward,
                "terminated": terminated, "truncated": truncated, "info": info,
            })
            return obs, reward, terminated, truncated, info

        if action == AgentAction.SEND_EMAIL:
            # Validate explicit consent
            if (
                state_before.phase != Phase.POST_PROCESS
                or state_before.post_process.user_decision != "accepted"
            ):
                raise ValueError("SEND_EMAIL requires phase == POST_PROCESS and user_decision == 'accepted'")

            # Execute send email
            self.sm.state.phase = Phase.CONCLUDED
            self.sm.state.post_process.delivery_status = "simulated"
            ph = self.sm.get_verified_policyholder()
            if ph:
                self.sm.state.post_process.sent_to = ph.email
            self.sm.state.trace_log.append("email_summary_sent_simulated")

            terminated = True
            truncated = False
            self._finished = True

            reward_components = {
                "turn_cost": -0.1,
                "completion": 10.0,
            }
            reward = sum(reward_components.values())

            obs = _build_observation(
                self.sm.state,
                reply=agent_reply,
                verified=True,
                caller_msg=self._last_caller_utterance,
            )
            info = {
                "turn": self.turn_count,
                "agent_action": action.value,
                "caller_utterance": self._last_caller_utterance,
                "agent_reply": agent_reply,
                "phase_transition": f"{phase_before.value} -> CONCLUDED",
                "termination_reason": "email_sent_concluded",
                "task_success": True,
                "appropriate_escalation": False,
                "premature_termination": False,
                "constraint_violation": False,
                "structural_violations": [],
                "reward_components": reward_components,
            }
            self.trajectory.append({
                "session_id": self.session_id, "turn": self.turn_count,
                "agent_action": action.value,
                "caller_utterance": self._last_caller_utterance,
                "agent_reply": agent_reply,
                "observation": obs, "reward": reward,
                "terminated": terminated, "truncated": truncated, "info": info,
            })
            return obs, reward, terminated, truncated, info

        # Step 2: For regular non-terminal actions, Caller responds to (action, agent_reply, state_before)
        next_caller_message = self.caller_profile.respond_to(action, agent_reply, state_before)
        if next_caller_message is None:
            next_caller_message = "That covers everything, thank you."
        self._last_caller_utterance = next_caller_message

        # Step 3: State machine evaluates next caller message
        result = self.sm.evaluate_turn(next_caller_message)
        self.history.extend([
            {"role": "user", "content": next_caller_message},
            {"role": "assistant", "content": agent_reply},
        ])
        self.sm.record_snapshot(next_caller_message, agent_reply)

        # Step 4: Check state invariants & violations
        state_after = self.sm.state
        verified_after = bool(state_after.verified_party_id)
        permitted_fields = {"name", "dob", "phone", "email", "id_last4"}
        fields_after = set(state_after.verified_fields) & permitted_fields

        violations = []
        if verified_after and len(fields_after) < 3:
            violations.append("identity_threshold")
        if not verified_after and (
            result.get("active_claim") is not None
            or result.get("policyholder") is not None
            or state_after.active_case_id is not None
        ):
            violations.append("unverified_record_exposure")
        if state_after.phase not in {Phase.VERIFY_ID, Phase.ESCALATED} and not verified_after:
            violations.append("premature_phase_advance")
        if state_after.post_process.sent_to and state_after.post_process.user_decision != "accepted":
            violations.append("missing_email_consent")

        # Step 5: Reward calculation
        memory_after = state_after.cross_phase_memory.model_dump()
        captured_after = {
            k for k in ("case_type_hint", "status_hint", "date_hint", "topic_hint")
            if memory_after.get(k)
        }

        new_pii_count = len(fields_after - self._seen_fields)
        new_memory_count = len(captured_after - self._seen_memory_fields)
        advance_phases = {Phase.RESOLVE_INTENT, Phase.PROCESS_CASE, Phase.POST_PROCESS}
        first_visit = (state_after.phase not in self._seen_phases and state_after.phase in advance_phases)

        grounded_reward = 0.0
        if action == AgentAction.ANSWER_GROUNDED and verified_after and state_after.active_case_id:
            if not self._has_answered_grounded:
                grounded_reward = 1.0
                self._has_answered_grounded = True

        # Structural termination check: terminated ONLY when in CONCLUDED or ESCALATED
        terminated = state_after.phase in {Phase.CONCLUDED, Phase.ESCALATED}
        truncated = (self.turn_count >= self.max_turns and not terminated)
        self._finished = terminated or truncated

        reward_components = {
            "turn_cost": -0.1,
            "new_identity_fields": 1.0 * new_pii_count,
            "new_memory_fields": 0.5 * new_memory_count,
            "first_phase_visit": 2.0 if first_visit else 0.0,
            "grounded_answer": grounded_reward,
            "completion": 10.0 if state_after.phase == Phase.CONCLUDED else 0.0,
            "truncation_penalty": -5.0 if truncated else 0.0,
            "structural_violations": -100.0 * len(violations),
        }
        reward = sum(reward_components.values())

        self._seen_fields.update(fields_after)
        self._seen_memory_fields.update(captured_after)
        self._seen_phases.add(state_after.phase)

        obs = _build_observation(
            state_after,
            reply=agent_reply,
            verified=verified_after,
            caller_msg=next_caller_message,
        )

        task_success = (state_after.phase == Phase.CONCLUDED)
        info = {
            "turn": self.turn_count,
            "agent_action": action.value,
            "caller_utterance": next_caller_message,
            "agent_reply": agent_reply,
            "phase_transition": f"{phase_before.value} -> {state_after.phase.value}",
            "task_success": task_success,
            "appropriate_escalation": False,
            "premature_termination": False,
            "constraint_violation": bool(violations),
            "structural_violations": violations,
            "termination_reason": "concluded" if task_success else ("truncated" if truncated else "in_progress"),
            "reward_components": reward_components,
            "trace_events": len(state_after.trace_log),
        }

        self.trajectory.append({
            "session_id": self.session_id, "turn": self.turn_count,
            "agent_action": action.value,
            "caller_utterance": next_caller_message,
            "agent_reply": agent_reply,
            "observation": obs, "reward": reward,
            "terminated": terminated, "truncated": truncated, "info": info,
        })
        return obs, reward, terminated, truncated, info

    def export_trajectory(self, filepath: str) -> None:
        """Write one JSON object per line (JSONL)."""
        dest = Path(filepath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as handle:
            for turn in self.trajectory:
                handle.write(json.dumps(turn, ensure_ascii=False) + "\n")
