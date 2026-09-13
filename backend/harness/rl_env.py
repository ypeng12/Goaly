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
from .dialogue import dialogue_signals, render_action


# Gymnasium's default Text charset is alphanumeric-only.  Caller and agent
# messages contain spaces and punctuation, so use a bounded printable
# Unicode set while keeping the declared space finite and serializable.
_TEXT_CHARSET = frozenset(chr(i) for i in range(32, 127)) | frozenset("\n\r\t—…’“”")
_MEMORY_KEYS = ("case_type_hint", "status_hint", "date_hint", "topic_hint")


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
    # Keep the observation shape stable.  Extra state such as raw snippets and
    # notes remains in the SOP state/audit trail, not in the RL observation.
    memory_slots = {k: memory.get(k) or "" for k in _MEMORY_KEYS}
    post_accepted = getattr(state.post_process, "user_decision", None) == "accepted"
    return {
        "caller_utterance": caller_msg,
        "last_agent_reply": reply,
        "reply": reply,
        "phase": state.phase.value,
        # Sequence spaces require tuples when stack=False.
        "verified_fields": tuple(state.verified_fields),
        "memory_slots": memory_slots,
        "data_shield_active": not verified,
        # Text spaces cannot contain None; an empty value represents the
        # shielded/unresolved case in the observation.
        "active_case_id": state.active_case_id if verified and state.active_case_id else "",
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

        terminated = state.phase in {Phase.CONCLUDED, Phase.ESCALATED}
        truncated = self.turn_count >= self.max_turns and not terminated
        self._finished = terminated or truncated

        reward_components = {
            "turn_cost": -0.1,
            "new_identity_fields": float(len(fields - self._seen_fields)),
            "structural_violations": -100.0 * len(violations),
        }
        reward = sum(reward_components.values())
        self._seen_fields.update(fields)

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
    """Constrained action environment with causal speech, caller and SOP transitions.

    Masks reject illegal actions at the boundary. The finite caller protocol
    reacts to speech, never to policy labels or future SOP state.
    A max-turn timeout is a failed finite-horizon task (with -5 reward).
    """

    metadata = {"render_modes": []}
    ENV_VERSION = 3

    def __init__(self, caller_profile=None, session_id=None, engine=None,
                 max_turns=20, enforce_mask=True):
        super().__init__()
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns
        self.enforce_mask = enforce_mask
        self.engine = engine or MockEngine()
        self.caller_profile = caller_profile or make_margaret_chen_profile()
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        text_space = lambda n: spaces.Text(max_length=n, min_length=0, charset=_TEXT_CHARSET)
        self.observation_space = spaces.Dict({
            "phase": text_space(32), "caller_utterance": text_space(6000),
            "last_agent_reply": text_space(6000), "reply": text_space(6000),
            "verified_fields": spaces.Sequence(text_space(32)),
            "memory_slots": spaces.Dict({k: text_space(128) for k in _MEMORY_KEYS}),
            "data_shield_active": spaces.Discrete(2), "active_case_id": text_space(32),
            "action_mask": spaces.MultiBinary(ACTION_SPACE_SIZE),
            "emotion": text_space(32), "privacy_concern": spaces.Discrete(2),
            "demands_human": spaces.Discrete(2), "refusal": spaces.Discrete(2),
            "resolution_attempts": spaces.Discrete(max_turns + 1),
            "grounded_answered": spaces.Discrete(2),
        })
        self._initialize(session_id or str(uuid.uuid4()))

    def _initialize(self, session_id):
        self.session_id = session_id
        self.sm = SOPStateMachine(session_id)
        self.history, self.trajectory = [], []
        self.turn_count = 0
        self._finished = False
        self._last_caller_utterance = ""
        self._last_agent_reply = ""
        self._has_answered_grounded = False
        self._resolution_attempts = 0
        self._seen_fields, self._seen_memory_fields = set(), set()
        self._seen_phases = {Phase.VERIFY_ID}

    @property
    def phase(self):
        return self.sm.state.phase

    @property
    def action_space_size(self):
        return ACTION_SPACE_SIZE

    def get_action_mask(self):
        state = self.sm.state
        mask = _compute_action_mask(self.phase, self.sm.get_verified_policyholder() is not None,
                                    state.post_process.user_decision == "accepted")
        # Asking for a closing decision requires an actual grounded answer first.
        if not self._has_answered_grounded:
            mask[AGENT_ACTIONS.index(AgentAction.OFFER_EMAIL_SUMMARY)] = False
        return mask

    def _observation(self):
        signals = dialogue_signals(self._last_caller_utterance)
        obs = _build_observation(self.sm.state, self._last_agent_reply,
                                 self.sm.get_verified_policyholder() is not None,
                                 self._last_caller_utterance)
        obs.update(
            action_mask=self.get_action_mask(), emotion=signals["emotion"],
            privacy_concern=signals["privacy_concern"], refusal=signals["is_refusal"],
            demands_human=signals["demands_human"],
            resolution_attempts=self._resolution_attempts,
            grounded_answered=self._has_answered_grounded,
        )
        return obs

    def save_state(self):
        """Include histories, caller state and local RNG for exact branching."""
        names = ("session_id", "history", "trajectory", "turn_count", "_finished",
                 "_last_caller_utterance", "_last_agent_reply", "_has_answered_grounded",
                 "_resolution_attempts", "_seen_fields", "_seen_memory_fields", "_seen_phases")
        return {
            "env": copy.deepcopy({k: getattr(self, k) for k in names}),
            "sm_state": copy.deepcopy(self.sm.state),
            "caller_state": self.caller_profile.get_state(),
            "rng": copy.deepcopy(self.np_random.bit_generator.state),
            "action_rng": copy.deepcopy(self.action_space.np_random.bit_generator.state),
        }

    def restore_state(self, snapshot):
        for key, value in copy.deepcopy(snapshot["env"]).items():
            setattr(self, key, value)
        self.sm = SOPStateMachine(self.session_id)
        self.sm.state = copy.deepcopy(snapshot["sm_state"])
        self.caller_profile.set_state(snapshot["caller_state"])
        self.np_random.bit_generator.state = copy.deepcopy(snapshot["rng"])
        self.action_space.np_random.bit_generator.state = copy.deepcopy(snapshot["action_rng"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.action_space.seed(seed)
        # Local seeding cannot reseed another environment or the training policy.
        caller_seed = int(self.np_random.integers(0, 2**31))
        self.caller_profile.reset(caller_seed)
        self._initialize(str(uuid.uuid4()))
        opening = self.caller_profile.get_opening_utterance()
        self.sm.evaluate_turn(opening)
        self._last_caller_utterance = opening
        self.history.append({"role": "user", "content": opening})
        self._seen_fields.update(self.sm.state.verified_fields)
        self._seen_memory_fields.update(k for k in _MEMORY_KEYS
                                        if getattr(self.sm.state.cross_phase_memory, k))
        return self._observation(), {"session_id": self.session_id, "turn_count": 0, "seed": seed}

    def _escalation_reason(self):
        state = self.sm.state
        if dialogue_signals(self._last_caller_utterance)["demands_human"]:
            return "caller_requested_human"
        if state.refusal_count >= 3:
            return "repeated_refusal"
        if state.out_of_scope_count >= 2:
            return "repeated_out_of_scope"
        if state.resolution_status in {"no_match", "ambiguous", "needs_intent"} and self._resolution_attempts >= 2:
            return "case_resolution_exhausted"
        return None

    def _violations(self):
        state = self.sm.state
        verified = self.sm.get_verified_policyholder() is not None
        errors = []
        if state.identity_verified and not verified:
            errors.append("identity_threshold_or_conflict")
        if not verified and state.active_case_id:
            errors.append("unverified_record_exposure")
        if state.phase not in {Phase.VERIFY_ID, Phase.ESCALATED} and not verified:
            errors.append("premature_phase_advance")
        if state.active_case_id and verified and self.sm.get_active_claim() is None:
            errors.append("case_ownership")
        if state.post_process.sent_to and state.post_process.user_decision != "accepted":
            errors.append("missing_email_consent")
        return errors

    def step(self, action):
        if self._finished:
            raise RuntimeError("Session is finished; call reset() before another step")
        if isinstance(action, (int, np.integer)) and not isinstance(action, bool):
            if not self.action_space.contains(action):
                raise ValueError("action index out of range")
            action = AGENT_ACTIONS[int(action)]
        if not isinstance(action, AgentAction):
            raise ValueError("action must be an AgentAction or integer index")
        legal = self.get_action_mask()[AGENT_ACTIONS.index(action)]
        if not legal and self.enforce_mask:
            raise ValueError(f"Action {action.value} is masked in {self.phase.value}")
        before = copy.deepcopy(self.sm.state)
        obs_before = self._observation()
        self.turn_count += 1
        reason = None
        caller_message = ""
        final_reply = ""
        grounded_reward = 0.0
        if not legal:
            # Diagnostic mask ablation records a violation; never executes it.
            agent_reply = "This action is unavailable at this workflow step."
        else:
            agent_reply = render_action(action, self.sm, self._last_caller_utterance)
            self.history.append({"role": "assistant", "content": agent_reply})
            if action == AgentAction.ESCALATE_HUMAN:
                reason = self._escalation_reason()
                self.sm.state.phase = Phase.ESCALATED
                self.sm._add_trace("AGENT_ESCALATION", bool(reason),
                                   reason or "premature_agent_escalation", before.phase)
            elif action == AgentAction.SEND_EMAIL:
                # Consent and ownership remain under the same SOP authority.
                self.sm.evaluate_turn("Yes, please send the email summary.")
            else:
                if action == AgentAction.ANSWER_GROUNDED and self.sm.get_active_claim():
                    if not self._has_answered_grounded:
                        grounded_reward = 1.0
                    self._has_answered_grounded = True
                if before.phase == Phase.RESOLVE_INTENT and action in {
                    AgentAction.RESOLVE_INTENT, AgentAction.ASK_CLAIM_CLARIFICATION
                }:
                    self._resolution_attempts += 1
                caller_message = self.caller_profile.respond_to(action, agent_reply)
                self._last_caller_utterance = caller_message
                self.history.append({"role": "user", "content": caller_message})
                self.sm.evaluate_turn(caller_message)
                if self.phase == Phase.ESCALATED:
                    reason = self._escalation_reason()
                if self.phase in {Phase.CONCLUDED, Phase.ESCALATED}:
                    final_reply = self.engine.generate_response(
                        caller_message, self.sm._result(before.phase), self.history)
                    self.history.append({"role": "assistant", "content": final_reply})

        state = self.sm.state
        fields = set(state.verified_fields)
        memory = {k for k in _MEMORY_KEYS if getattr(state.cross_phase_memory, k)}
        violations = self._violations() + ([] if legal else ["illegal_action"])
        terminated = self.phase in {Phase.CONCLUDED, Phase.ESCALATED}
        truncated = self.turn_count >= self.max_turns and not terminated
        success = (self.phase == Phase.CONCLUDED and self._has_answered_grounded
                   and state.post_process.user_decision in {"accepted", "declined"}
                   and not violations)
        appropriate = self.phase == Phase.ESCALATED and reason is not None
        premature = (self.phase == Phase.ESCALATED and not appropriate) or (
            self.phase == Phase.CONCLUDED and not success)
        components = {
            "turn_cost": -0.1, "new_identity_fields": float(len(fields - self._seen_fields)),
            # A soft customer-experience cost; no positive empathy bonus to farm.
            # Explicit human requests take precedence over continued persuasion.
            "ignored_distress": -0.5 if (
                obs_before['emotion'] in {'frustration', 'anger', 'anxiety'}
                and not obs_before['demands_human']
                and action not in {AgentAction.ACK_EMOTION, AgentAction.ESCALATE_HUMAN}
            ) else 0.0,
            "new_memory_fields": 0.5 * len(memory - self._seen_memory_fields),
            "first_phase_visit": 2.0 if self.phase not in self._seen_phases and self.phase in {
                Phase.RESOLVE_INTENT, Phase.PROCESS_CASE, Phase.POST_PROCESS} else 0.0,
            "grounded_answer": grounded_reward, "completion": 10.0 if success else 0.0,
            "escalation_reward": 1.0 if appropriate else (-3.0 if premature else 0.0),
            "truncation_penalty": -5.0 if truncated else 0.0,
            "structural_violations": -100.0 * len(violations),
        }
        self._finished = terminated or truncated
        self._last_agent_reply = final_reply or agent_reply
        self._seen_fields.update(fields)
        self._seen_memory_fields.update(memory)
        self._seen_phases.add(self.phase)
        obs = self._observation()
        info = {
            "turn": self.turn_count, "agent_action": action.value,
            "agent_reply": agent_reply, "caller_utterance": caller_message,
            "final_reply": final_reply, "phase_transition": f"{before.phase.value} -> {self.phase.value}",
            "task_success": success, "appropriate_escalation": appropriate,
            "premature_termination": premature, "constraint_violation": bool(violations),
            "structural_violations": violations, "reward_components": components,
            "escalation_reason": reason,
            "termination_reason": ("agent_escalation" if action == AgentAction.ESCALATE_HUMAN
                                   else reason or ("concluded" if success else
                                                   "truncated" if truncated else "in_progress")),
        }
        self.sm.record_snapshot(caller_message, final_reply or agent_reply)
        reward = sum(components.values())
        self.trajectory.append({
            "session_id": self.session_id, "turn": self.turn_count,
            "agent_action": action.value, "agent_reply": agent_reply,
            "caller_utterance": caller_message, "final_reply": final_reply,
            "observation_before": obs_before, "observation": obs,
            "reward": reward, "terminated": terminated, "truncated": truncated, "info": info,
        })
        return obs, reward, terminated, truncated, info

    def export_trajectory(self, filepath):
        dest = Path(filepath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in self.trajectory),
                        encoding="utf-8")
