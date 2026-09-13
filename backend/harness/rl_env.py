"""A caller-simulation adapter for experiments, with a Gym-like return signature.

This is not a gymnasium.Env: it has no action_space/observation_space and its
'action' is a caller utterance, not a learned assistant policy action. Rewards
are illustrative shaping scores, not calibrated customer outcomes or a CMDP
safety proof. Use the benchmark for explicit, separately reported assertions.
"""
import copy
import json
import uuid
from pathlib import Path
from typing import Any, Optional

from .types import Phase
from .state_machine import SOPStateMachine
from ..engine.mock_engine import MockEngine
from ..engine.base import BaseEngine


class InsuranceSOPEnv:
    def __init__(self, session_id: Optional[str] = None, engine: Optional[BaseEngine] = None, max_turns: int = 20):
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.engine = engine or MockEngine()
        self.max_turns = max_turns
        self._initialize(session_id or str(uuid.uuid4()))

    def _initialize(self, session_id: str):
        self.session_id = session_id
        self.sm = SOPStateMachine(session_id)
        self.history: list[dict[str, str]] = []
        self.trajectory: list[dict[str, Any]] = []
        self.turn_count = 0
        self._finished = False
        self._seen_fields: set[str] = set()
        self._seen_memory_fields: set[str] = set()
        self._seen_phases = {Phase.VERIFY_ID}

    @property
    def phase(self) -> Phase:
        return self.sm.state.phase

    def get_phase(self) -> str:
        return self.phase.value

    def reset(self, seed: Optional[int] = None):
        """Start a fresh session. Seed is metadata; this adapter has no RNG."""
        self._initialize(str(uuid.uuid4()))
        return {
            "reply": "Welcome to insurance claims support. We need three identity details before discussing a claim.",
            "phase": self.phase.value,
            "data_shield_active": True,
            "verified_fields": [],
            "active_case_id": None,
        }, {"session_id": self.session_id, "turn_count": 0, "seed": seed, "seed_used": False}

    def step(self, action: str):
        """Return (observation, reward, terminated, truncated, info)."""
        if self._finished:
            raise RuntimeError("Session is finished; call reset() before another step")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a nonempty caller utterance")
        self.turn_count += 1
        phase_before = self.phase
        result = self.sm.evaluate_turn(action)
        reply = self.engine.generate_response(action, result, self.history)
        self.history.extend([{"role": "user", "content": action}, {"role": "assistant", "content": reply}])
        self.sm.record_snapshot(action, reply)
        state = self.sm.state
        verified = bool(state.verified_party_id)
        permitted_fields = {"name", "dob", "phone", "email", "id_last4"}
        fields = set(state.verified_fields) & permitted_fields
        violations = []
        if verified and len(fields) < 3:
            violations.append("identity_threshold")
        if not verified and (result.get("active_claim") is not None or result.get("policyholder") is not None or state.active_case_id is not None):
            violations.append("unverified_record_exposure")
        if state.phase not in {Phase.VERIFY_ID, Phase.ESCALATED} and not verified:
            violations.append("premature_phase_advance")
        if state.post_process.sent_to and state.post_process.user_decision != "accepted":
            violations.append("missing_email_consent")

        memory = state.cross_phase_memory.model_dump()
        captured = {key for key in ("case_type_hint", "status_hint", "date_hint", "topic_hint") if memory.get(key)}
        components = {
            "turn_cost": -0.1,
            "new_identity_fields": float(len(fields - self._seen_fields)),
            "new_memory_fields": 0.5 * len(captured - self._seen_memory_fields),
            "first_phase_visit": 2.0 if state.phase not in self._seen_phases and state.phase in {Phase.RESOLVE_INTENT, Phase.PROCESS_CASE, Phase.POST_PROCESS} else 0.0,
            "completion": 5.0 if state.phase == Phase.CONCLUDED else 0.0,
            "structural_safety_cost": -100.0 * len(violations),
        }
        self._seen_fields.update(fields)
        self._seen_memory_fields.update(captured)
        self._seen_phases.add(state.phase)
        reward = sum(components.values())
        terminated = state.phase in {Phase.CONCLUDED, Phase.ESCALATED}
        truncated = self.turn_count >= self.max_turns and not terminated
        self._finished = terminated or truncated
        observation = {
            "reply": reply, "phase": state.phase.value,
            "data_shield_active": not verified,
            "verified_fields": list(state.verified_fields),
            "active_case_id": state.active_case_id if verified else None,
        }
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

    def export_trajectory(self, filepath: str):
        """Write one JSON object per line. Contains caller text; use synthetic data."""
        destination = Path(filepath)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for turn in self.trajectory:
                handle.write(json.dumps(turn, ensure_ascii=False) + "\n")
