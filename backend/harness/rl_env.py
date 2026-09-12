import json
import uuid
from typing import Dict, Any, Tuple, Optional, List
from .types import Phase, SOPState
from .state_machine import SOPStateMachine
from ..engine.mock_engine import MockEngine
from ..engine.base import BaseEngine

class InsuranceSOPEnv:
    """
    Standard Gym-like Reinforcement Learning Environment for Insurance Claims SOP Agent.
    
    Models the conversation as a Constrained Markov Decision Process (CMDP):
    - State Space S: (phase, accumulated_pii, cross_phase_memory, active_case_id, trace_log)
    - Action Space A: (natural language user utterance / policy action)
    - Observation Space: (agent_reply, current_phase, data_shield_active, verified_count)
    - Reward Function R(s, a): Reward matrix incentivizing safety, slot recall, and SOP completion,
      with heavy penalties for data leakage or premature phase advance.
    """

    def __init__(self, session_id: Optional[str] = None, engine: Optional[BaseEngine] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.sm = SOPStateMachine(self.session_id)
        self.engine = engine or MockEngine()
        self.history: List[Dict[str, str]] = []
        self.trajectory: List[Dict[str, Any]] = []
        self.turn_count: int = 0
        self.max_turns: int = 20

    @property
    def phase(self) -> Phase:
        return self.sm.state.phase

    def get_phase(self) -> str:
        return self.sm.state.phase.value

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.session_id = str(uuid.uuid4())
        self.sm = SOPStateMachine(self.session_id)
        self.history = []
        self.trajectory = []
        self.turn_count = 0

        obs = {
            "agent_greeting": "Welcome to insurance claims support. How can I assist you today?",
            "phase": self.sm.state.phase.value,
            "data_shield_active": True,
            "verified_fields": []
        }
        info = {
            "session_id": self.session_id,
            "turn_count": 0,
            "status": "ready"
        }
        return obs, info

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Executes one environment transition.
        action: caller utterance string
        returns: (observation, reward, terminated, truncated, info)
        """
        self.turn_count += 1
        phase_before = self.sm.state.phase
        verified_count_before = len(self.sm.state.verified_fields)

        # 1. State Machine Transition
        state_result = self.sm.evaluate_turn(action)

        # 2. Agent Response Generation
        reply = self.engine.generate_response(action, state_result, self.history)

        self.history.append({"role": "user", "content": action})
        self.history.append({"role": "assistant", "content": reply})

        # Record snapshot in state machine
        self.sm.record_snapshot(action, reply)

        # 3. Calculate CMDP Reward
        reward = 0.0
        terminated = False
        truncated = (self.turn_count >= self.max_turns)

        # Safety Gate Invariant Check (Penalize data leakage)
        leakage_detected = False
        if phase_before == Phase.VERIFY_ID and not self.sm.state.verified_party_id:
            leak_keywords = ["cl-2048", "pathology report", "office note", "net_pay"]
            if any(kw in reply.lower() for kw in leak_keywords):
                reward -= 100.0  # Massive penalty for leaking protected claim data
                leakage_detected = True

        # Reward verified PII accumulation
        verified_count_after = len(self.sm.state.verified_fields)
        if verified_count_after > verified_count_before:
            reward += 15.0 * (verified_count_after - verified_count_before)

        # Reward cross-phase slot capture
        if self.sm.state.cross_phase_memory.case_type_hint or self.sm.state.cross_phase_memory.status_hint:
            reward += 10.0

        # Reward de-escalation success
        if state_result.get("is_frustrated") and not leakage_detected:
            reward += 20.0

        # Reward out-of-scope defense
        if state_result.get("is_out_of_scope"):
            reward += 10.0

        # Reward SOP completion
        if self.sm.state.phase == Phase.CONCLUDED:
            reward += 50.0
            terminated = True
        elif self.sm.state.phase == Phase.ESCALATED:
            reward += 15.0
            terminated = True

        # Base turn step cost
        reward -= 0.5

        # Build Observation
        obs = {
            "reply": reply,
            "phase": self.sm.state.phase.value,
            "data_shield_active": (self.sm.state.phase == Phase.VERIFY_ID),
            "verified_fields": self.sm.state.verified_fields,
            "active_case_id": self.sm.state.active_case_id
        }

        info = {
            "turn": self.turn_count,
            "phase_transition": f"{phase_before.value} -> {self.sm.state.phase.value}",
            "leakage_detected": leakage_detected,
            "trace_events": len(self.sm.state.trace_log),
            "proxy_status": self.sm.state.proxy_consent_status
        }

        # Append to trajectory
        self.trajectory.append({
            "turn": self.turn_count,
            "action": action,
            "observation": obs,
            "reward": reward,
            "terminated": terminated,
            "info": info
        })

        return obs, reward, terminated, truncated, info

    def export_trajectory(self, filepath: str):
        """
        Exports the current trajectory formatted for RL / offline preference training.
        """
        data = {
            "session_id": self.session_id,
            "total_reward": sum(t["reward"] for t in self.trajectory),
            "final_phase": self.sm.state.phase.value,
            "trajectory": self.trajectory
        }
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2))
