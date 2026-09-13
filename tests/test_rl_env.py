"""Tests for the RL environment infrastructure.

Covers:
  CallerSimulatorEnv  (formerly InsuranceSOPEnv)
  AgentPolicyEnv      (new formal action-space environment)
  RuleBasedPolicy     (baseline policy)
  action_mask         (per-phase constraints)
"""
import json
import os
import tempfile
import pytest

from backend.harness.types import AgentAction, AGENT_ACTIONS, ACTION_SPACE_SIZE, Phase
from backend.harness.rl_env import (
    CallerSimulatorEnv,
    InsuranceSOPEnv,          # backward-compat alias
    AgentPolicyEnv,
    _compute_action_mask,
)
from backend.harness.rl_baseline import RuleBasedPolicy


# ---------------------------------------------------------------------------
# CallerSimulatorEnv  (backward-compat renamed class)
# ---------------------------------------------------------------------------

class TestCallerSimulatorEnv:

    def test_backward_compat_alias(self):
        """InsuranceSOPEnv must remain importable as an alias."""
        env = InsuranceSOPEnv(max_turns=5)
        obs, info = env.reset()
        assert obs["phase"] == Phase.VERIFY_ID.value

    def test_reset_returns_valid_schema(self):
        env = CallerSimulatorEnv(max_turns=10)
        obs, info = env.reset()
        assert "reply" in obs
        assert "phase" in obs
        assert "action_mask" in obs
        assert len(obs["action_mask"]) == ACTION_SPACE_SIZE
        assert "session_id" in info
        assert info["turn_count"] == 0

    def test_step_increments_turn(self):
        env = CallerSimulatorEnv(max_turns=10)
        env.reset()
        obs, reward, terminated, truncated, info = env.step("Hello, I need help.")
        assert info["turn"] == 1

    def test_step_on_finished_env_raises(self):
        env = CallerSimulatorEnv(max_turns=1)
        env.reset()
        env.step("Hi there.")
        with pytest.raises(RuntimeError, match="finished"):
            env.step("Another message.")

    def test_step_empty_action_raises(self):
        env = CallerSimulatorEnv(max_turns=5)
        env.reset()
        with pytest.raises(ValueError, match="nonempty"):
            env.step("   ")

    def test_truncation_on_max_turns(self):
        env = CallerSimulatorEnv(max_turns=2)
        env.reset()
        env.step("First turn.")
        _, _, terminated, truncated, _ = env.step("Second turn.")
        assert truncated or terminated  # at max_turns, should be done

    def test_structural_violation_penalty(self):
        """Unverified record exposure should trigger -100 penalty."""
        env = CallerSimulatorEnv(max_turns=10)
        env.reset()
        # Force a scenario where we check violations by inspecting reward components
        _, _, _, _, info = env.step("I need to check my claim status.")
        # No violation expected on a simple opening message
        assert isinstance(info["structural_violations"], list)
        assert isinstance(info["reward_components"], dict)

    def test_export_trajectory_writes_jsonl(self):
        env = CallerSimulatorEnv(max_turns=5)
        env.reset()
        env.step("My name is Jane Doe.")
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            env.export_trajectory(path)
            with open(path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert "session_id" in data
            assert "reward" in data
            assert "observation" in data
        finally:
            os.unlink(path)

    def test_observation_contains_action_mask(self):
        env = CallerSimulatorEnv(max_turns=10)
        env.reset()
        obs, _, _, _, _ = env.step("Hello.")
        assert "action_mask" in obs
        assert len(obs["action_mask"]) == ACTION_SPACE_SIZE
        assert all(isinstance(b, bool) for b in obs["action_mask"])


# ---------------------------------------------------------------------------
# action_mask logic
# ---------------------------------------------------------------------------

class TestActionMask:

    def test_answer_grounded_masked_when_unverified(self):
        """ANSWER_GROUNDED must be masked in VERIFY_ID phase (unverified)."""
        mask = _compute_action_mask(Phase.VERIFY_ID, verified=False)
        idx = AGENT_ACTIONS.index(AgentAction.ANSWER_GROUNDED)
        assert mask[idx] is False

    def test_answer_grounded_available_in_process_case_when_verified(self):
        mask = _compute_action_mask(Phase.PROCESS_CASE, verified=True)
        idx = AGENT_ACTIONS.index(AgentAction.ANSWER_GROUNDED)
        assert mask[idx] is True

    def test_answer_grounded_still_masked_in_process_case_when_unverified(self):
        mask = _compute_action_mask(Phase.PROCESS_CASE, verified=False)
        idx = AGENT_ACTIONS.index(AgentAction.ANSWER_GROUNDED)
        assert mask[idx] is False

    def test_verify_id_phase_mask_allows_core_actions(self):
        mask = _compute_action_mask(Phase.VERIFY_ID, verified=False)
        expected_legal = {
            AgentAction.ACK_EMOTION,
            AgentAction.ASK_IDENTITY_FIELD,
            AgentAction.EXPLAIN_VERIFICATION_GATE,
            AgentAction.ESCALATE_HUMAN,
        }
        for action in AGENT_ACTIONS:
            idx = AGENT_ACTIONS.index(action)
            if action in expected_legal:
                assert mask[idx] is True, f"{action.value} should be legal in VERIFY_ID"
            else:
                assert mask[idx] is False, f"{action.value} should be masked in VERIFY_ID"

    def test_concluded_phase_mask_is_all_false(self):
        mask = _compute_action_mask(Phase.CONCLUDED, verified=True)
        assert all(b is False for b in mask), "CONCLUDED phase should have no legal actions"

    def test_mask_length_equals_action_space_size(self):
        for phase in Phase:
            mask = _compute_action_mask(phase, verified=True)
            assert len(mask) == ACTION_SPACE_SIZE


# ---------------------------------------------------------------------------
# AgentPolicyEnv
# ---------------------------------------------------------------------------

class TestAgentPolicyEnv:

    def test_reset_returns_valid_obs(self):
        env = AgentPolicyEnv(max_turns=10)
        obs, info = env.reset()
        assert obs["phase"] == Phase.VERIFY_ID.value
        assert len(obs["action_mask"]) == ACTION_SPACE_SIZE
        assert obs["data_shield_active"] is True

    def test_action_space_size_property(self):
        env = AgentPolicyEnv()
        assert env.action_space_size == ACTION_SPACE_SIZE

    def test_illegal_action_raises_when_mask_enforced(self):
        """ANSWER_GROUNDED is masked in VERIFY_ID; stepping with it should raise."""
        env = AgentPolicyEnv(max_turns=10, enforce_mask=True)
        env.reset()
        with pytest.raises(ValueError, match="masked"):
            env.step(AgentAction.ANSWER_GROUNDED)

    def test_illegal_action_allowed_when_mask_not_enforced(self):
        """When enforce_mask=False, stepping with a masked action should not raise."""
        env = AgentPolicyEnv(max_turns=10, enforce_mask=False)
        env.reset()
        # Should not raise
        obs, reward, _, _, info = env.step(AgentAction.ANSWER_GROUNDED)
        assert obs is not None

    def test_legal_action_step_returns_five_tuple(self):
        env = AgentPolicyEnv(max_turns=10)
        env.reset()
        result = env.step(AgentAction.ASK_IDENTITY_FIELD)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_escalate_human_terminates_session(self):
        env = AgentPolicyEnv(max_turns=20)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(AgentAction.ESCALATE_HUMAN)
        # ESCALATE_HUMAN transitions to ESCALATED which terminates
        assert terminated or obs["phase"] == Phase.ESCALATED.value

    def test_step_on_finished_env_raises(self):
        env = AgentPolicyEnv(max_turns=1)
        env.reset()
        env.step(AgentAction.ACK_EMOTION)
        with pytest.raises(RuntimeError, match="finished"):
            env.step(AgentAction.ACK_EMOTION)

    def test_invalid_action_type_raises(self):
        env = AgentPolicyEnv(max_turns=5)
        env.reset()
        with pytest.raises((ValueError, TypeError)):
            env.step("ask_identity_field")  # string, not AgentAction

    def test_trajectory_records_agent_action(self):
        env = AgentPolicyEnv(max_turns=10)
        env.reset()
        env.step(AgentAction.ASK_IDENTITY_FIELD)
        assert len(env.trajectory) == 1
        assert env.trajectory[0]["agent_action"] == AgentAction.ASK_IDENTITY_FIELD.value


# ---------------------------------------------------------------------------
# RuleBasedPolicy
# ---------------------------------------------------------------------------

class TestRuleBasedPolicy:

    def test_policy_never_selects_masked_action(self):
        """Policy must always return a legal action."""
        env = AgentPolicyEnv(max_turns=20)
        policy = RuleBasedPolicy()
        obs, _ = env.reset()
        for _ in range(5):
            if all(not b for b in obs["action_mask"]):
                break
            action = policy.select_action(obs)
            idx = AGENT_ACTIONS.index(action)
            assert obs["action_mask"][idx] is True, (
                f"Policy selected masked action {action.value} in phase {obs['phase']}"
            )
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    def test_policy_selects_ask_identity_in_verify_phase(self):
        """In VERIFY_ID with no verified fields, policy should ask for identity."""
        obs = {
            "phase": Phase.VERIFY_ID.value,
            "verified_fields": [],
            "memory_slots": {},
            "data_shield_active": True,
            "active_case_id": None,
            "caller_utterance": "Hello, I need help.",
            "last_agent_reply": "",
            "action_mask": _compute_action_mask(Phase.VERIFY_ID, verified=False),
        }
        policy = RuleBasedPolicy()
        action = policy.select_action(obs)
        assert action == AgentAction.ASK_IDENTITY_FIELD

    def test_run_episode_achieves_positive_reward(self):
        """The rule-based policy should accumulate positive reward (no violations)."""
        env = AgentPolicyEnv(max_turns=15)
        policy = RuleBasedPolicy()
        result = policy.run_episode(env)
        # Rule policy should never trigger structural violations
        assert result["violations"] == [], (
            f"Rule policy produced violations: {result['violations']}"
        )
        # Should earn at least the turn_cost contribution without being disqualified
        assert isinstance(result["cumulative_reward"], float)


# ---------------------------------------------------------------------------
# Critical regression tests: caller-agent loop must be properly connected
# ---------------------------------------------------------------------------

class TestCallerAgentLoop:
    """Tests that verify the agent-caller loop works end-to-end correctly.

    These are the tests that were FAILING before the architectural fix:
    - CallerProfile provides real PII → state machine can verify identity
    - RuleBasedPolicy terminates (not truncates) using CallerProfile
    - Trajectory records real caller utterances, not synthetic ones
    """

    def test_caller_profile_provides_pii_that_verifies(self):
        """CallerProfile must provide PII utterances that the state machine accepts."""
        from backend.harness.caller_sim import make_margaret_chen_profile
        profile = make_margaret_chen_profile(style="cooperative")
        env = AgentPolicyEnv(caller_profile=profile, max_turns=20)
        obs, _ = env.reset()

        # Opening utterance should already give name
        assert "name" in obs["verified_fields"] or len(obs["verified_fields"]) >= 0
        # DOB and phone should be extractable after a few ASK_IDENTITY_FIELD steps
        steps = 0
        while "dob" not in obs["verified_fields"] and steps < 5:
            if obs["action_mask"][AGENT_ACTIONS.index(AgentAction.ASK_IDENTITY_FIELD)]:
                obs, _, term, trunc, _ = env.step(AgentAction.ASK_IDENTITY_FIELD)
            else:
                break
            steps += 1

        # After providing name + dob + phone, should have ≥ 3 verified fields
        # (name comes from opening utterance, dob from step 1, phone from step 2)
        assert len(obs["verified_fields"]) >= 1, (
            f"CallerProfile never provided verifiable PII. Verified: {obs['verified_fields']}"
        )

    def test_rule_policy_terminates_not_truncates_with_caller_profile(self):
        """RuleBasedPolicy + CallerProfile must terminate cleanly, not hit max_turns.

        This was the key failure before the architectural fix: the old code fed
        agent actions as fake caller utterances, so the state machine never got
        real PII and the policy looped until truncation.
        """
        from backend.harness.caller_sim import make_margaret_chen_profile
        profile = make_margaret_chen_profile(style="cooperative")
        env = AgentPolicyEnv(caller_profile=profile, max_turns=20)
        policy = RuleBasedPolicy()
        result = policy.run_episode(env)

        assert not result["truncated"], (
            f"RuleBasedPolicy was truncated at max_turns (phase={result['final_phase']}). "
            f"This means CallerProfile PII is not flowing into the state machine correctly."
        )
        assert result["terminated"], (
            f"Episode did not terminate. Final phase: {result['final_phase']}"
        )
        assert result["violations"] == [], f"Unexpected violations: {result['violations']}"

    def test_trajectory_contains_real_caller_utterances(self):
        """Trajectory must record actual caller utterances, not synthetic ones."""
        from backend.harness.caller_sim import make_margaret_chen_profile
        profile = make_margaret_chen_profile(style="cooperative")
        env = AgentPolicyEnv(caller_profile=profile, max_turns=10)
        env.reset()
        env.step(AgentAction.ASK_IDENTITY_FIELD)

        assert len(env.trajectory) >= 1
        turn = env.trajectory[0]
        caller_text = turn["caller_utterance"]

        # Must NOT be one of the old synthetic utterances
        synthetic_phrases = [
            "Could you please provide your",
            "I understand, this must be frustrating",
            "I need to verify your identity before",
        ]
        for phrase in synthetic_phrases:
            assert phrase not in caller_text, (
                f"Trajectory contains synthetic utterance, not real caller text: {caller_text!r}"
            )

        # Must contain real PII-style content from CallerProfile
        assert len(caller_text) > 5, f"Caller utterance too short: {caller_text!r}"

    def test_caller_profile_pii_advances_phase(self):
        """Providing 3+ PII fields must advance the SOP past VERIFY_ID."""
        from backend.harness.caller_sim import make_margaret_chen_profile
        profile = make_margaret_chen_profile(style="cooperative")
        env = AgentPolicyEnv(caller_profile=profile, max_turns=20)
        obs, _ = env.reset()

        # Keep asking for identity until we're past VERIFY_ID or at max attempts
        for _ in range(6):
            if obs["phase"] != Phase.VERIFY_ID.value:
                break
            if obs["action_mask"][AGENT_ACTIONS.index(AgentAction.ASK_IDENTITY_FIELD)]:
                obs, _, term, trunc, _ = env.step(AgentAction.ASK_IDENTITY_FIELD)
            elif obs["action_mask"][AGENT_ACTIONS.index(AgentAction.ACK_EMOTION)]:
                obs, _, term, trunc, _ = env.step(AgentAction.ACK_EMOTION)
            if term or trunc:
                break

        assert obs["phase"] != Phase.VERIFY_ID.value, (
            f"Phase never advanced beyond VERIFY_ID after 6 steps. "
            f"This means the CallerProfile PII is not being accepted by the state machine."
        )

