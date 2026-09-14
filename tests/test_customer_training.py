"""Causality, reward semantics and shared-runtime contract for customer PPO."""
import copy
import inspect

import numpy as np
import pytest

from backend.harness.customer_policy import decide
from backend.harness.customer_training import CustomerPolicyTrainingEnv, ReactiveCustomer, scenarios
from backend.harness.types import AgentAction, Phase
from backend.rl.featurizer import StateFeaturizer


def test_reactive_customer_receives_speech_only_and_changes_with_actual_reply():
    assert list(inspect.signature(ReactiveCustomer.respond).parameters) == ["self", "reply"]
    scenario = scenarios()[0]
    one, two = ReactiveCustomer(scenario, 42), ReactiveCustomer(scenario, 42)
    assert one.opening() == two.opening()
    assert "date of birth" in one.respond("Please share your date of birth.")
    assert "phone number" in two.respond("Please share your phone number.")
    assert one.given != two.given


def test_training_uses_customer_mask_and_refuses_illegal_actions_before_effects():
    env = CustomerPolicyTrainingEnv()
    obs, _ = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    before = copy.deepcopy(env.machine.state.model_dump())
    with pytest.raises(ValueError, match="masked"):
        env.step(AgentAction.ANSWER_GROUNDED)
    assert env.machine.state.model_dump() == before
    assert env.turn_count == 0
    assert env.caller.given == {"name"}


def test_training_next_caller_message_is_absorbed_after_the_actual_response():
    env = CustomerPolicyTrainingEnv()
    obs, _ = env.reset(seed=42)
    before = env.machine.state.phase
    obs, reward, terminal, truncated, info = env.step(AgentAction.ASK_IDENTITY_FIELD)
    turn = env.trajectory[-1]
    assert "date of birth" in turn["reply"]
    assert "date of birth" in turn["next_caller_message"]
    assert turn["observation_before"]["verified_fields"] == ["name"]
    assert set(obs["verified_fields"]) == {"name", "dob"}
    assert before == env.machine.state.phase == Phase.VERIFY_ID
    assert not terminal and not truncated


def test_text_empathy_cost_and_no_positive_ack_reward():
    scenario = next(s for s in scenarios() if s.name == "ma-angry")
    env = CustomerPolicyTrainingEnv(scenario)
    obs, _ = env.reset(seed=42)
    assert env.machine.state.phase == Phase.PROCESS_CASE
    _, _, _, _, ignored = env.step(AgentAction.ANSWER_GROUNDED)
    assert ignored["reward_components"]["ignored_distress"] == -1.0
    assert not ignored["empathy_delivered"]
    env = CustomerPolicyTrainingEnv(scenario)
    env.reset(seed=42)
    _, reward, _, _, empathetic = env.step(AgentAction.ACK_EMOTION)
    assert empathetic["empathy_delivered"]
    assert empathetic["reward_components"]["ignored_distress"] == 0.0
    assert reward == pytest.approx(-0.1)


def test_unnecessary_and_repeated_explanations_are_measured_from_text():
    env = CustomerPolicyTrainingEnv(scenarios()[0])
    env.reset(seed=42)
    _, _, _, _, first = env.step(AgentAction.EXPLAIN_VERIFICATION_GATE)
    assert first["unnecessary_explanation"]
    _, _, _, _, second = env.step(AgentAction.EXPLAIN_VERIFICATION_GATE)
    assert second["unnecessary_explanation"]
    assert second["reward_components"]["unnecessary_gate_explanation"] == -0.4


def test_five_followups_and_real_consent_needed_for_customer_success():
    env = CustomerPolicyTrainingEnv(scenarios()[0])
    obs, _ = env.reset(seed=42)
    while True:
        decision = decide(env.machine, env.message, env.history, "rule", env.runtime, env.context)
        obs, _, terminal, truncated, info = env.step(AgentAction(decision["selected_action"]))
        assert env.observation_space.contains(obs)
        if terminal or truncated:
            break
    assert terminal and not truncated
    assert info["task_success"]
    assert info["questions_answered"] == 5
    assert env.machine.state.post_process.user_decision == "declined"
    assert env.machine.state.mock_outbox == []
    assert env.trajectory[-1]["terminal_reply"]
    assert not any(obs["action_mask"])


def test_customer_feature_contract_is_repeatable_and_30_dimensional():
    one, two = CustomerPolicyTrainingEnv(), CustomerPolicyTrainingEnv()
    obs1, _ = one.reset(seed=42)
    obs2, _ = two.reset(seed=42)
    features = StateFeaturizer(max_turns=20)
    np.testing.assert_array_equal(features.featurize(obs1), features.featurize(obs2))
    assert features.featurize(obs1).shape == (30,)


def test_direct_policy_handoff_has_only_one_spoken_acknowledgement():
    scenario = next(s for s in scenarios() if s.name == "ava-no-record")
    env = CustomerPolicyTrainingEnv(scenario)
    env.reset(seed=42)
    while True:
        decision = decide(env.machine, env.message, env.history, "rule", env.runtime, env.context)
        _, _, terminal, truncated, info = env.step(AgentAction(decision["selected_action"]))
        if terminal or truncated:
            break
    assert env.trajectory[-1]["action"] == "ESCALATE_HUMAN"
    assert env.trajectory[-1]["terminal_reply"] == ""
    assert env.history[-1]["content"] == env.trajectory[-1]["reply"]
    assert sum(row["role"] == "assistant" for row in env.history) == len(env.trajectory)


def test_independent_customer_trainer_changes_weights_and_saves_versioned_artifact(tmp_path):
    import torch
    from backend.rl.ppo_trainer import PPOConfig
    from train_customer_policy import CustomerPPOTrainer
    torch.set_num_threads(1)
    trainer = CustomerPPOTrainer(PPOConfig(total_timesteps=128, max_turns=20, seed=42, device="cpu"))
    before = {key: value.clone() for key, value in trainer.policy.state_dict().items()}
    trainer.train(128)
    assert any(not torch.equal(before[key], value) for key, value in trainer.policy.state_dict().items())
    path = tmp_path / "policy.pt"
    trainer.save_checkpoint(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["environment_version"] == 4
    assert saved["feature_version"] == 2
    assert saved["checkpoint_family"] == "customer_multi_turn_v1"
    assert saved["history"][-1]["global_step"] == 128
    assert saved["source_hashes_at_start"]
