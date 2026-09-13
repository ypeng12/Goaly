"""Unit tests for Masked PPO training, featurizer, and DPO generation."""
import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np

from backend.harness.types import Phase, AGENT_ACTIONS, ACTION_SPACE_SIZE
from backend.harness.rl_env import AgentPolicyEnv
from backend.rl.featurizer import StateFeaturizer
from backend.rl.models import ActorCriticPolicy
from backend.rl.ppo_trainer import PPOTrainer, PPOConfig, RolloutBuffer
from eval.generate_dpo_pairs import generate_dpo_pairs


def test_featurizer_dimensions():
    featurizer = StateFeaturizer(max_turns=15)
    obs = {
        "phase": Phase.VERIFY_ID.value,
        "verified_fields": ["name", "dob"],
        "memory_slots": {"case_type_hint": "denied claim"},
        "data_shield_active": True,
        "active_case_id": None,
        "action_mask": [True, True, False, False, False, False, False, False, True],
    }

    vec = featurizer.featurize(obs, turn=3)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (20,)
    assert vec.dtype == np.float32

    # Phase VERIFY_ID is first
    assert vec[0] == 1.0
    # Verified fields: name (index 6) and dob (index 7)
    assert vec[6] == 1.0
    assert vec[7] == 1.0
    assert vec[8] == 0.0
    # Verified count ratio (2 / 5 = 0.4)
    assert abs(vec[11] - 0.4) < 1e-5
    # Data shield active
    assert vec[17] == 1.0
    # Active case id
    assert vec[18] == 0.0
    # Turn progress (3 / 15 = 0.2)
    assert abs(vec[19] - 0.2) < 1e-5


def test_model_action_masking():
    torch.manual_seed(42)
    policy = ActorCriticPolicy(state_dim=20, action_dim=ACTION_SPACE_SIZE, hidden_dim=32)
    state = torch.randn(1, 20)

    # Mask out all actions except index 1 and 8
    mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[0, 1] = True
    mask[0, 8] = True

    # Sample 50 actions; all MUST be either 1 or 8
    for _ in range(50):
        action, log_prob, entropy, value = policy.get_action_and_value(state, mask=mask)
        act_idx = action.item()
        assert act_idx in {1, 8}, f"Sampled masked action: {act_idx}"
        assert not torch.isnan(log_prob)
        assert not torch.isinf(log_prob)


def test_rollout_buffer_gae():
    device = torch.device("cpu")
    buffer = RolloutBuffer(size=4, state_dim=20, action_dim=9, device=device)

    for i in range(4):
        state = torch.randn(20)
        buffer.add(
            state=state,
            action=torch.tensor(0),
            log_prob=torch.tensor(-0.5),
            reward=1.0,
            done=(i == 3),
            value=torch.tensor(0.5),
            mask=torch.ones(9, dtype=torch.bool),
        )

    next_value = torch.tensor(0.0)
    buffer.compute_gae_and_returns(next_value=next_value, next_done=True, gamma=0.99, gae_lambda=0.95)

    assert buffer.advantages.shape == (4,)
    assert buffer.returns.shape == (4,)
    assert not torch.isnan(buffer.advantages).any()
    assert not torch.isnan(buffer.returns).any()


def test_ppo_short_training_and_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = str(Path(tmpdir) / "test_ppo.pt")

        cfg = PPOConfig(
            rollout_steps=32,
            total_timesteps=64,
            device="cpu",
            num_minibatches=2,
            update_epochs=2,
            seed=42,
        )
        trainer = PPOTrainer(config=cfg)
        history = trainer.train(total_timesteps=64)

        assert len(history) == 2  # 64 / 32 = 2 updates
        assert "mean_reward" in history[0]

        trainer.save_checkpoint(ckpt_path)
        assert Path(ckpt_path).exists()

        # Test reload
        new_trainer = PPOTrainer(config=cfg)
        new_trainer.load_checkpoint(ckpt_path)
        assert len(new_trainer.history) == 2


def test_dpo_pairs_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = str(Path(tmpdir) / "dpo_test.jsonl")
        pairs = generate_dpo_pairs(num_pairs=5, output_path=output_file)

        assert len(pairs) >= 5
        assert Path(output_file).exists()

        first = pairs[0]
        assert "prompt" in first
        assert "chosen" in first
        assert "rejected" in first
        assert "reward_delta" in first
        assert "action" in first["chosen"]
        assert "action" in first["rejected"]
