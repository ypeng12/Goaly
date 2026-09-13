"""Masked PPO Trainer for AgentPolicyEnv with formal GAE and anti-hacking metrics."""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import math
import random
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..harness.types import AGENT_ACTIONS, ACTION_SPACE_SIZE, AgentAction, Phase
from ..harness.rl_env import AgentPolicyEnv
from ..harness.caller_sim import (
    make_all_profiles,
    make_train_profiles,
    make_val_profiles,
    make_test_profiles,
    CallerProfile,
)
from .featurizer import StateFeaturizer
from .models import ActorCriticPolicy


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    total_timesteps: int = 50000
    max_turns: int = 15
    device: str = "auto"
    seed: int = 42


class RolloutBuffer:
    def __init__(self, size: int, state_dim: int, action_dim: int, device: torch.device):
        self.size = size
        self.device = device
        self.states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(size, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(size, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)
        self.masks = torch.zeros((size, action_dim), dtype=torch.bool, device=device)
        self.advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr = 0

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
        mask: torch.Tensor,
    ):
        if self.ptr >= self.size:
            raise IndexError("Buffer overflow")
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = 1.0 if done else 0.0
        self.values[self.ptr] = value
        self.masks[self.ptr] = mask
        self.ptr += 1

    def compute_gae_and_returns(self, next_value: torch.Tensor, next_done: bool, gamma: float, gae_lambda: float):
        """Compute GAE advantages and returns without off-by-one errors."""
        last_gae = 0.0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - float(next_done)
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value_t = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value_t * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def reset(self):
        self.ptr = 0


class PPOTrainer:
    """End-to-end PPO training loop for AgentPolicyEnv."""

    def __init__(self, config: Optional[PPOConfig] = None):
        self.cfg = config or PPOConfig()
        self._set_seed(self.cfg.seed)

        # Select device: mps -> cpu
        if self.cfg.device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.cfg.device)

        self.featurizer = StateFeaturizer(max_turns=self.cfg.max_turns)
        self.policy = ActorCriticPolicy(
            state_dim=self.featurizer.FEATURE_DIM,
            action_dim=ACTION_SPACE_SIZE,
            hidden_dim=64,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.buffer = RolloutBuffer(
            size=self.cfg.rollout_steps,
            state_dim=self.featurizer.FEATURE_DIM,
            action_dim=ACTION_SPACE_SIZE,
            device=self.device,
        )

        self.history: List[Dict[str, Any]] = []

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _create_env(self, split: str = "train") -> AgentPolicyEnv:
        if split == "train":
            profiles = make_train_profiles()
        elif split == "val":
            profiles = make_val_profiles()
        elif split == "test":
            profiles = make_test_profiles()
        else:
            profiles = make_all_profiles()

        profile = random.choice(profiles)
        return AgentPolicyEnv(caller_profile=profile, max_turns=self.cfg.max_turns)

    def train(self, total_timesteps: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run the complete PPO training loop and return training logs."""
        target_timesteps = total_timesteps or self.cfg.total_timesteps
        num_updates = math.ceil(target_timesteps / self.cfg.rollout_steps)
        actual_timesteps = num_updates * self.cfg.rollout_steps

        env = self._create_env(split="train")
        obs, info = env.reset()
        current_state = self.featurizer.featurize_tensor(obs, turn=0, device=self.device)
        current_mask = self.featurizer.extract_mask_tensor(obs, device=self.device)
        last_done = False

        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        episode_violations: List[int] = []
        episode_successes: List[bool] = []
        episode_premature: List[bool] = []

        current_ep_reward = 0.0
        current_ep_len = 0
        current_ep_violations = 0

        global_step = 0
        start_time = time.time()

        print(f"Starting PPO Training on device: {self.device}")
        print(f"Requested timesteps: {target_timesteps}, Actual timesteps: {actual_timesteps} ({num_updates} updates)")

        for update in range(1, num_updates + 1):
            self.buffer.reset()

            # --- 1. Collect Rollout ---
            self.policy.eval()
            with torch.no_grad():
                for _ in range(self.cfg.rollout_steps):
                    global_step += 1
                    current_ep_len += 1

                    action_idx, log_prob, _, value = self.policy.get_action_and_value(
                        current_state, mask=current_mask, deterministic=False
                    )

                    action_enum = AGENT_ACTIONS[action_idx.item()]
                    next_obs, reward, terminated, truncated, step_info = env.step(action_enum)
                    step_done = terminated or truncated
                    last_done = step_done

                    current_ep_reward += reward
                    violations = step_info.get("structural_violations", [])
                    current_ep_violations += len(violations)

                    self.buffer.add(
                        state=current_state,
                        action=action_idx,
                        log_prob=log_prob,
                        reward=reward,
                        done=step_done,
                        value=value,
                        mask=current_mask,
                    )

                    if step_done:
                        episode_rewards.append(current_ep_reward)
                        episode_lengths.append(current_ep_len)
                        episode_violations.append(current_ep_violations)
                        episode_successes.append(step_info.get("task_success", False))
                        episode_premature.append(step_info.get("premature_termination", False))

                        current_ep_reward = 0.0
                        current_ep_len = 0
                        current_ep_violations = 0

                        # Reset with a new randomized training caller profile
                        env = self._create_env(split="train")
                        next_obs, info = env.reset()
                        current_state = self.featurizer.featurize_tensor(next_obs, turn=0, device=self.device)
                        current_mask = self.featurizer.extract_mask_tensor(next_obs, device=self.device)
                    else:
                        current_state = self.featurizer.featurize_tensor(
                            next_obs, turn=env.turn_count, device=self.device
                        )
                        current_mask = self.featurizer.extract_mask_tensor(next_obs, device=self.device)

                # Bootstrap value for GAE
                next_value = self.policy.get_value(current_state)

            # Compute GAE advantages & returns
            self.buffer.compute_gae_and_returns(
                next_value=next_value,
                next_done=last_done,
                gamma=self.cfg.gamma,
                gae_lambda=self.cfg.gae_lambda,
            )

            # --- 2. Optimize Policy & Value Network ---
            self.policy.train()
            b_inds = np.arange(self.cfg.rollout_steps)
            minibatch_size = self.cfg.rollout_steps // self.cfg.num_minibatches

            clip_fracs = []
            pg_losses = []
            v_losses = []
            ent_losses = []
            approx_kls = []
            explained_vars = []

            for epoch in range(self.cfg.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.cfg.rollout_steps, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    mb_states = self.buffer.states[mb_inds]
                    mb_actions = self.buffer.actions[mb_inds]
                    mb_masks = self.buffer.masks[mb_inds]
                    mb_old_log_probs = self.buffer.log_probs[mb_inds]
                    mb_advantages = self.buffer.advantages[mb_inds]
                    mb_returns = self.buffer.returns[mb_inds]

                    # Normalize advantages at batch level
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    _, new_log_prob, entropy, new_value = self.policy.get_action_and_value(
                        mb_states, mask=mb_masks, action=mb_actions
                    )

                    log_ratio = new_log_prob - mb_old_log_probs
                    ratio = torch.exp(log_ratio)

                    with torch.no_grad():
                        clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()
                        clip_fracs.append(clip_frac)

                        # Schulman approx KL: (ratio - 1) - log(ratio)
                        approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                        approx_kls.append(approx_kl)

                        # Explained variance: 1 - Var(y - y_pred) / Var(y)
                        y_pred = new_value.detach().cpu().numpy()
                        y_true = mb_returns.cpu().numpy()
                        var_y = np.var(y_true)
                        ev = 1.0 - (np.var(y_true - y_pred) / (var_y + 1e-8)) if var_y > 1e-8 else 0.0
                        explained_vars.append(float(ev))

                    # Clipped surrogate objective
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef) * mb_advantages
                    pg_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

                    # Entropy bonus
                    entropy_loss = entropy.mean()

                    # Total Loss
                    loss = pg_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()

                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    ent_losses.append(entropy_loss.item())

            # Log metrics
            recent_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0
            recent_len = np.mean(episode_lengths[-20:]) if episode_lengths else 0.0
            recent_violations = np.sum(episode_violations[-20:]) if episode_violations else 0
            recent_success = np.mean(episode_successes[-20:]) * 100.0 if episode_successes else 0.0
            recent_premature = np.mean(episode_premature[-20:]) * 100.0 if episode_premature else 0.0

            metrics = {
                "update": update,
                "global_step": global_step,
                "mean_reward": float(recent_reward),
                "mean_ep_length": float(recent_len),
                "goal_success_rate": float(recent_success),
                "premature_termination_rate": float(recent_premature),
                "total_violations": int(recent_violations),
                "policy_loss": float(np.mean(pg_losses)),
                "value_loss": float(np.mean(v_losses)),
                "policy_entropy": float(np.mean(ent_losses)),
                "clip_fraction": float(np.mean(clip_fracs)),
                "approx_kl": float(np.mean(approx_kls)),
                "explained_variance": float(np.mean(explained_vars)),
                "fps": int(global_step / max(0.01, time.time() - start_time)),
            }
            self.history.append(metrics)

            if update % 10 == 0 or update == num_updates:
                print(
                    f"Update {update:3d}/{num_updates} | "
                    f"Step {global_step:5d} | "
                    f"Reward: {recent_reward:+.2f} | "
                    f"EpLen: {recent_len:.1f} | "
                    f"Success: {recent_success:5.1f}% | "
                    f"Premature: {recent_premature:4.1f}% | "
                    f"Violations: {recent_violations} | "
                    f"KL: {metrics['approx_kl']:.4f} | "
                    f"EV: {metrics['explained_variance']:.2f} | "
                    f"FPS: {metrics['fps']}"
                )

        return self.history

    def evaluate(
        self,
        n_episodes: int = 50,
        deterministic: bool = True,
        max_turns: Optional[int] = None,
        split: str = "all",
    ) -> Dict[str, Any]:
        """Run policy evaluation across fixture profiles."""
        self.policy.eval()
        turns_limit = max_turns or self.cfg.max_turns

        if split == "train":
            profile_fn = make_train_profiles
        elif split == "val":
            profile_fn = make_val_profiles
        elif split == "test":
            profile_fn = make_test_profiles
        else:
            profile_fn = make_all_profiles

        episode_summaries = []

        with torch.no_grad():
            for i in range(n_episodes):
                ep_profiles = profile_fn()
                profile = ep_profiles[i % len(ep_profiles)]
                env = AgentPolicyEnv(caller_profile=profile, max_turns=turns_limit)
                obs, info = env.reset()

                done = False
                cum_reward = 0.0
                all_violations = []

                while not done:
                    state_t = self.featurizer.featurize_tensor(obs, turn=env.turn_count, device=self.device)
                    mask_t = self.featurizer.extract_mask_tensor(obs, device=self.device)

                    action_idx, _, _, _ = self.policy.get_action_and_value(
                        state_t, mask=mask_t, deterministic=deterministic
                    )
                    action_enum = AGENT_ACTIONS[action_idx.item()]

                    obs, reward, terminated, truncated, step_info = env.step(action_enum)
                    cum_reward += reward
                    all_violations.extend(step_info.get("structural_violations", []))
                    done = terminated or truncated

                episode_summaries.append({
                    "cumulative_reward": cum_reward,
                    "turns": env.turn_count,
                    "terminated": terminated,
                    "truncated": truncated,
                    "task_success": step_info.get("task_success", False),
                    "appropriate_escalation": step_info.get("appropriate_escalation", False),
                    "premature_termination": step_info.get("premature_termination", False),
                    "violations": all_violations,
                    "final_phase": obs.get("phase", "unknown"),
                    "verified_fields": list(obs.get("verified_fields", [])),
                })

        terminal_phases = {"CONCLUDED", "ESCALATED"}
        consistency = sum(1 for e in episode_summaries if e["terminated"] and e["final_phase"] in terminal_phases)
        total_term = sum(1 for e in episode_summaries if e["terminated"])
        consistency_rate = (consistency / total_term * 100.0) if total_term > 0 else 100.0

        return {
            "mean_reward": float(np.mean([e["cumulative_reward"] for e in episode_summaries])),
            "mean_turns": float(np.mean([e["turns"] for e in episode_summaries])),
            "goal_success_rate": float(100.0 * sum(1 for e in episode_summaries if e["task_success"]) / n_episodes),
            "appropriate_escalation_rate": float(100.0 * sum(1 for e in episode_summaries if e["appropriate_escalation"]) / n_episodes),
            "premature_termination_rate": float(100.0 * sum(1 for e in episode_summaries if e["premature_termination"]) / n_episodes),
            "termination_rate": float(100.0 * sum(1 for e in episode_summaries if e["terminated"]) / n_episodes),
            "truncation_rate": float(100.0 * sum(1 for e in episode_summaries if e["truncated"]) / n_episodes),
            "violation_rate": float(100.0 * sum(1 for e in episode_summaries if e["violations"]) / n_episodes),
            "mean_verified_fields": float(np.mean([len(e["verified_fields"]) for e in episode_summaries])),
            "terminal_state_consistency": float(consistency_rate),
            "raw_episodes": episode_summaries,
        }

    def save_checkpoint(self, path: str):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.cfg),
                "history": self.history,
            },
            str(destination),
        )
        print(f"PPO checkpoint saved to {destination}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", [])
        print(f"PPO checkpoint loaded from {path}")
