"""
Policy comparison: RuleBasedPolicy vs RandomPolicy vs LearnedPPOPolicy on AgentPolicyEnv.

All policies are evaluated on task success, terminal-state consistency,
constraint violations, verified-field completion, and episode efficiency.
Safety is enforced structurally by the harness; reward is used to optimize
task completion within the legal action space.

Run:
    python3 eval/policy_comparison.py
"""
import sys
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from backend.harness.types import AgentAction, AGENT_ACTIONS, Phase
from backend.harness.rl_env import AgentPolicyEnv
from backend.harness.rl_baseline import RuleBasedPolicy
from backend.harness.caller_sim import (
    make_all_profiles,
    make_train_profiles,
    make_val_profiles,
    make_test_profiles,
)
from backend.rl.featurizer import StateFeaturizer
from backend.rl.models import ActorCriticPolicy


class RandomPolicy:
    """Uniformly samples from the legal action_mask each step."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def select_action(self, observation: dict) -> AgentAction:
        mask = observation.get("action_mask", [True] * len(AGENT_ACTIONS))
        legal = [a for a, ok in zip(AGENT_ACTIONS, mask) if ok]
        if not legal:
            raise RuntimeError("No legal actions available")
        return self.rng.choice(legal)

    def run_episode(self, env: AgentPolicyEnv, verbose: bool = False) -> dict:
        obs, info = env.reset()
        done = False
        cumulative_reward = 0.0
        all_violations = []
        terminated = truncated = False
        last_info = {}

        while not done:
            action = self.select_action(obs)
            obs, reward, terminated, truncated, last_info = env.step(action)
            cumulative_reward += reward
            all_violations.extend(last_info.get("structural_violations", []))
            done = terminated or truncated
            if verbose:
                caller = obs.get("caller_utterance", "")[:60]
                print(f"  turn={last_info['turn']:2d}  action={action.value:<28s}  reward={reward:+.2f}  phase={obs['phase']}")
                if caller:
                    print(f"    caller: {caller!r}")

        return {
            "cumulative_reward": cumulative_reward,
            "turns": env.turn_count,
            "terminated": terminated,
            "truncated": truncated,
            "task_success": last_info.get("task_success", False),
            "appropriate_escalation": last_info.get("appropriate_escalation", False),
            "premature_termination": last_info.get("premature_termination", False),
            "violations": all_violations,
            "trajectory": env.trajectory,
            "final_phase": obs.get("phase", "unknown"),
            "verified_fields": list(obs.get("verified_fields", [])),
        }


class LearnedPPOPolicy:
    """Policy driven by a trained ActorCritic neural network."""

    def __init__(self, model_path: str = "artifacts/ppo_policy.pt", device: str = "cpu"):
        self.device = torch.device(device)
        self.featurizer = StateFeaturizer()
        self.policy = ActorCriticPolicy(
            state_dim=self.featurizer.FEATURE_DIM,
            action_dim=len(AGENT_ACTIONS),
            hidden_dim=64,
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy.eval()

    def select_action(self, observation: dict, turn: int = 0) -> AgentAction:
        state_t = self.featurizer.featurize_tensor(observation, turn=turn, device=self.device)
        mask_t = self.featurizer.extract_mask_tensor(observation, device=self.device)
        with torch.no_grad():
            action_idx, _, _, _ = self.policy.get_action_and_value(
                state_t, mask=mask_t, deterministic=True
            )
        return AGENT_ACTIONS[action_idx.item()]

    def run_episode(self, env: AgentPolicyEnv, verbose: bool = False) -> dict:
        obs, info = env.reset()
        done = False
        cumulative_reward = 0.0
        all_violations = []
        terminated = truncated = False
        last_info = {}

        while not done:
            action = self.select_action(obs, turn=env.turn_count)
            obs, reward, terminated, truncated, last_info = env.step(action)
            cumulative_reward += reward
            all_violations.extend(last_info.get("structural_violations", []))
            done = terminated or truncated
            if verbose:
                caller = obs.get("caller_utterance", "")[:60]
                print(f"  turn={last_info['turn']:2d}  action={action.value:<28s}  reward={reward:+.2f}  phase={obs['phase']}")
                if caller:
                    print(f"    caller: {caller!r}")

        return {
            "cumulative_reward": cumulative_reward,
            "turns": env.turn_count,
            "terminated": terminated,
            "truncated": truncated,
            "task_success": last_info.get("task_success", False),
            "appropriate_escalation": last_info.get("appropriate_escalation", False),
            "premature_termination": last_info.get("premature_termination", False),
            "violations": all_violations,
            "trajectory": env.trajectory,
            "final_phase": obs.get("phase", "unknown"),
            "verified_fields": list(obs.get("verified_fields", [])),
        }


def run_agent_comparison(
    n_episodes: int = 50,
    max_turns: int = 15,
    seed: int = 42,
    ppo_model_path: Optional[str] = "artifacts/ppo_policy.pt",
    split: str = "all",
):
    """Compare RuleBasedPolicy vs RandomPolicy vs LearnedPPOPolicy."""
    if split == "train":
        profile_fn = make_train_profiles
    elif split == "val":
        profile_fn = make_val_profiles
    elif split == "test":
        profile_fn = make_test_profiles
    else:
        profile_fn = make_all_profiles

    profiles = profile_fn()
    rule_policy = RuleBasedPolicy()
    rand_policy = RandomPolicy(seed=seed)

    has_ppo = ppo_model_path and Path(ppo_model_path).exists()
    ppo_policy = LearnedPPOPolicy(model_path=ppo_model_path) if has_ppo else None

    policies = [
        ("RuleBased", rule_policy),
        ("Random", rand_policy),
    ]
    if ppo_policy:
        policies.append(("PPO (Learned)", ppo_policy))

    results_dict = {name: [] for name, _ in policies}

    for i in range(n_episodes):
        profile_idx = i % len(profiles)

        for name, pol in policies:
            ep_profiles = profile_fn()
            profile = ep_profiles[profile_idx]
            env = AgentPolicyEnv(caller_profile=profile, max_turns=max_turns)
            results_dict[name].append(pol.run_episode(env))

    def pct(results, pred):
        return 100.0 * sum(1 for r in results if pred(r)) / len(results)

    def avg(results, key):
        return sum(r[key] for r in results) / len(results)

    header_cols = "".join(f"{name:>16}" for name, _ in policies)
    line_len = 38 + 16 * len(policies)
    print(f"\n{'=' * line_len}")
    print(f"AgentPolicyEnv: Multi-Policy Benchmark Arena ({split} split)")
    print(f"({n_episodes} episodes each, {len(profiles)} caller profiles, max_turns={max_turns})")
    print(f"{'=' * line_len}")
    print(f"{'Metric':<38}{header_cols}")
    print("-" * line_len)

    terminal_phases = {"CONCLUDED", "ESCALATED"}
    metric_defs = [
        ("Goal success rate (%)", lambda res: f"{pct(res, lambda r: r['task_success']):>16.2f}"),
        ("Appropriate escalation rate (%)", lambda res: f"{pct(res, lambda r: r['appropriate_escalation']):>16.2f}"),
        ("Premature termination rate (%)", lambda res: f"{pct(res, lambda r: r['premature_termination']):>16.2f}"),
        ("Termination rate — clean end (%)", lambda res: f"{pct(res, lambda r: r['terminated']):>16.2f}"),
        ("Truncation rate — hit max_turns (%)", lambda res: f"{pct(res, lambda r: r['truncated']):>16.2f}"),
        ("Terminal consistency (%)", lambda res: f"{pct(res, lambda r: r['terminated'] and r['final_phase'] in terminal_phases):>16.2f}"),
        ("Constraint violation rate (%)", lambda res: f"{pct(res, lambda r: bool(r['violations'])):>16.2f}"),
        ("Mean verified fields collected", lambda res: f"{(sum(len(r['verified_fields']) for r in res) / len(res)):>16.2f}"),
        ("Mean episode length (turns)", lambda res: f"{avg(res, 'turns'):>16.2f}"),
        ("Mean cumulative reward", lambda res: f"{avg(res, 'cumulative_reward'):>16.2f}"),
    ]

    for label, fn in metric_defs:
        vals = "".join(fn(results_dict[name]) for name, _ in policies)
        print(f"  {label:<36}{vals}")

    print("=" * line_len)

    # Reward distribution
    print("\nReward distribution:")
    for name, _ in policies:
        rewards = sorted(r["cumulative_reward"] for r in results_dict[name])
        print(f"  {name}:")
        print(
            f"    min={rewards[0]:.2f}  p25={rewards[len(rewards)//4]:.2f}"
            f"  median={statistics.median(rewards):.2f}"
            f"  p75={rewards[3*len(rewards)//4]:.2f}  max={rewards[-1]:.2f}"
        )

    # Final phase breakdown
    print("\nFinal phase breakdown:")
    for name, _ in policies:
        counts = Counter(r["final_phase"] for r in results_dict[name])
        phases_str = "  ".join(f"{p}={c}" for p, c in sorted(counts.items()))
        print(f"  {name}: {phases_str}")

    return results_dict


if __name__ == "__main__":
    run_agent_comparison(n_episodes=50, max_turns=15)
