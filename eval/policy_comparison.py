"""
Policy comparison: RuleBasedPolicy vs RandomPolicy vs LearnedPPOPolicy on AgentPolicyEnv.

All policies are evaluated on task success, terminal-state consistency,
constraint violations, verified-field completion, and episode efficiency.
Safety is enforced structurally by the harness; reward is used to optimize
task completion within the legal action space.

Run:
    python3 eval/policy_comparison.py
"""
import argparse
import json
import hashlib
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


TERMINAL_PHASES = {"CONCLUDED", "ESCALATED"}
PROFILE_FACTORIES = {
    "all": make_all_profiles,
    "train": make_train_profiles,
    "val": make_val_profiles,
    "test": make_test_profiles,
}


def get_profile_factory(split: str):
    try:
        return PROFILE_FACTORIES[split]
    except KeyError as exc:
        raise ValueError(f"Unknown split {split!r}; choose from {sorted(PROFILE_FACTORIES)}") from exc


def summarize_policy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return stable, JSON-serializable policy metrics."""
    count = len(results)
    if count == 0:
        raise ValueError("Cannot summarize an empty policy result set")

    def pct(predicate):
        return round(100.0 * sum(1 for row in results if predicate(row)) / count, 2)

    final_phases = Counter(row["final_phase"] for row in results)
    return {
        "episodes": count,
        "support_outcome_rate": pct(lambda row: row['task_success'] or row['appropriate_escalation']),
        "unverified_success_rate": pct(lambda row: row['task_success'] and len(set(row['verified_fields'])) < 3),
        "goal_success_rate": pct(lambda row: row["task_success"]),
        "appropriate_escalation_rate": pct(lambda row: row["appropriate_escalation"]),
        "premature_termination_rate": pct(lambda row: row["premature_termination"]),
        "clean_termination_rate": pct(lambda row: row["terminated"]),
        "truncation_rate": pct(lambda row: row["truncated"]),
        "terminal_consistency_rate": pct(
            lambda row: bool(row["terminated"]) == (row["final_phase"] in TERMINAL_PHASES)
            and not (row["terminated"] and row["truncated"])
        ),
        "constraint_violation_rate": pct(lambda row: bool(row["violations"])),
        "mean_verified_fields": round(
            sum(len(row["verified_fields"]) for row in results) / count, 3
        ),
        "mean_episode_turns": round(sum(row["turns"] for row in results) / count, 3),
        "mean_cumulative_reward": round(
            sum(row["cumulative_reward"] for row in results) / count, 3
        ),
        "final_phase_distribution": dict(sorted(final_phases.items())),
    }


def _action_sequence(result: Dict[str, Any]) -> List[str]:
    return [turn["agent_action"] for turn in result["trajectory"]]


def build_report(
    results_dict: Dict[str, List[Dict[str, Any]]],
    *,
    split: str,
    n_episodes: int,
    max_turns: int,
    seed: int,
    profile_count: int,
) -> Dict[str, Any]:
    """Build an auditable report and acceptance decision from arena results."""
    policies = {name: summarize_policy(rows) for name, rows in results_dict.items()}
    report: Dict[str, Any] = {
        "schema_version": 2,
        "config": {
            "split": split,
            "episodes_per_policy": n_episodes,
            "profile_templates": profile_count,
            "max_turns": max_turns,
            "seed": seed,
        },
        "policies": policies,
        "limitations": "Finite synthetic profile templates are repeated; episodes are not independent human calls. Test holds out identity/style/scenario combinations, not identities or all styles.",
    }
    report['by_profile'] = {}
    for name, rows in results_dict.items():
        report['by_profile'][name] = {
            key: summarize_policy([r for r in rows if r.get('profile_id') == key])
            for key in sorted({r['profile_id'] for r in rows if 'profile_id' in r})
        }

    if "PPO (Learned)" in results_dict:
        rule_rows = results_dict["RuleBased"]
        ppo_rows = results_dict["PPO (Learned)"]
        paired = min(len(rule_rows), len(ppo_rows))
        outcome_matches = sum(
            rule_rows[i]["final_phase"] == ppo_rows[i]["final_phase"]
            for i in range(paired)
        )
        action_matches = sum(
            _action_sequence(rule_rows[i]) == _action_sequence(ppo_rows[i])
            for i in range(paired)
        )
        report["ppo_vs_rule"] = {
            "terminal_outcome_agreement_rate": round(100.0 * outcome_matches / paired, 2),
            "exact_action_sequence_agreement_rate": round(100.0 * action_matches / paired, 2),
        }

        ppo = policies["PPO (Learned)"]
        random_policy = policies["Random"]
        checks = {
            "terminal_consistency_is_100": ppo["terminal_consistency_rate"] == 100.0,
            "constraint_violation_is_0": ppo["constraint_violation_rate"] == 0.0,
            "premature_termination_is_0": ppo["premature_termination_rate"] == 0.0,
            "truncation_is_0": ppo["truncation_rate"] == 0.0,
            "completed_cases_verified": ppo['unverified_success_rate'] == 0.0,
            "support_outcomes_match_rule": ppo['support_outcome_rate'] >= policies['RuleBased']['support_outcome_rate'],
            "case_success_matches_rule": ppo['goal_success_rate'] >= policies['RuleBased']['goal_success_rate'],
            "reward_beats_random": (
                ppo["mean_cumulative_reward"] > random_policy["mean_cumulative_reward"]
            ),
        }
        report["acceptance"] = {"passed": all(checks.values()), "checks": checks}
    else:
        report["acceptance"] = {
            "passed": False,
            "checks": {"learned_checkpoint_loaded": False},
        }

    return report


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

    def run_episode(self, env: AgentPolicyEnv, verbose: bool = False, seed=None) -> dict:
        obs, info = env.reset(seed=seed)
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
        checkpoint = torch.load(model_path, map_location=self.device)
        if checkpoint.get('feature_version') != StateFeaturizer.VERSION or checkpoint.get('environment_version') != AgentPolicyEnv.ENV_VERSION:
            raise ValueError('Checkpoint schema/environment is incompatible. Run train_ppo.py to retrain.')
        self.featurizer = StateFeaturizer(max_turns=checkpoint['config']['max_turns'],
                                         use_emotion_features=checkpoint['config'].get('use_emotion_features', True))
        self.policy = ActorCriticPolicy(
            state_dim=self.featurizer.FEATURE_DIM,
            action_dim=len(AGENT_ACTIONS),
            hidden_dim=64,
        ).to(self.device)

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

    def run_episode(self, env: AgentPolicyEnv, verbose: bool = False, seed=None) -> dict:
        obs, info = env.reset(seed=seed)
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
    profile_fn = get_profile_factory(split)

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
            result = pol.run_episode(env, seed=seed + i)
            result['profile_id'] = f'{profile.ph.party_id}:{profile.style}:{profile.claim_hint}'
            result['episode_seed'] = seed + i
            results_dict[name].append(result)

    def pct(results, pred):
        return 100.0 * sum(1 for r in results if pred(r)) / len(results)

    def avg(results, key):
        return sum(r[key] for r in results) / len(results)

    header_cols = "".join(f"{name:>16}" for name, _ in policies)
    line_len = 38 + 16 * len(policies)
    print(f"\n{'=' * line_len}")
    print(f"AgentPolicyEnv: Multi-Policy Benchmark Arena ({split} split)")
    print(f"({n_episodes} episodes each, {len(profiles)} profile templates, max_turns={max_turns})")
    print(f"{'=' * line_len}")
    print(f"{'Metric':<38}{header_cols}")
    print("-" * line_len)

    metric_defs = [
        ("Goal success rate (%)", lambda res: f"{pct(res, lambda r: r['task_success']):>16.2f}"),
        ("Appropriate escalation rate (%)", lambda res: f"{pct(res, lambda r: r['appropriate_escalation']):>16.2f}"),
        ("Premature termination rate (%)", lambda res: f"{pct(res, lambda r: r['premature_termination']):>16.2f}"),
        ("Termination rate — clean end (%)", lambda res: f"{pct(res, lambda r: r['terminated']):>16.2f}"),
        ("Truncation rate — hit max_turns (%)", lambda res: f"{pct(res, lambda r: r['truncated']):>16.2f}"),
        ("Terminal consistency (%)", lambda res: f"{summarize_policy(res)['terminal_consistency_rate']:>16.2f}"),
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


def parse_args():
    parser = argparse.ArgumentParser(description="Compare constrained agent policies")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per policy")
    parser.add_argument("--max-turns", type=int, default=15, help="Maximum turns per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random-policy seed")
    parser.add_argument(
        "--split", choices=sorted(PROFILE_FACTORIES), default="all",
        help="Caller profile split to evaluate",
    )
    parser.add_argument(
        "--model-path", default="artifacts/ppo_policy.pt",
        help="Learned PPO checkpoint path",
    )
    parser.add_argument(
        "--json-output", default=None,
        help="Optional machine-readable report path",
    )
    parser.add_argument(
        "--assert-thresholds", action="store_true",
        help="Exit non-zero when PPO acceptance checks fail",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.max_turns < 1:
        parser.error("--episodes and --max-turns must be positive")
    return args


def main():
    args = parse_args()
    results = run_agent_comparison(
        n_episodes=args.episodes,
        max_turns=args.max_turns,
        seed=args.seed,
        ppo_model_path=args.model_path,
        split=args.split,
    )
    profile_count = len(get_profile_factory(args.split)())
    report = build_report(
        results,
        split=args.split,
        n_episodes=args.episodes,
        max_turns=args.max_turns,
        seed=args.seed,
        profile_count=profile_count,
    )
    if Path(args.model_path).is_file():
        checkpoint = torch.load(args.model_path, map_location='cpu')
        report['checkpoint'] = {
            'sha256': hashlib.sha256(Path(args.model_path).read_bytes()).hexdigest(),
            'feature_version': checkpoint['feature_version'],
            'environment_version': checkpoint['environment_version'],
            'training_seed': checkpoint['config']['seed'],
            'actual_steps': checkpoint['history'][-1]['global_step'],
        }

    comparison = report.get("ppo_vs_rule")
    if comparison:
        print("\nPPO vs RuleBased agreement:")
        print(
            "  Terminal outcome: "
            f"{comparison['terminal_outcome_agreement_rate']:.2f}%"
        )
        print(
            "  Exact action sequence: "
            f"{comparison['exact_action_sequence_agreement_rate']:.2f}%"
        )

    print(
        "Acceptance: "
        + ("PASS" if report["acceptance"]["passed"] else "FAIL")
    )
    for name, passed in report["acceptance"]["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    if args.json_output:
        destination = Path(args.json_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Machine-readable report: {destination}")

    if args.assert_thresholds and not report["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
