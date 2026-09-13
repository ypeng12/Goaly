"""
Policy comparison: RuleBasedPolicy vs RandomPolicy on AgentPolicyEnv.

Both policies are tested on the same AgentPolicyEnv + CallerProfile setup.
The CallerProfile provides real PII utterances to drive the SOP state machine.
The agent selects actions; rewards reflect PII field collection, phase transitions,
and structural violations.

Run:
    python3 eval/policy_comparison.py
"""
import sys
import random
import statistics
from collections import Counter

sys.path.insert(0, "/Users/yuliangpeng/Desktop/apps/insurance_claims")

from backend.harness.rl_env import AgentPolicyEnv, AgentAction, AGENT_ACTIONS
from backend.harness.rl_baseline import RuleBasedPolicy
from backend.harness.caller_sim import make_margaret_chen_profile, make_all_profiles


class RandomPolicy:
    """Uniformly samples from the legal action_mask each step.

    This is the simplest possible agent: it never selects masked actions,
    but otherwise has no preference over the legal action set.
    """
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def select_action(self, observation: dict) -> AgentAction:
        mask = observation.get("action_mask", [True] * len(AGENT_ACTIONS))
        legal = [a for a, ok in zip(AGENT_ACTIONS, mask) if ok]
        if not legal:
            raise RuntimeError("No legal actions available")
        return self.rng.choice(legal)

    def run_episode(self, env, verbose: bool = False) -> dict:
        obs, info = env.reset()
        done = False
        cumulative_reward = 0.0
        all_violations = []
        terminated = truncated = False
        while not done:
            action = self.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            all_violations.extend(info.get("structural_violations", []))
            done = terminated or truncated
            if verbose:
                caller = obs.get("caller_utterance", "")[:60]
                print(f"  turn={info['turn']:2d}  action={action.value:<28s}  reward={reward:+.2f}  phase={obs['phase']}")
                if caller:
                    print(f"    caller: {caller!r}")
        return {
            "cumulative_reward": cumulative_reward,
            "turns": env.turn_count,
            "terminated": terminated,
            "truncated": truncated,
            "violations": all_violations,
            "final_phase": obs.get("phase", "unknown"),
            "verified_fields": list(obs.get("verified_fields", [])),
        }



def run_agent_comparison(n_episodes: int = 50, max_turns: int = 15, seed: int = 42):
    """Compare RuleBasedPolicy vs RandomPolicy on AgentPolicyEnv + CallerProfile.

    Both policies use the same set of CallerProfiles (all fixture policyholders,
    cycled across episodes) so the comparison is controlled for caller quality.
    """
    profiles = make_all_profiles()
    rule_policy = RuleBasedPolicy()
    rand_policy = RandomPolicy(seed=seed)

    rule_results = []
    rand_results = []

    for i in range(n_episodes):
        # Create fresh profiles for each episode (CallerProfile tracks internal state)
        profiles_r = make_all_profiles()
        profiles_rnd = make_all_profiles()
        profile_idx = i % len(profiles_r)

        # RuleBasedPolicy episode
        env = AgentPolicyEnv(caller_profile=profiles_r[profile_idx], max_turns=max_turns)
        rule_results.append(rule_policy.run_episode(env))

        # RandomPolicy episode (same profile type, fresh instance)
        env2 = AgentPolicyEnv(caller_profile=profiles_rnd[profile_idx], max_turns=max_turns)
        rand_results.append(rand_policy.run_episode(env2))

    def pct(results, pred):
        return 100 * sum(1 for r in results if pred(r)) / len(results)

    def avg(results, key):
        return sum(r[key] for r in results) / len(results)

    print(f"\n{'='*72}")
    print(f"AgentPolicyEnv: RuleBasedPolicy vs RandomPolicy")
    print(f"({n_episodes} episodes each, {len(profiles)} caller profiles, max_turns={max_turns})")
    print(f"{'='*72}")
    print(f"{'Metric':<40} {'RuleBased':>13} {'Random':>13}")
    print("-" * 72)

    metrics = [
        ("Mean cumulative reward",
         avg(rule_results, "cumulative_reward"),
         avg(rand_results, "cumulative_reward")),
        ("Mean episode length (turns)",
         avg(rule_results, "turns"),
         avg(rand_results, "turns")),
        ("Termination rate — clean end (%)",
         pct(rule_results, lambda r: r["terminated"]),
         pct(rand_results, lambda r: r["terminated"])),
        ("Truncation rate — hit max_turns (%)",
         pct(rule_results, lambda r: r["truncated"]),
         pct(rand_results, lambda r: r["truncated"])),
        ("Violation rate — any violation (%)",
         pct(rule_results, lambda r: bool(r["violations"])),
         pct(rand_results, lambda r: bool(r["violations"]))),
        ("Mean verified fields collected",
         sum(len(r["verified_fields"]) for r in rule_results) / len(rule_results),
         sum(len(r["verified_fields"]) for r in rand_results) / len(rand_results)),
    ]

    for label, r_val, rnd_val in metrics:
        print(f"  {label:<38} {r_val:>13.2f} {rnd_val:>13.2f}")

    print("=" * 72)

    # Reward distribution
    print("\nReward distribution:")
    for name, results in [("RuleBasedPolicy", rule_results), ("RandomPolicy", rand_results)]:
        rewards = sorted(r["cumulative_reward"] for r in results)
        print(f"  {name}:")
        print(f"    min={rewards[0]:.2f}  p25={rewards[len(rewards)//4]:.2f}"
              f"  median={statistics.median(rewards):.2f}"
              f"  p75={rewards[3*len(rewards)//4]:.2f}  max={rewards[-1]:.2f}")

    # Final phase breakdown
    print("\nFinal phase breakdown:")
    for name, results in [("RuleBasedPolicy", rule_results), ("RandomPolicy", rand_results)]:
        counts = Counter(r["final_phase"] for r in results)
        phases_str = "  ".join(f"{p}={c}" for p, c in sorted(counts.items()))
        print(f"  {name}: {phases_str}")

    return rule_results, rand_results


if __name__ == "__main__":
    run_agent_comparison(n_episodes=50, max_turns=15)
