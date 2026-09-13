"""Generate DPO (Direct Preference Optimization) preference dataset from AgentPolicyEnv trajectories.

Pairs successful, SOP-compliant turns (chosen) with suboptimal or exploring turns (rejected)
across identical caller profiles and phases.

Usage:
    python3 -m eval.generate_dpo_pairs --num-pairs 50 --output artifacts/dpo_pairs.jsonl
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

from backend.harness.rl_env import AgentPolicyEnv, AGENT_ACTIONS, AgentAction
from backend.harness.rl_baseline import RuleBasedPolicy
from backend.harness.caller_sim import make_all_profiles
from eval.policy_comparison import RandomPolicy


def generate_dpo_pairs(num_pairs: int = 50, output_path: str = "artifacts/dpo_pairs.jsonl") -> List[Dict[str, Any]]:
    rule_policy = RuleBasedPolicy()
    rand_policy = RandomPolicy(seed=42)

    profiles = make_all_profiles()
    pairs = []

    print(f"Generating {num_pairs} DPO preference pairs...")

    episode_idx = 0
    while len(pairs) < num_pairs:
        profile_maker = profiles[episode_idx % len(profiles)]
        episode_idx += 1

        # Run expert / rule-based episode
        env_expert = AgentPolicyEnv(caller_profile=profile_maker, max_turns=12)
        res_expert = rule_policy.run_episode(env_expert)

        # Run suboptimal / random episode on fresh instance of same profile
        # Create fresh profile of same identity
        fresh_profiles = make_all_profiles()
        matching_profile = fresh_profiles[(episode_idx - 1) % len(fresh_profiles)]
        env_rand = AgentPolicyEnv(caller_profile=matching_profile, max_turns=12)
        res_rand = rand_policy.run_episode(env_rand)

        expert_traj = res_expert["trajectory"]
        rand_traj = res_rand["trajectory"]

        # Pair turns by turn index or phase
        min_turns = min(len(expert_traj), len(rand_traj))
        for t in range(min_turns):
            turn_exp = expert_traj[t]
            turn_rnd = rand_traj[t]

            # We form a preference pair if:
            # 1. Actions differ
            # 2. Expert reward >= Random reward, or Random had violations/truncation
            if turn_exp["agent_action"] != turn_rnd["agent_action"]:
                rew_exp = turn_exp["reward"]
                rew_rnd = turn_rnd["reward"]

                # Prefer higher reward or expert SOP action
                if rew_exp >= rew_rnd:
                    pair = {
                        "pair_id": f"dpo_{len(pairs) + 1:04d}",
                        "phase": turn_exp["observation"]["phase"],
                        "caller_utterance": turn_exp["caller_utterance"],
                        "context": {
                            "turn": t + 1,
                            "phase": turn_exp["observation"]["phase"],
                            "verified_fields": turn_exp["observation"]["verified_fields"],
                            "data_shield_active": turn_exp["observation"]["data_shield_active"],
                        },
                        "prompt": (
                            f"[System: SOP Phase={turn_exp['observation']['phase']}, "
                            f"Verified={turn_exp['observation']['verified_fields']}]\n"
                            f"Caller: {turn_exp['caller_utterance']}"
                        ),
                        "chosen": {
                            "action": turn_exp["agent_action"],
                            "reply": turn_exp["agent_reply"],
                            "step_reward": rew_exp,
                        },
                        "rejected": {
                            "action": turn_rnd["agent_action"],
                            "reply": turn_rnd["agent_reply"],
                            "step_reward": rew_rnd,
                        },
                        "reward_delta": round(rew_exp - rew_rnd, 3),
                    }
                    pairs.append(pair)
                    if len(pairs) >= num_pairs:
                        break

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Successfully wrote {len(pairs)} preference pairs to {dest}")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference dataset")
    parser.add_argument("--num-pairs", type=int, default=50, help="Number of preference pairs to generate")
    parser.add_argument("--output", type=str, default="artifacts/dpo_pairs.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    generate_dpo_pairs(num_pairs=args.num_pairs, output_path=args.output)


if __name__ == "__main__":
    main()
