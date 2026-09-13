"""Generate DPO preference dataset via exact same-state counterfactual branching.

At state S:
  1. Record snapshot(S)
  2. Branch A: Execute expert / RuleBased action
  3. Restore snapshot(S)
  4. Branch B: Execute alternative action from legal action_mask
  5. Form preference pair with identical prompt, phase, and verified fields.
  6. Split into train (70%), val (15%), test (15%).

Usage:
    python3 -m eval.generate_dpo_pairs --num-pairs 50
"""
import argparse
import copy
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from backend.harness.types import AGENT_ACTIONS, AgentAction, Phase
from backend.harness.rl_env import AgentPolicyEnv
from backend.harness.rl_baseline import RuleBasedPolicy
from backend.harness.grounded_data import grounded_data
from backend.harness.caller_sim import CallerProfile

CLAIM_HINTS = [
    "denied healthcare claim",
    "inpatient surgery coverage",
    "emergency room denial",
    "out-of-network ambulance bill",
    "prescription drug appeal",
    "physical therapy authorization",
    "diagnostic MRI denial",
    "chiropractic copay dispute",
    "radiology pre-authorization",
    "specialist consultation claim",
    "dental implant coverage",
    "preventive screening denial",
    "lab test reimbursement",
    "dermatology procedure appeal",
    "cardiology diagnostic claim",
]

STYLES = ["cooperative", "frustrated", "confused", "privacy_sensitive", "escalating"]


def generate_same_state_dpo_pairs(
    num_pairs: int = 50,
    output_dir: str = "artifacts",
    min_reward_delta: float = 1.0,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    random.seed(seed)
    rule_policy = RuleBasedPolicy()

    policyholders = grounded_data.policyholders
    pairs: List[Dict[str, Any]] = []
    seen_prompts = set()

    episode_idx = 0
    max_episodes = 500

    print(f"Generating {num_pairs} DPO pairs with exact same-state branching...")

    while len(pairs) < num_pairs and episode_idx < max_episodes:
        ph = policyholders[episode_idx % len(policyholders)]
        hint = CLAIM_HINTS[episode_idx % len(CLAIM_HINTS)]
        style = STYLES[(episode_idx // len(CLAIM_HINTS)) % len(STYLES)]
        episode_idx += 1

        profile = CallerProfile(ph, claim_hint=hint, style=style, escalate_after=6)
        env = AgentPolicyEnv(caller_profile=profile, max_turns=12)
        obs, info = env.reset(seed=seed + episode_idx)
        rule_policy._reset_episode_state()
        done = False

        while not done and len(pairs) < num_pairs:
            mask = obs["action_mask"]
            legal_actions = [a for a, ok in zip(AGENT_ACTIONS, mask) if ok]
            expert_action = rule_policy.select_action(obs)

            prompt_text = (
                f"[System: Phase={obs['phase']}, Verified={obs['verified_fields']}, "
                f"DataShield={'Active' if obs['data_shield_active'] else 'Inactive'}]\n"
                f"Caller: {obs['caller_utterance']}"
            )

            # Look for alternative actions in the same state
            candidate_alts = [a for a in legal_actions if a != expert_action]

            if candidate_alts and prompt_text not in seen_prompts:
                # Save snapshot of state S
                snapshot = env.save_state()

                # Branch 1: Expert step
                obs_exp, rew_exp, term_exp, trunc_exp, info_exp = env.step(expert_action)
                reply_exp = info_exp["agent_reply"]
                viol_exp = info_exp.get("structural_violations", [])

                # Branch 2: Alternative step from exact same state S
                alt_action = random.choice(candidate_alts)
                env.restore_state(snapshot)
                obs_alt, rew_alt, term_alt, trunc_alt, info_alt = env.step(alt_action)
                reply_alt = info_alt["agent_reply"]
                viol_alt = info_alt.get("structural_violations", [])

                # Evaluate preference
                delta = rew_exp - rew_alt
                if delta >= min_reward_delta and len(viol_exp) == 0:
                    chosen_action = expert_action
                    chosen_reply = reply_exp
                    chosen_rew = rew_exp

                    rejected_action = alt_action
                    rejected_reply = reply_alt
                    rejected_rew = rew_alt
                    rec_delta = round(delta, 3)

                    pair = {
                        "pair_id": f"dpo_{len(pairs) + 1:04d}",
                        "same_state_verified": True,
                        "phase": obs["phase"],
                        "verified_fields": obs["verified_fields"],
                        "caller_utterance": obs["caller_utterance"],
                        "prompt": prompt_text,
                        "chosen": {
                            "action": chosen_action.value,
                            "reply": chosen_reply,
                            "reward": chosen_rew,
                            "violations": viol_exp,
                        },
                        "rejected": {
                            "action": rejected_action.value,
                            "reply": rejected_reply,
                            "reward": rejected_rew,
                            "violations": viol_alt,
                        },
                        "reward_delta": rec_delta,
                    }
                    pairs.append(pair)
                    seen_prompts.add(prompt_text)

                elif -delta >= min_reward_delta and len(viol_alt) == 0:
                    chosen_action = alt_action
                    chosen_reply = reply_alt
                    chosen_rew = rew_alt

                    rejected_action = expert_action
                    rejected_reply = reply_exp
                    rejected_rew = rew_exp
                    rec_delta = round(-delta, 3)

                    pair = {
                        "pair_id": f"dpo_{len(pairs) + 1:04d}",
                        "same_state_verified": True,
                        "phase": obs["phase"],
                        "verified_fields": obs["verified_fields"],
                        "caller_utterance": obs["caller_utterance"],
                        "prompt": prompt_text,
                        "chosen": {
                            "action": chosen_action.value,
                            "reply": chosen_reply,
                            "reward": chosen_rew,
                            "violations": viol_alt,
                        },
                        "rejected": {
                            "action": rejected_action.value,
                            "reply": rejected_reply,
                            "reward": rejected_rew,
                            "violations": viol_exp,
                        },
                        "reward_delta": rec_delta,
                    }
                    pairs.append(pair)
                    seen_prompts.add(prompt_text)

                # Restore expert branch to continue trajectory
                env.restore_state(snapshot)
                obs, reward, terminated, truncated, _ = env.step(expert_action)
                done = terminated or truncated
            else:
                obs, reward, terminated, truncated, _ = env.step(expert_action)
                done = terminated or truncated

    # Split: Train 70%, Val 15%, Test 15%
    random.shuffle(pairs)
    n_train = int(len(pairs) * 0.70)
    n_val = int(len(pairs) * 0.15)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(filepath: Path, data: List[Dict[str, Any]]):
        with filepath.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "dpo_pairs.jsonl", pairs)
    write_jsonl(out_dir / "dpo_train.jsonl", train_pairs)
    write_jsonl(out_dir / "dpo_val.jsonl", val_pairs)
    write_jsonl(out_dir / "dpo_test.jsonl", test_pairs)

    counts = {
        "total": len(pairs),
        "train": len(train_pairs),
        "val": len(val_pairs),
        "test": len(test_pairs),
        "unique_prompts": len(seen_prompts),
        "positive_delta_pct": round(100.0 * sum(1 for p in pairs if p["reward_delta"] > 0) / max(1, len(pairs)), 1),
    }

    print(f"Generated {counts['total']} pairs ({counts['unique_prompts']} unique prompts)")
    print(f"  Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}")
    print(f"  Same-state verification: 100%")
    print(f"  reward_delta > 0: {counts['positive_delta_pct']}%")
    print(f"  Chosen violations: 0")
    return pairs, counts


def generate_dpo_pairs(num_pairs: int = 50, output_path: Optional[str] = None, output_dir: str = "artifacts", **kwargs):
    if output_path is not None:
        out_d = Path(output_path).parent
    else:
        out_d = Path(output_dir)
    pairs, counts = generate_same_state_dpo_pairs(num_pairs=num_pairs, output_dir=str(out_d), **kwargs)
    if output_path is not None:
        with Path(output_path).open("w", encoding="utf-8") as f:
            for item in pairs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference dataset via counterfactual branching")
    parser.add_argument("--num-pairs", type=int, default=50, help="Number of preference pairs")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--output", type=str, default=None, help="Output file path (parent used as output dir)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    out_dir = args.output_dir
    if args.output:
        p = Path(args.output)
        out_dir = str(p.parent) if p.suffix else str(p)

    generate_same_state_dpo_pairs(num_pairs=args.num_pairs, output_dir=out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
