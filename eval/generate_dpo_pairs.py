"""Same-state, full-continuation text preference generation (no DPO training)."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from backend.harness.types import AGENT_ACTIONS
from backend.harness.rl_env import AgentPolicyEnv
from backend.harness.rl_baseline import RuleBasedPolicy
from backend.harness.grounded_data import grounded_data
from backend.harness.caller_sim import CallerProfile

STYLES = ["cooperative", "frustrated", "confused", "privacy_sensitive", "escalating"]


def _json_default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def snapshot_digest(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=_json_default).encode()).hexdigest()


def branch_return(env, snapshot, action, gamma=0.99):
    """Take action a at exactly S, then finish with a common continuation policy."""
    env.restore_state(snapshot)
    digest = snapshot_digest(env.save_state())
    obs, reward, terminated, truncated, info = env.step(action)
    first_reply = info["agent_reply"]
    total, discount = reward, gamma
    violations = list(info["structural_violations"])
    policy = RuleBasedPolicy()
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(policy.select_action(obs))
        total += discount * reward
        discount *= gamma
        violations.extend(info["structural_violations"])
    return {"state_hash": digest, "reply": first_reply, "return": total,
            "violations": violations, "final_phase": obs["phase"]}


def split_pairs(pairs):
    """Keep each identity/style family entirely within one split."""
    groups = {}
    for pair in pairs:
        groups.setdefault(pair["group_id"], []).append(pair)
    ordered = sorted(groups, key=lambda key: hashlib.sha256(key.encode()).hexdigest())
    result = {"train": [], "val": [], "test": []}
    if len(ordered) < 3:
        result["train"] = list(pairs)
        return result
    n = max(1, round(len(ordered) * 0.15))
    for i, group in enumerate(ordered):
        split = "val" if i < n else "test" if i < n * 2 else "train"
        result[split].extend(groups[group])
    return result


def generate_same_state_dpo_pairs(num_pairs=50, output_dir="artifacts",
                                 min_reward_delta=0.1, seed=42):
    if num_pairs < 1 or min_reward_delta <= 0:
        raise ValueError("Pair count and preference margin must be positive")
    rng = random.Random(seed)
    pairs, seen_prompts, candidates = [], set(), []
    for ph in grounded_data.policyholders:
        claims = [c for c in grounded_data.claims if c.party_id == ph.party_id]
        hints = [f"{c.status} {c.case_type} claim from {c.created_at[:4]}" for c in claims]
        for style in STYLES:
            for hint in hints or ["healthcare claim"]:
                candidates.append((ph, style, hint))
    rng.shuffle(candidates)
    for episode_idx, (ph, style, hint) in enumerate(candidates):
        env = AgentPolicyEnv(caller_profile=CallerProfile(ph, claim_hint=hint, style=style,
                             email_accepted=episode_idx % 2 == 0), max_turns=15)
        obs, _ = env.reset(seed=seed + episode_idx)
        policy, done = RuleBasedPolicy(), False
        while not done and len(pairs) < num_pairs:
            action = policy.select_action(obs)
            claim = env.sm.get_active_claim()
            prompt = ("Choose a legal next insurance support response using only this visible context.\n" +
                      json.dumps({"observation": obs, "history": env.history,
                                  "grounded_claim": claim.model_dump() if claim else None}, sort_keys=True))
            snapshot = env.save_state()
            alternatives = [a for a, ok in zip(AGENT_ACTIONS, obs["action_mask"]) if ok and a != action]
            rng.shuffle(alternatives)
            if prompt not in seen_prompts and alternatives:
                expert = branch_return(env, snapshot, action)
                for alternative in alternatives:
                    other = branch_return(env, snapshot, alternative)
                    assert expert["state_hash"] == other["state_hash"] == snapshot_digest(snapshot)
                    if expert["reply"] == other["reply"]:
                        continue
                    chosen, rejected, ca, ra = ((expert, other, action, alternative)
                        if expert["return"] >= other["return"] else (other, expert, alternative, action))
                    delta = chosen["return"] - rejected["return"]
                    if delta < min_reward_delta or chosen["violations"]:
                        continue
                    pairs.append({
                        "schema_version": 2, "pair_id": f"dpo_{len(pairs)+1:04d}",
                        "group_id": f"{ph.party_id}:{style}",
                        "same_state_verified": True, "state_hash": expert["state_hash"],
                        "prompt": prompt, "chosen": chosen["reply"], "rejected": rejected["reply"],
                        "chosen_action": ca.value, "rejected_action": ra.value,
                        "chosen_return": chosen["return"], "rejected_return": rejected["return"],
                        "reward_delta": delta, "return_gamma": 0.99,
                        "chosen_violations": chosen["violations"], "phase": obs["phase"],
                    })
                    seen_prompts.add(prompt)
                    break
            env.restore_state(snapshot)
            obs, _, terminal, truncated, _ = env.step(action)
            done = terminal or truncated
        if len(pairs) >= num_pairs:
            break
    if len(pairs) != num_pairs:
        raise RuntimeError(f"Only {len(pairs)} valid pairs available; requested {num_pairs}")
    splits = split_pairs(pairs)
    for split, rows in splits.items():
        for row in rows:
            row["split"] = split
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in {"pairs": pairs, **splits}.items():
        (out / f"dpo_{name}.jsonl").write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in rows), encoding="utf-8")
    counts = {"total": len(pairs), **{s: len(v) for s, v in splits.items()},
              "unique_prompts": len(seen_prompts),
              "distinct_reply_pairs": sum(p["chosen"] != p["rejected"] for p in pairs),
              "positive_delta_pct": 100.0,
              "minimum_return_margin": min(p['reward_delta'] for p in pairs),
              "required_return_margin": min_reward_delta,
              "environment_version": AgentPolicyEnv.ENV_VERSION,
              "rejected_action_counts": {action: sum(p['rejected_action'] == action for p in pairs)
                                         for action in sorted({p['rejected_action'] for p in pairs})},
              "split_unit": "identity/style group", "training_performed": False}
    (out / "dpo_manifest.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    return pairs, counts


def generate_dpo_pairs(num_pairs=50, output_path=None, output_dir="artifacts", **kwargs):
    directory = Path(output_path).parent if output_path else Path(output_dir)
    pairs, _ = generate_same_state_dpo_pairs(num_pairs, str(directory), **kwargs)
    if output_path:
        Path(output_path).write_text("".join(json.dumps(p) + "\n" for p in pairs))
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-pairs", type=int, default=50)
    p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-reward-delta", type=float, default=0.1)
    args = p.parse_args()
    generate_dpo_pairs(args.num_pairs, args.output, args.output_dir,
                       seed=args.seed, min_reward_delta=args.min_reward_delta)


if __name__ == "__main__":
    main()
