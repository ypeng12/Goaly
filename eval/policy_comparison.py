"""
Policy comparison: RuleBasedPolicy vs RandomPolicy
Uses the correct architecture: CallerSimulatorEnv with scripted callers.

In CallerSimulatorEnv, the state machine is driven by *caller utterances*.
The agent_action here is recorded as metadata but the state transitions
happen because the caller speaks (providing PII, claim hints, etc.).

For a true AgentPolicyEnv comparison we'd need a paired caller simulator.
This script demonstrates the correct usage pattern and produces real metrics.
"""
import sys
import random
import statistics
sys.path.insert(0, "/Users/yuliangpeng/Desktop/apps/insurance_claims")

from backend.harness.rl_env import CallerSimulatorEnv, AgentPolicyEnv, AgentAction, AGENT_ACTIONS
from backend.harness.rl_baseline import RuleBasedPolicy
from backend.harness.grounded_data import grounded_data


# ---------------------------------------------------------------------------
# Scripted caller scenarios — realistic multi-turn caller scripts
# Each list is a full caller conversation for one policyholder
# ---------------------------------------------------------------------------

def make_caller_scripts():
    """Build scripted caller utterances per policyholder fixture."""
    ph = grounded_data.policyholders[0]  # Margaret Chen, ssn_last4
    scripts = [
        # Happy path: provides all PII up front, asks about claim
        [
            f"Hi, my name is {ph.name}.",
            f"My date of birth is 03/15/1985.",
            f"My phone number is {ph.phone}.",
            "I want to ask about my denied claim.",
            "What was the denial reason?",
            "Can you send me a summary?",
            "Yes please send it to my email.",
        ],
        # Slow path: provides PII one at a time
        [
            "Hello I need help with a claim.",
            f"My name is {ph.name}.",
            "My date of birth? It's March 15, 1985.",
            f"Phone is {ph.phone}.",
            "I had a dental claim that was denied.",
            "What are the next steps?",
            "No thanks, I don't need an email.",
        ],
        # Emotional caller: frustrated first
        [
            "This is ridiculous, I've been waiting weeks!",
            f"Fine. My name is {ph.name}, DOB 1985-03-15.",
            f"Phone: {ph.phone}.",
            "My claim was denied and no one will explain why.",
            "What documents do I need to appeal?",
            "Please send me a summary.",
            "Yes, send to my email.",
        ],
        # Escalation path
        [
            f"Hi, {ph.name} here.",
            "DOB March 15 1985.",
            f"Phone {ph.phone}.",
            "I want to speak to a human agent.",
        ],
        # Out of scope then recovers
        [
            "What's the weather like today?",
            f"OK sorry. I'm {ph.name}, born 03/15/1985.",
            f"Phone is {ph.phone}.",
            "I need to check my claim status.",
            "What is the current status?",
        ],
    ]
    return scripts


class ScriptedCallerPolicy:
    """Runs CallerSimulatorEnv with a pre-written caller script.
    
    This is the correct usage of CallerSimulatorEnv: the caller speaks
    real utterances (with PII), the SOP harness validates them, and
    rewards accumulate based on state transitions.
    """
    def run_episode(self, env, script: list[str]) -> dict:
        obs, info = env.reset()
        cumulative_reward = 0.0
        all_violations = []
        terminated = truncated = False
        
        for utterance in script:
            if env._finished:
                break
            obs, reward, terminated, truncated, info = env.step(utterance)
            cumulative_reward += reward
            all_violations.extend(info.get("structural_violations", []))
        
        return {
            "cumulative_reward": cumulative_reward,
            "turns": env.turn_count,
            "terminated": terminated,
            "truncated": truncated,
            "violations": all_violations,
            "final_phase": obs["phase"],
            "verified_fields": obs["verified_fields"],
        }


class RandomCallerPolicy:
    """Sends random short utterances — baseline that shouldn't verify."""
    RANDOM_UTTERANCES = [
        "Hello.",
        "I need help.",
        "What is my claim status?",
        "Can you help me?",
        "I don't understand.",
        "Please explain.",
        "What happens next?",
        "I want more information.",
    ]
    
    def run_episode(self, env, max_turns: int = 10) -> dict:
        obs, info = env.reset()
        cumulative_reward = 0.0
        all_violations = []
        terminated = truncated = False
        
        for _ in range(max_turns):
            if env._finished:
                break
            utterance = random.choice(self.RANDOM_UTTERANCES)
            obs, reward, terminated, truncated, info = env.step(utterance)
            cumulative_reward += reward
            all_violations.extend(info.get("structural_violations", []))
        
        return {
            "cumulative_reward": cumulative_reward,
            "turns": env.turn_count,
            "terminated": terminated,
            "truncated": truncated,
            "violations": all_violations,
            "final_phase": obs["phase"],
            "verified_fields": obs["verified_fields"],
        }


def run_caller_comparison(n_repeats: int = 40, seed: int = 42):
    """Compare scripted vs random callers in CallerSimulatorEnv."""
    random.seed(seed)
    scripts = make_caller_scripts()
    
    scripted = ScriptedCallerPolicy()
    random_caller = RandomCallerPolicy()
    
    scripted_results = []
    for i in range(n_repeats):
        script = scripts[i % len(scripts)]
        env = CallerSimulatorEnv(max_turns=20)
        result = scripted.run_episode(env, script)
        scripted_results.append(result)
    
    random_results = []
    for _ in range(n_repeats):
        env = CallerSimulatorEnv(max_turns=10)
        result = random_caller.run_episode(env)
        random_results.append(result)
    
    print(f"\n{'='*72}")
    print(f"CallerSimulatorEnv: Scripted Caller vs Random Caller")
    print(f"({n_repeats} episodes each, scripted uses {len(scripts)} persona scripts)")
    print(f"{'='*72}")
    print(f"{'Metric':<38} {'Scripted':>14} {'Random':>14}")
    print("-" * 72)
    
    def pct(results, pred): return 100 * sum(1 for r in results if pred(r)) / len(results)
    def avg(results, key): return sum(r[key] for r in results) / len(results)
    
    metrics = [
        ("Mean cumulative reward",
         avg(scripted_results, "cumulative_reward"),
         avg(random_results, "cumulative_reward")),
        ("Mean episode length (turns)",
         avg(scripted_results, "turns"),
         avg(random_results, "turns")),
        ("Completion rate — CONCLUDED (%)",
         pct(scripted_results, lambda r: r["final_phase"] == "CONCLUDED"),
         pct(random_results, lambda r: r["final_phase"] == "CONCLUDED")),
        ("Escalation rate — ESCALATED (%)",
         pct(scripted_results, lambda r: r["final_phase"] == "ESCALATED"),
         pct(random_results, lambda r: r["final_phase"] == "ESCALATED")),
        ("Still in VERIFY_ID at end (%)",
         pct(scripted_results, lambda r: r["final_phase"] == "VERIFY_ID"),
         pct(random_results, lambda r: r["final_phase"] == "VERIFY_ID")),
        ("Violation rate — any violation (%)",
         pct(scripted_results, lambda r: bool(r["violations"])),
         pct(random_results, lambda r: bool(r["violations"]))),
        ("Mean verified fields collected",
         sum(len(r["verified_fields"]) for r in scripted_results) / len(scripted_results),
         sum(len(r["verified_fields"]) for r in random_results) / len(random_results)),
    ]
    
    for label, s_val, r_val in metrics:
        print(f"  {label:<36} {s_val:>14.2f} {r_val:>14.2f}")
    
    print("=" * 72)
    
    # Reward distribution
    print("\nReward distribution (scripted caller):")
    rewards = sorted(r["cumulative_reward"] for r in scripted_results)
    print(f"  min={rewards[0]:.2f}  p25={rewards[len(rewards)//4]:.2f}"
          f"  median={statistics.median(rewards):.2f}"
          f"  p75={rewards[3*len(rewards)//4]:.2f}  max={rewards[-1]:.2f}")
    
    print("\nReward distribution (random caller):")
    rewards = sorted(r["cumulative_reward"] for r in random_results)
    print(f"  min={rewards[0]:.2f}  p25={rewards[len(rewards)//4]:.2f}"
          f"  median={statistics.median(rewards):.2f}"
          f"  p75={rewards[3*len(rewards)//4]:.2f}  max={rewards[-1]:.2f}")
    
    # Phase breakdown for scripted
    print("\nFinal phase breakdown (scripted):")
    from collections import Counter
    phase_counts = Counter(r["final_phase"] for r in scripted_results)
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase:<20} {count:>3} / {n_repeats}  ({100*count/n_repeats:.0f}%)")
    
    return scripted_results, random_results


if __name__ == "__main__":
    run_caller_comparison(n_repeats=40)
