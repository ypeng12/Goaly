#!/usr/bin/env python3
"""PPO Training CLI for Goaly Insurance Claims Agent Environment.

Usage:
    python3 train_ppo.py --timesteps 50000 --device cpu --seed 42
    python3 train_ppo.py --timesteps 50000 --device auto --save-path artifacts/ppo_policy.pt
"""
import argparse
import json
from pathlib import Path

from backend.rl.ppo_trainer import PPOTrainer, PPOConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Masked PPO Policy on AgentPolicyEnv")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps to train")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--rollout-steps", type=int, default=128, help="Rollout buffer size per update")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'mps', or 'cpu'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-path", type=str, default="artifacts/ppo_policy.pt", help="Checkpoint save path")
    parser.add_argument("--metrics-path", type=str, default="artifacts/ppo_training_metrics.json", help="Metrics JSON output path")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 72)
    print("           Goaly Insurance SOP - Masked PPO Training           ")
    print("=" * 72)

    cfg = PPOConfig(
        lr=args.lr,
        rollout_steps=args.rollout_steps,
        total_timesteps=args.timesteps,
        device=args.device,
        seed=args.seed,
    )

    trainer = PPOTrainer(config=cfg)
    history = trainer.train(total_timesteps=args.timesteps)

    # Save model checkpoint
    trainer.save_checkpoint(args.save_path)

    # Save metrics JSON
    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")

    # Evaluate trained policy on all profiles (50 episodes)
    print("\nRunning post-training evaluation across 50 episodes...")
    eval_results = trainer.evaluate(n_episodes=50, deterministic=True)
    print("=" * 72)
    print("Post-Training PPO Evaluation (50 episodes, multi-profile arena):")
    print(f"  Goal Success Rate:              {eval_results['goal_success_rate']:5.1f}%")
    print(f"  Appropriate Escalation Rate:    {eval_results['appropriate_escalation_rate']:5.1f}%")
    print(f"  Premature Termination Rate:     {eval_results['premature_termination_rate']:5.1f}%")
    print(f"  Clean Termination Rate:         {eval_results['termination_rate']:5.1f}%")
    print(f"  Truncation Rate:                {eval_results['truncation_rate']:5.1f}%")
    print(f"  Terminal State Consistency:     {eval_results['terminal_state_consistency']:5.1f}%")
    print(f"  Constraint Violation Rate:      {eval_results['violation_rate']:5.1f}%")
    print(f"  Mean Verified Fields Collected: {eval_results['mean_verified_fields']:5.2f} / 3.00")
    print(f"  Mean Turns per Episode:         {eval_results['mean_turns']:5.2f}")
    print(f"  Mean Cumulative Reward:         {eval_results['mean_reward']:+5.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
