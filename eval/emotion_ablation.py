"""Controlled input ablation: only the five emotion features change."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

from backend.rl.ppo_trainer import PPOConfig, PPOTrainer
from backend.harness.rl_env import AgentPolicyEnv
from backend.harness.caller_sim import make_margaret_chen_profile
from eval.policy_comparison import LearnedPPOPolicy, run_agent_comparison, build_report, get_profile_factory


def run(output_dir, timesteps, seeds):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inputs = sorted([*Path('backend/harness').glob('*.py'), *Path('backend/rl').glob('*.py'),
                     Path('backend/engine/mock_engine.py'), *Path('fixtures').glob('*.json')])
    digest = hashlib.sha256(b''.join(str(p).encode() + p.read_bytes() for p in inputs)).hexdigest()
    report = {'environment_version': AgentPolicyEnv.ENV_VERSION, 'source_data_sha256': digest,
              'design': 'Paired training seeds; same profiles, rewards, masks, architecture and step budget. Only emotion input dimensions 20:25 are zeroed in the ablation.',
              'requested_steps_per_run': timesteps, 'runs': [],
              'limitations': 'Small finite simulator and two seeds; no significance or deployment claim. These fresh runs are separate from the 75k delivery checkpoints.'}
    for seed in seeds:
        for emotion in (True, False):
            label = 'full' if emotion else 'no_emotion'
            dest = out / f'{label}_seed{seed}'
            dest.mkdir(exist_ok=True)
            trainer = PPOTrainer(PPOConfig(total_timesteps=timesteps, seed=seed, device='cpu',
                                            use_emotion_features=emotion))
            trainer.train()
            checkpoint = dest / 'policy.pt'
            trainer.save_checkpoint(str(checkpoint))
            (dest / 'training_metrics.json').write_text(json.dumps(trainer.history, indent=2))
            policy = LearnedPPOPolicy(str(checkpoint))
            row = {'condition': label, 'seed': seed, 'actual_steps': trainer.history[-1]['global_step'],
                   'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   'splits': {}, 'first_actions': {}}
            for style in ('cooperative', 'frustrated', 'confused', 'privacy_sensitive'):
                obs, _ = AgentPolicyEnv(caller_profile=make_margaret_chen_profile(style)).reset(seed=42)
                row['first_actions'][style] = policy.select_action(obs).value
            for split in ('train', 'val', 'test'):
                with contextlib.redirect_stdout(io.StringIO()):
                    results = run_agent_comparison(n_episodes=28, split=split, ppo_model_path=str(checkpoint), seed=42)
                row['splits'][split] = build_report(results, split=split, n_episodes=28, max_turns=15,
                                                    seed=42, profile_count=len(get_profile_factory(split)()))
            report['runs'].append(row)
            (out / 'report.json').write_text(json.dumps(report, indent=2))
            print('Ablation result:', seed, label, row['first_actions'], flush=True)
    lines = ['# Emotion feature ablation', '', report['design'], '',
             f'Each run requests {timesteps:,} steps. Source/data SHA256: `{digest}`.', '',
             '| Seed | Condition | Split | Case success | Appropriate handoff | Truncated | Turns | Return |',
             '|---:|---|---|---:|---:|---:|---:|---:|']
    for row in report['runs']:
        for split, evidence in row['splits'].items():
            m = evidence['policies']['PPO (Learned)']
            lines.append(f"| {row['seed']} | {row['condition']} | {split} | {m['goal_success_rate']}% | {m['appropriate_escalation_rate']}% | {m['truncation_rate']}% | {m['mean_episode_turns']} | {m['mean_cumulative_reward']} |")
    lines += ['', report['limitations'], '', 'Unsuccessful runs remain in this report; acceptance failures are outcomes, not grounds to discard a condition.', '']
    (out / 'report.md').write_text('\n'.join(lines))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', default='artifacts/emotion_ablation')
    p.add_argument('--timesteps', type=int, default=25000)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 7])
    args = p.parse_args()
    if args.timesteps < 1:
        p.error('timesteps must be positive')
    run(args.output_dir, args.timesteps, args.seeds)
