"""Reproduce versioned multi-seed policy evidence and causal regression probes."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from backend.harness.caller_sim import make_margaret_chen_profile
from backend.harness.rl_env import AgentPolicyEnv
from backend.harness.types import AgentAction as A
from backend.rl.featurizer import StateFeaturizer
from eval.policy_comparison import (LearnedPPOPolicy, run_agent_comparison,
                                    build_report, get_profile_factory)


def probes():
    env = AgentPolicyEnv(caller_profile=make_margaret_chen_profile('cooperative'))
    obs, _ = env.reset(seed=42)
    first = env.step(A.ASK_IDENTITY_FIELD)
    env.reset(seed=42)
    again = env.step(A.ASK_IDENTITY_FIELD)
    p = env.caller_profile
    snapshot = p.get_state()
    one = p.respond_to(A.ACK_EMOTION, 'Please share your phone number.')
    p.set_state(snapshot)
    two = p.respond_to(A.ESCALATE_HUMAN, 'Please share your phone number.')
    angry = dict(obs, emotion='frustration')
    neutral = dict(obs, emotion='neutral')
    f = StateFeaturizer()
    env = AgentPolicyEnv(caller_profile=make_margaret_chen_profile('frustrated'))
    env.reset()
    _, r, _, _, info = env.step(A.ESCALATE_HUMAN)
    return {
        'no_case_feature_is_zero': f.featurize(obs)[18] == 0,
        'emotion_is_observable': not np.array_equal(f.featurize(angry), f.featurize(neutral)),
        'reset_restores_caller': first[:4] == again[:4],
        'caller_ignores_action_label': one == two,
        'unrequested_escalation_penalized': r < 0 and info['premature_termination'],
    }


def write_markdown(report, path):
    lines = ['# RL audit', '', '**Result: ' + ('PASS' if report['passed'] else 'FAIL') + '**', '',
             f"Environment {report['environment_version']}; feature schema {report['feature_version']}.", '',
             '| Seed | Split | Policy | Case success | Appropriate handoff | Premature | Truncated | Mean turns | Return |',
             '|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for run in report['runs']:
        for split in ('all', 'test'):
            for policy, row in run['splits'][split]['policies'].items():
                lines.append(f"| {run['training_seed']} | {split} | {policy} | "
                             f"{row['goal_success_rate']}% | {row['appropriate_escalation_rate']}% | "
                             f"{row['premature_termination_rate']}% | {row['truncation_rate']}% | "
                             f"{row['mean_episode_turns']} | {row['mean_cumulative_reward']} |")
    lines += ['', '## Learned first actions', '', '| Seed | Cooperative | Frustrated | Confused | Privacy concern |',
              '|---:|---|---|---|---|']
    for run in report['runs']:
        actions = run['first_actions']
        lines.append('| ' + str(run['training_seed']) + ' | ' + ' | '.join(
            actions[k] for k in ('cooperative', 'frustrated', 'confused', 'privacy_sensitive')) + ' |')
    lines += ['', '## Interpretation', '', report['limitations'], '',
              'Case success and appropriate human handoff are separate outcomes. Safety rates are measured under '
              'the SOP mask. Matching terminal distributions does not mean identical action sequences.', '',
              'The JSON companion contains all splits, profile slices, gates, checkpoint hashes and actual training steps.', '',
              'See docs/EXPERIMENT_HISTORY.md for earlier failures and artifact provenance. '
              'This report evaluates only the checkpoints whose hashes and training segments appear in its JSON companion.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def run_audit(model_paths, output_dir, episodes=28):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {'environment_version': AgentPolicyEnv.ENV_VERSION,
              'feature_version': StateFeaturizer.VERSION,
              'regressions': {k: bool(v) for k, v in probes().items()}, 'runs': []}
    passed = all(report['regressions'].values())
    for path in model_paths:
        checkpoint = torch.load(path, map_location='cpu')
        learned = LearnedPPOPolicy(path)
        run = {'training_seed': checkpoint['config']['seed'],
               'requested_steps': checkpoint['config']['total_timesteps'],
               'actual_steps': checkpoint['history'][-1]['global_step'],
               'checkpoint_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
               'training_segments': checkpoint.get('training_segments', []),
               'splits': {}, 'first_actions': {}}
        for style in ('cooperative', 'frustrated', 'confused', 'privacy_sensitive'):
            obs, _ = AgentPolicyEnv(caller_profile=make_margaret_chen_profile(style)).reset(seed=42)
            run['first_actions'][style] = learned.select_action(obs).value
        run['behavior_checks'] = {
            'frustration_acknowledged': run['first_actions']['frustrated'] == A.ACK_EMOTION.value,
            'confusion_explained': run['first_actions']['confused'] == A.EXPLAIN_VERIFICATION_GATE.value,
            'privacy_alternative_explained': run['first_actions']['privacy_sensitive'] == A.EXPLAIN_VERIFICATION_GATE.value,
        }
        passed &= all(run['behavior_checks'].values())
        for split in ('train', 'val', 'test', 'all'):
            with contextlib.redirect_stdout(io.StringIO()):
                results = run_agent_comparison(n_episodes=episodes, split=split,
                                               ppo_model_path=path, seed=42)
            evidence = build_report(results, split=split, n_episodes=episodes,
                                    max_turns=15, seed=42,
                                    profile_count=len(get_profile_factory(split)()))
            run['splits'][split] = evidence
            passed &= evidence['acceptance']['passed']
            # One episode per profile and policy; preserves actual synthetic speech.
            if split == 'test':
                turns = []
                for policy, rows in results.items():
                    for row in rows[:len(get_profile_factory(split)())]:
                        for turn in row['trajectory']:
                            turns.append({'policy': policy, 'profile': row['profile_id'], **turn})
                (output / f"test_trajectories_seed{run['training_seed']}.jsonl").write_text(
                    ''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in turns))
        report['runs'].append(run)
        print(f"seed={run['training_seed']} first_actions={run['first_actions']}")
        print('split gates:', {k: v['acceptance']['passed'] for k, v in run['splits'].items()})
    report['passed'] = bool(passed)
    report['limitations'] = ('Two training seeds and finite repeated synthetic templates; no live-model '
                            'or human satisfaction study. DPO data generation does not train a model.')
    (output / 'rl_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    write_markdown(report, output / 'rl_audit.md')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-paths', nargs='+', default=['artifacts/ppo_policy.pt', 'artifacts/seed7/ppo_policy.pt'])
    p.add_argument('--output-dir', default='artifacts')
    p.add_argument('--episodes', type=int, default=28)
    args = p.parse_args()
    if args.episodes < 1:
        p.error('episodes must be positive')
    audit = run_audit(args.model_paths, args.output_dir, args.episodes)
    print('Audit:', 'PASS' if audit['passed'] else 'FAIL')
    raise SystemExit(0 if audit['passed'] else 1)
