"""Regression probes for causal simulation, policy inputs and reward semantics."""
import copy
import numpy as np
import pytest

from backend.harness.caller_sim import make_margaret_chen_profile
from backend.harness.rl_env import AgentPolicyEnv
from backend.harness.rl_baseline import RuleBasedPolicy
from backend.harness.types import AgentAction as A
from backend.rl.featurizer import StateFeaturizer
from eval.policy_comparison import summarize_policy


def environment(style='cooperative'):
    return AgentPolicyEnv(caller_profile=make_margaret_chen_profile(style), max_turns=15)


def test_reset_restores_caller_and_does_not_reseed_global_rng():
    e = environment()
    original, _ = e.reset(seed=42)
    first = e.step(A.ASK_IDENTITY_FIELD)
    e.step(A.ASK_IDENTITY_FIELD)
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    reset, _ = e.reset(seed=42)
    assert np.random.random() == expected
    repeated = e.step(A.ASK_IDENTITY_FIELD)
    assert original == reset
    assert first[:4] == repeated[:4]


def test_snapshot_restores_trajectory_and_causal_history():
    e = environment()
    e.reset(seed=7)
    snap = e.save_state()
    first = e.step(A.ASK_IDENTITY_FIELD)
    assert [h['role'] for h in e.history] == ['user', 'assistant', 'user']
    e.restore_state(snap)
    assert e.trajectory == []
    repeated = e.step(A.ASK_IDENTITY_FIELD)
    assert first[:4] == repeated[:4]
    assert len(e.trajectory) == 1


def test_caller_responds_to_speech_not_action_or_private_state():
    p = make_margaret_chen_profile('cooperative')
    p.reset(42)
    p.get_opening_utterance()
    snap = p.get_state()
    first = p.respond_to(A.ASK_IDENTITY_FIELD, 'Please share your date of birth.', object())
    p.set_state(snap)
    second = p.respond_to(A.ESCALATE_HUMAN, 'Please share your date of birth.', None)
    assert first == second
    p.set_state(snap)
    assert p.respond_to(A.ASK_IDENTITY_FIELD, 'Unrelated speech') != first


def test_empathy_changes_subsequent_cooperation_but_never_unlocks_identity():
    e = environment('frustrated')
    before, _ = e.reset()
    refused, _, _, _, ask = e.step(A.ASK_IDENTITY_FIELD)
    assert refused['verified_fields'] == before['verified_fields']
    _, _, _, _, empathy = e.step(A.ACK_EMOTION)
    assert empathy['agent_reply'] != ask['agent_reply']
    assert e.sm.get_active_claim() is None
    after, _, _, _, _ = e.step(A.ASK_IDENTITY_FIELD)
    assert len(after['verified_fields']) > len(before['verified_fields'])


def test_policy_features_distinguish_emotion_and_empty_case():
    f = StateFeaturizer()
    obs, _ = environment().reset()
    assert f.featurize(obs)[18] == 0
    obs['emotion'] = 'frustration'
    angry = f.featurize(obs)
    obs['emotion'] = 'neutral'
    assert not np.array_equal(angry, f.featurize(obs))
    obs['active_case_id'] = 'CL-2048'
    assert f.featurize(obs)[18] == 1


def test_frustration_alone_does_not_justify_escalation():
    e = environment('frustrated')
    e.reset()
    _, reward, terminal, _, info = e.step(A.ESCALATE_HUMAN)
    assert terminal and info['premature_termination']
    assert not info['appropriate_escalation'] and reward < 0
    assert e.sm.state.trace_log[-1].gate_evaluated == 'AGENT_ESCALATION'


def test_ignoring_distress_costs_more_without_rewarding_empathy_loops():
    e = environment('frustrated')
    e.reset(seed=42)
    snapshot = e.save_state()
    _, ignored, _, _, info = e.step(A.EXPLAIN_VERIFICATION_GATE)
    assert info['reward_components']['ignored_distress'] == -0.5
    e.restore_state(snapshot)
    _, acknowledged, _, _, info = e.step(A.ACK_EMOTION)
    assert info['reward_components']['ignored_distress'] == 0
    assert ignored < acknowledged < 0
    _, repeated, _, _, info = e.step(A.ACK_EMOTION)
    assert repeated < 0


@pytest.mark.parametrize('accept', [True, False])
def test_success_requires_grounded_answer_and_real_send_or_skip_choice(accept):
    e = environment()
    e.caller_profile.email_accepted = accept
    r = RuleBasedPolicy().run_episode(e)
    assert r['task_success'] and not r['violations']
    assert e.sm.state.post_process.user_decision == ('accepted' if accept else 'declined')
    assert len(e.sm.state.mock_outbox) == int(accept)
    assert any(t['agent_action'] == A.ANSWER_GROUNDED.value for t in e.trajectory)
    for t in e.trajectory:
        assert e.observation_space.contains(t['observation'])
        assert t['reward'] == sum(t['info']['reward_components'].values())


def test_normal_timeout_is_consistent_not_a_clean_end():
    e = environment()
    e.max_turns = 1
    result = RuleBasedPolicy().run_episode(e)
    assert result['truncated'] and not result['terminated']
    summary = summarize_policy([result])
    assert summary['terminal_consistency_rate'] == 100
    assert summary['clean_termination_rate'] == 0


def test_caller_driven_reward_accounts_for_identity_and_violations():
    from backend.harness.rl_env import CallerSimulatorEnv
    from backend.harness.types import Phase
    e = CallerSimulatorEnv()
    e.reset()
    _, reward, _, _, info = e.step('My name is Margaret Chen.')
    assert info['reward_components']['new_identity_fields'] == 1
    assert reward == sum(info['reward_components'].values())
    e.sm.state.phase = Phase.CONCLUDED
    _, reward, _, _, info = e.step('Thanks.')
    assert info['structural_violations']
    assert reward == sum(info['reward_components'].values()) and reward < -90


def test_profile_combinations_are_disjoint_across_train_validation_test():
    from backend.harness.caller_sim import make_train_profiles, make_val_profiles, make_test_profiles
    groups = [{(p.ph.party_id, p.style, p.claim_hint) for p in factory()}
              for factory in (make_train_profiles, make_val_profiles, make_test_profiles)]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]


def test_dpo_splits_do_not_leak_identity_style_groups():
    from eval.generate_dpo_pairs import split_pairs
    rows = [{'group_id': f'group_{i}', 'sample': j} for i in range(12) for j in range(3)]
    splits = split_pairs(rows)
    assert all(splits.values())
    groups = {key: {r['group_id'] for r in value} for key, value in splits.items()}
    assert not groups['train'] & groups['val']
    assert not groups['train'] & groups['test']
    assert not groups['val'] & groups['test']


def test_preference_branch_credit_includes_later_task_completion():
    from eval.generate_dpo_pairs import branch_return
    e = environment('frustrated')
    e.reset(seed=42)
    snapshot = e.save_state()
    empathy = branch_return(e, snapshot, A.ACK_EMOTION)
    early_exit = branch_return(e, snapshot, A.ESCALATE_HUMAN)
    assert empathy['state_hash'] == early_exit['state_hash']
    assert empathy['final_phase'] == 'CONCLUDED'
    assert empathy['return'] > early_exit['return'] + 1
    assert empathy['reply'] != early_exit['reply']


def test_checkpoint_rejects_previous_feature_schema(tmp_path):
    import torch
    from eval.policy_comparison import LearnedPPOPolicy
    path = tmp_path / 'old.pt'
    torch.save({'feature_version': 1, 'environment_version': 1}, path)
    with pytest.raises(ValueError, match='incompatible'):
        LearnedPPOPolicy(str(path))


def test_warm_start_records_added_steps_and_provenance(tmp_path):
    from backend.rl.ppo_trainer import PPOConfig, PPOTrainer
    config = PPOConfig(total_timesteps=16, rollout_steps=16, update_epochs=1, device='cpu')
    trainer = PPOTrainer(config)
    trainer.train(16)
    path = tmp_path / 'initial.pt'
    trainer.save_checkpoint(str(path))
    continued = PPOTrainer(PPOConfig(total_timesteps=16, rollout_steps=16,
                                    update_epochs=1, device='cpu'))
    continued.load_checkpoint(str(path))
    continued.train(16)
    assert [r['global_step'] for r in continued.history] == [16, 32]
    assert continued.cfg.total_timesteps == 32
    assert len(continued.training_segments) == 2
    assert continued.training_segments[-1]['initial_step'] == 16
    assert len(continued.training_segments[-1]['warm_start_sha256']) == 64
