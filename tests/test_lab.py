"""Lab traces come from the environment, while customer sessions stay private."""
import pytest
from fastapi.testclient import TestClient
from backend.app import app


def test_lab_comparison_has_real_probabilities_rewards_and_checkpoints():
    with TestClient(app) as client:
        response = client.post('/api/lab/run', json={'profile_id':'all:0','policy':'compare','seed':42})
        assert response.status_code == 200
        data = response.json()
        assert len(data['runs']) == 4
        for run in data['runs']:
            assert run['metrics']['return'] == pytest.approx(sum(t['reward'] for t in run['turns']))
            if run['id'].startswith('ppo'):
                assert len(run['checkpoint']['sha256']) == 64
                assert run['checkpoint']['actual_steps'] > 0
            for turn in run['turns']:
                assert sum(turn['probabilities']) == pytest.approx(1, abs=1e-6)
                assert all(p == 0 for p, legal in zip(turn['probabilities'], turn['observation_before']['action_mask']) if not legal)
                assert turn['reward'] == pytest.approx(sum(turn['info']['reward_components'].values()))
                if turn['observation_before']['data_shield_active']:
                    assert 'CL-2048' not in turn['agent_reply'] and 'pathology report' not in turn['agent_reply']


def test_lab_scenario_bounds_and_seed_replay():
    with TestClient(app) as client:
        for body in [{'profile_id':'../../private'}, {'profile_id':'test:99'}, {'policy':'/tmp/model.pt'}, {'seed':-1}]:
            assert client.post('/api/lab/run', json=body).status_code == 422
        request = {'profile_id':'test:1', 'policy':'random', 'seed':8}
        a = client.post('/api/lab/run',json=request).json()['runs'][0]
        b = client.post('/api/lab/run',json=request).json()['runs'][0]
        assert a['metrics'] == b['metrics']
        assert [t['agent_action'] for t in a['turns']] == [t['agent_action'] for t in b['turns']]


def test_lab_catalog_is_separate_from_customer_sessions():
    with TestClient(app) as client:
        sid=client.post('/api/reset',json={}).json()['session_id']
        before=client.get('/api/state/'+sid).json()
        assert client.get('/lab').status_code == 200
        assert len(client.get('/api/lab/catalog').json()['profiles']) >= 4
        client.post('/api/lab/run',json={'policy':'rule'})
        after=client.get('/api/state/'+sid).json()
        assert before['sop_state'] == after['sop_state']


def test_live_acceptance_without_credentials_does_not_claim_a_pass(tmp_path):
    from eval.live_acceptance import run
    report = run('http://unused.invalid',{'api_key':'','base_url':'https://unused.invalid','model':'test'},tmp_path/'report.json')
    assert report['status'] == 'not_run' and not report['passed'] and report['cases'] == []
