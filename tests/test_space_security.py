"""HF embedding enables the Hub parent without widening API write access."""
import pytest
from fastapi.testclient import TestClient

from backend.app import app, sessions


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv('AI_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('SPACE_ID', raising=False)
    sessions.clear()
    with TestClient(app) as client:
        yield client
    sessions.clear()


@pytest.mark.parametrize('space_id', ['', 'missing-namespace', '/demo', 'owner/',
                                    'owner/demo; frame-ancestors *', 'owner/demo\n'])
def test_local_and_malformed_space_config_keep_embedding_disabled(client, monkeypatch, space_id):
    monkeypatch.setenv('SPACE_ID', space_id)
    response = client.get('/')
    assert "frame-ancestors 'none';" in response.headers['content-security-policy']


def test_space_enables_only_hf_parent_and_keeps_other_directives(client, monkeypatch):
    local_policy = client.get('/').headers['content-security-policy']
    monkeypatch.setenv('SPACE_ID', 'Ypeng12/Goaly-SOP-RL')
    response = client.get('/')
    assert response.headers['content-security-policy'] == local_policy.replace(
        "frame-ancestors 'none'", 'frame-ancestors https://huggingface.co')
    assert response.headers['cache-control'] == 'no-store'
    assert response.headers['referrer-policy'] == 'no-referrer'


def test_request_headers_cannot_enable_embedding(client):
    response = client.get('/', headers={'space-id': 'Ypeng12/demo', 'host': 'demo.hf.space',
                                        'x-forwarded-host': 'huggingface.co'})
    assert "frame-ancestors 'none';" in response.headers['content-security-policy']


def test_hf_iframe_keeps_api_calls_same_origin(client, monkeypatch):
    monkeypatch.setenv('SPACE_ID', 'Ypeng12/Goaly-SOP-RL')
    headers = {'host': 'ypeng12-goaly-sop-rl.hf.space',
               'origin': 'https://ypeng12-goaly-sop-rl.hf.space'}
    assert client.post('/api/reset', json={}, headers=headers).status_code == 200
    for origin in ['https://huggingface.co', 'https://attacker.example']:
        assert client.post('/api/reset', json={}, headers={**headers, 'origin': origin}).status_code == 403
    assert client.post('/api/reset', json={}, headers={
        **headers, 'host': 'internal:8080',
        'x-forwarded-host': 'ypeng12-goaly-sop-rl.hf.space',
    }).status_code == 403
