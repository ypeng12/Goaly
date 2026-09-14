"""Bounded synthetic policy experiments, isolated from customer sessions."""
import asyncio
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
router = APIRouter()
RUN_LOCK = asyncio.Semaphore(1)
CHECKPOINTS = {'ppo42': ROOT / 'artifacts/ppo_policy.pt',
               'ppo7': ROOT / 'artifacts/seed7/ppo_policy.pt'}
LABELS = {'rule': 'Rule baseline', 'random': 'Random legal actions',
          'ppo42': 'PPO · seed 42', 'ppo7': 'PPO · seed 7'}


class LabRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    profile_id: str = Field(default='all:0', pattern=r'^(all|train|val|test):\d{1,2}$')
    policy: Literal['compare', 'rule', 'random', 'ppo42', 'ppo7'] = 'compare'
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


def profile_catalog():
    from .harness.caller_sim import make_all_profiles, make_train_profiles, make_val_profiles, make_test_profiles
    return {'all': make_all_profiles, 'train': make_train_profiles,
            'val': make_val_profiles, 'test': make_test_profiles}


@lru_cache(maxsize=4)
def load_policy(key, modified_ns):
    # The client supplies only allowlisted keys, never a filesystem path.
    from eval.policy_comparison import LearnedPPOPolicy
    import torch
    path = CHECKPOINTS[key]
    policy = LearnedPPOPolicy(str(path))
    checkpoint = torch.load(path, map_location='cpu')
    metadata = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'training_seed': checkpoint['config']['seed'],
                'actual_steps': checkpoint['history'][-1]['global_step'],
                'environment_version': checkpoint['environment_version'],
                'feature_version': checkpoint['feature_version']}
    return policy, metadata


def experiment(request):
    import torch
    from .harness.rl_env import AgentPolicyEnv
    from .harness.types import AGENT_ACTIONS
    from .harness.rl_baseline import RuleBasedPolicy
    from eval.policy_comparison import RandomPolicy
    split, index = request.profile_id.split(':')
    factory = profile_catalog()[split]
    if int(index) >= len(factory()):
        raise HTTPException(422, 'Unknown synthetic profile')
    runs = []
    for key in LABELS if request.policy == 'compare' else [request.policy]:
        profile = factory()[int(index)]
        metadata = None
        if key.startswith('ppo'):
            path = CHECKPOINTS[key]
            if not path.is_file():
                raise HTTPException(503, f'Missing {LABELS[key]} checkpoint. Run the documented training commands.')
            try:
                policy, metadata = load_policy(key, path.stat().st_mtime_ns)
            except (ValueError, RuntimeError, KeyError):
                raise HTTPException(503, 'Checkpoint is incompatible with this environment. Retrain before running the lab.')
        else:
            policy = RuleBasedPolicy() if key == 'rule' else RandomPolicy(seed=request.seed)
        env = AgentPolicyEnv(caller_profile=profile, max_turns=15)
        obs, _ = env.reset(seed=request.seed)
        opening, turns, total = obs['caller_utterance'], [], 0.0
        done = False
        while not done:
            mask = obs['action_mask']
            value = None
            if key.startswith('ppo'):
                with torch.no_grad():
                    state = policy.featurizer.featurize_tensor(obs, turn=env.turn_count)
                    masked_logits, value_tensor = policy.policy(state, policy.featurizer.extract_mask_tensor(obs))
                    probabilities = torch.softmax(masked_logits, dim=-1).tolist()
                    value = float(value_tensor)
                action = AGENT_ACTIONS[max(range(len(probabilities)), key=probabilities.__getitem__)]
                distribution = 'Masked policy probabilities; execution selects the largest probability.'
            else:
                action = policy.select_action(obs)
                probabilities = ([float(a == action) for a in AGENT_ACTIONS] if key == 'rule'
                                 else [float(ok) / sum(mask) for ok in mask])
                distribution = ('Rule decision shown as one-hot; not a neural probability.' if key == 'rule'
                                else 'Uniform probabilities over legal actions; execution samples with the chosen seed.')
            obs, reward, terminal, truncated, info = env.step(action)
            total += reward
            turns.append({**env.trajectory[-1], 'probabilities': probabilities,
                          'value_estimate': value, 'distribution_note': distribution})
            done = terminal or truncated
        last = turns[-1]['info']
        outcome = ('case_completed' if last['task_success'] else 'appropriate_handoff' if last['appropriate_escalation']
                   else 'timeout' if truncated else 'premature_exit')
        runs.append({'id': key, 'label': LABELS[key], 'checkpoint': metadata, 'opening': opening,
                     'turns': turns, 'metrics': {'outcome': outcome, 'final_phase': obs['phase'],
                     'turn_count': len(turns), 'return': round(total, 4),
                     'violations': sum(len(t['info']['structural_violations']) for t in turns),
                     'verified_fields': len(obs['verified_fields'])}})
    return {'profile_id': request.profile_id, 'seed': request.seed, 'synthetic': True,
            'actions': [a.value for a in AGENT_ACTIONS], 'runs': runs}


@router.get('/lab')
def lab_page():
    return FileResponse(ROOT / 'frontend/lab.html')


@router.get('/api/lab/catalog')
def catalog():
    return {'profiles': [{'id': f'{split}:{i}', 'split': split, 'name': p.ph.name,
                          'style': p.style, 'hint': p.claim_hint}
                         for split, factory in profile_catalog().items() for i, p in enumerate(factory())],
            'policies': LABELS}


@router.post('/api/lab/run')
async def run_lab(request: LabRequest):
    if RUN_LOCK.locked():
        raise HTTPException(429, 'Another experiment is running. Please retry shortly.')
    async with RUN_LOCK:
        return await asyncio.to_thread(experiment, request)


@router.get('/api/lab/evidence')
def evidence():
    paths = {'audit': ROOT / 'artifacts/rl_audit.json',
             'ablation': ROOT / 'artifacts/emotion_ablation/report.json',
             'preferences': ROOT / 'artifacts/dpo_manifest.json'}
    result = {}
    for key, path in paths.items():
        if path.exists():
            try:
                result[key] = json.loads(path.read_text())
            except json.JSONDecodeError:
                result[key] = None  # A training report may be being replaced.
        else:
            result[key] = None
    customer_path = ROOT / 'artifacts/customer_policy/comparison.json'
    result['customer_policy'] = None
    if customer_path.is_file():
        try:
            customer = json.loads(customer_path.read_text())
            # The initial page needs outcomes, not the full synthetic transcript.
            result['customer_policy'] = {
                key: customer.get(key) for key in
                ('schema_version', 'environment_version', 'mask_version', 'split', 'limitations', 'passed', 'checks')
            }
            result['customer_policy']['policies'] = {
                key: {field: data.get(field) for field in ('checkpoint', 'metrics')}
                for key, data in customer.get('policies', {}).items()
            }
        except (OSError, json.JSONDecodeError):
            pass
    return result


@router.get('/api/lab/customer-report')
def customer_policy_report():
    path = ROOT / 'artifacts/customer_policy/comparison.json'
    if not path.is_file():
        raise HTTPException(404, 'The customer policy report has not been generated.')
    return FileResponse(path, media_type='application/json', filename='customer-policy-comparison.json')
