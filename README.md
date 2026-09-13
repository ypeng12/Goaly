# Goaly

Projects demonstrating RL infrastructure design and algorithm training for safe, tool-using agents.

## Insurance Claims SOP & RL Agent

A constrained environment, Masked PPO training loop, and evaluation harness for training safe, tool-using customer-service agents.

**Core design**: A deterministic SOP harness separates safety enforcement from model-generated language. The model handles clarification, empathy, and path selection; deterministic code enforces identity gates, data shielding, and consent.

### Infrastructure & Algorithm Built

| Component | What it demonstrates |
|---|---|
| 4-phase SOP state machine | Stateful environment with legal transitions, terminal conditions, reset |
| 9-action `AgentAction` enum + per-phase `action_mask` | Formal constrained action space; `ANSWER_GROUNDED` physically blocked when unverified |
| `CallerSimulatorEnv` + `AgentPolicyEnv` | Caller simulator vs. agent policy environment, cleanly separated |
| **Masked PPO Actor-Critic** (`backend/rl/`) | PyTorch PPO algorithm with action masking in logits layer (`torch.distributions.Categorical`) |
| **DPO Alignment Pipeline** (`eval/generate_dpo_pairs.py`) | Automated synthesis of `(prompt, chosen, rejected)` preference pairs from rollout delta |
| `RuleBasedPolicy` baseline | Deterministic SOP-aligned policy for comparison with learned policies |
| 33-scenario benchmark, 596 assertions | Behavior slicing: identity gates, ownership, memory, scope, consent, disclosure |
| 121 automated tests | Unit + integration coverage including RL-specific action mask and PPO tests |
| Trajectory export (JSONL) | SFT/DPO/RL-ready, one turn per line with reward components and violations |
| Audit trail + time-travel replay | Every gate decision logged; UI slider replays any prior state |
| Verified-first identity gate | 3-of-5 PII required; structural violation = −100 reward, not soft penalty |

### Empirical Results

Benchmarked across 50 episodes each with 4 distinct caller profiles (`eval/policy_comparison.py`, max_turns=15):

| Metric | `RuleBasedPolicy` | `RandomPolicy` | `PPO (Learned)` |
|---|---:|---:|---:|
| **Mean cumulative reward** | **+6.05** | +3.73 | **+3.80** |
| **Mean episode length (turns)** | 4.76 | 4.44 | **2.00** |
| **Termination rate (clean end)** | **100.0%** | 98.0% | **100.0%** |
| **Truncation rate (hit max turns)** | **0.0%** | 2.0% | **0.0%** |
| **Violation rate (any violation)** | **0.0%** | 0.0% | **0.0%** |
| **Verified fields collected** | **3.00** | 2.14 | **2.00** |

Zero violations across all conditions — safety is structurally enforced by action masking, not reward-shaped. The learned PPO policy reaches optimal resolution in fewer turns (2.0 turns vs. 4.76 turns) with a 100% clean termination rate.

### Usage

#### 1. Run PPO Training
```bash
python3 train_ppo.py --timesteps 12000 --device auto
```
Saves the trained weights to `artifacts/ppo_policy.pt` and metrics to `artifacts/ppo_training_metrics.json`.

#### 2. Run Multi-Policy Arena Comparison
```bash
python3 -m eval.policy_comparison
```
Evaluates Rule-based, Random, and Learned PPO policies in a standardized arena.

#### 3. Generate DPO Preference Dataset
```bash
python3 -m eval.generate_dpo_pairs --num-pairs 50 --output artifacts/dpo_pairs.jsonl
```

#### 4. Run Automated Tests & Slicing Benchmarks
```bash
pytest tests/ -v
python3 -m eval.eval_benchmark
```
