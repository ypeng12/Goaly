# Goaly

Projects demonstrating RL infrastructure design and algorithm training for safe, tool-using agents.

## Insurance Claims SOP & RL Agent

A constrained environment, Masked PPO training loop, and evaluation harness for training safe, tool-using customer-service agents.

**Core design**: A deterministic SOP harness separates safety enforcement from model-generated language. The model handles clarification, empathy, and path selection; deterministic code enforces identity gates, data shielding, and consent.

### What the project demonstrates

| Capability | Concrete implementation | Evidence |
|---|---|---|
| SOP constraints | Explicit phase transitions plus per-state action masks | Claim access remains locked before 3-of-5 PII verification; email send remains locked before consent |
| RL post-training | 20-feature observation, masked Actor-Critic policy, GAE, clipped PPO objective, anti-hacking rewards | Learned checkpoint evaluated against RuleBased and Random policies |
| Closed-loop interaction | The policy selects an action, a stateful Caller Simulator reacts, and the SOP machine consumes the caller response | Five caller styles with separate train, validation, and held-out test profile factories |
| Evaluation and alignment | Slicing benchmark, machine-readable policy arena, acceptance gates, and same-state DPO branching | 650 SOP assertions, policy metrics JSON, and 50 preference pairs with positive reward margin |

The customer UI demonstrates the protected SOP runtime. PPO training and policy
selection run in `AgentPolicyEnv` and are evaluated in the policy arena; the UI
does not claim that a learned checkpoint controls customer-facing responses.

### Infrastructure & Algorithm Built

| Component | What it demonstrates |
|---|---|
| 4-phase SOP state machine | Stateful environment with legal transitions, terminal conditions, reset |
| 9-action `AgentAction` enum + per-phase `action_mask` | Formal constrained action space; `ANSWER_GROUNDED` physically blocked when unverified |
| `CallerSimulatorEnv` + `AgentPolicyEnv` | Caller simulator vs. agent policy environment, cleanly separated |
| **Masked PPO Actor-Critic** (`backend/rl/`) | PyTorch PPO algorithm with action masking in logits layer (`torch.distributions.Categorical`) |
| **DPO Alignment Pipeline** (`eval/generate_dpo_pairs.py`) | Automated synthesis of `(prompt, chosen, rejected)` preference pairs from rollout delta |
| `RuleBasedPolicy` baseline | Deterministic SOP-aligned policy for comparison with learned policies |
| 33-scenario benchmark, 650 assertions | Behavior slicing: identity gates, ownership, memory, scope, consent, disclosure |
| 127 automated tests | Unit + integration coverage including RL-specific action mask, Gym spaces, PPO math, and policy-report semantics |
| Trajectory export (JSONL) | SFT/DPO/RL-ready, one turn per line with reward components and violations |
| Audit trail + time-travel replay | Every gate decision logged; UI slider replays any prior state |
| Verified-first identity gate | 3-of-5 PII required; structural violation = −100 reward, not soft penalty |

### Empirical Results

Benchmarked across 50 episodes each with 4 distinct caller profiles (`eval/policy_comparison.py`, max_turns=15):

| Metric | `RuleBasedPolicy` | `RandomPolicy` | `PPO (Learned, 50k)` |
|---|---:|---:|---:|
| **Terminal State Consistency** | **100.0%** | 98.0% | **100.0%** |
| **Constraint Violation Rate** | **0.0%** | 0.0% | **0.0%** |
| **Premature Termination Rate** | **0.0%** | 40.0% | **0.0%** |
| **Goal Success Rate** | **76.0%** | 10.0% | **76.0%** |
| **Appropriate Escalation Rate** | **24.0%** | 48.0% | **24.0%** |
| **Clean Termination Rate** | **100.0%** | 98.0% | **100.0%** |
| **Truncation Rate** | **0.0%** | 2.0% | **0.0%** |
| **Mean verified fields collected** | **2.28** | 1.78 | **2.28** |
| **Mean episode length (turns)** | 3.52 | 4.00 | **3.28** |
| **Mean cumulative reward** | **+13.53** | +2.44 | **+13.08** |

The PPO policy is evaluated on task success, terminal-state consistency,
constraint violations, verified-field completion, and episode efficiency.
Safety is enforced structurally by the harness; reward is used to optimize
task completion within the legal action space.

### Usage

#### 0. Start the interactive demo

The demo uses synthetic policyholders and claims. No provider token is needed
for the deterministic offline flow.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```

Open <http://127.0.0.1:8080>. Choose **Offline demo** to run without an API
token. To use an OpenAI-compatible model, open **API settings**, select
**Live model**, and enter the token, base URL, and model name. The token stays
in server memory for that session and is never written to browser storage.

Docker provides the same UI and API:

```bash
cp .env.example .env
# Optional: set AI_API_KEY in .env for the default live-model configuration.
docker compose up --build
```

Then open <http://127.0.0.1:8080>. Set `AI_BASE_URL`, `AI_MODEL`, and `PORT` in
`.env` when needed. Email delivery and human transfer are simulated.

#### Demo acceptance path

1. Click **Margaret’s January claim**. The agent collects three identity
   fields while claim details remain locked, then reuses the remembered January
   denial hint after verification.
2. Ask about missing documents. The answer is composed from the matching
   synthetic claim record.
3. Click **That answers my question**, then choose **Yes, send summary** or
   **No thanks, skip**. Both choices are recorded and only the accepted choice
   produces a simulated email delivery.
4. Open **Safety trace / SOP audit** to inspect phase gates, data shielding,
   consent, and replay snapshots.

#### 1. Run PPO Training
```bash
python3 train_ppo.py --timesteps 50000 --device cpu --seed 42
```
Saves the trained weights to `artifacts/ppo_policy.pt` and metrics to `artifacts/ppo_training_metrics.json`.

#### 2. Run Multi-Policy Arena Comparison
```bash
python3 -m eval.policy_comparison \
  --split all \
  --episodes 50 \
  --json-output artifacts/policy_comparison.json \
  --assert-thresholds
```
Evaluates Rule-based, Random, and Learned PPO policies in a standardized arena.
Use `--split train`, `--split val`, or `--split test` to measure the profile
families independently. `--assert-thresholds` makes the command suitable for CI.

#### 3. Generate DPO Preference Dataset
```bash
python3 -m eval.generate_dpo_pairs --num-pairs 50 --output artifacts/dpo_pairs.jsonl
```

#### 4. Run Automated Tests & Slicing Benchmarks
```bash
pytest tests/ -v
python3 -m eval.eval_benchmark
```
