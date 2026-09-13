# Goaly

Projects demonstrating RL infrastructure design for safe, tool-using agents.

## [insurance_claims](./insurance_claims/)

A constrained environment and evaluation harness for training safe, tool-using customer-service agents.

**Core design**: A deterministic SOP harness separates safety enforcement from model-generated language. The model handles clarification, empathy, and path selection; deterministic code enforces identity gates, data shielding, and consent.

**Infrastructure built**:

| Component | What it demonstrates |
|---|---|
| 4-phase SOP state machine | Stateful environment with legal transitions, terminal conditions, reset |
| 9-action `AgentAction` enum + per-phase `action_mask` | Formal constrained action space; `ANSWER_GROUNDED` physically blocked when unverified |
| `CallerSimulatorEnv` + `AgentPolicyEnv` | Caller simulator vs. agent policy environment, cleanly separated |
| `RuleBasedPolicy` baseline | Deterministic SOP-aligned policy for comparison with learned policies |
| 33-scenario benchmark, 596 assertions | Behavior slicing: identity gates, ownership, memory, scope, consent, disclosure |
| 112 automated tests | Unit + integration coverage including RL-specific action mask tests |
| Trajectory export (JSONL) | SFT/DPO/RL-ready, one turn per line with reward components and violations |
| Audit trail + time-travel replay | Every gate decision logged; UI slider replays any prior state |
| Verified-first identity gate | 3-of-5 PII required; structural violation = −100 reward, not soft penalty |

**Empirical results** (40 episodes, `eval/policy_comparison.py`):

| Metric | Scripted caller | Random caller |
|---|---:|---:|
| Mean cumulative reward | +5.30 | −0.44 |
| Verified fields collected | 2.6 | 0.0 |
| Violation rate | 0% | 0% |

Zero violations in both conditions — safety is structurally enforced, not reward-shaped.
