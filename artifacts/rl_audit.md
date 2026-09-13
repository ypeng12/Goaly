# RL audit

**Result: PASS**

Environment 3; feature schema 2.

| Seed | Split | Policy | Case success | Appropriate handoff | Premature | Truncated | Mean turns | Return |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 42 | all | RuleBased | 75.0% | 25.0% | 0.0% | 0.0% | 4.25 | 14.325 |
| 42 | all | Random | 25.0% | 3.57% | 71.43% | 0.0% | 4.679 | 2.818 |
| 42 | all | PPO (Learned) | 75.0% | 25.0% | 0.0% | 0.0% | 4.25 | 14.325 |
| 42 | test | RuleBased | 50.0% | 50.0% | 0.0% | 0.0% | 4.25 | 11.325 |
| 42 | test | Random | 7.14% | 14.29% | 78.57% | 0.0% | 3.571 | 0.536 |
| 42 | test | PPO (Learned) | 50.0% | 50.0% | 0.0% | 0.0% | 4.25 | 11.325 |
| 7 | all | RuleBased | 75.0% | 25.0% | 0.0% | 0.0% | 4.25 | 14.325 |
| 7 | all | Random | 25.0% | 3.57% | 71.43% | 0.0% | 4.679 | 2.818 |
| 7 | all | PPO (Learned) | 75.0% | 25.0% | 0.0% | 0.0% | 7.0 | 14.05 |
| 7 | test | RuleBased | 50.0% | 50.0% | 0.0% | 0.0% | 4.25 | 11.325 |
| 7 | test | Random | 7.14% | 14.29% | 78.57% | 0.0% | 3.571 | 0.536 |
| 7 | test | PPO (Learned) | 50.0% | 50.0% | 0.0% | 0.0% | 4.25 | 10.325 |

## Learned first actions

| Seed | Cooperative | Frustrated | Confused | Privacy concern |
|---:|---|---|---|---|
| 42 | ASK_IDENTITY_FIELD | ACK_EMOTION | EXPLAIN_VERIFICATION_GATE | EXPLAIN_VERIFICATION_GATE |
| 7 | ACK_EMOTION | ACK_EMOTION | EXPLAIN_VERIFICATION_GATE | EXPLAIN_VERIFICATION_GATE |

## Interpretation

Two training seeds and finite repeated synthetic templates; no live-model or human satisfaction study. DPO data generation does not train a model.

Case success and appropriate human handoff are separate outcomes. Safety rates are measured under the SOP mask. Matching terminal distributions does not mean identical action sequences.

The JSON companion contains all splits, profile slices, gates, checkpoint hashes and actual training steps.

See docs/EXPERIMENT_HISTORY.md for earlier failures and artifact provenance. This report evaluates only the checkpoints whose hashes and training segments appear in its JSON companion.
