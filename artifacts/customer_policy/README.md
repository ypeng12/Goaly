# Customer dialogue policy experiments

These checkpoints select real customer response actions in `backend/harness/customer_policy.py`. They use the same action mask and response executor as `/api/chat`. They do not generate a fake customer message or call the simulator from the customer API.

The model is a 30-feature actor-critic, not a language model. It learns response order: ask for a missing detail, explain the identity gate, acknowledge distress before answering, clarify a claim, or offer the closing choice. Contextual language understanding and grounded document explanations are shared capabilities of all controllers. Those improvements are not PPO gains.

## Experiment history

Both seeds (42 and 7) receive the same training budget. Each requested 20,000-step segment executes 20,096 steps because rollout batches contain 128 steps.

1. **20,096 steps:** customer simulator with five follow-up questions, stepwise identity, anger, anxiety, privacy concerns and no-match recovery. Seed 42 matched the rule controller's business outcomes on development validation. Seed 7 still caused an unresolved handoff. The initial checkpoints and validation report are retained.
2. **40,192 steps:** same distribution and reward, another 20,096 steps. Seed 7 still failed completion and distress thresholds. More optimization alone did not cover missing state combinations.
3. **Dataset revision 2:** add frustration during verification and claim processing; missing-name and missing-case-hint openings; and anxiety before verification. Both seeds receive a further 20,096 steps, for **60,288 total steps each**. Current result and acceptance checks are in `comparison.json`; intermediate validation failures remain visible.

Final development evaluation: both new seeds complete 6 of 7 cases and appropriately hand off the remaining no-match case, matching the rule controller's business outcomes. Both old simulator policies complete 3 of 7, cause 2 unresolved handoffs and time out on 1. New policies have zero measured safety violations, timeouts, unresolved handoffs and missed distress in these seven scripted scenarios. This establishes recovery from the old policies' multi-turn failures; it does not demonstrate superiority over the rule controller.

`validation_20k.json` retains the original evaluator's broader handoff classification. Use `comparison.json` for comparisons under the final business-outcome definition; the historical report is kept to expose the evaluation correction, not as final success evidence.

No old `artifacts/ppo_policy.pt` or `artifacts/seed7/ppo_policy.pt` weights were overwritten. Those remain version 3 simulator baselines. Customer checkpoints declare environment version 4, feature version 2, family `customer_multi_turn_v1`, and mask `customer_sop_v1`.

## Reproduce

Use the pinned requirements from the repository. Training requires no model API key; the text-reactive caller and grounded executor run locally.

```bash
# One seed; repeat with --seed 7.
python train_customer_policy.py --timesteps 20000 --seed 42
python train_customer_policy.py --timesteps 20000 --seed 42 \
  --warm-start artifacts/customer_policy/seed42.pt

# Compare both customer checkpoints, both original checkpoints, and rules.
python -m eval.customer_policy_comparison --split test --assert-thresholds
```

To reproduce the historical three-stage experiment exactly, first overlay the original `training_source.zip` onto the recorded base commit in `source_manifest.json` for the initial segment. The hashed source archives referenced by `initial_40k` and final checkpoints preserve the subsequent source revisions, including the exact training scenarios and fixtures. Overlay the appropriate archive before each additional segment. The final checkpoint's `training_segments` records requested/actual steps, RNG resets and warm-start SHA values. A fresh 60,000-step run on revision 2 alone is a different experiment.

Source archives are overlays for this repository, not standalone apps. They include the actual `required_document_guideline.json` and the other fixture files. `source_hashes_at_start`, `source_hashes_at_save`, and each referenced archive make changes during training detectable.

## What the report does and does not show

The seven development evaluation scenarios use wording absent from PPO rollouts, but these scenarios were inspected while repairing shared language routing. They are not a blind holdout. Identities are reused, callers are finite text protocols, and there is no human study or statistical superiority claim.

The evaluator separates completed cases, expected no-match handoffs, and unresolved handoffs caused by poor conversation. Honoring a customer's human-transfer request is reported separately; it does not turn an unresolved claim into success. The historical training reward gave +1 for honoring human requests, including requests caused by frustration. Evaluation deliberately uses the stricter business outcome definition.

Distress checks use authored angry/anxious scenarios with distress visible on the current turn. Detector-only counts remain in the trajectories because a `why do you need` heuristic can mislabel confusion as frustration. Protocol progress uses coarse response keywords and cannot establish factual accuracy. API and browser content assertions validate document references, scope, privacy and consent separately.

`mean_policy_steps` counts selected policy actions. `mean_dialogue_responses` also counts mandatory SOP acknowledgements. The version 4 evaluator fixes a duplicate handoff acknowledgement; saved training source archives retain the exact earlier implementation. Safety remains enforced by the SOP and mask, so zero violations is not evidence that the neural policy learned unrestricted safety.

`mean_policy_choice_steps` counts turns with more than one permitted action. For example, a neutral, clearly understood claim question requires a grounded answer and offers no policy choice. Verification and distress often permit several response strategies. The customer trace marks single-action turns as forced so they are not mistaken for learned decisions.
