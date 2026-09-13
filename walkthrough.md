# Constrained dialogue policy: implementation walkthrough

## 1. Authority and interaction

The insurance SOP is the authority for identity, ownership, claim access and
consent. The policy selects one of nine dialogue acts; it cannot grant itself
access to a claim.

~~~mermaid
flowchart LR
    O["Visible observation + legal mask"] --> P["30-feature masked PPO policy"]
    P --> A["Selected dialogue act"]
    A --> R["Bounded action renderer"]
    R --> C["Caller reads spoken text"]
    C --> S["SOP consumes caller reply"]
    S --> O
    S --> E["Outcome, reward and causal trace"]
~~~

The runtime customer UI uses model-assisted interpretation plus the SOP.
The PPO simulation is a separate experiment that shares the same SOP authority.

## 2. Environment contract

AgentPolicyEnv is driven by an integer index or AgentAction. It exposes
Gymnasium spaces. The action mask must be used when sampling: the environment
rejects masked actions, so a generic unmasked random checker can raise.

Each step captures the pre-action observation and state, renders the chosen act,
appends the assistant message, lets the caller read it, then appends the caller
reply and evaluates it through the SOP. A terminal acknowledgement is a separate
final_reply, never retroactively substituted for the selected act.

The simulator's respond_to signature still accepts an action label and optional
state for compatibility. Its behavior uses neither: identical spoken text yields
the same reaction from the same caller state regardless of the label. This is a
finite text parser with explicit capabilities, not unrestricted language understanding.

reset clears PII progress, emotion recovery, consent decisions, history and reward
bookkeeping. Local RNGs do not reseed training or neighboring environments.
Snapshots restore the SOP, caller, trajectories, history and local RNG states.

## 3. Actions and safety

| Index | Act | Purpose |
|---|---|---|
| 0 | ACK_EMOTION | Acknowledge the caller's current concern |
| 1 | ASK_IDENTITY_FIELD | Request the next missing identity field |
| 2 | EXPLAIN_VERIFICATION_GATE | Explain privacy, alternatives and the next field |
| 3 | RESOLVE_INTENT | Reuse earlier hints and confirm the caller's need |
| 4 | ASK_CLAIM_CLARIFICATION | Request a reference or year |
| 5 | ANSWER_GROUNDED | Render facts from the owned, verified claim |
| 6 | OFFER_EMAIL_SUMMARY | Ask for an explicit send/skip choice |
| 7 | SEND_EMAIL | Execute only with previously recorded consent |
| 8 | ESCALATE_HUMAN | Record a handoff with a reason |

ANSWER_GROUNDED requires verification and PROCESS_CASE. Offering a summary also
requires a grounded answer in this episode. SEND_EMAIL requires accepted consent.
The shared SOP normally queues the simulated email atomically when the caller
accepts an offer; a separate SEND_EMAIL step is not required in that path.

Before answering, claim retrieval rechecks identity and ownership. A policy
action label alone cannot mark a case successful. Success requires CONCLUDED,
an actually rendered grounded answer, an accepted/declined email choice and no
recorded constraint violation.

An escalation receives credit for a caller's human request, repeated refusal or
out-of-scope requests, or two failed case-resolution attempts. Frustration alone
and an arbitrary elapsed-turn threshold do not justify it.

## 4. Observation and training

Feature schema version 2 has 30 components:

| Indices | Features |
|---|---|
| 0–5 | Phase one-hot |
| 6–11 | Verified PII flags and count |
| 12–16 | Present memory slots and count |
| 17–19 | Shield, nonempty case ID, turn/horizon |
| 20–24 | Observable emotion one-hot |
| 25–27 | Privacy concern, refusal, human request |
| 28–29 | Resolution attempts, grounded answer delivered |

Signals are derived from caller text and public dialogue progress; private
simulator style labels are not observations. Empty string and None both mean
there is no active case.

The network has two 64-unit hidden layers and separate action/value heads.
PPO stores the sampled mask, action log probability, reward, value and done flag.
The objective uses clipped policy ratios, value MSE, entropy and gradient
clipping. It does not perform language-model weight training or value clipping.

For the done flag of transition t:
delta_t = r_t + gamma * (1-d_t) * V(s_(t+1)) - V(s_t)
A_t = delta_t + gamma * lambda * (1-d_t) * A_(t+1)

The maximum-turn budget defines failure of this finite-horizon task, with a -5
penalty and no continuation value. This intentionally differs from an external
time-limit interruption of a continuing task.

Checkpoint metadata records feature/environment versions and the training
horizon. Inference loads the same horizon for feature normalization and rejects
incompatible old checkpoints. Two runs use seeds 42 and 7; each has 50,048 initial
steps plus 25,088 additional steps (75,136 actual for a 75,000-step total request).
Warm starts retain the policy, optimizer and training history while resetting
the episode and RNG. Segment metadata includes the source checkpoint hash and
separate requested/actual step counts. This is not exact mid-rollout resumption.

## 5. Rewards and metrics

| Event | Reward |
|---|---:|
| Each turn | -0.1 |
| Ignoring visible frustration, anger or anxiety without a human request | -0.5 |
| New verified identity field | +1 |
| New memory slot | +0.5 |
| First visit to an active processing phase | +2 |
| First grounded answer delivered | +1 |
| Completed supported case with email choice | +10 |
| Appropriate escalation | +1 |
| Premature exit | -3 |
| Timeout | -5 |
| Constraint violation | -100 per violation |

Rewards are experiment choices, not business guarantees. All step components sum
to the returned reward. Masks enforce safety separately from these incentives.
The distress cost is a soft preference, not an action mask. Acknowledging emotion
still costs a turn, and repeating it has no positive reward. A caller requesting
a human is eligible for handoff without more persuasion.

The version-2 findings are documented in docs/EXPERIMENT_HISTORY.md. Both
training seeds met outcome gates, but seed 42 failed the first-action empathy
probe. Version 3 adds this cost and explicitly seeds every training episode;
both seeds are retrained from scratch with the same 50,000-step request.
That intermediate audit is retained under artifacts/interaction_cost_50k: both
seeds passed empathy checks, but seed 7 timed out on a no-match training profile.
Both seeds then receive an equal additional 25,000-step training segment.
Reward magnitudes across different environment versions are not directly comparable.

Terminal consistency checks terminated iff phase is CONCLUDED/ESCALATED, with no
simultaneous termination and truncation. A valid timeout is consistent but is
not a clean end. Reports distinguish case success, appropriate handoff,
premature handoff, verified completion, turn count and return.

The audit compares both checkpoints against the same rule and random baselines,
using the same episode seeds. It reports results by profile, paired terminal
agreement and exact action-sequence agreement. It separately checks the learned
first action for frustration and privacy concerns.

## 6. Preference construction

For each visible state, take a complete environment/caller snapshot. Compare a
rule action and legal alternatives; restore the same snapshot before each.
Verify identical snapshot hashes. Continue each branch with the same rule policy
until the episode ends and compare discounted return (gamma 0.99).

Keep only pairs with different reply strings, margin >= 0.1, and no chosen-branch
violation. prompt includes the visible dialogue context and only an authorized
claim. chosen and rejected are plain text, with action and return metadata in
separate fields.
The earlier margin of 1 admitted only premature-escalation rejections. Lowering
the explicit threshold includes measurable wasted-turn and dialogue-quality
differences: the generated minimum is 0.199, and six rejection actions occur.

Split whole identity/style groups, not individual rows. The current 50-pair
artifact has 32/10/8 rows in train/validation/test. These are synthetic preference
labels derived from return. No DPO optimizer, preference-trained checkpoint or
human language-quality evaluation is claimed.

## 7. Reproduction and limits

Run the README commands for two-seed training, eval.audit_rl, policy_comparison,
preference generation and pytest. Machine evidence is in artifacts/rl_audit.json;
example held-out synthetic dialogue traces are in
artifacts/test_trajectories_seed42.jsonl and artifacts/test_trajectories_seed7.jsonl.

Train/validation/test contain disjoint identity/style/scenario combinations,
while reusing underlying synthetic identities and some styles. Canonical all is
a demonstration set, not the union of all splits. The results are finite fixture
evidence. They do not establish human satisfaction, live-provider quality,
general language understanding or safety against every adversarial interaction.
