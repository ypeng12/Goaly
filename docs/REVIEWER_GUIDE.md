# Five-minute reviewer guide

This is a constrained customer-service agent harness demonstrated with synthetic
insurance claims. A state machine owns privacy, phase transitions and consent;
Masked PPO learns dialogue actions in a reactive caller environment. The customer
demo and the policy experiments are separate entry points.

## 1. Show the customer workflow

Start the application using the root README, then open
<http://127.0.0.1:8080>. Choose Offline demo for the reproducible fixture flow.

1. Type: `My name is Margaret Chen. I'm calling about my denied healthcare claim from January.`
   Identity remains unverified. The claim hint is remembered without disclosing
   the claim record.
2. Type `DOB is 1985-03-15`, then `SSN last four is 4472`.
   Three matching identity fields unlock claim access. The remembered hint selects
   CL-2048 without restarting intent collection.
3. Ask `Can I use a scan of the pathology report?` The concise answer preserves
   the document requirements. Ask for more detail to expand them.
4. Type `That answers my question.` The agent offers the conversation summary
   and asks for email consent. Choose Skip; no simulated email is sent.
5. In a new session, try `This is ridiculous. Just tell me why my claim was denied.`
   The response acknowledges frustration, explains verification and offers
   alternatives. It does not unlock the protected claim.
6. Try unrelated questions twice in another session to demonstrate scope refusal
   and the human handoff path.

The verification card is an optional alternative to conversational collection.
Click **Verify using a form**. It opens blank; sample identities require explicit
filling and submission. Try a name alone: the call stays in VERIFY_ID and claim
details remain locked. An invalid date or email gets field-level feedback;
correcting a field retains the original claim hint. **How this works** opens the
inspector and replay; technical panels are hidden by default in customer chat.
Email delivery and human transfer are simulated.

## 2. Show what the learned policy actually does

Open **Developer lab** in the header or <http://127.0.0.1:8080/lab>.

1. Select a frustrated caller and compare all policies using the same environment
   seed: RuleBased, Random, PPO seed 42 and PPO seed 7.
2. Compare case completion, appropriate handoff, truncation, dialogue turns,
   cumulative reward and violations. A clean ending alone is not task success.
3. Select a PPO run and move the replay slider. Inspect the phase before and
   after the turn, actual agent and caller speech, legal action probabilities,
   selected action and individual reward components.
4. Show the checkpoint hash and training step count, then download the complete
   JSON trace. Probabilities are calculated from the loaded model; they are not
   decorative numbers.

The lab introduces the criteria before the numbers: protect the caller, reach
the appropriate outcome, then reduce effort. The selected run's takeaway is
computed from its actual result. Reward is labelled **Training score**, action
names are translated into plain language, and raw metadata stays expandable.

Customer chat uses the shared SOP runtime with either the mock or configured
language-model engine. The PPO checkpoint controls the synthetic policy lab;
it does not control the customer chat. The UI labels this boundary explicitly.

## 3. Evidence and limits

| Requirement or capability | Reviewable evidence |
|---|---|
| Fixed workflow and privacy gates | State machine, runtime checks, action masks, SOP benchmark |
| Early hints survive verification | Customer transcript and memory acceptance tests |
| Natural closing and optional consent | Free-text close, unresolved-question guard, send and skip tests |
| Causal interaction | Agent speech → reactive caller response → SOP transition, snapshot replay tests |
| Actual policy optimization | Masked actor-critic, GAE numerical checks, PPO logs and checkpoint versions |
| Reward and terminal correctness | Reward components, failure histories, terminal and reset regression tests |
| Emotion features | `artifacts/emotion_ablation/report.json` and `report.md` |
| Preference data | Same-state branches, continuation rewards, identity/style group splits |
| Desktop and phone usability | `artifacts/browser_acceptance/browser_acceptance.json` and screenshots |
| Real-model behavior | `artifacts/live_acceptance.json`; only `passed: true` is an acceptance pass |

The emotion experiment trains four fresh runs with paired seeds 42 and 7.
Only the five emotion inputs are zeroed in the ablated condition; profiles,
reward, masks, architecture and requested step budget stay fixed. All four runs
are reported. At this budget, full features reduce training-split truncation,
but held-out task success is unchanged. Two seeds and finite scripted callers
do not establish statistical significance or production generalization.

The trained network is an action policy, not a fine-tuned language model.
DPO preference pairs are data preparation; this repository does not claim to
have run a DPO optimizer. Structural safety results measure the tested harness
and fixtures, not universal resistance to every language-model attack.

## 4. Reproduce acceptance

```bash
python3 -m pytest -q
python3 -m eval.eval_benchmark
OMP_NUM_THREADS=1 python3 -m eval.audit_rl
python3 -m eval.generate_dpo_pairs --num-pairs 50 --min-reward-delta 0.1

# Optional browser dependency; start the app first.
python3 -m pip install playwright==1.58.0
python3 -m playwright install chromium
python3 -m eval.browser_acceptance

# Configure AI_API_KEY, AI_BASE_URL and AI_MODEL in .env first.
python3 -m eval.live_acceptance

# Fresh controlled experiment, separate from the delivery checkpoints.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m eval.emotion_ablation \
  --timesteps 25000 --seeds 42 7
```

The live suite exercises ten scenarios through HTTP. Missing credentials produce
`status: not_run` and exit code 2. Provider fallback is a failure, even if the
offline response is correct. Tokens are not included in the evidence file.
Docker CI runs offline checks and both browser sizes without paid model calls.
CI configuration alone is not proof that a build has passed; inspect the run.
