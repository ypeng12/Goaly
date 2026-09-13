# Insurance claims SOP harness

A runnable text demo with a fixed workflow, cross-phase memory, grounded claim answers, and optional model-assisted interpretation. The fixtures are synthetic. Email delivery and human handoffs are explicitly simulated.

## Run locally

Use Python 3.11. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./run.sh
```

Open **http://localhost:8080**. No API token is required for the deterministic demo. `run.sh` uses `.venv/bin/python` when available and binds to localhost. Set `PORT=8090` to change the port or `BIND_HOST=0.0.0.0` when an external bind is intentional.

To enable model-assisted interpretation, export a token before launching:

```bash
export AI_API_KEY='your-api-token'
export AI_BASE_URL='https://api.openai.com/v1'
export AI_MODEL='gpt-4o-mini'
./run.sh
```

You can also configure a token, endpoint, and model in the UI for the current session. Tokens are held in server memory and are not included in chat/state responses. Local execution reads environment variables; it does not automatically load `.env`.

The live adapter requires an **OpenAI-compatible `/chat/completions` endpoint supporting structured JSON-schema responses**. Native Anthropic or Gemini endpoints are not interchangeable with this protocol. The model receives the current caller message, which can contain PII, plus bounded conversational context. It does not receive fixture claim records or the policyholder database. Use the synthetic examples when trying a provider. See the [Chat Completions API reference](https://developers.openai.com/api/reference/resources/chat).

## Run with Docker

```bash
cp .env.example .env
# Optionally put a model token in .env; leave it empty for mock mode.
docker compose up --build
```

Open **http://localhost:8080**. Compose binds the published port to localhost and runs the container as a non-root user. To run validation in the same image:

```bash
docker compose run --rm insurance-sop-agent python -m pytest tests/ -q
docker compose run --rm insurance-sop-agent python -m eval.eval_benchmark
```

Docker was not available in the development environment, so these image build/run instructions have not been executed there.

## Try the complete conversation

Paste this first message:

> I’m the policyholder. My name is Margaret Chen, policy POL-9921. I’m calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.

The harness verifies name, DOB, and SSN last four, then reuses the denied/healthcare/January hints to select `CL-2048`. The policy number is a lookup hint and does **not** count toward the three required identity fields. Continue with:

1. “How do I submit the missing documents?”
2. “What if I cannot get the original pathology report?”
3. “That is all, thank you.”
4. “Yes, please send the email summary.” Alternatively, say “No thanks, skip it.”

The summary covers the discussion, claim status/outcome, and follow-up items. Accepting records a simulated delivery; declining concludes without sending. A subsequent message cannot change a completed session’s consent decision.

For progressive verification, start with just the name and denied-claim hint, then supply DOB and SSN on later turns. For a refusal, try “I already told you who I am. This is ridiculous. Just tell me why my claim was denied.” For scope handling, try “What is RL?” and then another irrelevant question. The agent stays within insurance customer support and offers a simulated human handoff after repeated unrelated requests.

The included claim dates are historical fixture values, including a March 18, 2026 appeal deadline for `CL-2048`. The demo must not imply that a past deadline is still available or promise an appeal outcome.

## How control and conversation fit together

```text
Caller message
    → extraction + optional model semantic proposal
    → deterministic gate/transition checks + persistent session slots
    → authorized claim lookup and bounded response composition
    → reply + redacted state/audit view

VERIFY_ID → RESOLVE_INTENT → PROCESS_CASE → POST_PROCESS
     └──────────────────── human handoff ──────────────────┘
                                             send / skip → CONCLUDED
```

| Phase | Harness control | Conversational flexibility |
| --- | --- | --- |
| `VERIFY_ID` | Requires three distinct matching fields from full name, DOB, phone, email, or SSN last four; gates all claim access. Policy numbers and national IDs do not count. | Accumulates partial answers, remembers later-phase hints, explains privacy, recognizes emotion/refusal, and offers alternate fields or a human. |
| `RESOLVE_INTENT` | Retrieves only the verified party’s cases; asks for clarification when multiple cases match. | Reuses remembered hints and interprets claim type, status, date, and requested topic. |
| `PROCESS_CASE` | Answers from the selected claim and document-guidance fixtures; does not adjudicate or modify a claim. | Chooses a bounded topic such as status, financial amounts, submission method, timing, or document alternatives. |
| `POST_PROCESS` | Offers a grounded summary; records explicit accept/decline before simulated delivery. | Handles clarification and further claim questions without treating them as consent. |

The model proposes evidence-backed slots, emotion, a response topic, and a bounded presentation style. Python validates those proposals and composes the customer-visible answer from allowed text and fixture facts. Arbitrary model prose is not displayed. This limits wording variety in exchange for predictable claim statements and SOP control. Mock mode uses deterministic extraction and the same response composer. Provider failure falls back to the deterministic path.

Early claim hints are caller assertions, not verified facts. Verification happens before lookup and disclosure even when both phases complete within one turn. Claim selection remains restricted to the verified owner. Listed representatives such as David Chen are handed off because the fixtures do not contain independently verifiable representative credentials or real consent evidence; relationship lookup alone does not unlock records. `consent_scenarios.json` is sample data, not authorization.

Useful entry points:

- `backend/harness/state_machine.py`: transitions, memory, consent, and simulated side effects.
- `backend/harness/grounded_data.py`: identity matching and ownership-restricted claim lookup.
- `backend/harness/context_builder.py`: allowed actions and context shielding.
- `backend/engine/`: deterministic response composition and optional live semantic adapter.
- `backend/app.py`: FastAPI session and chat endpoints.
- `frontend/index.html`: test conversation and state inspection.
- `fixtures/`: synthetic records, field definitions, and document guidance.

## Verification and measured results

```bash
python -m pytest tests/ -q
python -m eval.eval_benchmark
```

The benchmark writes [a Markdown report](eval/EVAL_REPORT.md) and a JSON report with every assertion and turn reply. It exits nonzero if any scenario fails. Checks cover per-turn identity/ownership gates, structured-data shielding, legal transitions, exact named memory slots, specified disclosure sentinels, scope/refusal recovery, and consent. Counts and timings are generated by the run; a blocked gate is not automatically a failure or a successful check.

The benchmark uses **MockEngine only**. It is a finite fixture regression suite, not proof of general privacy, grounding, or customer satisfaction. Live-provider integration is exercised with a mocked HTTP contract; no actual provider was tested without credentials. Regression assertions enforce the requested five-field policy and reject automatic proxy authorization.

For an optional browser check with the app running:

```bash
python -m pip install playwright
python -m playwright install chromium
python tests/browser_smoke.py
```

It checks the full send/skip flows, read-only replay, settings cleanup, and mobile overflow, and writes screenshots to `artifacts/`. [Desktop preview](artifacts/demo-desktop.png) · [Mobile preview](artifacts/demo-mobile.png).

## Experiment adapter

`InsuranceSOPEnv` provides `reset()` and `step(caller_utterance)` with a Gym-like five-value return signature:

```python
from backend.harness.rl_env import InsuranceSOPEnv

env = InsuranceSOPEnv(max_turns=20)
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(
    "My name is Margaret Chen. My claim was denied in January."
)
env.export_trajectory("eval/demo_trajectory.jsonl")
```

This is a caller simulator, **not a Gymnasium environment or a trained assistant policy**. It has no formal action/observation spaces. Rewards are illustrative, use incremental slot/phase gains, and separately record structural violations; they do not certify semantic safety or emotional success. Terminal/truncated sessions require `reset()`. Trajectories are actual JSONL, one turn per line, and contain caller text.

## API usage

Create an opaque session with `POST /api/reset {}`; use its `session_id` in `POST /api/chat {session_id, message}`. `POST /api/config {session_id, api_key, model, base_url, use_mock}` changes only that session. Inspect redacted data with `GET /api/state/{session_id}` or `GET /api/trajectory/{session_id}`. `POST /api/restore {session_id, turn_index}` branches the mock session from a zero-indexed completed turn; the UI slider uses local read-only replay instead.

Sessions expire after two hours of inactivity and have a 120-turn cap. The session ID is a bearer capability in this local demo. API calls do not return raw identity records or API tokens. Custom endpoint changes require supplying that endpoint's token; existing credentials are never silently redirected. Local HTTP endpoints require `AI_ALLOW_LOCAL_ENDPOINT=1`.

## Deployment limits

This deliverable is a local demo: sessions and tokens are in process memory, delivery/transfer integrations are mocks, and the verification fixture is not a production identity service. Before using real customer data or exposing it publicly, add authenticated access, managed storage with retention controls, rate limits, approved identity/representative-consent integrations, real idempotent delivery/handoff adapters, and security/privacy review. HTTPS and a trusted model endpoint are required for a real deployment. No HIPAA, SOC 2, or other certification is claimed.
