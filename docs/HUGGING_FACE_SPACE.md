# Goaly on Hugging Face Spaces

Deployment target: [Ypeng12/goaly-sop-rl](https://huggingface.co/spaces/Ypeng12/goaly-sop-rl).
This guide describes the release procedure; availability and hosted behavior
must be verified after upload. A repository URL or `RUNNING` status alone is
not a successful deployment.

## Try the demo

Use the text box to describe your question, or expand the optional suggestions.
Under **Testing the demo?**, select **Margaret’s January claim** for a synthetic
caller. Ask **What documents do I need? → What is the second one? → I can’t get
it. → Can I send photos?** Then say **That answers my questions** and choose
whether to send or skip the summary. You can also use the blank verification
form and its explicit sample-fill controls.

All identities and claims are synthetic. Email delivery and human transfer are
simulated; this demo does not contact an insurer or send real customer emails.
Use the supplied test identities, not personal insurance information.

Offline language mode needs no credentials and still runs the selected trained
PPO controller. In **Settings**, the action controller and language engine are
separate. **This response** shows the executed action, legal probabilities,
checkpoint identity and any fallback. Probability indicates action preference,
not answer accuracy. Required SOP responses are marked separately.

For live language interpretation, enter your own API token, trusted
OpenAI-compatible base URL and model name in **Settings**. The token stays in
that session’s server memory and is not saved in browser storage. Requests are
sent to the chosen provider. A live setting alone does not prove a provider
request succeeded; inspect the reported activity and fallback status.

## Deployment configuration

The existing FastAPI application uses a **Docker Space**, with `sdk: docker` and
`app_port: 8080` in the root README frontmatter. The Dockerfile listens on
`0.0.0.0:8080`, runs as UID 1000 and installs CPU PyTorch on the x86 host.
The small checked-in PPO networks use **CPU Basic**; no GPU or new training is
required to host them. CPU Basic has no hourly charge, but creating a Docker
Space requires an eligible paid account plan under the current
[Spaces requirements](https://huggingface.co/docs/hub/spaces-overview).

Do not set a shared `AI_API_KEY` or `OPENAI_API_KEY` in this public Space:
new sessions would inherit it. Visitors configure their own model credentials.
Keep `AI_ALLOW_LOCAL_ENDPOINT` unset. Hugging Face’s `SPACE_ID` runtime variable
enables embedding by `https://huggingface.co`; API writes still require the
app’s own origin. Local deployments retain the default no-embedding policy.

Sessions are held in one server process, with a two-hour inactivity expiry,
256-session capacity and 120-turn limit. A restart, sleep or rebuild loses
conversations and session credentials. There is no persistent customer store
or storage bucket. Keep the default single Uvicorn worker.

## Publish a reviewed release

Authenticate with a write-capable Hugging Face account. Review and commit the
intended release, then confirm `git status --short` is empty. Export tracked
files from that commit rather than uploading the working directory: Docker’s
`.dockerignore` does not control what gets published to the Hub.

```bash
hf auth whoami
hf repos create Ypeng12/goaly-sop-rl --type space --space-sdk docker \
  --flavor cpu-basic --public --exist-ok
git status --short
goaly_release_dir="$(mktemp -d /tmp/goaly-space.XXXXXX)"
git archive HEAD | tar -x -C "$goaly_release_dir"
hf upload Ypeng12/goaly-sop-rl "$goaly_release_dir" . --repo-type space \
  --exclude "**/__pycache__/**"
```

Check the export contains no credentials or `.env` files before upload. Record
the source commit and the resulting Space revision so hosted results can be
traced to their code and weights. The root Dockerfile, frontmatter, fixtures,
customer checkpoints and historical Lab checkpoints must all be included.

## Verify the hosted release

```bash
hf spaces info Ypeng12/goaly-sop-rl --expand runtime
hf spaces logs Ypeng12/goaly-sop-rl --tail 200
curl --fail https://ypeng12-goaly-sop-rl.hf.space/api/health
```

Confirm the health response identifies synthetic data and simulated email and
handoff. Then check the actual Hugging Face page as well as the direct app:

1. The iframe loads and a chat message succeeds without an origin error.
2. Name alone, and name plus DOB, remain in `VERIFY_ID` with claims locked.
   Three matching supported fields unlock the remembered claim request.
3. Contextual document follow-ups produce relevant grounded answers. Ending
   opens the email choice; skip produces no outbox item, while explicit send
   produces one simulated item.
4. Switch between rule, PPO seed 42 and PPO seed 7. Inspect actual decisions
   and checkpoint hashes; a missing-checkpoint fallback is not PPO acceptance.
5. Open `/lab`, read the current customer report, and run a policy comparison.
   The historical version-3 simulator and version-4 customer policies have
   separate weights and must not have their scores combined.

For automated desktop and mobile coverage, with Playwright installed locally:

```bash
python3 -m eval.browser_acceptance \
  --url https://ypeng12-goaly-sop-rl.hf.space \
  --require-customer-report --output-dir /tmp/goaly-space-browser
```

Read the runtime logs again after exercising the app, including model-load or
provider failures. Follow the [Docker Space documentation](https://huggingface.co/docs/hub/spaces-sdks-docker)
when diagnosing build and runtime differences.

## What the evidence establishes

The checked-in [live-model report](../artifacts/customer_runtime/live_acceptance.json)
is **`not_run`**: no provider credential was configured for that run. Offline
success, mocked HTTP tests and a healthy Space do not establish live-model
acceptance. Run `python3 -m eval.live_acceptance --url <direct-app-url>` with a
private local credential configuration to produce separate live evidence;
never include that credential file in the release.

On the [seven development scenarios](../artifacts/customer_policy/comparison.json),
both customer PPO seeds complete six cases and make one expected no-match
handoff, matching the rule baseline. These scenarios were inspected during
development; they are not a blind evaluation or evidence of superiority over
rules. Observed zero violations demonstrate the tested SOP and mask boundaries,
not that PPO learned safety independently. DPO artifacts are preference data,
not a trained DPO model. See [customer runtime details](CUSTOMER_RUNTIME.md).
