"use strict";
const $ = id => document.getElementById(id);
let experiment = null, selected = 0;
let customerReport = null;
const CUSTOMER_POLICY_LABELS = {
  customer_ppo42: 'Customer PPO · seed 42 (v4)', customer_ppo7: 'Customer PPO · seed 7 (v4)',
  rule: 'Rule baseline', legacy_ppo42: 'Original simulator PPO · seed 42 (v3)',
  legacy_ppo7: 'Original simulator PPO · seed 7 (v3)'
};
const metricNumber = value => typeof value === 'number' && Number.isFinite(value);
const percent = value => metricNumber(value) ? `${(value * 100).toLocaleString(undefined, {maximumFractionDigits: 1})}%` : 'Not reported';
const number = (value, digits = 0) => metricNumber(value) ? value.toLocaleString(undefined, {maximumFractionDigits: digits}) : 'Not reported';
const ACTION_LABELS = {
  ACK_EMOTION: 'Acknowledge the caller’s feelings', ASK_IDENTITY_FIELD: 'Ask for an identity detail',
  EXPLAIN_VERIFICATION_GATE: 'Explain why verification is needed', RESOLVE_INTENT: 'Identify the caller’s need',
  ASK_CLAIM_CLARIFICATION: 'Clarify which claim they mean', ANSWER_GROUNDED: 'Answer using the claim record',
  OFFER_EMAIL_SUMMARY: 'Offer an optional email summary', SEND_EMAIL: 'Send the agreed summary',
  ESCALATE_HUMAN: 'Request a human representative'
};
const PHASE_LABELS = {VERIFY_ID:'Verify identity',RESOLVE_INTENT:'Find the right claim',PROCESS_CASE:'Help with the claim',POST_PROCESS:'Offer follow-up',CONCLUDED:'Complete',ESCALATED:'Human requested'};
const OUTCOMES = {case_completed:'Case completed',appropriate_handoff:'Human help needed',premature_exit:'Ended before helping',timeout:'Ran out of turns'};
const text = (tag, content, cls) => { const e = document.createElement(tag); e.textContent = content; if (cls) e.className = cls; return e; };
async function api(path, body) {
  const response = await fetch(path, body ? {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)} : {});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The experiment request could not be completed.");
  return data;
}
function renderTurn() {
  const run = experiment.runs[selected], index = Number($("lab-turn").value), turn = run.turns[index];
  $("lab-turn-label").textContent = `Turn ${index + 1} / ${run.turns.length}`;
  $("lab-policy-title").textContent = run.label;
  $("lab-phase").textContent = `${PHASE_LABELS[turn.observation_before.phase]} → ${PHASE_LABELS[turn.observation.phase]}`;
  $("lab-checkpoint").textContent = run.checkpoint ? `Original simulator checkpoint · environment v${run.checkpoint.environment_version || 3} · ${run.checkpoint.actual_steps.toLocaleString()} training steps · SHA256 ${run.checkpoint.sha256}. This is not the customer conversation checkpoint.` : "This reference policy has no trained checkpoint.";
  $("lab-caller-before").textContent = turn.observation_before.caller_utterance;
  $("lab-action").textContent = ACTION_LABELS[turn.agent_action] || turn.agent_action;
  $("lab-agent").textContent = turn.agent_reply;
  $("lab-caller-after").textContent = turn.caller_utterance || "No further caller turn; the selected action ends this episode.";
  $("lab-final-box").hidden = !turn.final_reply;
  $("lab-final").textContent = turn.final_reply;
  $("lab-distribution").textContent = turn.distribution_note;
  $("lab-actions").replaceChildren();
  experiment.actions.forEach((action, i) => {
    const legal = turn.observation_before.action_mask[i], picked = action === turn.agent_action;
    const row = text("div", "", `lab-action-row${legal ? "" : " masked"}${picked ? " selected" : ""}`);
    const meta = text("div", "", "lab-action-meta");
    meta.append(text("span", `${picked ? "● " : ""}${ACTION_LABELS[action] || action}`), text("span", legal ? `${(turn.probabilities[i]*100).toFixed(1)}%` : "Blocked"));
    row.title = action;
    const track = text("div", "", "lab-action-track"), fill = text("div", "", "lab-action-fill");
    fill.style.width = `${turn.probabilities[i]*100}%`; track.append(fill); row.append(meta, track); $("lab-actions").append(row);
  });
  $("lab-reward").replaceChildren();
  Object.entries(turn.info.reward_components).forEach(([name, value]) => {
    if (value === 0) return;
    $("lab-reward").append(text("dt", name.replaceAll("_", " ")), text("dd", `${value>0?"+":""}${value.toFixed(2)}`));
  });
  $("lab-reward").append(text("dt", "Total this turn"), text("dd", turn.reward.toFixed(2)));
  $("lab-observation").textContent = JSON.stringify(turn.observation_before, null, 2);
}
function renderRun(index) {
  selected = index;
  $("lab-turn").max = experiment.runs[index].turns.length - 1;
  $("lab-turn").value = 0;
  [...$("lab-summary").children].forEach((row, i) => {row.classList.toggle("selected",i===index); row.querySelector("button").setAttribute("aria-pressed",String(i===index));});
  const run = experiment.runs[index], m = run.metrics;
  const baseline = experiment.runs.find(r => r.id === 'rule');
  let takeaway = `${run.label}: ${OUTCOMES[m.outcome] || m.outcome}, ${m.violations} rule violations, ${m.turn_count} turns. `;
  if (m.violations) takeaway += 'A rule violation takes priority over any reward or speed gain.';
  else if (['premature_exit', 'timeout'].includes(m.outcome)) takeaway += 'This call did not reach an acceptable outcome. A high training score would not make it successful.';
  else if (baseline && run.id !== 'rule' && baseline.metrics.outcome === m.outcome && !baseline.metrics.violations) {
    const diff = m.turn_count - baseline.metrics.turn_count;
    takeaway += `Same outcome as rules in this call; ${diff === 0 ? 'the same number of turns' : `${Math.abs(diff)} ${diff > 0 ? 'more' : 'fewer'} turns`}. One call does not establish an overall winner.`;
  } else takeaway += m.outcome === 'appropriate_handoff' ? 'The simulator judged human help appropriate. This is not counted as an agent-resolved case.' : 'The simulator marked the case complete. Review the conversation to assess whether the response is useful.';
  $('lab-takeaway').textContent = takeaway;
  renderTurn();
}
$("lab-form").addEventListener("submit", async event => {
  event.preventDefault(); $("lab-run").disabled = true; $("lab-status").className = "";
  $("lab-status").textContent = "Running the original v3 simulator policies through synthetic calls…";
  try {
    experiment = await api("/api/lab/run", {profile_id:$("lab-profile").value, policy:$("lab-policy").value, seed:Number($("lab-seed").value)});
    $("lab-summary").replaceChildren();
    experiment.runs.forEach((run, i) => {
      const row = document.createElement("tr");
      [run.label, OUTCOMES[run.metrics.outcome] || run.metrics.outcome, run.metrics.turn_count, run.metrics.return.toFixed(2), run.metrics.violations].forEach(value => row.append(text("td",String(value))));
      const cell = document.createElement("td"), button = text("button","Inspect"); button.type="button"; button.addEventListener("click",()=>renderRun(i)); cell.append(button); row.append(cell); $("lab-summary").append(row);
    });
    $("lab-results").hidden = false;
    renderRun(Math.max(0,experiment.runs.findIndex(r=>r.id==="ppo42")));
    $("lab-status").textContent = `Completed ${experiment.runs.length} original v3 simulator episodes with seed ${experiment.seed}. These results do not evaluate the customer conversation models.`;
  } catch (e) { $("lab-status").textContent=e.message; $("lab-status").className="lab-error"; }
  finally { $("lab-run").disabled=false; }
});
$("lab-turn").addEventListener("input",renderTurn);
$("lab-download").addEventListener("click",()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(experiment,null,2)],{type:"application/json"})); const a=document.createElement("a"); a.href=url;a.download="aegis-policy-experiment.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
function renderCustomerPolicy() {
  if (!customerReport) return;
  const key = $('customer-result-policy').value;
  const policy = customerReport.policies[key], metrics = policy.metrics, baseline = customerReport.policies.rule?.metrics;
  const episodes = Array.isArray(policy.episodes) ? policy.episodes : null;
  const distressedTurns = episodes?.reduce((sum, episode) => sum + (episode.distressed_turns || 0), 0);
  $('customer-evidence-scope').textContent = `${number(metrics.episodes)} scripted development scenarios; not a blind test or customer satisfaction study.`;
  $('customer-completion').textContent = percent(metrics.case_completion_rate);
  $('customer-ignored').textContent = distressedTurns === 0 ? 'No distressed turns' : percent(metrics.ignored_distress_rate);
  $('customer-repetition').textContent = number(metrics.unnecessary_gate_explanations);
  $('customer-safety').textContent = number(metrics.safety_violations);
  let takeaway = `${CUSTOMER_POLICY_LABELS[key] || key}: ${number(metrics.episodes)} scripted calls; ${percent(metrics.case_completion_rate)} completed and ${percent(metrics.appropriate_handoff_rate)} appropriately handed off. `;
  if (metricNumber(metrics.unresolved_handoff_rate)) takeaway += `${percent(metrics.unresolved_handoff_rate)} ended in an unresolved handoff. `;
  if (metrics.safety_violations > 0) takeaway += 'Investigate the violations before considering any improvement in speed or training score.';
  else if (baseline && key !== 'rule' && metricNumber(metrics.case_completion_rate) && metricNumber(baseline.case_completion_rate)) {
    if (metrics.case_completion_rate < baseline.case_completion_rate) takeaway += 'Completion was lower than the rule baseline on this suite. Better scores on another metric do not erase missed tasks.';
    else if (metrics.case_completion_rate === baseline.case_completion_rate) takeaway += 'Completion matches the rule baseline. Compare missed empathy and unnecessary explanations; this result alone does not show that PPO is better.';
    else takeaway += 'Completion was higher than the rule baseline on this finite suite. Review the individual calls before making a broader quality claim.';
  } else takeaway += 'Use this as a reference for completion and caller effort. The training score is not a customer satisfaction measure.';
  $('customer-evidence-takeaway').textContent = takeaway;
  const checkpoint = policy.checkpoint;
  $('customer-evidence-checkpoint').textContent = checkpoint
    ? `Selected checkpoint: environment v${checkpoint.environment_version} · ${number(checkpoint.actual_steps)} training steps · SHA256 ${checkpoint.sha256}`
    : 'The rule baseline has no trained checkpoint.';
  $('customer-evidence-other-metrics').textContent = `Premature exits: ${percent(metrics.premature_exit_rate)}. Timeouts: ${percent(metrics.truncation_rate)}. Terminal consistency: ${percent(metrics.terminal_consistency_rate)}. Mean policy decisions: ${number(metrics.mean_policy_steps ?? metrics.mean_turns, 2)}. Mean assistant responses: ${number(metrics.mean_dialogue_responses, 2)}. Mean questions answered: ${number(metrics.mean_questions_answered, 2)}. Required scope responses: ${number(metrics.forced_scope_responses)}. Training score: ${number(metrics.mean_return, 2)}.`;
}
function renderCustomerEvidence(report) {
  customerReport = report;
  $('customer-evidence-results').hidden = true;
  if (!report || report.environment_version !== 4 || !report.policies || !Object.values(report.policies).some(policy => policy.metrics)) {
    $('customer-evidence-status').textContent = 'No compatible v4 customer evaluation report has been generated yet. No result is claimed here. You can still inspect actual actions in customer chat.';
    return;
  }
  const checkStatus = report.passed === true ? 'Saved checks passed.' : report.passed === false ? 'Saved checks failed; inspect the report before claiming improvement.' : 'This report does not include an overall pass/fail judgment.';
  $('customer-evidence-status').textContent = `Saved ${report.split || 'unspecified'} split · development evaluation, not a blind test · environment v4 · ${checkStatus} These are finite synthetic conversations, not customer satisfaction ratings.`;
  const keys = Object.keys(CUSTOMER_POLICY_LABELS).filter(key => report.policies[key]?.metrics);
  $('customer-result-policy').replaceChildren();
  $('customer-comparison-rows').replaceChildren();
  keys.forEach(key => {
    const option = text('option', CUSTOMER_POLICY_LABELS[key]); option.value = key; $('customer-result-policy').append(option);
    const metrics = report.policies[key].metrics;
    const row = document.createElement('tr');
    [CUSTOMER_POLICY_LABELS[key], percent(metrics.case_completion_rate), percent(metrics.appropriate_handoff_rate), percent(metrics.unresolved_handoff_rate),
      percent(metrics.ignored_distress_rate), number(metrics.unnecessary_gate_explanations), number(metrics.safety_violations),
      number(metrics.mean_policy_steps ?? metrics.mean_turns, 2)].forEach(value => row.append(text('td', value)));
    $('customer-comparison-rows').append(row);
  });
  if (!keys.length) {
    $('customer-evidence-status').textContent = 'The saved report contains no recognized customer strategy. Inspect the report before drawing a conclusion.';
    return;
  }
  if (report.limitations) $('customer-evidence-limitations').textContent = Array.isArray(report.limitations) ? report.limitations.join(' ') : String(report.limitations);
  $('customer-evidence-results').hidden = false;
  renderCustomerPolicy();
}
$('customer-result-policy').addEventListener('change', renderCustomerPolicy);
async function init() {
  try {
    const catalog=await api("/api/lab/catalog");
    catalog.profiles.forEach(p=>{const option=text("option",`${p.name} · ${p.style.replaceAll('_',' ')} (${p.split})`);option.value=p.id;option.title=p.hint;$("lab-profile").append(option);});
    $("lab-status").textContent="Choose a caller to replay the original v3 simulator. All records are synthetic.";
  } catch(e) {$("lab-status").textContent=e.message;$("lab-run").disabled=true;}
  try {
    const evidence=await api("/api/lab/evidence");
    renderCustomerEvidence(evidence.customer_policy);
    const audit=evidence.audit, ablation=evidence.ablation, preferences=evidence.preferences;
    $("lab-evidence").textContent=`Original v3 multi-seed audit: ${audit ? (audit.passed?"PASS":"FAIL") : "not available"}. Emotion ablation: ${ablation?.runs?.length || 0} recorded runs. Preference pairs: ${preferences?.total || 0}. Inspect the report below for budgets, profile slices and failures.`;
    $("lab-evidence-json").textContent=JSON.stringify(evidence,null,2);
  } catch(e) {$("lab-evidence").textContent=e.message; $('customer-evidence-status').textContent = `Could not load the customer evaluation: ${e.message}`;}
}
init();
