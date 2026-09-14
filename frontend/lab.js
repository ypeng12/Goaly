"use strict";
const $ = id => document.getElementById(id);
let experiment = null, selected = 0;
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
  $("lab-checkpoint").textContent = run.checkpoint ? `${run.checkpoint.actual_steps.toLocaleString()} training steps · SHA256 ${run.checkpoint.sha256}` : "This reference policy has no trained checkpoint.";
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
  $("lab-status").textContent = "Running synthetic calls through the actual policy and SOP…";
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
    $("lab-status").textContent = `Completed ${experiment.runs.length} synthetic episodes with seed ${experiment.seed}. Results below were computed by this run.`;
  } catch (e) { $("lab-status").textContent=e.message; $("lab-status").className="lab-error"; }
  finally { $("lab-run").disabled=false; }
});
$("lab-turn").addEventListener("input",renderTurn);
$("lab-download").addEventListener("click",()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(experiment,null,2)],{type:"application/json"})); const a=document.createElement("a"); a.href=url;a.download="aegis-policy-experiment.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
async function init() {
  try {
    const catalog=await api("/api/lab/catalog");
    catalog.profiles.forEach(p=>{const option=text("option",`${p.name} · ${p.style.replaceAll('_',' ')} (${p.split})`);option.value=p.id;option.title=p.hint;$("lab-profile").append(option);});
    $("lab-status").textContent="Choose a caller and run an experiment. All records are synthetic.";
  } catch(e) {$("lab-status").textContent=e.message;$("lab-run").disabled=true;}
  try {
    const evidence=await api("/api/lab/evidence");
    const audit=evidence.audit, ablation=evidence.ablation, preferences=evidence.preferences;
    $("lab-evidence").textContent=`Saved multi-seed audit: ${audit ? (audit.passed?"PASS":"FAIL") : "not available"}. Emotion ablation: ${ablation?.runs?.length || 0} recorded runs. Preference pairs: ${preferences?.total || 0}. Inspect the report below for budgets, profile slices and failures.`;
    $("lab-evidence-json").textContent=JSON.stringify(evidence,null,2);
  } catch(e) {$("lab-evidence").textContent=e.message;}
}
init();
