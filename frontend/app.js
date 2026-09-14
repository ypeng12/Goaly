"use strict";

// Conversation, settings, and replay stay in memory. API tokens are never stored here.
const $ = (id) => document.getElementById(id);
const PHASES = ["VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE", "POST_PROCESS"];
const PHASE_LABELS = {
  VERIFY_ID: "Identity verification", RESOLVE_INTENT: "Understanding your need",
  PROCESS_CASE: "Working through your claim", POST_PROCESS: "Your follow-up choice",
  CONCLUDED: "Call complete", ESCALATED: "Human handoff requested"
};
const PII_ITEMS = {name: "pii-name", dob: "pii-dob", phone: "pii-phone", email: "pii-email", id_last4: "pii-id4"};
const PII_LABELS = {name: "Full name", dob: "Date of birth", phone: "Phone", email: "Email", id_last4: "SSN last four"};
const SCENARIOS = {
  "scenario-margaret": "I’m the policyholder. My name is Margaret Chen, policy POL-9921. I’m calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.",
  "scenario-partial": "Hi, my name is Ava Lopez. I’m calling about a claim, but I’d like to verify one detail at a time.",
  "scenario-frustrated": "I already told you who I am. This is ridiculous. Just tell me why my claim was denied.",
  "scenario-oos": "What is RL?",
  "scenario-proxy": "I am David Chen calling on behalf of my mother Margaret Chen, policy POL-9921. Her DOB is 1985-03-15 and her SSN last four is 4472. I’m calling about her denied healthcare claim from January."
};
const CONTROLLER_LABELS = {rule: 'Rule baseline', ppo42: 'Trained PPO · 42', ppo7: 'Trained PPO · 7', sop: 'Required SOP step'};
const ACTION_LABELS = {
  ACK_EMOTION: 'acknowledge your concern', ASK_IDENTITY_FIELD: 'ask for an identity detail',
  EXPLAIN_VERIFICATION_GATE: 'explain why verification is needed', RESOLVE_INTENT: 'understand your request',
  ASK_CLAIM_CLARIFICATION: 'clarify which claim you mean', ANSWER_GROUNDED: 'answer from the claim record',
  OFFER_EMAIL_SUMMARY: 'offer an optional summary', SEND_EMAIL: 'send the agreed summary',
  ESCALATE_HUMAN: 'request a human representative'
};
const WELCOME = "Hi, I’m Aegis. Tell me what happened with your claim — your request for insurance payment. You can ask follow-up questions in your own words.\n\nWe’ll verify your identity before opening your record. Use the form if that’s easier than typing your details.";
let requestedController = new URLSearchParams(window.location.search).get('controller');
if (!Object.hasOwn(CONTROLLER_LABELS, requestedController) || requestedController === 'sop') requestedController = null;
let sessionId = null;
let config = {};
let busy = false;
let replaying = false;
let liveMessages = [];
let snapshots = [];
let liveData = initialState();

function initialState() {
  return {current_phase: "VERIFY_ID", sop_state: {identity_verified: false, verified_fields: [], collected_fields: [], cross_phase_memory: {}, trace_log: [], post_process: {}}, active_case: null};
}
function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined && text !== null) el.textContent = String(text);
  return el;
}
function clone(value) { return JSON.parse(JSON.stringify(value)); }
function notice(message) {
  $("service-notice").textContent = message || "";
  $("service-notice").hidden = !message;
}
async function request(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 90000);
  try {
    const response = await fetch(path, {...options, signal: controller.signal, headers: {"Content-Type": "application/json", ...options.headers}});
    let data;
    try { data = await response.json(); } catch (_) { throw new Error("The server returned an unreadable response. Please try again."); }
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The server could not complete this request. Please try again.");
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The response took too long. Check the connection before trying again.");
    if (error instanceof TypeError) throw new Error("Could not reach the server. Check that the demo backend is running.");
    throw error;
  } finally { clearTimeout(timeout); }
}
function isTerminal() { return ["CONCLUDED", "ESCALATED"].includes(liveData.current_phase); }
function syncControls() {
  const unavailable = busy || replaying || !sessionId || isTerminal();
  $("user-input").disabled = unavailable;
  $("btn-send").disabled = unavailable;
  for (const id of [...Object.keys(SCENARIOS), "btn-accept-email", "btn-decline-email", "btn-wrap-up"]) $(id).disabled = unavailable;
  document.querySelectorAll('#customer-options button').forEach(button => { button.disabled = unavailable; });
  document.querySelectorAll('.chat-form-action').forEach(button => {
    button.disabled = unavailable || liveData.current_phase !== 'VERIFY_ID';
  });
  document.querySelectorAll('#security-card-el input, #security-card-el select, #security-card-el button').forEach(el => {
    el.disabled = unavailable || liveData.current_phase !== 'VERIFY_ID';
  });
  $("btn-reset").disabled = busy;
  $("btn-config").disabled = busy || replaying || !sessionId;
  $("btn-conversation-settings").disabled = busy || replaying || !sessionId;
  $("time-travel-slider").disabled = busy || snapshots.length < 2;
  $("replay-notice").hidden = !replaying;
  $("inspector-mode").textContent = replaying ? "REPLAY" : "LIVE";
  $("inspector-mode").className = replaying ? "tag pending" : "tag verified";
  $("chat-window").setAttribute("aria-busy", String(busy));
  $("user-input").placeholder = replaying ? "Return to live to continue…" : isTerminal() ? "This call has ended. Start a new call to continue." : liveData.current_phase === 'PROCESS_CASE' ? 'Ask a follow-up — for example, “What does the second item mean?”' : "Tell me what happened, or ask a follow-up…";
}
function renderMessage(message) {
  const role = ["user", "assistant", "system"].includes(message.role) ? message.role : "system";
  const row = node("div", `chat-msg ${role}`);
  const avatar = node("div", "msg-avatar", role === "assistant" ? "a" : role === "user" ? "You" : "!");
  avatar.setAttribute("aria-hidden", "true");
  const bubble = node("div", "msg-bubble");
  const meta = node("div", "msg-meta");
  meta.append(node("span", "msg-sender", role === "assistant" ? "Aegis support" : role === "user" ? "You" : "Connection notice"));
  if (message.phase) meta.append(node("span", "msg-phase", PHASE_LABELS[message.phase] || message.phase));
  const body = node("div", "msg-text");
  String(message.text).split(/\n\n+/).forEach(paragraph => body.append(node('p', '', paragraph)));
  // Keep the form choice beside the latest verification instruction, where a
  // caller is already reading, rather than relying on controls below the chat.
  if (role === 'assistant' && message.phase === 'VERIFY_ID') {
    document.querySelectorAll('#chat-window .chat-form-action').forEach(button => button.remove());
    const formAction = node('button', 'btn btn-primary chat-form-action', 'Verify using a form');
    formAction.type = 'button';
    formAction.addEventListener('click', openVerification);
    body.append(formAction);
  }
  bubble.append(meta, body);
  row.append(avatar, bubble);
  $("chat-window").append(row);
  $("chat-window").scrollTop = $("chat-window").scrollHeight;
}
function renderMessages(messages) {
  $("chat-window").replaceChildren();
  messages.forEach(renderMessage);
  renderVerificationCard();
}

function appendLive(message) { liveMessages.push(message); renderMessage(message); }
function showTyping() {
  const row = node("div", "chat-msg assistant typing");
  row.append(node("div", "msg-avatar", "a"), node("div", "msg-text", "Working on your request…"));
  $("chat-window").append(row);
  $("chat-window").scrollTop = $("chat-window").scrollHeight;
  return row;
}
async function sendMessage(text) {
  if (busy || replaying || !sessionId || isTerminal() || !text.trim()) return;
  const message = text.trim();
  const restoreComposerFocus = [$("user-input"), $("btn-send")].includes(document.activeElement);
  if (message.length > 6000) { notice("Please keep each message under 6,000 characters."); return; }
  busy = true;
  notice("");
  syncControls();
  $("user-input").value = "";
  appendLive({role: "user", text: message});
  const typing = showTyping();
  try {
    const data = await request("/api/chat", {method: "POST", body: JSON.stringify({session_id: sessionId, message})});
    typing.remove();
    if (!data.sop_state || typeof data.reply !== "string") throw new Error("The server returned an incomplete turn. Please start a new call.");
    liveData = data;
    appendLive({role: "assistant", text: data.reply, phase: data.current_phase});
    snapshots.push({data: clone(data), messages: clone(liveMessages)});
    renderInspector(data);
    renderEngine(data);
    renderReplayStatus(snapshots.length - 1);
    $("chat-window").scrollTop = $("chat-window").scrollHeight;
  } catch (error) {
    typing.remove();
    appendLive({role: "system", text: error.message});
    notice("Your message may have reached the server. Check the conversation before retrying; a new call starts a clean session.");
  } finally {
    busy = false;
    syncControls();
    if (!isTerminal() && restoreComposerFocus) $("user-input").focus();
  }
}
function renderReplayStatus(index) {
  const latest = snapshots.length - 1;
  $("time-travel-slider").max = Math.max(0, latest);
  $("time-travel-slider").value = index;
  $("time-travel-badge").textContent = replaying ? `Replay · Turn ${index} / ${latest}` : `Live · Turn ${Math.max(0, latest)}`;
  $("time-travel-slider").setAttribute("aria-valuetext", $("time-travel-badge").textContent);
  $("btn-revert-live").hidden = !replaying;
}
// The slider only renders local snapshots; it never restores or forks server state.
function applySnapshot(index) {
  if (busy || !snapshots[index]) return;
  replaying = index !== snapshots.length - 1;
  const snapshot = replaying ? snapshots[index] : {data: liveData, messages: liveMessages};
  renderMessages(snapshot.messages);
  renderInspector(snapshot.data);
  renderEngine(snapshot.data);
  renderReplayStatus(index);
  syncControls();
}
function renderStepper(data) {
  const phase = data.current_phase;
  const logs = data.sop_state.trace_log || [];
  let index = PHASES.indexOf(phase);
  if (phase === "CONCLUDED") index = 4;
  if (phase === "ESCALATED") {
    const beforeEscalation = [...logs].reverse().find(log => PHASES.includes(log.phase_before));
    index = PHASES.indexOf(beforeEscalation?.phase_before || "VERIFY_ID");
  }
  ["step-verify", "step-intent", "step-process", "step-post"].forEach((id, i) => {
    const step = $(id);
    step.className = "step-item";
    step.removeAttribute("aria-current");
    step.querySelector(".step-number").textContent = i < index ? "✓" : String(i + 1).padStart(2, "0");
    if (i < index) step.classList.add("completed");
    if (i === index) {
      step.classList.add(phase === "ESCALATED" ? "halted" : "active");
      step.setAttribute("aria-current", "step");
    }
  });
  $("call-status").textContent = PHASE_LABELS[phase] || phase;
}
function renderInspector(data) {
  const state = data.sop_state || {};
  const verified = [...new Set(state.verified_fields || [])].filter(field => Object.hasOwn(PII_ITEMS, field));
  const collected = (state.collected_fields || []).filter(field => Object.hasOwn(PII_ITEMS, field));
  const identityVerified = state.identity_verified === true;
  renderStepper(data);
  $("verify-status-badge").textContent = `${verified.length} / 3 matched`;
  $("verify-status-badge").className = identityVerified ? "tag verified" : "tag pending";
  $("pii-progress-fill").style.width = `${Math.min(100, verified.length / 3 * 100)}%`;
  $("pii-progress").setAttribute("aria-valuenow", Math.min(3, verified.length));
  for (const [field, id] of Object.entries(PII_ITEMS)) {
    const matched = verified.includes(field);
    const received = collected.includes(field);
    $(id).className = matched ? "pii-item verified" : received ? "pii-item collected" : "pii-item";
    $(id).querySelector(".pii-icon").textContent = matched ? "✓" : received ? "◐" : "○";
    $(id).setAttribute("aria-label", `${PII_LABELS[field]}: ${matched ? "matched" : received ? "collected, not matched" : "not collected"}`);
    $(id).title = matched ? "Matched against the identity record" : received ? "Collected; not matched against the record" : "Not collected";
  }
  $("data-shield-badge").textContent = identityVerified ? "Identity verified · Claim access permitted" : "Claim information is protected";
  $("data-shield-badge").className = identityVerified ? "shield-status verified" : "shield-status";
  renderMemory(state.cross_phase_memory || {});
  renderClaim(identityVerified ? data.active_case : null, identityVerified);
  renderSummary(state.post_process || {}, identityVerified);
  renderAudit(state.trace_log || data.trace || []);
  renderPolicyDecision(data);
  $("process-actions").hidden = data.current_phase !== "PROCESS_CASE";
  $("post-process-actions").hidden = data.current_phase !== "POST_PROCESS" || ["accepted", "declined"].includes(state.post_process?.user_decision);
  renderChoices(data);
}

function renderChoices(data) {
  const phase = data.current_phase;
  const terminal = ['CONCLUDED', 'ESCALATED'].includes(phase);
  $('customer-options').hidden = terminal;
  $('btn-open-verification').hidden = phase !== 'VERIFY_ID';
  $('verification-progress').hidden = phase !== 'VERIFY_ID';
  const matched = new Set(data.sop_state?.verified_fields || []).size;
  $('verification-progress').textContent = `Not verified · ${matched} of 3 details matched. Claim details are locked.`;
  $('btn-correct').hidden = phase !== 'VERIFY_ID' && phase !== 'RESOLVE_INTENT';
  const choices = $('topic-choices');
  choices.replaceChildren();
  let items = [];
  if (phase === 'VERIFY_ID' || phase === 'RESOLVE_INTENT') {
    items = [['Check progress', 'I want to check my claim status.'],
      ['Understand a rejection', 'Why was my claim denied?'],
      ['Help with documents', 'What documents does my claim need?'],
      ['Not sure where to start', "I don’t know where to start. What can I ask?"]];
    if (phase === 'RESOLVE_INTENT' && data.claim_choices?.length) {
      items = data.claim_choices.map(c => [`${c.case_type} · ${c.created_at} · ${c.case_id}`, `I mean claim ${c.case_id}.`]);
    }
  } else if (phase === 'PROCESS_CASE') {
    items = [['Check progress', 'What is my claim status?']];
    if (data.active_case?.denial_reason) items.push(['Why was it rejected?', 'Why was my claim denied?']);
    if (data.active_case?.documents_needed?.length) items.push(
      ['How to send documents', 'How do I submit the documents?'],
      ['I can’t get a document', 'I cannot get the documents. What alternatives are available?']);
    else items.push(['Understand the payment', 'What payment amounts are recorded for my claim?']);
  }
  const chooseClaim = phase === 'RESOLVE_INTENT' && data.claim_choices?.length;
  $('choices-label').textContent = chooseClaim ? 'Choose the claim you mean' : 'Need an idea? See suggested questions';
  $('suggested-topics').hidden = !items.length;
  if (chooseClaim) $('suggested-topics').open = true;
  items.forEach(([label, message]) => {
    const button = node('button', 'chip', label);
    button.type = 'button'; button.addEventListener('click', () => sendMessage(message)); choices.append(button);
  });
  if (phase !== 'VERIFY_ID' && $('verification-modal').open) $('verification-modal').close();
  if (phase !== 'VERIFY_ID' && !replaying) $('security-card-el')?.remove();
  syncControls();
}
function renderMemory(memory) {
  $("memory-chips-container").replaceChildren();
  for (const [key, label] of [["case_type_hint", "Type"], ["status_hint", "Status"], ["date_hint", "Date"], ["topic_hint", "Need"]]) {
    if (memory[key]) $("memory-chips-container").append(node("span", "memory-pill", `${label}: ${memory[key]}`));
  }
  if (!$("memory-chips-container").childElementCount) $("memory-chips-container").append(node("p", "placeholder", "No claim hints yet. Share your reason for calling at any time."));
}
function renderClaim(claim, identityVerified) {
  const container = $("claim-details-container");
  container.replaceChildren();
  $("claim-shield-status").textContent = !identityVerified ? "LOCKED" : claim ? "FIXTURE DATA" : "AWAITING MATCH";
  $("claim-shield-status").className = claim && identityVerified ? "tag verified" : "tag";
  if (!claim) {
    container.append(node("p", "placeholder", identityVerified ? "Identity verified. The agent will use your hints to identify the right claim." : "Claim details stay unavailable until the identity gate has passed."));
    return;
  }
  const rows = [["Claim", claim.case_id], ["Type", claim.case_type], ["Status", claim.status], ["Filed", claim.created_at], ["Appeal deadline", claim.appeal_deadline], ["Documents", Array.isArray(claim.documents_needed) ? claim.documents_needed.join(", ") : null], ["Net payment", claim.net_pay], ["Fee", claim.net_fee]];
  const record = node("dl", "claim-record");
  for (const [label, value] of rows) {
    if (value === undefined || value === null || value === "") continue;
    const row = node("div", "claim-row");
    row.append(node("dt", "", label), node("dd", "", value));
    record.append(row);
  }
  container.append(record);
  if (claim.denial_reason || claim.summary) container.append(node("p", "claim-note", claim.denial_reason || claim.summary));
}
function renderSummary(post, identityVerified) {
  $("summary-card").hidden = !identityVerified || (!post.email_offered && !post.draft_summary);
  $("summary-text").textContent = identityVerified ? (post.draft_summary || "Your conversation summary will appear here.") : "";
  const accepted = post.user_decision === "accepted";
  const declined = post.user_decision === "declined";
  $("summary-status").textContent = declined ? "SKIPPED" : accepted ? "SIMULATED" : "AWAITING CHOICE";
  $("summary-status").className = accepted ? "tag verified" : "tag";
  $("summary-note").textContent = declined ? "You chose to skip. No email was sent." : accepted ? "Consent recorded. Email delivery is simulated; no real email was sent." : "Sending requires your explicit choice. Delivery is simulated in this demo.";
}
function renderAudit(logs) {
  $("trace-counter").textContent = `${logs.length} events`;
  $("trace-stream").replaceChildren();
  if (!logs.length) $("trace-stream").append(node("p", "placeholder", "Gate checks and phase transitions will appear as the conversation progresses."));
  [...logs].reverse().forEach(log => {
    const entry = node("div", `trace-entry ${log.gate_passed ? "passed" : "failed"}`);
    const top = node("div", "trace-top");
    const timestamp = typeof log.timestamp === "string" ? log.timestamp.split("T")[1]?.slice(0, 8) || "" : "";
    top.append(node("span", "trace-gate", log.gate_evaluated || "Workflow event"), node("span", "trace-time", timestamp));
    entry.append(top, node("p", "trace-detail", log.details || ""));
    if (log.phase_before && log.phase_after && log.phase_before !== log.phase_after) entry.append(node("p", "phase-route", `${log.phase_before} → ${log.phase_after}`));
    $("trace-stream").append(entry);
  });
}
function renderEngine(data = {}) {
  const mode = String(data.engine_mode || "").toLowerCase();
  const fallback = mode.includes("fallback");
  const isLive = !fallback && (mode.includes("llm") || mode === "live" || (!mode && config.has_api_key && !config.use_mock_only));
  $("engine-badge").className = `engine-badge${fallback ? " fallback" : isLive ? " live" : ""}`;
  $("engine-label").textContent = fallback ? "Offline fallback" : isLive ? (mode ? "Live model" : "Live model ready") : "Offline demo";
  $('language-label').textContent = `Language: ${fallback ? 'offline fallback' : isLive ? 'live model' : 'offline'}`;
  const actualSource = data.policy_decision?.source;
  const controller = data.controller || config.controller;
  $('controller-label').textContent = `Strategy: ${CONTROLLER_LABELS[controller] || controller || 'not reported'}${actualSource && actualSource !== controller && actualSource !== 'sop' ? ` · this turn used ${CONTROLLER_LABELS[actualSource] || actualSource}` : ''}`;
  if (fallback) notice("The model could not complete this turn. The offline engine supplied the response; workflow gates remained active.");
  else if (replaying) notice("");
}
function renderPolicyDecision(data) {
  const decision = data.policy_decision;
  $('policy-decision').hidden = !decision;
  if (!decision) return;
  const action = ACTION_LABELS[decision.selected_action] || decision.selected_action;
  const source = CONTROLLER_LABELS[decision.source] || decision.source || 'SOP';
  $('policy-decision-title').textContent = action ? `This response: ${action} · ${source}` : `This response: ${source}`;
  $('policy-decision-reason').textContent = decision.reason || 'The server did not include a decision explanation.';
  const checkpoint = decision.checkpoint;
  $('policy-decision-model').hidden = !checkpoint;
  if (checkpoint) {
    const family = checkpoint.checkpoint_family === 'customer_multi_turn_v1' ? 'Customer conversation model'
      : checkpoint.checkpoint_family === 'legacy_simulator_v3' ? 'Original simulator model' : 'Trained model';
    $('policy-decision-model').textContent = `Model used: ${family}${checkpoint.environment_version ? ` · environment v${checkpoint.environment_version}` : ''}.`;
  }
  $('policy-decision-fallback').hidden = !decision.fallback_reason;
  $('policy-decision-fallback').textContent = decision.fallback_reason || '';
  const activity = data.model_activity;
  $('policy-decision-language').hidden = !activity;
  if (activity) {
    const interpretation = Number(activity.interpretation_requests || 0), planning = Number(activity.planning_requests || 0);
    const generation = {deterministic: 'local grounded reply', deterministic_grounded: 'local grounded reply',
      live_fact_plan: 'model-selected grounded facts', grounded_plan_fallback: 'local grounded fallback'}[decision.response_generation];
    $('policy-decision-language').textContent = interpretation + planning === 0
      ? 'No language-model request was sent for this turn. The reply used local grounding.'
      : `Language-model activity: ${activity.interpretation_succeeded || 0}/${interpretation} interpretation requests and ${activity.planning_succeeded || 0}/${planning} fact-planning requests succeeded.${generation ? ` Response: ${generation}.` : ''}`;
  }
  $('policy-decision-boundary').textContent = decision.forced
    ? 'The SOP required this step. The policy could not choose to skip it.'
    : 'The policy chose among the actions currently allowed by the SOP.';
  $('policy-decision-options').replaceChildren();
  (decision.allowed_actions || []).forEach(actionName => {
    const probability = decision.probabilities?.[actionName];
    const selected = actionName === decision.selected_action;
    const row = node('div', `decision-option${selected ? ' selected' : ''}`);
    row.append(node('span', '', `${selected ? '✓ ' : ''}${ACTION_LABELS[actionName] || actionName}`));
    if (typeof probability === 'number' && Number.isFinite(probability)) row.append(node('span', '', `${(probability * 100).toFixed(1)}%`));
    $('policy-decision-options').append(row);
  });
  $('policy-decision-record').textContent = JSON.stringify(decision, null, 2);
}
function applyConfig(data) {
  config = data;
  const ownerManaged = data.api_key_source === 'server';
  $('cfg-controller').value = data.controller || 'ppo42';
  $("cfg-engine-mode").value = data.has_api_key && !data.use_mock_only ? "llm" : "mock";
  $("cfg-base-url").value = data.base_url || "https://api.openai.com/v1";
  $("cfg-model").value = data.model || "gpt-4o-mini";
  $("cfg-api-key").value = "";
  $("cfg-api-key").placeholder = data.has_api_key ? "Leave blank to keep this session’s token" : "Paste your API token";
  $("llm-config-fields").hidden = $("cfg-engine-mode").value !== "llm" || ownerManaged;
  $('server-api-note').hidden = $("cfg-engine-mode").value !== "llm" || !ownerManaged;
  renderEngine();
}
async function resetSession() {
  if (busy) return;
  busy = true;
  syncControls();
  notice("");
  try {
    const data = await request("/api/reset", {method: "POST", body: JSON.stringify(sessionId ? {session_id: sessionId} : {})});
    sessionId = data.session_id;
    if (!sessionId) throw new Error("The server did not create a session. Please try New call again.");
    liveData = initialState();
    replaying = false;
    $('verification-modal').close();
    $('security-card-el')?.remove();
    liveMessages = [{role: "assistant", text: WELCOME, phase: "VERIFY_ID"}];
    snapshots = [{data: clone(liveData), messages: clone(liveMessages)}];
    $("user-input").value = "";
    renderMessages(liveMessages);
    renderInspector(liveData);
    renderReplayStatus(0);
    let sessionConfig = await request(`/api/config?session_id=${encodeURIComponent(sessionId)}`);
    if (requestedController) {
      sessionConfig = await request('/api/config', {method: 'POST', body: JSON.stringify({session_id: sessionId, controller: requestedController})});
      requestedController = null;
    }
    applyConfig(sessionConfig);
  } catch (error) {
    notice(error.message);
  } finally { busy = false; syncControls(); }
}
function closeConfig() {
  $("cfg-api-key").value = "";
  $("config-modal").close();
  $("btn-config").focus();
}
async function saveConfig(event) {
  event.preventDefault();
  if (busy || !sessionId) return;
  const mock = $("cfg-engine-mode").value === "mock";
  const token = $("cfg-api-key").value.trim();
  $("config-error").hidden = true;
  if (!mock && !token && !config.has_api_key) {
    $("config-error").textContent = "Add an API token to use the live model, or select the offline demo.";
    $("config-error").hidden = false;
    $("cfg-api-key").focus();
    return;
  }
  busy = true;
  $("config-save").disabled = true;
  $("config-save").textContent = "Saving…";
  syncControls();
  try {
    const ownerManaged = config.api_key_source === 'server';
    const payload = {session_id: sessionId, controller: $('cfg-controller').value, use_mock: mock};
    if (!ownerManaged) {
      payload.api_key = token || undefined;
      payload.base_url = $("cfg-base-url").value.trim() || undefined;
      payload.model = $("cfg-model").value.trim() || undefined;
    }
    const data = await request("/api/config", {method: "POST", body: JSON.stringify(payload)});
    applyConfig(data);
    notice(`${CONTROLLER_LABELS[data.controller] || data.controller || 'Strategy'} configured for the next turn. ${mock ? 'Language engine: offline.' : 'Live model configured; each response shows the engine actually used.'}`);
    closeConfig();
  } catch (error) {
    $("config-error").textContent = error.message;
    $("config-error").hidden = false;
  } finally {
    $("cfg-api-key").value = "";
    busy = false;
    $("config-save").disabled = false;
    $("config-save").textContent = "Save settings";
    syncControls();
  }
}

function renderVerificationCard() {
  if ($('security-card-el') || liveData.current_phase !== 'VERIFY_ID' || replaying) return;
  const card = node('form', 'security-card');
  card.id = 'security-card-el';
  card.noValidate = true;
  card.innerHTML = `
    <div class="security-card-grid">
      <div class="security-card-field"><label for="card-name">Full name <span class="field-tag">exactly as on policy</span></label><input id="card-name" autocomplete="off" maxlength="200" placeholder="Include middle name or suffix if shown"><span class="field-error" id="error-name" role="alert"></span></div>
      <div class="security-card-field"><label for="card-dob">Date of birth <span class="field-tag">YYYY-MM-DD</span></label><input type="date" id="card-dob" aria-describedby="dob-help error-dob"><span id="dob-help" class="field-help">Use the calendar or type 1985-03-15.</span><span class="field-error" id="error-dob" role="alert"></span></div>
      <div class="security-card-field"><label for="card-phone">Phone number</label><input type="tel" id="card-phone" autocomplete="off" maxlength="50" placeholder="10 digits, with optional +1" aria-describedby="error-phone"><span class="field-error" id="error-phone" role="alert"></span></div>
      <div class="security-card-field"><label for="card-email">Email address</label><input type="email" id="card-email" autocomplete="off" maxlength="200" placeholder="name@example.com" aria-describedby="error-email"><span class="field-error" id="error-email" role="alert"></span></div>
      <div class="security-card-field"><label for="card-id4">Last 4 digits of ID (optional)</label><div class="id-input-wrapper"><select id="card-id-type" aria-label="ID type"><option value="ssn_last4">SSN</option><option value="national_id_last4">National ID</option></select><input id="card-id4" inputmode="numeric" maxlength="4" pattern="[0-9]{4}" placeholder="4 digits" aria-label="Last four digits" aria-describedby="error-id_last4"></div><span class="field-error" id="error-id_last4" role="alert"></span></div>
    </div>
    <p class="field-help">You can leave unused fields blank. A National ID is an extra check; it does not replace the 3 required details.</p>
    <details class="demo-tools-tray"><summary>Demo identities — fill a sample</summary><div class="experiment-preset-chips">
      <button type="button" class="preset-chip" id="preset-margaret">Margaret Chen</button><button type="button" class="preset-chip" id="preset-ava">Ava Lopez</button><button type="button" class="preset-chip" id="preset-matian">Ma Tian</button><button type="button" class="preset-chip" id="preset-yawen">Ya Wen Li</button>
    </div><p class="field-help">Fills the form only. You review and submit it.</p></details>
    <p id="card-feedback" class="form-error" role="status" hidden></p>
    <div class="security-card-actions"><button type="button" class="btn btn-secondary" id="btn-card-fill">Fill sample data</button><button type="submit" class="btn btn-primary" id="btn-card-submit">Verify identity</button></div>`;
  $('verification-fields').append(card);
  $('card-dob').max = new Date().toISOString().slice(0, 10);
  const presets = {
    margaret: ['Margaret Chen', '1985-03-15', '+16505212836', 'margaret@email.com', '4472', 'ssn_last4'],
    ava: ['Ava Lopez', '1990-08-21', '+16503882920', 'ava.lopez@email.com', '9180', 'ssn_last4'],
    matian: ['Ma Tian', '1964-09-10', '+16502088799', 'matian@example.com', '6688', 'national_id_last4'],
    yawen: ['Ya Wen Li', '1989-12-03', '+16505212830', 'yawen.li@gmail.com', '5317', 'national_id_last4']
  };
  function fillPreset(key) {
    ['name', 'dob', 'phone', 'email', 'id4', 'id-type'].forEach((field, i) => { $(`card-${field}`).value = presets[key][i]; });
    clearCardErrors();
  }
  Object.keys(presets).forEach(key => $(`preset-${key}`).addEventListener('click', () => fillPreset(key)));
  $('btn-card-fill').addEventListener('click', () => fillPreset('margaret'));
  card.addEventListener('input', clearCardErrors);
  card.addEventListener('submit', event => {
    event.preventDefault();
    clearCardErrors();
    const form = {name: $('card-name').value.trim(), dob: $('card-dob').value,
      phone: $('card-phone').value.trim(), email: $('card-email').value.trim(),
      id_last4: $('card-id4').value.trim(), id_type: $('card-id-type').value};
    const errors = {};
    if (!$('card-dob').validity.valid) errors.dob = 'Choose a valid birth date.';
    if (!$('card-email').validity.valid) errors.email = 'Use a complete email address, such as name@example.com.';
    if (!$('card-id4').validity.valid) errors.id_last4 = 'Enter exactly 4 digits.';
    const digits = form.phone.replace(/\D/g, '');
    if (form.phone && !/^(?:1)?[0-9]{10}$/.test(digits)) errors.phone = 'Use a 10-digit US phone number, with optional +1.';
    if (Object.keys(errors).length) { showCardErrors(errors); return; }
    if (!['name', 'dob', 'phone', 'email', 'id_last4'].some(key => form[key])) {
      $('card-feedback').textContent = 'Add an identity detail, or fill a sample to try the demo.';
      $('card-feedback').hidden = false; $('card-name').focus(); return;
    }
    submitVerificationCard(form);
  });
  syncControls();
}
function clearCardErrors() {
  document.querySelectorAll('#security-card-el .field-error').forEach(el => { el.textContent = ''; });
  document.querySelectorAll('#security-card-el [aria-invalid]').forEach(el => el.removeAttribute('aria-invalid'));
  if ($('card-feedback')) $('card-feedback').hidden = true;
}
function showCardErrors(errors) {
  Object.entries(errors).forEach(([field, message]) => {
    const el = $(`card-${field === 'id_last4' ? 'id4' : field === 'id_type' ? 'id-type' : field}`);
    if (el) el.setAttribute('aria-invalid', 'true');
    if ($(`error-${field}`)) $(`error-${field}`).textContent = message;
  });
  document.querySelector('#security-card-el [aria-invalid]')?.focus();
}
function openVerification() {
  if (busy || replaying || !sessionId || liveData.current_phase !== 'VERIFY_ID') return;
  renderVerificationCard(); $('verification-modal').showModal();
}

async function submitVerificationCard(form) {
  if (busy || replaying || !sessionId || liveData.current_phase !== 'VERIFY_ID') return;
  busy = true;
  notice("");
  syncControls();
  const typing = showTyping();
  try {
    const data = await request("/api/verify-card", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, ...form })
    });
    typing.remove();
    if (!data.sop_state || typeof data.reply !== "string") throw new Error("Verification card submission failed.");
    if (data.field_errors) { showCardErrors(data.field_errors); return; }
    liveData = data;
    appendLive({ role: "user", text: "[Submitted Security Verification Card]" });
    appendLive({ role: "assistant", text: data.reply, phase: data.current_phase });
    snapshots.push({ data: clone(data), messages: clone(liveMessages) });
    renderInspector(data);
    renderEngine(data);
    renderReplayStatus(snapshots.length - 1);
    if (!data.sop_state.identity_verified && $('card-feedback')) {
      $('card-feedback').textContent = data.reply;
      $('card-feedback').hidden = false;
    }
    $("chat-window").scrollTop = $("chat-window").scrollHeight;
  } catch (error) {
    typing.remove();
    notice(error.message);
    if ($('card-feedback')) { $('card-feedback').textContent = error.message; $('card-feedback').hidden = false; }
  } finally {
    busy = false;
    syncControls();
  }
}

const toggleInspectorBtn = $("btn-toggle-inspector");
if (toggleInspectorBtn) {
  toggleInspectorBtn.addEventListener("click", () => {
    const drawer = document.querySelector(".inspector-panel");
    const layout = document.querySelector(".main-layout");
    if (drawer) {
      const isHidden = drawer.classList.toggle("hidden-drawer");
      if (layout) layout.classList.toggle("full-width-conversation", isHidden);
      toggleInspectorBtn.setAttribute("aria-expanded", String(!isHidden));
      toggleInspectorBtn.classList.toggle("active", !isHidden);
      document.querySelector('.replay-bar').hidden = isHidden;
    }
  });
}

$("chat-form").addEventListener("submit", event => { event.preventDefault(); sendMessage($("user-input").value); });
$("user-input").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); $("chat-form").requestSubmit(); }
});
Object.entries(SCENARIOS).forEach(([id, text]) => $(id).addEventListener("click", () => sendMessage(text)));
$("btn-reset").addEventListener("click", resetSession);
$('btn-open-verification').addEventListener('click', openVerification);
$('verification-close').addEventListener('click', () => $('verification-modal').close());
$('btn-human').addEventListener('click', () => sendMessage('I would like a human representative.'));
$('btn-correct').addEventListener('click', () => {
  if (liveData.current_phase === 'VERIFY_ID') openVerification();
  else { $('user-input').value = 'Correction: '; $('user-input').focus(); notice('Add the correct claim reference, type or date after “Correction:”. Review it before sending.'); }
});
$("btn-wrap-up").addEventListener("click", () => sendMessage("That answers my question. I’m ready to wrap up."));
$("btn-accept-email").addEventListener("click", () => sendMessage("Yes, please send the email summary."));
$("btn-decline-email").addEventListener("click", () => sendMessage("No thanks, please skip the email summary."));
$("time-travel-slider").addEventListener("input", event => applySnapshot(Number(event.target.value)));
$("btn-revert-live").addEventListener("click", () => { applySnapshot(snapshots.length - 1); if (!isTerminal()) $("user-input").focus(); });
$("btn-config").addEventListener("click", () => { applyConfig(config); $("config-error").hidden = true; $("config-modal").showModal(); });
$('btn-conversation-settings').addEventListener('click', () => $('btn-config').click());
$("modal-close").addEventListener("click", closeConfig);
$("modal-cancel").addEventListener("click", closeConfig);
$("config-modal").addEventListener("cancel", () => { $("cfg-api-key").value = ""; });
$("cfg-engine-mode").addEventListener("change", () => {
  const live = $("cfg-engine-mode").value === "llm";
  const ownerManaged = config.api_key_source === 'server';
  $("llm-config-fields").hidden = !live || ownerManaged;
  $('server-api-note').hidden = !live || !ownerManaged;
});
$("config-form").addEventListener("submit", saveConfig);
renderInspector(liveData);
resetSession();
