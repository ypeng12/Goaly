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
const WELCOME = "Hello, I’m your insurance claims support assistant. I can help you understand a claim and what to do next.\n\nBefore I access claim information, I’ll need to verify at least three details: your full name, date of birth, phone number, email, or the last four digits of your SSN. You can share whichever of those you prefer, one at a time or together.\n\nWhat brings you in today?";
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
  $("btn-reset").disabled = busy;
  $("btn-config").disabled = busy || replaying || !sessionId;
  $("time-travel-slider").disabled = busy || snapshots.length < 2;
  $("replay-notice").hidden = !replaying;
  $("inspector-mode").textContent = replaying ? "REPLAY" : "LIVE";
  $("inspector-mode").className = replaying ? "tag pending" : "tag verified";
  $("chat-window").setAttribute("aria-busy", String(busy));
  $("user-input").placeholder = replaying ? "Return to live to continue…" : isTerminal() ? "This call has ended. Start a new call to continue." : "Tell us what you need help with…";
}
function renderMessage(message) {
  const role = ["user", "assistant", "system"].includes(message.role) ? message.role : "system";
  const row = node("div", `chat-msg ${role}`);
  const avatar = node("div", "msg-avatar", role === "assistant" ? "a" : role === "user" ? "You" : "!");
  avatar.setAttribute("aria-hidden", "true");
  const bubble = node("div", "msg-bubble");
  const meta = node("div", "msg-meta");
  meta.append(node("span", "msg-sender", role === "assistant" ? "Aegis · Claims support" : role === "user" ? "You" : "Connection notice"));
  if (message.phase) meta.append(node("span", "msg-phase", message.phase));
  bubble.append(meta, node("div", "msg-text", message.text));
  row.append(avatar, bubble);
  $("chat-window").append(row);
  $("chat-window").scrollTop = $("chat-window").scrollHeight;
}
function renderMessages(messages) {
  $("chat-window").replaceChildren();
  messages.forEach(renderMessage);
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
    if (!isTerminal()) $("user-input").focus();
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
  $("process-actions").hidden = data.current_phase !== "PROCESS_CASE";
  $("post-process-actions").hidden = data.current_phase !== "POST_PROCESS" || ["accepted", "declined"].includes(state.post_process?.user_decision);
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
  if (fallback) notice("The model could not complete this turn. The offline engine supplied the response; workflow gates remained active.");
  else if (replaying) notice("");
}
function applyConfig(data) {
  config = data;
  $("cfg-engine-mode").value = data.has_api_key && !data.use_mock_only ? "llm" : "mock";
  $("cfg-base-url").value = data.base_url || "https://api.openai.com/v1";
  $("cfg-model").value = data.model || "gpt-4o-mini";
  $("cfg-api-key").value = "";
  $("cfg-api-key").placeholder = data.has_api_key ? "Leave blank to keep this session’s token" : "Paste your API token";
  $("llm-config-fields").hidden = $("cfg-engine-mode").value !== "llm";
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
    liveMessages = [{role: "assistant", text: WELCOME, phase: "VERIFY_ID"}];
    snapshots = [{data: clone(liveData), messages: clone(liveMessages)}];
    $("user-input").value = "";
    renderMessages(liveMessages);
    renderInspector(liveData);
    renderReplayStatus(0);
    applyConfig(await request(`/api/config?session_id=${encodeURIComponent(sessionId)}`));
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
    const data = await request("/api/config", {method: "POST", body: JSON.stringify({session_id: sessionId, use_mock: mock, api_key: token || undefined, base_url: $("cfg-base-url").value.trim() || undefined, model: $("cfg-model").value.trim() || undefined})});
    applyConfig(data);
    notice(mock ? "Offline demo enabled for this session." : "Live model configured for this session. The next response will show the engine actually used.");
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

$("chat-form").addEventListener("submit", event => { event.preventDefault(); sendMessage($("user-input").value); });
$("user-input").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); $("chat-form").requestSubmit(); }
});
Object.entries(SCENARIOS).forEach(([id, text]) => $(id).addEventListener("click", () => sendMessage(text)));
$("btn-reset").addEventListener("click", resetSession);
$("btn-wrap-up").addEventListener("click", () => sendMessage("That answers my question. I’m ready to wrap up."));
$("btn-accept-email").addEventListener("click", () => sendMessage("Yes, please send the email summary."));
$("btn-decline-email").addEventListener("click", () => sendMessage("No thanks, please skip the email summary."));
$("time-travel-slider").addEventListener("input", event => applySnapshot(Number(event.target.value)));
$("btn-revert-live").addEventListener("click", () => { applySnapshot(snapshots.length - 1); if (!isTerminal()) $("user-input").focus(); });
$("btn-config").addEventListener("click", () => { applyConfig(config); $("config-error").hidden = true; $("config-modal").showModal(); });
$("modal-close").addEventListener("click", closeConfig);
$("modal-cancel").addEventListener("click", closeConfig);
$("config-modal").addEventListener("cancel", () => { $("cfg-api-key").value = ""; });
$("cfg-engine-mode").addEventListener("change", () => { $("llm-config-fields").hidden = $("cfg-engine-mode").value !== "llm"; });
$("config-form").addEventListener("submit", saveConfig);
renderInspector(liveData);
resetSession();
