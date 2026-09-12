let sessionId = "session_" + Math.random().toString(36).substring(2, 10);
let currentPhase = "VERIFY_ID";

// DOM Elements
const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const btnReset = document.getElementById("btn-reset");
const btnConfig = document.getElementById("btn-config");
const configModal = document.getElementById("config-modal");
const modalClose = document.getElementById("modal-close");
const modalCancel = document.getElementById("modal-cancel");
const configForm = document.getElementById("config-form");
const cfgEngineMode = document.getElementById("cfg-engine-mode");
const llmConfigFields = document.getElementById("llm-config-fields");
const engineBadge = document.getElementById("engine-badge");
const engineLabel = document.getElementById("engine-label");
const postProcessActions = document.getElementById("post-process-actions");
const btnAcceptEmail = document.getElementById("btn-accept-email");
const btnDeclineEmail = document.getElementById("btn-decline-email");

// Time-Travel Debugger Elements
const timeTravelSlider = document.getElementById("time-travel-slider");
const timeTravelBadge = document.getElementById("time-travel-badge");
const btnRevertLive = document.getElementById("btn-revert-live");

let timeTravelSnapshots = [];

// Inspector Elements
const dataShieldBadge = document.getElementById("data-shield-badge");
const verifyStatusBadge = document.getElementById("verify-status-badge");
const piiProgressFill = document.getElementById("pii-progress-fill");
const phCard = document.getElementById("ph-card");
const phNameVal = document.getElementById("ph-name-val");
const phMetaVal = document.getElementById("ph-meta-val");
const memoryChipsContainer = document.getElementById("memory-chips-container");
const claimShieldStatus = document.getElementById("claim-shield-status");
const claimDetailsContainer = document.getElementById("claim-details-container");
const traceStream = document.getElementById("trace-stream");
const traceCounter = document.getElementById("trace-counter");

// Stepper Elements
const stepVerify = document.getElementById("step-verify");
const stepIntent = document.getElementById("step-intent");
const stepProcess = document.getElementById("step-process");
const stepPost = document.getElementById("step-post");
const div1 = document.getElementById("div-1");
const div2 = document.getElementById("div-2");
const div3 = document.getElementById("div-3");

// PII Grid Items
const piiElements = {
  name: document.getElementById("pii-name"),
  dob: document.getElementById("pii-dob"),
  policy_number: document.getElementById("pii-policy"),
  id_last4: document.getElementById("pii-id4"),
  phone: document.getElementById("pii-phone"),
  email: document.getElementById("pii-email"),
};

// Initialize
window.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  fetchConfig();
});

function setupEventListeners() {
  // Chat submit
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text) return;
    sendMessage(text);
    userInput.value = "";
    userInput.style.height = "auto";
  });

  // Enter to send (Shift+Enter for newline)
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Reset Session
  btnReset.addEventListener("click", resetSession);

  // Quick Scenarios
  document.getElementById("scenario-margaret").addEventListener("click", () => {
    sendMessage("I’m the policyholder. My name is Margaret Chen, policy POL-9921. I’m calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.");
  });

  document.getElementById("scenario-proxy")?.addEventListener("click", () => {
    sendMessage("I am David Chen calling on behalf of my mother Margaret Chen POL-9921, DOB 1985-03-15, SSN 4472. Calling about her denied healthcare claim from January.");
  });

  document.getElementById("scenario-frustrated").addEventListener("click", () => {
    sendMessage("I already told you who I am. This is ridiculous. Just tell me why my claim was denied.");
  });

  document.getElementById("scenario-oos").addEventListener("click", () => {
    sendMessage("What is RL?");
  });

  document.getElementById("scenario-partial").addEventListener("click", () => {
    sendMessage("Hi, my name is Ava Lopez.");
  });

  // Time-Travel Slider Event Listener
  timeTravelSlider.addEventListener("input", (e) => {
    const idx = parseInt(e.target.value, 10);
    if (idx >= 0 && idx < timeTravelSnapshots.length) {
      applySnapshot(idx);
    }
  });

  // Return to Live Button
  btnRevertLive.addEventListener("click", () => {
    if (timeTravelSnapshots.length > 0) {
      applySnapshot(timeTravelSnapshots.length - 1, true);
    }
  });

  // Post Process Quick Consent
  btnAcceptEmail.addEventListener("click", () => {
    sendMessage("Yes, please send the summary to my email.");
  });

  btnDeclineEmail.addEventListener("click", () => {
    sendMessage("No thanks, skip it.");
  });

  // Config Modal
  btnConfig.addEventListener("click", () => {
    configModal.style.display = "flex";
  });
  modalClose.addEventListener("click", () => { configModal.style.display = "none"; });
  modalCancel.addEventListener("click", () => { configModal.style.display = "none"; });

  cfgEngineMode.addEventListener("change", () => {
    llmConfigFields.style.display = cfgEngineMode.value === "llm" ? "block" : "none";
  });

  configForm.addEventListener("submit", saveConfig);
}

async function sendMessage(text) {
  appendMessage("user", text);

  // Show typing bubble
  const typingBubble = showTypingIndicator();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text })
    });

    typingBubble.remove();

    if (!res.ok) {
      appendMessage("assistant", "An error occurred communicating with the SOP server.", "ERROR");
      return;
    }

    const data = await res.json();
    appendMessage("assistant", data.reply, data.current_phase);
    updateInspector(data);

    // Save snapshot for Time-Travel Debugger
    timeTravelSnapshots.push({
      chatHTML: chatWindow.innerHTML,
      data: data
    });
    timeTravelSlider.disabled = false;
    timeTravelSlider.min = 0;
    timeTravelSlider.max = timeTravelSnapshots.length - 1;
    timeTravelSlider.value = timeTravelSnapshots.length - 1;
    timeTravelBadge.innerText = `Live Turn ${timeTravelSnapshots.length}`;
    timeTravelBadge.style.background = "rgba(56, 189, 248, 0.15)";
    btnRevertLive.style.display = "none";

  } catch (err) {
    typingBubble.remove();
    appendMessage("assistant", "Network connection failed. Please check the backend.", "ERROR");
  }
}

function applySnapshot(idx, isLive = false) {
  const snap = timeTravelSnapshots[idx];
  if (!snap) return;

  chatWindow.innerHTML = snap.chatHTML;
  updateInspector(snap.data);

  timeTravelSlider.value = idx;
  const isActuallyLive = isLive || (idx === timeTravelSnapshots.length - 1);
  if (isActuallyLive) {
    timeTravelBadge.innerText = `Live Turn ${timeTravelSnapshots.length}`;
    timeTravelBadge.style.background = "rgba(56, 189, 248, 0.15)";
    btnRevertLive.style.display = "none";
  } else {
    timeTravelBadge.innerText = `Replaying Turn ${idx + 1} / ${timeTravelSnapshots.length}`;
    timeTravelBadge.style.background = "rgba(245, 158, 11, 0.2)";
    btnRevertLive.style.display = "inline-block";
  }
}

function appendMessage(role, text, phase = null) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `chat-msg ${role}`;

  const avatarDiv = document.createElement("div");
  avatarDiv.className = `msg-avatar ${role}`;
  if (role === "assistant") {
    avatarDiv.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>`;
  } else {
    avatarDiv.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;
  }

  const bubbleDiv = document.createElement("div");
  bubbleDiv.className = "msg-bubble";

  const metaDiv = document.createElement("div");
  metaDiv.className = "msg-meta";

  const senderSpan = document.createElement("span");
  senderSpan.className = "msg-sender";
  senderSpan.innerText = role === "assistant" ? "Insurance SOP Agent" : "Customer";
  metaDiv.appendChild(senderSpan);

  if (phase && role === "assistant") {
    const phaseSpan = document.createElement("span");
    phaseSpan.className = `badge-phase badge-${phase.toLowerCase()}`;
    phaseSpan.innerText = phase;
    metaDiv.appendChild(phaseSpan);
  }

  const textDiv = document.createElement("div");
  textDiv.className = "msg-text";
  textDiv.innerText = text;

  bubbleDiv.appendChild(metaDiv);
  bubbleDiv.appendChild(textDiv);

  msgDiv.appendChild(avatarDiv);
  msgDiv.appendChild(bubbleDiv);

  chatWindow.appendChild(msgDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function showTypingIndicator() {
  const typingDiv = document.createElement("div");
  typingDiv.className = "chat-msg assistant";
  typingDiv.id = "typing-indicator";
  typingDiv.innerHTML = `
    <div class="msg-avatar assistant">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
      </svg>
    </div>
    <div class="msg-bubble" style="opacity: 0.7;">
      <div class="msg-text">Evaluating SOP gates & formulating response...</div>
    </div>
  `;
  chatWindow.appendChild(typingDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return typingDiv;
}

function updateInspector(data) {
  const state = data.sop_state;
  const phase = data.current_phase;
  currentPhase = phase;

  // 1. Update Stepper Pipeline
  updateStepper(phase);

  // 2. Data Shield Badge
  const isShielded = (phase === "VERIFY_ID");
  if (isShielded) {
    dataShieldBadge.className = "badge-shield active";
    dataShieldBadge.innerText = "🔒 DATA SHIELD ACTIVE";
    claimShieldStatus.className = "badge-shield active";
    claimShieldStatus.innerText = "Shielded";
  } else {
    dataShieldBadge.className = "badge-shield unlocked";
    dataShieldBadge.innerText = "🔓 DATA SHIELD UNLOCKED";
    claimShieldStatus.className = "badge-shield unlocked";
    claimShieldStatus.innerText = "Unlocked";
  }

  // 3. Verification Meter & PII Grid
  const verifiedList = state.verified_fields || [];
  const count = verifiedList.length;
  verifyStatusBadge.innerText = `${count} / 3 PII Verified`;
  if (count >= 3) {
    verifyStatusBadge.className = "badge-status verified";
  } else {
    verifyStatusBadge.className = "badge-status pending";
  }

  const pct = Math.min(100, Math.round((count / 3) * 100));
  piiProgressFill.style.width = `${pct}%`;

  // Update PII Grid Checkmarks
  Object.keys(piiElements).forEach((key) => {
    const el = piiElements[key];
    if (!el) return;
    const isVerified = verifiedList.includes(key);
    if (isVerified) {
      el.className = "pii-item verified";
      el.querySelector(".pii-icon").innerText = "✓";
    } else {
      el.className = "pii-item";
      el.querySelector(".pii-icon").innerText = "✗";
    }
  });

  // Policyholder card
  if (data.verified_policyholder) {
    phCard.style.display = "block";
    phNameVal.innerText = data.verified_policyholder.name;
    phMetaVal.innerText = `${data.verified_policyholder.policy_number} • ${data.verified_policyholder.email}`;
  } else {
    phCard.style.display = "none";
  }

  // 4. Cross-Phase Memory Drawer
  const mem = state.cross_phase_memory;
  memoryChipsContainer.innerHTML = "";
  const pills = [];
  if (mem.case_type_hint) pills.push(`Type: ${mem.case_type_hint}`);
  if (mem.status_hint) pills.push(`Status: ${mem.status_hint}`);
  if (mem.date_hint) pills.push(`Date: ${mem.date_hint}`);

  if (pills.length === 0) {
    memoryChipsContainer.innerHTML = `<span class="memory-placeholder">No early hints captured yet.</span>`;
  } else {
    pills.forEach((p) => {
      const chip = document.createElement("span");
      chip.className = "memory-pill";
      chip.innerText = p;
      memoryChipsContainer.appendChild(chip);
    });
  }

  // 5. Active Case Details (Shielded vs Unlocked)
  if (data.active_case && !isShielded) {
    const ac = data.active_case;
    claimDetailsContainer.innerHTML = `
      <div class="claim-unlocked-card">
        <div class="claim-row">
          <span class="claim-label">Case ID</span>
          <span class="claim-val">${ac.case_id} (${ac.case_type})</span>
        </div>
        <div class="claim-row">
          <span class="claim-label">Status</span>
          <span class="claim-val" style="color: #f43f5e; font-weight: 700;">${ac.status.toUpperCase()}</span>
        </div>
        <div class="claim-row">
          <span class="claim-label">Appeal Deadline</span>
          <span class="claim-val">${ac.appeal_deadline || "N/A"}</span>
        </div>
        <div class="claim-row">
          <span class="claim-label">Missing Documents</span>
          <span class="claim-val">${(ac.documents_needed || []).join(", ")}</span>
        </div>
        <div class="claim-row">
          <span class="claim-label">Net Pay / Fee</span>
          <span class="claim-val">$${ac.net_pay} / $${ac.net_fee}</span>
        </div>
        <div class="claim-denial-box">
          <strong>Denial Reason:</strong> ${ac.denial_reason || ac.summary}
        </div>
      </div>
    `;
  } else {
    claimDetailsContainer.innerHTML = `
      <div class="shielded-placeholder">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
          <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
        </svg>
        <p>Protected health & claim information is shielded by SOP policy until 3 PII items are verified.</p>
      </div>
    `;
  }

  // 6. Post-Process Quick Action Buttons
  if (phase === "POST_PROCESS") {
    postProcessActions.style.display = "flex";
  } else {
    postProcessActions.style.display = "none";
  }

  // 7. Audit Log Stream
  const logs = state.trace_log || [];
  traceCounter.innerText = `${logs.length} events`;
  traceStream.innerHTML = "";
  logs.slice().reverse().forEach((log) => {
    const entry = document.createElement("div");
    entry.className = `trace-entry ${log.gate_passed ? "passed" : "failed"}`;
    const timeStr = log.timestamp.split("T")[1]?.substring(0, 8) || "";
    entry.innerHTML = `
      <div class="trace-top">
        <span class="trace-gate">${log.gate_evaluated}</span>
        <span>${timeStr}</span>
      </div>
      <div class="trace-detail">${log.details}</div>
    `;
    traceStream.appendChild(entry);
  });
}

function updateStepper(phase) {
  [stepVerify, stepIntent, stepProcess, stepPost].forEach(s => s.className = "step-item");
  [div1, div2, div3].forEach(d => d.className = "step-divider");

  if (phase === "VERIFY_ID") {
    stepVerify.className = "step-item active";
  } else if (phase === "RESOLVE_INTENT") {
    stepVerify.className = "step-item completed";
    div1.className = "step-divider completed";
    stepIntent.className = "step-item active";
  } else if (phase === "PROCESS_CASE") {
    stepVerify.className = "step-item completed";
    div1.className = "step-divider completed";
    stepIntent.className = "step-item completed";
    div2.className = "step-divider completed";
    stepProcess.className = "step-item active";
  } else if (phase === "POST_PROCESS") {
    stepVerify.className = "step-item completed";
    div1.className = "step-divider completed";
    stepIntent.className = "step-item completed";
    div2.className = "step-divider completed";
    stepProcess.className = "step-item completed";
    div3.className = "step-divider completed";
    stepPost.className = "step-item active";
  } else if (phase === "CONCLUDED" || phase === "ESCALATED") {
    stepVerify.className = "step-item completed";
    div1.className = "step-divider completed";
    stepIntent.className = "step-item completed";
    div2.className = "step-divider completed";
    stepProcess.className = "step-item completed";
    div3.className = "step-divider completed";
    stepPost.className = "step-item completed";
  }
}

async function resetSession() {
  try {
    const res = await fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    const data = await res.json();
    sessionId = data.session_id;

    // Reset Chat Window
    chatWindow.innerHTML = `
      <div class="chat-msg system">
        <div class="msg-avatar system">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
        </div>
        <div class="msg-bubble">
          <div class="msg-meta">
            <span class="msg-sender">Aegis SOP Engine</span>
            <span class="badge-phase badge-verify">VERIFY_ID</span>
          </div>
          <div class="msg-text">
            Session reset. Identity verification gate active: at least 3 PII items required before claim access.
          </div>
        </div>
      </div>
    `;

    // Reset Stepper and Inspector
    updateStepper("VERIFY_ID");
    verifyStatusBadge.innerText = "0 / 3 PII Verified";
    verifyStatusBadge.className = "badge-status pending";
    piiProgressFill.style.width = "0%";
    Object.keys(piiElements).forEach(k => {
      piiElements[k].className = "pii-item";
      piiElements[k].querySelector(".pii-icon").innerText = "✗";
    });
    phCard.style.display = "none";
    memoryChipsContainer.innerHTML = `<span class="memory-placeholder">No early hints captured yet.</span>`;
    dataShieldBadge.className = "badge-shield active";
    dataShieldBadge.innerText = "🔒 DATA SHIELD ACTIVE";
    claimDetailsContainer.innerHTML = `
      <div class="shielded-placeholder">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
          <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
        </svg>
        <p>Protected health & claim information is shielded by SOP policy until 3 PII items are verified.</p>
      </div>
    `;
    traceCounter.innerText = "0 events";
    traceStream.innerHTML = `<div class="trace-empty">Audit events will stream here in real time...</div>`;
    postProcessActions.style.display = "none";

    // Reset Time-Travel State
    timeTravelSnapshots = [];
    timeTravelSlider.disabled = true;
    timeTravelSlider.min = 0;
    timeTravelSlider.max = 0;
    timeTravelSlider.value = 0;
    timeTravelBadge.innerText = "Live Turn 0";
    btnRevertLive.style.display = "none";

  } catch (err) {
    console.error("Failed to reset session:", err);
  }
}

async function fetchConfig() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) return;
    const cfg = await res.json();
    if (cfg.use_mock_only || !cfg.has_api_key) {
      engineLabel.innerText = "Mock Engine (Deterministic)";
      cfgEngineMode.value = "mock";
      llmConfigFields.style.display = "none";
    } else {
      engineLabel.innerText = `Live LLM (${cfg.model})`;
      cfgEngineMode.value = "llm";
      llmConfigFields.style.display = "block";
    }
  } catch (err) {
    console.warn("Could not load config:", err);
  }
}

async function saveConfig(e) {
  e.preventDefault();
  const isMock = cfgEngineMode.value === "mock";
  const apiKey = document.getElementById("cfg-api-key").value.trim() || undefined;
  const baseUrl = document.getElementById("cfg-base-url").value.trim() || undefined;
  const model = document.getElementById("cfg-model").value.trim() || undefined;

  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        use_mock: isMock,
        api_key: apiKey,
        base_url: baseUrl,
        model: model
      })
    });

    if (res.ok) {
      const cfg = await res.json();
      if (cfg.use_mock_only || !cfg.has_api_key) {
        engineLabel.innerText = "Mock Engine (Deterministic)";
      } else {
        engineLabel.innerText = `Live LLM (${cfg.model})`;
      }
      configModal.style.display = "none";
    }
  } catch (err) {
    alert("Failed to save config: " + err);
  }
}
