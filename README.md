# Aegis ClaimShield • Insurance SOP Agent Harness

A Standard Operating Procedure (SOP) harness for an insurance claims customer support agent that enforces strict business workflows and regulatory guardrails while preserving natural, empathetic LLM conversation.

---

## Key Architecture & Capabilities

### 1. The 4-Phase Fixed Business Workflow

```
[ VERIFY_ID ]  ──(3+ PII items)──►  [ RESOLVE_INTENT ]  ──(Grounded Match)──►  [ PROCESS_CASE ]  ──(Issue Resolved)──►  [ POST_PROCESS ]  ──(Consent)──►  [ CONCLUDED ]
      │                                                                               │                                            │
      ▼ (Refusals / Anger)                                                            ▼ (Repeated Out of Scope)                    ▼ (Skipped)
[ EMPATHY / ESCALATE ]                                                        [ ESCALATED TO HUMAN ]                        [ CONCLUDED ]
```

1. **`VERIFY_ID` (Strict SOP Security Gate)**:
   - **Physical Data Shielding**: Claim data is **never** injected into the LLM context during this phase. Hallucination or leak of claim details is architecturally impossible.
   - **Verification Requirement**: Requires at least 3 matching PII items from `[Full Name, Date of Birth, Phone, Email, Policy Number, SSN/National ID last 4 digits]`.
   - **Proactive Memory Capture**: If caller mentions claim details early (e.g. *"I'm calling about my denied healthcare claim from January"*), the harness stays in `VERIFY_ID` but caches the intent and case hints into `cross_phase_memory`.
   - **Emotional Support & Recovery**: When callers express frustration or refusal (*"I already told you who I am. Just tell me why my claim was denied"*), the agent acknowledges the frustration with empathy, explains *why* SOP verification is mandatory (protecting confidential health/financial data under privacy regulations), offers alternative ID fields (e.g. phone/email/policy number instead of SSN), and moves forward without leaking claim records.

2. **`RESOLVE_INTENT` (Cross-Phase Memory Resolution)**:
   - Reads `cross_phase_memory` populated during earlier turns.
   - For Margaret Chen: matches `party_id=P9`, `case_type=healthcare`, `status=denied`, `date=January` -> automatically resolves to claim `CL-2048`.
   - Transitions directly to `PROCESS_CASE` without asking the customer to repeat themselves from scratch.

3. **`PROCESS_CASE` (Grounded Claim Reasoning & Guardrails)**:
   - Grounded strictly in `claims.json`, `required_document_guideline.json`, and `claim_schema.json`.
   - Accurately explains denial reason (*missing pathology report and treating provider office note*), required documents, alternative documents, submission timing (*within a week*), review time (*usually less than a week*), and appeal deadline (*March 18, 2026*).
   - **Scope Guardrail**: Rejects out-of-scope inquiries (*"What is RL?"*) politely and re-centers on insurance. Consecutive out-of-scope attempts automatically trigger human escalation.

4. **`POST_PROCESS` (Mandatory Email Summary & Explicit Consent)**:
   - Generates an email summary covering:
     1. What was discussed (healthcare claim review).
     2. Claim status & outcome (denied for missing pathology report and office note; appeal deadline March 18, 2026).
     3. Major follow-up items / next steps (submit documents via member portal within a week).
   - Prompts caller whether they want the summary sent to their email on file (`margaret@email.com`) or prefer to skip it.
   - Customer choice governs the outcome (`yes` -> email sent; `no` -> skipped, gracefully concludes).

---

## Quick Start Guide

### Option 1: Local Setup (Recommended for immediate testing)

Requirements: Python 3.10+

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run automated test suite**:
   ```bash
   python3 -m pytest tests/ -v
   ```

3. **Start the application**:
   ```bash
   bash run.sh
   # or
   python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
   ```

4. Open your browser at **`http://localhost:8000`**.

---

### Option 2: Docker / Docker Compose

1. **Build and start with Docker Compose**:
   ```bash
   docker compose up --build
   ```

2. Open **`http://localhost:8000`** in your browser.

---

## AI Model & API Configuration

The system includes a **Dual-Engine Architecture**:

1. **Built-in Deterministic Simulation Engine (Default)**:
   - Enabled out-of-the-box with **zero API keys or tokens required**.
   - Fully passes all automated unit tests and demo scenarios.
2. **Live LLM Engine**:
   - Supports any OpenAI / Gemini / Anthropic / OpenRouter compatible API endpoint.
   - Configure via the Web UI (**"API Settings"** button) or via environment variables:
     ```bash
     export AI_API_KEY="your-api-key"
     export AI_BASE_URL="https://api.openai.com/v1" # or OpenRouter/Gemini endpoint
     export AI_MODEL="gpt-4o-mini"
     ```

---

## Pre-Loaded Interactive Test Scenarios in UI

The web interface includes 4 one-click scenario chips for instant evaluation:

1. **🎯 Margaret Chen Demo Case**:
   - Utterance: *"I’m the policyholder. My name is Margaret Chen, policy POL-9921. I’m calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."*
   - Verifies 4 PII items, stores January denied healthcare hint, unlocks `CL-2048`, explains denial reason and appeal deadline.
2. **⚡ Frustrated Caller Recovery**:
   - Utterance: *"I already told you who I am. This is ridiculous. Just tell me why my claim was denied."*
   - Agent acknowledges frustration, explains privacy regulations, offers alternative verification fields, and protects claim data from premature disclosure.
3. **🛡️ Out-of-Scope Guardrail**:
   - Utterance: *"What is RL?"* -> Polite refusal.
   - Follow-up out of scope -> Escalation to human representative.
4. **👤 Step-by-Step Progressive Verification**:
   - Demonstrates multi-turn identity verification where caller provides partial info across turns.

---

## Automated Test Coverage

Run the comprehensive pytest suite:
```bash
python3 -m pytest tests/ -v
```

Test files:
- `tests/test_demo_margaret_chen.py`: Complete one-shot Margaret Chen demo case and follow-up Q&A.
- `tests/test_emotional_recovery.py`: Frustration de-escalation, privacy justification, and human escalation.
- `tests/test_scope_guardrail.py`: Out-of-scope question rejection and consecutive attempt escalation.
- `tests/test_post_process.py`: Email summary structure (3 required components) and skip/accept consent.
- `tests/test_partial_pii.py`: Progressive verification across multiple conversation turns.
