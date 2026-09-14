"""Apply the trained dialogue-act policy to a real, already-assimilated turn.

This adapter never advances the SOP, invents a caller message, or calls a
simulation environment. The caller is the human using the chat UI. Its mask is
stricter than the training mask: answering once does not authorize ending a
real customer's conversation. Policy choices govern the order and form of the
response; protected facts and side effects remain under SOP authority.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import pickle
import re

from .dialogue import dialogue_signals
from .rl_baseline import RuleBasedPolicy
from .types import AGENT_ACTIONS, AgentAction, Phase


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = {
    "ppo42": ROOT / "artifacts/customer_policy/seed42.pt",
    "ppo7": ROOT / "artifacts/customer_policy/seed7.pt",
}
LEGACY_CHECKPOINTS = {"ppo42": ROOT / "artifacts/ppo_policy.pt", "ppo7": ROOT / "artifacts/seed7/ppo_policy.pt"}
CONTROLLER_MODES = ("rule", "ppo42", "ppo7")
UNRESOLVED_CASE_STATUSES = {"no_match", "ambiguous"}
MEMORY_KEYS = ("case_type_hint", "status_hint", "date_hint", "topic_hint")
IDENTITY_LABELS = {
    "name": "full name", "dob": "date of birth", "phone": "phone number",
    "email": "email address", "id_last4": "SSN last four digits",
}


def _signals(message, context=None):
    signals = dialogue_signals(message)
    emotion = (context or {}).get("emotion")
    # Only a bounded server-produced interpretation can supplement the text
    # features. It affects response order, never identity, ownership or consent.
    if emotion in {"neutral", "frustration", "anger", "anxiety", "confusion"}:
        signals["emotion"] = emotion
    return signals


@lru_cache(maxsize=4)
def _load_policy(mode, modified_ns, size, legacy=False):
    """Allowlisted, version-checked CPU inference; no caller-supplied paths."""
    import torch
    from ..rl.featurizer import StateFeaturizer
    from ..rl.models import ActorCriticPolicy

    path = (LEGACY_CHECKPOINTS if legacy else CHECKPOINTS)[mode]
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected_version = 3 if legacy else 4
    if (checkpoint.get("feature_version") != StateFeaturizer.VERSION
            or checkpoint.get("environment_version") != expected_version
            or (not legacy and (checkpoint.get("checkpoint_family") != "customer_multi_turn_v1"
                                or checkpoint.get("mask_version") != "customer_sop_v1"))):
        raise ValueError("Checkpoint feature or environment version is incompatible")
    config = checkpoint["config"]
    featurizer = StateFeaturizer(
        max_turns=config["max_turns"],
        use_emotion_features=config.get("use_emotion_features", True),
    )
    # Loading a model must not perturb an unrelated training process's RNG.
    with torch.random.fork_rng(devices=[]):
        policy = ActorCriticPolicy(state_dim=featurizer.FEATURE_DIM, action_dim=len(AGENT_ACTIONS))
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    metadata = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "training_seed": config["seed"],
        "actual_steps": checkpoint["history"][-1]["global_step"],
        "environment_version": checkpoint["environment_version"],
        "feature_version": checkpoint["feature_version"],
        "checkpoint_family": checkpoint.get("checkpoint_family", "legacy_simulator_v3"),
        "mask_version": checkpoint.get("mask_version", "legacy_training_mask"),
    }
    return policy, featurizer, metadata


def _identity_field(machine, message, history, runtime):
    """Ask for an unverified field, respecting a caller's stated alternative.

    These are requests, never guessed or fuzzily matched identity values.
    Previous requests do not count as matches. A field already supplied but
    mismatched is offered for correction rather than silently discarded.
    """
    fields = machine.state.accumulated_pii
    verified = set(machine.state.verified_fields)
    low = message.lower()
    avoided = set(runtime.get("avoided_identity_fields", []))
    # Keep the preference across turns using actual human history only.
    caller_text = " ".join(row.get("content", "") for row in history[-20:] if row.get("role") == "user") + " " + low
    if re.search(r"(?:no|not|never|avoid|rather|instead|without|don't|won't|cannot|can't).{0,35}(?:ssn|social security|government id)|(?:ssn|social security).{0,25}(?:private|sensitive)", caller_text, re.I):
        avoided.add("id_last4")
    preferred = next((f for f in ("phone", "email", "dob", "name")
                      if f not in verified and re.search(
                          rf"(?:use|give|share|provide|prefer).{{0,20}}\b{'date of birth|birthday|dob' if f == 'dob' else f}\b", low)), None)
    order = list(IDENTITY_LABELS)
    if preferred:
        order.remove(preferred)
        order.insert(0, preferred)
    # Do not repeatedly ask for a value that is already present. An invalid or
    # conflicting value needs an explicit correction, which the UI supports.
    conflicts = set(machine.state.pii_conflicts)
    if conflicts:
        return next((f for f in order if f in conflicts), "name"), True, sorted(avoided)
    missing = [f for f in order if f not in verified and f not in avoided and not getattr(fields, f)]
    if missing:
        return missing[0], False, sorted(avoided)
    unresolved = [f for f in order if f not in verified and f not in avoided]
    return (unresolved[0] if unresolved else "email"), True, sorted(avoided)


def _pending_question(machine, message, context):
    return bool(
        context.get("needs_clarification") or context.get("is_followup") or context.get("is_contextual_followup")
        or set(context.get("response_topics", [])) - {"unknown"}
        or machine._has_pending_question(message)
    )


def _observation(machine, message, runtime, mask, context=None):
    """The original 30-feature semantics, without raw PII in the audit payload."""
    state = machine.state
    signals = _signals(message, context)
    verified = machine.get_verified_policyholder() is not None
    return {
        "phase": state.phase.value,
        "verified_fields": list(state.verified_fields),
        # Featurizer consumes presence only; no names, dates or case IDs needed.
        "memory_slots": {key: "provided" if getattr(state.cross_phase_memory, key) else "" for key in MEMORY_KEYS},
        "data_shield_active": not verified,
        "active_case_id": "selected" if machine.get_active_claim() else "",
        "emotion": signals["emotion"],
        "privacy_concern": signals["privacy_concern"],
        "refusal": signals["is_refusal"],
        "demands_human": signals["demands_human"],
        "resolution_attempts": runtime.get("resolution_attempts", 0),
        "grounded_answered": bool(runtime.get("grounded_answered", False)),
        "action_mask": mask,
    }


def decide(machine, message, history, controller_mode="ppo42", runtime_state=None, conversation_context=None):
    """Return an auditable response action after ``machine.evaluate_turn``.

    ``conversation_context`` is bounded interpretation, not authorization. It
    may contain ``response_topics``, ``needs_clarification``,
    ``clarification_question``, ``is_followup``, and ``public_help``. The caller
    must send the actual response before calling ``record_execution``.
    """
    if controller_mode not in CONTROLLER_MODES:
        raise ValueError("Unknown customer controller")
    runtime = runtime_state or {}
    context = conversation_context or {}
    state = machine.state
    phase = state.phase
    verified = machine.get_verified_policyholder() is not None
    active = machine.get_active_claim() is not None
    signals = _signals(message, context)
    policy_phase = runtime.get("last_phase")
    attempts = runtime.get("resolution_attempts", 0) if policy_phase in {None, phase.value} else 0
    if phase == Phase.RESOLVE_INTENT and state.resolution_status == "needs_intent":
        # Asking what the caller needs is not a failed case lookup. Recheck
        # this even for sessions that retain an older retry counter.
        attempts = 0
    runtime = {**runtime, "resolution_attempts": attempts}
    legal, restrictions = set(), []
    mandatory = None

    if phase in machine.TERMINAL_PHASES:
        mandatory = "The SOP already completed or escalated this call; no policy is sampled."
    elif state.is_proxy_caller:
        mandatory = "Representative authorization must be handled by the SOP."
    elif state.out_of_scope_count:
        mandatory = "The SOP scope guard controls this response; no policy is sampled."
    elif context.get("public_help"):
        mandatory = "Public support guidance does not require access to a claim record."
    elif phase == Phase.VERIFY_ID:
        legal = {AgentAction.ASK_IDENTITY_FIELD, AgentAction.EXPLAIN_VERIFICATION_GATE}
    elif phase == Phase.RESOLVE_INTENT and verified:
        legal = {AgentAction.RESOLVE_INTENT, AgentAction.ASK_CLAIM_CLARIFICATION}
        if attempts >= 2 and state.resolution_status in UNRESOLVED_CASE_STATUSES:
            legal.add(AgentAction.ESCALATE_HUMAN)
        if state.resolution_status == "needs_intent":
            restrictions.append("Helping the caller describe their need does not exhaust case lookup attempts or authorize an automatic handoff.")
    elif phase == Phase.PROCESS_CASE and active:
        legal = {AgentAction.ASK_CLAIM_CLARIFICATION} if context.get("needs_clarification") else {AgentAction.ANSWER_GROUNDED}
        restrictions.append("Email is offered only after the customer explicitly finishes claim questions.")
    elif phase == Phase.POST_PROCESS and active:
        if _pending_question(machine, message, context):
            legal = {AgentAction.ASK_CLAIM_CLARIFICATION} if context.get("needs_clarification") else {AgentAction.ANSWER_GROUNDED}
            restrictions.append("A pending claim question is answered before requesting the email choice again.")
        else:
            legal = {AgentAction.OFFER_EMAIL_SUMMARY}
    else:
        mandatory = "The verified identity or selected claim gate must be restored before policy execution."

    if legal and signals["emotion"] in {"frustration", "anger", "anxiety", "confusion"}:
        legal.add(AgentAction.ACK_EMOTION)
    mask = [action in legal for action in AGENT_ACTIONS]
    obs = _observation(machine, message, runtime, mask, context)
    result = {
        "requested_controller": controller_mode,
        "selected_action": None,
        "probabilities": {action.value: 0.0 for action in AGENT_ACTIONS},
        "allowed_actions": [action.value for action in AGENT_ACTIONS if action in legal],
        "action_mask": mask,
        "source": "sop",
        "checkpoint_sha256": None,
        "checkpoint": None,
        "reason": mandatory or "",
        "forced": bool(mandatory or len(legal) == 1),
        "mask_source": "customer_sop_v1",
        "mask_restrictions": restrictions,
        "observation": obs,
        # Audit metadata only; the checkpoint feature contract is unchanged.
        "resolution_status": state.resolution_status,
        "turn_index": runtime.get("turn_count", 0),
        "execution": {"kind": "sop_response"},
        "fallback_reason": None,
        "emotion_source": "bounded_context" if context.get("emotion") in {"neutral", "frustration", "anger", "anxiety", "confusion"} else "text_detector",
    }
    if mandatory:
        return result

    action = None
    if controller_mode.startswith("ppo"):
        try:
            import torch
            legacy = not CHECKPOINTS[controller_mode].is_file()
            path = (LEGACY_CHECKPOINTS if legacy else CHECKPOINTS)[controller_mode]
            stat = path.stat()
            policy, featurizer, metadata = _load_policy(controller_mode, stat.st_mtime_ns, stat.st_size, legacy)
            with torch.inference_mode():
                features = featurizer.featurize_tensor(obs, turn=runtime.get("turn_count", 0))
                logits, value = policy(features, featurizer.extract_mask_tensor(obs))
                distribution = torch.softmax(logits, dim=-1)
                if (not torch.isfinite(features).all() or not torch.isfinite(logits).all()
                        or not torch.isfinite(distribution).all() or not torch.isfinite(value).all()
                        or abs(float(distribution.sum()) - 1.0) > 1e-5):
                    raise ValueError("Non-finite policy output")
                probabilities = distribution.tolist()
                if any(probability != 0.0 for probability, permitted in zip(probabilities, mask) if not permitted):
                    raise ValueError("Policy probabilities violated the action mask")
            action = AGENT_ACTIONS[max(range(len(probabilities)), key=probabilities.__getitem__)]
            result.update(
                source=controller_mode,
                probabilities={act.value: float(p) for act, p in zip(AGENT_ACTIONS, probabilities)},
                checkpoint_sha256=metadata["sha256"], checkpoint=metadata,
                value_estimate=float(value),
                reason="The trained policy selected the largest probability among actions permitted for this real customer turn.",
            )
            if legacy:
                result["fallback_reason"] = "Customer multi-turn checkpoint is not installed; using the original simulator-trained PPO checkpoint with the stricter customer mask."
        except (OSError, ValueError, RuntimeError, KeyError, ImportError, IndexError, TypeError, EOFError, pickle.UnpicklingError) as exc:
            # Availability fallback is visible. Never label a rule action PPO.
            result["fallback_reason"] = f"{controller_mode} unavailable ({type(exc).__name__}); used the rule controller."
    if action is None:
        action = RuleBasedPolicy().select_action(obs)
        if (phase == Phase.VERIFY_ID and obs["privacy_concern"]
                and runtime.get("last_action") == AgentAction.ACK_EMOTION.value
                and AgentAction.EXPLAIN_VERIFICATION_GATE in legal):
            action = AgentAction.EXPLAIN_VERIFICATION_GATE
        result.update(
            source="rule",
            probabilities={act.value: float(act == action) for act in AGENT_ACTIONS},
            reason=result["fallback_reason"] or "The rule controller selected a permitted response action; probabilities are a one-hot decision, not a neural prediction.",
        )

    result["selected_action"] = action.value
    result["execution"] = execution_for_action(machine, message, history, runtime, context, action)
    return result


def execution_for_action(machine, message, history, runtime, context, action):
    """Shared execution contract for masked training actions and inference."""
    phase = machine.state.phase
    active = machine.get_active_claim() is not None
    signals = _signals(message, context)
    execution = {"kind": action.value.lower(), "phase": phase.value,
                 "can_answer_claim": active, "pending_question": _pending_question(machine, message, context)}
    if action in {AgentAction.ASK_IDENTITY_FIELD, AgentAction.EXPLAIN_VERIFICATION_GATE, AgentAction.ACK_EMOTION} and phase == Phase.VERIFY_ID:
        field, correction, avoided = _identity_field(machine, message, history, runtime)
        execution.update(identity_field=field, identity_label=IDENTITY_LABELS[field],
                         correction_needed=correction, avoided_identity_fields=avoided)
    if action == AgentAction.ACK_EMOTION:
        execution.update(kind="empathy_then_progress", emotion=signals["emotion"],
                         followup_kind=("ask_claim_clarification" if context.get("needs_clarification")
                                        else "answer_grounded" if active and _pending_question(machine, message, context)
                                        else "ask_identity_field" if phase == Phase.VERIFY_ID
                                        else "offer_email_summary" if phase == Phase.POST_PROCESS
                                        else "answer_grounded" if active else "resolve_intent"))
    if action == AgentAction.ASK_CLAIM_CLARIFICATION:
        execution["clarification_question"] = context.get("clarification_question")
    if action == AgentAction.ESCALATE_HUMAN:
        execution["escalation_reason"] = "case_resolution_exhausted"
    return execution


def record_execution(runtime_state, decision, *, grounded_answered=False):
    """Commit response memory only after the actual selected response is sent."""
    observation = decision["observation"]
    phase = observation["phase"]
    resolution_status = decision.get("resolution_status")
    if (runtime_state.get("last_phase") != phase
            or (phase == Phase.RESOLVE_INTENT.value and resolution_status == "needs_intent")):
        runtime_state["resolution_attempts"] = 0
    action = decision["selected_action"]
    lookup_response = action in {
        AgentAction.RESOLVE_INTENT.value, AgentAction.ASK_CLAIM_CLARIFICATION.value,
    } or (action == AgentAction.ACK_EMOTION.value
          and decision["execution"].get("followup_kind") in {"resolve_intent", "ask_claim_clarification"})
    if (phase == Phase.RESOLVE_INTENT.value and resolution_status in UNRESOLVED_CASE_STATUSES
            and lookup_response):
        runtime_state["resolution_attempts"] = runtime_state.get("resolution_attempts", 0) + 1
    runtime_state["grounded_answered"] = bool(runtime_state.get("grounded_answered") or grounded_answered)
    runtime_state["last_phase"] = phase
    runtime_state["last_action"] = action
    runtime_state["turn_count"] = runtime_state.get("turn_count", 0) + 1
    if "avoided_identity_fields" in decision["execution"]:
        runtime_state["avoided_identity_fields"] = decision["execution"]["avoided_identity_fields"]
