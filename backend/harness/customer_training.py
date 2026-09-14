"""Multi-turn text-reactive training with the production customer executor.

The finite synthetic caller receives only the agent's actual reply. It cannot
read policy actions, masks, model scores, SOP state, or future observations.
This is deliberately a test protocol, not a claim of human-equivalent realism.
"""
from dataclasses import dataclass
import copy
import random
import re

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .conversation_context import prepare_conversation_turn
from .customer_policy import decide, execution_for_action, record_execution
from .customer_service import render_policy_reply
from .dialogue import dialogue_signals
from .grounded_data import grounded_data
from .state_machine import SOPStateMachine
from .types import AGENT_ACTIONS, AgentAction, Phase
from ..engine.mock_engine import EMPATHY


def has_empathy(reply):
    """Measure what was actually said, not the selected action label."""
    return any(text.strip() in reply for text in EMPATHY.values())


def is_gate_explanation(reply):
    low = reply.lower()
    return "privacy" in low and any(term in low for term in ("3 identity", "3 matching", "three matching"))


@dataclass(frozen=True)
class CustomerScenario:
    name: str
    policyholder: str
    hint: str
    style: str = "cooperative"
    wording: int = 0
    email_accepted: bool = False
    opening_verified: bool = False
    opening_name: bool = True


def scenarios(split="train"):
    """Held-out wording and identity/style combinations; identities are reused."""
    if split == "train":
        return [
            CustomerScenario("margaret-stepwise", "Margaret Chen", "denied healthcare claim from January"),
            CustomerScenario("ma-stepwise", "Ma Tian", "denied healthcare claim"),
            CustomerScenario("margaret-angry", "Margaret Chen", "denied healthcare claim from January", "angry", email_accepted=True),
            CustomerScenario("ma-angry", "Ma Tian", "denied healthcare claim", "angry", opening_verified=True),
            CustomerScenario("margaret-anxious", "Margaret Chen", "denied healthcare claim from January", "anxious", opening_verified=True),
            CustomerScenario("ma-private", "Ma Tian", "denied healthcare claim", "privacy_sensitive"),
            CustomerScenario("margaret-confused", "Margaret Chen", "denied healthcare claim from January", "confused"),
            CustomerScenario("ava-no-record", "Ava Lopez", "denied dental claim"),
            CustomerScenario("margaret-frustrated-stepwise", "Margaret Chen", "denied healthcare claim from January", "frustrated"),
            CustomerScenario("ma-frustrated-verified", "Ma Tian", "denied healthcare claim", "frustrated", opening_verified=True),
            CustomerScenario("margaret-frustrated-no-hint", "Margaret Chen", "", "frustrated"),
            CustomerScenario("margaret-frustrated-no-name", "Margaret Chen", "denied healthcare claim from January", "frustrated", opening_name=False),
            CustomerScenario("ma-anxious-stepwise", "Ma Tian", "denied healthcare claim", "anxious"),
            CustomerScenario("margaret-anxious-stepwise", "Margaret Chen", "denied healthcare claim from January", "anxious"),
        ]
    if split in {"val", "test"}:
        wording = 1 if split == "val" else 2
        return [
            CustomerScenario(f"margaret-stepwise-{split}", "Margaret Chen", "denied healthcare claim from January", wording=wording, email_accepted=True),
            CustomerScenario(f"ma-stepwise-{split}", "Ma Tian", "denied healthcare claim", wording=wording),
            CustomerScenario(f"margaret-angry-{split}", "Margaret Chen", "denied healthcare claim from January", "angry", wording, opening_verified=True),
            CustomerScenario(f"ma-anxious-{split}", "Ma Tian", "denied healthcare claim", "anxious", wording),
            CustomerScenario(f"margaret-private-{split}", "Margaret Chen", "denied healthcare claim from January", "privacy_sensitive", wording),
            CustomerScenario(f"ma-confused-{split}", "Ma Tian", "denied healthcare claim", "confused", wording),
            CustomerScenario(f"ava-no-record-{split}", "Ava Lopez", "denied dental claim", wording=wording),
        ]
    raise ValueError("Unknown scenario split")


class ReactiveCustomer:
    """A customer test double driven by response text and its own needs."""

    QUESTIONS = (
        ("What documents do I need?", "Can you explain the first one?", "I cannot get it.", "Would a scan work?", "How long will review take?"),
        ("Which documents are required?", "What is the first one?", "What if I cannot get it?", "Would a photo work?", "What is the processing time?"),
        ("Tell me which materials I need to send.", "Explain the first document.", "I cannot obtain that.", "Are copies acceptable?", "How many business days does review take?"),
    )
    TOPIC_WORDS = (("document", "report", "note", "materials"), ("report", "note", "document"),
                   ("provider", "alternative", "representative", "original", "report"),
                   ("scan", "copy", "copie", "readable", "legible", "image", "photo"),
                   ("review", "business", "working", "timeline", "turnaround"))

    def __init__(self, scenario, seed=None):
        self.scenario = scenario
        self.ph = next(person for person in grounded_data.policyholders if person.name == scenario.policyholder)
        self.desired_hint = scenario.hint or "denied healthcare claim from January"
        self.rng = random.Random(seed)
        self.given = {"name"} if scenario.opening_name else set()
        self.question_index = -1
        self.awaiting_answer = False
        self.gate_explained = False
        self.distress_ignored = 0
        self.answers_received = 0
        self.no_match_replies = 0
        self.email_decided = False
        self.reassured = False

    def _emotion(self, message):
        if self.scenario.style == "frustrated":
            return "I am frustrated. " + message
        if self.scenario.style == "angry":
            prefix = ("I am angry. ", "This is frustrating. ", "This is ridiculous. ")[self.scenario.wording]
            return prefix + message
        if self.scenario.style == "anxious":
            return ("I am worried. ", "I feel anxious. ", "I am scared. ")[self.scenario.wording] + message
        return message

    def opening(self):
        message = f"My name is {self.ph.name}." if self.scenario.opening_name else "Hello."
        if self.scenario.hint:
            message += f" I need help with my {self.scenario.hint}."
        if self.scenario.opening_verified:
            self.given.update({"dob", "phone"})
            message += f" My DOB is {self.ph.dob}. My phone is {self.ph.phone}."
        if self.scenario.style == "privacy_sensitive":
            message += " I prefer not to share sensitive government IDs."
        elif self.scenario.style == "confused":
            message += " I am confused. Why do you need my details?"
        return self._emotion(message)

    def _give_field(self, field):
        self.given.add(field)
        label = {"name": "name", "dob": "date of birth", "phone": "phone number", "email": "email"}[field]
        return f"My {label} is {getattr(self.ph, field)}."

    def respond(self, reply):
        # Sole input is speech. No policy action or SOP state may be passed.
        low = reply.lower()
        if is_gate_explanation(reply):
            self.gate_explained = True
        distressed = self.scenario.style in {"angry", "anxious", "frustrated"} and ((len(self.given) < 3 and not self.reassured) or self.awaiting_answer)
        if distressed and not has_empathy(reply):
            self.distress_ignored += 1
            if self.distress_ignored >= 3:
                return "I want a human representative now."
            pending = (self.QUESTIONS[self.scenario.wording][self.question_index] if self.awaiting_answer
                       else "Please understand my concern before asking for identity details.")
            return self._emotion(pending)
        if has_empathy(reply):
            self.distress_ignored = 0
            self.reassured = True
        request = re.search(r"please (?:share|correct) your (full name|date of birth|phone number|email address)", low)
        if request:
            if self.scenario.style in {"privacy_sensitive", "confused"} and not self.gate_explained:
                return "I am confused. I prefer not to share sensitive government IDs. Why do you need this? Can I use phone or email?"
            field = {"full name": "name", "date of birth": "dob", "phone number": "phone", "email address": "email"}[request.group(1)]
            if field in self.given:
                return "I already provided that identity detail. What else do you need?"
            return self._give_field(field)
        if "could not match" in low or "could not find a matching" in low:
            self.no_match_replies += 1
            if self.no_match_replies >= 3:
                return "Please connect me with a human representative."
            return f"My letter says {self.scenario.hint}. I have no claim reference or other year to add."
        if "which claim reference or year" in low or "more than one claim matches" in low:
            return f"I mean my {self.desired_hint}."
        if "email summary" in low and any(phrase in low for phrase in ("would you like", "send or skip", "send the", "send an")):
            self.email_decided = True
            return "Yes, please send the email summary." if self.scenario.email_accepted else "No, skip the email summary."
        if self.awaiting_answer:
            if "which item do you mean" in low:
                return "I mean the first requested document."
            if any(word in low for word in self.TOPIC_WORDS[self.question_index]):
                self.answers_received += 1
                self.awaiting_answer = False
            else:
                return self._emotion(self.QUESTIONS[self.scenario.wording][self.question_index])
        if self.question_index == -1 and not re.search(r"claim cl-\d+ is |claim.*(?:denied|review|open)|recorded.*denial", low):
            if "what would you like" in low or "what.*claim" in low:
                return f"I need the status and documents for my {self.desired_hint}."
            return "Could you explain the status of my claim?"
        self.question_index += 1
        if self.question_index >= len(self.QUESTIONS[self.scenario.wording]):
            return "That answers my questions."
        self.awaiting_answer = True
        return self._emotion(self.QUESTIONS[self.scenario.wording][self.question_index])


class CustomerPolicyTrainingEnv(gym.Env):
    """The real-customer mask/executor paired with the text-reactive test caller."""

    ENV_VERSION = 4
    MASK_VERSION = "customer_sop_v1"
    DATASET_REVISION = 2
    metadata = {"render_modes": []}

    def __init__(self, scenario=None, max_turns=20):
        super().__init__()
        self.scenario = scenario or scenarios()[0]
        self.max_turns = max_turns
        self.action_space = spaces.Discrete(len(AGENT_ACTIONS))
        # Native observation remains a mapping for the original 30-D featurizer.
        self.observation_space = spaces.Dict({
            "phase": spaces.Text(max_length=32, min_length=1, charset="ABCDEFGHIJKLMNOPQRSTUVWXYZ_"),
            "verified_fields": spaces.Sequence(spaces.Text(max_length=16, charset="abcdefghijklmnopqrstuvwxyz_")),
            "memory_slots": spaces.Dict({key: spaces.Text(max_length=8, min_length=0, charset="provided") for key in ("case_type_hint", "status_hint", "date_hint", "topic_hint")}),
            "data_shield_active": spaces.Discrete(2),
            "active_case_id": spaces.Text(max_length=8, min_length=0, charset="selected"),
            "emotion": spaces.Text(max_length=16, charset="abcdefghijklmnopqrstuvwxyz"),
            "privacy_concern": spaces.Discrete(2), "refusal": spaces.Discrete(2), "demands_human": spaces.Discrete(2),
            "resolution_attempts": spaces.Discrete(max_turns + 1), "grounded_answered": spaces.Discrete(2),
            "action_mask": spaces.MultiBinary(len(AGENT_ACTIONS)),
        })

    def _assimilate(self, message):
        self.message = message
        self.context = prepare_conversation_turn(self.machine, message, self.history)
        self.result = self.machine.evaluate_turn(message, conversation_context=self.context)
        self.history.append({"role": "user", "content": message})
        self.decision = decide(self.machine, message, self.history, "rule", self.runtime, self.context)
        obs = copy.deepcopy(self.decision["observation"])
        obs["verified_fields"] = tuple(obs["verified_fields"])
        obs["action_mask"] = np.array(obs["action_mask"], dtype=np.int8)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.caller = ReactiveCustomer(self.scenario, seed)
        self.machine = SOPStateMachine("customer-training")
        self.history, self.runtime, self.trajectory = [], {}, []
        self.turn_count = 0
        self.done = False
        self.gate_explanations = 0
        obs = self._assimilate(self.caller.opening())
        return obs, {"scenario": self.scenario.name, "seed": seed}

    def _violations(self):
        state = self.machine.state
        verified = self.machine.get_verified_policyholder() is not None
        errors = []
        if state.identity_verified and not verified:
            errors.append("invalid_identity_gate")
        if not verified and state.active_case_id:
            errors.append("unverified_claim_exposure")
        if state.phase not in {Phase.VERIFY_ID, Phase.ESCALATED} and not verified:
            errors.append("premature_phase_advance")
        if state.post_process.sent_to and state.post_process.user_decision != "accepted":
            errors.append("missing_email_consent")
        return errors

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has ended")
        if isinstance(action, (int, np.integer)):
            action = AGENT_ACTIONS[int(action)]
        if not isinstance(action, AgentAction) or action.value not in self.decision["allowed_actions"]:
            raise ValueError("Action is masked")
        before = copy.deepcopy(self.machine.state)
        decision = copy.deepcopy(self.decision)
        decision.update(selected_action=action.value, source="training_rollout",
                        execution=execution_for_action(self.machine, self.message, self.history,
                                                       self.runtime, self.context, action))
        original_message = self.message
        reply = render_policy_reply(self.machine, self.message, self.result, self.context, decision, history=self.history)
        grounded = self.result.get("grounded_answer_delivered", False)
        self.machine.record_grounded_topics(self.result.get("grounded_topics", []))
        self.machine.record_snapshot(self.message, reply)
        self.history.append({"role": "assistant", "content": reply})
        record_execution(self.runtime, decision, grounded_answered=grounded)
        self.turn_count += 1
        emotion = dialogue_signals(original_message)["emotion"]
        distressed = emotion in {"frustration", "anger", "anxiety"}
        explanation = is_gate_explanation(reply)
        unnecessary_explanation = explanation and (
            self.gate_explanations > 0 or (emotion == "neutral" and not decision["observation"]["privacy_concern"]))
        self.gate_explanations += int(explanation)
        caller_message = ""
        agent_already_terminated = self.machine.state.phase in self.machine.TERMINAL_PHASES
        if not agent_already_terminated:
            caller_message = self.caller.respond(reply)
            obs = self._assimilate(caller_message)
        else:
            self.decision = decide(self.machine, original_message, self.history, "rule", self.runtime, {})
            obs = self.decision["observation"]
        forced_responses = []
        # A scope refusal is a mandatory controller response, not a sampled
        # action. Consume these bounded external transitions honestly. Repeated
        # unrelated input is then escalated by the same SOP as real chat.
        while (self.machine.state.phase not in self.machine.TERMINAL_PHASES
               and not any(self.decision["action_mask"])
               and self.turn_count < self.max_turns):
            mandatory_reply = render_policy_reply(self.machine, self.message, self.result, self.context,
                                                  self.decision, history=self.history)
            self.machine.record_snapshot(self.message, mandatory_reply)
            self.history.append({"role": "assistant", "content": mandatory_reply})
            record_execution(self.runtime, self.decision)
            self.turn_count += 1
            forced_responses.append({"message": self.message, "reply": mandatory_reply, "source": "sop"})
            caller_message = self.caller.respond(mandatory_reply)
            obs = self._assimilate(caller_message)
        terminal = self.machine.state.phase in self.machine.TERMINAL_PHASES
        truncated = self.turn_count >= self.max_turns and not terminal
        self.done = terminal or truncated
        terminal_reply = ""
        if terminal and not agent_already_terminated:
            terminal_decision = decide(self.machine, self.message, self.history, "rule", self.runtime, self.context)
            terminal_reply = render_policy_reply(self.machine, self.message, self.result, self.context,
                                                 terminal_decision, history=self.history)
            self.machine.record_snapshot(self.message, terminal_reply)
            self.history.append({"role": "assistant", "content": terminal_reply})
        violations = self._violations()
        success = bool(self.machine.state.phase == Phase.CONCLUDED and self.runtime.get("grounded_answered")
                       and self.caller.answers_received == len(self.caller.QUESTIONS[self.scenario.wording])
                       and self.machine.state.post_process.user_decision in {"accepted", "declined"} and not violations)
        exhausted = (self.machine.state.phase == Phase.ESCALATED
                     and (self.caller.no_match_replies >= 2 or "human" in caller_message.lower()))
        premature = terminal and not (success or exhausted)
        components = {
            "turn_cost": -0.1 * (1 + len(forced_responses)),
            "new_identity_fields": 0.25 * len(set(self.machine.state.verified_fields) - set(before.verified_fields)),
            "ignored_distress": -1.0 if distressed and not has_empathy(reply) else 0.0,
            "unnecessary_gate_explanation": -0.4 if unnecessary_explanation else 0.0,
            "completion": 10.0 if success else 0.0,
            "appropriate_handoff": 1.0 if exhausted else 0.0,
            "premature_exit": -5.0 if premature else 0.0,
            "timeout": -5.0 if truncated else 0.0,
            "violations": -100.0 * len(violations),
        }
        info = {
            "task_success": success, "appropriate_escalation": exhausted,
            "premature_termination": premature, "structural_violations": violations,
            "reward_components": components, "questions_answered": self.caller.answers_received,
            "distressed_turn": distressed, "empathy_delivered": has_empathy(reply),
            "unnecessary_explanation": unnecessary_explanation,
            "phase_before": before.phase.value, "final_phase": self.machine.state.phase.value,
        }
        self.trajectory.append({"turn": self.turn_count, "message": original_message,
                                "reply": reply, "action": action.value, "next_caller_message": caller_message,
                                "observation_before": decision["observation"], "info": info,
                                "forced_responses": forced_responses, "terminal_reply": terminal_reply})
        obs = dict(obs)
        obs["verified_fields"] = tuple(obs["verified_fields"])
        obs["action_mask"] = np.array(obs["action_mask"], dtype=np.int8)
        return obs, sum(components.values()), terminal, truncated, info
