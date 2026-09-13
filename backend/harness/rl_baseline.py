"""Rule-based policy baseline for AgentPolicyEnv.

RuleBasedPolicy implements the deterministic SOP as a policy that can be
compared against random sampling or a learned dialogue-act policy. It prioritizes
human requests, empathy and gate explanations, then verification, clarification,
a grounded answer and the optional email choice. Unresolved case matching after
two attempts permits a reasoned handoff.

Usage:
    from backend.harness.rl_env import AgentPolicyEnv
    from backend.harness.rl_baseline import RuleBasedPolicy

    env = AgentPolicyEnv(max_turns=20)
    policy = RuleBasedPolicy()

    obs, info = env.reset()
    done = False
    cumulative_reward = 0.0
    while not done:
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        done = terminated or truncated

    print(f"Cumulative reward: {cumulative_reward:.2f}")
    print(f"Violations: {info['structural_violations']}")
"""
from .types import AgentAction, Phase, AGENT_ACTIONS


class RuleBasedPolicy:
    """Deterministic SOP-aligned policy for baseline comparison.

    Given an observation dict from AgentPolicyEnv, selects the action that
    follows the SOP rules exactly.  It never violates action_mask constraints.

    This policy is useful as:
    - A hand-designed performance reference for the learned policy
    - A sanity check that the environment rewards SOP-compliant behaviour
    - A reference for mask-compliance in tests
    """

    def __init__(self):
        self._grounded_answered = False

    def _reset_episode_state(self) -> None:
        self._grounded_answered = False

    def select_action(self, observation: dict) -> AgentAction:
        """Select the best legal action given the current observation.

        Args:
            observation: The dict returned by AgentPolicyEnv.step() or .reset().
                         Keys: phase, verified_fields, memory_slots,
                               data_shield_active, action_mask,
                               caller_utterance (new), last_agent_reply (new).

        Returns:
            An AgentAction that is legal according to observation['action_mask'].
        """
        phase_str = observation.get("phase", Phase.VERIFY_ID.value)
        verified_fields: list[str] = observation.get("verified_fields", [])
        mask: list[bool] = observation.get("action_mask", [True] * len(AGENT_ACTIONS))
        data_shield: bool = observation.get("data_shield_active", True)
        memory_slots: dict = observation.get("memory_slots", {})
        phase = Phase(phase_str)

        def legal(action: AgentAction) -> bool:
            idx = AGENT_ACTIONS.index(action)
            return mask[idx]

        if observation.get('demands_human') and legal(AgentAction.ESCALATE_HUMAN):
            return AgentAction.ESCALATE_HUMAN
        if observation.get('emotion') in {'frustration', 'anger', 'anxiety'} and legal(AgentAction.ACK_EMOTION):
            return AgentAction.ACK_EMOTION
        if (observation.get('privacy_concern') or observation.get('emotion') == 'confusion') and legal(AgentAction.EXPLAIN_VERIFICATION_GATE):
            return AgentAction.EXPLAIN_VERIFICATION_GATE

        # --- Phase-specific rule cascade ---

        if phase == Phase.VERIFY_ID:
            # Not yet verified: ask for missing fields, or explain the gate
            pii_fields = {"name", "dob", "phone", "email", "id_last4"}
            missing = pii_fields - set(verified_fields)
            if missing and legal(AgentAction.ASK_IDENTITY_FIELD):
                return AgentAction.ASK_IDENTITY_FIELD
            if legal(AgentAction.EXPLAIN_VERIFICATION_GATE):
                return AgentAction.EXPLAIN_VERIFICATION_GATE
            if legal(AgentAction.ACK_EMOTION):
                return AgentAction.ACK_EMOTION

        elif phase == Phase.RESOLVE_INTENT:
            # After 2 attempts without phase change, escalate (no matching claim)
            if observation.get('resolution_attempts', 0) >= 2 and legal(AgentAction.ESCALATE_HUMAN):
                return AgentAction.ESCALATE_HUMAN
            if not memory_slots.get('case_type_hint') and legal(AgentAction.RESOLVE_INTENT):
                return AgentAction.RESOLVE_INTENT
            # The remembered intent exists; ask for case-disambiguating details.
            if legal(AgentAction.ASK_CLAIM_CLARIFICATION):
                return AgentAction.ASK_CLAIM_CLARIFICATION
            # Fall through to escalation if stuck
            if legal(AgentAction.ESCALATE_HUMAN):
                return AgentAction.ESCALATE_HUMAN
            if legal(AgentAction.ACK_EMOTION):
                return AgentAction.ACK_EMOTION

        elif phase == Phase.PROCESS_CASE:
            # Provide grounded answer once if verified, then move to wrap up
            if not data_shield and legal(AgentAction.ANSWER_GROUNDED) and not self._grounded_answered:
                self._grounded_answered = True
                return AgentAction.ANSWER_GROUNDED
            if legal(AgentAction.OFFER_EMAIL_SUMMARY):
                return AgentAction.OFFER_EMAIL_SUMMARY
            if legal(AgentAction.ACK_EMOTION):
                return AgentAction.ACK_EMOTION

        elif phase == Phase.POST_PROCESS:
            if legal(AgentAction.SEND_EMAIL):
                return AgentAction.SEND_EMAIL
            if legal(AgentAction.OFFER_EMAIL_SUMMARY):
                return AgentAction.OFFER_EMAIL_SUMMARY
            if legal(AgentAction.ACK_EMOTION):
                return AgentAction.ACK_EMOTION

        # Fallback: pick first legal action
        for action in AGENT_ACTIONS:
            if legal(action):
                return action

        raise RuntimeError(
            f"No legal action available in phase={phase_str} with mask={mask}. "
            "This should not happen unless the episode is already done."
        )

    def run_episode(self, env, verbose: bool = False, seed=None) -> dict:
        """Run a full episode and return a summary dict.

        Args:
            env: An AgentPolicyEnv instance (already initialized with caller_profile).
            verbose: If True, print each turn's action, caller utterance and reward.

        Returns:
            {
              "cumulative_reward": float,
              "turns": int,
              "terminated": bool,
              "truncated": bool,
              "violations": list[str],
              "trajectory": list,
              "final_phase": str,
            }
        """
        obs, info = env.reset(seed=seed)
        self._reset_episode_state()
        done = False
        cumulative_reward = 0.0
        all_violations: list[str] = []
        terminated = truncated = False

        last_info: dict = {}
        while not done:
            action = self.select_action(obs)
            obs, reward, terminated, truncated, last_info = env.step(action)
            cumulative_reward += reward
            all_violations.extend(last_info.get("structural_violations", []))
            done = terminated or truncated
            if verbose:
                caller = obs.get("caller_utterance", "")[:60]
                print(
                    f"  turn={last_info['turn']:2d}  action={action.value:<28s}"
                    f"  reward={reward:+.2f}  phase={obs['phase']}"
                )
                if caller:
                    print(f"    caller: {caller!r}")

        return {
            "cumulative_reward": cumulative_reward,
            "turns": env.turn_count,
            "terminated": terminated,
            "truncated": truncated,
            "task_success": last_info.get("task_success", False),
            "appropriate_escalation": last_info.get("appropriate_escalation", False),
            "premature_termination": last_info.get("premature_termination", False),
            "violations": all_violations,
            "trajectory": env.trajectory,
            "final_phase": obs.get("phase", "unknown"),
            "verified_fields": list(obs.get("verified_fields", [])),
        }
