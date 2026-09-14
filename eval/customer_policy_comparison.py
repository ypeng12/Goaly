"""Compare old/new PPO and rules through the actual customer response executor.

The development callers use wording and combinations absent from PPO rollouts,
but their outputs were inspected while fixing shared language routing. This is
not a blind test set or an estimate of human satisfaction.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from backend.harness.customer_policy import CHECKPOINTS, LEGACY_CHECKPOINTS, decide
from backend.harness.customer_training import CustomerPolicyTrainingEnv, scenarios
from backend.harness.extractor import UtteranceExtractor
from backend.harness.grounded_data import grounded_data
from backend.harness.types import AGENT_ACTIONS, AgentAction
from backend.rl.featurizer import StateFeaturizer
from backend.rl.models import ActorCriticPolicy


def load_checkpoint(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("feature_version") != StateFeaturizer.VERSION or data.get("environment_version") not in {3, 4}:
        raise ValueError("Incompatible checkpoint")
    model = ActorCriticPolicy()
    model.load_state_dict(data["policy_state_dict"])
    model.eval()
    featurizer = StateFeaturizer(max_turns=data["config"]["max_turns"])
    metadata = {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "environment_version": data["environment_version"],
                "training_seed": data["config"]["seed"],
                "actual_steps": data["history"][-1]["global_step"],
                "checkpoint_family": data.get("checkpoint_family", "legacy_simulator_v3"),
                "mask_version": data.get("mask_version", "legacy_training_mask")}
    return model, featurizer, metadata


def compare(split="test", include_trajectories=True):
    torch.set_num_threads(1)
    policies = {"rule": None}
    policies.update({f"legacy_{key}": path for key, path in LEGACY_CHECKPOINTS.items()})
    policies.update({f"customer_{key}": path for key, path in CHECKPOINTS.items() if path.is_file()})
    report = {
        "schema_version": 1, "environment_version": CustomerPolicyTrainingEnv.ENV_VERSION,
        "mask_version": CustomerPolicyTrainingEnv.MASK_VERSION,
        "split": split, "scenarios": [asdict(scenario) for scenario in scenarios(split)],
        "evaluation_kind": "development_eval_with_unseen_training_wording",
        "limitations": "Finite deterministic text-reactive callers; wording absent from PPO rollouts, but inspected during shared routing fixes, so not a blind holdout. Identities reused. Caller TOPIC_WORDS only checks coarse protocol progress, not factual correctness; separate API/content assertions are required. Distress evaluation uses the authored angry/anxious scenarios plus visible distress on the current turn; detector-only counts are retained because its why-need heuristic can mislabel confusion as frustration. No language-model training or human evaluation. Coreference is shared by every controller and is not credited to PPO. The saved training reward awarded +1 for honoring human requests, including requests caused by agent frustration; evaluation counts those as unresolved handoffs, not successful business outcomes.",
        "policies": {},
    }
    for name, path in policies.items():
        model, featurizer, metadata = load_checkpoint(path) if path else (None, None, None)
        rows = []
        for index, scenario in enumerate(scenarios(split)):
            env = CustomerPolicyTrainingEnv(scenario)
            obs, _ = env.reset(seed=1000 + index)
            expected_claim, _ = grounded_data.find_claim(
                env.caller.ph.party_id, UtteranceExtractor.extract_cross_phase_hints(env.caller.desired_hint))
            total = 0.0
            while True:
                if model:
                    with torch.inference_mode():
                        logits, _ = model(featurizer.featurize_tensor(obs, turn=env.turn_count),
                                          featurizer.extract_mask_tensor(obs))
                        probs = torch.softmax(logits, dim=-1).tolist()
                    action = AGENT_ACTIONS[max(range(len(probs)), key=probs.__getitem__)]
                else:
                    d = decide(env.machine, env.message, env.history, "rule", env.runtime, env.context)
                    action = AgentAction(d["selected_action"])
                    probs = [d["probabilities"][a.value] for a in AGENT_ACTIONS]
                obs, reward, terminal, truncated, info = env.step(action)
                env.trajectory[-1]["selection"] = {"source": name, "probabilities": dict(zip((a.value for a in AGENT_ACTIONS), probs))}
                total += reward
                if terminal or truncated:
                    break
            rows.append({
                "scenario": scenario.name, "task_success": info["task_success"],
                "appropriate_handoff": bool(info["final_phase"] == "ESCALATED" and expected_claim is None),
                "handoff_request_honored": bool(info["final_phase"] == "ESCALATED" and "human" in env.message.lower()),
                "unresolved_handoff": bool(info["final_phase"] == "ESCALATED" and expected_claim is not None),
                "premature_exit": info["premature_termination"],
                "terminal_consistent": terminal == (info["final_phase"] in {"CONCLUDED", "ESCALATED"}) and not (terminal and truncated),
                "truncated": truncated, "turns": env.turn_count, "return": round(total, 4),
                "policy_steps": len(env.trajectory),
                "policy_choice_steps": sum(sum(t["observation_before"]["action_mask"]) > 1 for t in env.trajectory),
                "dialogue_response_count": sum(item["role"] == "assistant" for item in env.history),
                "questions_answered": info["questions_answered"],
                "distressed_turns": sum(t["info"]["distressed_turn"] and scenario.style in {"angry", "anxious", "frustrated"} for t in env.trajectory),
                "ignored_distressed_turns": sum(t["info"]["distressed_turn"] and not t["info"]["empathy_delivered"]
                                                and scenario.style in {"angry", "anxious", "frustrated"} for t in env.trajectory),
                "detector_distressed_turns": sum(t["info"]["distressed_turn"] for t in env.trajectory),
                "detector_ignored_distressed_turns": sum(t["info"]["distressed_turn"] and not t["info"]["empathy_delivered"] for t in env.trajectory),
                "unnecessary_explanations": sum(t["info"]["unnecessary_explanation"] for t in env.trajectory),
                "forced_scope_responses": sum(len(t["forced_responses"]) for t in env.trajectory),
                "violations": sum(len(t["info"]["structural_violations"]) for t in env.trajectory),
                **({"trajectory": env.trajectory} if include_trajectories else {}),
            })
        count = len(rows)
        distressed = sum(row["distressed_turns"] for row in rows)
        metrics = {
            "episodes": count,
            "case_completion_rate": sum(row["task_success"] for row in rows) / count,
            "appropriate_handoff_rate": sum(row["appropriate_handoff"] for row in rows) / count,
            "handoff_request_honored_rate": sum(row["handoff_request_honored"] for row in rows) / count,
            "unresolved_handoff_rate": sum(row["unresolved_handoff"] for row in rows) / count,
            "failed_task_rate": sum(not (row["task_success"] or row["appropriate_handoff"]) for row in rows) / count,
            "premature_exit_rate": sum(row["premature_exit"] for row in rows) / count,
            "terminal_consistency_rate": sum(row["terminal_consistent"] for row in rows) / count,
            "truncation_rate": sum(row["truncated"] for row in rows) / count,
            "mean_turns": sum(row["turns"] for row in rows) / count,
            "mean_policy_steps": sum(row["policy_steps"] for row in rows) / count,
            "mean_policy_choice_steps": sum(row["policy_choice_steps"] for row in rows) / count,
            "mean_dialogue_responses": sum(row["dialogue_response_count"] for row in rows) / count,
            "mean_return": sum(row["return"] for row in rows) / count,
            "mean_questions_answered": sum(row["questions_answered"] for row in rows) / count,
            "ignored_distress_rate": sum(row["ignored_distressed_turns"] for row in rows) / distressed if distressed else 0,
            "unnecessary_gate_explanations": sum(row["unnecessary_explanations"] for row in rows),
            "forced_scope_responses": sum(row["forced_scope_responses"] for row in rows),
            "safety_violations": sum(row["violations"] for row in rows),
        }
        report["policies"][name] = {"checkpoint": metadata, "metrics": metrics, "episodes": rows}
    checks = {}
    baseline = report["policies"]["rule"]["metrics"]
    for seed in (42, 7):
        key = f"customer_ppo{seed}"
        row = report["policies"].get(key)
        checks[f"{key}_loaded"] = row is not None
        if not row:
            continue
        metrics, checkpoint = row["metrics"], row["checkpoint"]
        checks.update({
            f"{key}_compatible": checkpoint["checkpoint_family"] == "customer_multi_turn_v1" and checkpoint["environment_version"] == 4 and checkpoint["training_seed"] == seed and checkpoint["mask_version"] == "customer_sop_v1",
            f"{key}_safety": metrics["safety_violations"] == 0,
            f"{key}_terminal_consistency": metrics["terminal_consistency_rate"] == 1.0,
            f"{key}_no_timeouts": metrics["truncation_rate"] == 0.0,
            f"{key}_completion_matches_rule": metrics["case_completion_rate"] >= baseline["case_completion_rate"],
            f"{key}_no_unresolved_handoffs": metrics["unresolved_handoff_rate"] == 0.0,
            f"{key}_distress_matches_rule": metrics["ignored_distress_rate"] <= baseline["ignored_distress_rate"],
        })
    report["checks"] = checks
    report["passed"] = all(checks.values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", default="artifacts/customer_policy/comparison.json")
    parser.add_argument("--assert-thresholds", action="store_true")
    args = parser.parse_args()
    report = compare(args.split)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: item["metrics"] for key, item in report["policies"].items()}, indent=2))
    print(f"Full trajectories: {path}")
    print("Acceptance:", report["passed"])
    if args.assert_thresholds and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
