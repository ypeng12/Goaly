"""Reproducible fixture benchmark with explicit assertions; no live model calls."""
import argparse
import datetime
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from backend.engine.mock_engine import MockEngine
from backend.harness.state_machine import SOPStateMachine

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
REPORT_PATH = EVAL_DIR / "EVAL_REPORT.md"
ALLOWED_PII = {"name", "dob", "phone", "email", "id_last4"}
TERMINAL = {"CONCLUDED", "ESCALATED"}
ALLOWED_EDGES = {
    "VERIFY_ID": {"VERIFY_ID", "RESOLVE_INTENT", "ESCALATED"},
    "RESOLVE_INTENT": {"RESOLVE_INTENT", "PROCESS_CASE", "ESCALATED"},
    "PROCESS_CASE": {"PROCESS_CASE", "RESOLVE_INTENT", "POST_PROCESS", "ESCALATED"},
    "POST_PROCESS": {"POST_PROCESS", "PROCESS_CASE", "CONCLUDED", "ESCALATED"},
    "CONCLUDED": {"CONCLUDED"}, "ESCALATED": {"ESCALATED"},
}


def _value_at(state: dict, path: str) -> Any:
    value = state
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Unknown assertion state path: {path}")
        value = value[part]
    return value


class EvalBenchmarkRunner:
    def __init__(self, dataset_path: Path = DATASET_PATH, report_path: Path = REPORT_PATH):
        self.dataset_path = Path(dataset_path)
        self.report_path = Path(report_path)
        self.dataset = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        if not self.dataset:
            raise ValueError("Benchmark dataset must not be empty")

    def run_benchmark(self) -> dict:
        cases = []
        counters = defaultdict(lambda: {"passed": 0, "total": 0})
        latency = []
        for case in self.dataset:
            sm = SOPStateMachine(session_id=f"eval_{case['id']}")
            engine, history, turns, assertions = MockEngine(), [], [], []

            def check(group: str, label: str, passed: bool, actual=None, expected=None):
                item = {"group": group, "check": label, "passed": bool(passed)}
                if not passed:
                    item.update(actual=actual, expected=expected)
                assertions.append(item)
                counters[group]["total"] += 1
                counters[group]["passed"] += int(bool(passed))

            def expect(spec: dict, reply: str, result: dict, label: str):
                for path, expected in spec.get("state", {}).items():
                    actual = _value_at(sm.state.model_dump(mode="json"), path)
                    group = "memory" if path.startswith("cross_phase_memory.") else "behavior"
                    check(group, f"{label}: {path}", actual == expected, actual, expected)
                for key, expected in spec.get("result", {}).items():
                    check("behavior", f"{label}: result.{key}", result.get(key) == expected, result.get(key), expected)
                for token in spec.get("contains", []):
                    check("response", f"{label}: response contains {token!r}", token.casefold() in reply.casefold())
                for alternatives in spec.get("contains_any", []):
                    check("response", f"{label}: response includes one of {alternatives!r}", any(t.casefold() in reply.casefold() for t in alternatives))
                for token in spec.get("forbidden", []):
                    check("sentinel_non_disclosure", f"{label}: response excludes {token!r}", token.casefold() not in reply.casefold())

            for index, utterance in enumerate(case["turns"], start=1):
                before = sm.state.model_dump(mode="json")
                trace_offset = len(sm.state.trace_log)
                start = time.perf_counter()
                result = sm.evaluate_turn(utterance)
                reply = engine.generate_response(utterance, result, history)
                elapsed = (time.perf_counter() - start) * 1000
                latency.append(elapsed)
                history.extend([{"role": "user", "content": utterance}, {"role": "assistant", "content": reply}])
                state, label = sm.state, f"turn {index}"
                verified = bool(state.verified_party_id)
                allowed = set(state.verified_fields) & ALLOWED_PII
                check("sop_invariants", f"{label}: only permitted verification fields", set(state.verified_fields) <= ALLOWED_PII)
                check("sop_invariants", f"{label}: verified identity requires three distinct PII", not verified or len(allowed) >= 3)
                check("sop_invariants", f"{label}: national ID is not counted as SSN", "id_last4" not in allowed or state.accumulated_pii.id_type == "ssn_last4")
                check("sop_invariants", f"{label}: active phases require identity", state.phase.value in {"VERIFY_ID", "ESCALATED"} or verified)
                check("sop_invariants", f"{label}: unverified structured data stays shielded", verified or (state.active_case_id is None and result.get("active_claim") is None and result.get("policyholder") is None))
                check("sop_invariants", f"{label}: context shield reflects verification", result.get("context", {}).get("data_shield_active") == (not verified))
                active_claim = sm.get_active_claim()
                check("sop_invariants", f"{label}: selected claim belongs to caller", active_claim is None or active_claim.party_id == state.verified_party_id)
                sent = state.post_process.sent_to
                check("consent", f"{label}: send requires offered and accepted summary", not sent or (verified and state.post_process.email_offered and state.post_process.user_decision == "accepted"))
                if before["phase"] in TERMINAL:
                    check("sop_invariants", f"{label}: terminal phase remains stable", state.phase.value == before["phase"])
                    check("consent", f"{label}: terminal consent remains stable", state.post_process.model_dump(mode="json") == before["post_process"])
                for event in state.trace_log[trace_offset:]:
                    check("sop_invariants", f"{label}: legal trace edge {event.phase_before.value} -> {event.phase_after.value}", event.phase_after.value in ALLOWED_EDGES[event.phase_before.value])
                expect(case.get("turn_expectations", {}).get(str(index), {}), reply, result, label)
                turns.append({"turn": index, "phase": state.phase.value, "reply": reply, "latency_ms": round(elapsed, 3)})
            expect(case.get("expect", {}), turns[-1]["reply"], result, "final")
            failures = [a for a in assertions if not a["passed"]]
            cases.append({"id": case["id"], "name": case["name"], "category": case["category"], "passed": not failures, "assertions": assertions, "failures": failures, "turns": turns})

        passed = sum(c["passed"] for c in cases)
        metrics = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "engine": "MockEngine (no live provider calls)", "dataset": self.dataset_path.name,
            "total_scenarios": len(cases), "passed_scenarios": passed,
            "total_turns": len(latency), "overall_pass_rate": round(100 * passed / len(cases), 2),
            "avg_latency_ms": round(sum(latency) / len(latency), 3),
            "assertion_groups": dict(counters), "case_results": cases,
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        self._generate_markdown_report(metrics)
        return metrics

    def _generate_markdown_report(self, metrics: dict):
        lines = [
            "# Insurance SOP fixture evaluation", "",
            f"Generated: `{metrics['timestamp']}`. Engine: {metrics['engine']}.", "",
            f"Scenarios: **{metrics['passed_scenarios']}/{metrics['total_scenarios']} passed** across {metrics['total_turns']} turns.",
            f"Mean measured turn execution: {metrics['avg_latency_ms']} ms (local state machine plus mock response; excludes HTTP and model latency).", "",
            "| Assertion group | Passed / checked | Result |", "| --- | ---: | --- |",
        ]
        for name, counts in metrics["assertion_groups"].items():
            lines.append(f"| {name} | {counts['passed']} / {counts['total']} | {'PASS' if counts['passed'] == counts['total'] else 'FAIL'} |")
        lines.extend(["", "| Scenario | Turns | Final phase | Result |", "| --- | ---: | --- | --- |"])
        for case in metrics["case_results"]:
            lines.append(f"| {case['id']}: {case['name']} | {len(case['turns'])} | {case['turns'][-1]['phase']} | {'PASS' if case['passed'] else 'FAIL'} |")
        failures = [(case, failure) for case in metrics["case_results"] for failure in case["failures"]]
        if failures:
            lines.extend(["", "## Failed assertions", ""])
            for case, failure in failures:
                lines.append(f"- **{case['id']}**: {failure['check']}; expected `{failure.get('expected')}`, got `{failure.get('actual')}`.")
        lines.extend([
            "", "## What these numbers measure", "",
            "Each turn checks identity thresholds, permitted PII types, claim ownership, shielding of structured data, legal transition edges, and consent prerequisites. A blocked gate is normal behavior; trace entries are not automatically counted as successful checks.", "",
            "Memory assertions compare the named field with its exact expected value. Response assertions check specified wording or alternatives. Sentinel checks search replies for explicitly listed protected fixture facts on designated turns. These checks do not measure all possible hallucinations, semantic privacy leakage, or whether a caller actually feels reassured.", "",
            "A scenario fails if any assertion fails. The JSON report includes every assertion and turn reply. This deterministic fixture suite is finite, is not an adversarial security proof, and does not evaluate live-provider conversation quality. All identities and claims are synthetic. Historical fixture deadlines are preserved.", "",
            "Reproduce with `python -m eval.eval_benchmark`; a failing scenario produces exit status 1.", "",
        ])
        self.report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    metrics = EvalBenchmarkRunner(args.dataset, args.report).run_benchmark()
    print(f"Mock benchmark: {metrics['passed_scenarios']}/{metrics['total_scenarios']} scenarios passed; {metrics['total_turns']} turns.")
    for name, counts in metrics["assertion_groups"].items():
        print(f"  {name}: {counts['passed']}/{counts['total']} assertions passed")
    print(f"Report: {args.report}")
    raise SystemExit(0 if metrics["passed_scenarios"] == metrics["total_scenarios"] else 1)


if __name__ == "__main__":
    main()
