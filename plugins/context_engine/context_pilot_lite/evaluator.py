"""Offline A/B evaluator for ContextPilot-lite (no model/network calls)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable

@dataclass
class ABResult:
    baseline_input_size: int
    candidate_input_size: int
    required_span_direct_retention: bool
    verified_recoverability: bool
    role_tool_structural_validity: bool
    task_success_baseline: bool
    task_success_candidate: bool
    passed: bool
    failures: list[str]


def _size(messages):
    return sum(len(str(m.get("content", ""))) for m in messages)


def _valid(messages):
    pending = set()
    for m in messages:
        role = m.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            return False
        if role == "assistant":
            for call in m.get("tool_calls", []) or []:
                if isinstance(call, dict) and call.get("id"):
                    pending.add(call["id"])
        if role == "tool":
            tool_call_id = m.get("tool_call_id")
            if not tool_call_id:
                # Legacy name-based tool results have no id to pair.
                if m.get("name"):
                    continue
                return False
            if tool_call_id not in pending:
                return False
            pending.remove(tool_call_id)
    return not pending


def evaluate_case(baseline: list[dict], candidate: list[dict], required_spans: Iterable[str] = (),
                  recovery_refs: Iterable[str] = (), task_success_baseline: bool = True,
                  task_success_candidate: bool = True, recover: Callable[[str], bool] | None = None) -> ABResult:
    required = list(required_spans)
    direct = all(any(span in str(m.get("content", "")) for m in candidate) for span in required)
    refs = list(recovery_refs)
    recoverable = all(recover(r) for r in refs) if refs and recover else (not required or direct)
    failures = []
    if required and not direct and not recoverable: failures.append("required span lost without verified recovery")
    if task_success_baseline and not task_success_candidate: failures.append("candidate task regression")
    valid = _valid(candidate)
    if not valid: failures.append("invalid role/tool structure")
    return ABResult(_size(baseline), _size(candidate), direct, recoverable, valid,
                    task_success_baseline, task_success_candidate, not failures, failures)


def evaluate_ab(cases: Iterable[dict]) -> dict:
    results = []
    for case in cases:
        results.append(asdict(evaluate_case(**case)))
    return {"passed": all(r["passed"] for r in results), "cases": results, "count": len(results)}

if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="Offline ContextPilot-lite A/B gate")
    p.add_argument("fixture", help="JSON list of evaluate_case kwargs")
    args = p.parse_args()
    with open(args.fixture, encoding="utf-8") as f: cases = json.load(f)
    print(json.dumps(evaluate_ab(cases), indent=2, sort_keys=True))
