"""Offline A/B evaluator for ContextPilot-lite (no model/network calls)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
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
    baseline_usage: dict[str, Any]
    candidate_usage: dict[str, Any]
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


def _usage(turns: Iterable[dict] | None) -> tuple[dict[str, Any], list[str]]:
    summary: dict[str, Any] = {
        "turns": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_hit_ratio": None,
        "cache_miss_turns": [],
        "per_turn": [],
        "output_tokens": 0,
        "latency_ms": 0.0,
        "recovery_calls": 0,
    }
    errors = []

    if turns is None:
        iterator = iter(())
    elif isinstance(turns, (dict, str, bytes)):
        return summary, ["usage evidence must be an iterable of turn objects"]
    else:
        try:
            iterator = iter(turns)
        except TypeError:
            return summary, ["usage evidence must be an iterable of turn objects"]

    for index, turn in enumerate(iterator, start=1):
        if not isinstance(turn, dict):
            errors.append(f"turn {index} is not an object")
            continue
        counts = [turn.get(name, 0) for name in (
            "input_tokens", "cached_input_tokens", "output_tokens", "recovery_calls"
        )]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in counts):
            errors.append(f"turn {index} contains an invalid count")
            continue
        input_tokens, cached_tokens, output_tokens, recovery_calls = counts
        latency_value = turn.get("latency_ms", 0.0)
        try:
            latency_ms = float(latency_value)
        except (TypeError, ValueError, OverflowError):
            errors.append(f"turn {index} contains an invalid latency")
            continue
        if (isinstance(latency_value, bool)
                or not isinstance(latency_value, (int, float))
                or latency_ms < 0
                or not math.isfinite(latency_ms)):
            errors.append(f"turn {index} contains an invalid latency")
            continue
        if cached_tokens > input_tokens:
            errors.append(f"turn {index} cached input exceeds input tokens")
            continue
        summary["turns"] += 1
        summary["input_tokens"] += input_tokens
        summary["cached_input_tokens"] += cached_tokens
        summary["output_tokens"] += output_tokens
        summary["latency_ms"] += latency_ms
        summary["recovery_calls"] += recovery_calls
        cache_miss = bool(input_tokens and not cached_tokens)
        if cache_miss:
            summary["cache_miss_turns"].append(index)
        summary["per_turn"].append({
            "turn": index,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "cache_hit_ratio": cached_tokens / input_tokens if input_tokens else None,
            "cache_miss": cache_miss,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 3),
            "recovery_calls": recovery_calls,
        })
    total_input = summary["input_tokens"]
    if total_input:
        summary["cache_hit_ratio"] = summary["cached_input_tokens"] / total_input
    summary["latency_ms"] = round(summary["latency_ms"], 3)
    return summary, errors


def evaluate_case(baseline: list[dict], candidate: list[dict], required_spans: Iterable[str] = (),
                  recovery_refs: Iterable[str] = (), task_success_baseline: bool = True,
                  task_success_candidate: bool = True, recover: Callable[[str], bool] | None = None,
                  baseline_usage: Iterable[dict] | None = None,
                  candidate_usage: Iterable[dict] | None = None) -> ABResult:
    required = list(required_spans)
    direct = all(any(span in str(m.get("content", "")) for m in candidate) for span in required)
    refs = list(recovery_refs)
    recoverable = all(recover(r) for r in refs) if refs and recover else (not required or direct)
    failures = []
    if required and not direct and not recoverable: failures.append("required span lost without verified recovery")
    if task_success_baseline and not task_success_candidate: failures.append("candidate task regression")
    valid = _valid(candidate)
    if not valid: failures.append("invalid role/tool structure")
    baseline_usage_summary, baseline_usage_errors = _usage(baseline_usage)
    candidate_usage_summary, candidate_usage_errors = _usage(candidate_usage)
    failures.extend(f"invalid baseline usage: {error}" for error in baseline_usage_errors)
    failures.extend(f"invalid candidate usage: {error}" for error in candidate_usage_errors)
    return ABResult(_size(baseline), _size(candidate), direct, recoverable, valid,
                    task_success_baseline, task_success_candidate, baseline_usage_summary,
                    candidate_usage_summary, not failures, failures)


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
