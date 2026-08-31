#!/usr/bin/env python3
"""Fail-closed offline evaluator for coding-context compression candidates.

Input is JSONL. Each row must contain ``original`` and ``compressed`` strings.
Optional ``required_spans`` lists exact identifiers/paths/errors that must
survive. Optional ``baseline_task_success`` and ``compressed_task_success``
booleans let the same gate reject downstream task regressions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def evaluate_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("at least one evaluation record is required")

    original_chars = 0
    compressed_chars = 0
    required_total = 0
    required_retained = 0
    task_pairs = 0
    task_regressions = 0
    failures: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        original = row.get("original")
        compressed = row.get("compressed")
        if not isinstance(original, str) or not isinstance(compressed, str):
            raise ValueError(f"record {index}: original/compressed must be strings")
        original_chars += len(original)
        compressed_chars += len(compressed)

        missing: list[str] = []
        spans = row.get("required_spans") or []
        if not isinstance(spans, list) or not all(isinstance(span, str) for span in spans):
            raise ValueError(f"record {index}: required_spans must be a string list")
        for span in spans:
            if not span:
                raise ValueError(f"record {index}: required_spans cannot contain an empty string")
            if span not in original:
                raise ValueError(
                    f"record {index}: required span is not present in original: {span!r}"
                )
            required_total += 1
            if span in compressed:
                required_retained += 1
            else:
                missing.append(span)

        baseline = row.get("baseline_task_success")
        candidate = row.get("compressed_task_success")
        regressed = False
        if baseline is not None or candidate is not None:
            if not isinstance(baseline, bool) or not isinstance(candidate, bool):
                raise ValueError(
                    f"record {index}: task success fields must both be booleans"
                )
            task_pairs += 1
            regressed = baseline and not candidate
            task_regressions += int(regressed)

        if missing or regressed:
            failures.append({
                "id": row.get("id", index),
                "missing_spans": missing,
                "task_regression": regressed,
            })

    return {
        "records": len(rows),
        "compression_ratio": (
            compressed_chars / original_chars if original_chars else 1.0
        ),
        "original_chars": original_chars,
        "compressed_chars": compressed_chars,
        "required_spans": required_total,
        "retained_spans": required_retained,
        "exact_span_recall": (
            required_retained / required_total if required_total else 1.0
        ),
        "task_pairs": task_pairs,
        "task_regressions": task_regressions,
        "failures": failures,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number}: expected a JSON object")
        records.append(row)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--min-exact-recall", type=float, default=1.0)
    parser.add_argument("--max-task-regressions", type=int, default=0)
    parser.add_argument("--max-compression-ratio", type=float, default=1.0)
    args = parser.parse_args(argv)

    try:
        report = evaluate_records(_read_jsonl(args.jsonl))
    except (OSError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        return 2

    passed = (
        report["exact_span_recall"] >= args.min_exact_recall
        and report["task_regressions"] <= args.max_task_regressions
        and report["compression_ratio"] <= args.max_compression_ratio
    )
    report["gates"] = {
        "min_exact_recall": args.min_exact_recall,
        "max_task_regressions": args.max_task_regressions,
        "max_compression_ratio": args.max_compression_ratio,
    }
    report["passed"] = passed
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
