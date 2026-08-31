import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evaluate_compression_segments import evaluate_records


def test_evaluator_reports_exact_span_recall_ratio_and_task_regression():
    report = evaluate_records([
        {"id": "kept", "original": "path/a.py ERR_1 keep", "compressed": "path/a.py ERR_1", "required_spans": ["path/a.py", "ERR_1"], "baseline_task_success": True, "compressed_task_success": True},
        {"id": "lost", "original": "path/b.py ERR_2 keep", "compressed": "summary", "required_spans": ["path/b.py", "ERR_2"], "baseline_task_success": True, "compressed_task_success": False},
    ])
    assert report["records"] == 2
    assert report["required_spans"] == 4
    assert report["exact_span_recall"] == pytest.approx(0.5)
    assert report["task_regressions"] == 1
    assert 0 < report["compression_ratio"] < 1
    assert report["failures"][0]["id"] == "lost"


def test_evaluator_rejects_required_span_absent_from_original():
    with pytest.raises(ValueError, match="not present in original"):
        evaluate_records([{"id": "bad", "original": "abc", "compressed": "abc", "required_spans": ["invented"]}])


def test_cli_fails_closed_on_regression_and_passes_clean_fixture(tmp_path):
    script = Path(__file__).parents[2] / "scripts" / "evaluate_compression_segments.py"
    failing = tmp_path / "failing.jsonl"
    failing.write_text(json.dumps({"id": "x", "original": "EXACT", "compressed": "summary", "required_spans": ["EXACT"], "baseline_task_success": True, "compressed_task_success": False}) + "\n", encoding="utf-8")
    failed = subprocess.run([sys.executable, str(script), str(failing)], capture_output=True, text=True, check=False)
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["passed"] is False

    passing = tmp_path / "passing.jsonl"
    passing.write_text(json.dumps({"id": "x", "original": "EXACT and filler", "compressed": "EXACT", "required_spans": ["EXACT"], "baseline_task_success": True, "compressed_task_success": True}) + "\n", encoding="utf-8")
    passed = subprocess.run([sys.executable, str(script), str(passing)], capture_output=True, text=True, check=False)
    assert passed.returncode == 0
    assert json.loads(passed.stdout)["passed"] is True
