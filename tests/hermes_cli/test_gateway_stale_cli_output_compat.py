"""Backward-compatible gateway import during an in-place update."""

import subprocess
import sys


def test_gateway_import_tolerates_pre_line_input_cli_output():
    code = """
import sys
import types

import hermes_cli
import hermes_cli.cli_output as current

stale = types.ModuleType("hermes_cli.cli_output")
stale.__dict__.update(current.__dict__)
stale.__dict__.pop("line_input", None)
sys.modules["hermes_cli.cli_output"] = stale
hermes_cli.cli_output = stale

import hermes_cli.gateway as gateway

assert gateway.line_input is input
assert stale.line_input is input
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
