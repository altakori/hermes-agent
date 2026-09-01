"""Regression for updating from releases before ``cli_output.line_input``."""

from __future__ import annotations

import subprocess
import sys


def test_gateway_import_survives_stale_cli_output_without_line_input():
    """The post-pull restart imports new gateway code in the old process."""
    probe = (
        # The old updater has already cached config before the checkout moves.
        "import hermes_cli.config; "
        "import hermes_cli.cli_output as cli_output; "
        "del cli_output.line_input; "
        "import hermes_cli.gateway"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr