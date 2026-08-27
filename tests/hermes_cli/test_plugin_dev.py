from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli.subcommands.plugins import build_plugins_parser


def _parse_plugins_args(*argv: str):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_plugins_parser(subparsers, cmd_plugins=lambda args: None)
    return parser.parse_args(["plugins", *argv])


def test_plugins_parser_exposes_doctor() -> None:
    doctor = _parse_plugins_args("doctor", "sample", "--ci", "--json")

    assert (doctor.plugins_action, doctor.target, doctor.ci, doctor.json) == (
        "doctor",
        "sample",
        True,
        True,
    )


def test_doctor_readiness_is_derived_from_current_report() -> None:
    from hermes_cli.plugin_dev import DoctorReport

    manifest = type(
        "Manifest",
        (),
        {"name": "sample", "version": "1.0.0", "kind": "standalone"},
    )()
    unknown = DoctorReport(Path("unknown"))
    ready = DoctorReport(Path("ready"), manifest=manifest)
    degraded = DoctorReport(Path("degraded"), manifest=manifest)
    degraded.warning("advisory mismatch")
    unavailable = DoctorReport(Path("unavailable"), manifest=manifest)
    unavailable.error("registration failed")

    assert unknown.readiness == "unknown"
    assert ready.readiness == "ready"
    assert degraded.readiness == "degraded"
    assert unavailable.readiness == "unavailable"
    assert unavailable.to_dict()["findings"] == [
        {"level": "error", "message": "registration failed"}
    ]


def test_plugin_doctor_json_reports_diagnostic_readiness(monkeypatch, capsys) -> None:
    from hermes_cli import plugins_cmd
    from hermes_cli.plugin_dev import DoctorReport

    report = DoctorReport(Path("sample"), manifest=type(
        "Manifest",
        (),
        {"name": "sample", "version": "1.2.3", "kind": "standalone"},
    )())
    report.warning("declared tool was not registered")

    def noisy_doctor(target):
        print("PLUGIN-IMPORT-NOISE")
        return report

    monkeypatch.setattr("hermes_cli.plugin_dev.doctor_plugin", noisy_doctor)

    plugins_cmd.cmd_plugin_doctor("sample", json_output=True)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == "PLUGIN-IMPORT-NOISE\n"
    assert payload["readiness"] == "degraded"
    assert payload["ok"] is True
    assert payload["manifest"] == {
        "name": "sample",
        "version": "1.2.3",
        "kind": "standalone",
    }


def test_plugin_doctor_ci_preserves_error_only_exit_gate(monkeypatch, capsys) -> None:
    from hermes_cli import plugins_cmd
    from hermes_cli.plugin_dev import DoctorReport

    degraded = DoctorReport(Path("degraded"), manifest=type(
        "Manifest",
        (),
        {"name": "degraded", "version": "1.0.0", "kind": "standalone"},
    )())
    degraded.warning("advisory")
    monkeypatch.setattr("hermes_cli.plugin_dev.doctor_plugin", lambda target: degraded)

    plugins_cmd.cmd_plugin_doctor("degraded", ci=True, json_output=True)
    assert json.loads(capsys.readouterr().out)["readiness"] == "degraded"

    unavailable = DoctorReport(Path("unavailable"))
    unavailable.error("cannot load")
    monkeypatch.setattr("hermes_cli.plugin_dev.doctor_plugin", lambda target: unavailable)

    with pytest.raises(SystemExit) as exit_info:
        plugins_cmd.cmd_plugin_doctor("unavailable", ci=True, json_output=True)
    assert exit_info.value.code == 1
    assert json.loads(capsys.readouterr().out)["readiness"] == "unavailable"


def test_doctor_uses_registration_to_reject_bad_hook_and_callback_signature(
    tmp_path: Path,
) -> None:
    from hermes_cli.plugin_dev import doctor_plugin

    plugin = tmp_path / "bad-plugin"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text(
        "\n".join(
            [
                "name: bad-plugin",
                "version: 0.1.0",
                "description: broken contract",
                "provides_hooks:",
                "  - typo_hook",
                "  - pre_tool_call",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        "def callback(tool_name):\n"
        "    return None\n\n"
        "def register(ctx):\n"
        "    ctx.register_hook('typo_hook', callback)\n"
        "    ctx.register_hook('pre_tool_call', callback)\n",
        encoding="utf-8",
    )

    report = doctor_plugin(plugin)
    messages = "\n".join(f.message for f in report.findings)
    assert report.ok is False
    assert "unknown hook 'typo_hook'" in messages
    assert "must accept **kwargs" in messages


def test_doctor_accepts_manifest_defaults_from_runtime_parser(tmp_path: Path) -> None:
    from hermes_cli.plugin_dev import doctor_plugin

    plugin = tmp_path / "minimal"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("name: minimal\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n    pass\n", encoding="utf-8"
    )

    report = doctor_plugin(plugin)
    assert report.ok, report.format_text()
    assert report.manifest is not None
    assert report.manifest.kind == "standalone"


def test_doctor_restores_global_tool_policy_and_module_state(tmp_path: Path) -> None:
    import sys

    from hermes_cli.plugin_dev import doctor_plugin
    from tools.registry import registry

    target = tmp_path / "cleanup-plugin"
    target.mkdir()
    (target / "plugin.yaml").write_text(
        "name: cleanup-plugin\nprovides_tools: [cleanup_plugin_ping]\n",
        encoding="utf-8",
    )
    (target / "__init__.py").write_text(
        "import json\n\n"
        "def ping(args, **kwargs):\n    return json.dumps({'ok': True})\n\n"
        "def register(ctx):\n"
        "    ctx.register_tool(name='cleanup_plugin_ping', toolset='cleanup', "
        "schema={'name': 'cleanup_plugin_ping', 'description': 'test', "
        "'parameters': {'type': 'object'}}, handler=ping)\n",
        encoding="utf-8",
    )
    before_policy = dict(registry._plugin_override_policy)
    before_modules = {
        name
        for name in sys.modules
        if name == "hermes_plugins" or name.startswith("hermes_plugins.")
    }

    report = doctor_plugin(target)

    assert report.ok, report.format_text()
    assert report.registered_tools == ("cleanup_plugin_ping",)
    assert registry.get_entry("cleanup_plugin_ping") is None
    assert registry._plugin_override_policy == before_policy
    after_modules = {
        name
        for name in sys.modules
        if name == "hermes_plugins" or name.startswith("hermes_plugins.")
    }
    assert after_modules == before_modules


def test_doctor_blocks_live_network(tmp_path: Path) -> None:
    from hermes_cli.plugin_dev import doctor_plugin

    plugin = tmp_path / "network-plugin"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("name: network-plugin\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "import socket\n\n"
        "def register(ctx):\n"
        "    socket.create_connection(('example.com', 443))\n",
        encoding="utf-8",
    )

    report = doctor_plugin(plugin)
    assert report.ok is False
    assert "network access is disabled while Plugin Doctor runs" in report.format_text()
