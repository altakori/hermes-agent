from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

PLUGIN = Path(__file__).parents[2] / "plugins" / "persistent-safety-guard" / "__init__.py"


def load_plugin():
    spec = importlib.util.spec_from_file_location("persistent_safety_guard_test", PLUGIN)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeContext:
    def __init__(self, data_dir: Path, config=None):
        self.state = type("State", (), {"data_dir": data_dir})()
        self.config = config or {}
        self.hooks, self.commands = {}, {}

    def register_hook(self, name, callback):
        self.hooks[name] = callback

    def register_command(self, name, callback, **kwargs):
        self.commands[name] = callback


def test_hash_chain_partition_and_no_raw_intent(tmp_path):
    mod = load_plugin()
    ledger = mod.SafetyLedger(tmp_path)
    ledger.add_flag("child-a", "risk-a", root_session_id="root-a")
    ledger.add_flag("child-b", "risk-b", root_session_id="root-b")
    ledger.deny_intent("do not retain this plaintext", "child-a", root_session_id="root-a")
    assert ledger.verify()
    assert ledger.state("root-a")["risk_flags"] == ["risk-a"]
    assert ledger.state("root-b")["risk_flags"] == ["risk-b"]
    disk = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert "do not retain this plaintext" not in disk
    lines = disk.splitlines()
    record = json.loads(lines[0]); record["event_type"] = "tampered"
    lines[0] = json.dumps(record)
    (tmp_path / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not ledger.verify()


def test_concurrent_appends_keep_one_chain(tmp_path):
    mod = load_plugin(); ledger = mod.SafetyLedger(tmp_path)

    def append(i):
        ledger.append(
            scope="agent",
            root_session_id="root",
            session_id="session",
            event_type=f"event-{i}",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append, range(40)))
    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["seq"] for record in records] == list(range(1, 41))
    assert ledger.verify()


def test_budget_is_explicit_and_non_decaying(tmp_path):
    mod = load_plugin(); ledger = mod.SafetyLedger(tmp_path)
    ledger.set_budget(2, "child", root_session_id="root")
    assert ledger.consume_budget("child", root_session_id="root")
    assert ledger.consume_budget("child", root_session_id="root")
    assert not ledger.consume_budget("child", root_session_id="root")
    assert ledger.state("root")["irreversible_action_budget"] == 0


def test_compression_and_worker_inherit_root_and_fingerprints(tmp_path):
    mod = load_plugin(); guard = mod.Guard(tmp_path)
    guard.pre_llm("parent", user_message="P" * 180)
    guard.session_started("compressed", old_session_id="parent", boundary_reason="compression")
    guard.child("worker", parent_session_id="compressed")
    assert guard.sessions["compressed"]["root"] == "parent"
    assert guard.sessions["worker"]["root"] == "parent"
    assert guard.sessions["worker"]["fingerprints"]
    # A fresh process-local Guard recovers the child/root binding and salted
    # fingerprints from disk without storing protected plaintext.
    fresh = mod.Guard(tmp_path)
    assert fresh._session("worker")["root"] == "parent"
    assert fresh._session("worker")["fingerprints"]

    script = """
import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location('guard_child', pathlib.Path(sys.argv[1]))
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
state = module.Guard(pathlib.Path(sys.argv[2]))._session('worker')
print(json.dumps({'root': state['root'], 'count': len(state['fingerprints'])}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(PLUGIN), str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    child_state = json.loads(result.stdout)
    assert child_state["root"] == "parent" and child_state["count"] > 0


def test_mcp_context_and_secret_policy(monkeypatch, tmp_path):
    mod = load_plugin(); from tools import mcp_tool
    tool = "mcp__evil_server__send"
    monkeypatch.setitem(mcp_tool._mcp_tool_server_names, tool, "evil-server")
    monkeypatch.setitem(mcp_tool._server_trust_levels, "evil-server", "untrusted")
    guard = mod.Guard(tmp_path); protected = "confidential user history " * 12
    guard.pre_llm("s1", user_message=protected)
    result = guard.pre_tool(tool, {"payload": "prefix::" + protected + "::suffix"}, "s1")
    assert result and "protected_context" in result["message"]
    assert "previously denied" in guard.pre_tool(
        tool, {"payload": "prefix::" + protected + "::suffix"}, "s1"
    )["message"]
    monkeypatch.setitem(mcp_tool._server_trust_levels, "evil-server", "full")
    assert guard.pre_tool(tool, {"payload": protected + " allowed"}, "s1") is None
    secret = "api_key=abcdefghijklmnopqrstuvwxyz"
    assert "secret" in guard.pre_tool(tool, {"payload": secret}, "s1")["message"]
    assert "secret" in guard.pre_tool(
        tool, {"api_key": "abcdefghijklmnopqrstuvwxyz"}, "fresh-json"
    )["message"]
    assert guard.pre_tool("web_search", {"query": "q" * 500}, "s1") is None
    assert secret not in (tmp_path / "events.jsonl").read_text(encoding="utf-8")


def test_authorization_persists_hashes_not_raw_metadata(tmp_path):
    mod = load_plugin(); ledger = mod.SafetyLedger(tmp_path)
    ledger.set_authorization(
        {"scope": "private scope token=not-for-disk", "provenance": "operator secret provenance", "expiry": "2099-01-01T00:00:00Z"},
        "s1",
        root_session_id="root",
    )
    disk = (tmp_path / "active_state.json").read_text(encoding="utf-8") + (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert "not-for-disk" not in disk and "operator secret provenance" not in disk
    state = ledger.state("root")["authorization"]
    assert state["scope_hash"] and state["provenance_hash"]


def test_active_state_and_salt_tamper_fail_closed(tmp_path):
    mod = load_plugin(); ledger = mod.SafetyLedger(tmp_path)
    ledger.add_flag("s", "persist", root_session_id="root")
    state_path = tmp_path / "active_state.json"
    original_state = state_path.read_bytes()
    state_path.write_text("{}", encoding="utf-8")
    with pytest.raises(mod.SafetyStateIntegrityError):
        ledger.state("root")
    assert not ledger.verify()
    state_path.write_bytes(original_state)
    (tmp_path / "ledger.salt").write_bytes(b"x" * 32)
    with pytest.raises(mod.SafetyStateIntegrityError):
        ledger.state("root")
    assert not ledger.verify()

    mod._guard = None
    ctx = FakeContext(tmp_path); mod.register(ctx)
    blocked = ctx.hooks["pre_tool_call"](
        tool_name="mcp__evil__send", args={"payload": "ordinary"}, session_id="s"
    )
    assert blocked["action"] == "block" and "failed closed" in blocked["message"]


def test_deleted_active_state_with_existing_ledger_fails_closed(tmp_path):
    mod = load_plugin(); ledger = mod.SafetyLedger(tmp_path)
    ledger.add_flag("s", "persist", root_session_id="root")
    (tmp_path / "active_state.json").unlink()
    (tmp_path / "active_state.mac").unlink()
    with pytest.raises(mod.SafetyStateIntegrityError):
        ledger.state("root")


def test_register_and_explicit_scoped_clear(tmp_path):
    mod = load_plugin(); mod._guard = None; ctx = FakeContext(tmp_path); mod.register(ctx)
    assert {"on_session_start", "pre_llm_call", "pre_tool_call", "subagent_start"} <= set(ctx.hooks)
    assert ctx.commands["safety-clear"]("").startswith("Usage:")
    mod._guard.ledger.add_flag("s1", "risk", root_session_id="root")
    mod._guard.session_started("s1", root_session_id="root")
    mod._guard.pre_llm("s1", user_message="protected " + "x" * 200)
    assert "s1" in mod._guard.sessions
    assert "root" in ctx.commands["safety-clear"]("root")
    state = mod._guard.ledger.state("root")
    assert state["risk_flags"] == [] and state["protected_fingerprints"] == []
    assert "s1" not in mod._guard.sessions


def test_registered_mcp_hook_fails_closed_on_guard_error(tmp_path, monkeypatch):
    mod = load_plugin(); mod._guard = None
    ctx = FakeContext(tmp_path, {"max_ledger_bytes": 2048}); mod.register(ctx)
    assert mod._guard.ledger.max_bytes == 2048
    monkeypatch.setattr(mod._guard, "pre_tool", lambda **kwargs: (_ for _ in ()).throw(OSError("disk unavailable")))
    assert ctx.hooks["pre_tool_call"](tool_name="mcp__evil__send", args={}, session_id="s")["action"] == "block"
    assert ctx.hooks["pre_tool_call"](tool_name="terminal", args={}, session_id="s") is None


def test_final_mcp_guard_cannot_be_skipped(monkeypatch):
    import hermes_cli.plugins as plugin_runtime
    import model_tools

    def callback(**kwargs):
        return {"action": "block", "message": "final guard block"}

    callback.__hermes_final_mcp_egress__ = True
    manager = type("Manager", (), {"_hooks": {"pre_tool_call": [callback]}})()
    monkeypatch.setattr(plugin_runtime, "get_plugin_manager", lambda: manager)
    assert plugin_runtime._dispatch_final_mcp_egress_hooks("mcp__evil__send", {}) == "final guard block"
    result = model_tools.handle_function_call(
        "mcp__evil__send", {}, task_id="t", session_id="s", skip_pre_tool_call_hook=True
    )
    assert "final guard block" in str(result)


def test_fingerprint_windows_cover_unsampled_offsets(tmp_path, monkeypatch):
    mod = load_plugin(); guard = mod.Guard(tmp_path)
    protected = "".join(f"{i:08x}" for i in range(5000))
    guard.pre_llm("s", user_message=protected)
    # A copied span at a non-aligned offset contains at least one uniformly
    # sampled source window.
    sample = protected[17:217]
    assert set(mod._fingerprints(sample, guard.ledger._salt())).intersection(
        guard.sessions["s"]["fingerprints"]
    )
    from tools import mcp_tool
    tool = "mcp__evil_server__send"
    monkeypatch.setitem(mcp_tool._mcp_tool_server_names, tool, "evil-server")
    monkeypatch.setitem(mcp_tool._server_trust_levels, "evil-server", "untrusted")
    # Candidate scanning is not prefix-limited, and uniform source sampling
    # covers later transcript regions for long copied spans.
    tail_copy = protected[-400:]
    assert guard.pre_tool(tool, {"payload": "z" * 5000 + tail_copy}, "s")["action"] == "block"
    oversized = guard.pre_tool(tool, {"payload": "x" * (mod._MAX_MCP_ARG_TEXT + 1)}, "fresh")
    assert "oversized_argument" in oversized["message"]


def test_nested_argument_and_history_fingerprint_coverage(tmp_path, monkeypatch):
    mod = load_plugin(); from tools import mcp_tool
    tool = "mcp__evil_server__send"
    monkeypatch.setitem(mcp_tool._mcp_tool_server_names, tool, "evil-server")
    monkeypatch.setitem(mcp_tool._server_trust_levels, "evil-server", "untrusted")
    guard = mod.Guard(tmp_path)
    older = "older protected history " + "h" * 600
    current = "".join(f"current-{i:05d};" for i in range(3000))
    guard.pre_llm("s", user_message=current, conversation_history=[{"role": "user", "content": older}])
    nested = [{"safe": index} for index in range(64)] + [{"payload": older}]
    blocked = guard.pre_tool(tool, {"items": nested}, "s")
    assert blocked and blocked["action"] == "block"


def test_fingerprint_capacity_exhaustion_fails_closed(tmp_path, monkeypatch):
    mod = load_plugin(); from tools import mcp_tool
    tool = "mcp__evil_server__send"
    monkeypatch.setitem(mcp_tool._mcp_tool_server_names, tool, "evil-server")
    monkeypatch.setitem(mcp_tool._server_trust_levels, "evil-server", "untrusted")
    guard = mod.Guard(tmp_path)
    full = [f"{index:064x}" for index in range(mod._MAX_FINGERPRINTS)]
    guard.ledger.set_fingerprints("s", full, root_session_id="s")
    guard.sessions.clear()
    guard.pre_llm("s", user_message="new protected context " + "n" * 300)
    state = guard.ledger.state("s")
    assert "fingerprint_capacity_exhausted" in state["risk_flags"]
    blocked = guard.pre_tool(tool, {"payload": "ordinary"}, "s")
    assert "fingerprint_capacity_exhausted" in blocked["message"]


def test_ledger_quota_failure_is_explicit(tmp_path):
    mod = load_plugin(); ledger = mod.SafetyLedger(tmp_path, max_bytes=1024)
    for i in range(10):
        try:
            ledger.append(scope="agent", root_session_id="r", session_id="s", event_type="event", metadata={"i": i, "padding": "x" * 300})
        except RuntimeError as exc:
            assert "quota exceeded" in str(exc); break
    else:
        pytest.fail("expected bounded ledger quota to fail explicitly")
