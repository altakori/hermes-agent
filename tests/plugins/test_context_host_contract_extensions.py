from __future__ import annotations

import json

from plugins.context_engine import discover_context_engines, load_context_engine
from plugins.context_engine.context_pilot_lite import ContextPilotLite
from plugins.context_engine.context_pilot_lite.evaluator import evaluate_case


def make_engine(tmp_path, mode="shadow"):
    return ContextPilotLite(model="test-model", provider="test-provider", context_length=4096, api_key="", base_url="", mode=mode, archive_dir=tmp_path)


def test_discovery_and_shadow_noop(tmp_path):
    names = {name for name, _description, _available in discover_context_engines()}
    assert "context_pilot_lite" in names
    engine = load_context_engine("context_pilot_lite")
    assert isinstance(engine, ContextPilotLite)
    request = [{"role": "system", "content": "stable"}, {"role": "user", "content": "question"}]
    before = json.dumps(request, sort_keys=True, ensure_ascii=False)
    assert engine.select_context(request, conversation_messages=list(request)) is None
    assert json.dumps(request, sort_keys=True, ensure_ascii=False) == before


def test_engine_config_and_full_update_model_contract(monkeypatch):
    import hermes_cli.config as config_module

    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {"context": {"engine_config": {"mode": "active", "session_id": "configured"}}},
    )
    engine = load_context_engine("context_pilot_lite")
    assert engine.mode == "active" and engine.session_id == "configured"
    engine.update_model(
        model="test-model",
        context_length=8192,
        base_url="http://127.0.0.1:1",
        api_key="",
        provider="test-provider",
        api_mode="chat",
    )
    assert engine.context_length == 8192
    assert engine.get_automatic_compaction_status_message(
        phase="preflight", default_message="Compacting", prompt_tokens=100
    ) == "Compacting"


def test_one_tool_schema_and_sensitive_archive_is_not_exact(tmp_path):
    engine = make_engine(tmp_path); schemas = engine.get_tool_schemas()
    assert len(schemas) == 1
    actions = schemas[0]["parameters"]["properties"]["action"]["enum"]
    assert actions == ["note", "archive", "compress", "exclude", "recall"]
    ref = engine._archive_segment("segment-1", "secret=abcdefghijklmnopqrstuvwxyz")
    recalled = json.loads(engine.handle_tool_call("context_manage", {"action": "recall", "content": ref}))
    assert recalled["ok"] is False
    assert "not exact/verified" in recalled["error"]
    assert "abcdefghijklmnopqrstuvwxyz" not in json.dumps(recalled)
    exact_ref = engine._archive_segment("segment-2", "verified public result")
    exact = json.loads(engine.handle_tool_call("context_manage", {"action": "recall", "content": exact_ref}))
    assert exact["ok"] and exact["content"] == "verified public result"


def test_active_mode_preserves_tool_pair(tmp_path):
    engine = make_engine(tmp_path, mode="active")
    messages = [{"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "c1", "content": "result"}]
    selected = engine.select_context(messages, conversation_messages=list(messages))
    assert selected is None
    malformed_current = [{"role": "tool", "content": "current result without id"}]
    assert engine.select_context(malformed_current) is None

def test_active_note_and_compress_actions_are_effective(tmp_path):
    engine = make_engine(tmp_path, mode="active")
    note = json.loads(engine.handle_tool_call("context_manage", {"action": "note", "content": "retain finding A"}))
    assert note["ok"]
    selected = engine.select_context([{"role": "user", "content": "next"}])
    assert selected is not None and "retain finding A" in selected[-1]["content"]
    result = json.loads(engine.handle_tool_call("context_manage", {"action": "compress"}))
    assert result["ok"] and engine.should_compress(0)


def test_compress_uses_keyword_contract(tmp_path, monkeypatch):
    engine = make_engine(tmp_path); seen = {}
    def fake(messages, **kwargs): seen.update(kwargs); return messages
    monkeypatch.setattr(engine._compressor, "compress", fake)
    messages = [{"role": "user", "content": "x"}]
    assert engine.compress(messages, current_tokens=12, focus_topic="topic") == messages
    assert seen["current_tokens"] == 12 and seen["focus_topic"] == "topic"


def test_wrapper_forwards_usage_metrics(tmp_path):
    engine = make_engine(tmp_path)
    engine._compressor.last_prompt_tokens = 123
    engine._compressor.last_completion_tokens = 7
    engine._compressor.last_total_tokens = 130
    engine._compressor.compression_count = 2
    assert engine.last_prompt_tokens == 123
    assert engine.last_completion_tokens == 7
    assert engine.last_total_tokens == 130
    assert engine.compression_count == 2


def test_evaluator_gates_evidence_and_dangling_tool_call():
    baseline = [{"role": "user", "content": "required evidence " + "x" * 100}]
    candidate = [{"role": "user", "content": "required evidence"}]
    result = evaluate_case(baseline, candidate, required_spans=["required evidence"])
    assert result.passed and result.candidate_input_size < result.baseline_input_size
    dangling = [{"role": "user", "content": "run"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]}]
    invalid = evaluate_case(baseline, dangling)
    assert not invalid.passed and "invalid role/tool structure" in invalid.failures


def test_evaluator_reports_cache_token_latency_and_recovery_usage():
    messages = [{"role": "user", "content": "retain evidence"}]
    result = evaluate_case(
        messages,
        messages,
        baseline_usage=[
            {"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 12,
             "latency_ms": 25.25, "recovery_calls": 1},
            {"input_tokens": 200, "cached_input_tokens": 150, "output_tokens": 8,
             "latency_ms": 10.5},
        ],
        candidate_usage=[
            {"input_tokens": 120, "cached_input_tokens": 100, "output_tokens": 10,
             "latency_ms": 12.0},
        ],
    )
    assert result.passed
    assert result.baseline_usage == {
        "turns": 2,
        "input_tokens": 300,
        "cached_input_tokens": 150,
        "cache_hit_ratio": 0.5,
        "cache_miss_turns": [1],
        "output_tokens": 20,
        "latency_ms": 35.75,
        "recovery_calls": 1,
    }
    assert result.candidate_usage["cache_hit_ratio"] == 100 / 120
    assert result.candidate_usage["cache_miss_turns"] == []


def test_evaluator_rejects_invalid_usage_evidence():
    messages = [{"role": "user", "content": "x"}]
    result = evaluate_case(
        messages,
        messages,
        candidate_usage=[{"input_tokens": 10, "cached_input_tokens": 11}],
    )
    assert not result.passed
    assert result.candidate_usage["turns"] == 0
    assert result.failures == [
        "invalid candidate usage: turn 1 cached input exceeds input tokens"
    ]

    invalid_latency = evaluate_case(
        messages,
        messages,
        candidate_usage=[{"latency_ms": float("inf")}],
    )
    assert not invalid_latency.passed
    assert invalid_latency.failures == [
        "invalid candidate usage: turn 1 contains an invalid latency"
    ]
