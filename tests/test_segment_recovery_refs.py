"""Typed, content-addressed tool-segment recovery tests."""

import json
from unittest.mock import patch

from agent.compression_segments import classify_tool_segment, make_recovery_ref
from agent.context_compressor import ContextCompressor, _lean_recovery_stub
from hermes_state import SessionDB
from tools.session_search_tool import session_search


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        value = ContextCompressor(model="test/model", threshold_percent=0.85, protect_first_n=1, protect_last_n=2, quiet_mode=True)
        _ = value.context_length
        return value


def test_segment_classifier_distinguishes_coding_outputs():
    assert classify_tool_segment("read_file", '{"path":"app.py"}', "1|print('x')") == "file_read"
    assert classify_tool_segment("terminal", '{"command":"pytest -q"}', "2 passed") == "test_log"
    assert classify_tool_segment("terminal", '{"command":"python app.py"}', "Traceback (most recent call last):") == "traceback"
    assert classify_tool_segment("terminal", '{"command":"git status"}', "clean") == "shell_output"


def test_recovery_ref_matches_durable_content_and_is_sensitive():
    first = make_recovery_ref("identifier = 'exact'")
    second = make_recovery_ref("identifier = 'changed'")
    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 64
    assert first != second
    assert make_recovery_ref("bad-\ud83d") == make_recovery_ref("bad-\ufffd")


def test_lean_stub_is_typed_and_has_exact_recovery_command():
    content = "Traceback (most recent call last):\nValueError: exact-error"
    ref = make_recovery_ref(content)
    stub = _lean_recovery_stub("terminal", len(content), "session-1", content=content, tool_args='{"command":"python app.py"}')
    assert stub.startswith(f"[SEGMENT:traceback][REF:{ref}]")
    assert f'session_search(query="ref:{ref}", session_id="session-1")' in stub


def test_duplicate_reread_points_to_exact_newest_content():
    compressor = _compressor()
    content = "1|exact line\n2|identifier = 'DO_NOT_PARAPHRASE'\n" * 20
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "old", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}]},
        {"role": "tool", "tool_call_id": "old", "tool_name": "read_file", "content": content},
        {"role": "assistant", "tool_calls": [{"id": "new", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}]},
        {"role": "tool", "tool_call_id": "new", "tool_name": "read_file", "content": content},
    ]
    pruned, count = compressor._prune_old_tool_results(messages, protect_tail_count=2)
    ref = make_recovery_ref(content)
    assert count == 1
    assert f"[SEGMENT:file_read][REF:{ref}]" in pruned[1]["content"]
    assert pruned[3]["content"] == content


def test_typed_stub_never_inflates_short_tool_output():
    compressor = _compressor()
    compressor._session_id = "session-" + "x" * 200
    content = "x" * 220
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call", "type": "function", "function": {"name": "terminal", "arguments": '{"command":"run"}'}}]},
        {"role": "tool", "tool_call_id": "call", "tool_name": "terminal", "content": content},
        {"role": "user", "content": "recent"},
        {"role": "assistant", "content": "tail"},
    ]
    pruned, count = compressor._prune_old_tool_results(messages, protect_tail_count=2)
    assert count == 0
    assert pruned[1]["content"] == content


def test_session_search_recovers_inactive_compacted_output_by_verified_ref(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="s1", source="cli", model="test")
        content = "Traceback (most recent call last):\nValueError: EXACT_123"
        db.append_message("s1", "tool", content, tool_call_id="call-1", tool_name="terminal")
        ref = make_recovery_ref(content)
        db.archive_and_compact("s1", [{"role": "user", "content": f"[SEGMENT:traceback][REF:{ref}] compacted"}])
        assert all(message["content"] != content for message in db.get_messages("s1"))
        payload = json.loads(session_search(query=f"ref:{ref}", session_id="s1", db=db))
        assert payload["success"] is True
        assert payload["mode"] == "recover"
        assert payload["verified"] is True
        assert payload["message"]["content"] == content
        assert payload["segment_type"] == "traceback"
    finally:
        db.close()


def test_session_search_ref_fails_closed_on_digest_mismatch(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="s1", source="cli", model="test")
        db.append_message("s1", "tool", "actual", tool_call_id="call-1", tool_name="terminal")
        payload = json.loads(session_search(query=f"ref:{make_recovery_ref('different')}", session_id="s1", db=db))
        assert payload["success"] is False
        assert payload["mode"] == "recover"
        assert "not found" in payload["error"].lower()
    finally:
        db.close()


def test_session_search_never_recovers_rewound_tool_message(tmp_path):
    db = SessionDB(db_path=tmp_path / "rewound.db")
    try:
        db.create_session(session_id="s2", source="cli", model="test")
        db.append_messages_batch(
            "s2",
            [
                {"role": "user", "content": "keep this user turn"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "removed-call",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "removed-call",
                    "tool_name": "terminal",
                    "content": "explicitly rewound secret",
                },
            ],
        )
        user_id = db.get_messages("s2")[0]["id"]
        db.rewind_to_message("s2", user_id)
        ref = make_recovery_ref("explicitly rewound secret")
        payload = json.loads(
            session_search(query=f"ref:{ref}", session_id="s2", db=db)
        )
        assert payload["success"] is False
        assert payload["mode"] == "recover"
    finally:
        db.close()
