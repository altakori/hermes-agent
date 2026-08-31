from __future__ import annotations

from types import SimpleNamespace

from agent.conversation_compression import _notify_context_engine_compression_complete


def test_committed_compression_emits_plugin_session_lineage(monkeypatch):
    events = []
    engine_events = []

    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **kwargs: events.append((name, kwargs)),
    )

    compressor = SimpleNamespace(
        on_session_start=lambda session_id, **kwargs: engine_events.append(
            (session_id, kwargs)
        )
    )
    agent = SimpleNamespace(
        context_compressor=compressor,
        model="test-model",
        platform="telegram",
        _root_session_id="root-session",
        _gateway_session_key="chat:thread",
    )

    assert _notify_context_engine_compression_complete(
        agent,
        new_session_id="child-session",
        old_session_id="parent-session",
    )

    plugin_events = [payload for name, payload in events if name == "on_session_start"]
    assert len(plugin_events) == 1
    assert plugin_events[0]["session_id"] == "child-session"
    assert plugin_events[0]["old_session_id"] == "parent-session"
    assert plugin_events[0]["root_session_id"] == "root-session"
    assert plugin_events[0]["boundary_reason"] == "compression"
    assert engine_events == [
        (
            "child-session",
            {
                "boundary_reason": "compression",
                "old_session_id": "parent-session",
                "platform": "telegram",
                "conversation_id": "chat:thread",
            },
        )
    ]
