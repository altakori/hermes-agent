"""Tests for gateway /yolo session scoping."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from tools.approval import disable_session_yolo, is_session_yolo_enabled


@pytest.fixture(autouse=True)
def _clean_yolo_state(monkeypatch):
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    disable_session_yolo("agent:main:telegram:dm:chat-a")
    disable_session_yolo("agent:main:telegram:dm:chat-b")
    yield
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    disable_session_yolo("agent:main:telegram:dm:chat-a")
    disable_session_yolo("agent:main:telegram:dm:chat-b")


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = None
    runner.config = None
    return runner


def _make_event(chat_id: str) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id=f"user-{chat_id}",
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
    )
    return MessageEvent(text="/yolo", source=source)


@pytest.mark.asyncio
async def test_yolo_command_toggles_only_current_session(monkeypatch):
    runner = _make_runner()

    event_a = _make_event("chat-a")
    session_a = runner._session_key_for_source(event_a.source)
    session_b = runner._session_key_for_source(_make_event("chat-b").source)

    result_on = await runner._handle_yolo_command(event_a)

    assert "ON" in result_on
    assert is_session_yolo_enabled(session_a) is True
    assert is_session_yolo_enabled(session_b) is False
    assert os.environ.get("HERMES_YOLO_MODE") is None

    result_off = await runner._handle_yolo_command(event_a)

    assert "OFF" in result_off
    assert is_session_yolo_enabled(session_a) is False
    assert os.environ.get("HERMES_YOLO_MODE") is None


@pytest.mark.asyncio
async def test_yolo_command_refreshes_current_telegram_topic_title(monkeypatch):
    runner = _make_runner()
    event = _make_event("chat-a")
    event.source.thread_id = "42"
    runner._telegram_topic_mode_enabled = lambda _source: True
    runner._session_db = SimpleNamespace(
        get_session_title=AsyncMock(return_value="Current Topic")
    )
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="session-42")
        ),
    )
    runner._schedule_telegram_topic_title_rename = MagicMock()

    await runner._handle_yolo_command(event)
    runner._schedule_telegram_topic_title_rename.assert_called_once_with(
        event.source,
        "session-42",
        "Current Topic",
    )

    runner._schedule_telegram_topic_title_rename.reset_mock()
    await runner._handle_yolo_command(event)
    runner._schedule_telegram_topic_title_rename.assert_called_once_with(
        event.source,
        "session-42",
        "Current Topic",
    )
