"""Deterministic typing and content-addressed recovery for compressed tool output.

The compressor may replace bulky tool results with short stubs, but the
persisted session transcript remains the immutable source.  A recovery
reference is the SHA-256 digest of the exact UTF-8 content, so retrieval can
verify the recovered bytes rather than trusting a mutable message position or
an LLM-generated description.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

_RECOVERY_REF_RE = re.compile(r"^(?:ref:)?(sha256:[0-9a-f]{64})$", re.IGNORECASE)
_TEST_COMMAND_RE = re.compile(
    r"(?:^|[\s;&|])(?:pytest|py\.test|jest|vitest|mocha|cargo\s+test|go\s+test|"
    r"npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|yarn\s+test)(?:\s|$)",
    re.IGNORECASE,
)
_TEST_OUTPUT_RE = re.compile(
    r"(?:\b\d+\s+(?:passed|failed|skipped|tests?)\b|tests?\s+(?:passed|failed)|"
    r"FAILURES|Test Suites:)",
    re.IGNORECASE,
)
_TRACEBACK_RE = re.compile(
    r"(?:Traceback \(most recent call last\):|\b[A-Za-z_][\w.]*Error:\s|"
    r"\bException:\s|panic(?:ked)? at)",
)


def make_recovery_ref(content: str) -> str:
    """Return a stable content address for exact string *content*."""
    if not isinstance(content, str):
        raise TypeError("recovery references require string content")
    # SessionDB scrubs lone UTF-16 surrogates to U+FFFD before persistence.
    # Hash that durable canonical form so a pre-persistence stub can still
    # recover the exact archived row.
    from agent.message_sanitization import _sanitize_surrogates

    durable_content = _sanitize_surrogates(content)
    digest = hashlib.sha256(durable_content.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def parse_recovery_ref(value: str) -> str | None:
    """Normalize ``ref:sha256:…`` or ``sha256:…``; reject partial digests."""
    if not isinstance(value, str):
        return None
    match = _RECOVERY_REF_RE.fullmatch(value.strip())
    return match.group(1).lower() if match else None


def _arguments(tool_args: str | Mapping[str, Any] | None) -> Mapping[str, Any]:
    if isinstance(tool_args, Mapping):
        return tool_args
    if isinstance(tool_args, str) and tool_args:
        try:
            parsed = json.loads(tool_args)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def classify_tool_segment(
    tool_name: str | None,
    tool_args: str | Mapping[str, Any] | None,
    content: str,
) -> str:
    """Classify a tool result into a small coding-agent segment taxonomy."""
    name = (tool_name or "").strip().lower()
    text = content if isinstance(content, str) else ""
    args = _arguments(tool_args)

    # Content wins over the launcher: a terminal command may emit a traceback
    # or test report, both more useful retrieval labels than ``shell_output``.
    if _TRACEBACK_RE.search(text):
        return "traceback"
    command = str(args.get("command") or "")
    if _TEST_COMMAND_RE.search(command) or _TEST_OUTPUT_RE.search(text):
        return "test_log"
    if name in {"read_file", "skill_view"}:
        return "file_read"
    if name in {"terminal", "process"}:
        return "shell_output"
    if name in {"search_files", "web_search"}:
        return "search_output"
    if name in {"web_extract", "browser_exec"} or name.startswith("browser_"):
        return "web_content"
    if name in {"patch", "write_file"}:
        return "edit_result"
    if name == "clarify":
        return "user_response"
    return "tool_output"


def format_segment_stub(
    *,
    segment_type: str,
    recovery_ref: str,
    summary: str,
    session_id: str = "",
) -> str:
    """Render a compact typed stub with an exact, fail-closed recovery call."""
    prefix = f"[SEGMENT:{segment_type}][REF:{recovery_ref}]"
    hint = ""
    if session_id:
        hint = (
            " Recover exact original with "
            "session_search("
            f"query={json.dumps(f'ref:{recovery_ref}')}, "
            f"session_id={json.dumps(session_id)})."
        )
    return f"{prefix} {summary.strip()}{hint}".strip()


def recover_message_by_ref(db: Any, session_id: str, recovery_ref: str) -> dict[str, Any] | None:
    """Return the exact persisted tool message whose content matches *ref*.

    Active and compaction-archived rows are included. Undo/rewind rows are
    deliberately excluded: those are ``active=0, compacted=0`` content the
    user explicitly removed, not a recovery source. The digest is recomputed
    from decoded DB content; a ref never resolves merely because metadata
    claims it.
    """
    normalized = parse_recovery_ref(recovery_ref)
    if not normalized or not session_id:
        return None
    for message in reversed(db.get_messages(session_id, include_compacted=True)):
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if make_recovery_ref(content) == normalized:
            return message
    return None
