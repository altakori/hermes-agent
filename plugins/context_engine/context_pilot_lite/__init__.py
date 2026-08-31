"""Experimental, opt-in ContextPilot-lite shadow context engine.

Shadow mode is deliberately a no-op at the prompt boundary: it records proposed
operations and metrics while ``select_context`` returns ``None``.  Active mode
only removes old, standalone tool-result messages and never mutates persisted
transcripts.  This module is intentionally small and wraps the built-in
:class:`ContextCompressor` for compression behavior.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

from agent.context_compressor import ContextCompressor
from agent.context_engine import ContextEngine
from agent.redact import redact_sensitive_text

_SECRET = re.compile(r"(?i)(api[_ -]?key|token|password|secret|private[_ -]?key)\s*[:=]")
MAX_NOTES = 32
MAX_NOTE_CHARS = 800

CONTEXT_MANAGE_SCHEMA = {
    "name": "context_manage",
    "description": "Experimental ContextPilot-lite context management. Shadow mode records proposals without changing the prompt.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["note", "archive", "compress", "exclude", "recall"]},
            "segment_ids": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def _ref(session_id: str, content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
    return f"contextpilot://{session_id}/{digest}"


class ContextPilotLite(ContextEngine):
    """A conservative ContextEngine wrapper; never deletes transcript data."""

    def __init__(self, *, mode: str = "shadow", session_id: str = "default", compressor: Any = None, **kwargs: Any):
        self.mode = mode if mode in {"shadow", "active"} else "shadow"
        self.session_id = str(session_id or "default")
        self._compressor = compressor or ContextCompressor(model=kwargs.get("model", "contextpilot-lite"), quiet_mode=True)
        self._notes: Dict[str, str] = {}
        self._archived: Dict[str, str] = {}
        self._archive_payload: Dict[str, str] = {}
        self._archive_verified: set[str] = set()
        self._proposals: List[Dict[str, Any]] = []
        self._force_compress_next = False
        self.metrics: Dict[str, int] = {"turns": 0, "proposed": 0, "excluded": 0, "archived": 0, "recalled": 0, "bytes_saved": 0}

    @property
    def name(self) -> str:
        return "context_pilot_lite"

    @property
    def context_length(self): return self._compressor.context_length
    @context_length.setter
    def context_length(self, value): self._compressor.context_length = value
    @property
    def threshold_tokens(self): return self._compressor.threshold_tokens
    @threshold_tokens.setter
    def threshold_tokens(self, value): self._compressor.threshold_tokens = value
    @property
    def last_prompt_tokens(self): return self._compressor.last_prompt_tokens
    @last_prompt_tokens.setter
    def last_prompt_tokens(self, value): self._compressor.last_prompt_tokens = value
    @property
    def last_completion_tokens(self): return self._compressor.last_completion_tokens
    @last_completion_tokens.setter
    def last_completion_tokens(self, value): self._compressor.last_completion_tokens = value
    @property
    def last_total_tokens(self): return self._compressor.last_total_tokens
    @last_total_tokens.setter
    def last_total_tokens(self, value): self._compressor.last_total_tokens = value
    @property
    def compression_count(self): return self._compressor.compression_count
    @compression_count.setter
    def compression_count(self, value): self._compressor.compression_count = value

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self._compressor.update_from_response(usage)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        if self.mode == "active" and self._force_compress_next:
            return True
        return self._compressor.should_compress(prompt_tokens)

    def should_compress_info(self, prompt_tokens: int = None):
        return self._compressor.should_compress_info(prompt_tokens)

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context=""):
        try:
            return self._compressor.compress(
                messages, current_tokens=current_tokens, focus_topic=focus_topic,
                force=force or self._force_compress_next, memory_context=memory_context,
            )
        finally:
            self._force_compress_next = False

    def prune_tool_results_only(self, messages, current_tokens=None):
        return self._compressor.prune_tool_results_only(messages, current_tokens)

    def should_compress_preflight(self, messages):
        return self._compressor.should_compress_preflight(messages)

    def should_defer_preflight_to_real_usage(self, rough_tokens):
        return self._compressor.should_defer_preflight_to_real_usage(rough_tokens)

    def has_content_to_compress(self, messages):
        return self._compressor.has_content_to_compress(messages)

    def get_status(self):
        status = self._compressor.get_status()
        status.update({"engine": self.name, "mode": self.mode, "shadow": self.mode == "shadow", "metrics": dict(self.metrics)})
        return status

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [deepcopy(CONTEXT_MANAGE_SCHEMA)]

    def on_session_start(self, session_id: str, **kwargs):
        self.session_id = str(session_id or self.session_id)
        self._compressor.on_session_start(session_id, **kwargs)

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self._compressor.on_session_end(session_id, messages)

    def on_session_reset(self):
        self._compressor.on_session_reset()
        self._notes.clear(); self._archived.clear(); self._archive_payload.clear(); self._archive_verified.clear(); self._proposals.clear(); self._force_compress_next = False; self.metrics = {k: 0 for k in self.metrics}

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        self._compressor.update_model(
            model=model,
            context_length=context_length,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            api_mode=api_mode,
        )

    def on_turn_complete(self, messages, usage=None, **kwargs):
        self.metrics["turns"] += 1
        # Observation only; do not retain transcript content in shadow state.

    def select_context(self, request_messages, *, conversation_messages=None, incoming_message=None, budget_tokens=0):
        """Return None in shadow mode; active mode injects only bounded notes.

        Archive/exclude requests remain proposals until a typed, structurally
        verified segment-selection contract can prove that no tool pair or
        protected turn would be orphaned. The raw transcript is never mutated.
        """
        if self.mode != "active":
            return None
        messages = deepcopy(request_messages)
        if self._notes:
            note_block = "\n\n[ContextPilot notes]\n" + "\n".join(
                f"- {value}" for value in list(self._notes.values())[-MAX_NOTES:]
            )
            for message in reversed(messages):
                if message.get("role") == "user" and isinstance(message.get("content"), str):
                    message["content"] += note_block[: 4 * MAX_NOTE_CHARS]
                    return messages
        return None

    def _archive_segment(self, sid: str, content: Any) -> str:
        text = str(content or "")
        safe = redact_sensitive_text(text, force=True, redact_url_credentials=True)
        verified = safe == text and not _SECRET.search(text)
        if not verified:
            safe = "[redacted sensitive tool output]"
        ref = _ref(self.session_id, sid + "\0" + safe)
        self._archived[sid] = ref
        self._archive_payload[sid] = safe
        if verified:
            self._archive_verified.add(sid)
        self.metrics["archived"] += 1
        self.metrics["bytes_saved"] += max(0, len(text) - len(ref))
        return ref

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        if name != "context_manage":
            return json.dumps({"error": "Unknown context engine tool", "ok": False}, sort_keys=True)
        action = args.get("action")
        if action == "note":
            content = str(args.get("content", "")).strip()
            safe = redact_sensitive_text(content, force=True, redact_url_credentials=True)
            if not content or _SECRET.search(content):
                return json.dumps({"ok": False, "error": "note rejected: empty or sensitive content"}, sort_keys=True)
            safe = safe[:MAX_NOTE_CHARS]
            key = hashlib.sha256(safe.encode()).hexdigest()[:24]
            if len(self._notes) >= MAX_NOTES and key not in self._notes:
                del self._notes[next(iter(self._notes))]
            self._notes[key] = safe
            ref = _ref(self.session_id, safe)
            return json.dumps({"ok": True, "ref": ref, "bounded": len(safe) < len(content)}, sort_keys=True)
        if action in {"archive", "exclude"}:
            ids = [str(x) for x in args.get("segment_ids", [])]
            self.metrics["proposed"] += 1
            proposal = {"action": action, "segment_ids": ids, "reason": str(args.get("reason", ""))[:300]}
            self._proposals.append(proposal)
            return json.dumps({"ok": True, "mode": self.mode, "proposal": proposal}, sort_keys=True)
        if action == "compress":
            if self.mode == "active":
                self._force_compress_next = True
            return json.dumps({"ok": True, "mode": self.mode, "delegated": "ContextCompressor"}, sort_keys=True)
        if action == "recall":
            ref = str(args.get("content", "") or (args.get("segment_ids") or [""])[0])
            prefix = f"contextpilot://{self.session_id}/"
            if not ref.startswith(prefix):
                return json.dumps({"ok": False, "error": "unverified recovery ref"}, sort_keys=True)
            for sid, candidate in self._archived.items():
                if candidate != ref:
                    continue
                if sid not in self._archive_verified:
                    return json.dumps({"ok": False, "error": "recovery is not exact/verified for this redacted archive"}, sort_keys=True)
                self.metrics["recalled"] += 1
                return json.dumps({"ok": True, "ref": ref, "segment_id": sid, "content": self._archive_payload.get(sid, "")}, sort_keys=True)
            return json.dumps({"ok": False, "error": "unverified recovery ref"}, sort_keys=True)
        return json.dumps({"ok": False, "error": "invalid action"}, sort_keys=True)


def register(ctx):
    """Explicit loader entry point; activation occurs only via context.engine."""
    config = getattr(ctx, "config", {})
    if not isinstance(config, dict):
        config = {}
    ctx.register_context_engine(
        ContextPilotLite(
            mode=str(config.get("mode", "shadow")),
            session_id=str(config.get("session_id", "default")),
        )
    )


__all__ = ["ContextPilotLite", "CONTEXT_MANAGE_SCHEMA", "register"]
