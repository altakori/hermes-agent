"""Persistent, opt-in safety state and high-confidence argument egress guard.

Only salted/keyed digests and bounded metadata are written.  This plugin is
intentionally conservative: it blocks only high-confidence egress to MCP or
network-capable tools; ambiguous calls are allowed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MAX_META = 512
_MAX_FINGERPRINTS = 4096
_MAX_NEW_FINGERPRINTS_PER_TURN = 512
_MAX_TEXT = 2 * 1024 * 1024
_MAX_MCP_ARG_TEXT = 256 * 1024
_SECRET_RE = re.compile(r"(?i)(?:sk-[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9_]{16,}|xox[baprs]-[A-Za-z0-9-]{16,}|Bearer\s+[A-Za-z0-9._~-]{16,}|(?:api[_-]?key|token|password|secret|private[_-]?key)[\"']?\s*[:=]\s*[\"']?[^\s,\"'}]{8,})")
_PATH_RE = re.compile(r"(?i)(?:[A-Z]:\\|/(?:home|Users|root|etc|private|var)|\\Users\\|\.ssh/|\.env(?:\b|$))")
_INVENTORY_RE = re.compile(r"(?i)(?:tool[_ -]?inventory|available[_ -]?tools|function[_ -]?definitions|tool[_ -]?schemas)")


class SafetyStateIntegrityError(RuntimeError):
    """Persisted enforcement state failed integrity verification."""


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return repr(value)


def _digest(value: Any, salt: bytes) -> str:
    return hashlib.sha256(salt + _json(value).encode("utf-8", "replace")).hexdigest()


def _strings(value: Any):
    """Yield every string leaf from a bounded argument object.

    Untrusted MCP arguments are size-gated before this traversal.  The iterative
    walk avoids both recursion-depth failures and a fixed per-container prefix
    that an attacker could bypass by placing data in a later element.
    """
    stack = [value]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            yield current[:_MAX_TEXT]
            continue
        if isinstance(current, dict):
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)
            for key, item in reversed(list(current.items())):
                stack.append(item)
                stack.append(key)
            continue
        if isinstance(current, (list, tuple)):
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)
            stack.extend(reversed(current))

def _fingerprints(text: str, salt: bytes, *, source: bool = False, limit: int | None = None):
    """Yield salted windows without retaining protected plaintext.

    Candidate arguments are scanned at every offset. Source text is sampled
    uniformly when it has more windows than its fair-share persistent budget,
    so later transcript regions remain covered instead of only the prefix.
    """
    width = 96
    if len(text) < width:
        return
    stop = len(text) - width + 1
    effective_limit = (_MAX_FINGERPRINTS if source else stop) if limit is None else limit
    if effective_limit <= 0:
        return
    step = max(1, math.ceil(stop / effective_limit)) if source else 1
    emitted = 0
    for i in range(0, stop, step):
        if emitted >= effective_limit:
            break
        yield _digest(text[i:i + width], salt)
        emitted += 1


def _active_authorization(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict): return {}
    expiry = value.get("expiry", "")
    if expiry:
        try:
            if datetime.fromisoformat(str(expiry).replace("Z", "+00:00")) <= datetime.now(timezone.utc):
                return {}
        except ValueError:
            return {}
    return _bounded(value)


def _bounded(value: Any) -> Any:
    if isinstance(value, str):
        return value[:_MAX_META]
    if isinstance(value, dict):
        return {str(k)[:80]: _bounded(v) for k, v in list(value.items())[:16]}
    if isinstance(value, (list, tuple)):
        return [_bounded(v) for v in list(value)[:16]]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return type(value).__name__


class SafetyLedger:
    """Append-only hash-chain ledger plus non-clearable active state."""
    def __init__(self, data_dir: Path, max_bytes: int = 10 * 1024 * 1024):
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "events.jsonl"
        self.state_path = self.data_dir / "active_state.json"
        self.state_mac_path = self.data_dir / "active_state.mac"
        self.max_bytes = max(1024, int(max_bytes))
        self._lock = threading.RLock()
        self._salt_path = self.data_dir / "ledger.salt"

    @contextmanager
    def _state_guard(self, path: Path):
        """Serialize one ledger/state file across threads and processes."""
        from hermes_cli.plugins import _locked_plugin_state

        with self._lock:
            with _locked_plugin_state(path):
                yield

    def _salt(self) -> bytes:
        with self._state_guard(self._salt_path):
            self.data_dir.mkdir(parents=True, exist_ok=True)
            try:
                return self._salt_path.read_bytes()
            except FileNotFoundError:
                salt = secrets.token_bytes(32)
                tmp = self._salt_path.with_suffix(".tmp")
                tmp.write_bytes(salt)
                os.replace(tmp, self._salt_path)
                return salt

    def _active(self, root_session_id: str = "") -> dict[str, Any]:
        obj = self._read_all_active()
        # State is partitioned by root scope; legacy flat state is ignored
        # rather than leaked into a different conversation.
        scope = obj.get("scopes", {}).get(root_session_id or "__global__", {})
        return scope if isinstance(scope, dict) else {}

    def _write_all_active(self, current: dict[str, Any]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(current, sort_keys=True, ensure_ascii=False).encode("utf-8")
        signature = hmac.new(self._salt(), payload, hashlib.sha256).hexdigest().encode("ascii")
        state_tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        mac_tmp = self.state_mac_path.with_name(self.state_mac_path.name + ".tmp")
        for path, data in ((state_tmp, payload), (mac_tmp, signature)):
            with path.open("wb") as fh:
                fh.write(data); fh.flush(); os.fsync(fh.fileno())
        os.replace(state_tmp, self.state_path)
        os.replace(mac_tmp, self.state_mac_path)

    def _save_active(self, state: dict[str, Any], root_session_id: str = "") -> None:
        current = self._read_all_active()
        current.setdefault("scopes", {})[root_session_id or "__global__"] = state
        self._write_all_active(current)

    def _read_all_active(self) -> dict[str, Any]:
        state_exists = self.state_path.exists()
        mac_exists = self.state_mac_path.exists()
        if not state_exists:
            if mac_exists or self.path.exists():
                raise SafetyStateIntegrityError("persistent safety state missing")
            return {}
        if not mac_exists:
            raise SafetyStateIntegrityError("persistent safety state signature missing")
        try:
            payload = self.state_path.read_bytes()
            actual = self.state_mac_path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as exc:
            raise SafetyStateIntegrityError("persistent safety state could not be verified") from exc
        expected = hmac.new(self._salt(), payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(actual, expected):
            raise SafetyStateIntegrityError("persistent safety state integrity check failed")
        try:
            obj = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise SafetyStateIntegrityError("persistent safety state is malformed") from exc
        if not isinstance(obj, dict):
            raise SafetyStateIntegrityError("persistent safety state root is invalid")
        return obj

    def bind_session(self, session_id: str, root_session_id: str) -> None:
        if not session_id or not root_session_id:
            return
        with self._state_guard(self.state_path):
            current = self._read_all_active()
            current.setdefault("session_roots", {})[str(session_id)[:128]] = str(root_session_id)[:128]
            self._write_all_active(current)

    def root_for(self, session_id: str) -> str:
        with self._state_guard(self.state_path):
            current = self._read_all_active()
            return str(current.get("session_roots", {}).get(session_id, ""))

    def set_fingerprints(self, session_id: str, fingerprints: list[str], *, root_session_id: str = "") -> None:
        root = root_session_id or self.root_for(session_id) or session_id
        bounded = [str(v) for v in fingerprints[:_MAX_FINGERPRINTS] if isinstance(v, str)]
        with self._state_guard(self.state_path):
            s = self._active(root)
            s["protected_fingerprints"] = bounded
            self._save_active(s, root)
        self.bind_session(session_id, root)

    def _ensure_state_initialized(self) -> None:
        with self._state_guard(self.state_path):
            current = self._read_all_active()
            if not self.state_path.exists():
                current.setdefault("scopes", {})
                current.setdefault("session_roots", {})
                self._write_all_active(current)

    def append(self, *, scope: str, root_session_id: str, session_id: str,
               event_type: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self._ensure_state_initialized()
        with self._state_guard(self.path):
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
                raise RuntimeError("persistent-safety-guard ledger quota exceeded")
            previous = "0" * 64
            seq = 1
            if self.path.exists():
                lines = self.path.read_text(encoding="utf-8").splitlines()
                if lines:
                    last = json.loads(lines[-1])
                    previous, seq = last.get("event_hash", previous), int(last.get("seq", 0)) + 1
            record = {"seq": seq, "timestamp": datetime.now(timezone.utc).isoformat(),
                      "scope": str(scope)[:128], "root_session_id": str(root_session_id)[:128],
                      "session_id": str(session_id)[:128], "event_type": str(event_type)[:96],
                      "metadata": _bounded(metadata or {}), "prev_hash": previous}
            record["event_hash"] = hmac.new(
                self._salt(), (previous + _json(record)).encode(), hashlib.sha256
            ).hexdigest()
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(_json(record) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            return record

    def verify(self) -> bool:
        try:
            with self._state_guard(self.state_path):
                self._read_all_active()
        except (OSError, SafetyStateIntegrityError):
            return False
        with self._state_guard(self.path):
            previous = "0" * 64; expected_seq = 1
            try:
                lines = self.path.read_text(encoding="utf-8").splitlines()
                for line in lines:
                    r = json.loads(line); supplied = r.pop("event_hash")
                    if r.get("seq") != expected_seq or r.get("prev_hash") != previous:
                        return False
                    actual = hmac.new(
                        self._salt(), (previous + _json(r)).encode(), hashlib.sha256
                    ).hexdigest()
                    if not hmac.compare_digest(actual, supplied): return False
                    previous, expected_seq = supplied, expected_seq + 1
                return True
            except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
                return False

    def add_flag(self, session_id: str, flag: str, *, root_session_id: str = "") -> None:
        with self._state_guard(self.state_path):
            s = self._active(root_session_id or session_id); flags = set(s.get("risk_flags", [])); flags.add(str(flag)[:128])
            s["risk_flags"] = sorted(flags); self._save_active(s, root_session_id or session_id)
        self.append(scope="agent", root_session_id=root_session_id or session_id, session_id=session_id,
                    event_type="risk_flag_added", metadata={"rule": flag})

    def deny_intent(self, intent: Any, session_id: str, *, root_session_id: str = "") -> str:
        digest = _digest(intent, self._salt())
        with self._state_guard(self.state_path):
            s = self._active(root_session_id or session_id); values = set(s.get("denied_intent_hashes", [])); values.add(digest)
            s["denied_intent_hashes"] = sorted(values)[-_MAX_FINGERPRINTS:]; self._save_active(s, root_session_id or session_id)
        self.append(scope="agent", root_session_id=root_session_id or session_id, session_id=session_id,
                    event_type="intent_denied", metadata={"intent_hash": digest})
        return digest

    def is_denied(self, intent: Any, session_id: str, *, root_session_id: str = "") -> bool:
        root = root_session_id or self.root_for(session_id) or session_id
        digest = _digest(intent, self._salt())
        return digest in set(self.state(root).get("denied_intent_hashes", []))

    def set_authorization(self, metadata: dict[str, Any], session_id: str, *, root_session_id: str = "") -> None:
        scope = str(metadata.get("scope", ""))
        provenance = str(metadata.get("provenance", ""))
        safe = {
            "scope_hash": _digest(scope, self._salt()) if scope else "",
            "provenance_hash": _digest(provenance, self._salt()) if provenance else "",
            "expiry": str(metadata.get("expiry", ""))[:64],
        }
        with self._state_guard(self.state_path):
            s = self._active(root_session_id or session_id); s["authorization"] = safe; self._save_active(s, root_session_id or session_id)
        self.append(scope="agent", root_session_id=root_session_id or session_id, session_id=session_id,
                    event_type="authorization_set", metadata=safe)

    def set_budget(self, amount: int, session_id: str, *, root_session_id: str = "") -> None:
        root = root_session_id or session_id
        with self._state_guard(self.state_path):
            s = self._active(root); s["irreversible_action_budget"] = max(0, min(int(amount), 1000)); self._save_active(s, root)
        self.append(scope="agent", root_session_id=root, session_id=session_id, event_type="budget_set", metadata={"remaining": s["irreversible_action_budget"]})

    def consume_budget(self, session_id: str, *, root_session_id: str = "") -> bool:
        root = root_session_id or session_id
        with self._state_guard(self.state_path):
            s = self._active(root); n = int(s.get("irreversible_action_budget", 0))
            if n <= 0: return False
            s["irreversible_action_budget"] = n - 1; self._save_active(s, root)
        self.append(scope="agent", root_session_id=root, session_id=session_id, event_type="budget_consumed", metadata={"remaining": n - 1})
        return True

    def clear_explicit(self, session_id: str = "", *, root_session_id: str = "") -> None:
        """Clear only through this explicit user/plugin API, never model data."""
        with self._state_guard(self.state_path):
            root = root_session_id or session_id
            s = self._active(root)
            s["risk_flags"] = []
            s["denied_intent_hashes"] = []
            s["protected_fingerprints"] = []
            s["authorization"] = {}
            s["irreversible_action_budget"] = 0
            self._save_active(s, root)
        self.append(scope="agent", root_session_id=root_session_id or session_id, session_id=session_id,
                    event_type="explicit_clear", metadata={"provenance": "slash_or_user"})

    def state(self, root_session_id: str = "") -> dict[str, Any]:
        with self._state_guard(self.state_path):
            return dict(self._active(root_session_id))


class Guard:
    def __init__(self, data_dir: Path, max_bytes: int = 10 * 1024 * 1024):
        self.ledger = SafetyLedger(data_dir, max_bytes); self.sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def _session(self, sid: str, root: str = "") -> dict[str, Any]:
        with self._lock:
            resolved_root = root or self.ledger.root_for(sid) or sid
            persisted = self.ledger.state(resolved_root)
            x = self.sessions.setdefault(
                sid,
                {
                    "root": resolved_root,
                    "fingerprints": set(persisted.get("protected_fingerprints", [])),
                },
            )
            if root:
                x["root"] = root
            self.ledger.bind_session(sid, x["root"])
            return x

    def pre_llm(self, session_id: str, user_message: Any = "", conversation_history: Any = None, **_: Any) -> dict[str, str]:
        s = self._session(session_id)
        texts = [user_message]
        if isinstance(conversation_history, list):
            texts += list(reversed(conversation_history))
        sources: list[str] = []
        for text in texts:
            for raw in _strings(text):
                if len(raw) >= 96:
                    sources.append(raw)
        if len(sources) > _MAX_NEW_FINGERPRINTS_PER_TURN:
            stride = math.ceil(len(sources) / _MAX_NEW_FINGERPRINTS_PER_TURN)
            sources = sources[::stride][:_MAX_NEW_FINGERPRINTS_PER_TURN]
        per_source = max(1, _MAX_NEW_FINGERPRINTS_PER_TURN // max(1, len(sources)))
        newest: list[str] = []
        salt = self.ledger._salt()
        for raw in sources:
            newest.extend(_fingerprints(raw, salt, source=True, limit=per_source))
        existing = list(s["fingerprints"])
        additions = [v for v in dict.fromkeys(newest) if v not in s["fingerprints"]]
        remaining = max(0, _MAX_FINGERPRINTS - len(existing))
        if len(additions) > remaining:
            self.ledger.add_flag(s["root"], "fingerprint_capacity_exhausted")
        merged = existing + additions[:remaining]
        s["fingerprints"] = set(merged)
        self.ledger.set_fingerprints(session_id, merged, root_session_id=s["root"])
        state = self.ledger.state(s.get("root", session_id))
        capsule = {"risk_flags": state.get("risk_flags", []),
                   "authorization": _active_authorization(state.get("authorization", {})),
                   "irreversible_action_budget": state.get("irreversible_action_budget", 0)}
        self.ledger.append(scope="agent", root_session_id=s["root"], session_id=session_id,
                           event_type="pre_llm", metadata={"fingerprint_count": len(s["fingerprints"]), "capsule": capsule})
        return {"context": "SAFETY CAPSULE (exact): " + _json(capsule)}

    @staticmethod
    def _external(tool_name: str, args: Any) -> tuple[bool, bool]:
        name = str(tool_name).lower()
        mcp = name.startswith("mcp__") or name.startswith("mcp_")
        # Built-in web/browser/platform tools are not in this gate unless an
        # operator explicitly configures them as MCP sinks. This avoids turning
        # ordinary user queries into false positives.
        if not mcp: return False, False
        server = ""
        trusted = False
        try:
            from tools import mcp_tool
            server = getattr(mcp_tool, "_mcp_tool_server_names", {}).get(
                str(tool_name), ""
            )
            if not server and name.startswith("mcp__"):
                # Compatibility fallback for tests and pre-registration calls.
                server = name[len("mcp__"):].split("__", 1)[0].replace("_", "-")
            trusted = getattr(mcp_tool, "_server_trust_levels", {}).get(server) == "full"
        except Exception: pass
        return True, trusted

    def pre_tool(self, tool_name: str = "", args: Any = None, session_id: str = "", **_: Any) -> dict[str, str] | None:
        external, trusted = self._external(tool_name, args)
        if not external: return None
        text = _json(args); s = self._session(session_id)
        intent = {"tool": str(tool_name), "args": args}
        if self.ledger.is_denied(intent, session_id, root_session_id=s["root"]):
            return {
                "action": "block",
                "message": "persistent-safety-guard blocked a previously denied external tool intent",
            }
        state = self.ledger.state(s["root"])
        capacity_exhausted = "fingerprint_capacity_exhausted" in state.get("risk_flags", [])
        oversized_untrusted = not trusted and len(text) > _MAX_MCP_ARG_TEXT
        explicit_secret = bool(_SECRET_RE.search(text))
        inventory = bool(_INVENTORY_RE.search(text)) and ("tool" in text.lower() or "schema" in text.lower())
        matching = 0
        if not trusted:
            for raw in _strings(args):
                if any(
                    fp in s["fingerprints"]
                    for fp in _fingerprints(raw, self.ledger._salt())
                ):
                    matching += 1
        exact_context = matching >= 1
        whole_transcript = matching >= 2
        path_exfil = bool(_PATH_RE.search(text))
        # A server explicitly trusted by the operator may receive ordinary
        # local material; secrets remain blocked regardless of trust.
        high = capacity_exhausted and not trusted or oversized_untrusted or explicit_secret or (not trusted and (inventory or exact_context or whole_transcript or path_exfil))
        if not high: return None
        rule = "fingerprint_capacity_exhausted" if capacity_exhausted and not trusted else "oversized_argument" if oversized_untrusted else "secret" if explicit_secret else "tool_inventory" if inventory else "protected_context" if exact_context or whole_transcript else "sensitive_path"
        self.ledger.append(scope="agent", root_session_id=s.get("root", session_id), session_id=session_id,
                           event_type="egress_blocked", metadata={"rule": rule, "tool": str(tool_name)[:128], "trusted": trusted})
        self.ledger.deny_intent(intent, session_id, root_session_id=s.get("root", session_id))
        return {"action": "block", "message": "persistent-safety-guard blocked high-confidence protected argument egress (rule: %s); no raw arguments retained" % rule}

    def child(self, child_session_id: str, parent_session_id: str = "", root_session_id: str = "", **_: Any) -> None:
        parent = self._session(parent_session_id)
        if root_session_id:
            parent["root"] = root_session_id
        self.sessions[child_session_id] = {"root": parent.get("root", root_session_id or child_session_id), "fingerprints": set(parent["fingerprints"])}
        self.ledger.bind_session(child_session_id, self.sessions[child_session_id]["root"])

    def session_started(
        self,
        session_id: str,
        old_session_id: str = "",
        parent_session_id: str = "",
        root_session_id: str = "",
        **_: Any,
    ) -> None:
        """Bind initial sessions and compression-created child sessions."""
        parent = old_session_id or parent_session_id
        if parent and parent != session_id:
            self.child(
                session_id,
                parent_session_id=parent,
                root_session_id=root_session_id,
            )
            return
        self._session(session_id, root_session_id or session_id)

    def clear_scope(self, root_session_id: str) -> None:
        self.ledger.clear_explicit(root_session_id, root_session_id=root_session_id)
        with self._lock:
            for session_id in [
                key for key, value in self.sessions.items()
                if value.get("root") == root_session_id
            ]:
                self.sessions.pop(session_id, None)


_guard: Guard | None = None

def _get(ctx=None) -> Guard:
    global _guard
    if _guard is None:
        data = getattr(getattr(ctx, "state", None), "data_dir", None)
        if data is None: data = Path.home() / ".hermes" / "plugin-data" / "persistent-safety-guard"
        configured = getattr(ctx, "config", {}) if ctx is not None else {}
        if not isinstance(configured, dict):
            configured = {}
        max_bytes = configured.get(
            "max_ledger_bytes",
            os.environ.get("PERSISTENT_SAFETY_GUARD_MAX_LEDGER_BYTES", 10 * 1024 * 1024),
        )
        _guard = Guard(Path(data), int(max_bytes))
    return _guard


def register(ctx) -> None:
    g = _get(ctx)
    ctx.register_hook("on_session_start", g.session_started)
    ctx.register_hook("pre_llm_call", g.pre_llm)
    def _fail_closed_pre_tool(**kwargs):
        try:
            return g.pre_tool(**kwargs)
        except Exception:
            tool_name = str(kwargs.get("tool_name", ""))
            # The guard owns only MCP egress. Internal tools stay available if
            # the safety ledger is unavailable; external egress fails closed.
            if tool_name.lower().startswith(("mcp__", "mcp_")):
                return {
                    "action": "block",
                    "message": "persistent-safety-guard is unavailable; external tool egress failed closed",
                }
            return None

    _fail_closed_pre_tool.__hermes_final_mcp_egress__ = True
    ctx.register_hook("pre_tool_call", _fail_closed_pre_tool)
    ctx.register_hook("subagent_start", g.child)

    def _clear(args):
        root = str(args or "").strip()
        if not root:
            return "Usage: /safety-clear <root-session-id>"
        g.clear_scope(root)
        return f"Safety state explicitly cleared for root session {root}"

    ctx.register_command(
        "safety-clear",
        _clear,
        description="Explicitly clear safety flags for one root session",
    )

# Small deterministic API for tests and trusted internal callers.
def clear_safety_state(session_id: str = "", root_session_id: str = "") -> None:
    root = root_session_id or session_id
    _get().clear_scope(root)

def add_risk_flag(flag: str, session_id: str = "", root_session_id: str = "") -> None:
    _get().ledger.add_flag(session_id, flag, root_session_id=root_session_id)

def deny_intent(intent: Any, session_id: str = "", root_session_id: str = "") -> str:
    return _get().ledger.deny_intent(intent, session_id, root_session_id=root_session_id)

def verify_ledger() -> bool:
    return _get().ledger.verify()
