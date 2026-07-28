"""Projects cursor-sdk stream messages into Hermes' messages list.

The translator that lets Hermes' memory/skill review, session persistence,
and background review keep working under the Cursor runtime: it converts the
SDK's typed ``SDKMessage`` stream (see cursor.com/docs/sdk/python) into the
standard OpenAI-shaped ``{role, content, tool_calls, tool_call_id}`` entries
the rest of Hermes already knows how to read.

Cursor emits messages with a discriminator field ``type``:
  - assistant   → {role: "assistant", content} (TextBlock content accumulated)
  - thinking    → stashed in the next assistant message's "reasoning" field
  - tool_call   → assistant tool_call(name=<mapped>) + tool result. Emitted
                  twice for most calls: status="running" with args, then
                  status="completed" (or "error") with result. We materialize
                  messages only on the terminal event, mirroring how Hermes
                  only writes assistant messages after streaming completes.
  - usage       → per-turn TokenUsage, tracked for accounting (no message)
  - system/user/status/task/request → display/accounting only (no messages)

Each tool call maps to exactly one assistant entry + one tool entry,
preserving Hermes' message-alternation invariants. Cursor's tool ``args`` /
``result`` payloads are explicitly documented as UNSTABLE ("treat as untyped
data and parse defensively") — only the envelope (type, call_id, name,
status) is stable, so everything below reads payloads best-effort.

The projector accepts both the SDK's frozen dataclasses and plain mappings
(the SDK yields ``Mapping[str, Any]`` for unknown message types, and tests
drive the projector with dicts).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Cursor-internal tool names → the Hermes display/projection names the rest
# of the stack (display emojis, background review, curator) already knows.
# Unknown names pass through unchanged — cursor renames tools between
# releases and we must not crash or hide activity when it does.
_CURSOR_TOOL_NAME_MAP = {
    "shell": "exec_command",
    "bash": "exec_command",
    "run_terminal_cmd": "exec_command",
    "terminal": "exec_command",
    "edit": "apply_patch",
    "write": "apply_patch",
    "str_replace": "apply_patch",
    "apply_patch": "apply_patch",
    "edit_file": "apply_patch",
    "create_file": "apply_patch",
    "delete_file": "apply_patch",
    "read": "read_file",
    "read_file": "read_file",
    "grep": "search_files",
    "glob": "search_files",
    "search": "search_files",
    "codebase_search": "search_files",
    "list_dir": "search_files",
    "ls": "search_files",
}


def map_cursor_tool_name(name: str) -> str:
    """Map a cursor-internal tool name onto Hermes' vocabulary.

    MCP tools arrive as ``mcp__server__tool`` or similar — keep them
    verbatim. Unknown names pass through so new cursor tools stay visible.
    """
    return _CURSOR_TOOL_NAME_MAP.get((name or "").strip().lower(), name or "unknown")


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from an SDK dataclass attribute or a mapping key."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _deterministic_call_id(name: str, call_id: str) -> str:
    """Stable id for tool_call message correlation.

    Uses cursor's call_id directly when present; falls back to a content
    hash so replay produces the same id across sessions and prefix caches
    stay valid (same rationale as the codex projector)."""
    if call_id:
        return f"cursor_{call_id}"
    digest = hashlib.sha256(f"cursor_{name}".encode()).hexdigest()[:16]
    return f"cursor_{name}_{digest}"


def _format_tool_args(args: Any) -> str:
    """Format tool args as JSON the way Hermes' tool_calls path does.

    Cursor's args payload is untyped — tolerate non-dict shapes."""
    if not isinstance(args, dict):
        args = {"arguments": args} if args is not None else {}
    try:
        return json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return json.dumps({"arguments": str(args)}, ensure_ascii=False)


def _format_tool_result(result: Any, error: bool = False) -> str:
    """Render a tool result payload as bounded text for the tool message."""
    if result is None:
        content = ""
    elif isinstance(result, str):
        content = result
    else:
        try:
            content = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            content = str(result)
    content = content[:4000]
    if error:
        return f"[error] {content}" if content else "[error]"
    return content


@dataclass
class CursorProjectionResult:
    """Output of projecting one cursor SDK stream message.

    ``messages`` is a list because a completed tool call produces two
    messages (assistant tool_call + tool result). Empty list = message
    ignored or display-only (e.g. a status event, or the "running" half of
    a tool call)."""

    messages: list[dict] = field(default_factory=list)
    is_tool_iteration: bool = False
    final_text: Optional[str] = None       # Set when an assistant message completes
    # Set on the "running" half of a tool call so the session can bridge it
    # to Hermes' tool-progress display without materializing messages yet:
    # (mapped_name, preview, args_dict)
    tool_started: Optional[tuple[str, str, dict]] = None
    # Set when a usage stream event arrives: raw TokenUsage-shaped object.
    usage: Any = None


class CursorEventProjector:
    """Stateful projector consuming cursor-sdk stream messages in order.

    Owns the in-progress reasoning content (cursor emits thinking as
    separate messages but Hermes stashes reasoning on the next assistant
    message) and the open tool-call table (running → completed pairing).
    """

    def __init__(self) -> None:
        self._pending_reasoning: list[str] = []
        # call_id → {"name": mapped name, "args": dict} for calls we've seen
        # start but not finish. finalize() closes leftovers.
        self._open_tool_calls: dict[str, dict] = {}
        self._assistant_texts: list[str] = []

    # ---------- per-message ----------

    def project(self, message: Any) -> CursorProjectionResult:
        """Project a single SDKMessage. Never raises on malformed payloads."""
        try:
            msg_type = str(_get(message, "type", "") or "")
            if msg_type == "assistant":
                return self._project_assistant(message)
            if msg_type == "thinking":
                text = str(_get(message, "text", "") or "")
                if text:
                    self._pending_reasoning.append(text)
                return CursorProjectionResult()
            if msg_type == "tool_call":
                return self._project_tool_call(message)
            if msg_type == "usage":
                return CursorProjectionResult(usage=_get(message, "usage"))
            # system / user / status / task / request / unknown → no messages.
            # (The user message is already in Hermes' list — run_conversation
            # appended it before dispatching the turn.)
            return CursorProjectionResult()
        except Exception:
            logger.debug("cursor projector: failed to project %r", message, exc_info=True)
            return CursorProjectionResult()

    def _project_assistant(self, message: Any) -> CursorProjectionResult:
        inner = _get(message, "message") or {}
        content = _get(inner, "content") or []
        parts: list[str] = []
        if isinstance(content, str):
            parts.append(content)
        else:
            for block in content:
                block_type = str(_get(block, "type", "") or "")
                if block_type == "text":
                    text = _get(block, "text")
                    if text:
                        parts.append(str(text))
                # ToolUseBlock values are surfaced separately via tool_call
                # stream messages — don't duplicate them here.
        text = "".join(parts)
        if not text:
            return CursorProjectionResult()
        self._assistant_texts.append(text)
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if self._pending_reasoning:
            msg["reasoning"] = "\n".join(self._pending_reasoning)
            self._pending_reasoning = []
        return CursorProjectionResult(messages=[msg], final_text=text)

    def _project_tool_call(self, message: Any) -> CursorProjectionResult:
        raw_name = str(_get(message, "name", "") or "unknown")
        mapped = map_cursor_tool_name(raw_name)
        call_id = _deterministic_call_id(mapped, str(_get(message, "call_id", "") or ""))
        status = str(_get(message, "status", "") or "").lower()

        args = _get(message, "args")
        if not isinstance(args, dict):
            args = {"arguments": args} if args is not None else {}

        if status == "running":
            self._open_tool_calls[call_id] = {"name": mapped, "args": args}
            preview = _tool_preview(mapped, args)
            return CursorProjectionResult(tool_started=(mapped, preview, args))

        # Terminal statuses ("completed", "error", anything else cursor may
        # add) — materialize the assistant tool_call + tool result pair.
        started = self._open_tool_calls.pop(call_id, None)
        if started is not None and not args:
            args = started["args"]
        is_error = status == "error"
        result = _get(message, "result")
        return self._materialize_tool_pair(mapped, call_id, args, result, is_error)

    def _materialize_tool_pair(
        self,
        name: str,
        call_id: str,
        args: dict,
        result: Any,
        is_error: bool,
    ) -> CursorProjectionResult:
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": _format_tool_args(args),
                    },
                }
            ],
        }
        if self._pending_reasoning:
            assistant_msg["reasoning"] = "\n".join(self._pending_reasoning)
            self._pending_reasoning = []
        tool_msg = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": _format_tool_result(result, error=is_error),
        }
        return CursorProjectionResult(
            messages=[assistant_msg, tool_msg], is_tool_iteration=True
        )

    # ---------- turn end ----------

    def finalize(self) -> CursorProjectionResult:
        """Close any tool calls that never reached a terminal status.

        A cancelled/errored run can strand "running" tool calls; leaving a
        dangling assistant tool_call without its tool result would break
        Hermes' alternation invariants, so close each with an error result.
        """
        messages: list[dict] = []
        tool_iterations = 0
        for call_id, info in list(self._open_tool_calls.items()):
            pair = self._materialize_tool_pair(
                info["name"], call_id, info["args"],
                "tool call did not complete (run ended)", True,
            )
            messages.extend(pair.messages)
            tool_iterations += 1
        self._open_tool_calls.clear()
        return CursorProjectionResult(
            messages=messages, is_tool_iteration=bool(tool_iterations)
        )

    @property
    def final_text(self) -> str:
        """Assistant text accumulated across the whole run."""
        return "\n\n".join(t for t in self._assistant_texts if t)


def _tool_preview(name: str, args: dict) -> str:
    """Short human preview for the tool-progress feed."""
    for key in ("command", "path", "file_path", "query", "pattern", "url"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:200]
    return name
