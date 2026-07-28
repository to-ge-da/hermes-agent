"""Hermes tools exposed to cursor turns via the SDK's custom_tools.

When ``provider: cursor`` drives a turn, the cursor agent owns the loop and
builds its own tool list (shell, read, edit, ...). By default that would make
Hermes' richer tool surface — web search, browser automation, vision, image
generation, skills, TTS, kanban handoff — unreachable for the duration of the
turn.

Where the codex app-server runtime bridges this gap with a stdio MCP sidecar
(``agent/transports/hermes_tools_mcp_server.py``), the cursor-sdk offers a
better primitive for LOCAL runs: ``LocalAgentOptions.custom_tools`` takes
plain Python callbacks that the SDK invokes in-process. No subprocess, no
extra serialization hop — each call dispatches straight through
``model_tools.handle_function_call()`` (the same code path as Hermes'
default runtime, including pre/post tool hooks and guardrails).

The exposed set is EXACTLY the codex runtime's ``EXPOSED_TOOLS`` — one
curated list, one policy. See that module's docstring for the reasoning on
what is deliberately NOT exposed (terminal/file ops are cursor's own;
``_AGENT_LOOP_TOOLS`` like delegate_task/memory need mid-loop AIAgent state
a stateless callback can't reach).

Cloud runs cannot use custom_tools (SDK limitation) — cloud agents get
capability via MCP passthrough instead (``cursor.inherit_mcp``).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

# Single source of truth for the exposed-tool policy — shared with the codex
# runtime's MCP sidecar so the two external runtimes never drift.
from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

logger = logging.getLogger(__name__)

# Bound each bridged tool result the same way plugin tools are bounded —
# cursor feeds the result back into its own context window.
_MAX_RESULT_CHARS = 100_000


def build_cursor_custom_tools(
    task_id: Optional[str] = None,
    *,
    session_id: Optional[str] = None,
    on_tool_event: Optional[Callable[[str, str, dict], None]] = None,
    enabled_toolsets: Optional[list[str]] = None,
    disabled_toolsets: Optional[list[str]] = None,
) -> dict[str, dict]:
    """Build the ``custom_tools`` mapping for ``LocalAgentOptions``.

    Returns ``{tool_name: {"description": ..., "input_schema": ...,
    "execute": callable}}`` — the mapping form the SDK accepts alongside its
    ``CustomTool`` dataclass, which keeps this module import-safe without
    cursor-sdk installed.

    Only tools that are actually registered in this Hermes process are
    included (missing API keys / disabled toolsets drop out naturally, same
    as the MCP sidecar).
    """
    from model_tools import get_tool_definitions, handle_function_call

    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (
            get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                quiet_mode=True,
            )
            or []
        )
        if isinstance(td, dict) and td.get("type") == "function"
    }

    tools: dict[str, dict] = {}
    for name in EXPOSED_TOOLS:
        spec = all_defs.get(name)
        if spec is None:
            logger.debug(
                "cursor custom_tools: skipping %s — not registered in this process",
                name,
            )
            continue

        description = spec.get("description") or f"Hermes {name} tool"
        params_schema = spec.get("parameters") or {"type": "object", "properties": {}}

        tools[name] = {
            "description": description,
            "input_schema": params_schema,
            "execute": _make_executor(
                name,
                handle_function_call,
                task_id=task_id,
                session_id=session_id,
                on_tool_event=on_tool_event,
            ),
        }

    logger.info(
        "cursor custom_tools: exposing %d/%d Hermes tools", len(tools), len(EXPOSED_TOOLS)
    )
    return tools


def _make_executor(
    tool_name: str,
    handle_function_call: Callable[..., str],
    *,
    task_id: Optional[str],
    session_id: Optional[str],
    on_tool_event: Optional[Callable[[str, str, dict], None]],
) -> Callable[..., str]:
    """Closure factory — one executor per tool so names bind correctly."""

    def _execute(args: Any = None, context: Any = None) -> str:
        parsed: dict[str, Any]
        if isinstance(args, dict):
            parsed = args
        elif args is None:
            parsed = {}
        else:
            # The SDK hands parsed argument mappings; tolerate strings anyway.
            try:
                parsed = json.loads(args)
                if not isinstance(parsed, dict):
                    parsed = {"value": parsed}
            except Exception:
                parsed = {"value": str(args)}

        if on_tool_event is not None:
            try:
                on_tool_event(tool_name, _preview(parsed), parsed)
            except Exception:
                logger.debug("cursor custom_tools: tool-event hook raised", exc_info=True)

        try:
            result = handle_function_call(
                tool_name,
                parsed,
                task_id=task_id,
                session_id=session_id,
            )
        except Exception as exc:
            logger.exception("cursor custom_tools: %s raised", tool_name)
            return json.dumps({"error": str(exc), "tool": tool_name})

        if not isinstance(result, str):
            try:
                result = json.dumps(result, ensure_ascii=False, default=str)
            except Exception:
                result = str(result)
        if len(result) > _MAX_RESULT_CHARS:
            result = (
                result[:_MAX_RESULT_CHARS]
                + f"\n... [truncated {len(result) - _MAX_RESULT_CHARS} chars]"
            )
        return result

    _execute.__name__ = tool_name
    return _execute


def _preview(args: dict) -> str:
    for key in ("query", "url", "path", "prompt", "text", "name"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:200]
    return ""
