"""Cursor agent runtime — turns driven through the official cursor-sdk.

Extracted runtime module in the mold of :mod:`agent.codex_runtime`: when
``provider: cursor`` is active (api_mode ``"cursor_agent"``),
``run_conversation()`` hands each turn to :func:`run_cursor_agent_turn`
instead of the chat_completions loop. The cursor agent owns the turn's
reason+act loop (its shell/read/edit tools, its own context window);
Hermes owns everything around it — sessions, memory review, skills,
gateway delivery — and additionally injects its own tool surface INTO the
cursor turn via the SDK's custom_tools (see
``agent/transports/cursor_hermes_tools.py``).

Each function takes the parent ``AIAgent`` as its first argument
(``agent``); AIAgent keeps a thin forwarder method for dispatch.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

from agent.memory_manager import build_memory_context_block

logger = logging.getLogger(__name__)

_CURSOR_HOST_CONTEXT = (
    "[Hermes host context]\n"
    "You are a Cursor agent running inside Hermes. Hermes provides the "
    "surrounding session, gateway, memory, and custom tools. Identify as the "
    "selected Cursor model, not as Hermes."
)


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    return 0


def _extract_images_from_content(content: Any) -> list[dict]:
    """Pull OpenAI-style image parts out of a rich user message.

    The gateway/TUI can hand ``original_user_message`` as a content-part
    list. Cursor's SDK takes ``{"data": <b64>, "mime_type": ...}`` or
    ``{"url": ...}`` image mappings — translate both data-URI and remote-URL
    shapes; ignore anything unparseable."""
    if not isinstance(content, list):
        return []
    images: list[dict] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in {"image", "image_url", "input_image"}:
            continue
        url_obj = part.get("image_url")
        url = ""
        if isinstance(url_obj, dict):
            url = str(url_obj.get("url") or "")
        elif isinstance(url_obj, str):
            url = url_obj
        elif isinstance(part.get("url"), str):
            url = part["url"]
        if not url:
            continue
        if url.startswith("data:"):
            try:
                header, payload = url.split(",", 1)
                mime = header[len("data:"):].split(";", 1)[0] or "image/png"
                # Validate the payload is real base64 before forwarding.
                base64.b64decode(payload, validate=True)
                images.append({"data": payload, "mime_type": mime})
            except Exception:
                logger.debug("cursor runtime: unparseable data-URI image part")
        else:
            images.append({"url": url})
    return images


def _compose_cursor_user_input(
    user_message: str,
    *,
    external_memory_context: str = "",
    plugin_user_context: str = "",
) -> str:
    """Add API-only host, memory, and plugin context to one user payload."""
    injections = [_CURSOR_HOST_CONTEXT]
    if external_memory_context:
        fenced_memory = build_memory_context_block(external_memory_context)
        if fenced_memory:
            injections.append(fenced_memory)
    if plugin_user_context:
        injections.append(plugin_user_context)
    return user_message + "\n\n" + "\n\n".join(injections)


def _load_cursor_runtime_config() -> tuple[dict, Optional[dict]]:
    """Read the ``cursor:`` section + ``mcp_servers`` from merged config.

    Read once per session build — never mid-conversation — so runtime
    behavior stays stable for the life of the session (cache/alternation
    policy). Both the CLI and the gateway resolve through
    ``hermes_cli.config.load_config()`` (DEFAULT_CONFIG deep-merged with the
    user's YAML), so the section is present even on raw gateway configs."""
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        logger.debug("cursor runtime: load_config failed; using defaults", exc_info=True)
        return {}, None
    cursor_cfg = config.get("cursor")
    mcp = config.get("mcp_servers")
    return (
        dict(cursor_cfg) if isinstance(cursor_cfg, dict) else {},
        mcp if isinstance(mcp, dict) else None,
    )


def _record_cursor_usage(agent, turn) -> Dict[str, Any]:
    """Translate cursor-sdk token usage into Hermes accounting.

    The SDK reports per-turn usage via ``usage`` stream events and a
    cumulative ``run.usage`` — both TokenUsage-shaped with snake_case
    fields: input_tokens, output_tokens, cache_read_tokens,
    cache_write_tokens, total_tokens, reasoning_tokens.

    Even when cursor omits usage for a run, Hermes still counts the turn as
    one API call for session/status accounting. Cost is "included" — SDK
    runs bill the user's Cursor subscription (see usage_pricing route).
    """
    agent.session_api_calls += 1

    last_usage = getattr(turn, "token_usage_last", None)
    total_usage = getattr(turn, "token_usage_total", None)
    if not isinstance(last_usage, dict) or not last_usage:
        last_usage = None
    if not isinstance(total_usage, dict) or not total_usage:
        total_usage = None

    # Cursor emits one usage event per internal model step and the SDK sums
    # those events into ``run.usage``. The sum is correct for billing/session
    # accounting but is not the occupancy of the final context window.
    billing_usage = total_usage or last_usage
    if billing_usage is None:
        if agent._session_db and agent.session_id:
            try:
                if not agent._session_db_created:
                    agent._ensure_db_session()
                agent._session_db.update_token_counts(
                    agent.session_id,
                    model=agent.model,
                    api_call_count=1,
                )
            except Exception as exc:
                logger.debug(
                    "cursor api-call persistence failed (session=%s): %s",
                    agent.session_id, exc,
                )
        return {}

    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    def _canonicalize(raw_usage: dict) -> CanonicalUsage:
        return CanonicalUsage(
            input_tokens=_coerce_usage_int(raw_usage.get("input_tokens")),
            output_tokens=_coerce_usage_int(raw_usage.get("output_tokens")),
            cache_read_tokens=_coerce_usage_int(
                raw_usage.get("cache_read_tokens")
            ),
            cache_write_tokens=_coerce_usage_int(
                raw_usage.get("cache_write_tokens")
            ),
            reasoning_tokens=_coerce_usage_int(
                raw_usage.get("reasoning_tokens")
            ),
            raw_usage=raw_usage,
        )

    canonical_usage = _canonicalize(billing_usage)
    prompt_tokens = canonical_usage.prompt_tokens
    completion_tokens = canonical_usage.output_tokens
    total_tokens = (
        _coerce_usage_int(billing_usage.get("total_tokens"))
        or canonical_usage.total_tokens
    )
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": canonical_usage.reasoning_tokens,
    }

    # Update the context bar only from the final individual usage event.
    # When the SDK supplies total-only metadata, retaining the previous meter
    # is more truthful than displaying a multi-step aggregate as window fill.
    context_prompt_tokens: Optional[int] = None
    context_usage: Optional[CanonicalUsage] = None
    compressor = getattr(agent, "context_compressor", None)
    if last_usage is not None:
        context_usage = _canonicalize(last_usage)
        context_prompt_tokens = context_usage.prompt_tokens
    if compressor is not None and context_usage is not None:
        try:
            compressor.update_from_response(
                {
                    "prompt_tokens": context_prompt_tokens,
                    "completion_tokens": context_usage.output_tokens,
                    "total_tokens": (
                        _coerce_usage_int(last_usage.get("total_tokens"))
                        or context_usage.total_tokens
                    ),
                    "input_tokens": context_usage.input_tokens,
                    "output_tokens": context_usage.output_tokens,
                    "cache_read_tokens": context_usage.cache_read_tokens,
                    "cache_write_tokens": context_usage.cache_write_tokens,
                    "reasoning_tokens": context_usage.reasoning_tokens,
                }
            )
        except Exception:
            logger.debug("cursor usage update failed", exc_info=True)

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens
    agent.session_reasoning_tokens += canonical_usage.reasoning_tokens

    cost_result = estimate_usage_cost(
        agent.model,
        canonical_usage,
        provider=agent.provider,
        base_url=agent.base_url,
        api_key=getattr(agent, "api_key", ""),
    )
    if cost_result.amount_usd is not None:
        agent.session_estimated_cost_usd += float(cost_result.amount_usd)
    agent.session_cost_status = cost_result.status
    agent.session_cost_source = cost_result.source

    if agent._session_db and agent.session_id:
        try:
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.update_token_counts(
                agent.session_id,
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                reasoning_tokens=canonical_usage.reasoning_tokens,
                estimated_cost_usd=float(cost_result.amount_usd)
                if cost_result.amount_usd is not None else None,
                cost_status=cost_result.status,
                cost_source=cost_result.source,
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode="subscription_included"
                if cost_result.status == "included" else None,
                model=agent.model,
                api_call_count=1,
            )
        except Exception as exc:
            logger.debug(
                "cursor token persistence failed (session=%s, tokens=%d): %s",
                agent.session_id, total_tokens, exc,
            )

    result = {
        **usage_dict,
        "estimated_cost_usd": float(cost_result.amount_usd)
        if cost_result.amount_usd is not None else None,
        "cost_status": cost_result.status,
        "cost_source": cost_result.source,
    }
    if context_prompt_tokens is not None:
        result["last_prompt_tokens"] = context_prompt_tokens
    return result


def _build_cursor_session(agent, effective_task_id: str):
    """Construct a CursorSDKSession bound to this AIAgent."""
    from agent.runtime_cwd import resolve_agent_cwd
    from agent.transports.cursor_sdk_session import CursorSDKSession

    cwd = getattr(agent, "session_cwd", None) or str(resolve_agent_cwd())
    cursor_cfg, hermes_mcp = _load_cursor_runtime_config()

    def _on_tool_event(tool_name: str, preview: str, args: dict) -> None:
        # Bridge cursor-internal tool starts (and bridged Hermes custom
        # tools) to Hermes tool-progress so gateways show verbose
        # "running X" breadcrumbs on this route too — same contract the
        # codex runtime fulfills via _codex_note_to_tool_progress.
        progress_callback = getattr(agent, "tool_progress_callback", None)
        if progress_callback is None:
            return
        try:
            progress_callback("tool.started", tool_name, preview, args)
        except Exception:
            logger.debug("cursor tool-progress callback raised", exc_info=True)

    def _on_text_delta(text: str) -> None:
        agent._fire_stream_delta(text)

    def _on_reasoning_delta(text: str) -> None:
        agent._fire_reasoning_delta(text)

    cursor_step_count = 0

    def _on_step(step: Any) -> None:
        nonlocal cursor_step_count
        callback = getattr(agent, "step_callback", None)
        if callback is None:
            return
        cursor_step_count += 1
        try:
            callback(cursor_step_count, [])
        except Exception:
            logger.debug("cursor step callback raised", exc_info=True)

    return CursorSDKSession(
        cwd=cwd,
        api_key=getattr(agent, "api_key", None),
        model=getattr(agent, "model", "") or "",
        cursor_config=cursor_cfg,
        hermes_mcp_servers=hermes_mcp,
        session_id=getattr(agent, "session_id", None),
        session_title=getattr(agent, "session_title", None),
        task_id=effective_task_id,
        on_tool_event=_on_tool_event,
        enabled_toolsets=getattr(agent, "enabled_toolsets", None),
        disabled_toolsets=getattr(agent, "disabled_toolsets", None),
        on_text_delta=_on_text_delta,
        on_reasoning_delta=_on_reasoning_delta,
        on_step=_on_step if getattr(agent, "step_callback", None) is not None else None,
        # Poll the agent's interrupt flag so /stop (gateway) and Ctrl+C (CLI)
        # cancel the in-flight cursor run.
        interrupt_check=lambda: bool(getattr(agent, "_interrupt_requested", False)),
    )


def _finalize_cursor_result(agent, result: Dict[str, Any]) -> Dict[str, Any]:
    """Restore the per-turn state normally cleared by ``finalize_turn``.

    Cursor returns early from :mod:`agent.conversation_loop`, so it must
    explicitly honor the result contract consumed by the CLI and gateway.
    """
    if result.get("interrupted") and getattr(agent, "_interrupt_message", None):
        result["interrupt_message"] = agent._interrupt_message

    pending_steer = agent._drain_pending_steer()
    if pending_steer:
        result["pending_steer"] = pending_steer

    agent.clear_interrupt()
    agent._stream_callback = None
    return result


def run_cursor_agent_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
    external_memory_context: str = "",
    plugin_user_context: str = "",
) -> Dict[str, Any]:
    """Cursor runtime path. Hands the entire turn to the cursor-sdk agent
    and projects its stream back into Hermes' messages list so memory/skill
    review keep working.

    Called from run_conversation() when agent.api_mode == "cursor_agent".
    Returns the same dict shape as the chat_completions path.
    """
    # Lazy session: one CursorSDKSession per AIAgent instance. Created on
    # first turn, reused across turns (the SDK agent keeps conversation
    # context server/bridge-side), retired on wedge/crash.
    if getattr(agent, "_cursor_session", None) is None:
        agent._cursor_session = _build_cursor_session(agent, effective_task_id)

    # NOTE: the user message is ALREADY appended to messages by the standard
    # run_conversation() flow before this early-return path runs. Do NOT
    # append again — that would duplicate it.

    images = _extract_images_from_content(original_user_message)
    outbound_user_message = _compose_cursor_user_input(
        user_message,
        external_memory_context=external_memory_context,
        plugin_user_context=plugin_user_context,
    )

    try:
        turn = agent._cursor_session.run_turn(
            outbound_user_message,
            images=images or None,
            # Pass the agent's current model every turn — the session tracks
            # the sticky selection and only sends an override after a
            # mid-session /model switch.
            model=getattr(agent, "model", "") or None,
        )
    except Exception as exc:
        logger.exception("cursor runtime turn failed")
        # Crash → unconditionally drop the session so the next turn
        # rebuilds bridge + agent instead of reusing a dead client.
        try:
            agent._cursor_session.retire()
        except Exception:
            pass
        agent._cursor_session = None
        return _finalize_cursor_result(agent, {
            "final_response": (
                f"Cursor runtime turn failed: {exc}. "
                "Check `hermes doctor` and your CURSOR_API_KEY, or switch "
                "providers with /model."
            ),
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "error": str(exc),
            "interrupted": False,
        })

    # Wedged bridge / expired run / auth failure → retire the session so the
    # next turn starts fresh (mirrors the codex runtime's retire semantics).
    if getattr(turn, "should_retire", False):
        logger.warning("cursor session retired (turn error: %s)", turn.error)
        try:
            agent._cursor_session.retire()
        except Exception:
            pass
        agent._cursor_session = None

    # Splice projected messages into the conversation. The projector emits
    # standard {role, content, tool_calls, tool_call_id} entries — exactly
    # what curator.py / the sessions DB expect.
    if turn.projected_messages:
        messages.extend(turn.projected_messages)
        # Persist the newly-projected assistant/tool rows ourselves — this
        # path bypasses conversation_loop's per-step _persist_session().
        # The inbound user turn was already flushed at turn start and the
        # flush dedups via _DB_PERSISTED_MARKER, so this writes only the new
        # rows (same reasoning as the codex runtime; see #860/#42039).
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug("cursor projected-message flush failed", exc_info=True)

    # Counter ticks for the agent-improvement loop. _turns_since_memory and
    # _user_turn_count are already incremented in run_conversation()'s
    # pre-loop block; only _iters_since_skill needs explicit bumping here
    # (the chat_completions loop does it per tool iteration).
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )
    usage_result = _record_cursor_usage(agent, turn)
    api_calls = 1

    should_review_skills = False
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    # External memory provider sync — skipped on interrupt/error to avoid
    # feeding partial transcripts to memory.
    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
                messages=messages,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    # Background review fork — same cadence + signature as the default path.
    if (
        turn.final_text
        and not turn.interrupted
        and (should_review_memory or should_review_skills)
    ):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=should_review_memory,
                review_skills=should_review_skills,
            )
        except Exception:
            logger.debug("background review spawn raised", exc_info=True)

    return _finalize_cursor_result(agent, {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": api_calls,
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "error": turn.error,
        "interrupted": bool(turn.interrupted),
        # Early-return path that bypasses conversation_loop — but the
        # projected assistant/tool rows are flushed above and the user turn
        # was flushed at turn start, so the agent is the sole persister.
        # agent_persisted=True tells the gateway to skip its own DB write
        # (see the codex runtime's identical note re #860/#42039).
        "agent_persisted": True,
        "cursor_agent_id": turn.agent_id,
        "cursor_run_id": turn.run_id,
        **usage_result,
    })
