"""Session adapter for the cursor-sdk agent runtime.

Owns one cursor-sdk ``Agent`` per Hermes session. Drives ``agent.send()``,
consumes the typed stream via :class:`CursorEventProjector`, translates
cancellation, and returns a clean turn result that
``AIAgent.run_conversation()`` can splice into its ``messages`` list —
the same contract :class:`CodexAppServerSession` fulfills for the codex
app-server runtime.

Lifecycle::

    session = CursorSDKSession(cwd=..., api_key=..., model=..., cursor_config=...)
    session.ensure_started()                    # bridge + Agent.create/resume
    result = session.run_turn(user_input="hi")  # blocks until the run finishes
    # result.final_text          → assistant text returned to caller
    # result.projected_messages  → list of {role, content, ...} for messages list
    # result.tool_iterations     → completed tool calls (skill nudge counter)
    # result.interrupted         → True if interrupt fired mid-turn
    session.close()                             # dispose agent + bridge client

Threading model: the adapter is single-threaded from the caller's
perspective (AIAgent.run_conversation() is synchronous). The SDK's stream
iterator blocks between events, so each run is consumed by a small pump
thread feeding a queue that the caller polls with timeouts — that is what
lets us honor interrupts (``run.cancel()``) and idle timeouts without an
asyncio layer. Mirrors the codex client's deliberate not-async design.

The cursor-sdk import is lazy (``tools/lazy_deps.py`` feature
``provider.cursor``) and injectable (``sdk_module=``) so tests never touch
the ~48 MB wheel or the network.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from agent.redact import redact_sensitive_text
from agent.transports.cursor_bridge import launch_cursor_bridge
from agent.transports.cursor_event_projector import CursorEventProjector
from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

# How many session→agent-id records to retain in the sidecar store.
_SESSION_STORE_MAX_ENTRIES = 200

# Grace period to drain remaining stream events after a cancel request.
_CANCEL_DRAIN_SECONDS = 15.0

# Queue poll cadence — how often we can notice an interrupt.
_POLL_SECONDS = 0.25

# Upper bound for a server-suggested retry_after sleep.
_MAX_RETRY_AFTER_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Session ↔ cursor agent-id persistence (resume support)
# ---------------------------------------------------------------------------

def _session_store_path() -> Path:
    return get_hermes_home() / "cursor" / "sessions.json"


def _load_session_store() -> dict:
    try:
        import json

        path = _session_store_path()
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.debug("cursor session store unreadable", exc_info=True)
        return {}


def load_persisted_agent_record(session_id: Optional[str]) -> Optional[dict]:
    """Return the persisted cursor agent record for a Hermes session id."""
    if not session_id:
        return None
    record = _load_session_store().get(str(session_id))
    return record if isinstance(record, dict) else None


def persist_agent_record(session_id: Optional[str], record: dict) -> None:
    """Persist ``{session_id → record}``, pruning the oldest entries."""
    if not session_id:
        return
    try:
        store = _load_session_store()
        record = dict(record)
        record["updated_at"] = time.time()
        store[str(session_id)] = record
        if len(store) > _SESSION_STORE_MAX_ENTRIES:
            oldest_first = sorted(
                store.items(),
                key=lambda kv: kv[1].get("updated_at", 0) if isinstance(kv[1], dict) else 0,
            )
            for key, _ in oldest_first[: len(store) - _SESSION_STORE_MAX_ENTRIES]:
                store.pop(key, None)
        path = _session_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, store)
    except Exception:
        logger.debug("cursor session store write failed", exc_info=True)


def clear_persisted_agent_record(session_id: Optional[str]) -> None:
    if not session_id:
        return
    try:
        store = _load_session_store()
        if str(session_id) in store:
            store.pop(str(session_id), None)
            atomic_json_write(_session_store_path(), store)
    except Exception:
        logger.debug("cursor session store clear failed", exc_info=True)


# ---------------------------------------------------------------------------
# Option builders
# ---------------------------------------------------------------------------

def build_model_selection(model_id: str, model_params: Any = None) -> Any:
    """Build the SDK model argument: bare id, or {"id", "params"} mapping.

    ``model_params`` accepts a model-keyed mapping, e.g.
    ``{"composer-2.5": {"fast": "true"}}``. A legacy flat mapping remains
    supported as a global default for backward compatibility.
    Values are stringified — the SDK's ModelParameterValue.value is a str.
    Invalid ids/values are passed through for the server to validate; we do
    not hardcode a parameter catalog that would rot.
    """
    model_id = (model_id or "").strip()
    if not model_id:
        return model_id
    if not isinstance(model_params, dict) or not model_params:
        return model_id

    scoped_params = model_params.get(model_id)
    if isinstance(scoped_params, dict):
        model_params = scoped_params
    elif any(isinstance(value, dict) for value in model_params.values()):
        default_params = model_params.get("default")
        if not isinstance(default_params, dict):
            return model_id
        model_params = default_params

    params = [
        {"id": str(key), "value": str(value)}
        for key, value in model_params.items()
        if str(key).strip()
    ]
    if not params:
        return model_id
    return {"id": model_id, "params": params}


def translate_hermes_mcp_servers(hermes_mcp: Any) -> dict[str, dict]:
    """Translate Hermes' ``mcp_servers`` config into SDK inline definitions.

    Hermes shape (config.yaml)::

        mcp_servers:
          filesystem: {command: npx, args: [...], env: {...}}
          docs:       {url: https://..., headers: {...}}

    SDK shape::

        {"filesystem": {"type": "stdio", "command": ..., "args": [...], "env": {...}},
         "docs":       {"type": "http", "url": ..., "headers": {...}}}

    Servers that require interactive OAuth are skipped — the SDK cannot open
    a browser to sign in. Disabled entries are skipped. Never raises.
    """
    if not isinstance(hermes_mcp, dict):
        return {}
    translated: dict[str, dict] = {}
    for name, entry in hermes_mcp.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("enabled") is False or entry.get("disabled") is True:
            continue
        if entry.get("oauth"):
            logger.info(
                "cursor mcp passthrough: skipping %s — interactive OAuth is "
                "not supported inside cursor-sdk runs",
                name,
            )
            continue
        command = str(entry.get("command") or "").strip()
        url = str(entry.get("url") or "").strip()
        if command:
            server: dict[str, Any] = {"type": "stdio", "command": command}
            args = entry.get("args")
            if isinstance(args, (list, tuple)):
                server["args"] = [str(a) for a in args]
            env = entry.get("env")
            if isinstance(env, dict) and env:
                server["env"] = {str(k): str(v) for k, v in env.items()}
            translated[str(name)] = server
        elif url:
            server = {"type": "http", "url": url}
            headers = entry.get("headers")
            if isinstance(headers, dict) and headers:
                server["headers"] = {str(k): str(v) for k, v in headers.items()}
            translated[str(name)] = server
        else:
            logger.debug(
                "cursor mcp passthrough: skipping %s — no command or url", name
            )
    return translated


def _clean_subagent_definitions(agents_cfg: Any) -> dict[str, dict]:
    """Validate ``cursor.agents`` config into SDK AgentDefinition mappings."""
    if not isinstance(agents_cfg, dict):
        return {}
    cleaned: dict[str, dict] = {}
    for name, entry in agents_cfg.items():
        if not isinstance(entry, dict):
            continue
        description = str(entry.get("description") or "").strip()
        prompt = str(entry.get("prompt") or "").strip()
        if not description or not prompt:
            logger.warning(
                "cursor.agents.%s ignored — both description and prompt are required",
                name,
            )
            continue
        definition: dict[str, Any] = {"description": description, "prompt": prompt}
        model = entry.get("model")
        if isinstance(model, str) and model.strip():
            definition["model"] = model.strip()
        cleaned[str(name)] = definition
    return cleaned


def _usage_to_dict(usage: Any) -> Optional[dict]:
    """Convert a TokenUsage dataclass/mapping into a plain dict."""
    if usage is None:
        return None
    fields = (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "total_tokens",
        "reasoning_tokens",
    )
    out: dict[str, Any] = {}
    for name in fields:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            try:
                out[name] = int(value)
            except (TypeError, ValueError):
                continue
    return out or None


def _model_id_of(selection: Any) -> str:
    """Best-effort model id from a ModelSelection / mapping / str."""
    if selection is None:
        return ""
    if isinstance(selection, str):
        return selection
    if isinstance(selection, dict):
        return str(selection.get("id") or "")
    return str(getattr(selection, "id", "") or "")


def _exc_class(sdk: Any, name: str) -> type:
    """Look up an SDK exception class, degrading to a never-raised dummy."""
    cls = getattr(sdk, name, None)
    if isinstance(cls, type) and issubclass(cls, BaseException):
        return cls

    class _Never(Exception):
        pass

    return _Never


# ---------------------------------------------------------------------------
# Turn result
# ---------------------------------------------------------------------------

@dataclass
class CursorTurnResult:
    """Result of one user→assistant→tool turn through the cursor runtime."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    status: str = ""
    model_used: str = ""
    token_usage_last: Optional[dict] = None   # last turn's TokenUsage
    token_usage_total: Optional[dict] = None  # cumulative run TokenUsage
    # Hint that the bridge/agent is likely wedged or unauthenticated and the
    # caller should retire the session so the next turn starts fresh.
    should_retire: bool = False


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------

class CursorSDKSession:
    """One cursor-sdk Agent per Hermes session, lifetime owned by AIAgent.

    Not thread-safe — one caller drives it at a time, matching
    AIAgent.run_conversation(). The internal stream pump thread only ever
    feeds this object's queue.
    """

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "",
        cursor_config: Optional[dict] = None,
        hermes_mcp_servers: Optional[dict] = None,
        session_id: Optional[str] = None,
        session_title: Optional[str] = None,
        task_id: Optional[str] = None,
        on_tool_event: Optional[Callable[[str, str, dict], None]] = None,
        interrupt_check: Optional[Callable[[], bool]] = None,
        enabled_toolsets: Optional[list[str]] = None,
        disabled_toolsets: Optional[list[str]] = None,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_reasoning_delta: Optional[Callable[[str], None]] = None,
        on_step: Optional[Callable[[Any], None]] = None,
        sdk_module: Any = None,
        custom_tools_builder: Optional[Callable[..., dict]] = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._api_key = api_key or os.environ.get("CURSOR_API_KEY", "")
        self._model = (model or "").strip()
        self._config = dict(cursor_config or {})
        self._hermes_mcp_servers = hermes_mcp_servers
        self._session_id = session_id
        self._session_title = session_title
        self._task_id = task_id
        self._on_tool_event = on_tool_event
        self._interrupt_check = interrupt_check
        self._enabled_toolsets = enabled_toolsets
        self._disabled_toolsets = disabled_toolsets
        self._on_text_delta = on_text_delta
        self._on_reasoning_delta = on_reasoning_delta
        self._on_step = on_step
        self._sdk = sdk_module
        self._custom_tools_builder = custom_tools_builder

        self._client: Any = None
        self._agent: Any = None
        self._agent_id: Optional[str] = None
        self._sticky_model: str = ""
        self._interrupt_event = threading.Event()
        self._closed = False

    # ---------- config accessors ----------

    @property
    def runtime(self) -> str:
        value = str(self._config.get("runtime") or "local").strip().lower()
        return value if value in {"local", "cloud"} else "local"

    @property
    def mode(self) -> str:
        value = str(self._config.get("mode") or "agent").strip().lower()
        return value if value in {"agent", "plan"} else "agent"

    @property
    def timeout_seconds(self) -> float:
        try:
            value = float(self._config.get("timeout_seconds") or 1800)
        except (TypeError, ValueError):
            value = 1800.0
        return max(30.0, value)

    # ---------- lifecycle ----------

    def _import_sdk(self) -> Any:
        if self._sdk is not None:
            return self._sdk
        from tools.lazy_deps import FeatureUnavailable, ensure

        try:
            ensure("provider.cursor", prompt=False)
        except FeatureUnavailable as exc:
            raise RuntimeError(
                "The Cursor provider needs the official cursor-sdk package "
                f"and it could not be installed automatically: {exc}\n"
                "Install it manually with:  pip install cursor-sdk\n"
                "or re-enable lazy installs (security.allow_lazy_installs) "
                "via `hermes tools`."
            ) from exc
        import cursor_sdk  # noqa: PLC0415 — lazy on purpose (48 MB wheel)

        self._sdk = cursor_sdk
        return self._sdk

    def _launch_client(self, sdk: Any) -> Any:
        return launch_cursor_bridge(sdk, workspace=self._cwd)

    def _build_create_kwargs(self) -> dict[str, Any]:
        # cursor-sdk 0.1.9 ``agents.create`` only accepts top-level
        # model/api_key/name/local/cloud/idempotency_key. Everything else
        # (mode, agents, mcp_servers, …) lives on the ``options`` mapping
        # (AgentOptions). Passing ``mode=`` as a top-level kwarg raises
        # TypeError: unexpected keyword argument 'mode'.
        model_selection = build_model_selection(
            self._model, self._config.get("model_params")
        )
        kwargs: dict[str, Any] = {
            "model": model_selection,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._session_title:
            kwargs["name"] = str(self._session_title)[:120]

        options: dict[str, Any] = {}
        mode = self.mode
        if mode in {"agent", "plan"}:
            options["mode"] = mode

        subagents = _clean_subagent_definitions(self._config.get("agents"))
        if subagents:
            options["agents"] = subagents

        if self._config.get("inherit_mcp"):
            mcp = translate_hermes_mcp_servers(self._hermes_mcp_servers)
            if mcp:
                options["mcp_servers"] = mcp

        if options:
            kwargs["options"] = options

        if self.runtime == "cloud":
            kwargs["cloud"] = self._build_cloud_options()
        else:
            kwargs["local"] = self._build_local_options()
        return kwargs

    def _build_local_options(self) -> dict[str, Any]:
        local: dict[str, Any] = {"cwd": self._cwd}
        sources = self._config.get("setting_sources")
        if isinstance(sources, (list, tuple)) and sources:
            local["setting_sources"] = [str(s) for s in sources]
        sandbox = self._config.get("sandbox")
        if isinstance(sandbox, dict) and sandbox:
            local["sandbox_options"] = dict(sandbox)
        if self._config.get("expose_hermes_tools", True):
            builder = self._custom_tools_builder
            if builder is None:
                from agent.transports.cursor_hermes_tools import (
                    build_cursor_custom_tools,
                )

                builder = build_cursor_custom_tools
            try:
                custom_tools = builder(
                    self._task_id,
                    session_id=self._session_id,
                    on_tool_event=self._on_tool_event,
                    enabled_toolsets=self._enabled_toolsets,
                    disabled_toolsets=self._disabled_toolsets,
                )
            except Exception:
                logger.exception("cursor custom_tools build failed; continuing without")
                custom_tools = {}
            if custom_tools:
                local["custom_tools"] = custom_tools
        return local

    def _build_cloud_options(self) -> dict[str, Any]:
        cloud_cfg = self._config.get("cloud")
        cloud_cfg = cloud_cfg if isinstance(cloud_cfg, dict) else {}
        cloud: dict[str, Any] = {}
        repos = []
        for entry in cloud_cfg.get("repos") or []:
            if isinstance(entry, str) and entry.strip():
                repos.append({"url": entry.strip()})
            elif isinstance(entry, dict) and entry.get("url"):
                repo: dict[str, Any] = {"url": str(entry["url"]).strip()}
                ref = entry.get("ref") or entry.get("starting_ref")
                if ref:
                    repo["starting_ref"] = str(ref)
                repos.append(repo)
        if repos:
            cloud["repos"] = repos
        if cloud_cfg.get("auto_create_pr"):
            cloud["auto_create_pr"] = True
        if cloud_cfg.get("work_on_current_branch"):
            cloud["work_on_current_branch"] = True
        env = cloud_cfg.get("env")
        if isinstance(env, dict) and env:
            cloud["env"] = dict(env)
        return cloud

    def ensure_started(self) -> str:
        """Launch the bridge and create/resume the SDK agent.

        Idempotent — repeated calls return the same agent id."""
        if self._agent is not None and self._agent_id:
            return self._agent_id

        sdk = self._import_sdk()
        if self._client is None:
            self._client = self._launch_client(sdk)

        agents_api = getattr(self._client, "agents", None)

        # Try resuming a previously persisted cursor agent for this Hermes
        # session (survives process restarts: local via bridge workspace
        # state, cloud server-side).
        record = load_persisted_agent_record(self._session_id)
        if record and agents_api is not None:
            persisted_id = str(record.get("agent_id") or "")
            same_context = (
                record.get("runtime") == self.runtime
                and (self.runtime == "cloud" or record.get("cwd") == self._cwd)
            )
            if persisted_id and same_context:
                try:
                    resume_options: dict[str, Any] = {}
                    if self._api_key:
                        resume_options["api_key"] = self._api_key
                    if self._model:
                        resume_options["model"] = build_model_selection(
                            self._model, self._config.get("model_params")
                        )
                    self._agent = agents_api.resume(persisted_id, resume_options or None)
                    self._agent_id = str(
                        getattr(self._agent, "agent_id", "") or persisted_id
                    )
                    self._sticky_model = self._model
                    logger.info(
                        "cursor session resumed agent %s (runtime=%s)",
                        self._agent_id[:16],
                        self.runtime,
                    )
                    return self._agent_id
                except Exception as exc:
                    logger.info(
                        "cursor resume of %s failed (%s); creating a fresh agent",
                        persisted_id[:16],
                        exc.__class__.__name__,
                    )
                    clear_persisted_agent_record(self._session_id)

        kwargs = self._build_create_kwargs()
        if agents_api is not None:
            self._agent = agents_api.create(**kwargs)
        else:  # pragma: no cover - very old SDK surface
            self._agent = sdk.Agent.create(client=self._client, **kwargs)
        self._agent_id = str(getattr(self._agent, "agent_id", "") or "")
        self._sticky_model = self._model
        logger.info(
            "cursor session created agent %s (runtime=%s mode=%s cwd=%s)",
            (self._agent_id or "?")[:16],
            self.runtime,
            self.mode,
            self._cwd,
        )
        persist_agent_record(
            self._session_id,
            {
                "agent_id": self._agent_id,
                "runtime": self.runtime,
                "cwd": self._cwd,
                "model": self._model,
            },
        )
        return self._agent_id or ""

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for target, verb in ((self._agent, "close"), (self._client, "close")):
            if target is None:
                continue
            try:
                getattr(target, verb)()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
        self._agent = None
        self._client = None

    def retire(self) -> None:
        """Close a wedged session and prevent later resume of its agent id."""
        self.close()
        clear_persisted_agent_record(self._session_id)

    def __enter__(self) -> "CursorSDKSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- interrupt ----------

    def request_interrupt(self) -> None:
        """Idempotent: signal the active turn loop to cancel the run."""
        self._interrupt_event.set()

    def _interrupted(self) -> bool:
        if self._interrupt_event.is_set():
            return True
        if self._interrupt_check is not None:
            try:
                return bool(self._interrupt_check())
            except Exception:
                return False
        return False

    # ---------- per-turn ----------

    def run_turn(
        self,
        user_input: Any,
        *,
        images: Optional[list[dict]] = None,
        model: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> CursorTurnResult:
        """Send a user message and block until the run reaches a terminal
        status, projecting stream messages into Hermes' messages shape.

        ``model``: per-turn override (a mid-session ``/model`` switch). The
        SDK makes overrides sticky, matching Hermes' own semantics.
        ``images``: list of ``{"data": <b64>, "mime_type": ...}`` mappings.
        """
        # ``request_interrupt()`` is scoped to the active turn. Without this,
        # one cancelled run poisons every later run on the reused SDK session.
        # The parent AIAgent flag is checked separately through
        # ``interrupt_check`` and remains authoritative for an interrupt that
        # races with turn startup.
        self._interrupt_event.clear()
        result = CursorTurnResult()
        try:
            self.ensure_started()
        except Exception as exc:
            result.error = self._format_error("cursor runtime startup failed", exc)
            result.should_retire = True
            return result
        result.agent_id = self._agent_id

        sdk = self._sdk
        message = self._build_message(user_input, images)
        send_options = self._build_send_options(model=model, mode=mode)

        try:
            run = self._send_with_recovery(sdk, message, send_options)
        except Exception as exc:
            result.error = self._format_error("cursor send failed", exc)
            result.should_retire = self._looks_like_auth_failure(exc)
            return result

        if model:
            self._sticky_model = model.strip()
        result.run_id = str(getattr(run, "id", "") or "")

        projector = CursorEventProjector()
        stream_drained = self._consume_stream(run, projector, result)

        # ``Run.wait()`` has no timeout and drains the same event stream.
        # Calling it after our cancel-drain deadline expired can hang forever
        # while the pump thread is still blocked in ``run.messages()``. Only
        # ask the SDK for terminal metadata after the stream ended normally.
        final = None
        if stream_drained:
            try:
                final = run.wait()
            except Exception:
                final = None

        finalize = projector.finalize()
        result.projected_messages.extend(finalize.messages)
        if finalize.messages:
            result.tool_iterations += sum(
                1 for m in finalize.messages if m.get("role") == "tool"
            )

        status = str(
            getattr(final, "status", None) or getattr(run, "status", "") or ""
        ).lower()
        result.status = status
        if status == "cancelled":
            result.interrupted = True

        terminal_text = str(
            getattr(final, "result", None) or getattr(run, "result", "") or ""
        )
        result.final_text = terminal_text or projector.final_text

        result.model_used = _model_id_of(
            getattr(final, "model", None) or getattr(run, "model", None)
        )
        total_usage = _usage_to_dict(
            getattr(final, "usage", None) or getattr(run, "usage", None)
        )
        if total_usage:
            result.token_usage_total = total_usage

        if status == "error" and not result.error:
            result.error = self._format_error(
                "cursor run ended in error",
                terminal_text or "(no error detail from the run)",
            )
        if status == "expired" and not result.error:
            result.error = "cursor run expired before completing"
            result.should_retire = True

        # Refresh the persisted record so resume survives restarts.
        persist_agent_record(
            self._session_id,
            {
                "agent_id": self._agent_id,
                "runtime": self.runtime,
                "cwd": self._cwd,
                "model": self._sticky_model or self._model,
            },
        )
        return result

    # ---------- send helpers ----------

    def _build_message(self, user_input: Any, images: Optional[list[dict]]) -> Any:
        text = user_input if isinstance(user_input, str) else str(user_input or "")
        if images:
            clean = [img for img in images if isinstance(img, dict)]
            if clean:
                return {"text": text, "images": clean}
        return text

    def _build_send_options(
        self, *, model: Optional[str], mode: Optional[str]
    ) -> Any:
        options: dict[str, Any] = {}
        model = (model or "").strip()
        if model and model != self._sticky_model:
            options["model"] = build_model_selection(
                model, self._config.get("model_params")
            )
        mode = (mode or "").strip().lower()
        if mode in {"agent", "plan"}:
            options["mode"] = mode

        # SDK callbacks are retained only when using the typed SendOptions
        # object; mapping options are serialized and discard Python callables.
        send_options_cls = getattr(self._sdk, "SendOptions", None)
        if send_options_cls is not None and (
            self._on_text_delta is not None
            or self._on_reasoning_delta is not None
            or self._on_step is not None
        ):
            return send_options_cls(
                model=options.get("model"),
                mode=options.get("mode"),
                on_delta=self._handle_sdk_delta,
                on_step=self._handle_sdk_step if self._on_step is not None else None,
            )
        return options or None

    def _handle_sdk_delta(self, update: Any) -> None:
        update_type = (
            update.get("type")
            if isinstance(update, dict)
            else getattr(update, "type", "")
        )
        text = (
            update.get("text", "")
            if isinstance(update, dict)
            else getattr(update, "text", "")
        )
        if update_type == "text-delta" and text and self._on_text_delta is not None:
            self._on_text_delta(str(text))
        elif (
            update_type == "thinking-delta"
            and text
            and self._on_reasoning_delta is not None
        ):
            self._on_reasoning_delta(str(text))

    def _handle_sdk_step(self, step: Any) -> None:
        if self._on_step is not None:
            self._on_step(step)

    def _send_with_recovery(self, sdk: Any, message: Any, options: Any) -> Any:
        """agent.send() with busy/stuck/rate-limit recovery (one retry each)."""
        agent_busy = _exc_class(sdk, "AgentBusyError")
        rate_limit = _exc_class(sdk, "RateLimitError")
        cursor_error = _exc_class(sdk, "CursorAgentError")

        try:
            return self._agent.send(message, options)
        except agent_busy:
            # Cloud agents allow one active run — cancel it and retry once.
            logger.info("cursor agent busy; cancelling the active run and retrying")
            self._cancel_active_run()
            return self._agent.send(message, options)
        except rate_limit as exc:
            self._sleep_for_retry(exc)
            return self._agent.send(message, options)
        except cursor_error as exc:
            if self.runtime == "local" and self._looks_like_stuck_run(exc):
                # Local agents don't raise AgentBusyError; a stuck active run
                # is expired via local.force per the SDK docs.
                logger.info("cursor local run appears stuck; retrying with local.force")
                if hasattr(options, "__dataclass_fields__"):
                    current_local = getattr(options, "local", None)
                    current_local = (
                        dict(current_local)
                        if isinstance(current_local, dict)
                        else {}
                    )
                    forced = replace(
                        options,
                        local={**current_local, "force": True},
                    )
                else:
                    forced = dict(options or {})
                    forced["local"] = {**(forced.get("local") or {}), "force": True}
                return self._agent.send(message, forced)
            if getattr(exc, "is_retryable", False):
                self._sleep_for_retry(exc)
                return self._agent.send(message, options)
            raise

    def _cancel_active_run(self) -> None:
        try:
            agents_api = getattr(self._client, "agents", None)
            if agents_api is None or not self._agent_id:
                return
            runs = agents_api.list_runs(self._agent_id)
            items = getattr(runs, "items", None) or []
            for item in items:
                if str(getattr(item, "status", "")).lower() == "running":
                    try:
                        item.cancel()
                    except Exception:
                        logger.debug("cursor active-run cancel failed", exc_info=True)
                    return
        except Exception:
            logger.debug("cursor active-run lookup failed", exc_info=True)

    @staticmethod
    def _sleep_for_retry(exc: Any) -> None:
        delay = 2.0
        retry_after = getattr(exc, "retry_after", None)
        if retry_after:
            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                delay = 2.0
        time.sleep(min(max(delay, 0.5), _MAX_RETRY_AFTER_SECONDS))

    @staticmethod
    def _looks_like_stuck_run(exc: Any) -> bool:
        text = str(exc).lower()
        return "busy" in text or "active run" in text or "already running" in text

    @staticmethod
    def _looks_like_auth_failure(exc: Any) -> bool:
        name = exc.__class__.__name__
        if name in {"AuthenticationError", "PermissionDeniedError"}:
            return True
        text = str(exc).lower()
        return "api key" in text or "unauthorized" in text or "unauthenticated" in text

    # ---------- stream consumption ----------

    def _consume_stream(
        self, run: Any, projector: CursorEventProjector, result: CursorTurnResult
    ) -> bool:
        """Pump run.messages() through the projector with interrupt + idle
        timeout support. Blocking iterator → pump thread + polled queue.

        Returns ``True`` only when the stream iterator drained normally. A
        ``False`` result means callers must not enter the SDK's unbounded
        ``run.wait()`` on the same partially-consumed stream.
        """
        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        def _pump() -> None:
            try:
                for message in run.messages():
                    events.put(("msg", message))
                events.put(("done", None))
            except BaseException as exc:  # noqa: BLE001 — surfaced to caller
                events.put(("error", exc))

        pump = threading.Thread(
            target=_pump, name="cursor-stream-pump", daemon=True
        )
        pump.start()

        idle_deadline = time.monotonic() + self.timeout_seconds
        cancel_requested = False
        cancel_deadline: Optional[float] = None

        while True:
            try:
                kind, payload = events.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                now = time.monotonic()
                if not cancel_requested and self._interrupted():
                    self._request_run_cancel(run)
                    cancel_requested = True
                    cancel_deadline = now + _CANCEL_DRAIN_SECONDS
                    result.interrupted = True
                    continue
                if cancel_requested and cancel_deadline and now > cancel_deadline:
                    # Cancel isn't draining — treat the bridge as wedged.
                    result.should_retire = True
                    return False
                if not cancel_requested and now > idle_deadline:
                    logger.warning(
                        "cursor run idle for %.0fs — cancelling and retiring the session",
                        self.timeout_seconds,
                    )
                    self._request_run_cancel(run)
                    result.error = (
                        f"cursor run produced no events for {int(self.timeout_seconds)}s "
                        "(idle timeout) and was cancelled"
                    )
                    result.should_retire = True
                    cancel_requested = True
                    cancel_deadline = time.monotonic() + _CANCEL_DRAIN_SECONDS
                continue

            if kind == "done":
                return True
            if kind == "error":
                if not result.error:
                    result.error = self._format_error("cursor stream failed", payload)
                return False

            idle_deadline = time.monotonic() + self.timeout_seconds
            projection = projector.project(payload)
            if projection.tool_started is not None and self._on_tool_event is not None:
                name, preview, args = projection.tool_started
                try:
                    self._on_tool_event(name, preview, args)
                except Exception:
                    logger.debug("cursor tool-progress hook raised", exc_info=True)
            if projection.messages:
                result.projected_messages.extend(projection.messages)
            if projection.is_tool_iteration:
                result.tool_iterations += 1
            if projection.usage is not None:
                usage = _usage_to_dict(projection.usage)
                if usage:
                    result.token_usage_last = usage
            if not cancel_requested and self._interrupted():
                self._request_run_cancel(run)
                cancel_requested = True
                cancel_deadline = time.monotonic() + _CANCEL_DRAIN_SECONDS
                result.interrupted = True

    def _request_run_cancel(self, run: Any) -> None:
        try:
            if str(getattr(run, "status", "")).lower() in {
                "finished",
                "error",
                "cancelled",
                "expired",
            }:
                return
            run.cancel()
        except Exception:
            logger.debug("cursor run.cancel failed", exc_info=True)

    # ---------- diagnostics ----------

    @staticmethod
    def _format_error(prefix: str, exc: Any = "") -> str:
        detail = str(exc) if exc not in ("", None) else ""
        request_id = getattr(exc, "request_id", None)
        base = f"{prefix}: {detail}" if detail else prefix
        if request_id:
            base = f"{base} (request_id={request_id})"
        return redact_sensitive_text(base, force=True)
