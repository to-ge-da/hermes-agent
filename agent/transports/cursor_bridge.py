"""Safe launch wrapper for the cursor-sdk bridge subprocess."""

from __future__ import annotations

import importlib
import threading
from typing import Any

from tools.environments.local import hermes_subprocess_env

_BRIDGE_LAUNCH_LOCK = threading.Lock()


def launch_cursor_bridge(sdk: Any, *, workspace: str) -> Any:
    """Launch cursor-sdk with Hermes' sanitized subprocess environment.

    cursor-sdk 0.1.9 copies ``os.environ`` internally and exposes no ``env=``
    argument. Patch its private environment builder only for the duration of
    the synchronized launch. This does not mutate process-global
    ``os.environ`` and is isolated here so it can be removed when the SDK
    exposes a public environment override.

    Test doubles and older SDKs without the pinned private module retain the
    plain launch path.
    """
    client_cls = getattr(sdk, "CursorClient", None) or getattr(sdk, "Client")
    module_name = getattr(sdk, "__name__", "")
    if not module_name:
        return client_cls.launch_bridge(workspace=workspace)

    try:
        bridge_module = importlib.import_module(f"{module_name}._bridge")
        original_env_builder = bridge_module._bridge_subprocess_env
    except (AttributeError, ImportError) as exc:
        if module_name == "cursor_sdk":
            raise RuntimeError(
                "cursor-sdk no longer exposes the bridge environment hook; "
                "refusing to launch with unsanitized Hermes secrets"
            ) from exc
        return client_cls.launch_bridge(workspace=workspace)

    sanitized_env = hermes_subprocess_env(inherit_credentials=False)
    sanitized_env.setdefault("CURSOR_SDK_CLIENT_LANGUAGE", "python")

    with _BRIDGE_LAUNCH_LOCK:
        bridge_module._bridge_subprocess_env = lambda: dict(sanitized_env)
        try:
            return client_cls.launch_bridge(
                workspace=workspace,
                # Run-scoped SDK RPCs (wait/cancel/conversation) carry only a
                # run id and 0.1.9 rejects them when this client-side guard is
                # disabled. The bridge environment is sanitized above, so
                # allowing the SDK's owned-bridge path cannot expose or fall
                # back to a process-environment API key; agent/get-run calls
                # still pass credentials explicitly.
                allow_api_key_env_fallback=True,
            )
        finally:
            bridge_module._bridge_subprocess_env = original_env_builder
