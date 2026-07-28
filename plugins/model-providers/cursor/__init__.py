"""Cursor provider profile (official cursor-sdk agent runtime).

Cursor sells Composer as an *agent harness*, not a raw chat-completions
endpoint — there is no OpenAI-compatible inference URL to point a transport
at. This profile therefore declares ``api_mode="cursor_agent"``: turns for
``provider: cursor`` are driven through the official ``cursor-sdk`` Python
package by ``agent/cursor_runtime.py`` + ``agent/transports/cursor_sdk_session.py``,
mirroring the codex app-server runtime (the in-tree precedent for an
external agent harness acting as a provider).

The profile itself stays SDK-free on purpose: the model catalog is fetched
over plain HTTPS from the Cloud Agents REST API so that the model picker
never triggers the ~48 MB lazy ``cursor-sdk`` install. The SDK is only
imported when a cursor turn actually runs (see tools/lazy_deps.py,
feature key ``provider.cursor``).

Auth is a plain API key (``CURSOR_API_KEY``) — a user key or service-account
key generated at Cursor Dashboard → Integrations. The SDK has no OAuth flow.
"""

from __future__ import annotations

import json
import logging
import urllib.request

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

CURSOR_API_BASE_URL = "https://api.cursor.com"
CURSOR_MODELS_URL = f"{CURSOR_API_BASE_URL}/v1/models"

# Populated as a side effect of fetch_models(): the /v1/models response also
# carries per-model parameter definitions and preset variants (the same
# ``model.params`` shape the SDK's ModelSelection uses). ``hermes cursor
# models`` and the runtime's params helper read this cache so they can show
# valid parameter ids without a second network round-trip. Best-effort only —
# consumers must tolerate an empty dict.
LAST_MODEL_CATALOG: dict[str, dict] = {}


class CursorProfile(ProviderProfile):
    """Cursor — agent-harness provider driven via the official cursor-sdk."""

    def resolve_model_id(self, value: str) -> str | None:
        """Resolve a live catalog id or alias to its canonical model id."""
        candidate = str(value or "").strip().lower()
        if not candidate:
            return None
        for model_id, item in LAST_MODEL_CATALOG.items():
            if model_id.lower() == candidate:
                return model_id
            aliases = item.get("aliases") if isinstance(item, dict) else None
            if isinstance(aliases, list) and any(
                str(alias).strip().lower() == candidate for alias in aliases
            ):
                return model_id
        return None

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch the recommended model catalog from the Cloud Agents API.

        ``GET https://api.cursor.com/v1/models`` (Bearer auth) returns::

            {"models": [{"id": "composer-2.5", "displayName": ...,
                         "aliases": [...], "parameters": [...],
                         "variants": [...]}, ...]}

        The legacy ``/v0/models`` shape (``{"models": ["id", ...]}``) is
        tolerated defensively. Returns None on any failure so callers fall
        back to ``fallback_models``. Note Cursor documents this catalog as a
        *recommended subset* — the API accepts other model keys too, so the
        picker list is advisory, not a validation gate.
        """
        if not api_key:
            return None

        effective_base = (base_url or CURSOR_API_BASE_URL).rstrip("/")
        url = effective_base + "/v1/models"

        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("fetch_models(cursor): %s", exc)
            return None

        # Current Cloud Agents API returns {"items": [...]}; legacy shapes used
        # {"models": [...]} (v1 objects or v0 bare id strings).
        raw_models = None
        if isinstance(data, dict):
            for key in ("items", "models"):
                candidate = data.get(key)
                if isinstance(candidate, list) and candidate:
                    raw_models = candidate
                    break
        if not isinstance(raw_models, list):
            return None

        ids: list[str] = []
        catalog: dict[str, dict] = {}
        for item in raw_models:
            if isinstance(item, str) and item.strip():
                # Legacy /v0 shape: bare id strings.
                ids.append(item.strip())
            elif isinstance(item, dict) and item.get("id"):
                model_id = str(item["id"]).strip()
                if not model_id:
                    continue
                ids.append(model_id)
                catalog[model_id] = item

        if not ids:
            return None

        if catalog:
            LAST_MODEL_CATALOG.clear()
            LAST_MODEL_CATALOG.update(catalog)
        return ids


cursor = CursorProfile(
    name="cursor",
    aliases=("cursor-sdk", "cursor-agent", "composer"),
    api_mode="cursor_agent",
    env_vars=("CURSOR_API_KEY",),
    base_url=CURSOR_API_BASE_URL,
    auth_type="api_key",
    display_name="Cursor",
    description="Cursor (First-party + API models via Cursor subscription, official SDK)",
    signup_url="https://cursor.com/dashboard?tab=integrations",
    # No OpenAI-compat /models endpoint on the inference path — the catalog
    # probe above uses the Cloud Agents REST API instead, and `hermes doctor`
    # must not probe {base_url}/models.
    supports_health_check=False,
    # Curated fallback shown when the live /v1/models fetch fails. Live fetch
    # is the primary source; keep this list short and load-bearing models only.
    fallback_models=(
        "composer-2.5",
        "composer-2",
        "claude-4.6-sonnet-thinking",
        "gpt-5.3-codex-high",
    ),
)

register_provider(cursor)
