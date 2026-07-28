"""Parser for the ``hermes cursor`` subcommand (Cursor cloud agents + catalog)."""

from __future__ import annotations

from typing import Callable


def build_cursor_parser(subparsers, *, cmd_cursor: Callable) -> None:
    """Attach the ``cursor`` subcommand to ``subparsers``."""
    cursor_parser = subparsers.add_parser(
        "cursor",
        help="Manage Cursor cloud agents and the Cursor model catalog",
        description=(
            "Drive Cursor's cloud agents and inspect the model catalog via the "
            "official cursor-sdk. Requires CURSOR_API_KEY (Cursor Dashboard → "
            "Integrations). To use Cursor models as your PRIMARY chat models, "
            "run `hermes model` and pick Cursor — this subcommand covers the "
            "cloud-delegation surface."
        ),
    )
    cursor_sub = cursor_parser.add_subparsers(dest="cursor_action")

    cursor_sub.add_parser(
        "models", help="List Cursor models, parameters, and variants"
    )
    cursor_sub.add_parser("me", help="Validate CURSOR_API_KEY and show the account")
    cursor_sub.add_parser(
        "repos", help="List repositories connected for cloud agents"
    )

    launch = cursor_sub.add_parser(
        "launch", help="Launch a Cursor cloud agent on a repository"
    )
    launch.add_argument("prompt", help="Task prompt for the cloud agent")
    launch.add_argument("--repo", default="", help="Repository URL to clone into the VM")
    launch.add_argument("--ref", default="", help="Starting ref/branch (default: repo default)")
    launch.add_argument("--model", default="", help="Model id (default: account default)")
    launch.add_argument("--name", default="", help="Human-readable agent name")
    launch.add_argument("--pr", action="store_true", help="Open a PR when the run finishes")
    launch.add_argument(
        "--branch-current", action="store_true",
        help="Push to the existing branch instead of a new one",
    )
    launch.add_argument(
        "--pool", default="", help="Self-hosted pool name (default: Cursor-hosted VMs)"
    )
    launch.add_argument(
        "--env-var", action="append", default=[], metavar="KEY=VALUE",
        help="Session-scoped env var injected into the VM (repeatable)",
    )
    launch.add_argument(
        "--follow", action="store_true", help="Stream the run output until it finishes"
    )

    list_parser = cursor_sub.add_parser(
        "list", aliases=["ls"], help="List cloud agents"
    )
    list_parser.add_argument(
        "--archived", action="store_true", help="Include archived agents"
    )

    status = cursor_sub.add_parser("status", help="Show one agent + recent runs")
    status.add_argument("agent_id", help="Cloud agent id (bc-...)")

    follow = cursor_sub.add_parser("follow", help="Stream a running agent's events")
    follow.add_argument("agent_id", help="Cloud agent id (bc-...)")

    send = cursor_sub.add_parser("send", help="Send a follow-up prompt to an agent")
    send.add_argument("agent_id", help="Cloud agent id (bc-...)")
    send.add_argument("prompt", help="Follow-up prompt")
    send.add_argument(
        "--follow", action="store_true", help="Stream the run output until it finishes"
    )

    cancel = cursor_sub.add_parser("cancel", help="Cancel an agent's active run")
    cancel.add_argument("agent_id", help="Cloud agent id (bc-...)")

    artifacts = cursor_sub.add_parser(
        "artifacts", help="List (and optionally download) an agent's artifacts"
    )
    artifacts.add_argument("agent_id", help="Cloud agent id (bc-...)")
    artifacts.add_argument(
        "--download", default="", metavar="DIR",
        help="Download all artifacts into DIR",
    )

    archive = cursor_sub.add_parser("archive", help="Archive an agent (reversible)")
    archive.add_argument("agent_id", help="Cloud agent id (bc-...)")

    unarchive = cursor_sub.add_parser("unarchive", help="Restore an archived agent")
    unarchive.add_argument("agent_id", help="Cloud agent id (bc-...)")

    delete = cursor_sub.add_parser("delete", help="Permanently delete an agent")
    delete.add_argument("agent_id", help="Cloud agent id (bc-...)")
    delete.add_argument("--yes", action="store_true", help="Confirm permanent deletion")

    cursor_parser.set_defaults(func=cmd_cursor)
