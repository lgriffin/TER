"""Command implementation module extracted from :mod:`ter_calculator.cli`."""

from __future__ import annotations

import sys


def _cmd_hook(args) -> int:
    """Dispatch hook sub-subcommands."""
    hook_cmd = getattr(args, "hook_command", None)
    if not hook_cmd:
        print("Usage: ter hook {monitor}", file=sys.stderr)
        return 1
    if hook_cmd == "monitor":
        return _cmd_hook_monitor(args)
    print(f"Unknown hook command: {hook_cmd}", file=sys.stderr)
    return 1


def _cmd_hook_monitor(args) -> int:
    """Execute the hook monitor: read stdin, process, output JSON."""
    import json as json_mod

    from ..hook_monitor import (
        HookConfig,
        format_guidance,
        format_notification,
        load_state,
        process_tool_event,
        save_state,
    )

    try:
        raw = sys.stdin.read()
        event_data = json_mod.loads(raw)
    except (json_mod.JSONDecodeError, ValueError) as e:
        print(f"Invalid JSON on stdin: {e}", file=sys.stderr)
        return 1

    session_id = event_data.get("session_id", "unknown")

    config = HookConfig(
        min_repetitive_reads=args.min_repetitive_reads,
        min_edit_fragments=args.min_edit_fragments,
        min_repeated_commands=args.min_repeated_commands,
        min_duplicate_calls=args.min_duplicate_calls,
        enable_bash_antipatterns=not args.no_bash_antipatterns,
        state_dir=args.state_dir,
    )

    state = load_state(session_id, config)
    alerts, state = process_tool_event(event_data, state, config)
    save_state(state, config)

    if alerts:
        guidance = format_guidance(alerts)
        notification = format_notification(alerts)
        output: dict[str, str] = {"additionalContext": guidance}
        if notification:
            output["systemMessage"] = notification
        print(json_mod.dumps(output))
    else:
        print(json_mod.dumps({}))

    return 0
