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
    """Execute the unified Phase 3 hook monitor."""
    import json as json_mod

    from ..hook_monitor import (
        HookConfig,
        format_guidance,
        format_notification,
        load_state,
        process_tool_event,
        save_state,
    )
    from ..intervention import process_intervention_event, record_tool_result

    try:
        event_data = json_mod.loads(sys.stdin.read())
    except (json_mod.JSONDecodeError, ValueError) as e:
        print(f"Invalid JSON on stdin: {e}", file=sys.stderr)
        return 1

    session_id = event_data.get("session_id", "unknown")
    config = HookConfig(
        min_repetitive_reads=args.min_repetitive_reads,
        min_edit_fragments=args.min_edit_fragments,
        min_repeated_commands=args.min_repeated_commands,
        min_duplicate_calls=args.min_duplicate_calls,
        min_denied_calls=args.min_denied_calls,
        min_reasoning_loops=args.min_reasoning_loops,
        reasoning_similarity_threshold=args.reasoning_similarity_threshold,
        enable_bash_antipatterns=not args.no_bash_antipatterns,
        state_dir=args.state_dir,
    )
    state = load_state(session_id, config)
    event_name = str(
        event_data.get("hook_event_name", event_data.get("event", "PostToolUse"))
    )

    alerts, state, output = process_intervention_event(event_data, state, config)
    if event_name == "PostToolUse":
        tool_alerts, state = process_tool_event(event_data, state, config)
        alerts.extend(tool_alerts)
        record_tool_result(event_data, state)

    if alerts:
        state.intervention_count += 1
        output.setdefault("additionalContext", format_guidance(alerts))
        output.setdefault("systemMessage", format_notification(alerts))
    save_state(state, config)
    print(json_mod.dumps(output))
    return 0
