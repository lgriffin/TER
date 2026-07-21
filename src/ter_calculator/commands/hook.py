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
    from ..closed_loop import append_lessons, record_outcome, resolve_project_root
    from ..threshold_tuning import load_tuned_policy_config

    try:
        event_data = json_mod.loads(sys.stdin.read())
    except (json_mod.JSONDecodeError, ValueError) as e:
        print(f"Invalid JSON on stdin: {e}", file=sys.stderr)
        return 1

    session_id = event_data.get("session_id", "unknown")
    root = resolve_project_root(event_data)
    tuned = load_tuned_policy_config(root)

    def selected(name, default):
        value = getattr(args, name, None)
        if value is not None:
            return value
        if tuned is not None:
            return getattr(tuned, name)
        return default

    config = HookConfig(
        min_repetitive_reads=args.min_repetitive_reads,
        min_edit_fragments=args.min_edit_fragments,
        min_repeated_commands=args.min_repeated_commands,
        min_duplicate_calls=args.min_duplicate_calls,
        min_denied_calls=args.min_denied_calls,
        min_reasoning_loops=args.min_reasoning_loops,
        reasoning_similarity_threshold=args.reasoning_similarity_threshold,
        enable_bash_antipatterns=not args.no_bash_antipatterns,
        enable_project_memory=not args.no_project_memory,
        memory_index=args.memory_index,
        memory_limit=args.memory_limit,
        memory_minimum_score=args.memory_minimum_score,
        lesson_store=args.lesson_store,
        outcome_store=args.outcome_store,
        policy_mode=args.policy_mode,
        ter_drop_warning=selected("ter_drop_warning", 0.12),
        ter_drop_replan=selected("ter_drop_replan", 0.20),
        waste_ratio_warning=selected("waste_ratio_warning", 0.25),
        waste_ratio_replan=selected("waste_ratio_replan", 0.40),
        degraded_windows_required=selected("degraded_windows_required", 3),
        refresh_cooldown_seconds=selected("refresh_cooldown_seconds", 120),
        replan_cooldown_seconds=selected("replan_cooldown_seconds", 180),
        state_dir=args.state_dir,
    )
    state = load_state(session_id, config)
    event_name = str(
        event_data.get("hook_event_name", event_data.get("event", "PostToolUse"))
    )

    for key in ("ter_metrics", "metrics", "ter_signal"):
        if isinstance(event_data.get(key), dict):
            event_data[key].setdefault("cost_per_1k_tokens", args.cost_per_1k_tokens)
    if not any(key in event_data for key in ("ter_metrics", "metrics", "ter_signal")):
        try:
            from ..transcript_metrics import derive_transcript_metrics

            derived_metrics = derive_transcript_metrics(event_data, state)
            if derived_metrics is not None:
                derived_metrics["cost_per_1k_tokens"] = args.cost_per_1k_tokens
                event_data["ter_metrics"] = derived_metrics
        except Exception:
            # Hook execution must never fail because transcript-derived metrics
            # are unavailable or a transcript is temporarily incomplete.
            pass

    alerts, state, output = process_intervention_event(event_data, state, config)
    if event_name == "PostToolUse":
        tool_alerts, state = process_tool_event(event_data, state, config)
        alerts.extend(tool_alerts)
        record_tool_result(event_data, state)

    if alerts:
        state.intervention_count += 1
        guidance = format_guidance(alerts)
        if output.get("additionalContext"):
            output["additionalContext"] += "\n\n" + guidance
        else:
            output["additionalContext"] = guidance
        output.setdefault("systemMessage", format_notification(alerts))
        root = resolve_project_root(event_data)
        lesson_store = config.lesson_store or str(
            root / ".ter" / "session-lessons.jsonl"
        )
        append_lessons(
            lesson_store, session_id=session_id, repository=str(root), alerts=alerts
        )
        outcome_store = config.outcome_store or str(
            root / ".ter" / "intervention-outcomes.jsonl"
        )
        for alert in alerts:
            record_outcome(
                outcome_store,
                session_id=session_id,
                intervention_type=alert.pattern_type,
                outcome="issued",
                details=alert.details,
            )
    save_state(state, config)
    print(json_mod.dumps(output))
    return 0
