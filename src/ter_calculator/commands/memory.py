"""Repository memory CLI commands for TER v2.0.14."""

from __future__ import annotations

import json
from pathlib import Path

from ..repository_memory import build_index, inspect_index, search_index
from ..closed_loop import analyze_trends, build_effectiveness_dashboard_html
from ..intervention_policy import PolicyConfig
from ..threshold_tuning import (
    describe_config_changes,
    load_tuned_policy_config,
    recommend_policy_config,
    save_tuned_policy_config,
)


def _index_path(args) -> Path:
    if args.index_path:
        return Path(args.index_path)
    root = Path(getattr(args, "root", ".")).resolve()
    return root / ".ter" / "memory-index.json"


def _cmd_memory(args) -> int:
    if args.memory_command == "index":
        result = build_index(args.root, args.output)
    elif args.memory_command == "search":
        result = search_index(
            _index_path(args), args.query, args.limit, args.minimum_score
        )
    elif args.memory_command == "inspect":
        result = inspect_index(_index_path(args))
    elif args.memory_command == "trends":
        root = Path(getattr(args, "root", ".")).resolve()
        lessons_arg = getattr(args, "lessons", None)
        outcomes_arg = getattr(args, "outcomes", None)
        lessons = (
            Path(lessons_arg)
            if lessons_arg
            else root / ".ter" / "session-lessons.jsonl"
        )
        outcomes = (
            Path(outcomes_arg)
            if outcomes_arg
            else root / ".ter" / "intervention-outcomes.jsonl"
        )
        result = analyze_trends(
            lessons, minimum_occurrences=args.minimum_occurrences, outcome_path=outcomes
        )
    elif args.memory_command == "tune":
        root = Path(args.root).resolve()
        lessons = root / ".ter" / "session-lessons.jsonl"
        outcomes = root / ".ter" / "intervention-outcomes.jsonl"
        trends = analyze_trends(lessons, outcome_path=outcomes)
        current = PolicyConfig()
        tuned = recommend_policy_config(
            trends.get("intervention_effectiveness", {}),
            current,
            min_sample_size=args.minimum_samples,
        )
        result = {
            "current": current.__dict__,
            "recommended": tuned.__dict__,
            "changes": describe_config_changes(
                current, tuned, trends.get("intervention_effectiveness", {})
            ),
            "applied": bool(args.apply),
        }
        if args.apply:
            save_tuned_policy_config(root, tuned)
    elif args.memory_command == "dashboard":
        root = Path(args.root).resolve()
        trends = analyze_trends(
            root / ".ter" / "session-lessons.jsonl",
            outcome_path=root / ".ter" / "intervention-outcomes.jsonl",
        )
        output = (
            Path(args.output)
            if args.output
            else root / ".ter" / "effectiveness-dashboard.html"
        )
        current = load_tuned_policy_config(root) or PolicyConfig()
        recommended = recommend_policy_config(
            trends.get("intervention_effectiveness", {}), current
        )
        tuning_preview = {
            "applied_config": (
                load_tuned_policy_config(root).__dict__
                if load_tuned_policy_config(root) is not None
                else None
            ),
            "changes": describe_config_changes(
                current,
                recommended,
                trends.get("intervention_effectiveness", {}),
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            build_effectiveness_dashboard_html(trends, tuning_preview=tuning_preview),
            encoding="utf-8",
        )
        result = {"output": str(output)}
    else:
        raise ValueError("Choose a memory subcommand: index, search, or inspect")

    if args.output_format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.memory_command == "index":
        print(
            f"Indexed {result['file_count']} files and {result['commit_count']} commits into {result['index_path']}"
        )
        print(
            f"Chunks: {result['chunk_count']} | duplicate groups: {result['duplicate_group_count']}"
        )
    elif args.memory_command == "inspect":
        print(f"Repository memory: {result['root']}")
        print(
            f"Chunks: {result['chunk_count']} | files: {result['file_count']} | commits: {result['commit_count']}"
        )
        print(
            f"Duplicate groups: {result['duplicate_group_count']} | semantic groups: {result.get('semantic_duplicate_group_count', 0)}"
        )
    elif args.memory_command == "trends":
        print(f"Recorded lessons: {result['lesson_count']}")
        print(
            f"Estimated ~${result.get('total_estimated_cost_saved_usd', 0):.4f} saved and ~${result.get('total_estimated_cost_wasted_usd', 0):.4f} wasted"
        )
        if not result["scenarios"]:
            print("No recurring scenarios yet.")
        for scenario in result["scenarios"]:
            print(f"WARNING {scenario['message']}")
        for kind, metrics in result.get("intervention_effectiveness", {}).items():
            print(
                f"{kind}: compliance {metrics['compliance_rate']:.0%}, "
                f"improved {metrics['improvement_rate']:.0%}, "
                f"mean TER delta {metrics['mean_ter_delta']:+.3f}"
            )
    elif args.memory_command == "tune":
        print(
            "Policy tuning preview"
            + (" (applied)" if result["applied"] else " (dry-run)")
        )
        for change in result.get("changes", []):
            print(
                f"{change['field']}: {change['old_value']} -> {change['new_value']} "
                f"({change['reason']})"
            )
    elif args.memory_command == "dashboard":
        print(f"Effectiveness dashboard written to {result['output']}")
    else:
        if not result["matches"]:
            print("No relevant repository memory found.")
        for match in result["matches"]:
            location = match["path"]
            if match["start_line"]:
                location += f":{match['start_line']}-{match['end_line']}"
            print(f"{match['score']:.3f}  {location}  [{match['source_type']}]")
            print(f"  {match['excerpt'].splitlines()[0][:180]}")
        for flag in result["risk_flags"]:
            print(f"WARNING {flag['type']}: {flag['path']}")
    return 0
