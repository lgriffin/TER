"""Repository memory CLI commands for TER v2.0.14."""

from __future__ import annotations

import json
from pathlib import Path

from ..repository_memory import build_index, inspect_index, search_index
from ..closed_loop import analyze_trends


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
        lessons = (
            Path(args.lessons)
            if args.lessons
            else Path(args.root).resolve() / ".ter" / "session-lessons.jsonl"
        )
        result = analyze_trends(lessons, minimum_occurrences=args.minimum_occurrences)
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
