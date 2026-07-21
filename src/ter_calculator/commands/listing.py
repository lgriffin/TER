"""Command implementation module extracted from :mod:`ter_calculator.cli`."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict


class SessionListing(TypedDict):
    path: str
    name: str
    size: int
    modified: float
    subagent_count: int


def _cmd_list(args) -> int:
    """Execute the list subcommand."""
    import json as json_mod
    from ..loader import discover_subagents

    project_path = args.project_path
    if project_path is None:
        home = Path.home()
        claude_dir = home / ".claude" / "projects"
        if not claude_dir.exists():
            print(
                "Error: No Claude Code projects found at ~/.claude/projects/",
                file=sys.stderr,
            )
            return 1
        project_path = str(claude_dir)

    project_dir = Path(project_path)
    if not project_dir.exists():
        print(f"Error: Directory not found: {project_path}", file=sys.stderr)
        return 1

    sessions: list[SessionListing] = []
    for jsonl_file in sorted(
        project_dir.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        if len(sessions) >= args.limit:
            break
        # Skip subagent files — they're shown as counts on their parent.
        if "subagents" in jsonl_file.parts:
            continue
        subagent_count = len(discover_subagents(jsonl_file))
        sessions.append(
            {
                "path": str(jsonl_file),
                "name": jsonl_file.stem,
                "size": jsonl_file.stat().st_size,
                "modified": jsonl_file.stat().st_mtime,
                "subagent_count": subagent_count,
            }
        )

    if args.output_format == "json":
        print(json_mod.dumps(sessions, indent=2))
    else:
        if not sessions:
            print("No sessions found.")
        else:
            print(f"Found {len(sessions)} session(s):\n")
            for i, s in enumerate(sessions, 1):
                size_kb = s["size"] / 1024
                sub_str = (
                    f", {s['subagent_count']} subagents" if s["subagent_count"] else ""
                )
                print(f"  {i}. {s['name']} ({size_kb:.1f} KB{sub_str})")
                print(f"     {s['path']}")

    return 0
