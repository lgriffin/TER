"""Claude Code native tool names mapped to harness-independent tool kinds.

This table is data: when Claude Code adds or renames a tool, only this table
changes. Unknown tools map to ``ToolKind.OTHER`` and stay visible in reports.

It lives beside, not inside, the Claude Code adapters because two of them
share it: the driven JSONL ``SessionSource`` and the driving hooks adapter.
Keeping it here means the hook entry point does not import the JSONL reader
(and, through it, the TER 3 loader and numpy) on every hook invocation.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..domain.events import ToolKind

CLAUDE_CODE_TOOL_KINDS: Mapping[str, ToolKind] = {
    "Read": ToolKind.FS_READ,
    "NotebookRead": ToolKind.FS_READ,
    "Grep": ToolKind.FS_SEARCH,
    "Glob": ToolKind.FS_SEARCH,
    "LS": ToolKind.FS_SEARCH,
    "Edit": ToolKind.FS_EDIT,
    "MultiEdit": ToolKind.FS_EDIT,
    "NotebookEdit": ToolKind.FS_EDIT,
    "Write": ToolKind.FS_WRITE,
    "Bash": ToolKind.EXEC_SHELL,
    # Claude Code's shell tool on Windows. Real sessions on a Windows PC ran
    # 2,880 PowerShell calls in 30 of 47 sessions (25 Nov 2026), checks
    # included; left unmapped they were invisible to every shell rule.
    "PowerShell": ToolKind.EXEC_SHELL,
    "BashOutput": ToolKind.EXEC_SHELL,
    "KillShell": ToolKind.EXEC_SHELL,
    "WebFetch": ToolKind.NET_FETCH,
    "WebSearch": ToolKind.NET_FETCH,
    "Task": ToolKind.AGENT_HANDOFF,
    "Agent": ToolKind.AGENT_HANDOFF,
    "TodoWrite": ToolKind.PLAN,
    "ExitPlanMode": ToolKind.PLAN,
}


def tool_kind(native_name: str | None) -> ToolKind:
    """Return the neutral kind for a Claude Code tool name."""
    if not native_name:
        return ToolKind.OTHER
    return CLAUDE_CODE_TOOL_KINDS.get(native_name, ToolKind.OTHER)
