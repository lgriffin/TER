"""Claude Code PostToolUse hook monitor for real-time waste detection.

Runs as a hook handler via ``ter hook monitor``.  Reads hook event JSON
from stdin, tracks tool-usage patterns in a lightweight session state
file, and returns ``{"additionalContext": "..."}`` guidance when waste
patterns are detected.

No heavyweight dependencies (numpy, sentence-transformers) — pure
stdlib so the hook starts in milliseconds.

Hook configuration example for ``.claude/settings.json``::

    {
      "hooks": {
        "PostToolUse": [
          {
            "matcher": "Bash|Read|Edit|Write|Glob|Grep",
            "hooks": [
              {
                "type": "command",
                "command": "ter hook monitor",
                "timeout": 15
              }
            ]
          }
        ]
      }
    }
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "HookConfig",
    "HookSessionState",
    "WasteAlert",
    "check_bash_antipattern",
    "check_duplicate_tool_call",
    "check_edit_fragmentation",
    "check_repeated_command",
    "check_repetitive_read",
    "format_guidance",
    "format_notification",
    "load_state",
    "process_tool_event",
    "save_state",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HookConfig:
    min_repetitive_reads: int = 3
    min_edit_fragments: int = 3
    min_repeated_commands: int = 3
    min_duplicate_calls: int = 2
    enable_bash_antipatterns: bool = True
    state_dir: str | None = None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

MAX_TOOL_CALL_ENTRIES = 500

@dataclass
class HookSessionState:
    session_id: str = ""
    file_read_counts: dict[str, int] = field(default_factory=dict)
    recent_edits: list[str] = field(default_factory=list)
    bash_command_counts: dict[str, int] = field(default_factory=dict)
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    total_events: int = 0


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@dataclass
class WasteAlert:
    pattern_type: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bash anti-pattern regexes (copied from waste.py to avoid heavy imports)
# ---------------------------------------------------------------------------

_BASH_ANTIPATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"(?:^|\|\s*)cat\s+"), "Read",
     "Use the Read tool instead of `cat` for reading files"),
    (re.compile(r"(?:^|\|\s*)head\s+"), "Read",
     "Use the Read tool with offset/limit instead of `head`"),
    (re.compile(r"(?:^|\|\s*)tail\s+"), "Read",
     "Use the Read tool with offset/limit instead of `tail`"),
    (re.compile(r"(?:^|\|\s*)grep\s+"), "Grep",
     "Use the Grep tool instead of `grep` for searching"),
    (re.compile(r"(?:^|\|\s*)rg\s+"), "Grep",
     "Use the Grep tool instead of `rg` for searching"),
    (re.compile(r"^find\s+"), "Glob",
     "Use the Glob tool instead of `find` for finding files"),
]


def _normalize_bash_command(cmd: str) -> str:
    """Normalize a bash command for deduplication (from waste.py)."""
    cmd = cmd.strip()
    cmd = re.sub(r'\s*\|\s*(tail|head)\s+-\d+\s*$', '', cmd)
    cmd = re.sub(r'\s+', ' ', cmd)
    return cmd


def _compute_tool_signature(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Stable hash for a tool call to detect duplicates."""
    raw = f"{tool_name}:{json.dumps(tool_input, sort_keys=True)}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Individual checkers
# ---------------------------------------------------------------------------

def check_bash_antipattern(
    tool_name: str,
    tool_input: dict[str, Any],
) -> WasteAlert | None:
    if tool_name != "Bash":
        return None
    cmd = tool_input.get("command", "").strip()
    if not cmd:
        return None
    for pattern, recommended_tool, guidance in _BASH_ANTIPATTERNS:
        if pattern.search(cmd):
            short_cmd = cmd[:80] + "..." if len(cmd) > 80 else cmd
            return WasteAlert(
                pattern_type="bash_antipattern",
                severity="info",
                message=(
                    f"Bash anti-pattern: `{short_cmd}`. "
                    f"{guidance} — it is faster and avoids injecting "
                    f"raw terminal output into context."
                ),
                details={"command": cmd, "recommended_tool": recommended_tool},
            )
    return None


def check_repetitive_read(
    tool_name: str,
    tool_input: dict[str, Any],
    state: HookSessionState,
    threshold: int = 3,
) -> WasteAlert | None:
    if tool_name != "Read":
        return None
    file_path = tool_input.get("file_path", "")
    if not file_path:
        return None
    normalized = os.path.normpath(file_path)
    state.file_read_counts[normalized] = state.file_read_counts.get(normalized, 0) + 1
    count = state.file_read_counts[normalized]
    if count >= threshold:
        short_path = os.path.basename(normalized)
        return WasteAlert(
            pattern_type="repetitive_read",
            severity="warning" if count >= threshold + 2 else "info",
            message=(
                f"File '{short_path}' has been read {count} times this session. "
                f"Its content is likely already in your context — avoid re-reading "
                f"files unless they have been modified since your last read."
            ),
            details={"file_path": normalized, "read_count": count},
        )
    return None


def check_edit_fragmentation(
    tool_name: str,
    tool_input: dict[str, Any],
    state: HookSessionState,
    threshold: int = 3,
) -> WasteAlert | None:
    if tool_name not in ("Edit", "Write"):
        state.recent_edits.clear()
        return None

    file_path = tool_input.get("file_path", "")
    if not file_path:
        return None
    normalized = os.path.normpath(file_path)

    state.recent_edits.append(normalized)
    if len(state.recent_edits) > 20:
        state.recent_edits = state.recent_edits[-20:]

    consecutive = 0
    for path in reversed(state.recent_edits):
        if path == normalized:
            consecutive += 1
        else:
            break

    if consecutive >= threshold:
        short_path = os.path.basename(normalized)
        return WasteAlert(
            pattern_type="edit_fragmentation",
            severity="warning",
            message=(
                f"{consecutive} consecutive edits to '{short_path}'. "
                f"Try to batch multiple changes into fewer Edit calls "
                f"to reduce round-trips and context churn."
            ),
            details={"file_path": normalized, "consecutive_edits": consecutive},
        )
    return None


def check_duplicate_tool_call(
    tool_name: str,
    tool_input: dict[str, Any],
    state: HookSessionState,
    threshold: int = 2,
) -> WasteAlert | None:
    sig = _compute_tool_signature(tool_name, tool_input)
    state.tool_call_counts[sig] = state.tool_call_counts.get(sig, 0) + 1

    if len(state.tool_call_counts) > MAX_TOOL_CALL_ENTRIES:
        sorted_sigs = sorted(state.tool_call_counts, key=state.tool_call_counts.get)  # type: ignore[arg-type]
        for s in sorted_sigs[:100]:
            del state.tool_call_counts[s]

    count = state.tool_call_counts[sig]
    if count >= threshold:
        return WasteAlert(
            pattern_type="duplicate_tool_call",
            severity="warning",
            message=(
                f"Duplicate {tool_name} call (identical parameters, "
                f"invocation #{count}). The result is already in your "
                f"context from a previous call."
            ),
            details={
                "tool_name": tool_name,
                "invocation_count": count,
                "signature": sig,
            },
        )
    return None


def check_repeated_command(
    tool_name: str,
    tool_input: dict[str, Any],
    state: HookSessionState,
    threshold: int = 3,
) -> WasteAlert | None:
    if tool_name != "Bash":
        return None
    cmd = tool_input.get("command", "")
    if not cmd:
        return None
    normalized = _normalize_bash_command(cmd)
    if not normalized:
        return None
    state.bash_command_counts[normalized] = (
        state.bash_command_counts.get(normalized, 0) + 1
    )
    count = state.bash_command_counts[normalized]
    if count >= threshold:
        short_cmd = normalized[:60] + "..." if len(normalized) > 60 else normalized
        return WasteAlert(
            pattern_type="repeated_command",
            severity="warning",
            message=(
                f"Command `{short_cmd}` has been run {count} times. "
                f"If re-running to verify a fix, consider whether the "
                f"approach needs to change rather than retrying."
            ),
            details={"command": normalized, "run_count": count},
        )
    return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_tool_event(
    event_data: dict[str, Any],
    state: HookSessionState,
    config: HookConfig | None = None,
) -> tuple[list[WasteAlert], HookSessionState]:
    cfg = config or HookConfig()

    tool_name = event_data.get("tool_name", "")
    tool_input = event_data.get("tool_input", {})
    if not isinstance(tool_input, dict):
        tool_input = {}

    state.total_events += 1
    alerts: list[WasteAlert] = []

    if cfg.enable_bash_antipatterns:
        alert = check_bash_antipattern(tool_name, tool_input)
        if alert:
            alerts.append(alert)

    alert = check_repetitive_read(
        tool_name, tool_input, state, cfg.min_repetitive_reads,
    )
    if alert:
        alerts.append(alert)

    alert = check_edit_fragmentation(
        tool_name, tool_input, state, cfg.min_edit_fragments,
    )
    if alert:
        alerts.append(alert)

    alert = check_duplicate_tool_call(
        tool_name, tool_input, state, cfg.min_duplicate_calls,
    )
    if alert:
        alerts.append(alert)

    alert = check_repeated_command(
        tool_name, tool_input, state, cfg.min_repeated_commands,
    )
    if alert:
        alerts.append(alert)

    return alerts, state


# ---------------------------------------------------------------------------
# Guidance formatting
# ---------------------------------------------------------------------------

def format_guidance(alerts: list[WasteAlert]) -> str:
    if not alerts:
        return ""
    lines = ["[TER Waste Monitor]"]
    for i, alert in enumerate(alerts, 1):
        prefix = f"  {i}. " if len(alerts) > 1 else "  "
        lines.append(f"{prefix}{alert.message}")
    return "\n".join(lines)


_SEVERITY_ICON = {"info": "~", "warning": "!", "error": "!!"}


def format_notification(alerts: list[WasteAlert]) -> str:
    if not alerts:
        return ""
    parts: list[str] = []
    for alert in alerts:
        icon = _SEVERITY_ICON.get(alert.severity, "~")
        short = alert.pattern_type.replace("_", " ")
        parts.append(f"[{icon}] {short}")
    return f"TER: {', '.join(parts)}"


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _get_state_dir(config: HookConfig | None = None) -> Path:
    if config and config.state_dir:
        return Path(config.state_dir)
    return Path(tempfile.gettempdir()) / "ter-hooks"


def load_state(
    session_id: str,
    config: HookConfig | None = None,
) -> HookSessionState:
    state_dir = _get_state_dir(config)
    state_file = state_dir / f"{session_id}.json"
    if state_file.exists():
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            return HookSessionState(**data)
        except (json.JSONDecodeError, TypeError):
            pass
    return HookSessionState(session_id=session_id)


def save_state(
    state: HookSessionState,
    config: HookConfig | None = None,
) -> None:
    state_dir = _get_state_dir(config)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / f"{state.session_id}.json"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(state_dir), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f)
        os.replace(tmp_path, str(state_file))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
