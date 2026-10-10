"""Countermeasures: what to change so the next session wastes less.

Each countermeasure answers findings of one detector and is built from those
findings (the files, commands and subjects they cite), never from a fixed
token threshold (point 139). Actions are concrete: lines to add to
``CLAUDE.md``, Claude Code hooks to install (with the settings snippet and
script), harness settings, and working practices. A countermeasure answering
only uncertain findings says so, and asks for verification first; their
cost still counts, since uncertain waste is waste until verified.

The hook scripts are examples to adapt: they read the hook payload with
``jq`` and use only documented Claude Code hook behaviour (exit code 2 blocks
a ``PreToolUse`` call, or feeds stderr back to the agent after a
``PostToolUse`` call).
"""

from __future__ import annotations

import json
import re
import shlex
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .model import Finding, LeanWaste, Step

__all__ = [
    "Action",
    "ActionKind",
    "Countermeasure",
    "FollowUp",
    "build_countermeasures",
    "follow_ups",
]


class ActionKind(StrEnum):
    CLAUDE_MD = "claude_md"
    HOOK = "hook"
    SETTING = "setting"
    PRACTICE = "practice"

    @property
    def label(self) -> str:
        return {
            ActionKind.CLAUDE_MD: "Add to CLAUDE.md",
            ActionKind.HOOK: "Install a hook",
            ActionKind.SETTING: "Harness setting",
            ActionKind.PRACTICE: "Practice",
        }[self]


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    text: str
    snippet: str | None = None
    language: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "text": self.text,
            "snippet": self.snippet,
            "language": self.language,
        }


@dataclass(frozen=True)
class Countermeasure:
    detector: str
    waste: LeanWaste
    title: str
    rationale: str
    addresses: tuple[str, ...]
    uncertain: bool
    actions: tuple[Action, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "detector": self.detector,
            "waste": self.waste.value,
            "title": self.title,
            "rationale": self.rationale,
            "addresses": list(self.addresses),
            "uncertain": self.uncertain,
            "actions": [a.as_dict() for a in self.actions],
        }


@dataclass(frozen=True)
class FollowUp:
    """What to measure on the next run to know the countermeasure worked."""

    metric: str
    current: str
    target: str
    how: str

    def as_dict(self) -> dict[str, str]:
        return {
            "metric": self.metric,
            "current": self.current,
            "target": self.target,
            "how": self.how,
        }


@dataclass(frozen=True)
class _Context:
    findings: tuple[Finding, ...]
    steps: tuple[Step, ...]

    @property
    def subjects(self) -> list[str]:
        return list(dict.fromkeys(f.subject for f in self.findings if f.subject))

    @property
    def validation_command(self) -> str | None:
        runs = Counter(s.command for s in self.steps if s.is_validation and s.command)
        return runs.most_common(1)[0][0] if runs else None


# A command of plain words needs no quoting to run inside a hook.
_PLAIN_COMMAND = re.compile(r"[\w./:=@%+,-]+(?: [\w./:=@%+,-]+)*")


def _settings(hooks: dict[str, object]) -> str:
    return json.dumps({"hooks": hooks}, indent=2)


def _script_hook(event: str, matcher: str, script: str) -> str:
    return _settings(
        {
            event: [
                {
                    "matcher": matcher,
                    "hooks": [
                        {
                            "type": "command",
                            "command": f'"$CLAUDE_PROJECT_DIR"/.claude/hooks/{script}',
                        }
                    ],
                }
            ]
        }
    )


def _list(items: Sequence[str], limit: int = 4) -> str:
    shown = [f"`{i}`" for i in items[:limit]]
    more = len(items) - limit
    return ", ".join(shown) + (f" and {more} more" if more > 0 else "")


_REREAD_SCRIPT = """#!/usr/bin/env bash
# .claude/hooks/no-reread.sh: PreToolUse(Read). Blocks re-reading a file whose
# contents have not changed since this session last read it.
input=$(cat)
session=$(jq -r '.session_id' <<<"$input")
file=$(jq -r '.tool_input.file_path // empty' <<<"$input")
range=$(jq -c '[.tool_input.offset, .tool_input.limit]' <<<"$input")
[ -n "$file" ] && [ -f "$file" ] || exit 0
seen="${TMPDIR:-/tmp}/claude-reads-$session"
key="$file $range $(sha1sum "$file" | cut -c1-16)"
if grep -qxF "$key" "$seen" 2>/dev/null; then
  echo "$file is unchanged since you read it; use the copy in context." >&2
  exit 2
fi
echo "$key" >> "$seen"
"""

_REPEAT_SCRIPT = """#!/usr/bin/env bash
# .claude/hooks/no-repeat.sh: PreToolUse(Bash). Blocks re-running an identical
# command while the working tree is unchanged since its last run.
input=$(cat)
session=$(jq -r '.session_id' <<<"$input")
command=$(jq -r '.tool_input.command // empty' <<<"$input")
[ -n "$command" ] || exit 0
state=$( (git diff; git status --porcelain) 2>/dev/null | sha1sum | cut -c1-16)
key="$(printf '%s' "$command" | sha1sum | cut -c1-16) $state"
seen="${TMPDIR:-/tmp}/claude-commands-$session"
if grep -qxF "$key" "$seen" 2>/dev/null; then
  echo "Nothing changed since you last ran this command; reuse its output." >&2
  exit 2
fi
echo "$key" >> "$seen"
"""

_SAME_FAILURE_SCRIPT = """#!/usr/bin/env bash
# .claude/hooks/same-failure.sh: PostToolUse(Bash). When a check fails exactly
# as it failed last time, tell the agent to stop patching and re-diagnose.
input=$(cat)
session=$(jq -r '.session_id' <<<"$input")
output=$(jq -r '.tool_response | tostring' <<<"$input")
grep -qE 'FAILED|[0-9]+ failed|Traceback|error:' <<<"$output" || exit 0
sig=$(grep -E 'FAILED|Error|error:' <<<"$output" | sed -E 's/[0-9.]+s//g' | sort -u | sha1sum | cut -c1-16)
last="${TMPDIR:-/tmp}/claude-failure-$session"
if [ "$(cat "$last" 2>/dev/null)" = "$sig" ]; then
  echo "Same failure as the previous run: the last edit did not change it." \\
       "Re-read the failure and state its root cause before editing again." >&2
  exit 2
fi
echo "$sig" > "$last"
"""

_NO_WRITE_SCRIPT = """#!/usr/bin/env bash
# .claude/hooks/edit-not-write.sh: PreToolUse(Write). Blocks whole-file rewrites
# of files that already exist, so changes go through Edit.
file=$(jq -r '.tool_input.file_path // empty')
if [ -n "$file" ] && [ -f "$file" ]; then
  echo "$file exists: change it with Edit instead of rewriting it." >&2
  exit 2
fi
"""


def _repeated_exploration(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Keep what was read; re-read only what changed",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Tell the agent to reuse file contents it already has.",
                "- Do not re-read a file or re-run a search whose result is already in "
                "context; re-read only after the file was edited or to see a different range.",
                "markdown",
            ),
            Action(
                ActionKind.CLAUDE_MD,
                "Name the files this task keeps needing, so they are found once.",
                "## Where things live\n"
                + "\n".join(f"- `{s}`: <what it holds>" for s in ctx.subjects[:6]),
                "markdown",
            ),
            Action(
                ActionKind.HOOK,
                f"Block re-reads of unchanged files ({_list(ctx.subjects)} were re-read). "
                "Save the script as .claude/hooks/no-reread.sh, make it executable, and "
                "register it in .claude/settings.json.",
                _script_hook("PreToolUse", "Read", "no-reread.sh")
                + "\n\n"
                + _REREAD_SCRIPT,
                "json+bash",
            ),
        ],
    )


def _repeated_tool_call(ctx: _Context) -> tuple[str, list[Action]]:
    validation = any("validation" in f.title for f in ctx.findings)
    actions = [
        Action(
            ActionKind.CLAUDE_MD,
            "Ask for reuse of results that cannot have changed.",
            "- Do not repeat a tool call with the same input unless something it "
            "depends on has changed since; reuse the earlier result.",
            "markdown",
        ),
        Action(
            ActionKind.HOOK,
            f"Block identical shell commands while the working tree is unchanged "
            f"(repeated: {_list(ctx.subjects)}).",
            _script_hook("PreToolUse", "Bash", "no-repeat.sh")
            + "\n\n"
            + _REPEAT_SCRIPT,
            "json+bash",
        ),
    ]
    if validation:
        actions.insert(
            1,
            Action(
                ActionKind.CLAUDE_MD,
                "Re-run checks only after a change.",
                "- Re-run tests or linters only after editing code since their last run.",
                "markdown",
            ),
        )
    return "Reuse results instead of repeating calls", actions


def _rework_cycle(ctx: _Context) -> tuple[str, list[Action]]:
    command = ctx.validation_command or "<test command>"
    return (
        "Re-diagnose after an identical failure instead of patching again",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Make an unchanged failure a stop signal.",
                f"- If `{command}` fails the same way after a fix, stop editing: re-read "
                "the full failure, state the root cause in one sentence, then change approach.",
                "markdown",
            ),
            Action(
                ActionKind.PRACTICE,
                "Run the failing test alone with full output before the next edit, "
                f"e.g. `{command} -x -vv` for pytest, so the diagnosis rests on the whole assertion.",
            ),
            Action(
                ActionKind.HOOK,
                "Tell the agent when a check fails exactly as before.",
                _script_hook("PostToolUse", "Bash", "same-failure.sh")
                + "\n\n"
                + _SAME_FAILURE_SCRIPT,
                "json+bash",
            ),
        ],
    )


def _unvalidated(ctx: _Context) -> tuple[str, list[Action]]:
    command = ctx.validation_command
    title = "Validate every change before reporting it"
    edited = f"(edited without validation: {_list(ctx.subjects)})"
    if command is None:
        # No check ran in this session, so there is no command to put in a
        # hook: a placeholder there would be executed as shell code.
        return (
            title,
            [
                Action(
                    ActionKind.CLAUDE_MD,
                    "Make validation part of done, and name the check to run.",
                    "## Validation\n"
                    "- Test command: <fill in the project's test command>\n"
                    "- After changing code, run the test command and report its result "
                    "before saying the task is done; say so explicitly if it cannot be run.",
                    "markdown",
                ),
                Action(
                    ActionKind.PRACTICE,
                    "No check ran in this session, so no edit hook is generated. Once the "
                    "project has a test command, add a PostToolUse hook on "
                    f"Edit|Write that runs it {edited}.",
                ),
            ],
        )
    # Both the command and the diagnostic are quoted as shell words, so
    # quotes, ``;``, ``#`` or newlines in the observed command can neither
    # break the hook nor detach the failure branch from the check.
    run = (
        command
        if _PLAIN_COMMAND.fullmatch(command)
        else f"bash -c {shlex.quote(command)}"
    )
    message = shlex.quote(f"{command} fails after this edit")
    check = (
        f'cd "$CLAUDE_PROJECT_DIR" && {run} >/dev/null 2>&1 '
        f"|| {{ echo {message} >&2; exit 2; }}"
    )
    hook = _settings(
        {
            "PostToolUse": [
                {
                    "matcher": "Edit|Write",
                    "hooks": [{"type": "command", "command": check, "timeout": 300}],
                }
            ]
        }
    )
    return (
        title,
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Make validation part of done.",
                f"- After changing code, run `{command}` and report its result before saying the "
                "task is done; say so explicitly if it cannot be run.",
                "markdown",
            ),
            Action(
                ActionKind.HOOK,
                "Run the check after every edit and feed a failure back to the agent "
                f"{edited}.",
                hook,
                "json",
            ),
        ],
    )


def _premature(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Collect evidence before changing code",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Require a look before a change.",
                "- Read a file (or the relevant range) before editing it. Before creating a "
                "file, read one neighbouring file to follow its conventions.",
                "markdown",
            ),
            Action(
                ActionKind.SETTING,
                "Start unfamiliar tasks in plan mode, so exploration comes before edits "
                f"(changed before reading: {_list(ctx.subjects)}).",
                json.dumps({"permissions": {"defaultMode": "plan"}}, indent=2),
                "json",
            ),
        ],
    )


def _planning(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Turn plans into the first action sooner",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Bound planning by action, not by length.",
                "- After planning, take the first concrete action (read, edit or run) "
                "before refining the plan; update the to-do list when a step completes.",
                "markdown",
            ),
            Action(
                ActionKind.PRACTICE,
                "For multi-step work, approve a short plan in plan mode once, then let the "
                "agent execute without re-planning between steps.",
            ),
        ],
    )


def _fragmented(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Batch edits to one file",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Ask for one round trip per coherent change.",
                "- Plan the whole change to a file before editing it, then send its "
                "Edit calls together in one turn (parallel tool calls) rather than "
                "one hunk per turn, waiting for each result "
                f"(fragmented: {_list(ctx.subjects)}).",
                "markdown",
            )
        ],
    )


def _unused(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Read with a purpose",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Ask the agent to justify reads, and give it a map so it needs fewer.",
                "- Before opening a file, name what you expect to learn from it; skip files "
                "the task and the code you have read do not point to.",
                "markdown",
            ),
            Action(
                ActionKind.PRACTICE,
                f"Verify first: {_list(ctx.subjects)} may have been read to rule something "
                "out. If they are never relevant to this kind of task, say so in CLAUDE.md, "
                "or run /compact after exploration so unused context stops being carried.",
            ),
        ],
    )


def _handoff(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Delegate only what will not be redone",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Say when subagents are worth their round trip.",
                "- Use a subagent only for broad searches or independent parallel work. "
                "Do small lookups directly, and rely on a subagent's result instead of redoing it.",
                "markdown",
            ),
            Action(
                ActionKind.SETTING,
                f"For small, focused tasks, turn off delegation (delegated then redone: "
                f"{_list(ctx.subjects)}).",
                json.dumps({"permissions": {"deny": ["Task"]}}, indent=2),
                "json",
            ),
        ],
    )


def _reasoning(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Act on a decision instead of restating it",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Discourage re-deriving the same plan.",
                "- When you notice you are restating an earlier plan, act on it; re-plan "
                "only when new evidence contradicts it.",
                "markdown",
            )
        ],
    )


def _regeneration(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Edit in place instead of rewriting files",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Reserve whole-file writes for new files.",
                "- Change existing files with Edit; use Write only for new files "
                "or rewrites the user asked for.",
                "markdown",
            ),
            Action(
                ActionKind.HOOK,
                f"Block whole-file rewrites of existing files (rewritten: {_list(ctx.subjects)}).",
                _script_hook("PreToolUse", "Write", "edit-not-write.sh")
                + "\n\n"
                + _NO_WRITE_SCRIPT,
                "json+bash",
            ),
        ],
    )


def _drift(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Stay inside the stated intent",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Ask the agent to propose extra work instead of doing it.",
                "- Do only what the current request asks. When you see something else worth "
                "doing, mention it in your answer and wait for the go-ahead; when the user "
                "changes the goal, drop the old one.",
                "markdown",
            ),
            Action(
                ActionKind.PRACTICE,
                f"Unrequested changes this session: {_list(ctx.subjects)}. If they were "
                "wanted, say so in the prompt next time, so the intent records them and they "
                "count as value; if not, revert them.",
            ),
        ],
    )


def _excessive_context(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Bound exploration by the size of the change",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Ask for a stated plan of what to read before reading widely.",
                "- Before exploring, list the few files the change will touch and read "
                "those first; widen the search only when they leave a named question open.",
                "markdown",
            ),
            Action(
                ActionKind.PRACTICE,
                f"Verify first: the context before changing {_list(ctx.subjects)} "
                "exceeded the band, but wide reading can be justified. If it was not, "
                "add a 'Where things live' map to CLAUDE.md so the agent finds the "
                "right files directly.",
            ),
        ],
    )


def _insufficient_context(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Acquire context in proportion to the change",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Require evidence for every file changed.",
                "- For each file you change in place, first read it or search for what "
                "you are changing in it within the current task; do not rely on memory "
                "of an earlier task.",
                "markdown",
            ),
            Action(
                ActionKind.SETTING,
                "Start tasks that change existing code in plan mode, so context comes "
                f"first (changed with too little context: {_list(ctx.subjects)}).",
                json.dumps({"permissions": {"defaultMode": "plan"}}, indent=2),
                "json",
            ),
        ],
    )


def _unused_traversal(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Search for a target, not a tour",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Point the agent at the layout so it does not walk the tree.",
                "## Where things live\n"
                "- <directory>: <what it holds>\n"
                "- Search for a symbol or a file name you expect, not whole directories.",
                "markdown",
            ),
            Action(
                ActionKind.PRACTICE,
                f"Verify first: {_list(ctx.subjects)} listed files nothing later used, "
                "which may have ruled a place out. If they recur across sessions, name "
                "the right places in CLAUDE.md.",
            ),
        ],
    )


def _failed_route(ctx: _Context) -> tuple[str, list[Action]]:
    routes = list(
        dict.fromkeys(
            s.split(": ", 1)[1].split(" failed", 1)[0]
            for s in ctx.subjects
            if ": " in s and " failed" in s
        )
    )
    named = _list(routes) if routes else _list(ctx.subjects)
    return (
        "Stop routing to routes that fail",
        [
            Action(
                ActionKind.SETTING,
                f"Demote or health-check the failing route(s) in the routing profile "
                f"({named} failed and the work was done elsewhere), so the first "
                "choice is a route that answers.",
            ),
            Action(
                ActionKind.PRACTICE,
                "Put a short timeout and a circuit breaker on routes that fail repeatedly, "
                "so a failover costs one fast failure rather than a wait per task.",
            ),
        ],
    )


def _unearned_escalation(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Escalate only on evidence, and give the escalated model something new",
        [
            Action(
                ActionKind.SETTING,
                "In the routing profile, escalate only on a detector signal with "
                "evidence (a failing check that does not move, a rework cycle), not "
                f"after any completed answer ({_list(ctx.subjects)} added nothing "
                "the earlier call lacked).",
            ),
            Action(
                ActionKind.PRACTICE,
                "When a task does escalate, hand the stronger model new evidence: the "
                "failing test output, the files the first answer did not read. An "
                "escalation that sees what the first model saw tends to answer the "
                "same way, after a wait.",
            ),
        ],
    )


def _unrelated(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Keep each task's edits inside its change surface",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Bound the change by the code the task names.",
                "- Change only the files the request names, the modules they import or "
                "that import them, and their tests. For anything else, say what you would "
                "change and why, and wait for the go-ahead.",
                "markdown",
            ),
            Action(
                ActionKind.PRACTICE,
                f"Edited with no import link to the task: {_list(ctx.subjects)}. Revert "
                "them or move them to their own task and commit; if they were wanted, "
                "name them in the prompt so they are inside the surface next time.",
            ),
        ],
    )


def _expansion(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Make ripple edits a stated decision",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Ask the agent to announce a change that spreads past the named code.",
                "- When a change has to spread beyond the files the request names and "
                "their direct imports, list those files and why before editing them.",
                "markdown",
            ),
            Action(
                ActionKind.PRACTICE,
                f"Verify first: {_list(ctx.subjects)} lie one import link beyond the "
                "surface, and callers often have to change with what they call. If "
                "these did not have to, ask for the narrower change next time.",
            ),
        ],
    )


def _exploration_drift(ctx: _Context) -> tuple[str, list[Action]]:
    return (
        "Keep exploration on what the change depends on",
        [
            Action(
                ActionKind.CLAUDE_MD,
                "Point the agent at the code the task depends on before it explores.",
                "- Explore the files the request names, what they import or what "
                "imports them, and their tests, docs and config first. Before "
                "reading another package, say what you expect to find there.",
                "markdown",
            ),
            Action(
                ActionKind.PRACTICE,
                f"Verify first: {_list(ctx.subjects)} lay off the change surface "
                "and nothing later used them. If they were needed, name them in "
                "the prompt or CLAUDE.md so the dependency is on record.",
            ),
        ],
    )


_LINT_IMPORTS_HOOK = (
    'cd "$CLAUDE_PROJECT_DIR" && lint-imports >/dev/null 2>&1 '
    "|| { echo 'lint-imports: an import breaks an architecture contract; "
    "run lint-imports to see which' >&2; exit 2; }"
)


def _boundary(ctx: _Context) -> tuple[str, list[Action]]:
    contracts = _list(ctx.subjects)
    hook = _settings(
        {
            "PostToolUse": [
                {
                    "matcher": "Edit|Write",
                    "hooks": [
                        {
                            "type": "command",
                            "command": _LINT_IMPORTS_HOOK,
                            "timeout": 120,
                        }
                    ],
                }
            ]
        }
    )
    return (
        "Check the architecture contracts while the agent edits",
        [
            Action(
                ActionKind.HOOK,
                "Run import-linter after every edit and feed a broken contract back "
                f"to the agent (broken this session: {contracts}).",
                hook,
                "json",
            ),
            Action(
                ActionKind.CLAUDE_MD,
                "Name the architecture rules the agent must keep.",
                "- The import contracts in the project's import-linter configuration "
                f"are rules, not suggestions ({contracts}). Before adding an import "
                "across packages, check they allow it; run `lint-imports` after "
                "changing imports.",
                "markdown",
            ),
        ],
    )


_CATALOGUE: dict[str, Callable[[_Context], tuple[str, list[Action]]]] = {
    "repeated_tool_call": _repeated_tool_call,
    "repeated_exploration": _repeated_exploration,
    "rework_cycle": _rework_cycle,
    "unvalidated_implementation": _unvalidated,
    "premature_implementation": _premature,
    "excessive_planning": _planning,
    "fragmented_edits": _fragmented,
    "unused_context": _unused,
    "unnecessary_handoff": _handoff,
    "repeated_reasoning": _reasoning,
    "regeneration": _regeneration,
    "intent_drift": _drift,
    "excessive_context": _excessive_context,
    "insufficient_context": _insufficient_context,
    "unused_traversal": _unused_traversal,
    "failed_route": _failed_route,
    "unearned_escalation": _unearned_escalation,
    "unrelated_modification": _unrelated,
    "surface_expansion": _expansion,
    "boundary_violation": _boundary,
    "exploration_drift": _exploration_drift,
}


def build_countermeasures(
    findings: Sequence[Finding],
    steps: Sequence[Step],
    allocated: Mapping[str, float] | None = None,
) -> tuple[Countermeasure, ...]:
    """One countermeasure per detector that fired, most costly first.

    A countermeasure's cost is generated tokens. With ``allocated`` (the
    scorecard's per-finding allocation, ``LeanAnalysis.allocated_waste_tokens``)
    a finding costs only the tokens charged to it, so two detectors claiming
    one event do not both count it. A finding without an allocation costs its
    own claim.
    """
    by_detector: dict[str, list[Finding]] = {}
    for finding in findings:
        by_detector.setdefault(finding.detector, []).append(finding)
    out: list[tuple[tuple[int, int, float, str], Countermeasure]] = []
    for detector, group in by_detector.items():
        make = _CATALOGUE.get(detector)
        if make is None:
            continue
        ctx = _Context(tuple(group), tuple(steps))
        title, actions = make(ctx)
        cost = round(
            sum(
                allocated.get(f.id, 0.0) if allocated is not None else float(f.tokens)
                for f in group
            )
        )
        uncertain = all(f.uncertain for f in group)
        confidence = max(f.confidence for f in group)
        rationale = (
            f"{len(group)} finding(s), {cost:,} generated tokens: "
            + "; ".join(f.title for f in group[:3])
            + ("; …" if len(group) > 3 else "")
            + "."
        )
        if uncertain:
            rationale += (
                " All below the confidence threshold: counted as waste until "
                "verified; verify before acting."
            )
        out.append(
            (
                (int(uncertain), -cost, -confidence, detector),
                Countermeasure(
                    detector=detector,
                    waste=group[0].waste,
                    title=title,
                    rationale=rationale,
                    addresses=tuple(f.id for f in group),
                    uncertain=uncertain,
                    actions=tuple(actions),
                ),
            )
        )
    return tuple(c for _, c in sorted(out, key=lambda p: p[0]))


_MEASURES: dict[str, tuple[str, str]] = {
    "repeated_tool_call": ("Repeated tool calls with unchanged results", "0"),
    "repeated_exploration": ("Re-reads and re-searches of unchanged content", "0"),
    "rework_cycle": ("Fix attempts that left a failure unchanged", "0"),
    "unvalidated_implementation": (
        "Responses after unvalidated edits or failing checks",
        "0",
    ),
    "premature_implementation": ("Edits to files not read first", "0"),
    "excessive_planning": ("Planning runs of 4+ steps without action", "0"),
    "fragmented_edits": ("Edit runs to one file over 3+ round trips", "0"),
    "unused_context": ("Files read and never used", "fewer"),
    "unnecessary_handoff": ("Handoffs redone by the agent", "0"),
    "repeated_reasoning": ("Restated reasoning blocks", "fewer"),
    "regeneration": ("Whole-file rewrites of existing content", "0"),
    "intent_drift": ("Edits departing from the intent with no intent change", "0"),
    "excessive_context": ("Tasks with context above the band (uncertain)", "fewer"),
    "insufficient_context": ("Tasks with context below the band", "0"),
    "unused_traversal": ("Traversals whose files were never used (uncertain)", "fewer"),
    "failed_route": ("Failed model routes waited on", "0"),
    "unearned_escalation": ("Escalations that added no evidence", "0"),
    "unrelated_modification": ("Edits with no import link to the task", "0"),
    "surface_expansion": (
        "Edits one import link beyond the change surface (uncertain)",
        "fewer",
    ),
    "boundary_violation": ("Imports that break an architecture contract", "0"),
    "exploration_drift": (
        "Reads off what the change depends on, with no intent change",
        "0",
    ),
}


def follow_ups(
    findings: Sequence[Finding],
    *,
    flow_efficiency: float | None,
    avoidable_share: float,
) -> tuple[FollowUp, ...]:
    """What to measure next run: each fired detector, then the flow headline.

    ``how`` is the field of ``ter a3 <next-session.jsonl> --json`` to read.
    """
    counts = Counter(f.detector for f in findings)
    out = [
        FollowUp(
            metric=_MEASURES[detector][0],
            current=str(n),
            target=_MEASURES[detector][1],
            how=f"findings[detector={detector}]",
        )
        for detector, n in sorted(counts.items(), key=lambda p: (-p[1], p[0]))
        if detector in _MEASURES
    ]
    if flow_efficiency is not None:
        out.append(
            FollowUp(
                metric="Agentic flow efficiency (tokens)",
                current=f"{flow_efficiency:.0%}",
                target="higher",
                how="scorecard.flow_efficiency_tokens",
            )
        )
    out.append(
        FollowUp(
            metric="Avoidable share of generated tokens",
            current=f"{avoidable_share:.0%}",
            target="lower",
            how="scorecard.activity_tokens.avoidable",
        )
    )
    return tuple(out)
