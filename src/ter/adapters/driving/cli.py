"""The ``python -m ter`` command line: a thin driving adapter.

It parses arguments, asks the composition root (through :class:`CliServices`)
for wired use cases, and renders :class:`~ter.domain.stream.StreamReport`
values as plain text or JSON. It holds no analysis logic.

Commands::

    python -m ter observe SESSION.jsonl [--timeline] [--json]
    python -m ter observe --event-log DIR [--session ID] [--timeline] [--json]
    python -m ter hook [--event-log DIR] [--record DIR]  # one payload on stdin
    python -m ter explain SESSION.jsonl [--json] [--graph FILE] [--outcome FILE]
                                        [--repo DIR [--repo-engine NAME]]
    python -m ter a3 SESSION.jsonl [--html FILE] [--json [FILE]] [--graph FILE]
                                   [--ter offline|model|off] [--outcome FILE]
                                   [--repo DIR [--repo-engine NAME]]
    python -m ter route SESSION.jsonl [--profile NAME] [--profiles DIR] [--json]
                                      [--repo DIR [--repo-engine NAME]]
    python -m ter hooks check RECORDINGS TRANSCRIPTS [--json FILE]
    python -m ter capabilities                # adapters per port, and problems
    python -m ter context bundle|report ...   # L3 context bundles (context_cli)
    python -m ter corpus import SRC... --out DIR [--labels CSV]
                                [--max-tool-output N] [--keep-tool NAME]
                                [--quote-files]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Protocol

from ...application.explain import ExplainedSession
from ...application.route import RoutedSession
from ...domain.capabilities import Capability, CapabilityError, CapabilityProblem
from ...domain.events import TEXT_LIMITS, describe_limit
from ...domain.lean import LeanAnalysis, SoftwareValueEfficiency
from ...domain.lean.surface import EditPlacement
from ...domain.outcome import OutcomeFormatError, OutcomeVerdict
from ...domain.repository import RepositoryEvidenceError
from ...domain.routing import RoutingProfileError
from ...domain.stack import StackKind, stack_label
from ...domain.stream import StreamReport
from ...ports.driven import Clock
from ...ports.driving import EventIngest
from .claude_hooks import HookStatus, run_hook
from .context_cli import ContextServices, add_context_parser, run_context

if TYPE_CHECKING:
    from ..driven.claude_code.corpus import CorpusImport
    from .claude_hooks.check import HookCheck

__all__ = [
    "CliServices",
    "format_capabilities",
    "format_corpus_import",
    "format_findings",
    "format_profile",
    "format_surfaces",
    "format_outcome",
    "format_report",
    "format_timeline",
    "main",
]

#: The built-in tokenizers; any installed ``Tokenizer.<name>`` capability is
#: accepted too, validated through ``CliServices.tokenizers``.
TOKENIZERS = ("regex", "tiktoken")
TOKENIZER_HELP = (
    "Tokenizer capability: regex (default), tiktoken or any installed "
    "Tokenizer.<name> (see `capabilities`)"
)
#: How the A3 obtains TER: offline (deterministic, lexical embedder), with the
#: TER 3 sentence-transformers model, or not at all.
TER_MODES = ("offline", "model", "off")
#: Checks listed under the outcome verdict in text output.
OUTCOME_ROWS = 12
#: Share of records the session source should map (TER-SRC-005).
CORPUS_COVERAGE_TARGET = 0.99
REPO_HELP = (
    "L3: the repository the session worked in, checked out at the commit the "
    "session started from: report each task's change surface, edits outside "
    "it and imports that break the repository's import-linter contracts"
)
REPO_ENGINE_HELP = (
    "RepositoryEvidence engine for --repo: syntax (default: the import graph "
    "of Python, TypeScript, JavaScript, Svelte and Vue files), python-ast "
    "(Python only), lexical (no import graph), git or any installed one"
)
OUTCOME_HELP = (
    "test results of the run (JUnit XML, e.g. from pytest --junitxml): judge "
    "the outcome and show the verdict beside the measures"
)


class ExplainTranscript(Protocol):
    """``explain_transcript(path, tokenizer, ter, outcome, repo, repo_engine)``;
    ``repo`` (L3) grounds the analysis on the repository at that path."""

    def __call__(
        self,
        path: Path,
        tokenizer: str,
        ter: str,
        outcome: Path | None = None,
        repo: Path | None = None,
        repo_engine: str = "syntax",
    ) -> ExplainedSession: ...


class RouteTranscript(Protocol):
    """``route_transcript(path, tokenizer, profile, repo, repo_engine,
    profiles_dir)`` (L3): classify and route a session's tasks by role."""

    def __call__(
        self,
        path: Path,
        tokenizer: str,
        profile: str | None = None,
        repo: Path | None = None,
        repo_engine: str = "syntax",
        profiles_dir: Path | None = None,
    ) -> RoutedSession: ...


@dataclass(frozen=True)
class CliServices:
    """Use cases the composition root hands to the CLI."""

    analyse_transcript: Callable[[Path, str], StreamReport]
    log_sessions: Callable[[Path], tuple[str, ...]]
    analyse_log: Callable[[Path, str, str], StreamReport]
    hook_ingest: Callable[[Path], EventIngest]
    default_log_dir: Path
    hook_clock: Clock | None = None
    explain_transcript: ExplainTranscript | None = None
    capabilities: (
        Callable[[], tuple[tuple[Capability, ...], tuple[CapabilityProblem, ...]]]
        | None
    ) = None
    #: Names of the registered ``Tokenizer`` capabilities, to validate
    #: ``--tokenizer``; ``None`` accepts the built-ins only.
    tokenizers: Callable[[], tuple[str, ...]] | None = None
    #: ``import_corpus(sources, out, labels_csv, max_tool_output, keep_tools,
    #: quote_files)``; raises ``ValueError`` for a bad label file.
    import_corpus: (
        Callable[
            [Sequence[Path], Path, Path | None, int, frozenset[str], bool],
            "CorpusImport",
        ]
        | None
    ) = None
    #: ``hooks_check(recordings, transcripts)``; raises ``OSError`` or
    #: ``ValueError`` when the recordings cannot be read.
    hooks_check: Callable[[Path, Path], "HookCheck"] | None = None
    #: L3 context bundles: ``python -m ter context`` (TER-CTX-001).
    context: ContextServices | None = None
    #: L3 advisory routing (``python -m ter route``).
    route_transcript: RouteTranscript | None = None


def main(
    argv: Sequence[str] | None,
    services: CliServices,
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    args = _parser(services.default_log_dir).parse_args(argv)
    if args.command == "hook":
        log_dir: Path = args.event_log
        result = run_hook(
            stdin or sys.stdin,
            out,
            lambda: services.hook_ingest(log_dir),
            clock=services.hook_clock,
            record_to=args.record,
        )
        # Still exit 0 so the agent carries on; stderr leaves a trace
        # (Claude Code shows it in verbose mode and debug logs).
        if result.record_error:
            err.write(f"ter hook: payload not recorded: {result.record_error}\n")
        if result.status is HookStatus.IGNORED and result.reason:
            err.write(f"ter hook: event not recorded: {result.reason}\n")
        return 0
    if args.command in ("observe", "explain", "a3", "context", "route"):
        known = TOKENIZERS if services.tokenizers is None else services.tokenizers()
        if args.tokenizer not in known:
            err.write(
                f"Unknown tokenizer {args.tokenizer!r} "
                f"(available: {', '.join(known) or 'none'}; "
                "`python -m ter capabilities` shows broken ones)\n"
            )
            return 2
    if args.command in ("explain", "a3"):
        return _explain(args, services, out, err)
    if args.command == "route":
        return _route(args, services, out, err)
    if args.command == "capabilities":
        return _capabilities(services, out, err)
    if args.command == "corpus":
        return _corpus(args, services, out, err)
    if args.command == "hooks":
        return _hooks_check(args, services, out, err)
    if args.command == "context":
        return run_context(args, services.context, out, err)
    return _observe(args, services, out, err)


def _write(path: Path, text: str) -> None:
    if path.parent != Path():
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def _explain(
    args: argparse.Namespace, services: CliServices, out: IO[str], err: IO[str]
) -> int:
    if services.explain_transcript is None:
        err.write("explain is not available in this installation\n")
        return 2
    if not args.path.exists():
        err.write(f"No such session file: {args.path}\n")
        return 2
    ter_mode = getattr(args, "ter", "off")
    outcome_path: Path | None = args.outcome
    if outcome_path is not None and not outcome_path.is_file():
        err.write(f"No such outcome file: {outcome_path}\n")
        return 2
    repo: Path | None = args.repo
    if repo is not None and not repo.is_dir():
        err.write(f"No such repository directory: {repo}\n")
        return 2
    try:
        if repo is None:
            explained = services.explain_transcript(
                args.path, args.tokenizer, ter_mode, outcome_path
            )
        else:
            explained = services.explain_transcript(
                args.path,
                args.tokenizer,
                ter_mode,
                outcome_path,
                repo,
                args.repo_engine,
            )
    except OutcomeFormatError as exc:
        err.write(f"Cannot read outcome: {exc}\n")
        return 2
    except (RepositoryEvidenceError, CapabilityError) as exc:
        err.write(f"Cannot read repository {repo}: {exc}\n")
        return 2
    grounding = explained.grounding
    if grounding is not None and grounding.contract_problem is not None:
        err.write(f"Architecture contracts not checked: {grounding.contract_problem}\n")
    if ter_mode != "off" and explained.analysis.scorecard.ter is None:
        # TER 3 reads Claude Code transcripts only (TER-SRC-017).
        err.write(
            f"No TER score: TER 3 scoring reads Claude Code transcripts, "
            f"not {explained.trace.source_format}\n"
        )
    if args.graph is not None:
        _write(args.graph, _json(explained.analysis.graph.as_dict()))
        err.write(f"Wrote {args.graph}\n")
    if args.command == "explain":
        if args.json:
            analysis = explained.analysis.as_dict()
            analysis["scorecard"] = explained.a3.scorecard_dict()
            if explained.a3.outcome is not None:
                analysis["outcome"] = explained.a3.as_dict()["outcome"]
            out.write(_json(analysis))
        else:
            out.write(
                format_findings(explained.analysis, explained.a3.value_efficiency)
            )
            out.write(format_limits(explained.a3.usage_limits))
            out.write(format_outcome(explained.a3.outcome, outcome_path))
        return 0

    from .reports.a3 import render_a3_html

    wrote = False
    if args.html is not None:
        _write(args.html, render_a3_html(explained.a3))
        err.write(f"Wrote {args.html}\n")
        wrote = True
    if args.json is not None:
        payload = _json(explained.a3.as_dict())
        if args.json == "-":
            out.write(payload)
        else:
            _write(Path(args.json), payload)
            err.write(f"Wrote {args.json}\n")
        wrote = True
    if not wrote:
        out.write(format_findings(explained.analysis, explained.a3.value_efficiency))
        out.write(format_limits(explained.a3.usage_limits))
        out.write(format_outcome(explained.a3.outcome, outcome_path))
    return 0


def _route(
    args: argparse.Namespace, services: CliServices, out: IO[str], err: IO[str]
) -> int:
    from .route_report import format_routing

    if services.route_transcript is None:
        err.write("route is not available in this installation\n")
        return 2
    if not args.path.exists():
        err.write(f"No such session file: {args.path}\n")
        return 2
    repo: Path | None = args.repo
    if repo is not None and not repo.is_dir():
        err.write(f"No such repository directory: {repo}\n")
        return 2
    try:
        routed = services.route_transcript(
            args.path,
            args.tokenizer,
            args.profile,
            repo,
            args.repo_engine,
            args.profiles,
        )
    except RoutingProfileError as exc:
        err.write(f"Cannot use routing profile: {exc}\n")
        return 2
    except (RepositoryEvidenceError, CapabilityError) as exc:
        err.write(f"Cannot read repository {repo}: {exc}\n")
        return 2
    if args.json:
        out.write(_json(routed.plan.as_dict()))
    else:
        out.write(format_routing(routed.plan, routed.analysis.repository is not None))
    return 0


def _hooks_check(
    args: argparse.Namespace, services: CliServices, out: IO[str], err: IO[str]
) -> int:
    from .claude_hooks.check import format_hook_check

    recordings: Path = args.recordings
    transcripts: Path = args.transcripts
    if services.hooks_check is None:
        err.write("hooks check is not available\n")
        return 2
    if not recordings.is_dir():
        err.write(f"ter hooks check: {recordings} is not a directory\n")
        return 2
    if not transcripts.exists():
        err.write(f"ter hooks check: {transcripts} does not exist\n")
        return 2
    try:
        check = services.hooks_check(recordings, transcripts)
    except (OSError, ValueError) as error:
        # The type and the file only: a parser's message can quote content.
        where = getattr(error, "filename", None) or ""
        err.write(
            f"ter hooks check: recordings unreadable: {type(error).__name__}"
            + (f" ({where})" if where else "")
            + "\n"
        )
        return 2
    out.write(format_hook_check(check))
    if args.json is not None:
        _write(args.json, _json(check.to_dict()))
    return 0


def _capabilities(services: CliServices, out: IO[str], err: IO[str]) -> int:
    """List every adapter per port; exit 1 when a registered one is broken."""
    if services.capabilities is None:
        err.write("capabilities are not available in this installation\n")
        return 2
    found, problems = services.capabilities()
    out.write(format_capabilities(found, problems))
    return 1 if problems else 0


def _corpus(
    args: argparse.Namespace, services: CliServices, out: IO[str], err: IO[str]
) -> int:
    """Redact real sessions into a research corpus (issue #34)."""
    if services.import_corpus is None:
        err.write("corpus import is not available in this installation\n")
        return 2
    sources: list[Path] = args.sources
    target = args.out.resolve()
    for source in sources:
        if not source.exists():
            err.write(f"No such session file or folder: {source}\n")
            return 2
        resolved = source.resolve()
        if resolved.is_dir() and (target == resolved or resolved in target.parents):
            # The next import would read the redacted copies back as sources.
            err.write(f"--out {args.out} must not be inside the source {source}\n")
            return 2
    if args.labels is not None and not args.labels.is_file():
        err.write(f"No such label file: {args.labels}\n")
        return 2
    if args.max_tool_output < 0:
        err.write("--max-tool-output must be 0 or more\n")
        return 2
    try:
        result = services.import_corpus(
            sources,
            args.out,
            args.labels,
            args.max_tool_output,
            frozenset(args.keep_tool),
            args.quote_files,
        )
    except ValueError as exc:
        err.write(f"ter corpus import: {exc}\n")
        return 2
    out.write(format_corpus_import(result))
    return 1 if result.load_failures else 0


def format_corpus_import(result: "CorpusImport") -> str:
    """What an import wrote, and what needs a look before the corpus is used."""
    redactions: dict[str, int] = {}
    for session in result.sessions:
        for kind, count in session.redactions.items():
            redactions[kind] = redactions.get(kind, 0) + count
    lines = [
        f"TER corpus import · {len(result.sessions)} session(s) into {result.out}"
        f" ({result.total} in its manifest)",
        "  redactions  "
        + (" · ".join(f"{k} {n:,}" for k, n in sorted(redactions.items())) or "-"),
    ]
    low = [
        s
        for s in result.sessions
        if s.coverage is not None and s.coverage < CORPUS_COVERAGE_TARGET
    ]
    if low:
        lines.append(
            f"  {len(low)} session(s) below {CORPUS_COVERAGE_TARGET:.0%} coverage "
            "(see unrecognised_by_type in manifest.json):"
        )
        lines += [f"    {s.file}  {s.coverage:.1%}" for s in low]
    for session in result.load_failures:
        lines.append(f"  ! {session.file}: {session.load_error}")
    if result.unlabelled:
        lines.append(f"  {len(result.unlabelled)} session id(s) have no labels")
    if result.unknown_labels:
        lines.append(
            "  labels for sessions not found: " + ", ".join(result.unknown_labels)
        )
    lines.append("  Review reports/ and the redacted sessions before sharing anything.")
    return "\n".join(lines) + "\n"


def format_capabilities(
    found: Sequence[Capability], problems: Sequence[CapabilityProblem]
) -> str:
    # A problem names the entry it rejected: a plugin clashing with a
    # built-in's key must not hide the built-in that keeps working.
    broken = {(p.key, p.target) for p in problems}
    usable = [c for c in found if (c.key, c.target) not in broken]
    lines = [f"TER capabilities · {len(usable)} usable, {len(problems)} problem(s)"]
    if usable:
        port_w = max(len(c.port) for c in usable)
        name_w = max(len(c.name) for c in usable)
        lines += [
            f"  {c.port:<{port_w}}  {c.name:<{name_w}}  {c.target}  [{c.origin}]"
            for c in usable
        ]
    for p in problems:
        lines.append(f"  ! {p.key} ({p.target}): {p.reason}")
    return "\n".join(lines) + "\n"


def _observe(
    args: argparse.Namespace, services: CliServices, out: IO[str], err: IO[str]
) -> int:
    if args.event_log is not None:
        sessions = services.log_sessions(args.event_log)
        session = args.session
        if session is None:
            if len(sessions) != 1:
                listing = "\n".join(f"  {s}" for s in sessions) or "  (none)"
                err.write(
                    f"{len(sessions)} sessions in {args.event_log}; "
                    f"choose one with --session:\n{listing}\n"
                )
                return 2
            session = sessions[0]
        report = services.analyse_log(args.event_log, session, args.tokenizer)
    elif args.path is not None:
        if not args.path.exists():
            err.write(f"No such session file: {args.path}\n")
            return 2
        report = services.analyse_transcript(args.path, args.tokenizer)
    else:
        err.write("observe needs a session file or --event-log DIR\n")
        return 2

    if args.json:
        out.write(json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n")
        return 0
    out.write(format_report(report))
    if args.timeline:
        out.write("\n" + format_timeline(report, limit=args.limit))
    return 0


def _parser(default_log_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ter", description="TER 4: Lean analysis of agent sessions."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    observe = commands.add_parser(
        "observe", help="L1 observables of a session, from its event stream"
    )
    observe.add_argument(
        "path", nargs="?", type=Path, help="Claude Code session .jsonl"
    )
    observe.add_argument(
        "--event-log",
        type=Path,
        default=None,
        metavar="DIR",
        help="analyse what `ter hook` recorded in DIR instead of a transcript",
    )
    observe.add_argument("--session", help="session id within --event-log")
    observe.add_argument("--timeline", action="store_true", help="print every event")
    observe.add_argument(
        "--limit", type=int, default=None, help="timeline rows to print"
    )
    observe.add_argument("--json", action="store_true", help="print the report as JSON")
    observe.add_argument("--tokenizer", default="regex", help=TOKENIZER_HELP)

    explain = commands.add_parser(
        "explain", help="L2: Lean findings, value stream and scorecard of a session"
    )
    explain.add_argument("path", type=Path, help="Claude Code session .jsonl")
    explain.add_argument(
        "--json", action="store_true", help="print the analysis as JSON"
    )
    explain.add_argument(
        "--graph", type=Path, metavar="FILE", help="write the evidence graph as JSON"
    )
    explain.add_argument("--tokenizer", default="regex", help=TOKENIZER_HELP)
    explain.add_argument("--outcome", type=Path, metavar="FILE", help=OUTCOME_HELP)
    explain.add_argument("--repo", type=Path, metavar="DIR", help=REPO_HELP)
    explain.add_argument("--repo-engine", default="syntax", help=REPO_ENGINE_HELP)

    a3 = commands.add_parser("a3", help="L2: a one-page Lean A3 report of a session")
    a3.add_argument("path", type=Path, help="Claude Code session .jsonl")
    a3.add_argument("--html", type=Path, metavar="FILE", help="write the A3 as HTML")
    a3.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        metavar="FILE",
        help="write the A3 as JSON to FILE (stdout when FILE is omitted)",
    )
    a3.add_argument(
        "--graph", type=Path, metavar="FILE", help="write the evidence graph as JSON"
    )
    a3.add_argument(
        "--ter",
        choices=TER_MODES,
        default="offline",
        help="how to compute TER: offline (deterministic, default), model "
        "(TER 3 sentence-transformers, may download) or off",
    )
    a3.add_argument("--tokenizer", default="regex", help=TOKENIZER_HELP)
    a3.add_argument("--outcome", type=Path, metavar="FILE", help=OUTCOME_HELP)
    a3.add_argument("--repo", type=Path, metavar="DIR", help=REPO_HELP)
    a3.add_argument("--repo-engine", default="syntax", help=REPO_ENGINE_HELP)

    add_context_parser(commands, default_log_dir, TOKENIZER_HELP, REPO_ENGINE_HELP)
    route = commands.add_parser(
        "route",
        help="L3: classify each task and choose a model role for it (advisory)",
    )
    route.add_argument("path", type=Path, help="session .jsonl or GARE run")
    route.add_argument(
        "--profile", help="routing profile name (default: the profiles' default)"
    )
    route.add_argument(
        "--profiles",
        type=Path,
        metavar="DIR",
        help="read routing profiles from DIR/*.json instead of the shipped ones",
    )
    route.add_argument("--json", action="store_true", help="print the plan as JSON")
    route.add_argument("--tokenizer", default="regex", help=TOKENIZER_HELP)
    route.add_argument("--repo", type=Path, metavar="DIR", help=REPO_HELP)
    route.add_argument("--repo-engine", default="syntax", help=REPO_ENGINE_HELP)

    commands.add_parser(
        "capabilities",
        help="adapters registered for each port (ter.capabilities), and any that are broken",
    )

    hook = commands.add_parser(
        "hook", help="Claude Code hook: record one hook payload read from stdin"
    )
    hook.add_argument(
        "--event-log",
        type=Path,
        default=default_log_dir,
        metavar="DIR",
        help=f"where live events are appended (default {default_log_dir})",
    )
    hook.add_argument(
        "--record",
        type=Path,
        default=None,
        metavar="DIR",
        help="also save the raw payload to DIR/<session>/<seq>-<hook>.json "
        "(for collecting real hook payloads; contains tool inputs and output)",
    )

    hooks = commands.add_parser(
        "hooks", help="check recorded hook payloads (ter hook --record)"
    )
    hooks_commands = hooks.add_subparsers(dest="hooks_command", required=True)
    hooks_check = hooks_commands.add_parser(
        "check",
        help="correlate recorded hook payloads with the sessions' transcripts "
        "(content-free report: counts, ids, reasons, field names)",
    )
    hooks_check.add_argument(
        "recordings", type=Path, help="the DIR given to `ter hook --record DIR`"
    )
    hooks_check.add_argument(
        "transcripts",
        type=Path,
        help="Claude Code projects folder (e.g. ~/.claude/projects) or a folder "
        "of .jsonl transcripts",
    )
    hooks_check.add_argument(
        "--json", type=Path, metavar="FILE", help="also write the report as JSON"
    )

    corpus = commands.add_parser(
        "corpus", help="build a redacted research corpus from real sessions"
    )
    corpus_commands = corpus.add_subparsers(dest="corpus_command", required=True)
    corpus_import = corpus_commands.add_parser(
        "import",
        help="redact sessions into DIR with a manifest and redaction reports",
    )
    corpus_import.add_argument(
        "sources",
        nargs="+",
        type=Path,
        metavar="SRC",
        help="session .jsonl files or folders of them (e.g. ~/.claude/projects)",
    )
    corpus_import.add_argument(
        "--out", type=Path, required=True, metavar="DIR", help="corpus folder"
    )
    corpus_import.add_argument(
        "--labels",
        type=Path,
        metavar="CSV",
        help="session_id,task_category,task,outcome,rating,licence",
    )
    corpus_import.add_argument(
        "--max-tool-output",
        type=int,
        default=2000,
        metavar="N",
        help="drop tool outputs longer than N characters (default 2000)",
    )
    corpus_import.add_argument(
        "--keep-tool",
        action="append",
        default=[],
        metavar="NAME",
        help="keep this tool's outputs whatever their length (repeatable)",
    )
    corpus_import.add_argument(
        "--quote-files",
        action="store_true",
        help="keep the content of files the agent read (only for code whose "
        "licence allows sharing)",
    )
    return parser


def _pairs(pairs: Sequence[tuple[object, int]]) -> str:
    return " · ".join(f"{_value(k)} {n:,}" for k, n in pairs) or "-"


def _value(key: object) -> str:
    return str(getattr(key, "value", key))


def format_report(report: StreamReport) -> str:
    """A readable plain-text summary of a report."""
    usage = report.usage
    trust = "exact" if report.tokens_exact else "estimate"
    reads = (
        ", ".join(f"{path} ×{n}" for path, n in report.repeated_reads)
        if report.repeated_reads
        else "-"
    )
    rows = [
        ("events", f"{report.total_events:,}  ({_pairs(report.by_class)})"),
        ("by kind", _pairs(report.by_kind)),
        ("by tool", _pairs(report.by_tool)),
        (
            "text tokens",
            f"{_pairs(report.tokens_by_class)}  [{report.tokenizer}, {trust}]"
            + _limits(report.usage_limits, text=True),
        ),
        (
            "usage",
            f"input {usage.input_tokens:,} · output {usage.output_tokens:,} · "
            f"cache write {usage.cache_creation_tokens:,} · "
            f"cache read {usage.cache_read_tokens:,}"
            + _limits(report.usage_limits, text=False),
        ),
        ("duplicate calls", f"{len(report.duplicate_tool_calls)}"),
        ("repeated reads", f"{report.repeated_read_count}  {reads}"),
        ("orphan results", f"{len(report.orphan_results)}"),
        ("open requests", f"{len(report.open_requests)}"),
        (
            "unvalidated edits",
            f"{report.edits_since_validation} since last shell "
            f"(peak {report.peak_edits_without_validation})",
        ),
    ]
    width = max(len(label) for label, _ in rows)
    lines = [f"TER observe · session {report.session_id or '-'}"]
    lines += [f"  {label:<{width}}  {value}" for label, value in rows]
    return "\n".join(lines) + "\n"


def _limits(limits: Sequence[str], *, text: bool) -> str:
    """The limits that qualify text token counts, or the usage figures."""
    return "".join(
        f"  [{describe_limit(limit)}]"
        for limit in limits
        if (limit in TEXT_LIMITS) is text
    )


def format_limits(limits: Sequence[str]) -> str:
    """Every usage limit of the trace an explanation was built from."""
    return "".join(f"  limit            {describe_limit(lim)}\n" for lim in limits)


def format_timeline(report: StreamReport, *, limit: int | None = None) -> str:
    """One line per accepted event, with the signals it raised."""
    rows = report.timeline if limit is None else report.timeline[:limit]
    # Routing kinds such as verification.completed are wider than the rest.
    width = max([15, *(len(row.kind.value) for row in rows)])
    header = f"  {'#':>4}  {'kind':<{width}} {'actor':<9} {'tool':<13} {'tokens':>7}  signals"
    lines = ["Timeline", header]
    for row in rows:
        tool = row.tool_kind.value if row.tool_kind else ""
        signals = ", ".join(s.value for s in row.signals)
        lines.append(
            f"  {row.index:>4}  {row.kind.value:<{width}} {row.actor.value:<9} "
            f"{tool:<13} {row.tokens:>7}  {signals}".rstrip()
        )
    hidden = len(report.timeline) - len(rows)
    if hidden > 0:
        lines.append(f"  … {hidden} more")
    return "\n".join(lines) + "\n"


def format_outcome(verdict: OutcomeVerdict | None, path: Path | None) -> str:
    """The outcome verdict, judged apart from the measures above it."""
    if path is None:
        return ""
    if verdict is None:
        return f"  outcome          none recorded in {path}\n"
    lines = [
        f"  outcome          {verdict.verdict.value}: {'; '.join(verdict.reasons)}"
    ]
    # Every check, open ones first, with its status and evidence source, so an
    # accepted verdict shows what it rests on too.
    ordered = sorted(
        verdict.results,
        key=lambda r: r.status is not None and r.status.value == "passed",
    )
    for r in ordered[:OUTCOME_ROWS]:
        status = "no evidence" if r.status is None else r.status.value
        optional = "" if r.check.required else " (optional)"
        detail = next((e.detail for e in r.evidence if e.detail), "")
        sources = ", ".join(e.source for e in r.evidence)
        lines.append(
            f"  - {status}: {r.check.id}{optional}"
            + (f" · {detail}" if detail else "")
            + (f" [{sources}]" if sources else "")
        )
    if len(ordered) > OUTCOME_ROWS:
        lines.append(
            f"  … {len(ordered) - OUTCOME_ROWS} more (--json lists every check)"
        )
    return "\n".join(lines) + "\n"


def format_surfaces(analysis: LeanAnalysis) -> str:
    """One line on the change surfaces of a grounded (L3) analysis."""
    g = analysis.repository
    placed = [e.placement for s in analysis.surfaces for e in s.edits]
    counts = ", ".join(
        f"{sum(p is kind for p in placed)} {kind.value.replace('_', ' ')}"
        for kind in EditPlacement
    )
    contracts = (
        "no contracts"
        if g is None or not g.contracts
        else f"{len(g.contracts)} contract(s) from {g.contract_source}"
    )
    line = (
        f"  change surface   {len(analysis.surfaces)} task(s); edits: {counts}; "
        f"{contracts}"
    )
    if g is not None and len(g.roots) > 1:
        # More than one checkout named the repository (TER-EVD-017).
        line += f"\n  repository roots {', '.join(g.roots)}"
    return line


def format_evidence(analysis: LeanAnalysis) -> str:
    """Two lines on evidence usage and outcome value of a grounded (L3)
    analysis (TER-EVD-008, TER-LEN-009); empty without them."""
    u, v = analysis.usage, analysis.value
    if u is None or v is None:
        return ""
    s = u.as_dict()["summary"]
    assert isinstance(s, dict)
    lines = [
        f"  evidence usage   {s['reads']} read(s): {s['used']} used "
        f"({s['material']} by a change, command or check), {s['unused']} unused "
        f"({s['unused_context_tokens']:,} tok), {s['pending']} pending; "
        f"{s['explored_and_changed']} of {s['files_explored']} file(s) read "
        "were changed"
    ]
    totals: dict[str, int] = {}
    for counts in v.counts().values():
        for k, n in counts.items():
            totals[k] = totals.get(k, 0) + n
    lines.append(
        "  outcome value    "
        + ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in totals.items())
        + f" ({sum(j.uncertain and j.value.value != 'unjudged' for j in v.judgements)}"
        " uncertain)"
    )
    return "\n".join(lines)


def format_profile(analysis: LeanAnalysis) -> str:
    """The session's languages and, when grounded, its stack (TER-STK-001,
    TER-STK-002); empty when it named no file and has no stack."""
    p = analysis.profile
    parts = [
        f"{u.language} {u.edits} edit(s)/{u.reads} read(s)" for u in p.languages[:4]
    ]
    if len(p.languages) > 4:
        parts.append(f"{len(p.languages) - 4} more")
    if p.unrecognised:
        parts.append(f"{sum(n for _, n in p.unrecognised)} unrecognised")
    lines: list[str] = []
    if parts:
        lines.append(
            f"  languages        {', '.join(parts)}"
            + (
                ""
                if p.dominant is None
                else f" · dominant {p.dominant} (by {p.dominant_basis})"
            )
        )
    if p.stack is not None:
        facts = ", ".join(
            f.name for f in p.stack.facts if f.kind is not StackKind.ECOSYSTEM
        )
        lines.append(
            f"  stack            {stack_label(p.stack)}"
            + (f" · {facts}" if facts else "")
            + f" · {len(p.stack.manifests)} manifest(s)"
        )
    return "\n".join(lines)


def format_findings(
    analysis: LeanAnalysis, sve: SoftwareValueEfficiency | None = None
) -> str:
    """A readable plain-text summary of an L2 analysis.

    With ``sve``, Software Value Efficiency is printed next to TER.
    """
    sc = analysis.scorecard
    lines = [f"TER explain · session {analysis.session_id or '-'}"]
    eff = sc.flow_efficiency_tokens
    lines.append(
        f"  flow efficiency  {'-' if eff is None else f'{eff:.0%}'} of generated tokens"
        + (
            ""
            if sc.flow_efficiency_time is None
            else f", {sc.flow_efficiency_time:.0%} of agent time"
        )
    )
    lines.append(
        "  activity         " + " · ".join(f"{k} {n:,}" for k, n in sc.activity_tokens)
    )
    if sc.ter is not None:
        lines.append(f"  TER              {sc.ter.value:.3f} ({sc.ter.method})")
    if sve is not None:
        lines.append(
            "  value efficiency "
            + (
                "unknown"
                if sve.tokens is None
                else f"{sve.tokens:.0%} of generated tokens"
                + ("" if sve.time is None else f", {sve.time:.0%} of agent time")
            )
            + f" · {sve.reason}"
        )
    wip = analysis.wip
    if wip.peak is not None:
        final = wip.final
        lines.append(
            f"  WIP              peak {wip.peak.total} ("
            + ", ".join(f"{k.value} {n}" for k, n in wip.peak_by_kind)
            + f"), {0 if final is None else final.total} open at the end"
        )
    profile = format_profile(analysis)
    if profile:
        lines.append(profile)
    if analysis.repository is not None:
        lines.append(format_surfaces(analysis))
        evidence = format_evidence(analysis)
        if evidence:
            lines.append(evidence)
    lines.append(
        f"  findings         {sc.findings} waste ({sc.uncertain_findings} uncertain, "
        "counted until verified), "
        f"{sc.risks} risk(s)"
    )
    for f in analysis.findings:
        flag = " (uncertain)" if f.uncertain else ""
        cost = (
            "risk" if f.kind.value == "risk" else f"{f.tokens + f.context_tokens} tok"
        )
        lines.append(
            f"  - [{f.confidence:.2f}{flag}] {f.waste.value}: {f.title} · {cost} · "
            f"evidence {', '.join(f.evidence[:4])}{' …' if len(f.evidence) > 4 else ''}"
        )
    return "\n".join(lines) + "\n"
