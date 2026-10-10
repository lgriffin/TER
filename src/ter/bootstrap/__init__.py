"""Composition root: the only place that chooses concrete adapters.

Entry points (CLI, hooks, CI gate) ask this package for wired use cases, and
it applies the installation's maturity ceiling while doing so. No other
module decides which capabilities are switched on.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ..adapters.driving.cli import CliServices
from ..adapters.driving.cli import main as cli_main
from ..adapters.driven.in_memory import SystemClock
from ..application.explain import ExplainedSession, ExplainSession
from ..application.route import RoutedSession, RouteSession
from .capabilities import (
    CapabilityRegistry,
    default_registry,
    detector_registry,
    repository_evidence,
)
from ..application.observe import (
    AnalyseEventLog,
    AnalyseTrace,
    ObserveEvent,
    RecordEvent,
)
from ..domain.capabilities import Capability, CapabilityProblem, UnknownCapabilityError
from ..domain.stream import StreamReport
from ..ports.driven import (
    ArchitectureContracts,
    OutcomeSource,
    SessionSource,
    TerScorer,
    Tokenizer,
)
from ..ports.driving import EventIngest

if TYPE_CHECKING:
    from ..adapters.driven.claude_code.corpus import CorpusImport
    from ..adapters.driving.claude_hooks.check import HookCheck
    from ..adapters.driving.control_cli import ControlServices
    from ..application.control import MeasuredSessions

__all__ = [
    "CapabilityRegistry",
    "cli_services",
    "default_registry",
    "default_event_log_dir",
    "detector_registry",
    "main",
    "make_contracts",
    "make_ingest",
    "make_outcome_source",
    "make_recorder",
    "make_ter_scorer",
    "make_tokenizer",
    "session_source_for",
]

#: Environment variable that relocates the live event log.
EVENT_LOG_ENV = "TER_EVENT_LOG_DIR"


def default_event_log_dir() -> Path:
    """``$TER_EVENT_LOG_DIR``, else ``ter/events`` in the user's cache directory.

    The log holds prompts and tool output, so the default lives under the
    user's home (``$XDG_CACHE_HOME`` or ``~/.cache``), not a shared temporary
    directory.
    """
    configured = os.environ.get(EVENT_LOG_ENV)
    if configured:
        return Path(configured)
    cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache) if cache else Path.home() / ".cache"
    return base / "ter" / "events"


def make_tokenizer(name: str = "regex") -> Tokenizer:
    """``regex`` is offline and deterministic; ``tiktoken`` needs its encoding.

    Any ``Tokenizer.<name>`` capability works; built-ins resolve without
    scanning installed packages, so hooks stay cheap.
    """
    try:
        tokenizer = default_registry().create("Tokenizer", name)
    except UnknownCapabilityError:
        raise ValueError(f"Unknown tokenizer {name!r}") from None
    assert isinstance(tokenizer, Tokenizer)  # checked by the registry
    return tokenizer


def make_outcome_source(name: str = "junit") -> OutcomeSource:
    """An ``OutcomeSource.<name>`` capability; ``junit`` reads JUnit XML results."""
    source = default_registry().create("OutcomeSource", name)
    assert isinstance(source, OutcomeSource)  # checked by the registry
    return source


def make_contracts(name: str = "import-linter") -> ArchitectureContracts:
    """An ``ArchitectureContracts.<name>`` capability; ``import-linter`` reads
    import-linter configuration (TER-EVD-007), ``dependency-cruiser`` the
    forbidden rules of ``.dependency-cruiser.json`` (TER-EVD-015)."""
    reader = default_registry().create("ArchitectureContracts", name)
    assert isinstance(reader, ArchitectureContracts)  # checked by the registry
    return reader


def make_ter_scorer(mode: str) -> TerScorer | None:
    """``offline`` pins TER 3 to the deterministic adapters; ``model`` uses its
    sentence-transformers model; ``off`` skips TER.

    The scorer, and the tokenizer and embedder it is pinned to, are
    capabilities (``TerScorer.ter3``, ``Tokenizer.regex``, ``Embedder.hashing``).
    """
    if mode == "off":
        return None
    registry = default_registry()
    if mode == "model":
        scorer = registry.create("TerScorer", "ter3")
    elif mode == "offline":
        scorer = registry.create(
            "TerScorer",
            "ter3",
            make_tokenizer("regex"),
            registry.create("Embedder", "hashing"),
        )
    else:
        raise ValueError(f"Unknown TER mode {mode!r}")
    assert isinstance(scorer, TerScorer)  # checked by the registry
    return scorer


def session_source_for(path: Path) -> SessionSource:
    """The agent adapter for a reference, from the ``SessionSource`` capabilities.

    A source that can tell its own files apart declares a static
    ``accepts(ref)``; the first, by name, that accepts ``path`` reads it (a
    GARE export, say). Anything else is a Claude Code transcript.
    """
    registry = default_registry()
    for cap in registry.capabilities("SessionSource"):
        if cap.name == "claude-code":
            continue
        try:
            accepts = getattr(registry.factory(cap.port, cap.name), "accepts", None)
            if not (callable(accepts) and bool(accepts(path))):
                continue
            source = registry.create(cap.port, cap.name)
        except Exception:  # a broken plugin must not stop Claude Code transcripts
            continue
        assert isinstance(source, SessionSource)  # checked by the registry
        return source
    source = registry.create("SessionSource", "claude-code")
    assert isinstance(source, SessionSource)
    return source


def make_ingest(tokenizer: str = "regex") -> EventIngest:
    """A fresh :class:`EventIngest` for analysing one recorded session.

    Every recorded path (transcript, GARE run, event log replay) applies its
    events through one of these (TER-OBS-001); the detectors it runs are the
    installed ``WasteDetector`` capabilities (TER-ARC-002).
    """
    return ObserveEvent(make_tokenizer(tokenizer), detectors=detector_registry())


def make_recorder(directory: Path) -> EventIngest:
    """The :class:`EventIngest` a hook process records live events through."""
    from ..adapters.driven.event_log import JsonlEventLog

    # Append-only: a hook's cost must not grow with the session. A hook
    # process never explains, so it skips plugin discovery too.
    return RecordEvent(make_tokenizer("regex"), JsonlEventLog(directory))


def cli_services() -> CliServices:
    """Wire the CLI's use cases. Heavy adapters are imported on first use."""

    def analyse_transcript(path: Path, tokenizer: str) -> StreamReport:
        return AnalyseTrace(
            session_source_for(path),
            make_tokenizer(tokenizer),
            lambda: make_ingest(tokenizer),
        )(path)

    def log_sessions(directory: Path) -> tuple[str, ...]:
        from ..adapters.driven.event_log import JsonlEventLog

        return JsonlEventLog(directory).sessions()

    def analyse_log(directory: Path, session_id: str, tokenizer: str) -> StreamReport:
        from ..adapters.driven.event_log import JsonlEventLog

        log = JsonlEventLog(directory)
        return AnalyseEventLog(
            log, make_tokenizer(tokenizer), lambda: make_ingest(tokenizer)
        )(session_id)

    def hook_ingest(directory: Path) -> EventIngest:
        return make_recorder(directory)

    def explain_transcript(
        path: Path,
        tokenizer: str,
        ter: str,
        outcome: Path | None = None,
        repo: Path | None = None,
        repo_engine: str = "syntax",
    ) -> ExplainedSession:
        from ..adapters.driven.claude_code import ClaudeCodeJsonlSource
        from ..adapters.driven.pricing import default_price_book

        source = session_source_for(path)
        # TER 3 scores Claude Code transcripts only; other sources get no TER
        # rather than a meaningless one (TER-SRC-017).
        scores = isinstance(source, ClaudeCodeJsonlSource)
        use_case = ExplainSession(
            source,
            make_tokenizer(tokenizer),
            make_ter_scorer(ter if scores else "off"),
            make_outcome_source() if outcome is not None else None,
            default_price_book(),
            lambda: make_ingest(tokenizer),
            # L3: the repository as it was when the session started.
            repository_evidence(repo, repo_engine) if repo is not None else None,
            (
                (make_contracts(), make_contracts("dependency-cruiser"))
                if repo is not None
                else None
            ),
        )
        return use_case(path, outcome)

    def route_transcript(
        path: Path,
        tokenizer: str,
        profile: str | None = None,
        repo: Path | None = None,
        repo_engine: str = "syntax",
        profiles_dir: Path | None = None,
    ) -> RoutedSession:
        from ..adapters.driven.routing_profiles import (
            JsonRoutingProfiles,
            default_routing_profiles,
        )

        # Advisory routing (L3): decisions and route.escalated events for
        # analysis, never sent to a live session (TER-INT-001).
        use_case = RouteSession(
            session_source_for(path),
            make_tokenizer(tokenizer),
            default_routing_profiles()
            if profiles_dir is None
            else JsonRoutingProfiles(profiles_dir),
            lambda: make_ingest(tokenizer),
            repository_evidence(repo, repo_engine) if repo is not None else None,
            make_contracts() if repo is not None else None,
        )
        return use_case(path, profile)

    def capabilities() -> tuple[tuple[Capability, ...], tuple[CapabilityProblem, ...]]:
        registry = default_registry()
        problems = registry.check()
        return registry.capabilities(), problems

    def tokenizers() -> tuple[str, ...]:
        # Only tokenizers that load and fit the port: a broken plugin is
        # reported by `capabilities`, not crashed into by a report command.
        registry = default_registry()
        broken = {(p.key, p.target) for p in registry.check("Tokenizer")}
        return tuple(
            c.name
            for c in registry.capabilities("Tokenizer")
            if (c.key, c.target) not in broken
        )

    def import_corpus(
        sources: Sequence[Path],
        out: Path,
        labels: Path | None,
        max_tool_output: int,
        keep_tools: frozenset[str],
        quote_files: bool,
    ) -> CorpusImport:
        from ..adapters.driven.claude_code.redaction import RedactionPolicy
        from ..adapters.driven.claude_code.corpus import (
            import_corpus as run,
            read_labels,
        )

        policy = RedactionPolicy(
            max_tool_output=max_tool_output,
            keep_tools=keep_tools,
            quote_file_contents=quote_files,
        )
        return run(
            sources,
            out,
            policy=policy,
            labels=read_labels(labels) if labels is not None else None,
        )

    def hooks_check(recordings: Path, transcripts: Path) -> "HookCheck":
        from ..adapters.driven.claude_code import ClaudeCodeJsonlSource
        from ..adapters.driving.claude_hooks.check import check_recordings

        return check_recordings(recordings, transcripts, ClaudeCodeJsonlSource().read)

    def control() -> "ControlServices":
        from ..adapters.driven.control_limits import JsonControlLimits
        from ..adapters.driving.control_cli import ControlServices
        from ..application.control import ChartSessions, measure_sessions

        def measure(paths: Sequence[Path], tokenizer: str) -> "MeasuredSessions":
            # TER off: the control measures are the Lean ones, and a corpus
            # must not download a model (as scripts/corpus_findings.py).
            return measure_sessions(
                paths, lambda path: explain_transcript(path, tokenizer, "off")
            )

        return ControlServices(measure, ChartSessions(JsonControlLimits()))

    from .context import context_services

    return CliServices(
        control=control(),
        context=context_services(session_source_for, make_tokenizer),
        hooks_check=hooks_check,
        capabilities=capabilities,
        import_corpus=import_corpus,
        tokenizers=tokenizers,
        analyse_transcript=analyse_transcript,
        log_sessions=log_sessions,
        analyse_log=analyse_log,
        hook_ingest=hook_ingest,
        default_log_dir=default_event_log_dir(),
        hook_clock=SystemClock(),
        explain_transcript=explain_transcript,
        route_transcript=route_transcript,
    )


def main(argv: Sequence[str] | None = None) -> int:
    return cli_main(argv, cli_services())
