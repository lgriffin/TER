"""Ports: the Protocols between the TER core and the outside world.

Driven ports (``ter.ports.driven``) are what TER needs from outside: session
sources, tokenizers, embedders, clocks, price books, repository evidence,
architecture contracts, routing profiles, control limits. Every driven port has one contract
suite under ``tests/contract``, and both the real adapter and its in-memory
fake must pass it.

Driving ports (``ter.ports.driving``) are what TER offers to entry points such
as hooks and the CLI.
"""

from __future__ import annotations

from .driven import (
    AlignmentScorer,
    ArchitectureContracts,
    Clock,
    ControlLimitsSource,
    Embedder,
    EventLog,
    OutcomeSource,
    PriceBook,
    RepositoryEvidence,
    RoutingProfiles,
    SessionSource,
    TerScorer,
    Tokenizer,
)
from .driving import EventIngest
from .plugins import WasteDetectorPlugin

#: Driven ports a capability (``ter.capabilities`` entry point) can plug into,
#: by the name its key uses: ``<Port>.<adapter>`` (ADR 0005).
DRIVEN_PORTS: dict[str, type[object]] = {
    "AlignmentScorer": AlignmentScorer,
    "ArchitectureContracts": ArchitectureContracts,
    "Clock": Clock,
    "ControlLimitsSource": ControlLimitsSource,
    "Embedder": Embedder,
    "EventLog": EventLog,
    "OutcomeSource": OutcomeSource,
    "PriceBook": PriceBook,
    "RepositoryEvidence": RepositoryEvidence,
    "RoutingProfiles": RoutingProfiles,
    "SessionSource": SessionSource,
    "TerScorer": TerScorer,
    "Tokenizer": Tokenizer,
}

#: Everything a capability can plug into: the driven ports, plus the analysis
#: plugins (waste detectors) that run inside the core (TER-ARC-002).
CAPABILITY_KINDS: dict[str, type[object]] = {
    **DRIVEN_PORTS,
    "WasteDetector": WasteDetectorPlugin,
}

__all__ = [
    "CAPABILITY_KINDS",
    "DRIVEN_PORTS",
    "AlignmentScorer",
    "ArchitectureContracts",
    "Clock",
    "ControlLimitsSource",
    "Embedder",
    "EventIngest",
    "EventLog",
    "OutcomeSource",
    "PriceBook",
    "RepositoryEvidence",
    "RoutingProfiles",
    "SessionSource",
    "TerScorer",
    "Tokenizer",
    "WasteDetectorPlugin",
]
