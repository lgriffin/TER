"""The capability registry: adapters for ports, discovered and loaded lazily.

TER's own adapters are built in (:data:`BUILTIN_CAPABILITIES`); an installed
package adds more by declaring entry points in the ``ter.capabilities`` group
(ADR 0005)::

    [project.entry-points."ter.capabilities"]
    "OutcomeSource.junit" = "ter.adapters.driven.junit:JUnitOutcomeSource"

The registry

* resolves a built-in without scanning installed packages, so a hook pays
  nothing for discovery;
* imports a capability's module only when a use case asks for it;
* rejects an object that does not satisfy its port's Protocol, naming the
  missing members;
* never raises out of discovery: a broken or conflicting entry is recorded
  in :attr:`CapabilityRegistry.problems` and the rest keep working;
* never lets an entry point replace a built-in of the same name.

Discovery reads installed package metadata, which is why the registry lives
in the composition root rather than the core.
"""

from __future__ import annotations

import dis
import importlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import CodeType
from typing import Any, Generic, Protocol

from ..domain.capabilities import (
    CAPABILITY_GROUP,
    Capability,
    CapabilityError,
    CapabilityProblem,
    UnknownCapabilityError,
    parse_capability_key,
)
from ..domain.lean.detectors import DEFAULT_REGISTRY, DetectorRegistry
from ..ports import CAPABILITY_KINDS, RepositoryEvidence, WasteDetectorPlugin

__all__ = [
    "BUILTIN_CAPABILITIES",
    "BUILTIN_ORIGIN",
    "CapabilityRegistry",
    "EntryLike",
    "builtin_detectors",
    "default_registry",
    "detector_registry",
    "installed_entry_points",
    "repository_evidence",
]

#: TER's own adapters, by capability key. ``pyproject.toml`` declares the same
#: table as entry points; ``tests/unit/test_ter4_capabilities.py`` keeps the
#: two equal. Built-ins also work from a source checkout that is not installed.
BUILTIN_CAPABILITIES: dict[str, str] = {
    "ArchitectureContracts.dependency-cruiser": "ter.adapters.driven.dependency_cruiser:DependencyCruiserContracts",
    "ArchitectureContracts.import-linter": "ter.adapters.driven.import_linter:ImportLinterContracts",
    "ControlLimitsSource.json": "ter.adapters.driven.control_limits:JsonControlLimits",
    "Embedder.hashing": "ter.adapters.driven.embedders:HashingEmbedder",
    "EventLog.jsonl": "ter.adapters.driven.event_log:JsonlEventLog",
    "OutcomeSource.junit": "ter.adapters.driven.junit:JUnitOutcomeSource",
    "PriceBook.json": "ter.adapters.driven.pricing:JsonPriceBook",
    "RepositoryEvidence.git": "ter.adapters.driven.repository:GitRepositoryEvidence",
    "RepositoryEvidence.lexical": "ter.adapters.driven.repository:LexicalRepositoryEvidence",
    "RepositoryEvidence.python-ast": "ter.adapters.driven.repository:PythonSyntaxEvidence",
    "RepositoryEvidence.syntax": "ter.adapters.driven.repository:SourceSyntaxEvidence",
    "RoutingProfiles.json": "ter.adapters.driven.routing_profiles:JsonRoutingProfiles",
    "SessionSource.claude-code": "ter.adapters.driven.claude_code:ClaudeCodeJsonlSource",
    "SessionSource.gare": "ter.adapters.driven.gare:GareRunSource",
    "TerScorer.ter3": "ter.adapters.driven.ter3:Ter3Scorer",
    "Tokenizer.regex": "ter.adapters.driven.tokenizers:RegexTokenizer",
    "Tokenizer.tiktoken": "ter.adapters.driven.tokenizers:TiktokenTokenizer",
}

BUILTIN_ORIGIN = "built-in"


def builtin_detectors(
    detectors: DetectorRegistry = DEFAULT_REGISTRY,
) -> dict[str, str]:
    """TER's own waste detectors as capabilities, ``WasteDetector.<id>``.

    Derived from the domain's detector catalogue, so a detector added to
    :data:`~ter.domain.lean.detectors.DEFAULT_REGISTRY` is listed here with
    no second table to keep in step.
    """
    return {
        f"WasteDetector.{d.id}": f"{type(d).__module__}:{type(d).__qualname__}"
        for d in detectors
    }


class EntryLike(Protocol):
    """What the registry needs of an entry point (``importlib.metadata.EntryPoint``)."""

    @property
    def name(self) -> str: ...

    @property
    def value(self) -> str: ...

    def load(self) -> Any: ...


@dataclass(frozen=True)
class _Target:
    """A built-in ``module:attribute`` reference, loaded like an entry point."""

    name: str
    value: str

    def load(self) -> Any:
        module, _, attribute = self.value.partition(":")
        loaded: Any = importlib.import_module(module)
        for part in attribute.split(".") if attribute else ():
            loaded = getattr(loaded, part)
        return loaded


def installed_entry_points() -> tuple[EntryLike, ...]:
    """Every installed entry point in the ``ter.capabilities`` group."""
    from importlib.metadata import entry_points

    return tuple(entry_points(group=CAPABILITY_GROUP))


def _origin(entry: EntryLike) -> str:
    dist = getattr(entry, "dist", None)
    name = getattr(dist, "name", None)
    return f"entry point ({name})" if name else "entry point"


def _protocols(port: type[object]) -> tuple[type[object], ...]:
    """The port and the Protocols it extends (a plugin Protocol may only
    re-export a domain Protocol, as ``WasteDetectorPlugin`` does)."""
    return tuple(
        k
        for k in port.__mro__
        if getattr(k, "_is_protocol", False) and k not in (Protocol, Generic)
    )


def _methods(port: type[object]) -> tuple[str, ...]:
    """The callable members a port Protocol declares."""
    return tuple(
        sorted(
            {
                n
                for k in _protocols(port)
                for n, v in vars(k).items()
                if not n.startswith("_") and callable(v)
            }
        )
    )


def _members(port: type[object]) -> tuple[str, ...]:
    annotated: set[str] = set()
    for k in _protocols(port):
        annotations: Mapping[str, object] = vars(k).get("__annotations__", {})
        annotated |= {n for n in annotations if not n.startswith("_")}
    return tuple(sorted(set(_methods(port)) | annotated))


def _self_assignments(code: CodeType) -> set[str]:
    """Attributes a method assigns on its first argument (``self.name = ...``).

    Reads the bytecode: a ``STORE_ATTR`` whose target was just loaded from the
    first local. Reads such as ``config.name`` do not count.
    """
    if code.co_argcount == 0:
        return set()
    this = code.co_varnames[0]
    assigned: set[str] = set()
    previous: dis.Instruction | None = None
    for instruction in dis.get_instructions(code):
        if (
            instruction.opname == "STORE_ATTR"
            and previous is not None
            and previous.opname.startswith("LOAD_FAST")
            and (
                previous.argval == this
                or (isinstance(previous.argval, tuple) and previous.argval[-1] == this)
            )
        ):
            assigned.add(str(instruction.argval))
        previous = instruction
    return assigned


def _undeclared(cls: type[object], port: type[object]) -> tuple[str, ...]:
    """Port members a class neither defines, annotates nor assigns in ``__init__``.

    A static check: nothing is instantiated, so no adapter does IO. An
    attribute assigned on ``self`` in ``__init__`` or ``__post_init__``
    (``self.name = ...``) counts as declared; merely reading a name of the
    same spelling (``config.name``) does not.
    """
    declared: set[str] = set()
    for klass in cls.__mro__:
        declared |= set(vars(klass).get("__annotations__", {}))
        for method in ("__init__", "__post_init__"):
            code = getattr(vars(klass).get(method), "__code__", None)
            if isinstance(code, CodeType):
                declared |= _self_assignments(code)
    return tuple(m for m in _members(port) if not hasattr(cls, m) and m not in declared)


@dataclass
class _Slot:
    capability: Capability
    entry: EntryLike
    factory: Callable[..., object] | None = None


class CapabilityRegistry:
    """Capabilities by port and adapter name; see the module docstring."""

    def __init__(
        self,
        *,
        builtins: Mapping[str, str] = BUILTIN_CAPABILITIES,
        discover: Callable[[], Iterable[EntryLike]] | None = installed_entry_points,
        ports: Mapping[str, type[object]] = CAPABILITY_KINDS,
    ) -> None:
        self._ports = dict(ports)
        self._discover = discover
        self._discovered = discover is None
        self._slots: dict[tuple[str, str], _Slot] = {}
        self._problems: list[CapabilityProblem] = []
        for key, target in builtins.items():
            self._admit(_Target(key, target), BUILTIN_ORIGIN)

    # -- registration ------------------------------------------------------

    def _problem(self, key: str, target: str, reason: str) -> None:
        problem = CapabilityProblem(key, target, reason)
        if problem not in self._problems:
            self._problems.append(problem)

    def _admit(self, entry: EntryLike, origin: str) -> None:
        try:
            key, target = str(entry.name), str(entry.value)
        except Exception as exc:  # a malformed entry object must not stop discovery
            self._problem("?", "?", f"unreadable entry point: {exc}")
            return
        try:
            port, name = parse_capability_key(key)
        except CapabilityError as exc:
            self._problem(key, target, str(exc))
            return
        if port not in self._ports:
            known = ", ".join(sorted(self._ports))
            self._problem(key, target, f"unknown port {port!r} (known: {known})")
            return
        existing = self._slots.get((port, name))
        if existing is not None:
            if existing.capability.target != target:
                self._problem(
                    key,
                    target,
                    f"{key} is already registered by {existing.capability.origin} "
                    f"as {existing.capability.target}; this entry is ignored",
                )
            return
        self._slots[(port, name)] = _Slot(Capability(port, name, target, origin), entry)

    def _ensure_discovered(self) -> None:
        if self._discovered:
            return
        self._discovered = True
        assert self._discover is not None
        try:
            entries = tuple(self._discover())
        except Exception as exc:  # broken package metadata must not crash TER
            self._problem(CAPABILITY_GROUP, "-", f"entry point discovery failed: {exc}")
            return
        for entry in entries:
            self._admit(entry, _origin(entry))

    # -- queries -----------------------------------------------------------

    @property
    def problems(self) -> tuple[CapabilityProblem, ...]:
        """Registered capabilities that cannot be used, after discovery."""
        self._ensure_discovered()
        return tuple(self._problems)

    def capabilities(self, port: str | None = None) -> tuple[Capability, ...]:
        """Every usable-looking registration (not yet loaded), sorted by key."""
        self._ensure_discovered()
        return tuple(
            slot.capability
            for (p, _), slot in sorted(self._slots.items())
            if port is None or p == port
        )

    def names(self, port: str) -> tuple[str, ...]:
        return tuple(c.name for c in self.capabilities(port))

    def _slot(self, port: str, name: str) -> _Slot:
        if port not in self._ports:
            raise UnknownCapabilityError(f"unknown port {port!r}")
        slot = self._slots.get((port, name))
        if slot is None:
            self._ensure_discovered()
            slot = self._slots.get((port, name))
        if slot is None:
            available = ", ".join(self.names(port)) or "none"
            raise UnknownCapabilityError(
                f"no {port} capability named {name!r} (available: {available})"
            )
        return slot

    # -- loading -----------------------------------------------------------

    def factory(self, port: str, name: str) -> Callable[..., object]:
        """Load (once) the class or factory registered as ``<port>.<name>``.

        Raises :class:`CapabilityError` when it fails to import, is not
        callable, or is a class that lacks a method of the port.
        """
        slot = self._slot(port, name)
        if slot.factory is not None:
            return slot.factory
        cap = slot.capability
        try:
            loaded = slot.entry.load()
        except Exception as exc:
            reason = f"failed to load: {type(exc).__name__}: {exc}"
            self._problem(cap.key, cap.target, reason)
            raise CapabilityError(
                f"capability {cap.key} ({cap.target}) {reason}"
            ) from exc
        if not callable(loaded):
            reason = f"{type(loaded).__name__} object is not a class or factory"
            self._problem(cap.key, cap.target, reason)
            raise CapabilityError(f"capability {cap.key} ({cap.target}): {reason}")
        if isinstance(loaded, type):
            missing = [
                m
                for m in _methods(self._ports[port])
                if not callable(getattr(loaded, m, None))
            ]
            if missing:
                reason = (
                    f"does not satisfy the {port} port: missing {', '.join(missing)}"
                )
                self._problem(cap.key, cap.target, reason)
                raise CapabilityError(f"capability {cap.key} ({cap.target}) {reason}")
        factory: Callable[..., object] = loaded
        slot.factory = factory
        return factory

    def create(self, port: str, name: str, *args: object, **kwargs: object) -> object:
        """Build the capability and check the instance satisfies its port."""
        cap = self._slot(port, name).capability
        built = self.factory(port, name)(*args, **kwargs)
        protocol = self._ports[port]
        if not isinstance(built, protocol):
            missing = [m for m in _members(protocol) if not hasattr(built, m)]
            reason = f"does not satisfy the {port} port: missing {', '.join(missing) or 'members'}"
            self._problem(cap.key, cap.target, reason)
            raise CapabilityError(f"capability {cap.key} ({cap.target}) {reason}")
        return built

    def check(self, port: str | None = None) -> tuple[CapabilityProblem, ...]:
        """Load every registered capability (of ``port``, if given) and return
        every problem found so far.

        Besides loading, a class is checked for every attribute of its port
        that :meth:`create` would demand of the instance (see
        :func:`_undeclared`). Never raises; nothing is instantiated, so no
        adapter does IO. A plain factory function can only be checked by
        :meth:`create`.
        """
        for cap in self.capabilities(port):
            try:
                loaded = self.factory(cap.port, cap.name)
            except CapabilityError:
                continue  # recorded as a problem
            if isinstance(loaded, type):
                missing = _undeclared(loaded, self._ports[cap.port])
                if missing:
                    self._problem(
                        cap.key,
                        cap.target,
                        f"does not satisfy the {cap.port} port: "
                        f"missing {', '.join(missing)}",
                    )
        return self.problems


@cache
def default_registry() -> CapabilityRegistry:
    """The installation's registry: built-ins (adapters and TER's own waste
    detectors) plus installed entry points."""
    return CapabilityRegistry(builtins={**BUILTIN_CAPABILITIES, **builtin_detectors()})


def detector_registry(
    registry: CapabilityRegistry | None = None,
    builtins: DetectorRegistry = DEFAULT_REGISTRY,
) -> DetectorRegistry:
    """The waste detectors an analysis runs: TER's catalogue, in catalogue
    order, then every other ``WasteDetector`` capability, by key.

    Installing a package that declares a ``WasteDetector.<id>`` entry point is
    what turns its detector on (TER-ARC-002). A plugin that fails to load,
    does not satisfy the protocol, or reuses a detector id already running is
    left out and recorded in :attr:`CapabilityRegistry.problems`; the
    analysis never fails because of a plugin.
    """
    registry = registry if registry is not None else default_registry()
    detectors = DetectorRegistry(builtins)
    for cap in registry.capabilities("WasteDetector"):
        if cap.origin == BUILTIN_ORIGIN and cap.name in builtins:
            continue
        try:
            detector = registry.create(cap.port, cap.name)
        except CapabilityError:
            continue  # recorded as a problem
        except Exception as exc:  # the plugin's own factory raised
            registry._problem(
                cap.key,
                cap.target,
                f"failed to construct: {type(exc).__name__}: {exc}",
            )
            continue
        assert isinstance(detector, WasteDetectorPlugin)  # checked by create
        if detector.id in detectors:
            registry._problem(
                cap.key,
                cap.target,
                f"detector id {detector.id!r} is already running; this plugin is ignored",
            )
            continue
        detectors.register(detector)
    return detectors


#: The repository engine used when a caller names none: the deterministic
#: lexical baseline (TER-EVD-002).
DEFAULT_REPOSITORY_ENGINE = "lexical"


def repository_evidence(
    root: str | Path,
    engine: str = DEFAULT_REPOSITORY_ENGINE,
    registry: CapabilityRegistry | None = None,
) -> RepositoryEvidence:
    """The repository engine ``RepositoryEvidence.<engine>`` for ``root``.

    Repository engines are capabilities like any adapter (TER-ARC-007): TER's
    own (``lexical``, ``git``, ``python-ast``, ``syntax``) are built in, and an installed
    package adds one with a ``RepositoryEvidence.<name>`` entry point whose
    class takes the repository root. Raises :class:`CapabilityError` for an
    unknown or broken engine, and the engine's own error (such as
    ``NotAWorkTreeError`` from ``git``) when it cannot serve ``root``.
    """
    registry = registry if registry is not None else default_registry()
    built = registry.create("RepositoryEvidence", engine, Path(root))
    assert isinstance(built, RepositoryEvidence)  # checked by create
    return built
