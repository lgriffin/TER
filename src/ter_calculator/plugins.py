"""Plugin system for extending the TER analyser.

Provides protocols for custom waste detectors, output formatters, and
pipeline middleware, plus a singleton registry for discovery and
registration.  Configuration is loaded from ``ter.toml`` / ``.terrc``
files (TOML format, parsed with :mod:`tomllib`).

Typical usage::

    from ter_calculator.plugins import PluginRegistry, waste_detector

    @waste_detector
    class MyDetector:
        name = "my_detector"
        description = "Finds custom waste patterns"

        def detect(self, spans: list[ClassifiedSpan]) -> list[WastePattern]:
            ...

    registry = PluginRegistry()
    registry.discover_plugins()
"""

from __future__ import annotations

import importlib
import logging
import tomllib
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ter_calculator.models import ClassifiedSpan, Session, TERResult, WastePattern

__all__ = [
    "OutputFormatter",
    "PipelineMiddleware",
    "PluginRegistry",
    "TERConfig",
    "WasteDetector",
    "output_formatter",
    "waste_detector",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class WasteDetector(Protocol):
    """Protocol that all waste-detector plugins must implement.

    A waste detector receives a list of classified spans and returns any
    waste patterns it identifies.
    """

    name: str
    description: str

    def detect(self, spans: list[ClassifiedSpan]) -> list[WastePattern]: ...


@runtime_checkable
class OutputFormatter(Protocol):
    """Protocol for output-formatting plugins.

    Formatters convert a :class:`~ter_calculator.models.TERResult` (or a
    list of results for comparison) into a string representation.
    """

    format_name: str

    def format_result(self, result: TERResult) -> str: ...

    def format_comparison(self, results: list[TERResult]) -> str: ...


@runtime_checkable
class PipelineMiddleware(Protocol):
    """Protocol for pre/post-processing middleware.

    Middleware can transform the session before analysis (``pre_process``)
    and/or adjust the result after computation (``post_process``).
    """

    name: str

    def pre_process(self, session: Session) -> Session: ...

    def post_process(self, result: TERResult) -> TERResult: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Filenames searched in order when looking for a config file.
_CONFIG_FILENAMES: tuple[str, ...] = ("ter.toml", ".terrc")

# Default threshold values drawn from the patent / spec.
_DEFAULT_SIMILARITY: float = 0.40
_DEFAULT_CONFIDENCE: float = 0.75
_DEFAULT_RESTATEMENT: float = 0.85

# Default phase weights (reasoning, tool_use, generation).
_DEFAULT_WEIGHT_REASONING: float = 0.3
_DEFAULT_WEIGHT_TOOL_USE: float = 0.4
_DEFAULT_WEIGHT_GENERATION: float = 0.3


@dataclass
class ThresholdsConfig:
    """Configurable thresholds for TER classification."""

    similarity: float = _DEFAULT_SIMILARITY
    confidence: float = _DEFAULT_CONFIDENCE
    restatement: float = _DEFAULT_RESTATEMENT


@dataclass
class WeightsConfig:
    """Per-phase weights for weighted aggregate TER computation."""

    reasoning: float = _DEFAULT_WEIGHT_REASONING
    tool_use: float = _DEFAULT_WEIGHT_TOOL_USE
    generation: float = _DEFAULT_WEIGHT_GENERATION


@dataclass
class OutputConfig:
    """Output formatting preferences."""

    format: str = "text"
    color: bool = True


@dataclass
class PluginsConfig:
    """Declarative plugin references loaded from config."""

    waste_detectors: list[str] = field(default_factory=list)
    formatters: list[str] = field(default_factory=list)
    middleware: list[str] = field(default_factory=list)


@dataclass
class TERConfig:
    """Central configuration for the TER analyser.

    Loaded from ``ter.toml`` or ``.terrc`` (TOML format).  Falls back to
    sensible defaults when no config file is found.
    """

    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)

    # Path to the file this config was loaded from, if any.
    config_path: Path | None = None

    # -----------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> TERConfig:
        """Load configuration from a TOML file.

        Search order:

        1. Explicit *path* argument.
        2. ``ter.toml`` / ``.terrc`` in the current working directory.
        3. ``ter.toml`` / ``.terrc`` in the user's home directory.

        If no file is found, a :class:`TERConfig` with default values is
        returned.

        Parameters
        ----------
        path:
            Explicit path to a TOML config file.  When given, no search
            is performed.

        Returns
        -------
        TERConfig
        """
        resolved = cls._resolve_config_path(path)
        if resolved is None:
            logger.debug("No config file found; using defaults")
            return cls()

        logger.debug("Loading config from %s", resolved)
        try:
            with open(resolved, "rb") as fh:
                raw = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            logger.warning("Failed to load config from %s: %s", resolved, exc)
            return cls()

        return cls._from_dict(raw, config_path=resolved)

    @classmethod
    def _resolve_config_path(cls, explicit: Path | None) -> Path | None:
        """Find the first existing config file."""
        if explicit is not None:
            return explicit if explicit.is_file() else None

        search_dirs = [Path.cwd(), Path.home()]
        for directory in search_dirs:
            for name in _CONFIG_FILENAMES:
                candidate = directory / name
                if candidate.is_file():
                    return candidate
        return None

    @classmethod
    def _from_dict(
        cls,
        raw: dict[str, Any],
        *,
        config_path: Path | None = None,
    ) -> TERConfig:
        """Construct a :class:`TERConfig` from a parsed TOML dict."""
        thresholds_raw = raw.get("thresholds", {})
        weights_raw = raw.get("weights", {})
        output_raw = raw.get("output", {})
        plugins_raw = raw.get("plugins", {})

        thresholds = ThresholdsConfig(
            similarity=float(thresholds_raw.get("similarity", _DEFAULT_SIMILARITY)),
            confidence=float(thresholds_raw.get("confidence", _DEFAULT_CONFIDENCE)),
            restatement=float(thresholds_raw.get("restatement", _DEFAULT_RESTATEMENT)),
        )

        weights = WeightsConfig(
            reasoning=float(weights_raw.get("reasoning", _DEFAULT_WEIGHT_REASONING)),
            tool_use=float(weights_raw.get("tool_use", _DEFAULT_WEIGHT_TOOL_USE)),
            generation=float(weights_raw.get("generation", _DEFAULT_WEIGHT_GENERATION)),
        )

        output = OutputConfig(
            format=str(output_raw.get("format", "text")),
            color=bool(output_raw.get("color", True)),
        )

        plugins = PluginsConfig(
            waste_detectors=list(plugins_raw.get("waste_detectors", [])),
            formatters=list(plugins_raw.get("formatters", [])),
            middleware=list(plugins_raw.get("middleware", [])),
        )

        return cls(
            thresholds=thresholds,
            weights=weights,
            output=output,
            plugins=plugins,
            config_path=config_path,
        )

    # -----------------------------------------------------------------
    # CLI override
    # -----------------------------------------------------------------

    def merge_cli_args(self, args: Namespace) -> TERConfig:
        """Return a new config with CLI arguments overriding file values.

        Only attributes that are explicitly set (not ``None``) on *args*
        override the current config.  Unknown attributes are silently
        ignored.

        Parameters
        ----------
        args:
            Parsed CLI arguments (from :mod:`argparse`).

        Returns
        -------
        TERConfig
            A new config instance with merged values.
        """
        thresholds = ThresholdsConfig(
            similarity=_first_not_none(
                getattr(args, "similarity_threshold", None),
                self.thresholds.similarity,
            ),
            confidence=_first_not_none(
                getattr(args, "confidence_threshold", None),
                self.thresholds.confidence,
            ),
            restatement=_first_not_none(
                getattr(args, "restatement_threshold", None),
                self.thresholds.restatement,
            ),
        )

        # Phase weights may arrive as a comma-separated string or a list.
        raw_weights = getattr(args, "phase_weights", None)
        if raw_weights is not None:
            parsed = _parse_phase_weights(raw_weights)
            weights = WeightsConfig(
                reasoning=parsed[0],
                tool_use=parsed[1],
                generation=parsed[2],
            )
        else:
            weights = WeightsConfig(
                reasoning=self.weights.reasoning,
                tool_use=self.weights.tool_use,
                generation=self.weights.generation,
            )

        output = OutputConfig(
            format=_first_not_none(
                getattr(args, "format", None),
                self.output.format,
            ),
            color=_first_not_none(
                getattr(args, "color", None),
                self.output.color,
            ),
        )

        return TERConfig(
            thresholds=thresholds,
            weights=weights,
            output=output,
            plugins=PluginsConfig(
                waste_detectors=list(self.plugins.waste_detectors),
                formatters=list(self.plugins.formatters),
                middleware=list(self.plugins.middleware),
            ),
            config_path=self.config_path,
        )


# ---------------------------------------------------------------------------
# Plugin registry (singleton)
# ---------------------------------------------------------------------------


class PluginRegistry:
    """Singleton registry for waste detectors, formatters, and middleware.

    Obtain the shared instance via ``PluginRegistry()`` -- repeated
    instantiation returns the same object.  Call :meth:`reset` in tests
    to clear all registrations.
    """

    _instance: PluginRegistry | None = None

    def __new__(cls) -> PluginRegistry:
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._waste_detectors = []
            inst._formatters = {}
            inst._middleware = []
            cls._instance = inst
        return cls._instance

    # -- Registration ----------------------------------------------------

    def register_waste_detector(self, detector: WasteDetector) -> None:
        """Register a custom waste detector.

        Parameters
        ----------
        detector:
            An object satisfying the :class:`WasteDetector` protocol.

        Raises
        ------
        TypeError
            If *detector* does not implement the required interface.
        """
        if not isinstance(detector, WasteDetector):
            raise TypeError(
                f"{type(detector).__name__} does not satisfy the "
                "WasteDetector protocol"
            )
        # Avoid duplicate registration by name.
        if any(d.name == detector.name for d in self._waste_detectors):
            logger.debug(
                "Waste detector '%s' already registered; skipping", detector.name
            )
            return
        self._waste_detectors.append(detector)
        logger.debug("Registered waste detector '%s'", detector.name)

    def register_formatter(self, formatter: OutputFormatter) -> None:
        """Register a custom output formatter.

        Parameters
        ----------
        formatter:
            An object satisfying the :class:`OutputFormatter` protocol.

        Raises
        ------
        TypeError
            If *formatter* does not implement the required interface.
        """
        if not isinstance(formatter, OutputFormatter):
            raise TypeError(
                f"{type(formatter).__name__} does not satisfy the "
                "OutputFormatter protocol"
            )
        name = formatter.format_name
        if name in self._formatters:
            logger.debug(
                "Formatter '%s' already registered; replacing", name
            )
        self._formatters[name] = formatter
        logger.debug("Registered output formatter '%s'", name)

    def register_middleware(self, middleware: PipelineMiddleware) -> None:
        """Register pipeline middleware.

        Parameters
        ----------
        middleware:
            An object satisfying the :class:`PipelineMiddleware` protocol.

        Raises
        ------
        TypeError
            If *middleware* does not implement the required interface.
        """
        if not isinstance(middleware, PipelineMiddleware):
            raise TypeError(
                f"{type(middleware).__name__} does not satisfy the "
                "PipelineMiddleware protocol"
            )
        if any(m.name == middleware.name for m in self._middleware):
            logger.debug(
                "Middleware '%s' already registered; skipping", middleware.name
            )
            return
        self._middleware.append(middleware)
        logger.debug("Registered middleware '%s'", middleware.name)

    # -- Retrieval -------------------------------------------------------

    def get_waste_detectors(self) -> list[WasteDetector]:
        """Return all registered waste detectors."""
        return list(self._waste_detectors)

    def get_formatter(self, name: str) -> OutputFormatter | None:
        """Look up a formatter by its ``format_name``.

        Parameters
        ----------
        name:
            The format name (e.g. ``"html"``, ``"json"``).

        Returns
        -------
        OutputFormatter | None
            The registered formatter, or ``None`` if not found.
        """
        return self._formatters.get(name)

    def get_middleware(self) -> list[PipelineMiddleware]:
        """Return all registered middleware in registration order."""
        return list(self._middleware)

    # -- Discovery -------------------------------------------------------

    def discover_plugins(self, package: str = "ter_plugins") -> None:
        """Auto-discover plugins via :func:`importlib.metadata.entry_points`.

        Looks for entry-point groups named:

        * ``{package}.waste_detectors``
        * ``{package}.formatters``
        * ``{package}.middleware``

        Each entry point should reference a callable that returns (or is)
        a plugin instance.

        Parameters
        ----------
        package:
            The entry-point group namespace prefix (default
            ``"ter_plugins"``).
        """
        try:
            from importlib.metadata import entry_points
        except ImportError:  # pragma: no cover – Python < 3.9
            logger.warning("importlib.metadata unavailable; skipping plugin discovery")
            return

        _GROUPS: dict[str, str] = {
            f"{package}.waste_detectors": "waste_detector",
            f"{package}.formatters": "formatter",
            f"{package}.middleware": "middleware",
        }

        for group, kind in _GROUPS.items():
            try:
                eps = entry_points(group=group)
            except TypeError:
                # Python 3.9/3.10 return a dict-like; fall back.
                all_eps = entry_points()
                eps = all_eps.get(group, [])  # type: ignore[union-attr]

            for ep in eps:
                try:
                    obj = ep.load()
                    # Support both class references and factory functions.
                    instance = obj() if isinstance(obj, type) else obj
                    self._register_by_kind(instance, kind, ep.name)
                except Exception:
                    logger.warning(
                        "Failed to load entry point '%s' from group '%s'",
                        ep.name,
                        group,
                        exc_info=True,
                    )

    def _register_by_kind(self, instance: Any, kind: str, ep_name: str) -> None:
        """Dispatch registration based on *kind* string."""
        if kind == "waste_detector":
            self.register_waste_detector(instance)
        elif kind == "formatter":
            self.register_formatter(instance)
        elif kind == "middleware":
            self.register_middleware(instance)
        else:
            logger.warning(
                "Unknown plugin kind '%s' for entry point '%s'", kind, ep_name
            )

    def load_from_config(self, config_path: Path) -> None:
        """Load and register plugins declared in a TOML config file.

        The ``[plugins]`` section of the config can list dotted Python
        paths to plugin classes::

            [plugins]
            waste_detectors = ["my_pkg.MyDetector"]
            formatters = ["my_pkg.HTMLFormatter"]
            middleware = ["my_pkg.TimingMiddleware"]

        Each path is imported, instantiated (if it is a class), and
        registered.

        Parameters
        ----------
        config_path:
            Path to a ``ter.toml`` or ``.terrc`` file.
        """
        try:
            with open(config_path, "rb") as fh:
                raw = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            logger.warning(
                "Failed to read plugin config from %s: %s", config_path, exc
            )
            return

        plugins_raw = raw.get("plugins", {})

        _CONFIG_KEYS: list[tuple[str, str]] = [
            ("waste_detectors", "waste_detector"),
            ("formatters", "formatter"),
            ("middleware", "middleware"),
        ]

        for config_key, kind in _CONFIG_KEYS:
            for dotted_path in plugins_raw.get(config_key, []):
                try:
                    obj = _import_dotted_path(dotted_path)
                    instance = obj() if isinstance(obj, type) else obj
                    self._register_by_kind(instance, kind, dotted_path)
                except Exception:
                    logger.warning(
                        "Failed to load plugin '%s' from config", dotted_path,
                        exc_info=True,
                    )

    # -- Utility ---------------------------------------------------------

    def reset(self) -> None:
        """Clear all registrations.  Intended for use in tests."""
        self._waste_detectors.clear()
        self._formatters.clear()
        self._middleware.clear()


# ---------------------------------------------------------------------------
# Registration decorators
# ---------------------------------------------------------------------------

_F = Any  # type alias for decorated class


def waste_detector(cls: type[_F]) -> type[_F]:
    """Class decorator that auto-registers a waste detector.

    Usage::

        @waste_detector
        class MyDetector:
            name = "my_detector"
            description = "Finds Foo patterns"

            def detect(self, spans: list[ClassifiedSpan]) -> list[WastePattern]:
                ...

    The class is instantiated (with no arguments) and registered with
    the global :class:`PluginRegistry` at import time.

    Returns the class unchanged so it can still be used directly.
    """
    try:
        instance = cls()
        PluginRegistry().register_waste_detector(instance)
    except Exception:
        logger.warning(
            "Failed to auto-register waste detector '%s'",
            getattr(cls, "__name__", cls),
            exc_info=True,
        )
    return cls


def output_formatter(cls: type[_F]) -> type[_F]:
    """Class decorator that auto-registers an output formatter.

    Usage::

        @output_formatter
        class HTMLFormatter:
            format_name = "html"

            def format_result(self, result: TERResult) -> str:
                ...

            def format_comparison(self, results: list[TERResult]) -> str:
                ...

    The class is instantiated (with no arguments) and registered with
    the global :class:`PluginRegistry` at import time.

    Returns the class unchanged so it can still be used directly.
    """
    try:
        instance = cls()
        PluginRegistry().register_formatter(instance)
    except Exception:
        logger.warning(
            "Failed to auto-register output formatter '%s'",
            getattr(cls, "__name__", cls),
            exc_info=True,
        )
    return cls


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _import_dotted_path(dotted_path: str) -> Any:
    """Import an object from a dotted Python path like ``'pkg.mod.Class'``.

    The rightmost component is treated as an attribute on the module
    formed by everything to its left.

    Parameters
    ----------
    dotted_path:
        Fully-qualified dotted path (e.g. ``"my_plugin.CustomDetector"``).

    Returns
    -------
    Any
        The imported object.

    Raises
    ------
    ImportError
        If the module cannot be imported.
    AttributeError
        If the attribute does not exist on the module.
    """
    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(
            f"Invalid dotted path '{dotted_path}': must contain at least "
            "one dot separating module from attribute"
        )
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _first_not_none(*values: Any) -> Any:
    """Return the first value that is not ``None``.

    Raises
    ------
    ValueError
        If all values are ``None``.
    """
    for v in values:
        if v is not None:
            return v
    raise ValueError("All values are None")


def _parse_phase_weights(raw: str | list[float] | tuple[float, ...]) -> tuple[float, float, float]:
    """Parse phase weights from CLI input.

    Accepts a comma-separated string (``"0.3,0.4,0.3"``) or a sequence
    of three floats.

    Parameters
    ----------
    raw:
        The raw phase-weights value from CLI args.

    Returns
    -------
    tuple[float, float, float]
        ``(reasoning, tool_use, generation)`` weights.

    Raises
    ------
    ValueError
        If *raw* cannot be parsed into exactly three floats.
    """
    if isinstance(raw, str):
        parts = [float(p.strip()) for p in raw.split(",")]
    else:
        parts = [float(p) for p in raw]

    if len(parts) != 3:
        raise ValueError(
            f"phase_weights must contain exactly 3 values, got {len(parts)}"
        )
    return (parts[0], parts[1], parts[2])
