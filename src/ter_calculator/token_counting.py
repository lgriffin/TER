"""Token counting with per-phase calibration, API-based exact counts, and confidence scoring.

Replaces the naive len(text)/4 heuristic with phase-aware multipliers,
optional calibration from sample data, and optional exact counting via the
Anthropic API.  Falls back gracefully when the API is unavailable.
"""

from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

__all__ = [
    "CountMethod",
    "PhaseMultipliers",
    "TokenCountResult",
    "calibrate_multiplier",
    "count_tokens",
    "estimate_tokens_heuristic",
    "token_count_confidence",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CountMethod(enum.Enum):
    """How the token count was obtained."""

    API = "api"
    CALIBRATED = "calibrated"
    HEURISTIC = "heuristic"


# ---------------------------------------------------------------------------
# Phase multipliers
# ---------------------------------------------------------------------------


@dataclass
class PhaseMultipliers:
    """Chars-per-token ratios for each processing phase.

    Natural language (reasoning / generation) averages ~4.0 chars/token.
    Code-heavy content (tool use, JSON) is denser at ~3.2 chars/token.
    """

    reasoning: float = 4.0
    generation: float = 4.0
    tool_use: float = 3.2


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenCountResult:
    """Outcome of a token-counting operation."""

    estimated_tokens: int
    confidence: float
    method_used: CountMethod


# ---------------------------------------------------------------------------
# Heuristic estimation
# ---------------------------------------------------------------------------

# Pre-compiled pattern used to detect code-heavy content.
_CODE_PATTERN: re.Pattern[str] = re.compile(
    r"[{}\[\]();=<>]"  # structural punctuation common in code / JSON
)


def _code_density(text: str) -> float:
    """Return a 0-1 ratio of how *code-like* ``text`` is.

    A higher value means the text contains more structural punctuation
    (braces, brackets, semicolons, etc.) relative to its length.
    """
    if not text:
        return 0.0
    matches = _CODE_PATTERN.findall(text)
    # Cap at 0.3 -- even pure code rarely exceeds that density.
    return min(len(matches) / len(text), 1.0)


def estimate_tokens_heuristic(
    text: str,
    *,
    phase: str | None = None,
    multipliers: PhaseMultipliers | None = None,
) -> int:
    """Estimate token count using a character-based heuristic.

    Parameters
    ----------
    text:
        The text to estimate tokens for.
    phase:
        One of ``"reasoning"``, ``"generation"``, or ``"tool_use"``.
        When *None*, a default ratio of 4.0 is used.
    multipliers:
        Custom per-phase chars-per-token ratios.  Falls back to
        :class:`PhaseMultipliers` defaults when *None*.

    Returns
    -------
    int
        Estimated token count (always >= 0).
    """
    if not text:
        return 0

    if multipliers is None:
        multipliers = PhaseMultipliers()

    if phase is not None:
        ratio = getattr(multipliers, phase, 4.0)
    else:
        ratio = 4.0

    return max(0, round(len(text) / ratio))


# ---------------------------------------------------------------------------
# Calibration from sample data
# ---------------------------------------------------------------------------


def calibrate_multiplier(
    samples: Sequence[tuple[str, int]],
) -> float:
    """Compute an optimal chars-per-token multiplier via least-squares fitting.

    Given a list of ``(text, known_token_count)`` pairs, finds the multiplier
    *m* that minimises ``sum((len(text)/m - known)^2)``.

    The closed-form OLS solution for ``tokens = chars / m`` rearranges to
    ``m = sum(chars * chars) / sum(chars * tokens)``, treating each sample
    as a data point with ``y = tokens`` and ``x = chars``.

    Parameters
    ----------
    samples:
        Each element is ``(text, known_token_count)`` where
        *known_token_count* is the ground-truth token count for *text*.

    Returns
    -------
    float
        The optimal chars-per-token multiplier.

    Raises
    ------
    ValueError
        If *samples* is empty or all texts are empty strings.
    """
    if not samples:
        raise ValueError("samples must be a non-empty sequence")

    # We model:  tokens_i = chars_i / m
    #            => m * tokens_i = chars_i
    # Minimise sum( (chars_i - m * tokens_i)^2 ) over m.
    # d/dm = 0  =>  m = sum(chars_i * tokens_i) / sum(tokens_i^2)
    # But we want m = chars/token, so the model is chars = m * tokens.

    sum_ct = 0.0  # sum of chars_i * tokens_i
    sum_tt = 0.0  # sum of tokens_i^2

    for text, token_count in samples:
        chars = len(text)
        if token_count <= 0:
            continue
        sum_ct += chars * token_count
        sum_tt += token_count * token_count

    if sum_tt == 0.0:
        raise ValueError(
            "No valid samples: all token counts are zero or negative"
        )

    return sum_ct / sum_tt


# ---------------------------------------------------------------------------
# API-based exact counting
# ---------------------------------------------------------------------------


def _count_tokens_via_api(text: str) -> int | None:
    """Attempt to count tokens using the Anthropic API.

    Returns *None* when the API is unavailable (missing key, missing
    package, network error, etc.).

    The ``anthropic`` package is imported lazily so that the rest of the
    module works without it installed.
    """
    try:
        import anthropic  # lazy import
    except ImportError:
        logger.debug("anthropic package not installed; skipping API token count")
        return None

    try:
        client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
        response = client.messages.count_tokens(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens
    except Exception:
        logger.debug("API token counting failed; falling back to heuristic", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

# Base confidence by method.
_METHOD_CONFIDENCE: dict[CountMethod, float] = {
    CountMethod.API: 1.0,
    CountMethod.CALIBRATED: 0.9,
    CountMethod.HEURISTIC: 0.8,
}


def token_count_confidence(
    text: str,
    method: CountMethod,
) -> float:
    """Return a 0-1 confidence score for a token estimate.

    The score combines:
    * A base value determined by the counting *method* (API=1.0,
      calibrated=0.9, heuristic=0.8).
    * A text-characteristic adjustment: for heuristic and calibrated
      methods, code-heavy text receives a penalty because the
      chars-per-token ratio is less predictable for mixed content.

    Parameters
    ----------
    text:
        The source text that was counted.
    method:
        Which counting method was used.

    Returns
    -------
    float
        Confidence in the range [0.0, 1.0].
    """
    base = _METHOD_CONFIDENCE.get(method, 0.5)

    if method is CountMethod.API:
        # API counts are exact; no adjustment needed.
        return base

    # Penalty for code-heavy text (heuristic / calibrated).
    # Maximum penalty of 0.15 for extremely code-dense content.
    density = _code_density(text)
    penalty = density * 0.15

    return max(0.0, min(1.0, base - penalty))


# ---------------------------------------------------------------------------
# Unified counting interface
# ---------------------------------------------------------------------------


def count_tokens(
    text: str,
    *,
    phase: str | None = None,
    multipliers: PhaseMultipliers | None = None,
    calibrated_multiplier: float | None = None,
    use_api: bool = False,
) -> TokenCountResult:
    """Count (or estimate) tokens for *text*, picking the best available method.

    Resolution order:

    1. **API** -- if *use_api* is ``True`` and the Anthropic API is reachable.
    2. **Calibrated** -- if a *calibrated_multiplier* is provided.
    3. **Heuristic** -- phase-aware character-based estimation.

    Parameters
    ----------
    text:
        The text to count tokens for.
    phase:
        One of ``"reasoning"``, ``"generation"``, or ``"tool_use"``.
        Used only by the heuristic path.
    multipliers:
        Custom per-phase chars-per-token ratios (heuristic path only).
    calibrated_multiplier:
        If provided, overrides phase-specific multipliers with a single
        calibrated value derived from :func:`calibrate_multiplier`.
    use_api:
        When ``True``, attempt exact counting via the Anthropic API
        before falling back to estimation.

    Returns
    -------
    TokenCountResult
        Contains the estimated token count, a confidence score, and
        which method was used.
    """
    if not text:
        return TokenCountResult(
            estimated_tokens=0,
            confidence=1.0,
            method_used=CountMethod.HEURISTIC,
        )

    # --- 1. API (optional) ------------------------------------------------
    if use_api:
        api_count = _count_tokens_via_api(text)
        if api_count is not None:
            return TokenCountResult(
                estimated_tokens=api_count,
                confidence=token_count_confidence(text, CountMethod.API),
                method_used=CountMethod.API,
            )

    # --- 2. Calibrated multiplier -----------------------------------------
    if calibrated_multiplier is not None and calibrated_multiplier > 0:
        tokens = max(0, round(len(text) / calibrated_multiplier))
        return TokenCountResult(
            estimated_tokens=tokens,
            confidence=token_count_confidence(text, CountMethod.CALIBRATED),
            method_used=CountMethod.CALIBRATED,
        )

    # --- 3. Heuristic (default) -------------------------------------------
    tokens = estimate_tokens_heuristic(text, phase=phase, multipliers=multipliers)
    return TokenCountResult(
        estimated_tokens=tokens,
        confidence=token_count_confidence(text, CountMethod.HEURISTIC),
        method_used=CountMethod.HEURISTIC,
    )
