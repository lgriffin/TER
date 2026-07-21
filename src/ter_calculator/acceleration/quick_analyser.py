"""Extracted acceleration responsibility."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ter" / "analysis"
CACHE_VERSION = 1
DEFAULT_TTL_HOURS = 168
DEFAULT_WATCH_INTERVAL = 30
DEFAULT_WATCH_DIR = Path.home() / ".claude" / "projects"
EMBEDDING_DIM = 384
_DEFAULT_THRESHOLDS = {
    "similarity_threshold": 0.40,
    "confidence_threshold": 0.75,
    "restatement_threshold": 0.85,
}
_PHASE_WEIGHTS = {"reasoning": 0.3, "tool_use": 0.4, "generation": 0.3}
_MIN_KEYWORD_LEN = 3
_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "shall",
        "may",
        "might",
        "can",
        "must",
        "need",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "no",
        "if",
        "then",
        "else",
        "for",
        "of",
        "in",
        "on",
        "at",
        "to",
        "from",
        "by",
        "with",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "my",
        "your",
        "his",
        "her",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "all",
        "each",
        "every",
        "any",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "also",
        "only",
        "so",
        "up",
        "out",
        "about",
        "into",
        "over",
        "after",
        "before",
    }
)


class QuickAnalyser:
    """Fast approximate TER calculation using keyword heuristics.

    Skips the embedding step entirely, replacing cosine similarity with a
    keyword-overlap ratio.  This makes analysis near-instant (~1-2 seconds)
    at the cost of reduced accuracy.

    The keyword extraction uses simple TF-based scoring with no external
    dependencies:

    1. Tokenise user prompts into words, remove stop words, and count term
       frequencies.
    2. Select top-N keywords by frequency (ties broken alphabetically).
    3. For each span, compute ``keywords_found_in_span / total_keywords``
       as the alignment score (analogous to cosine similarity).
    4. Apply the standard threshold logic to classify spans and compute TER.
    """

    def __init__(self, top_n_keywords: int = 30) -> None:
        self.top_n_keywords = max(1, top_n_keywords)

    def analyse_quick(
        self,
        session_path: str,
        thresholds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run a fast approximate TER analysis on a session file.

        Parameters
        ----------
        session_path:
            Path to a JSONL session file.
        thresholds:
            Override thresholds.  Keys: ``similarity_threshold``,
            ``confidence_threshold``.  Defaults mirror the main classifier.

        Returns
        -------
        dict
            A dictionary matching the TERResult structure with fields:
            ``session_id``, ``aggregate_ter``, ``raw_ratio``,
            ``phase_scores``, ``total_tokens``, ``aligned_tokens``,
            ``waste_tokens``, ``waste_patterns``, ``method``.
        """
        effective_thresholds = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
        sim_threshold = effective_thresholds["similarity_threshold"]

        # -- 1. Parse session minimally -----------------------------------
        session_data = self._parse_session(session_path)
        if not session_data["spans"]:
            return self._empty_result(session_data["session_id"])

        # -- 2. Extract keywords from user prompts ------------------------
        keywords = self._extract_keywords(session_data["user_prompts"])
        if not keywords:
            # No meaningful keywords -- treat everything as aligned.
            return self._all_aligned_result(session_data)

        # -- 3. Score each span by keyword overlap ------------------------
        scored_spans: list[dict[str, Any]] = []
        for span in session_data["spans"]:
            score = self._keyword_overlap_score(span["text"], keywords)
            label = "aligned" if score >= sim_threshold else "waste"
            scored_spans.append({**span, "score": score, "label": label})

        # -- 4. Compute per-phase and aggregate TER -----------------------
        return self._compute_result(session_data["session_id"], scored_spans)

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _parse_session(session_path: str) -> dict[str, Any]:
        """Minimal JSONL parsing -- extracts spans and user prompts.

        This is a lightweight parser that avoids importing the full loader
        module, keeping QuickAnalyser self-contained for speed.
        """
        path = Path(session_path)
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {session_path}")

        messages: list[dict[str, Any]] = []
        session_id = ""

        # Merge sibling lines that share a requestId.
        # Claude Code writes one JSONL line per content block (thinking,
        # tool_use, text) for the same API response — all sharing the same
        # requestId.  Keeping only the "best" entry by output_tokens silently
        # dropped the other blocks.  Instead, merge their content lists.
        seen_requests: dict[str, dict[str, Any]] = {}

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not session_id:
                    session_id = entry.get("sessionId", "")

                msg = entry.get("message", {})
                if not msg:
                    continue

                request_id = msg.get("requestId") or entry.get("requestId")
                usage = msg.get("usage", {})

                if request_id:
                    if request_id not in seen_requests:
                        seen_requests[request_id] = dict(entry)
                        seen_requests[request_id].setdefault("message", {})
                    else:
                        base_msg = seen_requests[request_id].setdefault("message", {})
                        base_content = base_msg.get("content", [])
                        new_content = msg.get("content", [])
                        if isinstance(new_content, list) and new_content:
                            if isinstance(base_content, list):
                                base_content.extend(new_content)
                            else:
                                base_msg["content"] = list(new_content)
                        if not base_msg.get("usage") and usage:
                            base_msg["usage"] = usage
                else:
                    messages.append(entry)

        messages.extend(seen_requests.values())
        # Sort by timestamp if available.
        messages.sort(key=lambda m: m.get("timestamp", ""))

        # Extract user prompts and content spans.
        user_prompts: list[str] = []
        spans: list[dict[str, Any]] = []
        position = 0

        for entry in messages:
            msg = entry.get("message", {})
            role = msg.get("role", entry.get("type", ""))
            content = msg.get("content", "")

            # Handle string content.
            if isinstance(content, str) and content.strip():
                if role == "user":
                    user_prompts.append(content)
                else:
                    from ter_calculator.embedding_cache import estimate_tokens

                    spans.append(
                        {
                            "text": content,
                            "phase": "generation",
                            "position": position,
                            "token_count": estimate_tokens(content),
                        }
                    )
                    position += 1
                continue

            # Handle block-based content.
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue

                    block_type = block.get("type", "")
                    text = ""

                    if block_type == "text":
                        text = block.get("text", "")
                        phase = "generation"
                    elif block_type == "thinking":
                        text = block.get("thinking", "") or block.get("text", "")
                        phase = "reasoning"
                    elif block_type == "tool_use":
                        tool_input = block.get("input", {})
                        text = json.dumps(tool_input) if tool_input else ""
                        tool_name = block.get("name", "unknown")
                        text = f"{tool_name}: {text}"
                        phase = "tool_use"
                    elif block_type == "tool_result":
                        text = ""
                        result_content = block.get("content", "")
                        if isinstance(result_content, str):
                            text = result_content
                        elif isinstance(result_content, list):
                            text = " ".join(
                                b.get("text", "")
                                for b in result_content
                                if isinstance(b, dict)
                            )
                        phase = "tool_use"
                    else:
                        continue

                    if role == "user" and block_type == "text" and text.strip():
                        user_prompts.append(text)
                        continue

                    if text.strip():
                        from ter_calculator.embedding_cache import estimate_tokens

                        spans.append(
                            {
                                "text": text,
                                "phase": phase,
                                "position": position,
                                "token_count": estimate_tokens(text),
                            }
                        )
                        position += 1

        if not session_id:
            session_id = path.stem

        return {
            "session_id": session_id,
            "user_prompts": user_prompts,
            "spans": spans,
        }

    def _extract_keywords(self, prompts: list[str]) -> set[str]:
        """Extract top-N keywords from user prompts using TF scoring.

        Words are lowercased, stop words and very short tokens are removed.
        The most frequent remaining words are selected as keywords.
        """
        if not prompts:
            return set()

        combined = " ".join(prompts)
        # Tokenise: split on non-alphanumeric characters.
        words = [
            w.lower()
            for w in combined.replace("-", " ").replace("_", " ").split()
            if len(w) >= _MIN_KEYWORD_LEN
        ]

        # Remove stop words.
        words = [w for w in words if w not in _STOP_WORDS]

        if not words:
            return set()

        counter = Counter(words)
        # Select top-N by frequency; break ties alphabetically for determinism.
        top = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        return {word for word, _ in top[: self.top_n_keywords]}

    @staticmethod
    def _keyword_overlap_score(text: str, keywords: set[str]) -> float:
        """Compute the fraction of keywords found in *text*.

        Returns a float in [0.0, 1.0].
        """
        if not keywords:
            return 0.0

        text_lower = text.lower()
        found = sum(1 for kw in keywords if kw in text_lower)
        return found / len(keywords)

    @staticmethod
    def _compute_result(
        session_id: str,
        scored_spans: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate scored spans into a TERResult-compatible dict."""
        # Per-phase token counts.
        phase_aligned: dict[str, int] = {"reasoning": 0, "tool_use": 0, "generation": 0}
        phase_total: dict[str, int] = {"reasoning": 0, "tool_use": 0, "generation": 0}

        total_tokens = 0
        aligned_tokens = 0

        for span in scored_spans:
            phase = span["phase"]
            tc = span["token_count"]
            total_tokens += tc

            if phase not in phase_total:
                phase_total[phase] = 0
                phase_aligned[phase] = 0

            phase_total[phase] += tc

            if span["label"] == "aligned":
                aligned_tokens += tc
                phase_aligned[phase] += tc

        waste_tokens = total_tokens - aligned_tokens

        # Per-phase scores.
        phase_scores: dict[str, float] = {}
        for phase in ("reasoning", "tool_use", "generation"):
            pt = phase_total.get(phase, 0)
            pa = phase_aligned.get(phase, 0)
            phase_scores[phase] = pa / pt if pt > 0 else 1.0

        # Weighted aggregate TER.
        aggregate_ter = sum(
            _PHASE_WEIGHTS.get(phase, 0.0) * phase_scores[phase]
            for phase in phase_scores
        )

        # Raw ratio.
        raw_ratio = aligned_tokens / total_tokens if total_tokens > 0 else 1.0

        return {
            "session_id": session_id,
            "aggregate_ter": round(aggregate_ter, 4),
            "raw_ratio": round(raw_ratio, 4),
            "phase_scores": {k: round(v, 4) for k, v in phase_scores.items()},
            "total_tokens": total_tokens,
            "aligned_tokens": aligned_tokens,
            "waste_tokens": waste_tokens,
            "waste_patterns": [],
            "method": "quick_keyword",
        }

    @staticmethod
    def _empty_result(session_id: str) -> dict[str, Any]:
        """Return a TERResult-compatible dict for an empty session."""
        return {
            "session_id": session_id,
            "aggregate_ter": 1.0,
            "raw_ratio": 1.0,
            "phase_scores": {"reasoning": 1.0, "tool_use": 1.0, "generation": 1.0},
            "total_tokens": 0,
            "aligned_tokens": 0,
            "waste_tokens": 0,
            "waste_patterns": [],
            "method": "quick_keyword",
        }

    @staticmethod
    def _all_aligned_result(session_data: dict[str, Any]) -> dict[str, Any]:
        """Return a TERResult treating all tokens as aligned (no keywords)."""
        total = sum(s["token_count"] for s in session_data["spans"])
        return {
            "session_id": session_data["session_id"],
            "aggregate_ter": 1.0,
            "raw_ratio": 1.0,
            "phase_scores": {"reasoning": 1.0, "tool_use": 1.0, "generation": 1.0},
            "total_tokens": total,
            "aligned_tokens": total,
            "waste_tokens": 0,
            "waste_patterns": [],
            "method": "quick_keyword",
        }
