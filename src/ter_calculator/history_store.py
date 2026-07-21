"""Persistent, privacy-conscious cross-session TER intelligence.

Only aggregate metrics and an optional normalized prompt fingerprint are stored.
Raw session content is never persisted by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_PATH = Path.home() / ".claude" / "ter" / "history.db"
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class HistoryRecord:
    session_id: str
    project: str
    timestamp: float
    aggregate_ter: float
    phase_ter: dict[str, float]
    waste_breakdown: dict[str, int]
    token_count: int
    waste_tokens: int
    cost_usd: float
    waste_cost_usd: float
    prompt_fingerprint: dict[str, float] | None = None


def prompt_fingerprint(text: str, dimensions: int = 128) -> dict[str, float]:
    """Create a deterministic hashed bag-of-words vector without storing prompt text."""
    counts: dict[int, float] = {}
    for token in _TOKEN_RE.findall(text.lower()):
        bucket = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % dimensions
        counts[bucket] = counts.get(bucket, 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    return {str(key): value / norm for key, value in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(key, 0.0) for key, value in a.items())


class TERHistoryStore:
    """SQLite store for aggregate TER session history."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.path = Path(db_path) if db_path else _DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS ter_sessions (
                session_id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                timestamp REAL NOT NULL,
                aggregate_ter REAL NOT NULL,
                phase_ter TEXT NOT NULL,
                waste_breakdown TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                waste_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                waste_cost_usd REAL NOT NULL,
                prompt_fingerprint TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ter_project_time
                ON ter_sessions(project, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_ter_score
                ON ter_sessions(aggregate_ter);
            """
        )
        self.connection.commit()

    def put(self, record: HistoryRecord) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO ter_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.session_id,
                record.project,
                record.timestamp,
                record.aggregate_ter,
                json.dumps(record.phase_ter, sort_keys=True),
                json.dumps(record.waste_breakdown, sort_keys=True),
                record.token_count,
                record.waste_tokens,
                record.cost_usd,
                record.waste_cost_usd,
                json.dumps(record.prompt_fingerprint, sort_keys=True)
                if record.prompt_fingerprint
                else None,
            ),
        )
        self.connection.commit()

    def query(
        self,
        *,
        project: str | None = None,
        since: float | None = None,
        until: float | None = None,
        min_ter: float | None = None,
        max_ter: float | None = None,
        limit: int = 100,
    ) -> list[HistoryRecord]:
        clauses, values = [], []
        for expression, value in (
            ("project = ?", project),
            ("timestamp >= ?", since),
            ("timestamp <= ?", until),
            ("aggregate_ter >= ?", min_ter),
            ("aggregate_ter <= ?", max_ter),
        ):
            if value is not None:
                clauses.append(expression)
                values.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(max(1, limit))
        rows = self.connection.execute(
            f"SELECT * FROM ter_sessions {where} ORDER BY timestamp DESC LIMIT ?",
            values,
        ).fetchall()
        return [self._row(row) for row in rows]

    def projects(self) -> list[str]:
        return [
            row[0]
            for row in self.connection.execute(
                "SELECT DISTINCT project FROM ter_sessions ORDER BY project"
            )
        ]

    def profile(self, project: str | None = None) -> dict[str, object]:
        records = self.query(project=project, limit=100000)
        if not records:
            return {"sessions": 0, "project": project}
        waste: dict[str, int] = {}
        for record in records:
            for key, value in record.waste_breakdown.items():
                waste[key] = waste.get(key, 0) + int(value)
        main_waste = max(waste, key=lambda name: waste[name]) if waste else None
        return {
            "project": project or "all",
            "sessions": len(records),
            "average_ter": sum(r.aggregate_ter for r in records) / len(records),
            "total_tokens": sum(r.token_count for r in records),
            "waste_tokens": sum(r.waste_tokens for r in records),
            "total_cost_usd": sum(r.cost_usd for r in records),
            "waste_cost_usd": sum(r.waste_cost_usd for r in records),
            "main_waste_source": main_waste,
            "waste_breakdown": waste,
        }

    def predict(self, prompt: str, project: str, *, k: int = 5) -> dict[str, object]:
        target = prompt_fingerprint(prompt)
        candidates = [
            r for r in self.query(project=project, limit=100000) if r.prompt_fingerprint
        ]
        ranked = sorted(
            ((_cosine(target, r.prompt_fingerprint or {}), r) for r in candidates),
            key=lambda item: item[0],
            reverse=True,
        )[: max(1, k)]
        useful = [(score, record) for score, record in ranked if score > 0]
        if not useful:
            return {
                "available": False,
                "project": project,
                "sample_size": len(candidates),
            }
        weight = sum(score for score, _ in useful)
        predicted = (
            sum(score * record.aggregate_ter for score, record in useful) / weight
        )
        return {
            "available": True,
            "project": project,
            "predicted_ter": predicted,
            "neighbors": len(useful),
            "sample_size": len(candidates),
            "confidence": "high" if len(candidates) >= 50 else "experimental",
            "recommendation": _prediction_recommendation(predicted),
        }

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> HistoryRecord:
        return HistoryRecord(
            session_id=row["session_id"],
            project=row["project"],
            timestamp=row["timestamp"],
            aggregate_ter=row["aggregate_ter"],
            phase_ter=json.loads(row["phase_ter"]),
            waste_breakdown=json.loads(row["waste_breakdown"]),
            token_count=row["token_count"],
            waste_tokens=row["waste_tokens"],
            cost_usd=row["cost_usd"],
            waste_cost_usd=row["waste_cost_usd"],
            prompt_fingerprint=json.loads(row["prompt_fingerprint"])
            if row["prompt_fingerprint"]
            else None,
        )


def _prediction_recommendation(score: float) -> str:
    if score >= 0.9:
        return "Prompt resembles historically efficient work; keep scope explicit."
    if score >= 0.75:
        return (
            "Moderate efficiency expected; name target files and acceptance criteria."
        )
    return "Low efficiency risk: narrow scope, specify target files, and define a stopping condition."


def waste_breakdown(result) -> dict[str, int]:
    totals: dict[str, int] = {}
    for pattern in result.waste_patterns:
        key = getattr(pattern, "pattern_type", None) or getattr(
            pattern, "category", None
        )
        name = getattr(key, "value", str(key or "unknown"))
        tokens = int(
            getattr(pattern, "token_count", 0) or getattr(pattern, "waste_tokens", 0)
        )
        totals[name] = totals.get(name, 0) + tokens
    return totals
