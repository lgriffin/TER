"""Production hardening utilities for TER runtime state."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    db_path: Path
    log_level: str = "WARNING"
    busy_timeout_ms: int = 5000
    backup_retention: int = 7

    @classmethod
    def from_env(cls, db_path: str | Path | None = None) -> "RuntimeConfig":
        raw_path: str | Path
        if db_path is not None:
            raw_path = db_path
        else:
            raw_path = os.getenv("TER_DB_PATH") or (
                Path.home() / ".claude" / "ter" / "history.db"
            )
        path = Path(raw_path)
        config = cls(
            db_path=path.expanduser(),
            log_level=os.getenv("TER_LOG_LEVEL", "WARNING").upper(),
            busy_timeout_ms=_env_int("TER_BUSY_TIMEOUT_MS", 5000),
            backup_retention=_env_int("TER_BACKUP_RETENTION", 7),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.log_level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError(f"Invalid TER_LOG_LEVEL: {self.log_level}")
        if not 100 <= self.busy_timeout_ms <= 120_000:
            raise ValueError("TER_BUSY_TIMEOUT_MS must be between 100 and 120000")
        if not 1 <= self.backup_retention <= 365:
            raise ValueError("TER_BACKUP_RETENTION must be between 1 and 365")
        if self.db_path.exists() and self.db_path.is_dir():
            raise ValueError(f"History database path is a directory: {self.db_path}")


@dataclass(frozen=True)
class HealthReport:
    healthy: bool
    db_path: str
    schema_version: int | None
    integrity: str
    writable: bool
    secure_permissions: bool
    journal_mode: str | None
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def secure_state_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        path.parent.chmod(0o700)
        if path.exists():
            path.chmod(0o600)


def inspect_database(path: str | Path) -> HealthReport:
    db_path = Path(path).expanduser()
    issues: list[str] = []
    if not db_path.exists():
        return HealthReport(
            False,
            str(db_path),
            None,
            "missing",
            False,
            False,
            None,
            ["database does not exist"],
        )

    writable = os.access(db_path, os.W_OK)
    if not writable:
        issues.append("database is not writable")
    secure_permissions = True
    if os.name == "posix":
        secure_permissions = db_path.stat().st_mode & 0o077 == 0
        if not secure_permissions:
            issues.append("database permissions allow group or other access")

    schema_version: int | None = None
    integrity = "unknown"
    journal_mode: str | None = None
    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
            row = connection.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            ).fetchone()
            schema_version = int(row[0]) if row else 0
        finally:
            connection.close()
    except (sqlite3.Error, OSError) as exc:
        issues.append(f"database inspection failed: {exc}")

    if integrity != "ok":
        issues.append(f"integrity check returned {integrity}")
    return HealthReport(
        not issues,
        str(db_path),
        schema_version,
        integrity,
        writable,
        secure_permissions,
        journal_mode,
        issues,
    )


def report_json(report: HealthReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
