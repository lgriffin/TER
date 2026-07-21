from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ter_calculator.cli import main
from ter_calculator.history_store import HistoryRecord, TERHistoryStore
from ter_calculator.production import RuntimeConfig, inspect_database


def _record() -> HistoryRecord:
    return HistoryRecord(
        session_id="s1",
        project="demo",
        timestamp=1.0,
        aggregate_ter=0.9,
        phase_ter={"reasoning": 0.9},
        waste_breakdown={"repeat": 10},
        token_count=100,
        waste_tokens=10,
        cost_usd=0.1,
        waste_cost_usd=0.01,
    )


def test_runtime_config_rejects_invalid_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TER_BUSY_TIMEOUT_MS", "nope")
    with pytest.raises(ValueError, match="must be an integer"):
        RuntimeConfig.from_env("history.db")


def test_store_migration_permissions_and_health(tmp_path: Path):
    path = tmp_path / "state" / "history.db"
    store = TERHistoryStore(path)
    try:
        assert store.schema_version == 1
        assert store.integrity_check() == "ok"
        assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        store.close()
    report = inspect_database(path)
    assert report.healthy
    assert report.schema_version == 1
    if os.name == "posix":
        assert path.stat().st_mode & 0o077 == 0


def test_atomic_backup_and_restore(tmp_path: Path):
    source = tmp_path / "history.db"
    backup = tmp_path / "backups" / "history.db"
    restored = tmp_path / "restored.db"
    store = TERHistoryStore(source)
    try:
        store.put(_record())
        assert store.backup(backup) == backup
    finally:
        store.close()
    TERHistoryStore.restore(backup, restored)
    restored_store = TERHistoryStore(restored)
    try:
        assert restored_store.query()[0].session_id == "s1"
    finally:
        restored_store.close()


def test_doctor_and_history_backup_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    db = tmp_path / "history.db"
    backup = tmp_path / "backup.db"
    assert main(["doctor", "--db", str(db), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["healthy"] is True
    assert main(["history", "backup", str(backup), "--db", str(db)]) == 0
    assert backup.exists()
