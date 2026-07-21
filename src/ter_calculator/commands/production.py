"""Phase 5 production readiness commands."""

from __future__ import annotations

from ..history_store import TERHistoryStore
from ..production import RuntimeConfig, inspect_database, report_json


def _cmd_doctor(args) -> int:
    config = RuntimeConfig.from_env(args.db)
    # Opening the store applies safe idempotent migrations and permissions.
    store = TERHistoryStore(config.db_path)
    store.close()
    report = inspect_database(config.db_path)
    if args.output_format == "json":
        print(report_json(report))
    else:
        print("TER production readiness")
        print("========================")
        print(f"Database: {report.db_path}")
        print(f"Schema: {report.schema_version}")
        print(f"Integrity: {report.integrity}")
        print(f"Journal mode: {report.journal_mode}")
        print(f"Writable: {'yes' if report.writable else 'no'}")
        print(f"Secure permissions: {'yes' if report.secure_permissions else 'no'}")
        for issue in report.issues:
            print(f"Issue: {issue}")
    return 0 if report.healthy else 1
