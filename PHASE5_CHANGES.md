# Phase 5 — Production Hardening

TER v2.0.5 adds production-safe configuration validation, SQLite migrations and durability settings, health diagnostics, restrictive local permissions, and atomic backup/restore workflows.

## Commands

```bash
ter doctor --format json
ter history backup ~/.claude/ter/backups/history.db
ter history restore ~/.claude/ter/backups/history.db --force
```
