# TER v2.0.9 — Repository Memory and Risk Retrieval

Phase 9 implements Step 1 of the feed-forward roadmap: a private, project-scoped memory that indexes repository text and Git history, retrieves semantically similar code and prior fixes, and flags duplicate or defect-related patterns before implementation begins.

## Commands

```bash
ter memory index .
ter memory search "authentication retry loop"
ter memory inspect
```

The deterministic hashed-vector index is stored in `.ter/memory-index.json` by default. It requires no network service or embedding model, retains source paths and line ranges, excludes common confidential/generated directories, and can be rebuilt incrementally by rerunning `memory index`.

## Risk report

Search results include similarity scores, source provenance, excerpts, duplicate-pattern flags, and prior defect/fix indicators. This is the retrieval substrate for the next live feed-forward intervention phase.
