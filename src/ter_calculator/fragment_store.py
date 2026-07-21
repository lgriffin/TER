"""Content-addressable fragment storage with SQLite persistence.

Implements the Fragment Sharding Engine from the Token Aware Microprompt
Orchestrator patent: decompose session context into discrete, deterministically
hashed fragments with semantic embeddings and token count annotations.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np

from .models import Fragment, SpanPhase, TokenSpan

_DEFAULT_DB_DIR = Path.home() / ".cache" / "ter"
_SCHEMA_VERSION = 1


def _default_db_path() -> Path:
    return _DEFAULT_DB_DIR / "fragments.db"


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def _compute_hash(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


class FragmentStore:
    """SQLite-backed content-addressable fragment store."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), timeout=5)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)"
        )
        row = cur.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if row is None:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS fragments ("
                "  id TEXT PRIMARY KEY,"
                "  text TEXT NOT NULL,"
                "  embedding BLOB,"
                "  token_count INTEGER NOT NULL,"
                "  phase TEXT NOT NULL,"
                "  origin_session TEXT NOT NULL,"
                "  created_at REAL NOT NULL,"
                "  ttl_seconds INTEGER NOT NULL DEFAULT 3600"
                ")"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_fragments_session "
                "ON fragments(origin_session)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_fragments_phase ON fragments(phase)"
            )
            cur.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (_SCHEMA_VERSION,),
            )
            self._conn.commit()

    def put(self, fragment: Fragment) -> None:
        emb_blob = (
            fragment.embedding.astype(np.float32).tobytes()
            if fragment.embedding is not None
            else None
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO fragments "
            "(id, text, embedding, token_count, phase, origin_session, "
            "created_at, ttl_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fragment.id,
                fragment.text,
                emb_blob,
                fragment.token_count,
                fragment.phase.value,
                fragment.origin_session,
                fragment.created_at,
                fragment.ttl_seconds,
            ),
        )
        self._conn.commit()

    def put_many(self, fragments: Sequence[Fragment]) -> None:
        rows = []
        for f in fragments:
            emb_blob = (
                f.embedding.astype(np.float32).tobytes()
                if f.embedding is not None
                else None
            )
            rows.append(
                (
                    f.id,
                    f.text,
                    emb_blob,
                    f.token_count,
                    f.phase.value,
                    f.origin_session,
                    f.created_at,
                    f.ttl_seconds,
                )
            )
        self._conn.executemany(
            "INSERT OR REPLACE INTO fragments "
            "(id, text, embedding, token_count, phase, origin_session, "
            "created_at, ttl_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def get(self, fragment_id: str) -> Fragment | None:
        row = self._conn.execute(
            "SELECT id, text, embedding, token_count, phase, origin_session, "
            "created_at, ttl_seconds FROM fragments WHERE id = ?",
            (fragment_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_fragment(row)

    def get_many(self, fragment_ids: list[str]) -> list[Fragment]:
        if not fragment_ids:
            return []
        placeholders = ",".join("?" for _ in fragment_ids)
        rows = self._conn.execute(
            f"SELECT id, text, embedding, token_count, phase, origin_session, "
            f"created_at, ttl_seconds FROM fragments WHERE id IN ({placeholders})",
            fragment_ids,
        ).fetchall()
        return [self._row_to_fragment(r) for r in rows]

    def exists(self, fragment_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM fragments WHERE id = ?", (fragment_id,)
        ).fetchone()
        return row is not None

    def find_by_session(self, session_id: str) -> list[Fragment]:
        rows = self._conn.execute(
            "SELECT id, text, embedding, token_count, phase, origin_session, "
            "created_at, ttl_seconds FROM fragments WHERE origin_session = ?",
            (session_id,),
        ).fetchall()
        return [self._row_to_fragment(r) for r in rows]

    def find_by_phase(self, phase: SpanPhase) -> list[Fragment]:
        rows = self._conn.execute(
            "SELECT id, text, embedding, token_count, phase, origin_session, "
            "created_at, ttl_seconds FROM fragments WHERE phase = ?",
            (phase.value,),
        ).fetchall()
        return [self._row_to_fragment(r) for r in rows]

    def gc(self, max_age_hours: float = 24.0) -> int:
        cutoff = time.time() - (max_age_hours * 3600)
        cur = self._conn.execute(
            "DELETE FROM fragments WHERE (created_at + ttl_seconds) < ?",
            (cutoff,),
        )
        self._conn.commit()
        return cur.rowcount

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM fragments").fetchone()
        return row[0] if row else 0

    def all_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT id FROM fragments").fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self._conn.close()

    def _row_to_fragment(self, row: tuple) -> Fragment:
        fid, text, emb_blob, token_count, phase, origin, created, ttl = row
        embedding = None
        if emb_blob is not None:
            embedding = np.frombuffer(emb_blob, dtype=np.float32).copy()
        return Fragment(
            id=fid,
            text=text,
            embedding=embedding,
            token_count=token_count,
            phase=SpanPhase(phase),
            origin_session=origin,
            created_at=created,
            ttl_seconds=ttl,
        )


class FragmentShardingEngine:
    """Decomposes token spans into content-addressable fragments."""

    def __init__(self, store: FragmentStore) -> None:
        self._store = store

    def shard(
        self,
        spans: list[TokenSpan],
        session_id: str,
        *,
        embed: bool = True,
    ) -> list[Fragment]:
        now = time.time()
        fragments: list[Fragment] = []
        new_fragments: list[Fragment] = []

        for span in spans:
            if not span.text or not span.text.strip():
                continue
            normalized = _normalize_text(span.text)
            frag_id = _compute_hash(normalized)

            existing = self._store.get(frag_id)
            if existing is not None:
                fragments.append(existing)
                continue

            frag = Fragment(
                id=frag_id,
                text=normalized,
                token_count=span.token_count,
                phase=span.phase,
                origin_session=session_id,
                created_at=now,
                embedding=span.embedding,
            )
            fragments.append(frag)
            new_fragments.append(frag)

        if embed and new_fragments:
            texts_to_embed = [f.text for f in new_fragments if f.embedding is None]
            if texts_to_embed:
                try:
                    from .intent import embed_texts

                    embeddings = embed_texts(texts_to_embed)
                    idx = 0
                    for frag in new_fragments:
                        if frag.embedding is None:
                            frag.embedding = embeddings[idx]
                            idx += 1
                except ImportError:
                    pass

        if new_fragments:
            self._store.put_many(new_fragments)

        return fragments
