"""Project-scoped repository memory with deterministic semantic retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

INDEX_VERSION = "2.0.14"
DEFAULT_INDEX = Path(".ter/memory-index.json")
_TEXT_EXTENSIONS = {
    ".py",
    ".pyi",
    ".md",
    ".rst",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".sh",
    ".sql",
}
_IGNORED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "data",
    "dist",
    "build",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,}|\d+(?:\.\d+)?")


@dataclass(frozen=True)
class MemoryChunk:
    chunk_id: str
    source_type: str
    path: str
    start_line: int
    end_line: int
    text: str
    fingerprint: str
    vector: dict[str, float]
    metadata: dict[str, Any]


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _vector(text: str, dimensions: int = 384) -> dict[str, float]:
    counts: defaultdict[str, float] = defaultdict(float)
    tokens = _tokens(text)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % dimensions
        counts[str(bucket)] += 1.0
    norm = math.sqrt(sum(value * value for value in counts.values()))
    return {key: value / norm for key, value in counts.items()} if norm else {}


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def _fingerprint(text: str) -> str:
    normalized = " ".join(_tokens(text))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _semantic_fingerprint(text: str) -> str:
    """Approximate structural equivalence while ignoring local naming/literals."""
    keywords = {
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "case",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "false",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "match",
        "none",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "true",
        "try",
        "while",
        "with",
        "yield",
    }
    normalized: list[str] = []
    for token in _tokens(text):
        if token in keywords:
            normalized.append(token)
        elif token.replace(".", "", 1).isdigit():
            normalized.append("<number>")
        else:
            normalized.append("<identifier>")
    return hashlib.sha256(" ".join(normalized).encode("utf-8")).hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        relative_parts = path.relative_to(root).parts
        if not path.is_file() or any(part in _IGNORED_PARTS for part in relative_parts):
            continue
        if path.suffix.lower() not in _TEXT_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > 1_000_000:
                continue
        except OSError:
            continue
        yield path


def _chunk_file(path: Path, root: Path, lines_per_chunk: int = 80) -> list[MemoryChunk]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    chunks: list[MemoryChunk] = []
    relative = path.relative_to(root).as_posix()
    for offset in range(0, len(lines), lines_per_chunk):
        block = lines[offset : offset + lines_per_chunk]
        text = "\n".join(block).strip()
        if not text:
            continue
        start = offset + 1
        end = offset + len(block)
        fp = _fingerprint(text)
        chunks.append(
            MemoryChunk(
                chunk_id=hashlib.sha256(
                    f"file:{relative}:{start}:{fp}".encode()
                ).hexdigest()[:20],
                source_type="file",
                path=relative,
                start_line=start,
                end_line=end,
                text=text,
                fingerprint=fp,
                vector=_vector(text),
                metadata={
                    "size_bytes": len(text.encode("utf-8")),
                    "semantic_fingerprint": _semantic_fingerprint(text),
                },
            )
        )
    return chunks


def _git_chunks(root: Path, limit: int = 200) -> list[MemoryChunk]:
    try:
        output = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "log",
                f"-{limit}",
                "--pretty=format:%H%x09%s%x09%b",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    chunks: list[MemoryChunk] = []
    for row in output.splitlines():
        parts = row.split("\t", 2)
        if len(parts) < 2:
            continue
        sha, subject = parts[:2]
        body = parts[2] if len(parts) > 2 else ""
        text = f"{subject}\n{body}".strip()
        fp = _fingerprint(text)
        chunks.append(
            MemoryChunk(
                chunk_id=f"commit-{sha[:16]}",
                source_type="commit",
                path=sha,
                start_line=0,
                end_line=0,
                text=text,
                fingerprint=fp,
                vector=_vector(text),
                metadata={"commit": sha, "subject": subject},
            )
        )
    return chunks


def build_index(root: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    project_root = Path(root).resolve()
    if not project_root.is_dir():
        raise ValueError(f"Repository root does not exist: {project_root}")
    chunks = [
        chunk
        for path in _iter_files(project_root)
        for chunk in _chunk_file(path, project_root)
    ]
    chunks.extend(_git_chunks(project_root))
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        groups[chunk.fingerprint].append(chunk.chunk_id)
    duplicate_groups = [ids for ids in groups.values() if len(ids) > 1]
    semantic_groups_map: defaultdict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        semantic = chunk.metadata.get("semantic_fingerprint")
        if semantic and len(_tokens(chunk.text)) >= 6:
            semantic_groups_map[str(semantic)].append(chunk.chunk_id)
    semantic_duplicate_groups = [
        ids for ids in semantic_groups_map.values() if len(ids) > 1
    ]
    payload = {
        "version": INDEX_VERSION,
        "root": str(project_root),
        "chunk_count": len(chunks),
        "file_count": len({c.path for c in chunks if c.source_type == "file"}),
        "commit_count": sum(c.source_type == "commit" for c in chunks),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "semantic_duplicate_group_count": len(semantic_duplicate_groups),
        "semantic_duplicate_groups": semantic_duplicate_groups,
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    destination = Path(output) if output else project_root / DEFAULT_INDEX
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=destination.name, dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    payload["index_path"] = str(destination)
    return payload


def load_index(path: str | Path) -> dict[str, Any]:
    index_path = Path(path)
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Repository memory index must contain a JSON object")
    payload: dict[str, Any] = raw
    if payload.get("version") != INDEX_VERSION:
        raise ValueError(f"Unsupported memory index version: {payload.get('version')}")
    return payload


def search_index(
    path: str | Path, query: str, limit: int = 8, minimum_score: float = 0.10
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    query_vector = _vector(query)
    payload = load_index(path)
    results = []
    for raw in payload.get("chunks", []):
        score = _cosine(query_vector, raw.get("vector", {}))
        if score < minimum_score:
            continue
        results.append(
            {
                "score": round(score, 6),
                "source_type": raw["source_type"],
                "path": raw["path"],
                "start_line": raw["start_line"],
                "end_line": raw["end_line"],
                "excerpt": raw["text"][:500],
                "fingerprint": raw["fingerprint"],
                "metadata": raw.get("metadata", {}),
            }
        )
    results.sort(key=lambda item: (-item["score"], item["path"], item["start_line"]))
    selected = results[:limit]
    return {
        "query": query,
        "matches": selected,
        "match_count": len(selected),
        "risk_flags": _risk_flags(selected, payload),
    }


def _risk_flags(
    matches: list[dict[str, Any]], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    duplicate_fingerprints = {
        payload["chunks"][index]["fingerprint"]
        for group in payload.get("duplicate_groups", [])
        for index, chunk in enumerate(payload.get("chunks", []))
        if chunk.get("chunk_id") in group
    }
    semantic_duplicate_ids = {
        chunk_id
        for group in payload.get("semantic_duplicate_groups", [])
        for chunk_id in group
    }
    chunk_ids_by_location = {
        (chunk.get("path"), chunk.get("start_line")): chunk.get("chunk_id")
        for chunk in payload.get("chunks", [])
    }
    flags = []
    for match in matches:
        if match["fingerprint"] in duplicate_fingerprints:
            flags.append(
                {
                    "type": "duplicate_pattern",
                    "path": match["path"],
                    "score": match["score"],
                }
            )
        if (
            chunk_ids_by_location.get((match["path"], match["start_line"]))
            in semantic_duplicate_ids
        ):
            flags.append(
                {
                    "type": "semantic_duplicate_pattern",
                    "path": match["path"],
                    "score": match["score"],
                }
            )
        text = match["excerpt"].lower()
        if any(
            term in text
            for term in ("failed", "failure", "bug", "regression", "revert", "fix")
        ):
            flags.append(
                {
                    "type": "prior_defect_or_fix",
                    "path": match["path"],
                    "score": match["score"],
                }
            )
    return flags


def inspect_index(path: str | Path) -> dict[str, Any]:
    payload = load_index(path)
    by_type = Counter(chunk["source_type"] for chunk in payload.get("chunks", []))
    return {
        "version": payload["version"],
        "root": payload["root"],
        "chunk_count": payload["chunk_count"],
        "file_count": payload["file_count"],
        "commit_count": payload["commit_count"],
        "duplicate_group_count": payload["duplicate_group_count"],
        "semantic_duplicate_group_count": payload.get(
            "semantic_duplicate_group_count", 0
        ),
        "sources": dict(by_type),
    }
