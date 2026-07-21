from pathlib import Path

from ter_calculator.repository_memory import build_index, inspect_index, search_index


def test_index_search_and_inspect(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text(
        "def retry_login():\n    # fix repeated authentication failure\n    return True\n"
    )
    (tmp_path / "README.md").write_text(
        "Authentication retries must reuse the existing backoff helper.\n"
    )
    result = build_index(tmp_path)
    assert result["file_count"] == 2
    report = inspect_index(tmp_path / ".ter" / "memory-index.json")
    assert report["chunk_count"] >= 2
    search = search_index(
        tmp_path / ".ter" / "memory-index.json", "authentication retry failure"
    )
    assert search["matches"]
    assert search["matches"][0]["path"] in {"README.md", "src/auth.py"}


def test_duplicate_groups_are_reported(tmp_path: Path):
    text = "def shared_pattern():\n    return 'same implementation'\n"
    (tmp_path / "a.py").write_text(text)
    (tmp_path / "b.py").write_text(text)
    result = build_index(tmp_path)
    assert result["duplicate_group_count"] == 1
    search = search_index(
        tmp_path / ".ter" / "memory-index.json", "shared pattern implementation"
    )
    assert any(flag["type"] == "duplicate_pattern" for flag in search["risk_flags"])
