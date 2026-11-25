"""TER-RTE-001: TER references language models only through role names.

Model ids live in data: the routing profiles (``ter/data/routing_profiles``)
and the price book (``ter/data/price_book.json``, ADR 0003). No ``ter``
source module may spell one out in a string literal; the adapters that read
the data are allowed to, though today they do not need to. Docstrings and
comments are prose, not references, and are not checked.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "src" / "ter"
DATA = SOURCE / "data"

#: The adapters that read model ids from data.
ALLOWED = {
    SOURCE / "adapters" / "driven" / "routing_profiles.py",
    SOURCE / "adapters" / "driven" / "pricing" / "__init__.py",
}

#: Model id spellings of the providers TER meets, and the price book's
#: family aliases. Matched against whole string literals' words.
MODEL_ID = re.compile(
    r"(?<![\w-])("
    r"claude-(?:instant|haiku|sonnet|opus|fable|mythos|\d)[\w.:-]*"
    r"|gpt-(?:\d|4o)[\w.:-]*|o[134]-(?:mini|preview|pro)[\w.-]*"
    r"|gemini-\d[\w.:-]*|qwen\d[\w.:-]*|llama-?\d[\w.:-]*"
    r"|(?:codestral|mistral|mixtral|deepseek)-[\w.:-]+"
    r"|haiku|sonnet|opus"
    r")(?![\w-])",
    re.IGNORECASE,
)


def _docstrings(tree: ast.AST) -> set[int]:
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def model_ids_in(path: Path) -> list[str]:
    """``line: model id`` for every model id in a string literal of ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _docstrings(tree)
    return [
        f"{node.lineno}: {m.group(1)}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in skip
        for m in MODEL_ID.finditer(node.value)
    ]


@pytest.mark.req("TER-RTE-001")
def test_no_source_module_hard_codes_a_model_id() -> None:
    offenders = [
        hit
        for path in sorted(SOURCE.rglob("*.py"))
        if path not in ALLOWED
        for hit in (f"{path.relative_to(REPO)}:{h}" for h in model_ids_in(path))
    ]
    assert offenders == []


@pytest.mark.req("TER-RTE-001")
def test_the_check_catches_a_model_id(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        '"""Mentions claude-opus-4-6 in prose."""\n'
        'MODEL = "claude-sonnet-4-6"\n'
        'ALIAS = {"tier": "opus"}\n'
        'NAME = "claude-code-jsonl"\n'
        'ROLE = "escalate"\n',
        encoding="utf-8",
    )
    assert model_ids_in(probe) == ["2: claude-sonnet-4-6", "3: opus"]


@pytest.mark.req("TER-RTE-001")
def test_model_ids_live_in_the_data() -> None:
    profiles = sorted((DATA / "routing_profiles").glob("*.json"))
    assert profiles
    models = {
        role["model"]
        for path in profiles
        for role in json.loads(path.read_text(encoding="utf-8"))["roles"].values()
    }
    assert all(MODEL_ID.search(m) for m in models), models
    book = json.loads((DATA / "price_book.json").read_text(encoding="utf-8"))
    assert all(MODEL_ID.search(e["model"]) for e in book["prices"])
