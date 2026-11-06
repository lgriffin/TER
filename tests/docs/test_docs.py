"""Cheap documentation checks: links resolve, shown commands exist, index is fresh.

These run offline in well under a second. They read Markdown as text and ask
the real argument parsers for ``--help``; no command in an example is run.
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
import unicodedata
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Pages whose relative links (files and anchors) must resolve.
LINKED_PAGES: tuple[Path, ...] = (
    ROOT / "README.md",
    *sorted((ROOT / "docs").rglob("*.md")),
    *sorted((ROOT / "presentation").glob("*.md")),
    *sorted((ROOT / "plugins").rglob("*.md")),
)

#: Pages whose fenced shell examples must name real commands and options.
COMMAND_PAGES: tuple[Path, ...] = (
    ROOT / "README.md",
    *sorted((ROOT / "docs" / "guides").glob("*.md")),
    *sorted((ROOT / "docs" / "ter4").glob("*.md")),
    ROOT / "docs" / "hooks-guide.md",
    *sorted((ROOT / "presentation").glob("*.md")),
    *sorted((ROOT / "plugins").rglob("*.md")),
)

SHELL_LANGUAGES = {"bash", "sh", "shell", "console"}
FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([\w+-]*)")
LINK = re.compile(
    r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|!\[[^\]]*\]\(([^)\s]+)\)"
)
INLINE_CODE = re.compile(r"`+[^`]*`+")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
HTML_ID = re.compile(r"""\bid=["']([^"']+)["']""")


# --------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------


def _lines_outside_fences(text: str) -> Iterator[tuple[int, str]]:
    fence: str | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        match = FENCE.match(line)
        if match:
            marker = match.group(2)
            if fence is None:
                fence = marker
                continue
            if marker.startswith(fence[0]) and len(marker) >= len(fence):
                fence = None
                continue
        if fence is None:
            yield number, line


def fenced_blocks(text: str) -> Iterator[tuple[int, str, list[str]]]:
    """Yield ``(first line number, language, lines)`` for every fenced block."""
    fence: str | None = None
    language = ""
    start = 0
    body: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = FENCE.match(line)
        if fence is None:
            if match:
                fence, language, start, body = (
                    match.group(2),
                    match.group(3),
                    number,
                    [],
                )
            continue
        if match and match.group(2).startswith(fence[0]) and not match.group(3):
            yield start, language.lower(), body
            fence = None
            continue
        body.append(line)


def slugify(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to -."""
    text = INLINE_CODE.sub(lambda m: m.group(0).strip("`"), heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links keep their text
    text = re.sub(r"<[^>]+>", "", text)
    text = text.strip().lower()
    kept = []
    for char in text:
        category = unicodedata.category(char)
        if char in "-_ " or category[0] in "LN":
            kept.append(char)
    return "".join(kept).replace(" ", "-")


@cache
def anchors(path: Path) -> frozenset[str]:
    text = path.read_text(encoding="utf-8")
    seen: dict[str, int] = {}
    found: set[str] = set()
    for _, line in _lines_outside_fences(text):
        match = HEADING.match(line)
        if match:
            slug = slugify(match.group(2))
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            found.add(slug if count == 0 else f"{slug}-{count}")
        found.update(HTML_ID.findall(line))
    return frozenset(found)


def relative_links(path: Path) -> Iterator[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    for number, line in _lines_outside_fences(text):
        for match in LINK.finditer(INLINE_CODE.sub("", line)):
            target = match.group(1) or match.group(2)
            if re.match(r"^[a-z][a-z0-9+.-]*:", target):  # http:, https:, mailto:
                continue
            yield number, target


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name


# --------------------------------------------------------------------------
# Links
# --------------------------------------------------------------------------


def broken_links(page: Path, root: Path = ROOT) -> list[str]:
    problems = []
    for number, target in relative_links(page):
        file_part, _, anchor = target.partition("#")
        resolved = (page.parent / file_part).resolve() if file_part else page
        where = f"{page.relative_to(root)}:{number}: {target}"
        if not resolved.exists():
            problems.append(f"{where} (no such file)")
        elif anchor and resolved.suffix == ".md" and anchor not in anchors(resolved):
            problems.append(f"{where} (no heading #{anchor})")
    return problems


@pytest.mark.req("TER-REQ-007")
@pytest.mark.parametrize("page", LINKED_PAGES, ids=_rel)
def test_relative_links_resolve(page: Path) -> None:
    assert broken_links(page) == []


@pytest.mark.req("TER-REQ-007")
def test_link_checker_reports_missing_files_and_anchors(tmp_path: Path) -> None:
    (tmp_path / "other.md").write_text("# Real heading\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text(
        "# Title: with `code`\n\n"
        "[ok](other.md#real-heading) [self](#title-with-code) "
        "[web](https://example.com/x) `[not a link](nowhere.md)`\n"
        "[gone](missing.md) [bad anchor](other.md#nope)\n\n"
        "```text\n[inside a fence](ignored.md)\n```\n",
        encoding="utf-8",
    )
    problems = [p.split(": ", 1)[1] for p in broken_links(page, tmp_path)]
    assert problems == [
        "missing.md (no such file)",
        "other.md#nope (no heading #nope)",
    ]


def test_slugs_follow_github_rules() -> None:
    assert slugify("L0 gate, as built") == "l0-gate-as-built"
    assert slugify("The `ter-req` command") == "the-ter-req-command"
    assert slugify("1. Grammar lint (`ter-req lint`)") == "1-grammar-lint-ter-req-lint"
    assert slugify("Planned → verified") == "planned--verified"


# --------------------------------------------------------------------------
# Command examples
# --------------------------------------------------------------------------


def _help(main: Callable[[list[str]], object], argv: list[str]) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            main([*argv, "--help"])
        except SystemExit as exit_:
            code = int(exit_.code or 0)
    return code, out.getvalue() + err.getvalue()


def _ter(argv: list[str]) -> object:
    from ter_calculator.cli import main

    return main(argv)


def _ter4(argv: list[str]) -> object:
    from ter.bootstrap import main

    return main(argv)


def _ter_req(argv: list[str]) -> object:
    from ter.adapters.driving.req_cli import main

    return main(argv)


PROGRAMS: dict[str, Callable[[list[str]], object]] = {
    "ter": _ter,
    "python -m ter_calculator": _ter,
    "python -m ter": _ter4,
    "ter-req": _ter_req,
    "python -m ter.adapters.driving.req_cli": _ter_req,
}
SUBCOMMANDS = re.compile(r"^\s{2}\{([\w,-]+)\}", re.MULTILINE)
OPERATORS = {"|", "||", "&&", ";", "&"}
REDIRECTS = {"<", ">", ">>", "2>", "2>&1"}


@cache
def help_text(program: str, path: tuple[str, ...]) -> tuple[int, str]:
    return _help(PROGRAMS[program], list(path))


@dataclass(frozen=True)
class Example:
    page: Path
    line: int
    words: tuple[str, ...]

    def __str__(self) -> str:
        return f"{_rel(self.page)}:{self.line}: {' '.join(self.words)}"


def _commands(source: str) -> Iterator[list[str]]:
    lexer = shlex.shlex(source, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    words: list[str] = []
    skip_next = False
    for token in lexer:
        if skip_next:
            skip_next = False
            continue
        if token in OPERATORS:
            if words:
                yield words
            words = []
        elif token in REDIRECTS:
            skip_next = True
        else:
            words.append(token)
    if words:
        yield words


def shell_examples(page: Path) -> Iterator[Example]:
    text = page.read_text(encoding="utf-8")
    for start, language, lines in fenced_blocks(text):
        if language not in SHELL_LANGUAGES:
            continue
        joined: list[tuple[int, str]] = []
        pending = ""
        pending_line = 0
        for offset, raw in enumerate(lines, start=1):
            line = raw.strip()
            if line.startswith("$ "):
                line = line[2:]
            if not pending:
                pending_line = start + offset
            if line.endswith("\\"):
                pending += line[:-1] + " "
                continue
            joined.append((pending_line, pending + line))
            pending = ""
        for number, command in joined:
            try:
                commands = list(_commands(command))
            except ValueError:  # unbalanced quotes: not a shell command
                continue
            for words in commands:
                while words and re.match(r"^[A-Z_][A-Z0-9_]*=", words[0]):
                    words = words[1:]
                if words:
                    yield Example(page, number, tuple(words))


def _program(words: tuple[str, ...]) -> tuple[str, list[str]] | None:
    """The documented program an example runs, and the words after it."""
    for name in sorted(PROGRAMS, key=len, reverse=True):
        parts = tuple(name.split())
        if words[: len(parts)] == parts:
            return name, list(words[len(parts) :])
    return None


def check_example(example: Example) -> list[str]:
    found = _program(example.words)
    if found is None:
        return []
    program, rest = found
    path: list[str] = []
    code, text = help_text(program, ())
    problems: list[str] = []
    for word in rest:
        if word.startswith("-"):
            continue
        choices = SUBCOMMANDS.search(text)
        if not choices:
            break
        options = choices.group(1).split(",")
        if word not in options:
            problems.append(
                f"{example}: `{word}` is not a subcommand of {program} {' '.join(path)}".rstrip()
            )
            return problems
        path.append(word)
        code, text = help_text(program, tuple(path))
        if code != 0:
            problems.append(
                f"{example}: `{program} {' '.join(path)} --help` exited {code}"
            )
            return problems
    for word in rest:
        flag = word.split("=", 1)[0]
        if re.match(r"^--?[A-Za-z]", flag) and not re.search(
            rf"(?<![\w-]){re.escape(flag)}(?![\w-])", text
        ):
            problems.append(f"{example}: option `{flag}` is not defined")
    return problems


EXAMPLES: tuple[Example, ...] = tuple(
    example for page in COMMAND_PAGES for example in shell_examples(page)
)


@pytest.mark.req("TER-REQ-008")
@pytest.mark.parametrize("page", COMMAND_PAGES, ids=_rel)
def test_shell_examples_name_real_commands_and_options(page: Path) -> None:
    problems = [p for e in EXAMPLES if e.page == page for p in check_example(e)]
    assert problems == []


@pytest.mark.req("TER-REQ-008")
def test_every_subcommand_has_an_example_in_the_docs() -> None:
    shown = {
        (found[0], found[1][0])
        for e in EXAMPLES
        if (found := _program(e.words)) and found[1] and not found[1][0].startswith("-")
    }
    for program in ("ter", "ter-req", "python -m ter"):
        _, text = help_text(program, ())
        match = SUBCOMMANDS.search(text)
        assert match
        missing = {c for c in match.group(1).split(",") if (program, c) not in shown}
        assert not missing, f"{program} subcommands with no example: {sorted(missing)}"


@pytest.mark.req("TER-REQ-008")
def test_command_checker_catches_unknown_subcommands_and_options(
    tmp_path: Path,
) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        "```bash\n"
        "ter analyze session.jsonl --format json | jq .\n"
        "TER_UPDATE_GOLDEN=1 pytest tests/golden && ter-req trace --results r.json \\\n"
        "  --gate L2\n"
        'echo \'{"a": "x|y"}\' | python -m ter hook --event-log /tmp/ev\n'
        "ter anaylze session.jsonl\n"
        "ter report session.jsonl --pdf out.pdf\n"
        "ter-req points --verify\n"
        "```\n"
        "```text\nter not-a-command\n```\n",
        encoding="utf-8",
    )
    problems = [
        p.split(": ", 2)[2] for e in shell_examples(page) for p in check_example(e)
    ]
    assert problems == [
        "`anaylze` is not a subcommand of ter",
        "option `--pdf` is not defined",
        "option `--verify` is not defined",
    ]


# --------------------------------------------------------------------------
# Points index
# --------------------------------------------------------------------------


@pytest.mark.req("TER-REQ-005")
def test_committed_points_index_matches_the_catalogue() -> None:
    from ter.adapters.driven.requirements_yaml import load_catalogue
    from ter.adapters.driving.req_report import render_points_index

    catalogue = load_catalogue(ROOT / "requirements")
    expected = render_points_index(catalogue.points, catalogue.requirements)
    committed = (ROOT / "docs" / "ter4" / "points.md").read_text(encoding="utf-8")
    assert committed == expected, "docs/ter4/points.md is stale: run `ter-req points`"
