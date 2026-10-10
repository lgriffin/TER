"""Facts read from single events: what a shell command is for, whether a
validation run passed, which files a call touches, which words a text uses.

Every function here is pure, deterministic and conservative: when a fact
cannot be read reliably it is reported as unknown (``Outcome.UNKNOWN``,
``ShellIntent.OTHER``) rather than guessed.
"""

from __future__ import annotations

import hashlib
import re
import shlex
from collections.abc import Mapping

from .model import Outcome, ShellIntent

__all__ = [
    "STOPWORDS",
    "content_words",
    "defined_identifiers",
    "failure_signature",
    "fingerprint",
    "output_fingerprint",
    "normalise_command",
    "overlap",
    "question_words",
    "runs_check",
    "shell_intent",
    "source_lines",
    "tool_paths",
    "validation_outcome",
]

_PATH_KEYS = ("file_path", "notebook_path", "path")

# Each pattern is matched at the start of one simple command (see
# ``_command_heads``), never anywhere in the line: ``echo pytest`` prints a
# word and validates nothing. Checked in this order: a change in any command
# makes the line a change; otherwise a validation in any command makes it a
# validation.
_CHANGE = re.compile(
    r"(?:"
    r"(?:pip3?|uv pip|poetry|npm|yarn|pnpm|cargo|go|gem|bundle|apt(?:-get)?|brew)"
    r"\s+(?:install|add|remove|uninstall|get|update|upgrade)"
    r"|git\s+(?:add|commit|checkout|switch|reset|restore|apply|merge|rebase|mv|rm|push|stash|cherry-pick)"
    r"|mkdir|rm|mv|cp|touch|chmod|chown|ln|patch|sed\s+-i|tee"
    r")(?:\s|$)"
)
_VALIDATE = re.compile(
    r"(?:"
    r"pytest|py\.test|tox|nox|unittest|doctest|jest|vitest|mocha|ava|karma|rspec"
    r"|phpunit|ctest|ruff|mypy|pyright|flake8|pylint|bandit|eslint|tsc|prettier\s+--check"
    r"|black\s+--check|isort\s+--check|shellcheck|hadolint|golangci-lint|lint-imports"
    r"|go\s+(?:test|vet|build)|cargo\s+(?:test|check|clippy|build)|mvn|gradlew?\s+\S*(?:test|check|build)"
    r"|dotnet\s+(?:test|build)|make\s+(?:test|check|lint|ci)|(?:npm|yarn|pnpm)\s+(?:run\s+)?(?:test|lint|check|build|typecheck)"
    r"|playwright\s+test|cypress\s+run|bazel\s+test|swift\s+test"
    r"|pre-commit\s+run|gh\s+(?:pr\s+checks|run\s+(?:watch|view))"
    r")(?:\s|$)"
)
# Running code to check it by hand: ``python -c``, ``python - <<EOF``,
# ``python -m pkg``, ``node script.js``. Counted as validation only when the
# command is not also a change.
_ADHOC_RUN = re.compile(
    r"(?:python3?|node|deno|bun|ruby|php)\s+(?:-c(?:\s|$)|-(?:\s|$)|-m\s+\w|[\w./-]+\.(?:py|js|mjs|ts|rb|php)(?:\s|$))"
)
_EXPLORE = re.compile(
    r"(?:ls|cat|head|tail|less|more|find|grep|rg|ag|tree|pwd|wc|file|stat|du"
    r"|which|type|echo|env|printenv|git\s+(?:status|log|diff|show|blame|branch|remote|ls-files|grep))(?:\s|$)"
)
# Words that run the command after them: ``sudo pytest``, ``uv run pytest``.
_PREFIXES = frozenset(
    {
        "sudo",
        "env",
        "time",
        "nohup",
        "exec",
        "command",
        "nice",
        "xargs",
        "npx",
        "bunx",
        "pnpx",
    }
)
_RUNNERS = frozenset(
    {
        "uv run",
        "poetry run",
        "pipenv run",
        "pdm run",
        "hatch run",
        "rye run",
        "pnpm exec",
        "yarn exec",
        "bundle exec",
    }
)
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_SHELL_PUNCTUATION = "();<>|&\n"

# Output markers. Failure markers are specific on purpose (precision over
# recall): a bare "error" in prose output is not a failure.
_FAILED = re.compile(
    r"(?:\b\d+\s+(?:failed|failures?|errors?)\b(?<!\b0 failed)(?<!\b0 errors)(?<!\b0 error)"
    r"|^FAILED\b|^FAIL\b|\bFAILED\s+\S+::|^ERROR[:\s]|^E\s{2,}\S"
    r"|Traceback \(most recent call last\)|^\w*Error: |^error(?:\[\w+\])?: "
    r"|\bexit (?:code|status) [1-9]\d*|\bExit code [1-9]\d*|command not found"
    r"|npm ERR!|^\s*[✗✘×] |\bFound \d+ errors?\b(?<!Found 0 errors)|\bTests?:\s+\d+ failed)",
    re.MULTILINE,
)
_PASSED = re.compile(
    r"(?:\b\d+\s+passed\b|^OK\b|\bAll checks passed\b|\bSuccess(?:fully)?\b|\bchecks? passed\b"
    r"|\bno issues found\b|\b0 errors?\b|\bTests?:\s+\d+ passed\b|^ok\s|\bBUILD SUCCESSFUL\b"
    r"|\bFinished\b.*\btarget\(s\)|^PASS\b)",
    re.MULTILINE | re.IGNORECASE,
)
_NOISE = re.compile(
    r"(?:\bin\s+\d+(?:\.\d+)?\s*m?s\b|\d+(?:\.\d+)?s\b|0x[0-9a-f]+|\b\d{4}-\d\d-\d\d[T ][\d:.]+Z?)",
    re.IGNORECASE,
)
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:[.-][A-Za-z0-9_]+)*")
_DEFINED = re.compile(
    r"(?:\b(?:def|class|function|func|fn|const|let|var|struct|interface|type|enum|trait|impl)\s+"
    r"([A-Za-z_][A-Za-z0-9_]{2,})|^([A-Z][A-Z0-9_]{2,})\s*[:=])",
    re.MULTILINE,
)

_STOPWORDS = frozenset(
    """
    a an the and or but if then else so to of in on at by for with from into onto
    is are was were be been being it its this that these those there here i we you
    he she they them our my your me us do does did done can could should would will
    shall may might must need needs let lets now just also still again only very
    what which who whom whose when where why how all any each some more most other
    such no not nor too than as up out about over under before after while
    make sure first next last think file files use using used
    """.split()
)

#: Words too common to say what a text is about (shared with :mod:`.intent`).
STOPWORDS = _STOPWORDS


def normalise_command(command: str) -> str:
    """Collapse whitespace, so equal commands compare equal."""
    return " ".join(command.split())


def _strip_heredocs(command: str) -> str:
    """Drop heredoc bodies: their lines are data for a command, not commands."""
    out: list[str] = []
    lines = command.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for match in _HEREDOC.finditer(line):
            delimiter = match.group(2)
            while i < len(lines) and lines[i].strip() != delimiter:
                i += 1
            i += 1  # the closing delimiter
    return "\n".join(out)


def _tokens(command: str) -> list[str]:
    """Shell words and operators; quoted text stays one word."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=_SHELL_PUNCTUATION)
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:  # unbalanced quotes: fall back to bare splitting
        return re.findall(r"[();<>|&\n]+|[^\s();<>|&]+", command)


def _command_heads(command: str) -> list[str]:
    """Each simple command in a shell line, from its command word on.

    Segments split on ``;``, ``&&``, ``||``, ``|``, ``&``, newlines and
    parentheses. Redirections and their targets are dropped; leading
    ``VAR=value`` assignments, wrappers (``sudo``, ``uv run``, ...) and a
    program's directory (``.venv/bin/pytest``) are peeled off, so the pattern
    sees the program that actually runs.
    """
    segments: list[list[str]] = [[]]
    skip_target = False
    for token in _tokens(_strip_heredocs(command)):
        if token and all(c in _SHELL_PUNCTUATION for c in token):
            if "<" in token or ">" in token:  # a redirection names a target
                skip_target = True
                continue
            segments.append([])
            skip_target = False
            continue
        if skip_target:
            skip_target = False
            continue
        segments[-1].append(token)
    heads: list[str] = []
    for words in segments:
        while words and _ASSIGNMENT.match(words[0]):
            words = words[1:]
        while words:
            first = words[0].rsplit("/", 1)[-1]
            if first in _PREFIXES:
                words = words[1:]
                while words and (
                    words[0].startswith("-") or _ASSIGNMENT.match(words[0])
                ):
                    words = words[1:]
            elif first == "timeout":  # ``timeout [opts] DURATION cmd``
                words = words[1:]
                while words and words[0].startswith("-"):
                    words = words[1:]
                words = words[1:]
            elif len(words) > 1 and f"{first} {words[1]}" in _RUNNERS:
                words = words[2:]
            else:
                break
        if words:
            words = [words[0].rsplit("/", 1)[-1] or words[0], *words[1:]]
            heads.append(" ".join(words))
    return heads


def shell_intent(command: str) -> ShellIntent:
    """Classify a shell command line by what it is for.

    Only command words count: a test runner's name printed by ``echo`` or
    searched for by ``grep`` is an argument, not a validation run.
    """
    heads = [h for h in _command_heads(command) if h.split(" ", 1)[0] != "cd"]
    if not heads:
        return ShellIntent.OTHER
    # ``python -m pip install x`` is a change, ``python -m pytest`` a check.
    unwrapped = [
        h.split(" ", 2)[2] if re.match(r"python3?\s+-m\s+\S", h) else h for h in heads
    ]
    if any(_CHANGE.match(h) for h in unwrapped):
        return ShellIntent.CHANGE
    if any(_VALIDATE.match(h) or _ADHOC_RUN.match(h) for h in [*heads, *unwrapped]):
        return ShellIntent.VALIDATE
    if _EXPLORE.match(heads[0]):
        return ShellIntent.EXPLORE
    return ShellIntent.OTHER


def runs_check(command: str) -> bool:
    """Whether any simple command in the line runs a named check tool.

    Unlike :func:`shell_intent`, a change elsewhere in the line does not hide
    the check: ``sed -i ... && pytest`` edits and then validates. Real
    sessions chain checks this way often (15 of 260 check-running lines in
    this project's own transcripts were classified as changes). Ad hoc runs
    (``python - <<EOF``) do not count here: beside a change they are more
    often an editing script than a check.
    """
    heads = _command_heads(command)
    unwrapped = [
        h.split(" ", 2)[2] if re.match(r"python3?\s+-m\s+\S", h) else h for h in heads
    ]
    return any(_VALIDATE.match(h) for h in [*heads, *unwrapped])


def validation_outcome(output: str) -> Outcome:
    """Read pass or fail from a validation run's output; unknown when unclear."""
    if _FAILED.search(output):
        return Outcome.FAILED
    if _PASSED.search(output):
        return Outcome.PASSED
    return Outcome.UNKNOWN


def failure_signature(output: str) -> str:
    """A hash of a failing run's failure lines, with timings and addresses removed.

    Two runs with equal signatures failed the same way.
    """
    lines = sorted(
        {
            _NOISE.sub("#", line.strip())
            for line in output.splitlines()
            if _FAILED.search(line.strip())
        }
    )
    basis = "\n".join(lines) if lines else _NOISE.sub("#", output.strip())
    return fingerprint(basis)


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def output_fingerprint(text: str) -> str:
    """Fingerprint of tool output with timings, addresses and timestamps removed,
    so two results that differ only in how long they took compare equal."""
    return fingerprint(_NOISE.sub("#", text.strip()))


def tool_call_failed(output: str) -> bool:
    """Whether a tool completion reports that the call itself failed: the
    harness's error envelope (``<tool_use_error>`` in Claude Code), as when
    an edit's text is not found or a write is refused."""
    return output.lstrip().startswith("<tool_use_error>")


def write_created(output: str) -> bool | None:
    """Whether a write's result says it created the file: True for a new
    file (Claude Code: ``File created successfully at: …``), False for an
    existing one it replaced (``The file … has been updated``), None when the
    result does not say."""
    text = output.lstrip()
    if text.startswith("File created successfully"):
        return True
    if text.startswith("The file ") and "has been updated" in text[:2000]:
        return False
    return None


def tool_paths(arguments: Mapping[str, object]) -> tuple[str, ...]:
    """File paths a file tool call names (empty for searches and shell)."""
    for key in _PATH_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return (value,)
    return ()


def content_words(text: str) -> frozenset[str]:
    """Lower-cased content words of a text, without stopwords or short words."""
    return frozenset(
        w
        for w in (m.group(0).lower() for m in _WORD.finditer(text))
        if len(w) > 2 and w not in _STOPWORDS
    )


def defined_identifiers(text: str) -> frozenset[str]:
    """Names a source text defines (functions, classes, constants)."""
    return frozenset(a or b for a, b in _DEFINED.findall(text))


_LINE_NUMBER = re.compile(r"^\s*\d+(?:→|\t)")


def source_lines(text: str) -> frozenset[str]:
    """Distinct non-blank lines of a file's text, stripped, without the line
    numbers a read tool may prefix."""
    return frozenset(
        stripped
        for stripped in (
            _LINE_NUMBER.sub("", line).strip() for line in text.splitlines()
        )
        if stripped
    )


def overlap(a: frozenset[str], b: frozenset[str]) -> float:
    """Overlap coefficient ``|a ∩ b| / min(|a|, |b|)``; 0 when either is empty."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# A sentence ends at ``.``, ``!`` or ``?`` followed by space, or at a line end;
# ``cli.py`` and ``v1.2`` stay whole.
_SENTENCE = re.compile(r"[^.!?\n]*(?:[.!?](?=\s|$)|\n|$)")
_DOTTED = re.compile(r"(?<=\w)\.(?=\w)")


def question_words(text: str) -> frozenset[str]:
    """Content words of the questions a text asks: sentences ending in ``?``.

    These are the open questions a prompt, reasoning block or response
    records; exploration that names one of their words addresses them.
    """
    words: set[str] = set()
    for line in text.splitlines():
        guarded = _DOTTED.sub("\0", line)
        for match in _SENTENCE.finditer(guarded):
            sentence = match.group(0).strip()
            if sentence.endswith("?"):
                words |= content_words(sentence.replace("\0", "."))
    return frozenset(words)
