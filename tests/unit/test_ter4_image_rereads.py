"""Image re-reads compare the image's bytes (TER-SRC-025, TER-DET-016).

A Claude Code image read returns base64 data and no text, so before this
every re-read of a screenshot looked identical. Reviewed on the owner's
sessions (25 Nov 2026): 65 image re-reads, 17 of them of an image a render
or script had redrawn in between. Synthetic records only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from ter4_lean_builder import Script

from ter.adapters.driven.claude_code import ClaudeCodeJsonlSource
from ter.adapters.driven.claude_code.session_source import image_digest
from ter.adapters.driven.tokenizers import RegexTokenizer
from ter.domain import ToolKind
from ter.domain.events import IMAGE_RESULT, EventKind, image_result_text
from ter.domain.lean import Finding, explain

RED = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
BLUE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPj/HwADBwIAMCbHYQAAAABJRU5ErkJggg=="


def _write(path: Path, records: list[Any]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def _read(uuid: str, call_id: str, file_path: str) -> dict[str, Any]:
    return {
        "type": "assistant",
        "uuid": uuid,
        "sessionId": "s1",
        "timestamp": "2026-11-25T08:00:00Z",
        "requestId": f"r-{uuid}",
        "message": {
            "id": f"m-{uuid}",
            "role": "assistant",
            "model": "claude-test",
            "content": [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": "Read",
                    "input": {"file_path": file_path},
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    }


def _result(uuid: str, call_id: str, content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "user",
        "uuid": uuid,
        "sessionId": "s1",
        "timestamp": "2026-11-25T08:00:01Z",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": content}
            ],
        },
    }


def _image(data: str) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


def _completions(tmp_path: Path, *results: list[dict[str, Any]]) -> list[str]:
    records: list[Any] = []
    for i, content in enumerate(results):
        records.append(_read(f"a{i}", f"t{i}", "/w/shot.png"))
        records.append(_result(f"u{i}", f"t{i}", content))
    trace = ClaudeCodeJsonlSource().read(_write(tmp_path / "s.jsonl", records))
    return [e.text for e in trace.events if e.kind is EventKind.TOOL_COMPLETED]


# --- TER-SRC-025: an image result names its bytes ------------------------------


@pytest.mark.req("TER-SRC-025")
def test_an_image_result_names_the_digest_of_its_bytes(tmp_path: Path) -> None:
    [text] = _completions(tmp_path, [_image(RED)])
    digest = hashlib.sha256(RED.encode("utf-8")).hexdigest()[:12]
    assert text == f"[image sha256:{digest}]"
    assert IMAGE_RESULT.match(text)


@pytest.mark.req("TER-SRC-025")
def test_the_same_image_gives_the_same_text_and_a_redrawn_one_does_not(
    tmp_path: Path,
) -> None:
    first, again, redrawn = _completions(
        tmp_path, [_image(RED)], [_image(RED)], [_image(BLUE)]
    )
    assert first == again != redrawn


@pytest.mark.req("TER-SRC-025")
def test_a_redacted_image_keeps_the_digest_of_the_original(tmp_path: Path) -> None:
    redacted = f"[image dropped: sha256:{image_digest(RED)}]"
    raw, kept = _completions(tmp_path, [_image(RED)], [_image(redacted)])
    assert raw == kept


@pytest.mark.req("TER-SRC-025")
def test_several_images_are_named_in_order(tmp_path: Path) -> None:
    [text] = _completions(tmp_path, [_image(RED), _image(BLUE)])
    assert text == image_result_text([image_digest(RED), image_digest(BLUE)])
    assert IMAGE_RESULT.match(text)


@pytest.mark.req("TER-SRC-025")
def test_a_result_with_text_keeps_its_text(tmp_path: Path) -> None:
    [text] = _completions(tmp_path, [{"type": "text", "text": "page 1"}, _image(RED)])
    assert text == "page 1"


@pytest.mark.req("TER-SRC-025")
@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "image"}],
        [{"type": "image", "source": "not a mapping"}],
        ["not a block"],
    ],
)
def test_a_result_without_image_data_stays_empty(
    tmp_path: Path, content: list[Any]
) -> None:
    assert _completions(tmp_path, content) == [""]


@pytest.mark.req("TER-SRC-025")
@pytest.mark.parametrize(
    ("text", "matches"),
    [
        ("[image sha256:abc123]", True),
        ("[image sha256:abc123,def456]", True),
        ("[image sha256:]", False),
        ("[image sha256:ABC]", False),
        ("see [image sha256:abc123]", False),
        ("[image sha256:abc123] and more", False),
    ],
)
def test_the_image_result_pattern(text: str, matches: bool) -> None:
    assert bool(IMAGE_RESULT.match(text)) is matches


# --- TER-DET-016: a redrawn image is not a re-read -----------------------------


def _rereads(first: str, second: str, path: str = "/w/shot.png") -> list[Finding]:
    s = Script()
    s.prompt("Make the chart's bars blue")
    s.call("Read", ToolKind.FS_READ, {"file_path": path}, first)
    s.bash("python render.py", "ok")
    s.call("Read", ToolKind.FS_READ, {"file_path": path}, second)
    s.say("The bars are blue.")
    analysis = explain(s.events, RegexTokenizer())
    return [f for f in analysis.findings if f.detector == "repeated_exploration"]


@pytest.mark.req("TER-DET-016", "TER-DET-002")
def test_a_redrawn_image_is_not_a_re_read() -> None:
    red = image_result_text([image_digest(RED)])
    blue = image_result_text([image_digest(BLUE)])
    assert _rereads(red, blue) == []


@pytest.mark.req("TER-DET-016", "TER-DET-002")
@pytest.mark.parametrize("path", ["/w/shot.png", "/w/SHOT.JPG", "C:\\w\\a.webp"])
def test_the_same_image_read_again_stays_uncertain(path: str) -> None:
    red = image_result_text([image_digest(RED)])
    [f] = _rereads(red, red, path)
    assert f.uncertain and f.confidence == 0.55
    assert "same image" in f.explanation


@pytest.mark.req("TER-DET-016", "TER-DET-002")
def test_an_image_without_a_digest_is_still_uncertain() -> None:
    [f] = _rereads("", "")
    assert f.confidence == 0.55 and "no text" in f.explanation


@pytest.mark.req("TER-DET-016", "TER-DET-002")
@pytest.mark.parametrize("path", ["/w/notes.md", "/w/report.pdf"])
def test_identical_text_is_still_a_confident_re_read(path: str) -> None:
    [f] = _rereads("the same words", "the same words", path)
    assert f.confidence == 0.85
