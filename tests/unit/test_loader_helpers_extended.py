from datetime import datetime
from pathlib import Path
import json
import pytest

import ter_calculator.loader as loader
from ter_calculator.models import ContentBlock, Message, SpanPhase


def test_parse_content_blocks_all_shapes():
    assert loader._parse_content_blocks(None) == []
    blocks = loader._parse_content_blocks("hello")
    assert blocks[0].block_type == "text" and blocks[0].text == "hello"
    blocks = loader._parse_content_blocks(
        [
            "x",
            {"type": "thinking", "thinking": "why"},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "a"}},
            {"type": "tool_result", "content": [{"text": "ok"}]},
        ]
    )
    assert [b.block_type for b in blocks] == [
        "text",
        "thinking",
        "tool_use",
        "tool_result",
    ]
    assert blocks[1].text == "why" and blocks[2].tool_name == "Read"


def test_parse_usage_timestamp_and_phase_helpers():
    assert loader._parse_usage(None) is None
    u = loader._parse_usage(
        {
            "input_tokens": 1,
            "output_tokens": 2,
            "cache_creation_input_tokens": 3,
            "cache_read_input_tokens": 4,
        }
    )
    assert (
        u.input_tokens
        + u.output_tokens
        + u.cache_creation_input_tokens
        + u.cache_read_input_tokens
    ) == 10
    assert loader._parse_timestamp(None) is None
    assert loader._parse_timestamp("bad") is None
    assert loader._parse_timestamp("2024-01-02T03:04:05Z").year == 2024
    assert loader._block_type_to_phase("thinking") == SpanPhase.REASONING
    assert loader._block_type_to_phase("tool_use") == SpanPhase.TOOL_USE
    assert loader._block_type_to_phase("tool_result") == SpanPhase.TOOL_USE
    assert loader._block_type_to_phase("anything") == SpanPhase.GENERATION


def test_block_text_variants():
    assert loader._get_block_text(ContentBlock(block_type="text", text="abc")) == "abc"
    assert (
        loader._get_block_text(ContentBlock(block_type="thinking", text="hmm")) == "hmm"
    )
    tool = ContentBlock(block_type="tool_use", tool_name="Read", tool_input={"x": 1})
    assert "Read" in loader._get_block_text(tool)
    result = ContentBlock(block_type="tool_result", text="done")
    assert loader._get_block_text(result) == "done"
    assert loader._get_block_text(ContentBlock(block_type="unknown")) == ""


def test_deduplicate_entries_and_prompt_extraction():
    entries = [
        {"uuid": "a", "x": 1},
        {"uuid": "a", "x": 2},
        {"uuid": "b"},
        {"x": 3},
        {"x": 3},
    ]
    dedup = loader._deduplicate_entries(entries)
    assert len(dedup) == 5
    msgs = [
        Message(
            uuid="1",
            role="user",
            content_blocks=[ContentBlock(block_type="text", text="hello")],
        ),
        Message(
            uuid="2",
            role="assistant",
            content_blocks=[ContentBlock(block_type="text", text="answer")],
        ),
        Message(
            uuid="3",
            role="user",
            content_blocks=[ContentBlock(block_type="tool_result", text="ignored")],
        ),
    ]
    assert loader._extract_user_prompts(msgs) == ["hello"]


def test_find_latest_session_explicit_dir_and_failures(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text("{}\n")
    b.write_text("{}\n")
    a.touch()
    b.touch()
    assert loader.find_latest_session(tmp_path).suffix == ".jsonl"
    with pytest.raises(FileNotFoundError):
        loader.find_latest_session(tmp_path / "missing")


def test_discover_subagents_layouts(tmp_path):
    parent = tmp_path / "parent.jsonl"
    parent.write_text("")
    direct = tmp_path / "parent" / "subagents"
    direct.mkdir(parents=True)
    (direct / "one.jsonl").write_text("")
    found = loader.discover_subagents(parent)
    assert any(p.name == "one.jsonl" for p in found)
