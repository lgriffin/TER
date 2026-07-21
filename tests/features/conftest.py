"""Shared fixtures for BDD feature tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ter_calculator.real_time import RollingTERState


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_sessions_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "sample_sessions"


@pytest.fixture
def empty_rolling_state() -> RollingTERState:
    return RollingTERState()


@pytest.fixture
def sample_jsonl_file(tmp_path: Path) -> Path:
    """Create a minimal valid JSONL session file."""
    lines = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "test-session",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Add a login page with email and password"}
                ],
            },
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": "test-session",
            "message": {
                "role": "assistant",
                "requestId": "req-1",
                "usage": {"input_tokens": 100, "output_tokens": 200},
                "content": [
                    {
                        "type": "thinking",
                        "text": "I need to create a login page component with email and password fields.",
                    },
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {
                            "file_path": "/app/login.py",
                            "content": "def login(): pass",
                        },
                    },
                ],
                "stop_reason": "tool_use",
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "sessionId": "test-session",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": "File written successfully",
                    },
                ],
            },
        },
        {
            "type": "assistant",
            "uuid": "a2",
            "sessionId": "test-session",
            "message": {
                "role": "assistant",
                "requestId": "req-2",
                "usage": {"input_tokens": 300, "output_tokens": 150},
                "content": [
                    {
                        "type": "text",
                        "text": "I have created the login page with email and password fields.",
                    },
                ],
                "stop_reason": "end_turn",
            },
        },
    ]
    path = tmp_path / "test_session.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


def build_jsonl_lines(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Helper to build JSONL line dicts from simplified message specs."""
    lines = []
    for i, msg in enumerate(messages):
        line: dict[str, Any] = {
            "type": msg.get("role", "assistant"),
            "uuid": f"msg-{i}",
            "sessionId": msg.get("session_id", "test-session"),
            "message": msg,
        }
        lines.append(line)
    return lines


def write_jsonl(path: Path, lines: list[dict[str, Any]]) -> None:
    """Write a list of dicts as JSONL lines."""
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
