"""Tests for input analysis: token breakdown and prompt similarity."""

from __future__ import annotations

import json

import pytest

from ter_calculator.input_analysis import (
    analyze_input,
    compute_intent_drift,
    compute_prompt_response_alignment,
    compute_prompt_similarity,
    compute_token_breakdown,
)
from ter_calculator.loader import load_session
from ter_calculator.models import ContentBlock, Message, Session


def _make_session(messages: list[Message], prompts: list[str] | None = None) -> Session:
    """Helper to create a Session with pre-built messages."""
    if prompts is None:
        prompts = [
            block.text
            for msg in messages
            if msg.role == "user"
            for block in msg.content_blocks
            if block.block_type == "text" and block.text
        ]
    return Session(
        session_id="test",
        file_path="test.jsonl",
        messages=messages,
        user_prompts=prompts,
    )


class TestTokenBreakdown:
    def test_user_text_counted(self):
        session = _make_session([
            Message(
                uuid="u1", role="user",
                content_blocks=[ContentBlock(block_type="text", text="Hello world")],
            ),
            Message(
                uuid="a1", role="assistant",
                content_blocks=[ContentBlock(block_type="text", text="Hi there!")],
            ),
        ])
        bd = compute_token_breakdown(session)
        assert bd.user_input_tokens > 0
        assert bd.user_result_tokens == 0
        assert bd.model_generation_tokens > 0
        assert bd.total_user_tokens == bd.user_input_tokens
        assert bd.total_model_tokens == bd.model_generation_tokens

    def test_tool_result_is_user_centric(self):
        session = _make_session([
            Message(
                uuid="u1", role="user",
                content_blocks=[
                    ContentBlock(block_type="text", text="read the file"),
                    ContentBlock(block_type="tool_result", text="file contents here blah blah blah"),
                ],
            ),
        ])
        bd = compute_token_breakdown(session)
        assert bd.user_input_tokens > 0
        assert bd.user_result_tokens > 0
        assert bd.total_user_tokens == bd.user_input_tokens + bd.user_result_tokens

    def test_assistant_phases(self):
        session = _make_session([
            Message(
                uuid="a1", role="assistant",
                content_blocks=[
                    ContentBlock(block_type="thinking", text="Let me think about this problem carefully"),
                    ContentBlock(block_type="tool_use", text="Read file.py"),
                    ContentBlock(block_type="text", text="Here is the answer to your question"),
                ],
            ),
        ], prompts=[])
        bd = compute_token_breakdown(session)
        assert bd.model_reasoning_tokens > 0
        assert bd.model_tool_tokens > 0
        assert bd.model_generation_tokens > 0
        assert bd.total_model_tokens == (
            bd.model_reasoning_tokens + bd.model_tool_tokens + bd.model_generation_tokens
        )

    def test_user_ratio(self):
        session = _make_session([
            Message(
                uuid="u1", role="user",
                content_blocks=[ContentBlock(block_type="text", text="short")],
            ),
            Message(
                uuid="a1", role="assistant",
                content_blocks=[ContentBlock(block_type="text", text="a much longer response with many words in it")],
            ),
        ])
        bd = compute_token_breakdown(session)
        assert 0.0 < bd.user_ratio < 1.0
        assert bd.user_ratio < 0.5  # user sent less

    def test_empty_session(self):
        session = _make_session([], prompts=[])
        bd = compute_token_breakdown(session)
        assert bd.total_user_tokens == 0
        assert bd.total_model_tokens == 0
        assert bd.user_ratio == 0.0


class TestPromptSimilarity:
    def test_empty_prompts(self):
        result = compute_prompt_similarity([])
        assert result.prompt_count == 0
        assert result.similarity_matrix == []
        assert result.similar_pairs == []
        assert result.prompt_redundancy_score == 0.0

    def test_single_prompt(self):
        result = compute_prompt_similarity(["fix the bug"])
        assert result.prompt_count == 1
        assert result.similarity_matrix == [[1.0]]
        assert result.similar_pairs == []
        assert result.prompt_redundancy_score == 0.0

    def test_identical_prompts_detected(self):
        result = compute_prompt_similarity([
            "fix the login bug",
            "fix the login bug",
        ])
        assert result.prompt_count == 2
        assert len(result.similar_pairs) == 1
        assert result.similar_pairs[0].similarity > 0.99
        assert result.prompt_redundancy_score == 1.0

    def test_similar_prompts_detected(self):
        result = compute_prompt_similarity([
            "fix the login authentication error",
            "repair the auth login failure",
            "write unit tests for the database",
        ], similarity_threshold=0.5)
        # The auth-related prompts should be similar; the test one should not
        auth_pairs = [
            p for p in result.similar_pairs
            if {p.prompt_a_index, p.prompt_b_index} == {0, 1}
        ]
        assert len(auth_pairs) == 1

    def test_dissimilar_prompts_no_pairs(self):
        result = compute_prompt_similarity([
            "fix the login bug in the authentication module",
            "add a new chart to the analytics dashboard",
            "update the database migration scripts for PostgreSQL",
        ], similarity_threshold=0.85)
        assert result.similar_pairs == []
        assert result.prompt_redundancy_score == 0.0

    def test_redundancy_score_fraction(self):
        result = compute_prompt_similarity([
            "fix the bug",
            "fix the bug",
            "fix the bug",
            "add a completely new unrelated feature to the system",
        ])
        # 3 of 4 prompts are redundant
        assert result.prompt_redundancy_score == 0.75

    def test_matrix_shape(self):
        prompts = ["a", "b", "c"]
        result = compute_prompt_similarity(prompts)
        assert len(result.similarity_matrix) == 3
        assert all(len(row) == 3 for row in result.similarity_matrix)
        # Diagonal is 1.0
        for i in range(3):
            assert result.similarity_matrix[i][i] == 1.0


class TestIntentDrift:
    def test_single_prompt_stable(self):
        result = compute_intent_drift(["fix the bug"])
        assert result.steps == []
        assert result.overall_trajectory == "stable"
        assert result.average_drift == 0.0

    def test_empty_prompts_stable(self):
        result = compute_intent_drift([])
        assert result.steps == []
        assert result.overall_trajectory == "stable"

    def test_identical_prompts_convergent(self):
        result = compute_intent_drift([
            "fix the login authentication bug",
            "fix the login authentication bug",
        ])
        assert len(result.steps) == 1
        assert result.steps[0].drift_type == "convergent"
        assert result.steps[0].similarity > 0.99
        assert result.overall_trajectory == "convergent"

    def test_divergent_prompts(self):
        result = compute_intent_drift([
            "fix the login authentication bug in the user module",
            "add a new chart to the analytics dashboard for sales data",
        ])
        assert len(result.steps) == 1
        # These are semantically very different
        assert result.steps[0].similarity < 0.6

    def test_mixed_trajectory(self):
        result = compute_intent_drift([
            "fix the login authentication error",
            "repair the auth login failure",
            "add a new analytics dashboard with charts",
            "create a visualization panel for metrics",
        ])
        assert len(result.steps) == 3
        # First step: similar (both about auth)
        # Middle step: divergent (auth → dashboard)
        # Last step: similar (both about visualization)
        assert result.steps[0].drift_type == "convergent"

    def test_step_indices(self):
        result = compute_intent_drift(["a", "b", "c"])
        assert result.steps[0].from_index == 0
        assert result.steps[0].to_index == 1
        assert result.steps[1].from_index == 1
        assert result.steps[1].to_index == 2


class TestPromptResponseAlignment:
    def test_aligned_pair(self):
        session = _make_session([
            Message(
                uuid="u1", role="user",
                content_blocks=[ContentBlock(block_type="text", text="fix the login authentication bug")],
            ),
            Message(
                uuid="a1", role="assistant",
                content_blocks=[
                    ContentBlock(block_type="text", text="I fixed the login authentication bug by updating the password validation"),
                ],
            ),
        ])
        result = compute_prompt_response_alignment(session)
        assert len(result.pairs) == 1
        assert result.pairs[0].alignment > 0.5
        assert result.average_alignment > 0.5
        assert result.low_alignment_count == 0

    def test_misaligned_pair(self):
        session = _make_session([
            Message(
                uuid="u1", role="user",
                content_blocks=[ContentBlock(block_type="text", text="fix the login authentication bug")],
            ),
            Message(
                uuid="a1", role="assistant",
                content_blocks=[
                    ContentBlock(block_type="text", text="The weather today is sunny with clear skies and mild temperatures"),
                ],
            ),
        ])
        result = compute_prompt_response_alignment(session)
        assert len(result.pairs) == 1
        assert result.pairs[0].alignment < 0.5

    def test_skips_tool_result_messages(self):
        session = _make_session([
            Message(
                uuid="u1", role="user",
                content_blocks=[ContentBlock(block_type="text", text="read the config file")],
            ),
            Message(
                uuid="a1", role="assistant",
                content_blocks=[ContentBlock(block_type="tool_use", text="Read config.yaml")],
            ),
            Message(
                uuid="u2", role="user",
                content_blocks=[ContentBlock(block_type="tool_result", text="key: value")],
            ),
            Message(
                uuid="a2", role="assistant",
                content_blocks=[ContentBlock(block_type="text", text="The config file contains the key-value settings")],
            ),
        ])
        result = compute_prompt_response_alignment(session)
        # Only one real user prompt, tool_result is intermediate
        assert len(result.pairs) == 1
        assert result.pairs[0].prompt_text == "read the config file"

    def test_multiple_turns(self):
        session = _make_session([
            Message(
                uuid="u1", role="user",
                content_blocks=[ContentBlock(block_type="text", text="fix the login bug")],
            ),
            Message(
                uuid="a1", role="assistant",
                content_blocks=[ContentBlock(block_type="text", text="I fixed the login bug")],
            ),
            Message(
                uuid="u2", role="user",
                content_blocks=[ContentBlock(block_type="text", text="now add unit tests for the database")],
            ),
            Message(
                uuid="a2", role="assistant",
                content_blocks=[ContentBlock(block_type="text", text="I added unit tests for the database module")],
            ),
        ])
        result = compute_prompt_response_alignment(session)
        assert len(result.pairs) == 2
        assert result.pairs[0].prompt_index == 0
        assert result.pairs[1].prompt_index == 1

    def test_empty_session(self):
        session = _make_session([], prompts=[])
        result = compute_prompt_response_alignment(session)
        assert result.pairs == []
        assert result.average_alignment == 0.0
        assert result.low_alignment_count == 0

    def test_no_response(self):
        """User prompt with no assistant response yields no pairs."""
        session = _make_session([
            Message(
                uuid="u1", role="user",
                content_blocks=[ContentBlock(block_type="text", text="fix the bug")],
            ),
        ])
        result = compute_prompt_response_alignment(session)
        assert result.pairs == []

    def test_response_from_multiple_assistant_messages(self):
        """Response text is collected across consecutive assistant messages."""
        session = _make_session([
            Message(
                uuid="u1", role="user",
                content_blocks=[ContentBlock(block_type="text", text="create a login page")],
            ),
            Message(
                uuid="a1", role="assistant",
                content_blocks=[ContentBlock(block_type="text", text="I'll create the login form")],
            ),
            Message(
                uuid="a2", role="assistant",
                content_blocks=[ContentBlock(block_type="text", text="Login page is ready with validation")],
            ),
        ])
        result = compute_prompt_response_alignment(session)
        assert len(result.pairs) == 1
        assert "login form" in result.pairs[0].response_text
        assert "validation" in result.pairs[0].response_text


class TestAnalyzeInput:
    def test_full_analysis(self):
        session = _make_session([
            Message(
                uuid="u1", role="user",
                content_blocks=[ContentBlock(block_type="text", text="fix the login bug")],
            ),
            Message(
                uuid="a1", role="assistant",
                content_blocks=[
                    ContentBlock(block_type="thinking", text="I need to find the login code"),
                    ContentBlock(block_type="text", text="I found and fixed the bug"),
                ],
            ),
            Message(
                uuid="u2", role="user",
                content_blocks=[ContentBlock(block_type="text", text="fix the login authentication error")],
            ),
        ])
        result = analyze_input(session, similarity_threshold=0.5)

        assert result.token_breakdown.total_user_tokens > 0
        assert result.token_breakdown.total_model_tokens > 0
        assert result.prompt_similarity.prompt_count == 2
        assert len(result.prompt_similarity.similar_pairs) >= 1
        # Drift should show convergent (similar asks)
        assert result.intent_drift.steps[0].drift_type == "convergent"
        # Alignment should have one pair (first prompt → assistant response)
        assert len(result.prompt_response_alignment.pairs) >= 1
