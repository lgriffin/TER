"""Tests for the shared analysis pipeline."""

import argparse
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from ter_calculator.analyze_pipeline import analyze_session, default_analyze_args
from ter_calculator.models import (
    ClassifiedSpan,
    IntentVector,
    SpanLabel,
    SpanPhase,
    TERResult,
    TokenSpan,
)


class TestDefaultAnalyzeArgs:
    def test_returns_namespace(self):
        args = default_analyze_args("/path/to/session.jsonl")
        assert isinstance(args, argparse.Namespace)

    def test_session_path_set(self):
        args = default_analyze_args("/tmp/test.jsonl")
        assert args.session_path == "/tmp/test.jsonl"

    def test_default_thresholds(self):
        args = default_analyze_args("x.jsonl")
        assert args.similarity_threshold == 0.40
        assert args.confidence_threshold == 0.75
        assert args.restatement_threshold == 0.85

    def test_default_weights(self):
        args = default_analyze_args("x.jsonl")
        assert args.phase_weights == "0.3,0.4,0.3"

    def test_waste_patterns_enabled(self):
        args = default_analyze_args("x.jsonl")
        assert args.no_waste_patterns is False

    def test_default_cost_model(self):
        args = default_analyze_args("x.jsonl")
        assert args.cost_model == "sonnet"

    def test_input_analysis_enabled(self):
        args = default_analyze_args("x.jsonl")
        assert args.no_input_analysis is False

    def test_fine_segmentation_disabled(self):
        args = default_analyze_args("x.jsonl")
        assert args.fine_segmentation is False


class TestAnalyzeSession:
    @pytest.fixture
    def mock_args(self):
        return argparse.Namespace(
            session_path="test.jsonl",
            similarity_threshold=0.40,
            confidence_threshold=0.75,
            restatement_threshold=0.85,
            phase_weights="0.3,0.4,0.3",
            no_waste_patterns=True,
            cost_model="sonnet",
            no_input_analysis=True,
            prompt_similarity_threshold=0.75,
            fine_segmentation=False,
            segment_min_tokens=12,
            segment_max_tokens=180,
        )

    def _make_mock_span(self, phase, label, tokens=100):
        return ClassifiedSpan(
            span=TokenSpan(
                text="test " * (tokens // 5),
                phase=phase,
                position=0,
                token_count=tokens,
                source_message_uuid="msg-1",
            ),
            label=label,
            confidence=0.9,
            cosine_similarity=0.8,
        )

    @patch("ter_calculator.economics.compute_economics")
    @patch("ter_calculator.compute.compute_ter")
    @patch("ter_calculator.classifier.classify_spans")
    @patch("ter_calculator.intent_extraction.SlidingIntentExtractor")
    @patch("ter_calculator.loader.segment_spans")
    @patch("ter_calculator.loader.load_session")
    def test_returns_ter_result(
        self,
        mock_load,
        mock_segment,
        mock_extractor_cls,
        mock_classify,
        mock_compute,
        mock_economics,
        mock_args,
    ):
        mock_session = MagicMock()
        mock_session.session_id = "s1"
        mock_session.user_prompts = ["fix the bug"]
        mock_load.return_value = mock_session
        mock_segment.return_value = []

        intent = IntentVector(
            text="fix the bug",
            embedding=np.zeros(384),
            confidence=0.9,
        )
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [intent]
        mock_extractor_cls.return_value = mock_extractor

        classified = [
            self._make_mock_span(SpanPhase.REASONING, SpanLabel.ALIGNED_REASONING),
        ]
        mock_classify.return_value = classified

        ter_result = TERResult(
            session_id="s1",
            aggregate_ter=0.85,
            raw_ratio=0.85,
            phase_scores={"reasoning": 0.85, "tool_use": 1.0, "generation": 1.0},
            total_tokens=100,
            aligned_tokens=85,
            waste_tokens=15,
            classified_spans=classified,
        )
        mock_compute.return_value = ter_result
        mock_economics.return_value = MagicMock()

        result = analyze_session(mock_args)

        assert isinstance(result, TERResult)
        assert result.session_id == "s1"
        mock_load.assert_called_once_with("test.jsonl")

    @patch("ter_calculator.economics.compute_economics")
    @patch("ter_calculator.compute.compute_ter")
    @patch("ter_calculator.classifier.classify_spans")
    @patch("ter_calculator.intent_extraction.SlidingIntentExtractor")
    @patch("ter_calculator.loader.segment_spans")
    @patch("ter_calculator.loader.load_session")
    @patch("ter_calculator.waste.detect_waste_patterns")
    def test_waste_patterns_included_by_default(
        self,
        mock_waste,
        mock_load,
        mock_segment,
        mock_extractor_cls,
        mock_classify,
        mock_compute,
        mock_economics,
        mock_args,
    ):
        mock_session = MagicMock()
        mock_session.session_id = "s1"
        mock_session.user_prompts = ["test"]
        mock_load.return_value = mock_session
        mock_segment.return_value = []

        intent = IntentVector(
            text="test", embedding=np.zeros(384), confidence=0.9
        )
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [intent]
        mock_extractor_cls.return_value = mock_extractor
        mock_classify.return_value = []
        mock_waste.return_value = []

        ter_result = TERResult(
            session_id="s1",
            aggregate_ter=1.0,
            raw_ratio=1.0,
            phase_scores={},
            total_tokens=0,
            aligned_tokens=0,
            waste_tokens=0,
        )
        mock_compute.return_value = ter_result
        mock_economics.return_value = MagicMock()

        mock_args.no_waste_patterns = False
        result = analyze_session(mock_args)
        mock_waste.assert_called_once()
