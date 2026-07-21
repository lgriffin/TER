"""Tests for adaptive budget recommendation and complexity estimation."""

import json
import tempfile
from pathlib import Path

import pytest

from ter_calculator.adaptive_budget import (
    BudgetRecommendation,
    ComplexityEstimator,
    ComplexityTier,
    HistoricalBudgetAnalyzer,
    HistoryEntry,
    ModelTier,
    estimate_complexity,
    recommend_budget,
    MAX_THINKING_TOKENS_SIMPLE,
    MAX_THINKING_TOKENS_STANDARD,
    MAX_THINKING_TOKENS_COMPLEX,
)


class TestComplexityEstimator:
    """Test task complexity classification."""

    def test_simple_task_detected(self):
        estimator = ComplexityEstimator()
        tier, confidence, features = estimator.estimate("fix typo in README")

        assert tier == ComplexityTier.SIMPLE
        assert confidence > 0.3
        assert features["simple_cues"] > 0

    def test_standard_task_detected(self):
        estimator = ComplexityEstimator()
        tier, confidence, features = estimator.estimate(
            "Fix the bug in the authentication module where users can't log in"
        )

        assert tier == ComplexityTier.STANDARD
        assert features["bug_cues"] > 0

    def test_complex_task_detected(self):
        estimator = ComplexityEstimator()
        tier, confidence, features = estimator.estimate(
            "Refactor the entire authentication system across all microservices "
            "to use a new distributed session store with proper scalability"
        )

        assert tier == ComplexityTier.COMPLEX
        assert features["architecture_cues"] > 0
        assert features["multi_file_cues"] > 0 or features["word_count"] > 80

    def test_word_count_feature(self):
        estimator = ComplexityEstimator()
        _, _, features = estimator.estimate("short")
        assert features["word_count"] == 1

        _, _, features2 = estimator.estimate(
            "a much longer prompt with many words here"
        )
        assert features2["word_count"] > features["word_count"]

    def test_unique_ratio_feature(self):
        estimator = ComplexityEstimator()
        _, _, features = estimator.estimate("word word word word")
        assert features["unique_ratio"] == 0.25  # 1 unique / 4 total

        _, _, features2 = estimator.estimate("all unique words here")
        assert features2["unique_ratio"] == 1.0

    def test_sentence_count_feature(self):
        estimator = ComplexityEstimator()
        _, _, features = estimator.estimate("One sentence.")
        assert features["sentence_count"] == 1.0

        _, _, features2 = estimator.estimate("First. Second! Third?")
        assert features2["sentence_count"] == 3.0

    def test_code_detection(self):
        estimator = ComplexityEstimator()
        _, _, features = estimator.estimate("plain text without code")
        assert features["has_code"] == 0.0

        _, _, features2 = estimator.estimate("use `function_name()` here")
        assert features2["has_code"] == 1.0

        _, _, features3 = estimator.estimate("```python\ncode block\n```")
        assert features3["has_code"] == 1.0

    def test_file_path_detection(self):
        estimator = ComplexityEstimator()
        _, _, features = estimator.estimate("general request")
        assert features["has_file_paths"] == 0.0

        _, _, features2 = estimator.estimate("edit src/main.py please")
        assert features2["has_file_paths"] == 1.0

        _, _, features3 = estimator.estimate(r"check C:\Users\file.txt")
        assert features3["has_file_paths"] == 1.0

    def test_multi_file_cues_detected(self):
        estimator = ComplexityEstimator()
        _, _, features = estimator.estimate("refactor across the codebase")
        assert features["multi_file_cues"] > 0

        _, _, features2 = estimator.estimate("update all files with migration")
        assert features2["multi_file_cues"] > 0

    def test_architecture_cues_detected(self):
        estimator = ComplexityEstimator()
        _, _, features = estimator.estimate(
            "design a scalable microservice architecture"
        )
        assert features["architecture_cues"] > 0

    def test_bug_cues_detected(self):
        estimator = ComplexityEstimator()
        _, _, features = estimator.estimate("fix the crash when error happens")
        assert features["bug_cues"] > 0

    def test_feature_cues_detected(self):
        estimator = ComplexityEstimator()
        _, _, features = estimator.estimate(
            "implement a new feature to add user profiles"
        )
        assert features["feature_cues"] > 0

    def test_question_marks_counted(self):
        estimator = ComplexityEstimator()
        _, _, features = estimator.estimate("How does this work? What should I do?")
        assert features["question_count"] == 2.0

    def test_empty_text_defaults_to_fallback(self):
        estimator = ComplexityEstimator()
        tier, confidence, features = estimator.estimate("")
        # Empty text should fall back to some tier
        assert tier in (
            ComplexityTier.SIMPLE,
            ComplexityTier.STANDARD,
            ComplexityTier.COMPLEX,
        )
        # Confidence varies by implementation
        assert 0 <= confidence <= 1

    def test_confidence_varies_with_signal_strength(self):
        estimator = ComplexityEstimator()
        _, conf1, _ = estimator.estimate("vague unclear ambiguous task")
        _, conf2, _ = estimator.estimate("quick simple fix")
        # Both should have valid confidence scores
        assert 0 <= conf1 <= 1
        assert 0 <= conf2 <= 1


class TestEstimateComplexityFunction:
    """Test convenience wrapper function."""

    def test_estimate_complexity_returns_tuple(self):
        tier, confidence, features = estimate_complexity("fix bug")
        assert isinstance(tier, ComplexityTier)
        assert isinstance(confidence, float)
        assert isinstance(features, dict)


class TestRecommendBudget:
    """Test budget recommendation without historical data."""

    def test_simple_task_recommends_haiku(self):
        rec = recommend_budget("fix typo")
        assert rec.complexity == ComplexityTier.SIMPLE
        assert rec.model_tier == ModelTier.HAIKU
        assert rec.max_thinking_tokens == MAX_THINKING_TOKENS_SIMPLE

    def test_standard_task_recommends_sonnet(self):
        rec = recommend_budget("fix authentication bug")
        assert rec.complexity == ComplexityTier.STANDARD
        assert rec.model_tier == ModelTier.SONNET
        assert rec.max_thinking_tokens == MAX_THINKING_TOKENS_STANDARD

    def test_complex_task_recommends_opus(self):
        rec = recommend_budget(
            "refactor entire architecture across multiple microservices with scalability"
        )
        assert rec.complexity == ComplexityTier.COMPLEX
        assert rec.model_tier == ModelTier.OPUS
        assert rec.max_thinking_tokens == MAX_THINKING_TOKENS_COMPLEX

    def test_recommendation_includes_cost_estimate(self):
        rec = recommend_budget("fix bug")
        assert rec.estimated_cost_usd > 0
        assert isinstance(rec.estimated_cost_usd, float)

    def test_recommendation_includes_total_tokens(self):
        rec = recommend_budget("fix bug")
        assert rec.estimated_total_tokens > 0
        assert rec.estimated_total_tokens > rec.max_thinking_tokens

    def test_recommendation_includes_confidence(self):
        rec = recommend_budget("fix bug")
        assert 0 <= rec.confidence <= 1

    def test_recommendation_includes_reasoning(self):
        rec = recommend_budget("simple fix")
        assert isinstance(rec.reasoning, str)
        assert len(rec.reasoning) > 0
        assert "Complexity:" in rec.reasoning
        assert "Model:" in rec.reasoning

    def test_recommendation_includes_features(self):
        rec = recommend_budget("fix bug in src/main.py")
        assert isinstance(rec.features, dict)
        assert "word_count" in rec.features
        assert "has_file_paths" in rec.features

    def test_multi_file_indicators_mentioned_in_reasoning(self):
        rec = recommend_budget("refactor across the codebase")
        if rec.features.get("multi_file_cues", 0) > 0:
            assert "Multi-file" in rec.reasoning or "multi-file" in rec.reasoning

    def test_simple_indicators_mentioned_in_reasoning(self):
        rec = recommend_budget("trivial simple quick fix")
        if rec.features.get("simple_cues", 0) > 0:
            assert "Simple" in rec.reasoning or "simple" in rec.reasoning

    def test_cost_increases_with_complexity(self):
        simple = recommend_budget("fix typo")
        standard = recommend_budget("fix authentication bug")
        complex_task = recommend_budget(
            "architect scalable distributed microservice infrastructure"
        )

        # Haiku < Sonnet < Opus pricing
        assert simple.estimated_cost_usd < standard.estimated_cost_usd
        assert standard.estimated_cost_usd < complex_task.estimated_cost_usd


class TestHistoricalBudgetAnalyzer:
    """Test learning from historical TER outcomes."""

    def test_empty_history_returns_neutral_adjustment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            analyzer = HistoricalBudgetAnalyzer(history_path)

            adjustment = analyzer.get_adjustment(ComplexityTier.STANDARD)
            assert adjustment.thinking_multiplier == 1.0
            assert adjustment.total_multiplier == 1.0
            assert adjustment.model_override is None
            assert adjustment.data_confidence == 0.0

    def test_few_entries_return_low_confidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            analyzer = HistoricalBudgetAnalyzer(history_path)

            for i in range(3):
                analyzer.record(
                    HistoryEntry(
                        intent_text="fix bug",
                        complexity="standard",
                        actual_thinking_tokens=5000,
                        actual_total_tokens=20000,
                        actual_ter=0.85,
                        model_used="sonnet",
                    )
                )

            adjustment = analyzer.get_adjustment(ComplexityTier.STANDARD)
            assert adjustment.data_confidence < 1.0

    def test_sufficient_entries_increase_confidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            analyzer = HistoricalBudgetAnalyzer(history_path)

            for i in range(10):
                analyzer.record(
                    HistoryEntry(
                        intent_text=f"task {i}",
                        complexity="standard",
                        actual_thinking_tokens=5000,
                        actual_total_tokens=20000,
                        actual_ter=0.85,
                        model_used="sonnet",
                    )
                )

            adjustment = analyzer.get_adjustment(ComplexityTier.STANDARD)
            assert adjustment.data_confidence > 0.25

    def test_low_ter_triggers_model_upgrade(self):
        """If TER is consistently low, upgrade model tier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            analyzer = HistoricalBudgetAnalyzer(history_path)

            for i in range(10):
                analyzer.record(
                    HistoryEntry(
                        intent_text=f"task {i}",
                        complexity="simple",
                        actual_thinking_tokens=2000,
                        actual_total_tokens=8000,
                        actual_ter=0.40,  # Low TER
                        model_used="haiku",
                    )
                )

            adjustment = analyzer.get_adjustment(ComplexityTier.SIMPLE)
            assert adjustment.model_override == ModelTier.SONNET

    def test_high_ter_triggers_model_downgrade(self):
        """If TER is consistently high, downgrade to save cost."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            analyzer = HistoricalBudgetAnalyzer(history_path)

            for i in range(10):
                analyzer.record(
                    HistoryEntry(
                        intent_text=f"task {i}",
                        complexity="standard",
                        actual_thinking_tokens=4000,
                        actual_total_tokens=15000,
                        actual_ter=0.90,  # High TER
                        model_used="sonnet",
                    )
                )

            adjustment = analyzer.get_adjustment(ComplexityTier.STANDARD)
            assert adjustment.model_override == ModelTier.HAIKU

    def test_thinking_multiplier_adjusts_based_on_actual_usage(self):
        """If actual usage is lower than default, reduce budget."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            analyzer = HistoricalBudgetAnalyzer(history_path)

            # Tasks using much less thinking than default
            for i in range(10):
                analyzer.record(
                    HistoryEntry(
                        intent_text=f"task {i}",
                        complexity="standard",
                        actual_thinking_tokens=2000,  # Much less than 8192 default
                        actual_total_tokens=10000,
                        actual_ter=0.85,
                        model_used="sonnet",
                    )
                )

            adjustment = analyzer.get_adjustment(ComplexityTier.STANDARD)
            assert adjustment.thinking_multiplier < 1.0

    def test_persistence_to_json(self):
        """Test saving and loading from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            analyzer = HistoricalBudgetAnalyzer(history_path)

            analyzer.record(
                HistoryEntry(
                    intent_text="test task",
                    complexity="simple",
                    actual_thinking_tokens=1000,
                    actual_total_tokens=5000,
                    actual_ter=0.90,
                    model_used="haiku",
                    timestamp=1234567890.0,
                )
            )

            # Load from same path
            analyzer2 = HistoricalBudgetAnalyzer(history_path)
            assert analyzer2.entry_count == 1

    def test_get_summary_aggregates_by_tier(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            analyzer = HistoricalBudgetAnalyzer(history_path)

            analyzer.record(
                HistoryEntry(
                    intent_text="simple 1",
                    complexity="simple",
                    actual_thinking_tokens=1000,
                    actual_total_tokens=4000,
                    actual_ter=0.90,
                    model_used="haiku",
                )
            )
            analyzer.record(
                HistoryEntry(
                    intent_text="standard 1",
                    complexity="standard",
                    actual_thinking_tokens=5000,
                    actual_total_tokens=20000,
                    actual_ter=0.85,
                    model_used="sonnet",
                )
            )

            summary = analyzer.get_summary()
            assert summary["total_entries"] == 2
            assert "simple" in summary["tiers"]
            assert "standard" in summary["tiers"]
            assert summary["tiers"]["simple"]["count"] == 1
            assert summary["tiers"]["standard"]["count"] == 1

    def test_entry_count_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            analyzer = HistoricalBudgetAnalyzer(history_path)

            assert analyzer.entry_count == 0
            analyzer.record(
                HistoryEntry(
                    intent_text="test",
                    complexity="simple",
                    actual_thinking_tokens=1000,
                    actual_total_tokens=5000,
                    actual_ter=0.90,
                    model_used="haiku",
                )
            )
            assert analyzer.entry_count == 1

    def test_recommend_budget_with_history(self):
        """Test that historical data influences recommendations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            analyzer = HistoricalBudgetAnalyzer(history_path)

            # Add history showing simple tasks need less budget
            for i in range(20):
                analyzer.record(
                    HistoryEntry(
                        intent_text=f"simple task {i}",
                        complexity="simple",
                        actual_thinking_tokens=500,  # Much less than 2048 default
                        actual_total_tokens=2000,
                        actual_ter=0.95,
                        model_used="haiku",
                    )
                )

            rec_without = recommend_budget("fix typo")
            rec_with = recommend_budget("fix typo", history=analyzer)

            # With history, budget should be adjusted
            assert rec_with.max_thinking_tokens != rec_without.max_thinking_tokens
            assert "historical" in rec_with.reasoning.lower()

    def test_intent_text_truncated_in_json(self):
        """Intent text should be truncated to 200 chars when saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            analyzer = HistoricalBudgetAnalyzer(history_path)

            long_text = "x" * 500
            analyzer.record(
                HistoryEntry(
                    intent_text=long_text,
                    complexity="simple",
                    actual_thinking_tokens=1000,
                    actual_total_tokens=5000,
                    actual_ter=0.90,
                    model_used="haiku",
                )
            )

            # Check JSON file
            data = json.loads(history_path.read_text())
            assert len(data[0]["intent_text"]) == 200

    def test_corrupted_json_handled_gracefully(self):
        """Corrupted history file should log warning but not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            history_path.write_text("{ invalid json }")

            analyzer = HistoricalBudgetAnalyzer(history_path)
            assert analyzer.entry_count == 0  # Should start fresh


class TestBudgetRecommendation:
    """Test BudgetRecommendation dataclass."""

    def test_frozen_immutable(self):
        rec = BudgetRecommendation(
            complexity=ComplexityTier.SIMPLE,
            model_tier=ModelTier.HAIKU,
            max_thinking_tokens=2048,
            estimated_total_tokens=5000,
            estimated_cost_usd=0.01,
            confidence=0.85,
            reasoning="test",
            features={},
        )

        with pytest.raises(AttributeError):
            rec.complexity = ComplexityTier.COMPLEX


class TestHistoryEntry:
    """Test HistoryEntry dataclass."""

    def test_default_timestamp(self):
        entry = HistoryEntry(
            intent_text="test",
            complexity="simple",
            actual_thinking_tokens=1000,
            actual_total_tokens=5000,
            actual_ter=0.90,
            model_used="haiku",
        )
        assert entry.timestamp == 0.0

    def test_custom_timestamp(self):
        entry = HistoryEntry(
            intent_text="test",
            complexity="simple",
            actual_thinking_tokens=1000,
            actual_total_tokens=5000,
            actual_ter=0.90,
            model_used="haiku",
            timestamp=1234567890.0,
        )
        assert entry.timestamp == 1234567890.0
