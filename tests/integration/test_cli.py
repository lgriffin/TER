"""Integration tests for TER CLI."""

import json
from pathlib import Path

import pytest

from ter_calculator.cli import main


FIXTURE_PATH = str(
    Path(__file__).parent.parent / "fixtures" / "sample_session.jsonl"
)


class TestAnalyzeCommand:
    def test_analyze_text_output(self, capsys):
        exit_code = main(["analyze", FIXTURE_PATH])
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "TER" in output
        assert "Waste" in output

    def test_analyze_json_output(self, capsys):
        exit_code = main(["analyze", FIXTURE_PATH, "--format", "json"])
        assert exit_code == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "aggregate_ter" in data
        assert "phase_scores" in data
        assert "total_tokens" in data
        assert 0.0 <= data["aggregate_ter"] <= 1.0

    def test_analyze_missing_file(self, capsys):
        exit_code = main(["analyze", "/nonexistent/file.jsonl"])
        assert exit_code == 1

    def test_analyze_no_waste_patterns(self, capsys):
        exit_code = main([
            "analyze", FIXTURE_PATH, "--no-waste-patterns"
        ])
        assert exit_code == 0

    def test_analyze_custom_thresholds(self, capsys):
        exit_code = main([
            "analyze", FIXTURE_PATH,
            "--similarity-threshold", "0.50",
            "--confidence-threshold", "0.80",
        ])
        assert exit_code == 0

    def test_analyze_includes_economics(self, capsys):
        exit_code = main(["analyze", FIXTURE_PATH])
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "Economics" in output
        assert "Growth" in output

    def test_analyze_json_includes_economics(self, capsys):
        exit_code = main(["analyze", FIXTURE_PATH, "--format", "json"])
        assert exit_code == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "economics" in data
        econ = data["economics"]
        assert "total_input_tokens" in econ
        assert "cache_hit_rate" in econ
        assert "positional" in econ
        assert "input_growth" in econ
        assert "cost_model" in econ

    def test_analyze_custom_cost_model(self, capsys):
        exit_code = main([
            "analyze", FIXTURE_PATH,
            "--cost-model", "6.0,30.0,0.60,7.50",
        ])
        assert exit_code == 0

    def test_analyze_cost_model_sonnet(self, capsys):
        exit_code = main([
            "analyze", FIXTURE_PATH,
            "--cost-model", "sonnet",
        ])
        assert exit_code == 0

    def test_no_command(self, capsys):
        exit_code = main([])
        assert exit_code == 1
