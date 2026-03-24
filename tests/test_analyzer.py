"""Unit tests for SessionAnalyzer (Hermes-style pattern extraction)."""

from pathlib import Path

import pytest

from amos.analyzer import SessionAnalyzer, categorize_query
from amos.memory import MemoryLayer


@pytest.fixture
def memory(tmp_path: Path) -> MemoryLayer:
    """Fresh MemoryLayer for each test."""
    return MemoryLayer(db_path=str(tmp_path / "test_analyzer.db"))


@pytest.fixture
def analyzer(memory: MemoryLayer) -> SessionAnalyzer:
    return SessionAnalyzer(memory=memory, analyze_every=5)


class TestCategorizeQuery:
    """Tests for the keyword-based query categorizer."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("debug this function", "code"),
            ("implement a new class", "code"),
            ("calculate the equation", "math"),
            ("write an essay about AI", "writing"),
            ("translate this to Korean", "translation"),
            ("번역해줘", "translation"),
            ("analyze the data trends", "analysis"),
            ("비교해서 분석해줘", "analysis"),
            ("create a poem about nature", "creative"),
            ("hello world", "general"),
        ],
    )
    def test_categories(self, text: str, expected: str) -> None:
        assert categorize_query(text) == expected


class TestShouldAnalyze:
    """Tests for the analysis trigger logic."""

    def test_not_enough_records(self, analyzer: SessionAnalyzer, memory: MemoryLayer) -> None:
        """Should not trigger when fewer than analyze_every new records exist."""
        for i in range(4):
            memory.record(f"query {i}", "local", "model-a", True, 100)
        assert analyzer.should_analyze() is False

    def test_enough_records(self, analyzer: SessionAnalyzer, memory: MemoryLayer) -> None:
        """Should trigger once analyze_every new records accumulate."""
        for i in range(5):
            memory.record(f"query {i}", "local", "model-a", True, 100)
        assert analyzer.should_analyze() is True

    def test_resets_after_analysis(self, analyzer: SessionAnalyzer, memory: MemoryLayer) -> None:
        """After analysis, counter resets and requires another batch."""
        for i in range(5):
            memory.record(f"query {i}", "local", "model-a", True, 100)
        analyzer.analyze()
        assert analyzer.should_analyze() is False

        # Add more records to trigger again
        for i in range(5):
            memory.record(f"new query {i}", "local", "model-a", True, 100)
        assert analyzer.should_analyze() is True


class TestAnalyze:
    """Tests for the analysis logic itself."""

    def test_empty_memory(self, analyzer: SessionAnalyzer) -> None:
        """Analysis on empty memory returns empty hints."""
        result = analyzer.analyze()
        assert result.sample_size == 0
        assert result.routing_hints == {}

    def test_extracts_cloud_hint_from_failures(
        self, analyzer: SessionAnalyzer, memory: MemoryLayer
    ) -> None:
        """When local consistently fails for a category, hint should prefer cloud."""
        # Code queries: local fails, cloud succeeds
        for _ in range(5):
            memory.record("debug this function please", "local", "local-model", False, 200)
        for _ in range(5):
            memory.record("debug this class method", "cloud", "cloud-model", True, 500)

        result = analyzer.analyze()
        assert result.sample_size > 0
        # "code" category should hint toward cloud
        assert result.routing_hints.get("code") == "cloud"

    def test_extracts_local_hint_from_successes(
        self, analyzer: SessionAnalyzer, memory: MemoryLayer
    ) -> None:
        """When local consistently succeeds and cloud fails, hint prefers local."""
        for _ in range(5):
            memory.record("calculate the equation", "local", "local-model", True, 100)
        for _ in range(5):
            memory.record("calculate the formula", "cloud", "cloud-model", False, 500)

        result = analyzer.analyze()
        assert result.routing_hints.get("math") == "local"

    def test_get_routing_hints_returns_cached(
        self, analyzer: SessionAnalyzer, memory: MemoryLayer
    ) -> None:
        """get_routing_hints returns the cached result from last analyze()."""
        for _ in range(5):
            memory.record("debug the function", "local", "model-a", False, 200)
        for _ in range(5):
            memory.record("debug this class", "cloud", "model-b", True, 300)

        assert analyzer.get_routing_hints() == {}  # Before analysis
        analyzer.analyze()
        hints = analyzer.get_routing_hints()
        assert isinstance(hints, dict)
        assert "code" in hints

    def test_last_result_property(self, analyzer: SessionAnalyzer, memory: MemoryLayer) -> None:
        """last_result returns None before analysis, AnalysisResult after."""
        assert analyzer.last_result is None
        for i in range(5):
            memory.record(f"query {i}", "local", "model-a", True, 100)
        result = analyzer.analyze()
        assert analyzer.last_result is result
        assert result.sample_size > 0
