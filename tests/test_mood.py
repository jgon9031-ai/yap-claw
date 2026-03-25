"""Unit tests for the MOOD layer — MoodDetector, MoodContext, NEST integration."""

from __future__ import annotations

import pytest

from amos.memory import MemoryLayer
from amos.mood import MoodContext, MoodDetector, MoodState


@pytest.fixture()
def detector() -> MoodDetector:
    return MoodDetector()


@pytest.fixture()
def memory() -> MemoryLayer:
    return MemoryLayer(db_path=":memory:")


# ── MoodDetector.detect() ───────────────────────────────────────────────────


class TestMoodDetect:
    """Tests for MoodDetector.detect() returning the correct MoodState."""

    def test_frustrated_korean(self, detector: MoodDetector) -> None:
        assert detector.detect("또 실패네 왜 안 되는 거야") == MoodState.FRUSTRATED

    def test_frustrated_english(self, detector: MoodDetector) -> None:
        assert detector.detect("This is still failing, not working at all") == MoodState.FRUSTRATED

    def test_satisfied_korean(self, detector: MoodDetector) -> None:
        assert detector.detect("응 좋아 완벽해") == MoodState.SATISFIED

    def test_satisfied_english(self, detector: MoodDetector) -> None:
        assert detector.detect("That's perfect, great work") == MoodState.SATISFIED

    def test_curious_korean(self, detector: MoodDetector) -> None:
        assert detector.detect("어떻게 가능한가 설명해줘") == MoodState.CURIOUS

    def test_curious_english(self, detector: MoodDetector) -> None:
        assert detector.detect("How does this work? Explain please") == MoodState.CURIOUS

    def test_urgent_korean(self, detector: MoodDetector) -> None:
        assert detector.detect("지금 당장 빨리 해줘") == MoodState.URGENT

    def test_urgent_english(self, detector: MoodDetector) -> None:
        assert detector.detect("Do this right now, immediately!") == MoodState.URGENT

    def test_neutral_fallback(self, detector: MoodDetector) -> None:
        assert detector.detect("76.5") == MoodState.NEUTRAL

    def test_neutral_no_signals(self, detector: MoodDetector) -> None:
        assert detector.detect("점심 라면 저녁 떡볶이") == MoodState.NEUTRAL

    def test_empty_string(self, detector: MoodDetector) -> None:
        assert detector.detect("") == MoodState.NEUTRAL


# ── MoodDetector.detect_with_context() ──────────────────────────────────────


class TestMoodDetectWithContext:
    """Tests for detect_with_context returning MoodContext with correct fields."""

    def test_returns_mood_context(self, detector: MoodDetector) -> None:
        ctx = detector.detect_with_context("응 좋아")
        assert isinstance(ctx, MoodContext)

    def test_frustrated_signals_found(self, detector: MoodDetector) -> None:
        ctx = detector.detect_with_context("또 실패네 계속 실패")
        assert ctx.mood == MoodState.FRUSTRATED
        assert "또 실패" in ctx.signals_found or "계속 실패" in ctx.signals_found
        assert len(ctx.signals_found) >= 2

    def test_satisfied_confidence(self, detector: MoodDetector) -> None:
        ctx = detector.detect_with_context("완벽 훌륭 잘했어")
        assert ctx.mood == MoodState.SATISFIED
        assert 0.0 < ctx.confidence <= 1.0

    def test_neutral_directive_is_empty(self, detector: MoodDetector) -> None:
        ctx = detector.detect_with_context("76.5")
        assert ctx.mood == MoodState.NEUTRAL
        assert ctx.directive == ""

    def test_non_neutral_directive_is_nonempty(self, detector: MoodDetector) -> None:
        for text, mood in [
            ("또 실패", MoodState.FRUSTRATED),
            ("응 좋아", MoodState.SATISFIED),
            ("어떻게 가능한가", MoodState.CURIOUS),
            ("지금 당장", MoodState.URGENT),
        ]:
            ctx = detector.detect_with_context(text)
            assert ctx.mood == mood
            assert ctx.directive != "", f"Expected non-empty directive for {mood}"

    def test_multiple_signals_one_text(self, detector: MoodDetector) -> None:
        ctx = detector.detect_with_context("빨리 즉시 당장 바로")
        assert ctx.mood == MoodState.URGENT
        assert len(ctx.signals_found) >= 3

    def test_neutral_confidence_is_one(self, detector: MoodDetector) -> None:
        ctx = detector.detect_with_context("평범한 텍스트")
        assert ctx.mood == MoodState.NEUTRAL
        assert ctx.confidence == 1.0

    def test_neutral_signals_empty(self, detector: MoodDetector) -> None:
        ctx = detector.detect_with_context("hello there")
        assert ctx.mood == MoodState.NEUTRAL
        assert ctx.signals_found == []


# ── Memory integration ──────────────────────────────────────────────────────


class TestMoodMemoryIntegration:
    """Tests for mood stored in Experience via memory.record() and mood_breakdown()."""

    def test_record_with_mood(self, memory: MemoryLayer) -> None:
        memory.record("응 좋아", "local", "qwen2.5-coder:7b", True, 200, mood="satisfied")
        exps = memory.get_recent(1)
        assert len(exps) == 1
        assert exps[0].mood == "satisfied"

    def test_record_without_mood_defaults_none(self, memory: MemoryLayer) -> None:
        memory.record("hello", "local", "qwen2.5-coder:7b", True, 200)
        exps = memory.get_recent(1)
        assert len(exps) == 1
        assert exps[0].mood is None

    def test_mood_breakdown_returns_dict(self, memory: MemoryLayer) -> None:
        memory.record("a", "local", "m", True, 100, mood="satisfied")
        memory.record("b", "local", "m", True, 100, mood="satisfied")
        memory.record("c", "cloud", "m", True, 100, mood="frustrated")
        bd = memory.mood_breakdown()
        assert isinstance(bd, dict)
        assert bd["satisfied"] == 2
        assert bd["frustrated"] == 1

    def test_mood_breakdown_groups_null_as_neutral(self, memory: MemoryLayer) -> None:
        memory.record("a", "local", "m", True, 100)
        memory.record("b", "local", "m", True, 100, mood="curious")
        bd = memory.mood_breakdown()
        assert "neutral" in bd
        assert bd["neutral"] == 1
        assert bd["curious"] == 1

    def test_mood_breakdown_empty_db(self, memory: MemoryLayer) -> None:
        bd = memory.mood_breakdown()
        assert bd == {}

    def test_mood_stored_in_experience_model(self) -> None:
        from amos.models import Experience
        exp = Experience(
            query_text="test",
            target="local",
            model_used="test",
            success=True,
            latency_ms=100,
            mood="urgent",
        )
        assert exp.mood == "urgent"

    def test_mood_none_in_experience_model(self) -> None:
        from amos.models import Experience
        exp = Experience(
            query_text="test",
            target="local",
            model_used="test",
            success=True,
            latency_ms=100,
        )
        assert exp.mood is None


# ── Dashboard mood_breakdown ────────────────────────────────────────────────


class TestDashboardMoodBreakdown:
    """Tests for RoutingDashboard.mood_breakdown() display."""

    def test_mood_breakdown_display(self, memory: MemoryLayer) -> None:
        from amos.dashboard import RoutingDashboard

        for i in range(10):
            memory.record(f"q{i}", "local", "m", True, 100, mood="neutral")
        for i in range(5):
            memory.record(f"q{i}", "local", "m", True, 100, mood="satisfied")
        memory.record("q", "cloud", "m", True, 100, mood="frustrated")

        dash = RoutingDashboard(memory)
        text = dash.mood_breakdown()
        assert "Mood distribution:" in text
        assert "neutral" in text
        assert "satisfied" in text
        assert "frustrated" in text

    def test_mood_breakdown_empty(self, memory: MemoryLayer) -> None:
        from amos.dashboard import RoutingDashboard

        dash = RoutingDashboard(memory)
        text = dash.mood_breakdown()
        assert "No mood data" in text
