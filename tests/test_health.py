"""Unit tests for SkillHealthTracker (cognee-style failure tracking)."""

from pathlib import Path

import pytest

from amos.health import SkillHealthTracker
from amos.memory import MemoryLayer
from amos.models import Experience


@pytest.fixture
def memory(tmp_path: Path) -> MemoryLayer:
    """Fresh MemoryLayer for each test."""
    return MemoryLayer(db_path=str(tmp_path / "test_health.db"))


@pytest.fixture
def tracker(memory: MemoryLayer) -> SkillHealthTracker:
    return SkillHealthTracker(memory=memory, failure_threshold=0.4)


def _make_experience(
    text: str, target: str = "local", success: bool = True, latency_ms: int = 100
) -> Experience:
    return Experience(
        query_text=text,
        target=target,
        model_used="test-model",
        success=success,
        latency_ms=latency_ms,
    )


class TestObserve:
    """Tests for recording outcomes."""

    def test_observe_increments_counters(self, tracker: SkillHealthTracker) -> None:
        tracker.observe(_make_experience("debug this function", success=True))
        tracker.observe(_make_experience("implement a class", success=False))

        # Both are "code" category — should have 1 success, 1 failure for local
        issues = tracker.inspect()
        # Only 2 samples, below the minimum of 3
        assert len(issues) == 0

    def test_observe_separates_targets(self, tracker: SkillHealthTracker) -> None:
        tracker.observe(_make_experience("debug this", target="local", success=True))
        tracker.observe(_make_experience("debug that", target="cloud", success=False))

        # Separate target counts — neither has 3+ samples
        assert tracker.inspect() == []


class TestInspect:
    """Tests for failure detection."""

    def test_no_issues_when_healthy(self, tracker: SkillHealthTracker) -> None:
        for i in range(5):
            tracker.observe(_make_experience(f"debug function {i}", success=True))
        assert tracker.inspect() == []

    def test_detects_high_failure_rate(self, tracker: SkillHealthTracker) -> None:
        """Should flag categories exceeding the failure threshold."""
        # 3 failures, 1 success = 75% failure rate > 40% threshold
        tracker.observe(_make_experience("debug function A", success=False))
        tracker.observe(_make_experience("debug class B", success=False))
        tracker.observe(_make_experience("implement feature C", success=False))
        tracker.observe(_make_experience("refactor code D", success=True))

        issues = tracker.inspect()
        assert len(issues) == 1
        assert issues[0].category == "code"
        assert issues[0].failure_rate == 0.75
        assert issues[0].current_target == "local"

    def test_ignores_small_samples(self, tracker: SkillHealthTracker) -> None:
        """Categories with fewer than 3 observations should be ignored."""
        tracker.observe(_make_experience("debug function", success=False))
        tracker.observe(_make_experience("implement class", success=False))
        assert tracker.inspect() == []


class TestAmend:
    """Tests for routing amendment suggestions."""

    def test_suggests_opposite_target(self, tracker: SkillHealthTracker) -> None:
        # Create a health issue on local
        for _ in range(4):
            tracker.observe(_make_experience("debug this function", success=False))

        issues = tracker.inspect()
        assert len(issues) == 1

        amendment = tracker.amend(issues[0])
        assert amendment.old_target == "local"
        assert amendment.new_target == "cloud"
        assert amendment.category == "code"
        assert "switching to cloud" in amendment.reason

    def test_suggests_local_for_cloud_failures(self, tracker: SkillHealthTracker) -> None:
        for _ in range(4):
            tracker.observe(
                _make_experience("debug this function", target="cloud", success=False)
            )

        issues = tracker.inspect()
        amendment = tracker.amend(issues[0])
        assert amendment.old_target == "cloud"
        assert amendment.new_target == "local"

    def test_amendment_stored_in_active(self, tracker: SkillHealthTracker) -> None:
        for _ in range(4):
            tracker.observe(_make_experience("debug function", success=False))

        issues = tracker.inspect()
        tracker.amend(issues[0])

        active = tracker.get_active_amendments()
        assert "code" in active
        assert active["code"].new_target == "cloud"


class TestEvaluate:
    """Tests for the amendment evaluation loop."""

    def test_keeps_amendment_when_improved(self, tracker: SkillHealthTracker) -> None:
        # Set up failure baseline
        for _ in range(4):
            tracker.observe(_make_experience("debug function", success=False))

        issues = tracker.inspect()
        amendment = tracker.amend(issues[0])

        # New experiences on cloud are successful
        new_exps = [
            _make_experience("debug this class", target="cloud", success=True),
            _make_experience("implement feature", target="cloud", success=True),
        ]

        result = tracker.evaluate(amendment, new_exps)
        assert result is True
        assert "code" in tracker.get_active_amendments()

    def test_rolls_back_when_not_improved(self, tracker: SkillHealthTracker) -> None:
        # Set up some successes on local baseline
        tracker.observe(_make_experience("debug function A", success=True))
        tracker.observe(_make_experience("debug function B", success=True))
        tracker.observe(_make_experience("debug function C", success=False))
        tracker.observe(_make_experience("debug function D", success=False))
        tracker.observe(_make_experience("debug function E", success=False))

        issues = tracker.inspect()
        amendment = tracker.amend(issues[0])

        # New experiences on cloud also fail — no improvement
        new_exps = [
            _make_experience("debug this class", target="cloud", success=False),
            _make_experience("implement feature", target="cloud", success=False),
        ]

        result = tracker.evaluate(amendment, new_exps)
        assert result is False
        assert "code" not in tracker.get_active_amendments()

    def test_keeps_amendment_with_no_data(self, tracker: SkillHealthTracker) -> None:
        """When no relevant new experiences exist, keep the amendment pending."""
        for _ in range(4):
            tracker.observe(_make_experience("debug function", success=False))

        issues = tracker.inspect()
        amendment = tracker.amend(issues[0])

        # No relevant new experiences (different category)
        new_exps = [_make_experience("calculate equation", target="cloud", success=True)]

        result = tracker.evaluate(amendment, new_exps)
        assert result is True  # Keep until we have data
