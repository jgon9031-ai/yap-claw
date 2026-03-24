"""Unit tests for the AMOS Routing Dashboard."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone, timedelta

import pytest

from amos.dashboard import RoutingDashboard, _bar, _spark
from amos.memory import MemoryLayer


@pytest.fixture()
def memory() -> MemoryLayer:
    """In-memory MemoryLayer for testing."""
    ml = MemoryLayer(db_path=":memory:")
    return ml


def _seed_data(memory: MemoryLayer, local_count: int = 10, cloud_count: int = 5) -> None:
    """Seed the memory with sample experiences."""
    for i in range(local_count):
        success = i % 5 != 0  # 80% success rate
        memory.record(
            query_text=f"local query {i}",
            target="local",
            model_used="qwen2.5-coder:7b",
            success=success,
            latency_ms=200 + i * 10,
        )
    for i in range(cloud_count):
        success = i % 10 != 0  # 90% success rate
        memory.record(
            query_text=f"analyze and compare topic {i}",
            target="cloud",
            model_used="gpt-4o-mini",
            success=success,
            latency_ms=1500 + i * 50,
        )


class TestRoutingDashboard:
    """Tests for the dashboard summary and export."""

    def test_empty_memory(self, memory: MemoryLayer) -> None:
        dash = RoutingDashboard(memory)
        summary = dash.summary()
        assert "AMOS Routing Dashboard" in summary
        assert "Total queries:       0" in summary

    def test_summary_with_data(self, memory: MemoryLayer) -> None:
        _seed_data(memory, local_count=10, cloud_count=5)
        dash = RoutingDashboard(memory)
        summary = dash.summary()

        assert "Total queries:" in summary
        assert "Local queries:" in summary
        assert "Cloud queries:" in summary
        assert "Success rates:" in summary
        assert "Avg latency:" in summary
        assert "ms" in summary

    def test_export_json_structure(self, memory: MemoryLayer) -> None:
        _seed_data(memory, local_count=8, cloud_count=4)
        dash = RoutingDashboard(memory)
        data = dash.export_json()

        assert data["total_queries"] == 12
        assert data["local"]["count"] == 8
        assert data["cloud"]["count"] == 4
        assert 0.0 <= data["local"]["success_rate"] <= 1.0
        assert 0.0 <= data["cloud"]["success_rate"] <= 1.0
        assert isinstance(data["local"]["avg_latency_ms"], int)
        assert isinstance(data["cloud"]["avg_latency_ms"], int)
        assert isinstance(data["routing_reasons"], dict)
        assert isinstance(data["trends"], dict)

    def test_export_json_empty(self, memory: MemoryLayer) -> None:
        dash = RoutingDashboard(memory)
        data = dash.export_json()
        assert data["total_queries"] == 0
        assert data["local"]["count"] == 0
        assert data["cloud"]["count"] == 0

    def test_export_json_serializable(self, memory: MemoryLayer) -> None:
        _seed_data(memory)
        dash = RoutingDashboard(memory)
        data = dash.export_json()
        # Must be JSON-serializable
        serialized = json.dumps(data)
        assert isinstance(serialized, str)

    def test_print_does_not_raise(self, memory: MemoryLayer, capsys: pytest.CaptureFixture[str]) -> None:
        _seed_data(memory)
        dash = RoutingDashboard(memory)
        dash.print()
        captured = capsys.readouterr()
        assert "AMOS Routing Dashboard" in captured.out


class TestDashboardHelpers:
    """Tests for helper rendering functions."""

    def test_bar_full(self) -> None:
        assert _bar(1.0, 10) == "██████████"

    def test_bar_empty(self) -> None:
        assert _bar(0.0, 10) == "░░░░░░░░░░"

    def test_bar_half(self) -> None:
        result = _bar(0.5, 10)
        assert result.count("█") == 5
        assert result.count("░") == 5

    def test_spark_empty(self) -> None:
        assert _spark([]) == ""

    def test_spark_uniform(self) -> None:
        result = _spark([5.0, 5.0, 5.0])
        assert len(result) == 3

    def test_spark_ascending(self) -> None:
        result = _spark([0, 1, 2, 3, 4, 5, 6, 7])
        assert len(result) == 8
        # Last char should be tallest
        assert result[-1] == "█"

    def test_spark_single_value(self) -> None:
        result = _spark([42.0])
        assert len(result) == 1


class TestDashboardFromHarness:
    """Test that AMOS.dashboard() returns a working dashboard."""

    def test_harness_dashboard_method(self) -> None:
        """Verify AMOS.dashboard() creates a RoutingDashboard instance."""
        from unittest.mock import patch, MagicMock

        with patch("amos.harness.LocalExecutor"), \
             patch("amos.harness.AMOSRouter"):
            from amos.harness import AMOS
            harness = AMOS({"memory_db": ":memory:"})
            dash = harness.dashboard()
            assert isinstance(dash, RoutingDashboard)
            summary = dash.summary()
            assert "AMOS Routing Dashboard" in summary
            harness.close()
