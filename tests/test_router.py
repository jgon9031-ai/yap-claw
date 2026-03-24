"""Unit tests for AMOS router (System M).

Tests routing logic without any real model calls.
"""

import tempfile
from pathlib import Path

import pytest

from amos.memory import MemoryLayer
from amos.models import Query
from amos.router import AMOSRouter


@pytest.fixture
def memory(tmp_path: Path) -> MemoryLayer:
    """Create a fresh in-memory-like MemoryLayer for each test."""
    db_path = str(tmp_path / "test_memory.db")
    return MemoryLayer(db_path=db_path)


@pytest.fixture
def router(memory: MemoryLayer) -> AMOSRouter:
    return AMOSRouter(memory=memory)


class TestPrivacyRouting:
    """Personal/private queries must always route to local."""

    @pytest.mark.parametrize("text", [
        "내 체중 기록을 보여줘",
        "my personal notes about the meeting",
        "Show me my private documents",
        "나의 식단 계획을 알려줘",
        "What is my 비밀번호?",
        "Tell me about my 건강 records",
    ])
    def test_personal_queries_route_local(self, router: AMOSRouter, text: str) -> None:
        query = Query(text=text)
        decision = router.route(query)
        assert decision.target == "local"
        assert "personal" in decision.reason.lower() or "private" in decision.reason.lower()
        assert decision.confidence >= 0.9


class TestComplexityRouting:
    """Complex queries should route to cloud."""

    @pytest.mark.parametrize("text", [
        "Analyze the performance characteristics of this distributed system "
        "and compare it with alternative architectures. Explain the trade-offs "
        "in terms of scalability and fault tolerance.",
        "이 데이터를 분석하고 비교해서 설명해줘. 특히 트렌드와 패턴을 조사해서 요약해줘. "
        "각 지표별로 상세하게 분석하고 전체적인 결론을 도출해줘. 시계열 변화도 비교 분석 부탁해.",
    ])
    def test_complex_queries_route_cloud(self, router: AMOSRouter, text: str) -> None:
        query = Query(text=text)
        decision = router.route(query)
        assert decision.target == "cloud"
        assert "complexity" in decision.reason.lower()

    def test_simple_query_routes_local(self, router: AMOSRouter) -> None:
        query = Query(text="Hello")
        decision = router.route(query)
        assert decision.target == "local"
        assert "default" in decision.reason.lower()


class TestFailureRouting:
    """Past local failures should push routing to cloud."""

    def test_high_failure_rate_routes_cloud(
        self, memory: MemoryLayer, router: AMOSRouter
    ) -> None:
        # Seed memory with failures for a specific topic
        for _ in range(8):
            memory.record(
                query_text="explain quantum computing basics",
                target="local",
                model_used="qwen2.5-coder:7b",
                success=False,
                latency_ms=500,
            )
        for _ in range(2):
            memory.record(
                query_text="explain quantum computing fundamentals",
                target="local",
                model_used="qwen2.5-coder:7b",
                success=True,
                latency_ms=400,
            )

        query = Query(text="explain quantum computing")
        decision = router.route(query)
        assert decision.target == "cloud"
        assert "failure" in decision.reason.lower()

    def test_low_failure_rate_stays_local(
        self, memory: MemoryLayer, router: AMOSRouter
    ) -> None:
        # Seed memory with mostly successes
        for _ in range(9):
            memory.record(
                query_text="simple math question",
                target="local",
                model_used="qwen2.5-coder:7b",
                success=True,
                latency_ms=200,
            )
        memory.record(
            query_text="simple math problem",
            target="local",
            model_used="qwen2.5-coder:7b",
            success=False,
            latency_ms=300,
        )

        query = Query(text="simple math")
        decision = router.route(query)
        assert decision.target == "local"


class TestMemoryLayer:
    """Tests for the memory layer itself."""

    def test_record_and_retrieve(self, memory: MemoryLayer) -> None:
        memory.record("test query alpha", "local", "test-model", True, 100)
        results = memory.retrieve_similar("test query alpha")
        assert len(results) >= 1
        assert results[0].query_text == "test query alpha"
        assert results[0].success is True

    def test_stats_empty(self, memory: MemoryLayer) -> None:
        stats = memory.stats()
        assert stats["total_records"] == 0
        assert stats["per_target"] == {}

    def test_stats_populated(self, memory: MemoryLayer) -> None:
        memory.record("q1", "local", "model-a", True, 100)
        memory.record("q2", "local", "model-a", False, 200)
        memory.record("q3", "cloud", "model-b", True, 500)

        stats = memory.stats()
        assert stats["total_records"] == 3
        assert stats["per_target"]["local"]["total"] == 2
        assert stats["per_target"]["local"]["success_rate"] == 0.5
        assert stats["per_target"]["cloud"]["total"] == 1
        assert stats["per_target"]["cloud"]["success_rate"] == 1.0

    def test_local_failure_rate_no_data(self, memory: MemoryLayer) -> None:
        rate = memory.get_local_failure_rate("nonexistent topic")
        assert rate == 0.0


class TestComplexityScore:
    """Tests for the internal complexity scoring heuristic."""

    def test_short_simple_query(self, router: AMOSRouter) -> None:
        score = router._complexity_score("hello")
        assert score < 0.3

    def test_long_complex_query(self, router: AMOSRouter) -> None:
        text = (
            "analyze the performance and compare the results across "
            "multiple dimensions. explain the underlying patterns and "
            "summarize the key findings in a comprehensive report."
        )
        score = router._complexity_score(text)
        assert score >= 0.6

    def test_multiple_questions_increase_score(self, router: AMOSRouter) -> None:
        single = router._complexity_score("what is this?")
        multi = router._complexity_score("what is this? how does it work? why?")
        assert multi > single
