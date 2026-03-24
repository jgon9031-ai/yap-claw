"""Integration tests for AMOS — requires real Ollama running.

Run with: pytest -m integration
Skip automatically if Ollama is not reachable.
"""

from __future__ import annotations

import pytest

from amos.exceptions import LocalUnavailableError
from amos.executor import LocalExecutor
from amos.models import RoutingDecision


def _ollama_available() -> bool:
    """Check if Ollama is running and reachable."""
    try:
        executor = LocalExecutor({"local_timeout": 5.0, "local_max_retries": 0})
        routing = RoutingDecision(target="local", reason="probe", confidence=1.0)
        executor.execute("hi", routing)
        return True
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama is not running or model not available",
)


@pytest.mark.integration
@requires_ollama
class TestOllamaIntegration:
    """Integration tests against a real Ollama instance."""

    def test_hello_world_python(self) -> None:
        """Send a simple coding prompt and verify the response contains 'print'."""
        executor = LocalExecutor({
            "local_model": "qwen2.5-coder:7b",
            "local_timeout": 60.0,
            "local_max_retries": 1,
        })
        routing = RoutingDecision(target="local", reason="integration test", confidence=1.0)

        response = executor.execute("print hello world in python", routing)

        assert response.text, "Response should not be empty"
        assert "print" in response.text.lower(), (
            f"Expected 'print' in response, got: {response.text[:200]}"
        )
        assert response.model_used == "qwen2.5-coder:7b"
        assert response.latency_ms > 0

    def test_response_has_routing_decision(self) -> None:
        """Verify the response carries the routing decision through."""
        executor = LocalExecutor({"local_timeout": 60.0})
        routing = RoutingDecision(target="local", reason="test", confidence=0.8)

        response = executor.execute("What is 2+2?", routing)
        assert response.routing_decision.target == "local"
        assert response.routing_decision.reason == "test"


@pytest.mark.integration
class TestOllamaUnavailable:
    """Test behavior when Ollama is running on a wrong port (simulating unavailability)."""

    def test_wrong_port_raises_local_unavailable(self) -> None:
        """Connecting to a port where Ollama is NOT running should raise."""
        executor = LocalExecutor({
            "local_base_url": "http://localhost:19999/v1",
            "local_timeout": 3.0,
            "local_max_retries": 0,
        })
        routing = RoutingDecision(target="local", reason="test", confidence=0.9)

        with pytest.raises(LocalUnavailableError):
            executor.execute("test", routing)
