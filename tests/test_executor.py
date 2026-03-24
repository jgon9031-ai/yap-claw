"""Unit tests for AMOS executors — mocked, no real Ollama/cloud needed."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from amos.exceptions import (
    CloudUnavailableError,
    LocalUnavailableError,
    ModelNotFoundError,
)
from amos.executor import CloudExecutor, LocalExecutor
from amos.models import RoutingDecision


@pytest.fixture()
def routing() -> RoutingDecision:
    return RoutingDecision(target="local", reason="test", confidence=0.9)


@pytest.fixture()
def cloud_routing() -> RoutingDecision:
    return RoutingDecision(target="cloud", reason="test", confidence=0.9)


# ---------------------------------------------------------------------------
# Helper to build a fake OpenAI completion response
# ---------------------------------------------------------------------------

def _fake_completion(text: str = "Hello!") -> MagicMock:
    choice = MagicMock()
    choice.message.content = text
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _make_api_status_error(status_code: int) -> APIStatusError:
    """Create an APIStatusError with the given status code."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {}
    return APIStatusError(
        message=f"HTTP {status_code}",
        response=resp,
        body=None,
    )


def _make_api_connection_error() -> APIConnectionError:
    return APIConnectionError(request=MagicMock())


def _make_api_timeout_error() -> APITimeoutError:
    return APITimeoutError(request=MagicMock())


# ===========================================================================
# LocalExecutor tests
# ===========================================================================


class TestLocalExecutorSuccess:
    """Test successful local execution."""

    @patch("amos.executor.OpenAI")
    def test_execute_returns_response(self, mock_openai_cls: MagicMock, routing: RoutingDecision) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_completion("print('hello')")
        mock_openai_cls.return_value = mock_client

        executor = LocalExecutor({"local_model": "test-model"})
        resp = executor.execute("say hello", routing)

        assert resp.text == "print('hello')"
        assert resp.model_used == "test-model"
        assert resp.latency_ms >= 0
        assert resp.routing_decision == routing

    @patch("amos.executor.OpenAI")
    def test_execute_with_empty_response(self, mock_openai_cls: MagicMock, routing: RoutingDecision) -> None:
        mock_client = MagicMock()
        choice = MagicMock()
        choice.message.content = None  # None content
        completion = MagicMock()
        completion.choices = [choice]
        mock_client.chat.completions.create.return_value = completion
        mock_openai_cls.return_value = mock_client

        executor = LocalExecutor()
        resp = executor.execute("test", routing)
        assert resp.text == ""


class TestLocalExecutorConnectionError:
    """Test local executor when Ollama is not running."""

    @patch("amos.executor.OpenAI")
    def test_connection_error_raises_local_unavailable(
        self, mock_openai_cls: MagicMock, routing: RoutingDecision
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _make_api_connection_error()
        mock_openai_cls.return_value = mock_client

        executor = LocalExecutor()
        with pytest.raises(LocalUnavailableError, match="Cannot connect to Ollama"):
            executor.execute("test query", routing)


class TestLocalExecutorModelNotFound:
    """Test local executor when model is not pulled."""

    @patch("amos.executor.OpenAI")
    def test_404_raises_model_not_found(
        self, mock_openai_cls: MagicMock, routing: RoutingDecision
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _make_api_status_error(404)
        mock_openai_cls.return_value = mock_client

        executor = LocalExecutor({"local_model": "nonexistent:latest"})
        with pytest.raises(ModelNotFoundError, match="nonexistent:latest"):
            executor.execute("test", routing)


class TestLocalExecutorOtherStatusError:
    """Test local executor with non-404 HTTP errors."""

    @patch("amos.executor.OpenAI")
    def test_500_raises_local_unavailable(
        self, mock_openai_cls: MagicMock, routing: RoutingDecision
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _make_api_status_error(500)
        mock_openai_cls.return_value = mock_client

        executor = LocalExecutor()
        with pytest.raises(LocalUnavailableError, match="HTTP 500"):
            executor.execute("test", routing)


class TestLocalExecutorTimeout:
    """Test timeout handling with retry."""

    @patch("amos.executor.OpenAI")
    def test_timeout_retries_once_then_raises(
        self, mock_openai_cls: MagicMock, routing: RoutingDecision
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _make_api_timeout_error()
        mock_openai_cls.return_value = mock_client

        executor = LocalExecutor({"local_timeout": 5.0, "local_max_retries": 1})
        with pytest.raises(LocalUnavailableError, match="timed out"):
            executor.execute("test", routing)

        # Should have been called twice (initial + 1 retry)
        assert mock_client.chat.completions.create.call_count == 2

    @patch("amos.executor.OpenAI")
    def test_timeout_then_success_on_retry(
        self, mock_openai_cls: MagicMock, routing: RoutingDecision
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_api_timeout_error(),
            _fake_completion("retried ok"),
        ]
        mock_openai_cls.return_value = mock_client

        executor = LocalExecutor({"local_max_retries": 1})
        resp = executor.execute("test", routing)
        assert resp.text == "retried ok"
        assert mock_client.chat.completions.create.call_count == 2

    @patch("amos.executor.OpenAI")
    def test_no_retry_when_max_retries_zero(
        self, mock_openai_cls: MagicMock, routing: RoutingDecision
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _make_api_timeout_error()
        mock_openai_cls.return_value = mock_client

        executor = LocalExecutor({"local_max_retries": 0})
        with pytest.raises(LocalUnavailableError, match="timed out"):
            executor.execute("test", routing)

        assert mock_client.chat.completions.create.call_count == 1


class TestLocalExecutorConfig:
    """Test configuration options."""

    @patch("amos.executor.OpenAI")
    def test_default_config(self, mock_openai_cls: MagicMock) -> None:
        executor = LocalExecutor()
        assert executor._model == "qwen2.5-coder:7b"
        assert executor._base_url == "http://localhost:11434/v1"
        assert executor._timeout == 30.0
        assert executor._max_retries == 1

    @patch("amos.executor.OpenAI")
    def test_custom_config(self, mock_openai_cls: MagicMock) -> None:
        executor = LocalExecutor({
            "local_model": "llama3:8b",
            "local_base_url": "http://remote:11434/v1",
            "local_timeout": 60.0,
            "local_max_retries": 3,
        })
        assert executor._model == "llama3:8b"
        assert executor._base_url == "http://remote:11434/v1"
        assert executor._timeout == 60.0
        assert executor._max_retries == 3


# ===========================================================================
# CloudExecutor tests
# ===========================================================================


class TestCloudExecutorSuccess:
    """Test successful cloud execution."""

    @patch("amos.executor.OpenAI")
    def test_execute_returns_response(
        self, mock_openai_cls: MagicMock, cloud_routing: RoutingDecision
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_completion("cloud response")
        mock_openai_cls.return_value = mock_client

        executor = CloudExecutor({"cloud_api_key": "test-key"})
        resp = executor.execute("complex question", cloud_routing)

        assert resp.text == "cloud response"
        assert resp.model_used == "gpt-4o-mini"
        assert resp.routing_decision == cloud_routing


class TestCloudExecutorErrors:
    """Test cloud executor error handling."""

    @patch("amos.executor.OpenAI")
    def test_connection_error_raises_cloud_unavailable(
        self, mock_openai_cls: MagicMock, cloud_routing: RoutingDecision
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _make_api_connection_error()
        mock_openai_cls.return_value = mock_client

        executor = CloudExecutor({"cloud_api_key": "test-key"})
        with pytest.raises(CloudUnavailableError, match="Cannot connect"):
            executor.execute("test", cloud_routing)

    @patch("amos.executor.OpenAI")
    def test_404_raises_model_not_found(
        self, mock_openai_cls: MagicMock, cloud_routing: RoutingDecision
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _make_api_status_error(404)
        mock_openai_cls.return_value = mock_client

        executor = CloudExecutor({"cloud_api_key": "key", "cloud_model": "gpt-5-turbo"})
        with pytest.raises(ModelNotFoundError, match="gpt-5-turbo"):
            executor.execute("test", cloud_routing)

    @patch("amos.executor.OpenAI")
    def test_500_raises_cloud_unavailable(
        self, mock_openai_cls: MagicMock, cloud_routing: RoutingDecision
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _make_api_status_error(500)
        mock_openai_cls.return_value = mock_client

        executor = CloudExecutor({"cloud_api_key": "key"})
        with pytest.raises(CloudUnavailableError, match="HTTP 500"):
            executor.execute("test", cloud_routing)

    @patch("amos.executor.OpenAI")
    def test_timeout_raises_cloud_unavailable(
        self, mock_openai_cls: MagicMock, cloud_routing: RoutingDecision
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _make_api_timeout_error()
        mock_openai_cls.return_value = mock_client

        executor = CloudExecutor({"cloud_api_key": "key"})
        with pytest.raises(CloudUnavailableError, match="timed out"):
            executor.execute("test", cloud_routing)


class TestCloudExecutorApiKey:
    """Test API key resolution."""

    def test_no_api_key_raises_value_error(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="No cloud API key"):
                CloudExecutor({})

    @patch("amos.executor.OpenAI")
    def test_explicit_config_key(self, mock_openai_cls: MagicMock) -> None:
        CloudExecutor({"cloud_api_key": "explicit-key"})
        mock_openai_cls.assert_called_once()

    @patch("amos.executor.OpenAI")
    def test_env_openai_key(self, mock_openai_cls: MagicMock) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            CloudExecutor({})
        mock_openai_cls.assert_called_once()
