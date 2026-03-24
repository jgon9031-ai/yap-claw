"""Model executors — local (Ollama) and cloud (OpenAI/Anthropic) backends."""

from __future__ import annotations

import logging
import os
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from amos.exceptions import (
    CloudUnavailableError,
    LocalUnavailableError,
    ModelNotFoundError,
)
from amos.models import Response, RoutingDecision

logger = logging.getLogger("amos.executor")


class LocalExecutor:
    """Executes queries against a local Ollama instance via its OpenAI-compatible API.

    Handles:
      - Connection errors (Ollama not running) -> LocalUnavailableError
      - Model not found (404) -> ModelNotFoundError
      - Slow responses (configurable timeout, default 30s)
      - Automatic retry (1 retry on timeout)
    """

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self._base_url = config.get("local_base_url", "http://localhost:11434/v1")
        self._model = config.get("local_model", "qwen2.5-coder:7b")
        self._timeout = config.get("local_timeout", 30.0)
        self._max_retries = config.get("local_max_retries", 1)
        self._client = OpenAI(
            base_url=self._base_url,
            api_key="ollama",
            timeout=self._timeout,
        )

    def execute(self, query_text: str, routing: RoutingDecision) -> Response:
        """Send a query to the local Ollama model and return the response.

        Raises:
            LocalUnavailableError: If Ollama is not running or unreachable.
            ModelNotFoundError: If the requested model is not pulled/available.
        """
        last_error: Exception | None = None

        for attempt in range(1 + self._max_retries):
            try:
                start = time.perf_counter_ns()
                completion = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": query_text}],
                )
                latency_ms = (time.perf_counter_ns() - start) // 1_000_000
                text = completion.choices[0].message.content or ""
                return Response(
                    text=text,
                    model_used=self._model,
                    latency_ms=latency_ms,
                    routing_decision=routing,
                )

            except APITimeoutError as exc:
                # Must be caught before APIConnectionError (it's a subclass)
                last_error = exc
                if attempt < self._max_retries:
                    logger.warning(
                        "Local request timed out (attempt %d/%d), retrying...",
                        attempt + 1,
                        1 + self._max_retries,
                    )
                    continue
                logger.error(
                    "Local request timed out after %d attempts", 1 + self._max_retries
                )

            except APIConnectionError as exc:
                logger.error("Local Ollama unreachable: %s", exc)
                raise LocalUnavailableError(
                    f"Cannot connect to Ollama at {self._base_url}. "
                    "Is Ollama running? Try: ollama serve"
                ) from exc

            except APIStatusError as exc:
                if exc.status_code == 404:
                    logger.error("Model not found: %s", self._model)
                    raise ModelNotFoundError(
                        f"Model '{self._model}' not found. "
                        f"Try: ollama pull {self._model}"
                    ) from exc
                # Other status errors — no retry
                logger.error("Local API error (status %d): %s", exc.status_code, exc)
                raise LocalUnavailableError(
                    f"Local API error (HTTP {exc.status_code}): {exc}"
                ) from exc

        # All retries exhausted on timeout
        raise LocalUnavailableError(
            f"Local model timed out after {1 + self._max_retries} attempts "
            f"(timeout={self._timeout}s)"
        ) from last_error


class CloudExecutor:
    """Executes queries against a cloud LLM (OpenAI or Anthropic-compatible endpoint)."""

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self._model = config.get("cloud_model", "gpt-4o-mini")
        self._base_url = config.get("cloud_base_url", None)  # None = default OpenAI
        self._timeout = config.get("cloud_timeout", 60.0)

        # Resolve API key: explicit config > OPENAI_API_KEY > ANTHROPIC_API_KEY
        api_key = config.get("cloud_api_key")
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "No cloud API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY env var, "
                "or pass cloud_api_key in config."
            )

        kwargs: dict = {"api_key": api_key, "timeout": self._timeout}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)

    def execute(self, query_text: str, routing: RoutingDecision) -> Response:
        """Send a query to the cloud model and return the response.

        Raises:
            CloudUnavailableError: If the cloud API is unreachable.
            ModelNotFoundError: If the requested cloud model doesn't exist.
        """
        try:
            start = time.perf_counter_ns()
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": query_text}],
            )
            latency_ms = (time.perf_counter_ns() - start) // 1_000_000
            text = completion.choices[0].message.content or ""
            return Response(
                text=text,
                model_used=self._model,
                latency_ms=latency_ms,
                routing_decision=routing,
            )

        except APITimeoutError as exc:
            raise CloudUnavailableError(
                f"Cloud request timed out (timeout={self._timeout}s)"
            ) from exc

        except APIConnectionError as exc:
            raise CloudUnavailableError(
                f"Cannot connect to cloud API: {exc}"
            ) from exc

        except APIStatusError as exc:
            if exc.status_code == 404:
                raise ModelNotFoundError(
                    f"Cloud model '{self._model}' not found"
                ) from exc
            raise CloudUnavailableError(
                f"Cloud API error (HTTP {exc.status_code}): {exc}"
            ) from exc
