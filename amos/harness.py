"""Main AMOS harness — orchestrates routing, execution, and memory."""

from __future__ import annotations

import logging
import threading

from amos.analyzer import SessionAnalyzer
from amos.exceptions import LocalUnavailableError
from amos.executor import CloudExecutor, LocalExecutor
from amos.health import SkillHealthTracker
from amos.memory import MemoryLayer
from amos.models import Experience, Query, Response
from amos.router import AMOSRouter

logger = logging.getLogger("amos")


class AMOS:
    """Autonomous Meta-Orchestration System.

    Orchestrates the full pipeline: route -> execute -> record -> return.
    Learns from past outcomes to improve future routing decisions.

    Includes two self-improving subsystems:
      - SessionAnalyzer (Hermes-inspired): periodic pattern extraction
      - SkillHealthTracker (cognee-inspired): failure detection + amendment
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._memory = MemoryLayer(
            db_path=self._config.get("memory_db", "~/.amos/memory.db")
        )
        self._router = AMOSRouter(self._memory, self._config)
        self._local = LocalExecutor(self._config)

        # Cloud executor is lazy-initialized — only created when first needed
        self._cloud: CloudExecutor | None = None

        # Self-improving subsystems
        self._analyzer = SessionAnalyzer(
            self._memory,
            analyze_every=self._config.get("analyze_every", 10),
        )
        self._health = SkillHealthTracker(
            self._memory,
            failure_threshold=self._config.get("failure_threshold", 0.4),
        )

    @property
    def analyzer(self) -> SessionAnalyzer:
        """Access the session analyzer for inspection."""
        return self._analyzer

    @property
    def health(self) -> SkillHealthTracker:
        """Access the health tracker for inspection."""
        return self._health

    def run(self, query_text: str, metadata: dict | None = None) -> Response:
        """Route a query, execute it, record the outcome, and return the response.

        If the local executor raises LocalUnavailableError (infrastructure issue),
        automatically falls back to cloud without recording a routing failure.

        After recording, triggers the self-improving subsystems:
          1. SkillHealthTracker.observe() — track failure patterns
          2. SessionAnalyzer.should_analyze() — if yes, run analysis in background

        Args:
            query_text: The user's natural language query.
            metadata: Optional metadata to attach to the query.

        Returns:
            The model's response with routing information.
        """
        query = Query(text=query_text, metadata=metadata or {})
        decision = self._router.route(query)

        logger.info(
            "Routing to %s (reason: %s, confidence: %.2f)",
            decision.target,
            decision.reason,
            decision.confidence,
        )

        try:
            if decision.target == "local":
                try:
                    response = self._local.execute(query_text, decision)
                except LocalUnavailableError:
                    # Infrastructure issue — fallback to cloud silently
                    logger.warning(
                        "Local unavailable, falling back to cloud "
                        "(not recorded as routing failure)"
                    )
                    response = self._get_cloud().execute(query_text, decision)
            else:
                response = self._get_cloud().execute(query_text, decision)
            success = True
        except Exception as exc:
            logger.error("Execution failed on %s: %s", decision.target, exc)
            # Record the failure, feed to health tracker, then re-raise
            self._memory.record(
                query_text=query_text,
                target=decision.target,
                model_used="error",
                success=False,
                latency_ms=0,
            )
            self._health.observe(
                Experience(
                    query_text=query_text,
                    target=decision.target,
                    model_used="error",
                    success=False,
                    latency_ms=0,
                )
            )
            raise

        # Record successful experience
        self._memory.record(
            query_text=query_text,
            target=decision.target,
            model_used=response.model_used,
            success=success,
            latency_ms=response.latency_ms,
        )

        # Feed to health tracker
        experience = Experience(
            query_text=query_text,
            target=decision.target,
            model_used=response.model_used,
            success=success,
            latency_ms=response.latency_ms,
        )
        self._health.observe(experience)

        # Trigger background analysis if enough new records have accumulated
        if self._analyzer.should_analyze():
            self._run_analysis_background()

        return response

    def feedback(self, response: Response, success: bool) -> None:
        """Allow the user to mark a past response as good or bad.

        This retroactive feedback helps the memory layer learn which
        routing decisions lead to satisfactory results.
        """
        self._memory.record(
            query_text="[feedback]",
            target=response.routing_decision.target,
            model_used=response.model_used,
            success=success,
            latency_ms=response.latency_ms,
        )
        logger.info(
            "Feedback recorded: %s on %s -> %s",
            response.model_used,
            response.routing_decision.target,
            "success" if success else "failure",
        )

    def stats(self) -> dict:
        """Return memory layer statistics."""
        return self._memory.stats()

    def dashboard(self) -> "RoutingDashboard":
        """Return a RoutingDashboard instance backed by this harness's memory."""
        from amos.dashboard import RoutingDashboard

        return RoutingDashboard(self._memory)

    def close(self) -> None:
        """Close underlying resources."""
        self._memory.close()

    def _get_cloud(self) -> CloudExecutor:
        """Lazy-initialize the cloud executor."""
        if self._cloud is None:
            self._cloud = CloudExecutor(self._config)
        return self._cloud

    def _run_analysis_background(self) -> None:
        """Run session analysis in a background thread.

        After analysis completes, merges routing hints from the analyzer
        and active health amendments into the router.
        """

        def _analyze() -> None:
            try:
                self._analyzer.analyze()
                self._apply_hints()
            except Exception:
                logger.exception("Background session analysis failed")

        thread = threading.Thread(target=_analyze, daemon=True)
        thread.start()

    def _apply_hints(self) -> None:
        """Merge analyzer hints and health amendments into the router."""
        hints = self._analyzer.get_routing_hints()

        # Health amendments override analyzer hints (more specific signal)
        for category, amendment in self._health.get_active_amendments().items():
            hints[category] = amendment.new_target

        self._router.set_routing_hints(hints)
