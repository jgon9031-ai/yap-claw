"""cognee-style skill health tracker for AMOS.

Inspired by cognee-skills observe-inspect-amend-evaluate loop: tracks per-
category failure patterns, flags issues when failure rate exceeds a threshold,
suggests routing amendments, and evaluates whether amendments actually improve
outcomes.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Literal

from amos.memory import MemoryLayer
from amos.models import Experience, HealthIssue, RoutingAmendment

logger = logging.getLogger("amos.health")


class SkillHealthTracker:
    """Tracks failure patterns by query category and suggests routing fixes.

    Implements the cognee observe -> inspect -> amend -> evaluate loop:
      - observe(): record an experience outcome by category
      - inspect(): find categories with failure rate above threshold
      - amend(): suggest a routing change for a health issue
      - evaluate(): check if an amendment improved success rate
    """

    def __init__(
        self,
        memory: MemoryLayer,
        failure_threshold: float = 0.4,
    ) -> None:
        self._memory = memory
        self._failure_threshold = failure_threshold

        # In-memory per-category counters: category -> {target -> {success, failure}}
        self._counters: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"success": 0, "failure": 0})
        )

        # Active amendments: category -> RoutingAmendment
        self._amendments: dict[str, RoutingAmendment] = {}

    def observe(self, experience: Experience) -> None:
        """Record an experience outcome, incrementing per-category counters.

        Uses the same categorization as SessionAnalyzer for consistency.
        """
        from amos.analyzer import categorize_query

        category = categorize_query(experience.query_text)
        key = "success" if experience.success else "failure"
        self._counters[category][experience.target][key] += 1

    def inspect(self) -> list[HealthIssue]:
        """Find categories where failure rate exceeds the threshold.

        Returns a list of HealthIssue objects describing each problematic
        category and its current routing target.
        """
        issues: list[HealthIssue] = []

        for category, per_target in self._counters.items():
            for target, counts in per_target.items():
                total = counts["success"] + counts["failure"]
                if total < 3:  # Need minimum sample
                    continue
                failure_rate = counts["failure"] / total
                if failure_rate > self._failure_threshold:
                    issues.append(
                        HealthIssue(
                            category=category,
                            failure_rate=failure_rate,
                            total_queries=total,
                            failed_queries=counts["failure"],
                            current_target=target,
                        )
                    )
                    logger.warning(
                        "Health issue: category=%s target=%s failure_rate=%.0f%% (%d/%d)",
                        category,
                        target,
                        failure_rate * 100,
                        counts["failure"],
                        total,
                    )

        return issues

    def amend(self, issue: HealthIssue) -> RoutingAmendment:
        """Suggest a routing change to fix a health issue.

        Simple strategy: if the current target has a high failure rate,
        suggest routing to the other target instead.
        """
        new_target: Literal["local", "cloud"] = (
            "cloud" if issue.current_target == "local" else "local"
        )
        amendment = RoutingAmendment(
            category=issue.category,
            old_target=issue.current_target,
            new_target=new_target,
            reason=(
                f"Failure rate {issue.failure_rate:.0%} on {issue.current_target} "
                f"for '{issue.category}' queries — switching to {new_target}"
            ),
        )
        self._amendments[issue.category] = amendment
        logger.info(
            "Amendment suggested: route '%s' from %s -> %s",
            issue.category,
            issue.current_target,
            new_target,
        )
        return amendment

    def evaluate(
        self,
        amendment: RoutingAmendment,
        new_experiences: list[Experience],
    ) -> bool:
        """Check if an amendment improved success rate.

        Compares the success rate of new experiences on the amended target
        against the original failure rate. If improved, keeps the amendment;
        if not, marks it for rollback.

        Returns:
            True if the amendment improved outcomes, False to rollback.
        """
        from amos.analyzer import categorize_query

        relevant = [
            e
            for e in new_experiences
            if categorize_query(e.query_text) == amendment.category
            and e.target == amendment.new_target
        ]

        if not relevant:
            logger.info(
                "No new experiences for '%s' on %s — cannot evaluate yet",
                amendment.category,
                amendment.new_target,
            )
            return True  # Keep amendment until we have data

        new_success_rate = sum(1 for e in relevant if e.success) / len(relevant)
        old_success_rate = 1.0 - self._get_failure_rate(
            amendment.category, amendment.old_target
        )

        if new_success_rate > old_success_rate:
            amendment.applied = True
            logger.info(
                "Amendment for '%s' confirmed: %.0f%% -> %.0f%% success rate",
                amendment.category,
                old_success_rate * 100,
                new_success_rate * 100,
            )
            return True

        # Rollback
        self._amendments.pop(amendment.category, None)
        logger.info(
            "Amendment for '%s' rolled back: %.0f%% -> %.0f%% (no improvement)",
            amendment.category,
            old_success_rate * 100,
            new_success_rate * 100,
        )
        return False

    def get_active_amendments(self) -> dict[str, RoutingAmendment]:
        """Return all active (not rolled back) amendments."""
        return dict(self._amendments)

    def _get_failure_rate(self, category: str, target: str) -> float:
        """Get failure rate for a category/target from internal counters."""
        counts = self._counters.get(category, {}).get(target)
        if not counts:
            return 0.0
        total = counts["success"] + counts["failure"]
        if total == 0:
            return 0.0
        return counts["failure"] / total
