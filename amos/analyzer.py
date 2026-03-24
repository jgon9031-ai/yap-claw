"""Hermes-style session analyzer for AMOS.

Inspired by NousResearch Hermes Agent pattern: periodically reviews recent
interactions to extract routing patterns and user preferences, then feeds
routing hints back into the router for improved decisions.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Literal

from amos.memory import MemoryLayer
from amos.models import AnalysisResult, Experience

logger = logging.getLogger("amos.analyzer")

# Simple keyword-based category detection
_CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("code", re.compile(r"(code|function|class|debug|refactor|implement|bug)", re.I)),
    ("math", re.compile(r"(math|calcul|equation|formula|algebra|statistic)", re.I)),
    ("writing", re.compile(r"(writ|essay|summar|article|draft|blog|report)", re.I)),
    ("translation", re.compile(r"(translat|번역|翻訳|convert.*language)", re.I)),
    ("analysis", re.compile(r"(analy[sz]|compare|evaluat|assess|분석|비교)", re.I)),
    ("creative", re.compile(r"(creat|story|poem|imagin|design|brainstorm)", re.I)),
]


def categorize_query(text: str) -> str:
    """Assign a simple category to a query based on keyword matching."""
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(text):
            return category
    return "general"


class SessionAnalyzer:
    """Periodically analyzes recent interactions to extract routing patterns.

    After every `analyze_every` new interactions, reviews the batch to
    determine which query categories perform better on local vs. cloud,
    then caches routing hints for the router to consume.
    """

    def __init__(self, memory: MemoryLayer, analyze_every: int = 10) -> None:
        self._memory = memory
        self._analyze_every = analyze_every
        self._last_analyzed_id: int = memory.max_id()
        self._cached_hints: dict[str, Literal["local", "cloud"]] = {}
        self._last_result: AnalysisResult | None = None

    def should_analyze(self) -> bool:
        """Return True if enough new records have accumulated since last analysis."""
        return self._memory.count_since(self._last_analyzed_id) >= self._analyze_every

    def analyze(self) -> AnalysisResult:
        """Review recent experiences and extract routing patterns.

        For each query category, computes success rate and average latency
        per target (local/cloud). If one target is clearly better — higher
        success rate or significantly lower latency at equal success — it
        becomes the preferred routing hint for that category.
        """
        experiences = self._memory.get_recent(limit=self._analyze_every * 5)
        if not experiences:
            return AnalysisResult(sample_size=0)

        # Group by category -> target -> outcomes
        stats: dict[str, dict[str, list[Experience]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for exp in experiences:
            cat = categorize_query(exp.query_text)
            stats[cat][exp.target].append(exp)

        hints: dict[str, Literal["local", "cloud"]] = {}

        for category, per_target in stats.items():
            local_exps = per_target.get("local", [])
            cloud_exps = per_target.get("cloud", [])

            local_rate = _success_rate(local_exps)
            cloud_rate = _success_rate(cloud_exps)
            local_latency = _avg_latency(local_exps)
            cloud_latency = _avg_latency(cloud_exps)

            # Only generate hint when we have data for at least one target
            if not local_exps and not cloud_exps:
                continue

            # Cloud clearly better on success rate (>20% difference)
            if cloud_rate - local_rate > 0.2 and len(cloud_exps) >= 2:
                hints[category] = "cloud"
            # Local clearly better on success rate
            elif local_rate - cloud_rate > 0.2 and len(local_exps) >= 2:
                hints[category] = "local"
            # Equal success: prefer lower latency (>30% faster)
            elif local_latency > 0 and cloud_latency > 0:
                if local_latency < cloud_latency * 0.7:
                    hints[category] = "local"
                elif cloud_latency < local_latency * 0.7:
                    hints[category] = "cloud"

        # Update state
        self._last_analyzed_id = self._memory.max_id()
        self._cached_hints = hints
        result = AnalysisResult(
            routing_hints=hints,
            analyzed_at=datetime.now(timezone.utc),
            sample_size=len(experiences),
        )
        self._last_result = result
        logger.info(
            "Session analysis complete: %d experiences, %d hints extracted",
            len(experiences),
            len(hints),
        )
        return result

    def get_routing_hints(self) -> dict[str, Literal["local", "cloud"]]:
        """Return current cached routing hints for the router."""
        return dict(self._cached_hints)

    @property
    def last_result(self) -> AnalysisResult | None:
        """Return the most recent analysis result, if any."""
        return self._last_result


def _success_rate(experiences: list[Experience]) -> float:
    """Compute success rate for a list of experiences."""
    if not experiences:
        return 0.0
    return sum(1 for e in experiences if e.success) / len(experiences)


def _avg_latency(experiences: list[Experience]) -> float:
    """Compute average latency for a list of experiences."""
    if not experiences:
        return 0.0
    return sum(e.latency_ms for e in experiences) / len(experiences)
