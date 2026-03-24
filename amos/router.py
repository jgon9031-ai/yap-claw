"""System M — AMOS routing logic.

Decides whether a query should be handled by a local (Ollama) model
or escalated to a cloud (Claude/GPT) model based on:
  1. Privacy constraints (personal data stays local)
  2. Past failure patterns (learned from memory)
  3. Complexity heuristics (hard queries go to cloud)
"""

from __future__ import annotations

from amos.memory import MemoryLayer
from amos.models import Query, RoutingDecision


# Keywords indicating personal/private content — must stay local
PERSONAL_KEYWORDS: list[str] = [
    "나의", "내", "my", "personal", "private",
    "체중", "식단", "건강", "비밀번호",
]

# Keywords indicating complex reasoning — prefer cloud
COMPLEX_KEYWORDS: list[str] = [
    "analyze", "compare", "explain", "research", "summarize",
    "분석", "비교", "설명", "조사", "요약",
]

# Threshold above which local failure rate triggers cloud routing
FAILURE_RATE_THRESHOLD = 0.6

# Query length (chars) above which we consider it complex
LENGTH_THRESHOLD = 200

# Number of complex keywords that tips the scale
COMPLEX_KEYWORD_THRESHOLD = 2


class AMOSRouter:
    """System M: routes queries to local or cloud models."""

    def __init__(self, memory: MemoryLayer, config: dict | None = None) -> None:
        self._memory = memory
        self._config = config or {}
        self._routing_hints: dict[str, str] = {}

    def set_routing_hints(self, hints: dict[str, str]) -> None:
        """Accept routing hints from SessionAnalyzer or SkillHealthTracker.

        Hints map query category -> preferred target ("local" or "cloud").
        """
        self._routing_hints = dict(hints)

    def route(self, query: Query) -> RoutingDecision:
        """Determine the best target for a query.

        Decision cascade:
          1. Privacy check — personal keywords force local
          2. Routing hints — analyzer/health-based overrides
          3. Past failure check — high local failure rate pushes to cloud
          4. Complexity check — complex queries go to cloud
          5. Default — prefer local
        """
        text_lower = query.text.lower()

        # 1. Privacy: personal data stays on-device
        if self._contains_personal(text_lower):
            return RoutingDecision(
                target="local",
                reason="Query contains personal/private content — keeping local",
                confidence=0.95,
            )

        # 2. Routing hints from SessionAnalyzer / SkillHealthTracker
        if self._routing_hints:
            from amos.analyzer import categorize_query

            category = categorize_query(query.text)
            if category in self._routing_hints:
                hint_target = self._routing_hints[category]
                return RoutingDecision(
                    target=hint_target,
                    reason=f"Routing hint for '{category}' queries — preferring {hint_target}",
                    confidence=0.80,
                )

        # 3. Past failure: if local has been failing on similar queries, escalate
        failure_rate = self._memory.get_local_failure_rate(query.text)
        if failure_rate > FAILURE_RATE_THRESHOLD:
            return RoutingDecision(
                target="cloud",
                reason=f"Local failure rate {failure_rate:.0%} for similar queries — escalating to cloud",
                confidence=0.85,
            )

        # 4. Complexity: score the query and route accordingly
        score = self._complexity_score(text_lower)
        if score >= 0.6:
            return RoutingDecision(
                target="cloud",
                reason=f"High complexity score ({score:.2f}) — routing to cloud",
                confidence=0.7 + score * 0.2,
            )

        # 5. Default: prefer local for cost and latency
        return RoutingDecision(
            target="local",
            reason="Default routing — local preferred for speed and cost",
            confidence=0.6,
        )

    @staticmethod
    def _contains_personal(text_lower: str) -> bool:
        """Check if the text contains personal/private keywords."""
        return any(kw in text_lower for kw in PERSONAL_KEYWORDS)

    @staticmethod
    def _complexity_score(text_lower: str) -> float:
        """Compute a 0-1 complexity score from simple heuristics.

        Factors:
          - Query length (longer = more complex)
          - Presence of complex keywords
          - Question depth (number of question marks)
        """
        score = 0.0

        # Length factor: 0.0 at 0 chars, 0.3 at LENGTH_THRESHOLD+
        length_factor = min(len(text_lower) / LENGTH_THRESHOLD, 1.0) * 0.3
        score += length_factor

        # Keyword factor: each complex keyword adds 0.15, max 0.6
        keyword_hits = sum(1 for kw in COMPLEX_KEYWORDS if kw in text_lower)
        keyword_factor = min(keyword_hits * 0.15, 0.6)
        score += keyword_factor

        # Question depth: multiple questions suggest compound reasoning
        question_marks = text_lower.count("?")
        question_factor = min(question_marks * 0.1, 0.25)
        score += question_factor

        return min(score, 1.0)
