"""MOOD Layer — keyword-rule based mood detection for YAP-CLAW."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MoodState(Enum):
    FRUSTRATED = "frustrated"
    SATISFIED = "satisfied"
    CURIOUS = "curious"
    URGENT = "urgent"
    NEUTRAL = "neutral"


@dataclass
class MoodContext:
    mood: MoodState
    confidence: float
    signals_found: list[str] = field(default_factory=list)
    directive: str = ""  # injected into agent prompt


class MoodDetector:
    """Detect user mood from text using keyword signal matching.

    Pure-Python, no ML dependencies. Scans for predefined signal phrases
    in Korean and English, returning the mood with the most matches.
    """

    SIGNALS: dict[MoodState, list[str]] = {
        MoodState.FRUSTRATED: [
            "또 실패",
            "왜 안 되",
            "배우는 게 없",
            "계속 실패",
            "왜 이렇게",
            "제대로 안",
            "안 되네",
            "또야",
            "실망",
            "짜증",
            "failed again",
            "still failing",
            "not working",
            "keeps failing",
            "again",
        ],
        MoodState.SATISFIED: [
            "응 좋아",
            "잘했어",
            "멋지다",
            "완벽",
            "훌륭",
            "잘 됐다",
            "맞아",
            "그렇지",
            "좋네",
            "잘 됐어",
            "good",
            "perfect",
            "great",
            "excellent",
            "well done",
            "nice",
        ],
        MoodState.CURIOUS: [
            "어떻게",
            "왜",
            "가능한가",
            "방법이",
            "어떤 방식",
            "원리",
            "설명해줘",
            "알려줘",
            "이유가",
            "무슨 이유",
            "how",
            "why",
            "what is",
            "explain",
            "tell me about",
            "how does",
        ],
        MoodState.URGENT: [
            "빠르게",
            "바로",
            "지금 당장",
            "빨리",
            "즉시",
            "당장",
            "urgent",
            "asap",
            "quickly",
            "right now",
            "immediately",
            "fast",
        ],
    }

    DIRECTIVES: dict[MoodState, str] = {
        MoodState.FRUSTRATED: (
            "User is frustrated. Acknowledge the issue first in one sentence. "
            "Be concise and direct. No filler phrases. Focus on solution."
        ),
        MoodState.SATISFIED: (
            "User is satisfied. Keep momentum. "
            "Naturally suggest next step if applicable."
        ),
        MoodState.CURIOUS: (
            "User is curious. Be thorough. "
            "Include background context and explanation."
        ),
        MoodState.URGENT: (
            "User needs speed. Skip preamble entirely. "
            "Lead with result. Maximum 3 sentences total."
        ),
        MoodState.NEUTRAL: "",
    }

    def detect(self, text: str) -> MoodState:
        """Return the best-matching mood for *text*, or NEUTRAL if no signals match."""
        text_lower = text.lower()
        scores: dict[MoodState, int] = {mood: 0 for mood in MoodState}
        for mood, signals in self.SIGNALS.items():
            scores[mood] = sum(1 for s in signals if s in text_lower)
        best_mood = max(scores, key=lambda m: scores[m])
        if scores[best_mood] == 0:
            return MoodState.NEUTRAL
        return best_mood

    def detect_with_context(self, text: str) -> MoodContext:
        """Detect mood and return full context including matched signals and directive."""
        text_lower = text.lower()
        scores: dict[MoodState, list[str]] = {mood: [] for mood in MoodState}
        for mood, signals in self.SIGNALS.items():
            scores[mood] = [s for s in signals if s in text_lower]

        best_mood = max(scores, key=lambda m: len(scores[m]))
        found = scores[best_mood]

        if not found:
            best_mood = MoodState.NEUTRAL

        total_signals = sum(len(v) for v in scores.values()) or 1
        confidence = round(len(found) / total_signals, 2) if found else 1.0

        return MoodContext(
            mood=best_mood,
            confidence=confidence,
            signals_found=found,
            directive=self.DIRECTIVES[best_mood],
        )
