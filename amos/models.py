"""Pydantic models for AMOS data structures."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from amos.mood import MoodContext


class Query(BaseModel):
    """An incoming query to be routed and executed."""

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RoutingDecision(BaseModel):
    """The result of System M's routing logic."""

    target: Literal["local", "cloud"]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class Experience(BaseModel):
    """A recorded experience from a past query execution."""

    query_text: str
    target: Literal["local", "cloud"]
    model_used: str
    success: bool
    latency_ms: int
    mood: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Response(BaseModel):
    """The final response returned to the caller."""

    text: str
    model_used: str
    latency_ms: int
    routing_decision: RoutingDecision
    mood_context: MoodContext | None = None


# --- Self-Improving Loop models ---


class AnalysisResult(BaseModel):
    """Output of SessionAnalyzer: extracted routing patterns from recent experiences."""

    routing_hints: dict[str, Literal["local", "cloud"]] = Field(
        default_factory=dict,
        description="Mapping of detected pattern -> preferred routing target",
    )
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sample_size: int = 0


class HealthIssue(BaseModel):
    """A category with an abnormally high failure rate detected by SkillHealthTracker."""

    category: str
    failure_rate: float = Field(ge=0.0, le=1.0)
    total_queries: int
    failed_queries: int
    current_target: Literal["local", "cloud"]
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RoutingAmendment(BaseModel):
    """A suggested routing change to fix a HealthIssue."""

    category: str
    old_target: Literal["local", "cloud"]
    new_target: Literal["local", "cloud"]
    reason: str
    applied: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
