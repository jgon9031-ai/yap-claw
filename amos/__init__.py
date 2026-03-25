"""AMOS — Autonomous Meta-Orchestration System.

A hybrid AI harness that routes queries between local (Ollama) and cloud
(Claude/GPT) models, records outcomes in a memory layer, and improves
routing over time.

Inspired by: "Why AI systems do not learn and what to do about it" (arXiv 2603.15381)
"""

from amos.models import (
    AnalysisResult,
    Experience,
    HealthIssue,
    Query,
    Response,
    RoutingAmendment,
    RoutingDecision,
)
from amos.harness import AMOS
from amos.analyzer import SessionAnalyzer
from amos.dashboard import RoutingDashboard
from amos.exceptions import (
    AMOSError,
    AllModelsFailedError,
    CloudUnavailableError,
    LocalUnavailableError,
    ModelNotFoundError,
)
from amos.health import SkillHealthTracker
from amos.mood import MoodContext, MoodDetector, MoodState

__all__ = [
    "AMOS",
    "AMOSError",
    "AllModelsFailedError",
    "AnalysisResult",
    "CloudUnavailableError",
    "Experience",
    "HealthIssue",
    "LocalUnavailableError",
    "ModelNotFoundError",
    "MoodContext",
    "MoodDetector",
    "MoodState",
    "Query",
    "Response",
    "RoutingAmendment",
    "RoutingDashboard",
    "RoutingDecision",
    "SessionAnalyzer",
    "SkillHealthTracker",
]
__version__ = "0.3.0"
