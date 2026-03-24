"""Custom exceptions for AMOS error handling."""


class AMOSError(Exception):
    """Base exception for all AMOS errors."""


class LocalUnavailableError(AMOSError):
    """Raised when the local Ollama instance is not reachable."""


class ModelNotFoundError(AMOSError):
    """Raised when the requested model is not available on the target."""


class CloudUnavailableError(AMOSError):
    """Raised when the cloud API endpoint is not reachable."""


class AllModelsFailedError(AMOSError):
    """Raised when both local and cloud execution have failed."""
