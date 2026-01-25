# src/easy_scaffold/common/custom_exceptions.py
from typing import Any, Dict, Optional


class OlympiadAgentException(Exception):
    """Base exception for the Olympiad Agent project."""
    pass


class ConfigException(OlympiadAgentException):
    """Exception related to configuration errors."""
    pass


class DatabaseException(OlympiadAgentException):
    """Exception related to database operations."""
    pass


class WorkflowException(OlympiadAgentException):
    """Exception related to workflow execution."""
    pass


class RateLimitException(WorkflowException):
    """Exception raised when an API rate limit is encountered."""
    pass


class ContentBlockedException(WorkflowException):
    """Raised when the API returns no content, likely due to safety filters."""
    pass


class APIServerException(WorkflowException):
    """Raised when the API returns a 500-level server error."""
    pass


class EmptyResponseException(WorkflowException):
    """Raised when the LLM returns a valid but empty response."""
    pass


class StageExecutionError(WorkflowException):
    """Raised when a stage fails so that the workflow can handle logging centrally."""

    def __init__(
        self,
        status: str,
        message: str,
        inputs: Dict[str, Any],
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.status = status
        self.inputs = inputs
        self.cause = cause

