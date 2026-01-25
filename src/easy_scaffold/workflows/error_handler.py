# src/easy_scaffold/workflows/error_handler.py
import logging
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from easy_scaffold.common.custom_exceptions import (
    APIServerException,
    ContentBlockedException,
    EmptyResponseException,
    RateLimitException,
    StageExecutionError,
)

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of error types by recoverability."""
    TRANSIENT = "transient"  # Should retry with backoff
    SEMI_TRANSIENT = "semi_transient"  # Retry a few times, then fallback/fail
    PERMANENT = "permanent"  # Fail immediately


class StageCriticality(Enum):
    """Classification of stage criticality."""
    CRITICAL = "critical"  # Workflow cannot proceed without this stage
    IMPORTANT = "important"  # Workflow can degrade but should try fallback
    OPTIONAL = "optional"  # Workflow can skip this stage


class WorkflowErrorHandler:
    """
    Centralized error handling for workflows.
    
    Classifies errors, determines retry strategies, and provides fallbacks
    for non-critical stages.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize error handler with configuration.
        
        Args:
            config: Error handling configuration dict with:
                - retry_config: Retry settings for different error types
                - stage_criticality: Mapping of stage names to criticality levels
                - fallbacks: Fallback values for important stages
        """
        self._config = config or {}
        
        # Default retry configuration
        retry_config = self._config.get("retry_config", {})
        transient_config = retry_config.get("transient_errors", {})
        semi_transient_config = retry_config.get("semi_transient_errors", {})
        
        self._transient_max_retries = transient_config.get("max_retries", 3)
        self._transient_backoff_factor = transient_config.get("backoff_factor", 2.0)
        self._transient_initial_delay = transient_config.get("initial_delay", 1.0)
        
        self._semi_transient_max_retries = semi_transient_config.get("max_retries", 2)
        self._semi_transient_backoff_factor = semi_transient_config.get("backoff_factor", 1.5)
        self._semi_transient_initial_delay = semi_transient_config.get("initial_delay", 0.5)
        
        # Default stage criticality
        stage_criticality = self._config.get("stage_criticality", {})
        self._critical_stages = set(stage_criticality.get("critical", [
            "Generator", "GeneratorContinuation", "SolutionAnalyzer"
        ]))
        self._important_stages = set(stage_criticality.get("important", [
            "Verifier", "ProgressAssessor", "HintGenerator"
        ]))
        self._optional_stages = set(stage_criticality.get("optional", [
            "Translator", "CoTShortener", "SolutionDetector"
        ]))
        
        # Default fallbacks
        self._fallbacks_enabled = self._config.get("fallbacks", {}).get("enabled", True)
        fallbacks = self._config.get("fallbacks", {})
        self._verifier_fallback_correctness = fallbacks.get("verifier_fallback_correctness", False)
        self._progress_assessor_fallback_pointer = fallbacks.get("progress_assessor_fallback_pointer", None)
        self._hint_generator_fallback_hint = fallbacks.get(
            "hint_generator_fallback_hint",
            "Continue with the correct approach based on the reference solution."
        )
    
    def classify_error(self, stage_error: StageExecutionError) -> ErrorType:
        """
        Classify error type based on the underlying exception.
        
        Args:
            stage_error: The StageExecutionError to classify
            
        Returns:
            ErrorType enum value
        """
        # Check the status field first (set by configurable_stage.py)
        status = stage_error.status
        
        if status in ["rate_limited", "api_error"]:
            return ErrorType.TRANSIENT
        elif status in ["empty_response"]:
            return ErrorType.SEMI_TRANSIENT
        elif status in ["blocked", "failed"]:
            return ErrorType.PERMANENT
        
        # Fallback: check the underlying cause
        cause = stage_error.cause
        if isinstance(cause, (RateLimitException, APIServerException)):
            return ErrorType.TRANSIENT
        elif isinstance(cause, EmptyResponseException):
            return ErrorType.SEMI_TRANSIENT
        elif isinstance(cause, ContentBlockedException):
            return ErrorType.PERMANENT
        
        # Default: treat as permanent
        return ErrorType.PERMANENT
    
    def get_stage_criticality(self, stage_name: str) -> StageCriticality:
        """
        Get the criticality level of a stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            StageCriticality enum value
        """
        if stage_name in self._critical_stages:
            return StageCriticality.CRITICAL
        elif stage_name in self._important_stages:
            return StageCriticality.IMPORTANT
        elif stage_name in self._optional_stages:
            return StageCriticality.OPTIONAL
        else:
            # Default: treat unknown stages as important (conservative)
            logger.warning(f"Unknown stage '{stage_name}', treating as IMPORTANT")
            return StageCriticality.IMPORTANT
    
    def should_retry(
        self,
        error_type: ErrorType,
        retry_count: int,
        stage_name: str
    ) -> Tuple[bool, float]:
        """
        Determine if an error should be retried and what delay to use.
        
        Args:
            error_type: Classification of the error
            retry_count: Current retry attempt number (0-indexed)
            stage_name: Name of the stage that failed
            
        Returns:
            Tuple of (should_retry: bool, delay_seconds: float)
        """
        if error_type == ErrorType.TRANSIENT:
            if retry_count < self._transient_max_retries:
                delay = self._transient_initial_delay * (
                    self._transient_backoff_factor ** retry_count
                )
                return True, delay
            return False, 0.0
        
        elif error_type == ErrorType.SEMI_TRANSIENT:
            if retry_count < self._semi_transient_max_retries:
                delay = self._semi_transient_initial_delay * (
                    self._semi_transient_backoff_factor ** retry_count
                )
                return True, delay
            return False, 0.0
        
        else:  # PERMANENT
            return False, 0.0
    
    def get_fallback(
        self,
        stage_name: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get fallback outputs for a failed stage.
        
        Args:
            stage_name: Name of the stage that failed
            context: Current workflow context
            
        Returns:
            Dict with fallback outputs, or None if no fallback available
        """
        if not self._fallbacks_enabled:
            return None
        
        criticality = self.get_stage_criticality(stage_name)
        
        # Only provide fallbacks for important stages
        if criticality != StageCriticality.IMPORTANT:
            return None
        
        if stage_name == "Verifier":
            logger.warning("Verifier failed, using fallback: assuming solution is incorrect")
            return {
                "detection_correctness": self._verifier_fallback_correctness,
                "detection_errors": [],
                "detection_contradiction_errors": [],
                "detection_explanation": "Verifier stage failed, assuming incorrect based on fallback."
            }
        
        elif stage_name == "ProgressAssessor":
            logger.warning("ProgressAssessor failed, using fallback: assuming no progress")
            cot_history = context.get("cot_history", [])
            # Use current step count as fallback
            last_useful_step_index = len(cot_history) if cot_history else 0
            return {
                "progress_assessment": {
                    "objective": "unknown",
                    "prefix_correctness": False,
                    "last_useful_step_index": last_useful_step_index
                },
                "progress_pointer": self._progress_assessor_fallback_pointer,
                "last_useful_step_index": last_useful_step_index
            }
        
        elif stage_name == "HintGenerator":
            logger.warning("HintGenerator failed, using fallback: generic hint")
            return {
                "hint": self._hint_generator_fallback_hint,
                "translated_hint": self._hint_generator_fallback_hint,
                "translated_hint_steps": []
            }
        
        # No fallback for other stages
        return None
    
    def preserve_state(
        self,
        context: Dict[str, Any],
        error: Exception,
        stage_name: str,
        cot_history: list,
        final_answer: str,
        iteration_count: int,
        hint_level: int
    ) -> Dict[str, Any]:
        """
        Preserve workflow state before failure.
        
        Args:
            context: Current workflow context
            error: The exception that occurred
            stage_name: Name of the stage that failed
            cot_history: Current CoT history
            final_answer: Current final answer
            iteration_count: Current iteration count
            hint_level: Current hint level
            
        Returns:
            Updated context with preserved state
        """
        # Preserve CoT and answer
        from easy_scaffold.workflows.agents.generator_critique_agent import (
            GeneratorCritiqueWorkflow
        )
        # We can't import here due to circular dependency, so we'll use a helper
        # The workflow will call this with the joined CoT
        
        context["generator_cot"] = context.get("generator_cot", "")
        context["generator_answer"] = final_answer
        context["iteration_count"] = iteration_count
        context["hint_level"] = hint_level
        
        # Add error information
        context["error"] = str(error)
        context["error_type"] = self.classify_error(error).value if isinstance(error, StageExecutionError) else "unknown"
        context["failed_stage"] = stage_name
        
        return context
    
    def handle_stage_error(
        self,
        stage_error: StageExecutionError,
        stage_name: str,
        retry_count: int,
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[float]]:
        """
        Handle a stage execution error and determine next action.
        
        Args:
            stage_error: The StageExecutionError that occurred
            stage_name: Name of the stage that failed
            retry_count: Current retry attempt number
            context: Current workflow context
            
        Returns:
            Tuple of:
            - should_retry: Whether to retry the stage
            - fallback_outputs: Fallback outputs if available, None otherwise
            - delay_seconds: Delay before retry if should_retry is True
        """
        error_type = self.classify_error(stage_error)
        criticality = self.get_stage_criticality(stage_name)
        
        logger.info(
            f"Stage '{stage_name}' failed: {stage_error.status} "
            f"(type={error_type.value}, criticality={criticality.value})"
        )
        
        # Check if we should retry
        should_retry, delay = self.should_retry(error_type, retry_count, stage_name)
        
        if should_retry:
            logger.info(f"Will retry stage '{stage_name}' after {delay:.2f}s (attempt {retry_count + 1})")
            return True, None, delay
        
        # No retry - check for fallback
        if criticality == StageCriticality.CRITICAL:
            logger.error(f"Critical stage '{stage_name}' failed and cannot be retried. Workflow must fail.")
            return False, None, None
        
        # Try to get fallback
        fallback_outputs = self.get_fallback(stage_name, context)
        
        if fallback_outputs:
            logger.info(f"Using fallback for stage '{stage_name}'")
            return False, fallback_outputs, None
        
        # No fallback available
        if criticality == StageCriticality.OPTIONAL:
            logger.warning(f"Optional stage '{stage_name}' failed. Skipping and continuing.")
            return False, {}, None  # Empty outputs to continue
        
        # Important stage with no fallback - must fail
        logger.error(f"Important stage '{stage_name}' failed with no fallback available. Workflow must fail.")
        return False, None, None


