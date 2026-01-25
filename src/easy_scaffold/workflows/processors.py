from typing import Any, Dict

from pydantic import BaseModel


def process_verification_result(result: BaseModel) -> Dict[str, Any]:
    """Checks for a 'yes' in the 'verdict' field of a Pydantic model."""
    is_correct = False
    if hasattr(result, "verdict"):
        is_correct = "yes" in str(getattr(result, "verdict", "")).lower()
    return {"is_correct": is_correct}


def process_completeness_result(result: BaseModel) -> Dict[str, Any]:
    """Checks for a 'yes' in the 'verdict' field and returns 'is_complete'."""
    is_complete = False
    if hasattr(result, "verdict"):
        is_complete = "yes" in str(getattr(result, "verdict", "")).lower()
    return {"is_complete": is_complete}


def process_rich_verification_result(result: BaseModel) -> Dict[str, Any]:
    """Process RichVerificationOutput to extract key flags and update proof state"""
    output = {}

    # Basic flags for backward compatibility
    if hasattr(result, "verdict"):
        output["is_correct"] = result.verdict == "correct"

    if hasattr(result, "is_complete"):
        output["is_complete"] = result.is_complete

    # Extract blocking issues count
    if hasattr(result, "issues"):
        blocking_count = sum(1 for issue in result.issues if issue.blocking)
        output["has_blocking_issues"] = blocking_count > 0
        output["blocking_issues_count"] = blocking_count

    # Extract verification data for proof state update
    if hasattr(result, "verified_claims"):
        output["newly_verified_claims"] = result.verified_claims

    if hasattr(result, "next_subgoals"):
        output["new_subgoals"] = result.next_subgoals

    return output


def process_rich_completeness_result(result: BaseModel) -> Dict[str, Any]:
    """Process RichVerificationOutput for completeness check stage"""
    return {
        "is_complete": getattr(result, "is_complete", False),
        "missing_aspects": getattr(result, "missing_lemmas", []),
    }


def initialize_proof_state(result: BaseModel) -> Dict[str, Any]:
    """Initialize proof state from extracted evidence"""
    proof_state = {
        "claims": getattr(result, "claims", []),
        "subgoals": getattr(result, "subgoals", []),
        "issues": getattr(result, "issues", []),
        "lemmas_used": getattr(result, "lemmas_used", []),
        "proof_structure": getattr(result, "proof_structure", None),
        "progress_score": 0.0,
    }

    # Compute initial progress
    if hasattr(result, "compute_progress"):
        proof_state["progress_score"] = result.compute_progress()

    return {"proof_state": proof_state}


