# src/easy_scaffold/workflows/agents/solution_verification_agent.py
import logging
import re
from typing import Any, Dict, Optional

from easy_scaffold.db.pydantic_models import RunLog
from easy_scaffold.workflows.base import AbstractWorkflow
from easy_scaffold.workflows.workflow_models import SolutionVerificationPayload, WorkItem

logger = logging.getLogger(__name__)


class SolutionVerificationWorkflow(AbstractWorkflow[SolutionVerificationPayload]):
    """
    Workflow for verifying correctness of existing solutions.
    
    This is a simpler, focused workflow compared to EvalWorkflow:
    - Takes an existing solution from a document field
    - Evaluates it using a judge (ProofCheckerJudge or MathJudge)
    - Saves the verification result to a specified field
    
    No generation, no Pass@k, just verification.
    """

    def _extract_response_from_thinking_model(self, full_output: str) -> str:
        """
        Extract response from thinking model output.
        Thinking models output CoT in <think>...</think> blocks.
        This method extracts everything after the closing tag as the response.
        """
        if not full_output:
            return ""
        
        # Patterns for thinking block closing tags (matching eval_agent pattern)
        thinking_end_patterns = [
            r'`</think>`',   # Backtick-wrapped (most specific)
            r'</think>',     # Plain tag
        ]
        
        last_match = None
        last_pos = -1
        
        for pattern in thinking_end_patterns:
            matches = list(re.finditer(pattern, full_output, re.IGNORECASE))
            if matches:
                match = matches[-1]
                if match.end() > last_pos:
                    last_match = match
                    last_pos = match.end()
        
        if last_match:
            response = full_output[last_pos:].strip()
            logger.debug(f"Extracted response from thinking model (CoT length: {last_pos}, Response length: {len(response)})")
            return response
        
        return full_output.strip()

    async def _run(
        self,
        work_item: WorkItem[SolutionVerificationPayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        """
        Verify an existing solution.
        
        Configuration:
        - task_type: "math_correctness" or "proof_correctness"
        - judge_stage_name: "MathJudge" or "ProofCheckerJudge"
        """
        payload = work_item.payload
        context: Dict[str, Any] = payload.model_dump()
        context["run_log"] = run_log

        # Configuration
        task_type = self._config.get("task_type", "proof_correctness")
        judge_stage_name = self._config.get("judge_stage_name", "ProofCheckerJudge")
        
        problem_id = context.get("problem_id", "unknown")
        solution_to_verify = context.get("solution_to_verify", "")
        
        logger.info(f"Starting Solution Verification ({task_type}) for {problem_id}")

        if not solution_to_verify:
            logger.warning("No solution provided to verify")
            return {
                "status": "failed",
                "correct": False,
                "reasoning": "No solution provided to verify",
                "method": "missing_solution"
            }

        # Extract response from thinking model if applicable
        response_text = self._extract_response_from_thinking_model(solution_to_verify)
        has_thinking = response_text != solution_to_verify.strip()

        # Prepare judge context
        judge_context = context.copy()
        judge_context["generator_solution"] = response_text  # Use extracted response, not full CoT
        judge_context["problem_statement"] = context.get("problem_statement")
        
        # Add task-specific inputs
        if task_type == "math_correctness":
            judge_context["ground_truth_answer"] = context.get("ground_truth_answer")
        elif task_type == "proof_correctness":
            judge_context["reference_solution"] = context.get("reference_solution")
            if not judge_context["reference_solution"]:
                logger.warning("Proof correctness task requires reference_solution, but none provided")
                return {
                    "status": "failed",
                    "correct": False,
                    "reasoning": "Proof correctness evaluation requires a reference solution",
                    "method": "missing_reference"
                }

        # Execute judge
        logger.info(f"Using {judge_stage_name} to evaluate solution")
        judge_result = await self._execute_stage(
            judge_stage_name, judge_context, run_log, document_id, workflow_name
        )
        
        outputs = judge_result.get("outputs", {})
        
        # Build result
        result = {
            "status": "completed",
            "correct": outputs.get("correct", False),
            "reasoning": outputs.get("reasoning", "No reasoning provided"),
            "method": judge_stage_name.lower()
        }
        
        # Add task-specific outputs
        if task_type == "proof_correctness":
            errors = outputs.get("errors", [])
            # Convert ProofError Pydantic objects to dictionaries for MongoDB storage
            errors_dict = []
            if errors and isinstance(errors, list):
                for error in errors:
                    if hasattr(error, "model_dump"):
                        errors_dict.append(error.model_dump())
                    elif isinstance(error, dict):
                        errors_dict.append(error)
                    else:
                        errors_dict.append({
                            "type": getattr(error, "type", str(error)),
                            "description": getattr(error, "description", ""),
                            "location": getattr(error, "location", "")
                        })
            result["errors"] = errors_dict
        
        # Store full output if thinking model was used
        if has_thinking:
            result["solution_full"] = solution_to_verify
        
        logger.info(f"Verification Complete. Correct: {result['correct']}")
        
        return result



