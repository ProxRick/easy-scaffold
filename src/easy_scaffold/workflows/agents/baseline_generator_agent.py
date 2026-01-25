# src/easy_scaffold/workflows/agents/baseline_generator_agent.py
import logging
import re
from typing import Any, Dict

from easy_scaffold.db.pydantic_models import RunLog
from easy_scaffold.workflows.base import AbstractWorkflow
from easy_scaffold.workflows.workflow_models import (
    BaselineGeneratorPayload,
    WorkItem,
)

logger = logging.getLogger(__name__)


class BaselineGeneratorWorkflow(AbstractWorkflow[BaselineGeneratorPayload]):
    """
    Baseline workflow that generates a solution in one shot without critique.
    This serves as a control group to study the impact of the critique mechanism.
    """

    def _extract_cot_and_solution(self, full_text: str) -> tuple[str, str]:
        """
        Extract Chain of Thought (CoT) and solution from generator output.
        Looks for thinking block markers (e.g., `</think>`).
        """
        thinking_end_patterns = [
            r'`</think>`',
            r'`</think>`',
            r'</think>',
            r'</think>',
        ]
        
        cot_text = full_text
        solution_text = ""
        
        for pattern in thinking_end_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                end_pos = match.end()
                start_pos = match.start()
                # CoT is everything before the tag
                cot_text = full_text[:start_pos].strip()
                # Solution is everything after the tag
                solution_text = full_text[end_pos:].strip()
                return cot_text, solution_text
        
        # If no tag found, return full text as CoT (assuming incomplete or no separator)
        return cot_text, ""

    async def _run(
        self,
        work_item: WorkItem[BaselineGeneratorPayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        payload = work_item.payload
        context: Dict[str, Any] = payload.model_dump()
        context["run_log"] = run_log

        logger.info(f"Starting baseline generator workflow for {payload.problem_id}")

        # Execute single generation call
        result = await self._execute_stage(
            "Generator", context, run_log, document_id, workflow_name
        )
        
        generator_output = result.get("outputs", {}).get("generator_solution", "")
        
        if not generator_output:
            logger.warning("Generator produced empty output. Marking as failed.")
            context["status"] = "failed"
            context["baseline_cot"] = ""
            context["baseline_answer"] = ""
            return context

        # Extract CoT and solution
        cot_text, solution_text = self._extract_cot_and_solution(generator_output)
        
        logger.info(
            f"Generated CoT length: {len(cot_text)} chars, "
            f"Solution length: {len(solution_text)} chars"
        )

        # Set outputs
        context["status"] = "completed"
        context["baseline_cot"] = cot_text
        context["baseline_answer"] = solution_text

        return context



