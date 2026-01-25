# src/easy_scaffold/workflows/agents/cot_conclusion_workflow.py
import logging
from typing import Any, Dict

from easy_scaffold.db.pydantic_models import RunLog
from easy_scaffold.workflows.base import AbstractWorkflow
from easy_scaffold.workflows.workflow_models import (
    CoTConclusionPayload,
    WorkItem,
)

logger = logging.getLogger(__name__)


class CoTConclusionWorkflow(AbstractWorkflow[CoTConclusionPayload]):
    """
    Workflow that adds a strong, conclusive paragraph to the end of a Chain of Thought.
    This helps terminate loopy CoTs that have repetitive or uncertain endings.
    """

    async def _run(
        self,
        work_item: WorkItem[CoTConclusionPayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        """
        Execute the CoT conclusion workflow:
        1. Call CoTConclusion stage to generate conclusive paragraph
        2. Append the paragraph to the original CoT
        3. Return both original and concluded CoT
        """
        payload = work_item.payload
        context: Dict[str, Any] = {
            "problem_statement": payload.problem_statement,
            "generator_cot": payload.generator_cot,
        }

        # Execute CoTConclusion stage
        try:
            conclusion_result = await self._execute_stage(
                "CoTConclusion",
                context,
                run_log,
                document_id,
                workflow_name,
            )
            conclusive_paragraph = conclusion_result.get("outputs", {}).get("conclusive_paragraph", "").strip()
            
            if not conclusive_paragraph:
                logger.warning("CoTConclusion stage returned empty paragraph. Using original CoT.")
                conclusive_paragraph = ""
            
            # Append conclusive paragraph to original CoT with double newline
            if conclusive_paragraph:
                concluded_cot = payload.generator_cot.rstrip() + "\n\n" + conclusive_paragraph
            else:
                concluded_cot = payload.generator_cot
            
            context["generator_cot"] = payload.generator_cot  # Original CoT
            context["concluded_cot"] = concluded_cot  # CoT with conclusion appended
            context["conclusive_paragraph"] = conclusive_paragraph  # Just the paragraph
            context["status"] = "completed"
            
            logger.info(f"CoT conclusion added. Original length: {len(payload.generator_cot)}, "
                       f"Concluded length: {len(concluded_cot)}, "
                       f"Paragraph length: {len(conclusive_paragraph)}")
            
        except Exception as e:
            logger.error(f"CoTConclusion stage failed: {e}")
            # Return original CoT if conclusion fails
            context["generator_cot"] = payload.generator_cot
            context["concluded_cot"] = payload.generator_cot
            context["conclusive_paragraph"] = ""
            context["status"] = "failed"
            raise

        return context



