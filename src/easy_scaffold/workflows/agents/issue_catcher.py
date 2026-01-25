# src/easy_scaffold/workflows/agents/issue_catcher.py
import logging
from typing import Any, Dict

from easy_scaffold.db.pydantic_models import RunLog
from easy_scaffold.workflows.base import AbstractWorkflow
from easy_scaffold.workflows.workflow_models import IssueCatcherPayload, WorkItem
from easy_scaffold.common.custom_exceptions import WorkflowException

logger = logging.getLogger(__name__)


class IssueCatcherWorkflow(AbstractWorkflow[IssueCatcherPayload]):
    """
    Workflow that checks if generated solutions admit self-reported issues,
    uncertainty, or flaws. Uses a single stage to analyze the solution text.
    """
    
    async def _run(
        self,
        work_item: WorkItem[IssueCatcherPayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        payload = work_item.payload
        context: Dict[str, Any] = payload.model_dump()
        context["run_log"] = run_log

        logger.info(f"Analyzing solution for problem {payload.problem_id}")

        # Execute single IssueCatcher stage
        result = await self._execute_stage(
            "IssueCatcher", context, run_log, document_id, workflow_name
        )

        # Extract structured output from raw_output (Pydantic model)
        raw_output = result.get("raw_output")
        
        if raw_output and hasattr(raw_output, "admits_issue"):
            # Successfully extracted structured output
            context["admits_issue"] = raw_output.admits_issue
            context["reasoning"] = raw_output.reasoning
            context["evidence"] = list(raw_output.evidence) if raw_output.evidence else []
            
            logger.info(
                f"Issue catcher result for problem {payload.problem_id}: "
                f"admits_issue={raw_output.admits_issue}, "
                f"evidence_count={len(raw_output.evidence) if raw_output.evidence else 0}"
            )
        else:
            # Failed to extract structured output - raise exception to prevent saving
            error_msg = (
                f"Failed to extract structured output from IssueCatcher stage for problem {payload.problem_id}. "
                f"Raw output type: {type(raw_output)}, value: {raw_output}"
            )
            logger.error(error_msg)
            raise WorkflowException(error_msg)

        context["status"] = "completed"
        return context



