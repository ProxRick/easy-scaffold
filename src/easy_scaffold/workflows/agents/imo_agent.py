from typing import Any, Dict

from easy_scaffold.workflows.workflow_models import ProblemPayload, WorkItem
from easy_scaffold.db.pydantic_models import RunLog
from easy_scaffold.workflows.base import AbstractWorkflow
from easy_scaffold.workflows.processors import process_verification_result


class ImoAgentWorkflow(AbstractWorkflow[ProblemPayload]):
    async def _run(
        self,
        work_item: WorkItem[ProblemPayload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = work_item.payload.model_dump()
        context["run_log"] = run_log

        # The outer retry loop, formerly the 'max_runs' handler
        for attempt in range(self._config.get("max_runs", 10)):
            context["attempt"] = attempt + 1
            run_is_successful = await self._attempt_solution(context, run_log, document_id, workflow_name)
            if run_is_successful:
                context["status"] = "completed"
                return context  # Success
        
        context["status"] = "failed"
        return context  # All attempts failed

    async def _attempt_solution(
        self,
        context: Dict[str, Any],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> bool:
        """Runs one full attempt of the problem-solving process."""
        raw_output = None

        # Initial sequence of stages
        for stage_name in ["InitialDraft", "SelfImprovement", "CompletenessCheck"]:
            result = await self._execute_stage(stage_name, context, run_log, document_id, workflow_name)
            raw_output = result.get("raw_output")

        # The stateful verification loop
        is_verified = await self._run_verification_loop(context, run_log, document_id, workflow_name)
        return is_verified

    async def _run_verification_loop(
        self,
        context: Dict[str, Any],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> bool:
        """
        Runs the verification/correction loop, formerly the 'stateful_verification' handler.
        """
        raw_output = None
        success_counter = 0
        failure_counter = 0

        max_turns = self._config.get("max_verification_turns", 30)
        success_threshold = self._config.get("success_threshold", 5)
        failure_threshold = self._config.get("failure_threshold", 10)

        for turn in range(max_turns):
            context["verification_turn"] = turn + 1

            # Execute verification stages
            result = await self._execute_stage("GenerateVerification", context, run_log, document_id, workflow_name)
            raw_output = result.get("raw_output")

            result = await self._execute_stage("InterpretVerification", context, run_log, document_id, workflow_name)
            raw_output = result.get("raw_output")
            
            # The logic that replaces the 'verification' result_processor
            processed_result = process_verification_result(raw_output)
            context.update(processed_result)

            if context.get("is_correct"):
                success_counter += 1
                failure_counter = 0
                if success_counter >= success_threshold:
                    return True  # Verification successful
            else:
                failure_counter += 1
                success_counter = 0
                if failure_counter >= failure_threshold:
                    return False  # Verification failed

                # On failure, run the correction stage
                result = await self._execute_stage("Correction", context, run_log, document_id, workflow_name)
                raw_output = result.get("raw_output")

        return False  # Loop finished without meeting success threshold


