from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional

from easy_scaffold.workflows.workflow_models import Payload, WorkItem
from easy_scaffold.db.pydantic_models import RunLog
from easy_scaffold.workflows.configurable_stage import StageFactory
from easy_scaffold.workflows.logger import WorkflowLogger
from easy_scaffold.common.custom_exceptions import StageExecutionError
from easy_scaffold.configs.pydantic_models import StageLoggingConfig

class AbstractWorkflow(ABC, Generic[Payload]):
    def __init__(
        self,
        stage_factory: StageFactory,
        logger: WorkflowLogger,
        **kwargs,
    ):
        self._stage_factory = stage_factory
        self._logger = logger
        self._config = kwargs
        
        # Extract logging config if present
        logging_config = kwargs.get("logging")
        if isinstance(logging_config, dict):
            # Convert dict to StageLoggingConfig if needed (Hydra may pass as dict)
            self._logging_config = StageLoggingConfig(**logging_config)
        elif isinstance(logging_config, StageLoggingConfig):
            self._logging_config = logging_config
        else:
            self._logging_config = None

    async def execute(
        self,
        work_item: WorkItem[Payload],
        workflow_name: str,
        model_name: str,
        temperature: float,
    ) -> Dict[str, Any]:
        """Template method handling logging lifecycle around workflow execution."""

        document_id = str(work_item.metadata.document_id)
        run_log = await self._logger.start_run(
            document_id=document_id,
            workflow_name=workflow_name,
            model_name=model_name,
            temperature=temperature,
        )

        try:
            final_context = await self._run(work_item, run_log, document_id, workflow_name)
            status = final_context.get("status", "failed")
            await self._logger.end_run(document_id, workflow_name, run_log.run_id, status)
            return final_context
        except Exception:
            if run_log.status == "running":
                await self._logger.end_run(document_id, workflow_name, run_log.run_id, "failed")
            raise

    def _should_log_stage(self, stage_name: str) -> bool:
        """
        Determine if a stage should be logged based on logging configuration.
        
        Rules:
        1. If no logging config is specified, log all stages (backward compatibility)
        2. If logging.enabled = False, don't log any stage
        3. If exclude_stages is specified, log all stages except those listed
        4. If include_stages is specified, log only those stages listed
        
        Args:
            stage_name: Name of the stage to check
            
        Returns:
            True if the stage should be logged, False otherwise
        """
        # No logging config: default behavior (log everything)
        if self._logging_config is None:
            return True
        
        # Global toggle disabled: don't log anything
        if not self._logging_config.enabled:
            return False
        
        # Whitelist pattern: only log stages in include_stages
        if self._logging_config.include_stages is not None:
            return stage_name in self._logging_config.include_stages
        
        # Blacklist pattern: log all stages except those in exclude_stages
        if self._logging_config.exclude_stages is not None:
            return stage_name not in self._logging_config.exclude_stages
        
        # Default: log everything (if enabled=True but no patterns specified)
        return True

    @abstractmethod
    async def _run(
        self,
        work_item: WorkItem[Payload],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
    ) -> Dict[str, Any]:
        """Executes the workflow logic for a single work item."""
        pass

    async def _execute_stage(
        self,
        stage_name: str,
        context: Dict[str, Any],
        run_log: RunLog,
        document_id: str,
        workflow_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        stage = self._stage_factory.create_stage(stage_name)
        should_log = self._should_log_stage(stage_name)
        
        try:
            result = await stage.execute(context, overrides)
        except StageExecutionError as err:
            # Always log errors for debugging, even if stage is excluded from normal logging
            await self._logger.log_stage_result(
                document_id=document_id,
                workflow_name=workflow_name,
                run_id=run_log.run_id,
                stage_name=stage.name,
                result={
                    "inputs": err.inputs,
                    "outputs": {},
                    "status": err.status,
                    "error_message": str(err),
                    "token_stats": None,
                },
            )
            if err.cause:
                raise err.cause from err
            raise

        # Only log successful stage execution if logging is enabled for this stage
        if should_log:
            await self._logger.log_stage_result(
                document_id=document_id,
                workflow_name=workflow_name,
                run_id=run_log.run_id,
                stage_name=stage.name,
                result={
                    "inputs": result.get("inputs", {}),
                    "outputs": result.get("outputs", {}),
                    "status": "completed",
                    "token_stats": result.get("token_stats"),
                },
            )
        
        context.update(result.get("outputs", {}))
        return result


