# src/easy_scaffold/workflows/logger.py
import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

from pymongo import ReturnDocument

from easy_scaffold.db.pydantic_models import RunLog, SolverLog, StageLog
from easy_scaffold.db.repository_base import AbstractRepository

logger = logging.getLogger(__name__)


class WorkflowLogger:
    """Handles logging of workflow execution to the database."""

    def __init__(
        self,
        repository: AbstractRepository,
        log_collection: str,
        log_database: Optional[str] = None,
    ):
        self._repository = repository
        self._log_collection = log_collection
        self._log_database = log_database

    @staticmethod
    def _normalize_workflow_name(workflow_name: str) -> str:
        """Convert workflow name to a MongoDB field-safe name."""
        # Replace spaces and special characters with underscores, then lowercase
        normalized = re.sub(r"[^a-zA-Z0-9_]", "_", workflow_name)
        normalized = normalized.lower()
        # Remove consecutive underscores
        normalized = re.sub(r"_+", "_", normalized)
        # Remove leading/trailing underscores
        return normalized.strip("_")

    async def start_run(
        self,
        document_id: str,
        workflow_name: str,
        model_name: str,
        temperature: float,
    ) -> RunLog:
        """Starts a new run for a given document and workflow."""
        solver_log = await self._ensure_log_document(document_id, workflow_name)
        workflow_field = self._normalize_workflow_name(workflow_name)

        # Get current runs for this workflow to compute run_id
        workflow_runs = solver_log.runs.get(workflow_field, [])
        run_id = len(workflow_runs) + 1

        new_run = RunLog(
            run_id=run_id,
            workflow_name=workflow_name,
            model_name=model_name,
            temperature=temperature,
            status="running",
        )

        await self._repository.find_one_and_update(
            self._log_collection,
            {"document_id": document_id},
            {"$push": {f"runs.{workflow_field}": new_run.model_dump(by_alias=True, exclude_none=True)}},
            database=self._log_database,
        )
        return new_run

    async def log_stage_result(
        self,
        document_id: str,
        workflow_name: str,
        run_id: int,
        stage_name: str,
        result: Dict[str, Any],
    ) -> None:
        """Logs the result of a single stage to the corresponding run."""
        workflow_field = self._normalize_workflow_name(workflow_name)
        stage_log = StageLog(stage_name=stage_name, **result)
        
        updated = await self._repository.find_one_and_update(
            self._log_collection,
            {"document_id": document_id, f"runs.{workflow_field}.run_id": run_id},
            {"$push": {f"runs.{workflow_field}.$.stages": stage_log.model_dump(by_alias=True, exclude_none=True)}},
            database=self._log_database,
            return_document=ReturnDocument.AFTER,
        )
        if updated is None:
            logger.error(
                "Failed to append stage log for document '%s' workflow '%s' (run_id=%s).",
                document_id,
                workflow_name,
                run_id,
            )

    async def end_run(self, document_id: str, workflow_name: str, run_id: int, status: str) -> None:
        """Marks a run as completed or failed."""
        workflow_field = self._normalize_workflow_name(workflow_name)
        updated = await self._repository.find_one_and_update(
            self._log_collection,
            {"document_id": document_id, f"runs.{workflow_field}.run_id": run_id},
            {
                "$set": {
                    f"runs.{workflow_field}.$.status": status,
                    f"runs.{workflow_field}.$.end_time": datetime.utcnow(),
                }
            },
            database=self._log_database,
            return_document=ReturnDocument.AFTER,
        )
        if updated is None:
            logger.error(
                "Failed to end run %s for document '%s' workflow '%s' (status=%s).",
                run_id,
                document_id,
                workflow_name,
                status,
            )

    async def _ensure_log_document(self, document_id: str, workflow_name: str) -> SolverLog:
        """Create the log document if it does not exist and return it."""
        workflow_field = self._normalize_workflow_name(workflow_name)
        
        # Ensure document exists with runs dict
        doc = await self._repository.find_one_and_update(
            self._log_collection,
            {"document_id": document_id},
            {"$setOnInsert": {"document_id": document_id, "runs": {}}},
            database=self._log_database,
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        if doc is None:
            message = f"Unable to create log document for document '{document_id}'."
            logger.error(message)
            raise RuntimeError(message)
        
        # Ensure the workflow field exists in runs dict
        if workflow_field not in doc.get("runs", {}):
            updated_doc = await self._repository.find_one_and_update(
                self._log_collection,
                {"document_id": document_id},
                {"$set": {f"runs.{workflow_field}": []}},
                database=self._log_database,
                return_document=ReturnDocument.AFTER,
            )
            if updated_doc is None:
                message = f"Unable to initialize workflow field '{workflow_field}' for document '{document_id}'."
                logger.error(message)
                raise RuntimeError(message)
            doc = updated_doc
        
        return SolverLog(**doc)


