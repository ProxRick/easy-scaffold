# src/easy_scaffold/workflows/orchestrator.py
import asyncio
import logging
import random
from typing import Any, Dict

from easy_scaffold.common.custom_exceptions import RateLimitException
from easy_scaffold.common.utils import import_from_string
from easy_scaffold.configs.pydantic_models import AppConfig, LLMConfig
from easy_scaffold.db.repository_base import AbstractRepository
from easy_scaffold.workflows.binding_resolver import BindingResolver
from easy_scaffold.workflows.configurable_stage import StageFactory
from easy_scaffold.workflows.logger import WorkflowLogger
from easy_scaffold.workflows.workflow_models import (
    WorkItem,
    WorkItemMetadata,
    WorkflowBindingConfig,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """Executes workflows using binding-driven payload construction."""

    def __init__(
        self,
        logger: WorkflowLogger,
        lm_config: LLMConfig,
        stage_factory: StageFactory,
        repository: AbstractRepository,
        app_config: AppConfig,
    ):
        self._logger = logger
        self._lm_config = lm_config
        self._stage_factory = stage_factory
        self._repository = repository
        self._app_config = app_config

    async def execute(
        self,
        binding_config: WorkflowBindingConfig,
        workflow_runtime: Dict[str, Any],
    ) -> None:
        resolver = BindingResolver(binding_config)

        query_filters = resolver.build_query(workflow_runtime)
        logger.debug(f"Resolved query filters: {query_filters}")
        collection_name = binding_config.query.collection
        
        # Get database name from repository's default or workflow runtime
        database_name = getattr(self._repository, '_default_database', None) or workflow_runtime.get("database")
        
        # Check if randomization is requested
        shuffle = workflow_runtime.get("shuffle", False)
        limit = workflow_runtime.get("limit")
        
        # Parse and validate limit
        parsed_limit = None
        if limit is not None:
            try:
                parsed_limit = int(limit)  # Ensure it's an integer (Hydra might pass as string)
                if parsed_limit <= 0:
                    parsed_limit = None
                    logger.warning(f"Invalid limit value '{limit}' (must be > 0), ignoring limit")
            except (ValueError, TypeError):
                logger.warning(f"Invalid limit value '{limit}', ignoring limit")
        
        # Fetch documents with randomization if supported, otherwise use standard fetch
        if shuffle and hasattr(self._repository, 'fetch_many_randomized'):
            # Use MongoDB-specific randomized fetch (uses shard_hash if available, otherwise $rand aggregation)
            logger.info("Using MongoDB randomized fetch (shard_hash or $rand aggregation)")
            documents = await self._repository.fetch_many_randomized(
                collection_name,
                query_filters,
                limit=parsed_limit,
                database=database_name,
            )
            logger.info(f"Found {len(documents)} documents (randomized and limited in MongoDB)")
        else:
            # Standard fetch
            documents = await self._repository.fetch_many(
                collection_name,
                query_filters,
                database=database_name,
            )
            logger.info(f"Found {len(documents)} documents matching query")
            
            # Apply randomization in Python if requested but not supported by repository
            if shuffle:
                logger.info("Shuffling documents in Python (repository doesn't support randomized fetch)")
                random.shuffle(documents)
            
            # Apply limit if specified and not already applied by randomized fetch
            if parsed_limit is not None:
                original_count = len(documents)
                documents = documents[:parsed_limit]
                logger.info(f"Limited to {parsed_limit} documents (found {original_count} matching documents, processing {len(documents)})")

        workflow_cls = import_from_string(binding_config.workflow.class_)
        payload_model_cls = import_from_string(binding_config.workflow.payload_model)
        workflow_name = workflow_runtime.get("name", binding_config.workflow.class_)
        default_profile = self._lm_config.resolve_profile()

        # Process documents with concurrency limit using dynamic task creation
        # Only create tasks when semaphore slots are available (no artificial delays)
        semaphore = asyncio.Semaphore(self._app_config.max_concurrent_tasks)

        async def process_document(doc):
            async with semaphore:
                payload_data = resolver.extract_payload_data(doc)
                payload = payload_model_cls(**payload_data)
                work_item = WorkItem(
                    payload=payload,
                    metadata=WorkItemMetadata(document_id=str(doc.get("_id"))),
                )

                workflow_instance = workflow_cls(
                    stage_factory=self._stage_factory,
                    logger=self._logger,
                    **workflow_runtime,
                )

                try:
                    final_context = await workflow_instance.execute(
                        work_item,
                        workflow_name=workflow_name,
                        model_name=default_profile.model,
                        temperature=default_profile.temperature,
                    )
                    await resolver.apply_outputs(doc.get("_id"), final_context, self._repository, workflow_runtime)
                except RateLimitException as err:
                    logger.error(f"Rate limit exceeded. Stopping execution: {err}")
                    raise  # Stop all processing

        # Dynamic task creation: maintain exactly max_concurrent_tasks tasks in flight
        # Workers pull from queue and process documents; semaphore naturally limits concurrency
        async def process_with_dynamic_tasks():
            doc_queue = asyncio.Queue()
            for doc in documents:
                await doc_queue.put(doc)
            
            # Use None as sentinel to signal workers to stop
            STOP_SENTINEL = None
            completed_count = 0
            total_docs = len(documents)
            workers = []  # Will be populated below
            
            async def worker():
                """Worker coroutine that processes documents from queue until sentinel."""
                nonlocal completed_count
                while True:
                    # Get next document (blocks until available)
                    doc = await doc_queue.get()
                    
                    # Check for stop sentinel
                    if doc is STOP_SENTINEL:
                        doc_queue.task_done()
                        break
                    
                    # Process document (semaphore inside process_document limits concurrency)
                    try:
                        await process_document(doc)
                        completed_count += 1
                        if completed_count % 10 == 0 or completed_count == total_docs:
                            logger.info(f"Progress: {completed_count}/{total_docs} documents completed")
                    except RateLimitException:
                        # Signal all workers to stop by putting sentinels
                        for _ in range(self._app_config.max_concurrent_tasks):
                            await doc_queue.put(STOP_SENTINEL)
                        raise
                    except Exception as e:
                        logger.error(f"Error processing document: {e}")
                    finally:
                        doc_queue.task_done()
            
            # Spawn exactly max_concurrent_tasks worker coroutines
            # Each worker processes documents sequentially; semaphore ensures only
            # max_concurrent_tasks are executing process_document at once
            workers = [
                asyncio.create_task(worker())
                for _ in range(self._app_config.max_concurrent_tasks)
            ]
            
            # Wait for queue to be empty (all documents processed)
            await doc_queue.join()
            
            # Signal all workers to stop
            for _ in range(self._app_config.max_concurrent_tasks):
                await doc_queue.put(STOP_SENTINEL)
            
            # Wait for all workers to finish
            await asyncio.gather(*workers, return_exceptions=False)
            
            logger.info(f"Completed processing all {total_docs} documents")

        # Process all documents with dynamic task creation
        try:
            await process_with_dynamic_tasks()
        except RateLimitException as err:
            logger.error(f"Rate limit exceeded. Stopped processing documents: {err}")
            raise



