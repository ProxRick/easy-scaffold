# src/easy_scaffold/app.py
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from easy_scaffold.configs.pydantic_models import MainConfig
from easy_scaffold.llm_client.clients import create_llm_client
from easy_scaffold.workflows.configurable_stage import StageFactory
from easy_scaffold.workflows.logger import WorkflowLogger
from easy_scaffold.workflows.orchestrator import Orchestrator


logger = logging.getLogger(__name__)


class App:
    """The main application class that orchestrates the agentic workflows."""

    def __init__(self, cfg: DictConfig):
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        self._config = MainConfig(**config_dict)
        self._raw_cfg = cfg # Keep raw config for hydra instantiation
        
        # Hydrate all components using Hydra for maximum flexibility
        self._repository = hydra.utils.instantiate(self._raw_cfg.repository)

        workflow_logger = WorkflowLogger(
            repository=self._repository,
            log_collection=self._config.db.logs_collection,
            log_database=self._config.db.logs_db_name,
        )

        llm_client = create_llm_client(self._config.llm)
        
        stage_factory = StageFactory(
            stage_configs=self._config.stages,
            llm_client=llm_client,
            llm_config=self._config.llm,
        )

        self._orchestrator = Orchestrator(
            logger=workflow_logger,
            lm_config=self._config.llm,
            stage_factory=stage_factory,
            repository=self._repository,
            app_config=self._config.app,
        )

        # Configure tenacity logger to show retry attempts
        logging.getLogger("tenacity").setLevel(logging.INFO)

    async def run(self):
        """Initializes components and runs the main orchestrator."""
        workflow_config = self._config.workflow
        binding_config = workflow_config.binding
        runtime_overrides = workflow_config.model_dump(exclude={"binding"})

        logger.info(f"Starting workflow: {workflow_config.name}")
        logger.debug(f"Workflow runtime config: {runtime_overrides}")

        await self._orchestrator.execute(
            binding_config=binding_config,
            workflow_runtime=runtime_overrides,
        )
        logger.info("All processing is complete.")


