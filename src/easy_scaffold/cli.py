# src/easy_scaffold/cli.py
import asyncio
import logging
import sys
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from easy_scaffold.app import App
from easy_scaffold.common.custom_exceptions import ConfigException

logger = logging.getLogger(__name__)

# We no longer register the schema with Hydra, as Pydantic handles validation.
# cs = ConfigStore.instance()
# cs.store(name="config_schema", node=MainConfig)

# Load environment variables BEFORE Hydra decorator runs (so config resolution can access env vars)
# Find project root (where .env file should be) - go up from this file to project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")


@hydra.main(config_path="../../configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for the Easy Scaffold CLI."""

    # Set up logging from the configuration
    log_level = cfg.logging.level.upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    try:
        app = App(cfg)
        asyncio.run(app.run())

    except ConfigException as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


