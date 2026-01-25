from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable

from bson import ObjectId

from easy_scaffold.db.repository_base import AbstractRepository
from easy_scaffold.workflows.workflow_models import WorkflowBindingConfig


class BindingResolver:
    """Resolves query placeholders and maps documents to workflow payloads and back."""

    def __init__(self, config: WorkflowBindingConfig):
        self._config = config

    # ---------------------------------------------------------------------
    # Query Resolution
    # ---------------------------------------------------------------------
    def build_query(self, workflow_runtime_config: Dict[str, Any]) -> Dict[str, Any]:
        import logging
        logger = logging.getLogger(__name__)
        template = deepcopy(self._config.query.template)
        logger.info(f"Query template (before resolution): {template}")
        resolved = self._resolve_structure(template, workflow_runtime_config)
        logger.info(f"Query filters (after resolution): {resolved}")
        
        # Add sharding condition if num_shards and shard_index are provided
        num_shards = workflow_runtime_config.get("num_shards")
        shard_index = workflow_runtime_config.get("shard_index")
        
        if num_shards is not None and shard_index is not None:
            try:
                num_shards = int(num_shards)
                shard_index = int(shard_index)
                
                # Validate sharding parameters
                if num_shards <= 0:
                    logger.warning(f"Invalid num_shards ({num_shards}): must be > 0. Skipping sharding.")
                elif shard_index < 0 or shard_index >= num_shards:
                    logger.warning(
                        f"Invalid shard_index ({shard_index}): must be >= 0 and < num_shards ({num_shards}). "
                        "Skipping sharding."
                    )
                else:
                    # Build sharding condition using pre-computed shard_hash field
                    # This field should be added to documents using scripts/add_shard_index.py
                    # The hash value works with any num_shards via modulo operation
                    shard_condition = {"shard_hash": {"$mod": [num_shards, shard_index]}}
                    
                    # Combine with existing query using $and
                    if "$and" in resolved:
                        # Append to existing $and array
                        resolved["$and"].append(shard_condition)
                        logger.info(f"Added sharding condition to existing $and: {shard_condition}")
                    else:
                        # Convert existing query to $and format
                        # Copy all existing conditions (excluding any $and that might exist)
                        existing_conditions = {k: v for k, v in resolved.items() if k != "$and"}
                        resolved = {
                            "$and": [existing_conditions, shard_condition]
                        }
                        logger.info(f"Wrapped query in $and and added sharding condition: {shard_condition}")
                    
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Invalid sharding parameters (num_shards={num_shards}, shard_index={shard_index}): {e}. "
                    "Skipping sharding."
                )
        elif num_shards is not None or shard_index is not None:
            # One parameter provided but not both
            logger.warning(
                f"Sharding requires both num_shards and shard_index. "
                f"Got num_shards={num_shards}, shard_index={shard_index}. Skipping sharding."
            )
        
        logger.info(f"Final query filters: {resolved}")
        return resolved

    # ---------------------------------------------------------------------
    # Payload Extraction
    # ---------------------------------------------------------------------
    def extract_payload_data(self, document: Dict[str, Any]) -> Dict[str, Any]:
        payload_data: Dict[str, Any] = {}
        for field_name, binding in self._config.bindings.inputs.items():
            value = self._get_nested_value(document, binding.from_)
            if isinstance(value, ObjectId):
                value = str(value)
            payload_data[field_name] = value
        return payload_data

    # ---------------------------------------------------------------------
    # Output Application
    # ---------------------------------------------------------------------
    async def apply_outputs(
        self,
        document_id: Any,
        workflow_result: Dict[str, Any],
        repository: AbstractRepository,
        workflow_runtime_config: Dict[str, Any] = None,
    ) -> None:
        if not workflow_result:
            return

        set_ops: Dict[str, Any] = {}
        for field_name, binding in self._config.bindings.outputs.items():
            if field_name in workflow_result:
                # Resolve dynamic path if workflow_runtime_config is provided
                if workflow_runtime_config:
                    resolved_path = self._resolve_token(binding.to, workflow_runtime_config, expect_string=True)
                else:
                    resolved_path = binding.to
                set_ops[resolved_path] = workflow_result[field_name]

        if set_ops:
            await repository.update_one(
                self._config.query.collection,
                document_id,
                {"$set": set_ops},
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_structure(
        self,
        value: Any,
        workflow_runtime_config: Dict[str, Any],
    ) -> Any:
        if isinstance(value, dict):
            resolved_dict: Dict[str, Any] = {}
            for key, val in value.items():
                resolved_key = self._resolve_token(key, workflow_runtime_config, expect_string=True)
                resolved_dict[resolved_key] = self._resolve_structure(val, workflow_runtime_config)
            return resolved_dict
        if isinstance(value, list):
            return [self._resolve_structure(item, workflow_runtime_config) for item in value]
        if isinstance(value, str):
            return self._resolve_token(value, workflow_runtime_config)
        return value

    def _resolve_token(
        self,
        token: str,
        workflow_runtime_config: Dict[str, Any],
        expect_string: bool = False,
    ) -> Any:
        import re

        # Handle full template variable: {workflow_config.key}
        if isinstance(token, str) and token.startswith("{") and token.endswith("}"):
            inner = token[1:-1]
            if inner.startswith("input."):
                field = inner.split(".", 1)[1]
                binding = self._config.bindings.inputs.get(field)
                return binding.from_ if binding else token

            if inner.startswith("workflow_config."):
                key = inner.split(".", 1)[1]
                value = workflow_runtime_config.get(key)
                return str(value) if expect_string else value

            return token
        
        # Handle embedded template variables: "prefix${workflow_config.key}suffix"
        # Pattern: ${workflow_config.key} or ${workflow.key}
        pattern = r'\$\{workflow(?:\.|_config\.)([^}]+)\}'
        
        def replace_var(match):
            key = match.group(1)
            value = workflow_runtime_config.get(key)
            if value is None:
                return match.group(0)  # Return original if not found
            return str(value)
        
        resolved = re.sub(pattern, replace_var, token)
        return resolved

    @staticmethod
    def _get_nested_value(data: Dict[str, Any], path: str) -> Any:
        current: Any = data
        for part in BindingResolver._split_path(path):
            if current is None:
                return None
            if isinstance(current, list):
                try:
                    index = int(part)
                    current = current[index]
                except (ValueError, IndexError):
                    return None
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    @staticmethod
    def _split_path(path: str) -> Iterable[str]:
        return path.split(".") if path else []



