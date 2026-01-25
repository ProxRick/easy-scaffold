from __future__ import annotations

from typing import Any, Dict, List, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument

from easy_scaffold.db.repository_base import AbstractRepository


class MongoRepository(AbstractRepository):
    """MongoDB-backed repository implementing the generic persistence interface."""

    def __init__(
        self,
        connection_string: str,
        default_database: str,
        **client_kwargs: Any,
    ):
        self._client = AsyncIOMotorClient(connection_string, **client_kwargs)
        self._default_database = default_database

    async def fetch_many(
        self,
        collection: str,
        filters: Dict[str, Any],
        *,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        import logging
        logger = logging.getLogger(__name__)
        db_name = database or self._default_database
        col = self._get_collection(collection, database)
        logger.debug(f"MongoDB query filters (before normalization): {filters}")
        normalized_filters = self._normalize_filter(filters)
        logger.debug(f"MongoDB query filters (after normalization): {normalized_filters}")
        cursor = col.find(normalized_filters)
        results = await cursor.to_list(length=None)
        logger.info(f"MongoDB query returned {len(results)} documents")
        return results
    
    async def fetch_many_randomized(
        self,
        collection: str,
        filters: Dict[str, Any],
        limit: Optional[int] = None,
        *,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        MongoDB-specific method to fetch documents with randomization.
        
        Uses pre-computed shard_hash field if available (deterministic shuffling),
        otherwise falls back to $rand aggregation (requires MongoDB 4.4+).
        
        Args:
            collection: Collection name
            filters: Query filters (same format as fetch_many)
            limit: Optional limit to apply
            database: Optional database name (uses default if not provided)
            
        Returns:
            List of documents, shuffled and optionally limited
        """
        import logging
        logger = logging.getLogger(__name__)
        col = self._get_collection(collection, database)
        
        logger.debug(f"MongoDB randomized query filters (before normalization): {filters}")
        normalized_filters = self._normalize_filter(filters)
        logger.debug(f"MongoDB randomized query filters (after normalization): {normalized_filters}")
        
        # Check if collection has shard_hash field (sample one document)
        sample_doc = await col.find_one(normalized_filters, {"shard_hash": 1})
        has_shard_hash = sample_doc is not None and "shard_hash" in sample_doc
        
        # Always use aggregation pipeline to support allowDiskUse for large sorts
        # This works even with indexed fields when sorting large result sets
        if has_shard_hash:
            # Use pre-computed shard_hash field in aggregation pipeline (supports allowDiskUse)
            logger.debug("Using pre-computed shard_hash field in aggregation pipeline for shuffling")
            pipeline = [
                # Step 1: Match filters (includes any sharding conditions)
                {"$match": normalized_filters},
                
                # Step 2: Sort by shard_hash (uses index, but aggregation allows allowDiskUse)
                {"$sort": {"shard_hash": 1}},
            ]
        else:
            # Fallback to $rand aggregation (slower, requires MongoDB 4.4+)
            logger.debug("Using $rand aggregation for randomization (consider adding shard_hash field for better performance)")
            pipeline = [
                # Step 1: Match filters (includes any sharding conditions)
                {"$match": normalized_filters},
                
                # Step 2: Add random field using $rand (MongoDB 4.4+)
                {"$addFields": {"_random": {"$rand": {}}}},
                
                # Step 3: Sort by random value
                {"$sort": {"_random": 1}},
            ]
        
        # Step 3/4: Apply limit if provided
        if limit is not None and limit > 0:
            pipeline.append({"$limit": limit})
        
        # Execute aggregation with allowDiskUse to handle large sorts
        # Motor's aggregate() accepts allowDiskUse as a keyword argument (same as PyMongo)
        logger.debug(f"Executing aggregation pipeline with allowDiskUse=True (pipeline length: {len(pipeline)}, using {'shard_hash' if has_shard_hash else '$rand'})")
        # Use explicit keyword argument - Motor wraps PyMongo and accepts same parameters
        cursor = col.aggregate(pipeline, allowDiskUse=True)
        results = await cursor.to_list(length=None)
        logger.info(f"MongoDB randomized query returned {len(results)} documents (limit={limit}, using {'shard_hash' if has_shard_hash else '$rand aggregation'})")
        return results

    async def fetch_one(
        self,
        collection: str,
        filters: Dict[str, Any],
        *,
        database: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        col = self._get_collection(collection, database)
        normalized_filters = self._normalize_filter(filters)
        return await col.find_one(normalized_filters)

    async def update_one(
        self,
        collection: str,
        object_id: Any,
        update_doc: Dict[str, Any],
        *,
        database: Optional[str] = None,
    ) -> None:
        col = self._get_collection(collection, database)
        identifier = self._ensure_object_id(object_id)
        await col.update_one({"_id": identifier}, update_doc)

    async def find_one_and_update(
        self,
        collection: str,
        filter_doc: Dict[str, Any],
        update_doc: Dict[str, Any],
        *,
        database: Optional[str] = None,
        upsert: bool = False,
        return_document: Optional[ReturnDocument] = None,
    ) -> Optional[Dict[str, Any]]:
        col = self._get_collection(collection, database)
        normalized_filter = self._normalize_filter(filter_doc)
        kwargs: Dict[str, Any] = {"upsert": upsert}
        if return_document is not None:
            kwargs["return_document"] = return_document
        return await col.find_one_and_update(normalized_filter, update_doc, **kwargs)

    async def replace_one(
        self,
        collection: str,
        filter_doc: Dict[str, Any],
        replacement: Dict[str, Any],
        *,
        database: Optional[str] = None,
        upsert: bool = False,
    ) -> None:
        col = self._get_collection(collection, database)
        normalized_filter = self._normalize_filter(filter_doc)
        await col.replace_one(normalized_filter, replacement, upsert=upsert)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_collection(self, collection: str, database: Optional[str]):
        db_name = database or self._default_database
        return self._client[db_name][collection]

    @staticmethod
    def _ensure_object_id(identifier: Any) -> ObjectId:
        if isinstance(identifier, ObjectId):
            return identifier
        return ObjectId(str(identifier))

    def _normalize_filter(self, filter_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize filter document, converting _id values to ObjectIds.
        Handles both simple _id values and MongoDB operator queries like $in/in.
        """
        if "_id" not in filter_doc:
            return filter_doc
        
        normalized = dict(filter_doc)
        _id_value = normalized["_id"]
        
        # Handle MongoDB operator queries (e.g., {"$in": [...]} or {"in": [...]})
        if isinstance(_id_value, dict):
            # Check for $in or in operator
            if "$in" in _id_value:
                # Convert array values to ObjectIds
                normalized["_id"] = {
                    "$in": [
                        self._ensure_object_id(item) if not isinstance(item, ObjectId) else item
                        for item in _id_value["$in"]
                    ]
                }
            elif "in" in _id_value:
                # Convert array values to ObjectIds (supporting both $in and in)
                normalized["_id"] = {
                    "$in": [  # MongoDB uses $in, so convert "in" to "$in"
                        self._ensure_object_id(item) if not isinstance(item, ObjectId) else item
                        for item in _id_value["in"]
                    ]
                }
            else:
                # Other operators - keep as is (could extend this if needed)
                normalized["_id"] = _id_value
        else:
            # Simple _id value - convert to ObjectId as before
            normalized["_id"] = self._ensure_object_id(_id_value)
        
        return normalized



