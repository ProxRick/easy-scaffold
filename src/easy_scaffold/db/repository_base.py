from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pymongo import ReturnDocument


class AbstractRepository(ABC):
    """Generic async persistence interface used across the application."""

    @abstractmethod
    async def fetch_many(
        self,
        collection: str,
        filters: Dict[str, Any],
        *,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return all documents matching ``filters``."""
        raise NotImplementedError

    @abstractmethod
    async def fetch_one(
        self,
        collection: str,
        filters: Dict[str, Any],
        *,
        database: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return a single document matching ``filters``."""
        raise NotImplementedError

    @abstractmethod
    async def update_one(
        self,
        collection: str,
        object_id: Any,
        update_doc: Dict[str, Any],
        *,
        database: Optional[str] = None,
    ) -> None:
        """Apply ``update_doc`` to the document identified by ``_id``."""
        raise NotImplementedError

    @abstractmethod
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
        """Atomically update and optionally return a single document."""
        raise NotImplementedError

    @abstractmethod
    async def replace_one(
        self,
        collection: str,
        filter_doc: Dict[str, Any],
        replacement: Dict[str, Any],
        *,
        database: Optional[str] = None,
        upsert: bool = False,
    ) -> None:
        """Replace a document matching ``filter_doc`` entirely."""
        raise NotImplementedError


