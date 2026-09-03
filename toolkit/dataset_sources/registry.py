"""
Registry for RemoteDatasetSource implementations.

Sources register themselves at startup by calling register_source().
Extensions do this in their __init__.py so they are only loaded on demand.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from toolkit.dataset_sources.base import RemoteDatasetSource

_registry: Dict[str, Type["RemoteDatasetSource"]] = {}


def register_source(source_class: Type["RemoteDatasetSource"]) -> None:
    """Register a RemoteDatasetSource subclass by its type_id."""
    if not source_class.type_id:
        raise ValueError(f"{source_class.__name__} must set type_id")
    _registry[source_class.type_id] = source_class


def get_source(type_id: str) -> Optional[Type["RemoteDatasetSource"]]:
    """Return the source class for *type_id*, or None if not registered."""
    return _registry.get(type_id)


def get_all_sources() -> List[Type["RemoteDatasetSource"]]:
    """Return all registered source classes."""
    return list(_registry.values())


def resolve_dataset_source(source_config: dict, settings: dict, cache_dir: str) -> str:
    """
    Convenience function called by the dataloader.

    Looks up the correct RemoteDatasetSource by type_id, instantiates it
    with the current settings, and calls resolve().

    Parameters
    ----------
    source_config:
        The ``dataset_source`` dict from the YAML config.
    settings:
        Key/value settings dict loaded from the AI-Toolkit settings DB.
    cache_dir:
        Root directory for cached dataset folders.

    Returns
    -------
    str
        Absolute local folder path ready for AiToolkitDataset.
    """
    type_id = source_config.get("type")
    if not type_id:
        raise ValueError("dataset_source config is missing required 'type' field")

    source_class = get_source(type_id)
    if source_class is None:
        raise ValueError(
            f"No RemoteDatasetSource registered for type '{type_id}'. "
            f"Registered types: {list(_registry.keys())}"
        )

    source = source_class(settings)
    return source.resolve(source_config, cache_dir)


def load_settings_from_db(db_path: str) -> dict:
    """
    Load the AI-Toolkit settings from the SQLite database.
    Returns an empty dict if the DB does not exist or cannot be read.
    """
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM Settings")
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}
    except Exception:
        return {}
