# dev/__init__.py
# Public API — everything the user needs, in one import.

from .core.engine import HybridSearchEngine
from .pipeline.update_manager import UpdateManager
from .adapters.sql_adapter import SQLAdapter

__all__ = [
    "HybridSearchEngine",
    "UpdateManager",
    "SQLAdapter",
]