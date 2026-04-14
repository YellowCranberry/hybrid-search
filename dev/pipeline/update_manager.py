"""
update_manager.py
-----------------
Orchestrates the ingestion pipeline:
    fetch from DB/manual → chunk → delta-check → embed → upsert to pgvector
"""

from typing import List, Optional, TYPE_CHECKING, Dict, Any

from ..adapters.sql_adapter import SQLAdapter
from ..core.delta_sync import DeltaSyncTracker

if TYPE_CHECKING:
    from ..core.engine import HybridSearchEngine


class UpdateManager:
    """
    Ties together chunking, delta-sync, and storage.

    Usage example:
        engine = HybridSearchEngine(db_url="postgresql://...")
        manager = UpdateManager(engine)

        # Pull from your existing SQL database
        manager.run_from_db(
            source_db_url="mysql+pymysql://...",
            table_name="posts",
            content_col="body",
            meta_cols=["id", "title", "slug", "published_at"],
        )
    """

    def __init__(self, engine: "HybridSearchEngine"):
        self.engine = engine
        self.tracker = DeltaSyncTracker(db_url=engine.db_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_from_db(
        self,
        source_db_url: str,
        table_name: str,
        content_col: str,
        meta_cols: List[str],
        url_col: Optional[str] = None,
    ) -> dict:
        """
        Pull rows from your SQL database and index them.

        Args:
            source_db_url: SQLAlchemy URL of the source database.
            table_name:    Table containing your content.
            content_col:   Column with the main text content.
            meta_cols:     Columns to include as metadata (title, slug, etc.).
            url_col:       Column containing the canonical URL (used for delta-sync).
                           If None, delta-sync is skipped for DB rows.

        Returns:
            Summary dict: {"indexed": int, "skipped": int, "failed": int}
        """
        print(f"[UpdateManager] Fetching rows from {table_name}...")
        adapter = SQLAdapter(source_db_url)
        rows = adapter.fetch_data(table_name, content_col, meta_cols)

        stats = {"indexed": 0, "skipped": 0, "failed": 0}

        for row in rows:
            try:
                text = row["text"]
                metadata = row["metadata"]
                url = metadata.get(url_col, f"db://{table_name}") if url_col else f"db://{table_name}/{stats['indexed'] + stats['skipped']}"
                content_hash = DeltaSyncTracker.compute_hash(text)

                if self.tracker.has_changed(url, content_hash):
                    self.engine.ingest_text(text=text, metadata=metadata)
                    self.tracker.mark_synced(url, content_hash)
                    stats["indexed"] += 1
                    print(f"  ✓ Indexed: {url}")
                else:
                    stats["skipped"] += 1

            except Exception as e:
                print(f"  ✗ Failed to index DB row: {e}")
                stats["failed"] += 1

        self._print_summary(stats)
        return stats

    def run_from_documents(self, documents: List[Dict[str, Any]], force: bool = False) -> dict:
        """
        Manually push documents format: [{"text": "...", "metadata": {"url": "..."}}]
        """
        stats = {"indexed": 0, "skipped": 0, "failed": 0}

        for doc in documents:
            url = doc["metadata"].get("url", "unknown-temp")
            text = doc["text"]
            content_hash = DeltaSyncTracker.compute_hash(text)

            try:
                if not force and self.tracker.has_changed(url, content_hash) is False:
                    stats["skipped"] += 1
                    continue

                self.engine.ingest_text(text=text, metadata=doc["metadata"])
                self.tracker.mark_synced(url, content_hash)
                stats["indexed"] += 1
                print(f"  ✓ Indexed: {url}")
            except Exception as e:
                print(f"  ✗ Failed: {url} - {e}")
                stats["failed"] += 1

        self._print_summary(stats)
        return stats
        
    @staticmethod
    def _print_summary(stats: dict) -> None:
        print(
            f"\n[UpdateManager] Done — "
            f"Indexed: {stats['indexed']} | "
            f"Skipped (unchanged): {stats['skipped']} | "
            f"Failed: {stats['failed']}"
        )
