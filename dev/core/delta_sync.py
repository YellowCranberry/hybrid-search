"""
delta_sync.py
-------------
Tracks which URLs/URIs have been ingested and whether their content has changed.
Uses a lightweight `rag_sync_log` table in the same Postgres database.
"""

import hashlib
from typing import Optional

import psycopg


class DeltaSyncTracker:
    """
    Persists ingestion state in a `rag_sync_log` Postgres table.

    Table schema (auto-created on first use):
        url          TEXT PRIMARY KEY
        content_hash TEXT        NOT NULL
        ingested_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS rag_sync_log (
        url          TEXT PRIMARY KEY,
        content_hash TEXT        NOT NULL,
        ingested_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS rag_sync_log_hash_idx ON rag_sync_log (content_hash);
    """

    def __init__(self, db_url: str):
        # psycopg3-style URL
        self.db_url = db_url.replace("postgresql+psycopg://", "postgresql://")
        self._ensure_table()

    def has_changed(self, url: str, content_hash: str) -> bool:
        """Returns True if this URL is new or its content has changed."""
        existing_hash = self._get_hash(url)
        if existing_hash is None:
            return True
        return existing_hash != content_hash

    def mark_synced(self, url: str, content_hash: str) -> None:
        """Upserts the URL -> hash mapping after successful ingestion."""
        sql = """
        INSERT INTO rag_sync_log (url, content_hash, ingested_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (url) DO UPDATE
            SET content_hash = EXCLUDED.content_hash,
                ingested_at  = EXCLUDED.ingested_at;
        """
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (url, content_hash))
            conn.commit()

    @staticmethod
    def compute_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _ensure_table(self) -> None:
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(self.CREATE_TABLE_SQL)
            conn.commit()

    def _get_hash(self, url: str) -> Optional[str]:
        sql = "SELECT content_hash FROM rag_sync_log WHERE url = %s;"
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (url,))
                row = cur.fetchone()
        return row[0] if row else None
