"""Lazy PostgreSQL connection-pool access for the phase2 ingestion/search helpers."""

from __future__ import annotations

import os
from typing import Any

_pool: Any = None


def get_pool():
    """Create the shared psycopg connection pool on first use."""

    global _pool
    if _pool is None:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL is not set")
        from psycopg_pool import ConnectionPool

        _pool = ConnectionPool(conninfo=database_url, min_size=1, max_size=5)
    return _pool


def get_conn():
    """Return a context-managed pooled connection."""

    return get_pool().connection()
