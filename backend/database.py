# backend/database.py
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

# ------------------ paths ------------------

DB_PATH = (Path(__file__).parent / "floatchat.db").resolve()
# Kept for backward compatibility with other modules
DATABASE_PATH: str = str(DB_PATH)


# ------------------ connection helper ------------------

@asynccontextmanager
async def _open_db() -> aiosqlite.Connection:
    """
    Async context manager that opens an aiosqlite connection with sensible defaults:
    - Foreign keys ON
    - WAL journaling (better for concurrent readers)
    - NORMAL synchronous (good dev default)
    - Row factory set to a dict-like row
    Ensures the connection is always closed.
    """
    db = await aiosqlite.connect(DATABASE_PATH)
    try:
        await db.execute("PRAGMA foreign_keys = ON;")
        await db.execute("PRAGMA journal_mode = WAL;")
        await db.execute("PRAGMA synchronous = NORMAL;")
        db.row_factory = aiosqlite.Row
        yield db
    finally:
        await db.close()


# ------------------ schema ------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT DEFAULT (CURRENT_TIMESTAMP),
    last_updated TEXT DEFAULT (CURRENT_TIMESTAMP)
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,             -- 'user' | 'assistant' | 'system'
    content TEXT NOT NULL,
    timestamp TEXT DEFAULT (CURRENT_TIMESTAMP),
    provider TEXT,                  -- e.g. 'azure', 'openai', 'db'
    FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS netcdf_files (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    upload_timestamp TEXT DEFAULT (CURRENT_TIMESTAMP),
    file_size INTEGER NOT NULL,
    dimensions TEXT NOT NULL,        -- JSON
    variables TEXT NOT NULL,         -- JSON
    global_attributes TEXT NOT NULL, -- JSON
    processed INTEGER DEFAULT 0      -- 0/1
);

CREATE TABLE IF NOT EXISTS argo_data (
    id TEXT PRIMARY KEY,
    file_id TEXT NOT NULL,
    float_id TEXT,
    latitude REAL,
    longitude REAL,
    timestamp REAL,       -- seconds since epoch or CF numeric
    depth REAL,
    temperature REAL,
    salinity REAL,
    pressure REAL,
    oxygen REAL,
    quality_flag INTEGER,
    FOREIGN KEY (file_id) REFERENCES netcdf_files (id) ON DELETE CASCADE
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_time
ON chat_messages(session_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_netcdf_files_uploaded
ON netcdf_files(upload_timestamp);

CREATE INDEX IF NOT EXISTS idx_argo_file_time
ON argo_data(file_id, timestamp);
"""


# ------------------ init ------------------

async def init_database() -> None:
    """
    Initialize the SQLite database (tables + indexes).
    Safe to call multiple times.
    """
    async with _open_db() as db:
        await db.executescript(SCHEMA)
        await db.commit()


# ------------------ chat I/O ------------------

async def save_chat_message(
    session_id: str,
    role: str,
    content: str,
    provider: Optional[str] = None,
) -> str:
    """
    Save a chat message and ensure the session exists/updates.
    Returns the new message_id (UUID4).
    """
    message_id = str(uuid.uuid4())
    async with _open_db() as db:
        # Ensure session exists
        await db.execute(
            "INSERT OR IGNORE INTO chat_sessions (id) VALUES (?)",
            (session_id,),
        )
        # Insert message
        await db.execute(
            """
            INSERT INTO chat_messages (id, session_id, role, content, provider)
            VALUES (?, ?, ?, ?, ?)
            """,
            (message_id, session_id, role, content, provider),
        )
        # Touch session
        await db.execute(
            "UPDATE chat_sessions SET last_updated = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,),
        )
        await db.commit()
    return message_id


async def get_chat_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Return up to `limit` messages for a session in chronological order.
    """
    async with _open_db() as db:
        query = """
            SELECT role, content, timestamp, provider
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        async with db.execute(query, (session_id, limit)) as cur:
            rows = await cur.fetchall()

    # rows come back newest->oldest; reverse to chronological
    rows = list(rows)[::-1]
    return [
        {
            "role": r["role"],
            "content": r["content"],
            "timestamp": r["timestamp"],
            "provider": r["provider"],
        }
        for r in rows
    ]


# ------------------ NetCDF metadata ------------------

async def save_netcdf_metadata(
    filename: str,
    file_size: int,
    metadata: Dict[str, Any],
) -> str:
    """
    Persist NetCDF file metadata (dimensions/variables/global_attributes).
    Returns generated file_id.
    """
    file_id = str(uuid.uuid4())

    # Compact JSON to keep rows small; UTF-8 friendly
    dims_json = json.dumps(metadata.get("dimensions", {}), ensure_ascii=False, separators=(",", ":"))
    vars_json = json.dumps(metadata.get("variables", {}), ensure_ascii=False, separators=(",", ":"))
    gatt_json = json.dumps(metadata.get("global_attributes", {}), ensure_ascii=False, separators=(",", ":"))

    async with _open_db() as db:
        await db.execute(
            """
            INSERT INTO netcdf_files
                (id, filename, file_size, dimensions, variables, global_attributes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (file_id, filename, file_size, dims_json, vars_json, gatt_json),
        )
        await db.commit()
    return file_id


async def get_netcdf_files() -> List[Dict[str, Any]]:
    """
    List uploaded NetCDF files (most recent first).
    """
    async with _open_db() as db:
        query = """
            SELECT id, filename, upload_timestamp, file_size, processed
            FROM netcdf_files
            ORDER BY upload_timestamp DESC
        """
        async with db.execute(query) as cur:
            rows = await cur.fetchall()

    return [
        {
            "id": r["id"],
            "filename": r["filename"],
            "upload_timestamp": r["upload_timestamp"],
            "file_size": r["file_size"],
            "processed": bool(r["processed"]),
        }
        for r in rows
    ]


# ------------------ extras ------------------

async def mark_file_processed(file_id: str, processed: bool = True) -> None:
    """
    Flip the processed flag for a NetCDF file.
    """
    async with _open_db() as db:
        await db.execute(
            "UPDATE netcdf_files SET processed = ? WHERE id = ?",
            (1 if processed else 0, file_id),
        )
        await db.commit()


async def get_netcdf_metadata(file_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch dimensions/variables/global_attributes JSON for one file.
    """
    async with _open_db() as db:
        query = """
            SELECT dimensions, variables, global_attributes
            FROM netcdf_files
            WHERE id = ?
            LIMIT 1
        """
        async with db.execute(query, (file_id,)) as cur:
            row = await cur.fetchone()

    if not row:
        return None
    return {
        "dimensions": json.loads(row["dimensions"]),
        "variables": json.loads(row["variables"]),
        "global_attributes": json.loads(row["global_attributes"]),
    }
