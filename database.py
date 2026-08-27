from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path


def database_path() -> Path:
    return Path(os.getenv("DATABASE_PATH", "data/safetyvision.db"))


def _connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def create_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                environment TEXT NOT NULL,
                risk_count INTEGER NOT NULL,
                highest_severity TEXT,
                provider TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_analysis(
    filename: str,
    environment: str,
    result: dict,
    provider: str = "huggingface",
) -> int:
    summary = result.get("summary", {})
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analysis_results
            (filename, environment, risk_count, highest_severity, provider, result_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                environment,
                int(summary.get("risk_count", 0)),
                summary.get("highest_severity"),
                provider,
                json.dumps(result, ensure_ascii=False),
            ),
        )
        return int(cursor.lastrowid)


def get_analyses(limit: int = 50) -> list[dict]:
    safe_limit = min(max(int(limit), 1), 100)
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, filename, environment, risk_count, highest_severity,
                   provider, created_at
            FROM analysis_results
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_analysis(analysis_id: int) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, filename, environment, provider, result_json, created_at
            FROM analysis_results
            WHERE id = ?
            """,
            (analysis_id,),
        ).fetchone()

    if row is None:
        return None

    result = json.loads(row["result_json"])
    result["analysis_id"] = row["id"]
    result["provider"] = row["provider"]
    result["created_at"] = row["created_at"]
    return result

