from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "runtime_data"
DB_PATH = DATA_DIR / "promptshield.sqlite3"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            prompt TEXT NOT NULL,
            classification TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_score INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            severity TEXT NOT NULL,
            result_json TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def save_analysis(prompt: str, result: dict[str, Any]) -> str:
    analysis_id = uuid.uuid4().hex
    created_at = utc_now()
    saved_result = dict(result)
    saved_result["id"] = analysis_id
    saved_result["created_at"] = created_at

    with connect() as connection:
        connection.execute(
            """
            INSERT INTO analyses (
                id, created_at, prompt, classification, confidence,
                risk_score, risk_level, severity, result_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                created_at,
                prompt,
                result["classification"],
                float(result["confidence"]),
                int(result["risk_score"]),
                result["risk_level"],
                result["severity"],
                json.dumps(saved_result, ensure_ascii=True),
            ),
        )
        connection.commit()
    return analysis_id


def get_analysis(analysis_id: str) -> dict[str, Any] | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT prompt, result_json FROM analyses WHERE id = ?",
            (analysis_id,),
        ).fetchone()
    if row is None:
        return None
    result = json.loads(str(row["result_json"]))
    result["prompt"] = str(row["prompt"])
    return result


def history(limit: int = 50) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 200))
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, prompt, classification, confidence,
                   risk_score, risk_level, severity
            FROM analyses
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "prompt": row["prompt"],
            "classification": row["classification"],
            "confidence": row["confidence"],
            "risk_score": row["risk_score"],
            "risk_level": row["risk_level"],
            "severity": row["severity"],
        }
        for row in rows
    ]


def stats() -> dict[str, Any]:
    with connect() as connection:
        total = connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        safe = connection.execute(
            "SELECT COUNT(*) FROM analyses WHERE classification = 'Safe Prompt'"
        ).fetchone()[0]
        threats = total - safe
        high_risk = connection.execute(
            "SELECT COUNT(*) FROM analyses WHERE risk_score >= 65"
        ).fetchone()[0]
        by_type = connection.execute(
            """
            SELECT classification, COUNT(*) AS count
            FROM analyses
            GROUP BY classification
            ORDER BY count DESC
            """
        ).fetchall()
        trend = connection.execute(
            """
            SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS count,
                   SUM(CASE WHEN classification = 'Safe Prompt' THEN 0 ELSE 1 END)
                   AS threats
            FROM analyses
            GROUP BY day
            ORDER BY day ASC
            """
        ).fetchall()
    return {
        "total": int(total),
        "safe": int(safe),
        "threats": int(threats),
        "high_risk": int(high_risk),
        "by_type": [
            {"label": row["classification"], "count": int(row["count"])}
            for row in by_type
        ],
        "trend": [
            {
                "day": row["day"],
                "count": int(row["count"]),
                "threats": int(row["threats"] or 0),
            }
            for row in trend
        ],
    }


def clear_history() -> None:
    with connect() as connection:
        connection.execute("DELETE FROM analyses")
        connection.commit()
