from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DB_PATH = Path("data/qa_audit.sqlite")
ASK_AUDIT_FIELDS = [
    "request_id",
    "created_at",
    "status",
    "question",
    "answer",
    "top_k",
    "use_rewrite",
    "rewritten_query",
    "used_queries_json",
    "retrieval_count",
    "sources_json",
    "distances_json",
    "citations_json",
    "confidence",
    "latency_ms",
    "fallback_reason",
    "error_type",
    "error_message",
]

logger = logging.getLogger(__name__)
JSON_TEXT_FIELDS = (
    "used_queries_json",
    "sources_json",
    "distances_json",
    "citations_json",
)
MIGRATION_COLUMNS = {
    "citations_json": "TEXT",
    "confidence": "TEXT",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def parse_json_text(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value

    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.text_factory = lambda value: value.decode("utf-8", errors="replace")
    conn.row_factory = sqlite3.Row
    return conn


def init_ask_audit_db() -> None:
    with _connect() as conn:
        conn.execute("PRAGMA encoding = 'UTF-8'")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ask_audit (
                request_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT,
                top_k INTEGER NOT NULL,
                use_rewrite INTEGER NOT NULL,
                rewritten_query TEXT,
                used_queries_json TEXT,
                retrieval_count INTEGER NOT NULL,
                sources_json TEXT,
                distances_json TEXT,
                citations_json TEXT,
                confidence TEXT,
                latency_ms INTEGER NOT NULL,
                fallback_reason TEXT,
                error_type TEXT,
                error_message TEXT
            )
            """
        )
        existing_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(ask_audit)").fetchall()
        }
        for column_name, column_type in MIGRATION_COLUMNS.items():
            if column_name not in existing_columns:
                conn.execute(
                    f"ALTER TABLE ask_audit ADD COLUMN {column_name} {column_type}"
                )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ask_audit_created_at "
            "ON ask_audit(created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ask_audit_status "
            "ON ask_audit(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ask_audit_fallback_reason "
            "ON ask_audit(fallback_reason)"
        )


def save_ask_audit(record: dict[str, Any]) -> None:
    payload = {
        "request_id": record["request_id"],
        "created_at": record.get("created_at") or utc_now_iso(),
        "status": record["status"],
        "question": record["question"],
        "answer": record.get("answer"),
        "top_k": int(record.get("top_k") or 0),
        "use_rewrite": 1 if record.get("use_rewrite") else 0,
        "rewritten_query": record.get("rewritten_query"),
        "used_queries_json": record.get("used_queries_json") or to_json([]),
        "retrieval_count": int(record.get("retrieval_count") or 0),
        "sources_json": record.get("sources_json") or to_json([]),
        "distances_json": record.get("distances_json") or to_json([]),
        "citations_json": record.get("citations_json") or to_json([]),
        "confidence": record.get("confidence"),
        "latency_ms": int(record.get("latency_ms") or 0),
        "fallback_reason": record.get("fallback_reason"),
        "error_type": record.get("error_type"),
        "error_message": record.get("error_message"),
    }
    placeholders = ", ".join(["?"] * len(ASK_AUDIT_FIELDS))

    with _connect() as conn:
        conn.execute(
            f"""
            INSERT INTO ask_audit ({", ".join(ASK_AUDIT_FIELDS)})
            VALUES ({placeholders})
            """,
            [payload[field] for field in ASK_AUDIT_FIELDS],
        )


def safe_save_ask_audit(record: dict[str, Any]) -> None:
    try:
        save_ask_audit(record)
    except Exception as exc:  # pragma: no cover - audit must not break ask
        logger.warning("Failed to save ask audit record: %s", exc)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["use_rewrite"] = bool(data["use_rewrite"])
    for field in JSON_TEXT_FIELDS:
        data[field] = parse_json_text(data.get(field))
    return data


def list_ask_audits(
    limit: int = 20,
    status: str | None = None,
    fallback_reason: str | None = None,
) -> list[dict[str, Any]]:
    where_parts: list[str] = []
    params: list[Any] = []

    if status:
        where_parts.append("status = ?")
        params.append(status)
    if fallback_reason:
        where_parts.append("fallback_reason = ?")
        params.append(fallback_reason)

    where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(
            f"""
            SELECT {", ".join(ASK_AUDIT_FIELDS)}
            FROM ask_audit
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


def get_ask_audit(request_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            f"""
            SELECT {", ".join(ASK_AUDIT_FIELDS)}
            FROM ask_audit
            WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()

    return _row_to_dict(row) if row else None
