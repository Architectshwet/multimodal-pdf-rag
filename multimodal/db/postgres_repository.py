from typing import Any

from multimodal.db.db_singleton import DatabaseSingleton


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversation_history (
    id BIGSERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversation_history_conversation_id
ON conversation_history (conversation_id, created_at DESC);
"""

MIGRATE_DATASET_SQL = """
ALTER TABLE conversation_history
ADD COLUMN IF NOT EXISTS dataset_id TEXT;

ALTER TABLE conversation_history
ALTER COLUMN dataset_id SET DEFAULT 'default';

UPDATE conversation_history
SET dataset_id = 'default'
WHERE dataset_id IS NULL;
"""


async def initialize_conversation_table() -> None:
    pool = await DatabaseSingleton.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLE_SQL)
        await conn.execute(MIGRATE_DATASET_SQL)


async def append_message(
    conversation_id: str,
    role: str,
    content: str,
) -> dict[str, Any]:
    pool = await DatabaseSingleton.get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO conversation_history (conversation_id, role, content)
            VALUES ($1, $2, $3)
            RETURNING id, conversation_id, role, created_at
            """,
            conversation_id,
            role,
            content,
        )

    return {
        "id": int(row["id"]),
        "conversation_id": row["conversation_id"],
        "role": row["role"],
        "created_at": row["created_at"].isoformat(),
    }


async def get_history(conversation_id: str, limit: int) -> list[dict[str, Any]]:
    pool = await DatabaseSingleton.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, content, created_at
            FROM conversation_history
            WHERE conversation_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            conversation_id,
            limit,
        )

    ordered = list(reversed(rows))
    return [dict(row) for row in ordered]
