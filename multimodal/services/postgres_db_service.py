from multimodal.config import config
from multimodal.db import postgres_repository
from multimodal.db.db_singleton import DatabaseSingleton


class PostgresDBService:
    def __init__(self) -> None:
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        await DatabaseSingleton.get_pool()
        await postgres_repository.initialize_conversation_table()
        self._initialized = True

    async def append_conversation_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> dict:
        if not self._initialized:
            return {"success": False, "error": "Service not initialized"}

        result = await postgres_repository.append_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )
        result["success"] = True
        return result

    async def get_history(self, conversation_id: str) -> list[dict]:
        if not self._initialized:
            return []
        return await postgres_repository.get_history(conversation_id, config.history_limit)

    async def close(self) -> None:
        if self._initialized:
            await DatabaseSingleton.close()
            self._initialized = False
