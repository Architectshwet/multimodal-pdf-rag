import json
import uuid
from typing import Any

from multimodal.config import config
from multimodal.services.openai_service import OpenAIService
from multimodal.services.postgres_db_service import PostgresDBService
from multimodal.services.qdrant_service import QdrantService
from multimodal.utils.logger import get_logger

logger = get_logger(__name__)


class ChatService:
    def __init__(
        self,
        db_service: PostgresDBService,
        qdrant_service: QdrantService,
        openai_service: OpenAIService,
    ) -> None:
        self.db_service = db_service
        self.qdrant_service = qdrant_service
        self.openai_service = openai_service

    async def stream_chat(
        self,
        message: str,
        thread_id: str | None,
    ):
        conversation_id = thread_id.strip() if thread_id else f"rag-thread-{uuid.uuid4().hex}"
        logger.info(
            "stream_chat start | conversation_id=%s | message=%s",
            conversation_id,
            message,
        )

        try:
            await self.db_service.append_conversation_message(
                conversation_id=conversation_id,
                role="user",
                content=message,
            )

            history = await self.db_service.get_history(conversation_id=conversation_id)
            if len(history) <= 1:
                rewritten = message
            else:
                rewritten = await self.openai_service.rewrite_query(latest_question=message, history=history)
            logger.info(
                "stream_chat query rewritten | conversation_id=%s | rewritten_query=%s",
                conversation_id,
                rewritten,
            )

            docs = self.qdrant_service.similarity_search(query=rewritten, k=config.retrieval_k)
            image_paths: list[str] = []
            sources: list[dict[str, Any]] = []

            for doc in docs:
                source = doc.metadata.get("source")
                page_number = doc.metadata.get("page_number")
                image_path = doc.metadata.get("image_path")

                if image_path and image_path not in image_paths:
                    image_paths.append(image_path)

                sources.append(
                    {
                        "source": source,
                        "page_number": page_number,
                        "image_path": image_path,
                    }
                )
            yield self._event(
                {
                    "type": "session",
                    "thread_id": conversation_id,
                }
            )
            yield self._event(
                {
                    "type": "retrieval",
                    "thread_id": conversation_id,
                    "rewritten_query": rewritten,
                    "sources": sources,
                }
            )

            if not image_paths:
                fallback = "No indexed PDF pages found. Please call POST /index first."
                await self.db_service.append_conversation_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=fallback,
                )
                yield self._event(
                    {
                        "type": "end_of_response",
                        "thread_id": conversation_id,
                        "content": fallback,
                        "sources": [],
                    }
                )
                return

            answer = ""
            async for token in self.openai_service.stream_answer_from_images(
                question=message,
                rewritten_query=rewritten,
                image_paths=image_paths[:4],
            ):
                answer += token
                yield self._event(
                    {
                        "type": "token",
                        "thread_id": conversation_id,
                        "content": token,
                    }
                )

            final_answer = answer.strip() or "I could not generate an answer from retrieved PDF pages."
            await self.db_service.append_conversation_message(
                conversation_id=conversation_id,
                role="assistant",
                content=final_answer,
            )
            logger.info(
                "stream_chat completed | conversation_id=%s | final_answer=%s",
                conversation_id,
                final_answer,
            )

            yield self._event(
                {
                    "type": "end_of_response",
                    "thread_id": conversation_id,
                    "content": final_answer,
                    "sources": sources,
                }
            )

        except Exception as exc:
            logger.exception("stream_chat failed | conversation_id=%s", conversation_id)
            yield self._event(
                {
                    "type": "error",
                    "thread_id": conversation_id,
                    "error": str(exc),
                }
            )

    @staticmethod
    def _event(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
