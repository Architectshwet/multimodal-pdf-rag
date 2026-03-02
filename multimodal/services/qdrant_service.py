"""Qdrant service for storing and searching multimodal PDF page embeddings."""

import logging
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from multimodal.config import config

logger = logging.getLogger(__name__)


class QdrantService:
    """Service for interacting with Qdrant vector database."""

    def __init__(self) -> None:
        self.path = str(config.qdrant_path)
        self.collection_name = config.qdrant_collection_name
        self.embedding_model = config.embedding_model

        Path(self.path).mkdir(parents=True, exist_ok=True)

        self.client = QdrantClient(path=self.path)
        self.embedder = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=config.openai_api_key,
        )
        self._initialize_collection()

    def _initialize_collection(self) -> None:
        collections = self.client.get_collections().collections
        names = [col.name for col in collections]
        if self.collection_name not in names:
            vector_size = len(self.embedder.embed_query("multimodal pdf page"))
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", self.collection_name)

    def clear_collection(self) -> None:
        """Delete the collection and recreate it empty."""
        try:
            collections = self.client.get_collections().collections
            names = [col.name for col in collections]
            if self.collection_name in names:
                self.client.delete_collection(collection_name=self.collection_name)
                logger.info("Deleted Qdrant collection: %s", self.collection_name)
        except Exception as exc:
            logger.warning("Collection delete check failed: %s", exc)
        self._initialize_collection()

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return

        self._initialize_collection()

        texts = [doc.page_content for doc in documents]
        embeddings = self.embedder.embed_documents(texts)

        points: list[PointStruct] = []
        for i, doc in enumerate(documents):
            payload: dict[str, Any] = dict(doc.metadata or {})
            payload["document"] = doc.page_content
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i],
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def similarity_search(self, query: str, k: int) -> list[Document]:
        self._initialize_collection()
        query_vector = self.embedder.embed_query(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
        ).points

        docs: list[Document] = []
        for result in results:
            payload = dict(result.payload or {})
            page_content = str(payload.pop("document", ""))
            docs.append(Document(page_content=page_content, metadata=payload))
        return docs

    def get_index_stats(self) -> dict[str, Any]:
        self._initialize_collection()
        details = self.client.get_collection(self.collection_name)
        vectors_config = details.config.params.vectors
        vector_size = getattr(vectors_config, "size", None)
        distance = getattr(vectors_config, "distance", None)
        return {
            "collection_name": self.collection_name,
            "qdrant_path": str(Path(self.path).resolve()),
            "total_docs": details.points_count,
            "vector_size": vector_size,
            "distance": str(distance) if distance is not None else None,
        }

    def get_page_document(self, page_number: int) -> dict[str, Any] | None:
        self._initialize_collection()
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="page_number",
                    match=MatchValue(value=page_number),
                )
            ]
        )
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None

        point = points[0]
        payload = dict(point.payload or {})
        page_content = str(payload.pop("document", ""))
        metadata: dict[str, Any] = {
            **payload,
            "point_id": str(point.id),
        }
        return {
            "page_number": page_number,
            "page_content": page_content,
            "metadata": metadata,
        }

@lru_cache
def get_qdrant_service() -> QdrantService:
    return QdrantService()
