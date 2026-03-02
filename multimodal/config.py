import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    postgres_url: str = os.getenv("POSTGRESQL_URL", "").strip()

    chat_model: str = os.getenv("CHAT_MODEL", "gpt-5.1")
    vision_model: str = os.getenv("VISION_MODEL", "gpt-5-mini")
    rewrite_model: str = os.getenv("REWRITE_MODEL", "gpt-5.1")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))
    history_limit: int = int(os.getenv("HISTORY_LIMIT", "12"))

    uploads_dir: Path = Path(os.getenv("UPLOADS_DIR", "uploads"))
    page_images_dir: Path = Path(os.getenv("PAGE_IMAGES_DIR", "page_images"))
    qdrant_path: Path = Path(os.getenv("QDRANT_PATH", "qdrant_vectordb"))
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "multimodal_pdf_pages")

    cors_allowed_origins: list[str] | None = None

    def __post_init__(self) -> None:
        origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "*").strip()
        if origins_env:
            self.cors_allowed_origins = [origin.strip() for origin in origins_env.split(",") if origin.strip()]
        else:
            self.cors_allowed_origins = ["*"]


config = Config()
