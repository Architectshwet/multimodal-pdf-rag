import asyncio

import asyncpg

from multimodal.config import config
from multimodal.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseSingleton:
    _pool: asyncpg.Pool | None = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        if cls._pool is None:
            async with cls._lock:
                if cls._pool is None:
                    postgres_url = config.postgres_url.strip()
                    if not postgres_url:
                        raise RuntimeError("POSTGRESQL_URL is required for database persistence")

                    logger.info("Creating PostgreSQL connection pool using POSTGRESQL_URL")
                    cls._pool = await asyncpg.create_pool(
                        dsn=postgres_url,
                        min_size=1,
                        max_size=10,
                        command_timeout=30,
                    )
                    logger.info("PostgreSQL connection pool created successfully")
        return cls._pool

    @classmethod
    async def close(cls) -> None:
        if cls._pool is not None:
            await cls._pool.close()
            cls._pool = None
            logger.info("PostgreSQL connection pool closed")
