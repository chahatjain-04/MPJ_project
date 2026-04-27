"""
Database Engine & Session Factory
==================================
Sets up the SQLAlchemy async engine (backed by asyncpg) and provides
a session factory for the rest of the application.

Usage:
    from backend.database_engine import async_engine, AsyncSessionLocal, init_db

    # At startup — create tables
    await init_db()

    # In request handlers — get a session
    async with AsyncSessionLocal() as session:
        ...
"""

import logging
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from backend.config import DATABASE_URL
from backend.db_models import Base

logger = logging.getLogger(__name__)

# =============================================================================
# Async Engine
# =============================================================================
# echo=False in production; set to True for SQL query logging during development
async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # Detect stale connections before use
)

# =============================================================================
# Session Factory
# =============================================================================
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# =============================================================================
# Table Creation
# =============================================================================

async def init_db():
    """
    Create all tables defined in db_models.py if they don't exist.
    Uses the async engine to run DDL statements.
    """
    logger.info("Initializing database tables...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized successfully.")


async def dispose_engine():
    """
    Dispose of the async engine, closing all pooled connections.
    Call this during application shutdown.
    """
    await async_engine.dispose()
    logger.info("Database engine disposed.")
