"""
Database Service (PostgreSQL + SQLAlchemy)
==========================================
Handles all PostgreSQL operations for the Criminal Face Detection system.
Uses SQLAlchemy async sessions with asyncpg for non-blocking database access,
which integrates naturally with FastAPI's async request handling.

Face embeddings are stored as JSON strings in a TEXT column. Cosine
similarity search is performed in Python using numpy against an in-memory
cache of all embeddings, which is fast (~1ms for 10K criminals).

Cache strategy:
  - On startup: load all criminal embeddings into memory
  - On add_criminal: insert into DB and append to cache
  - On search: compare query embedding against cached embeddings using numpy
  - Periodically refreshable if external DB changes are expected

The SQLAlchemy ORM models (Criminal, DetectionAlert) are defined in
backend/db_models.py and used for all insert/select operations.
"""

import numpy as np
import json
import logging
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from backend.database_engine import AsyncSessionLocal, init_db, dispose_engine
from backend.db_models import Criminal, DetectionAlert

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Async PostgreSQL database service for criminal records and detection alerts.
    Uses SQLAlchemy ORM with an in-memory embedding cache for fast
    cosine similarity search.
    """

    def __init__(self):
        """Initialize the service (cache populated in connect())."""
        # In-memory cache: list of dicts with 'id', 'name', 'crime', 'embedding' (np.array)
        self._embedding_cache = []

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self):
        """
        Initialize database tables (if they don't exist) and load the
        embedding cache from the database.
        """
        logger.info("Connecting to PostgreSQL database...")

        # Create tables from ORM models if they don't exist
        await init_db()

        logger.info("PostgreSQL connection established.")

        # Load all embeddings into memory for fast similarity search
        await self._load_embedding_cache()

    async def disconnect(self):
        """Dispose of the SQLAlchemy engine, closing all pooled connections."""
        await dispose_engine()
        logger.info("PostgreSQL connection pool closed.")

    # =========================================================================
    # Embedding Cache (In-Memory Vector Search)
    # =========================================================================

    async def _load_embedding_cache(self):
        """
        Load all criminal embeddings from the database into memory.

        This allows fast cosine similarity search without querying the DB
        on every frame. For a database of ~10K criminals, this uses roughly
        ~20 MB of RAM (10K × 512 floats × 4 bytes).
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Criminal.id, Criminal.name, Criminal.crime, Criminal.embedding)
            )
            rows = result.all()

        self._embedding_cache = []
        for row in rows:
            # Parse JSON embedding string → numpy array
            embedding_list = json.loads(row.embedding)
            emb = np.array(embedding_list, dtype=np.float32)

            # L2-normalize so dot product == cosine similarity in search_similar
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            self._embedding_cache.append({
                "id": row.id,
                "name": row.name,
                "crime": row.crime,
                "embedding": emb,
            })

        logger.info(
            f"Embedding cache loaded: {len(self._embedding_cache)} criminal(s)"
        )

    async def refresh_cache(self):
        """Reload the embedding cache from the database."""
        await self._load_embedding_cache()

    # =========================================================================
    # Criminal Records — Search
    # =========================================================================

    async def search_similar(
        self, embedding: np.ndarray, threshold: float = 0.6, limit: int = 5
    ) -> list:
        """
        Search for criminals with similar face embeddings using cosine similarity.

        This computation happens entirely in Python (not SQL). The in-memory
        cache makes this fast: ~1ms for 10K criminals on modern hardware.

        Cosine similarity formula:
            cos(A, B) = (A · B) / (||A|| × ||B||)
        
        Since our embeddings are L2-normalized (||A|| = ||B|| = 1):
            cos(A, B) = A · B  (just the dot product!)

        Args:
            embedding: Query face embedding as numpy array (512-dim, L2-normalized).
            threshold: Minimum cosine similarity to consider a match (0.0–1.0).
            limit: Maximum number of results to return.

        Returns:
            List of dicts with 'id', 'name', 'crime', 'similarity'.
            Sorted by similarity descending (best match first).
        """
        if not self._embedding_cache:
            return []

        # Ensure query embedding is L2-normalized
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        results = []

        for entry in self._embedding_cache:
            # Cosine similarity = dot product for L2-normalized vectors
            similarity = float(np.dot(embedding, entry["embedding"]))

            if similarity >= threshold:
                results.append({
                    "id": entry["id"],
                    "name": entry["name"],
                    "crime": entry["crime"],
                    "similarity": round(similarity, 4),
                })

        # Sort by similarity descending (best match first)
        results.sort(key=lambda r: r["similarity"], reverse=True)

        # Return top matches up to the limit
        return results[:limit]

    # =========================================================================
    # Criminal Records — Add
    # =========================================================================

    async def add_criminal(
        self,
        name: str,
        crime: str,
        embedding: np.ndarray,
        lbp_histogram: dict = None,
    ) -> int:
        """
        Insert a new criminal record with face embedding and optional LBP features.

        The embedding is stored as a JSON string of 512 floats in PostgreSQL.
        After insertion, the in-memory cache is updated immediately so the
        new criminal can be recognized without a restart.

        Args:
            name: Criminal's name.
            crime: Description of the crime committed.
            embedding: 512-dim ArcFace face embedding (numpy array).
            lbp_histogram: Optional dict of LBP histograms per face region.

        Returns:
            The auto-generated criminal ID.
        """
        # Convert numpy array to JSON-serializable string
        embedding_json = json.dumps(embedding.tolist())
        lbp_json = json.dumps(lbp_histogram) if lbp_histogram else None

        async with AsyncSessionLocal() as session:
            criminal = Criminal(
                name=name,
                crime=crime,
                embedding=embedding_json,
                lbp_histogram=lbp_json,
            )
            session.add(criminal)
            await session.commit()
            await session.refresh(criminal)
            criminal_id = criminal.id

        # Update the in-memory cache immediately (L2-normalize for cosine search)
        emb_copy = embedding.copy()
        norm = np.linalg.norm(emb_copy)
        if norm > 0:
            emb_copy = emb_copy / norm

        self._embedding_cache.append({
            "id": criminal_id,
            "name": name,
            "crime": crime,
            "embedding": emb_copy,
        })

        logger.info(
            f"Added criminal '{name}' with ID {criminal_id} "
            f"(cache size: {len(self._embedding_cache)})"
        )
        return criminal_id

    # =========================================================================
    # Criminal Records — LBP Features
    # =========================================================================

    async def get_all_lbp_features(self) -> list:
        """
        Retrieve all criminals with their LBP histograms for partial matching.

        Used by the disguise handler when ArcFace embedding matching fails.

        Returns:
            List of dicts with 'id', 'name', 'lbp_histogram'.
            Only entries that have LBP data are included.
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Criminal.id, Criminal.name, Criminal.lbp_histogram)
                .where(Criminal.lbp_histogram.isnot(None))
            )
            rows = result.all()

        results = []
        for row in rows:
            lbp_data = row.lbp_histogram
            if isinstance(lbp_data, str):
                lbp_data = json.loads(lbp_data)
            results.append({
                "id": row.id,
                "name": row.name,
                "lbp_histogram": lbp_data,
            })
        return results

    # =========================================================================
    # Detection Alerts
    # =========================================================================

    async def log_alert(
        self,
        criminal_id: int = None,
        confidence: float = 0.0,
        is_disguised: bool = False,
        image_data: str = None,
    ) -> int:
        """
        Log a detection alert to the database.

        Called whenever a criminal is identified (or a suspicious face
        is detected). The alert includes the matched criminal (if any),
        confidence score, and an optional image snapshot for audit.

        Args:
            criminal_id: ID of the matched criminal (None if unknown).
            confidence: Similarity/confidence score.
            is_disguised: Whether the face appeared disguised.
            image_data: Optional base64-encoded image snapshot.

        Returns:
            The auto-generated alert ID.
        """
        async with AsyncSessionLocal() as session:
            alert = DetectionAlert(
                criminal_id=criminal_id,
                confidence=confidence,
                is_disguised=is_disguised,
                image_data=image_data,
            )
            session.add(alert)
            await session.commit()
            await session.refresh(alert)
            alert_id = alert.id

        logger.info(
            f"Alert logged: criminal_id={criminal_id}, "
            f"confidence={confidence:.2f}, disguised={is_disguised}"
        )
        return alert_id

    async def get_recent_alerts(self, limit: int = 20) -> list:
        """
        Retrieve the most recent detection alerts, ordered newest first.

        Joins with the criminals table to include the name (if matched).

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of dicts with alert details including criminal name.
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(DetectionAlert)
                .options(joinedload(DetectionAlert.criminal))
                .order_by(DetectionAlert.detected_at.desc())
                .limit(limit)
            )
            alerts = result.scalars().all()

        return [
            {
                "id": alert.id,
                "criminal_name": alert.criminal.name if alert.criminal else None,
                "confidence": round(float(alert.confidence), 4),
                "is_disguised": bool(alert.is_disguised),
                "detected_at": alert.detected_at,
            }
            for alert in alerts
        ]
