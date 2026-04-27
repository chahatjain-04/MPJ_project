"""
SQLAlchemy ORM Models
=====================
Defines the database schema using SQLAlchemy's declarative ORM.
These models map directly to PostgreSQL tables and are used by
the DatabaseService for all CRUD operations.

Tables:
  - criminals: Stores criminal records with face embeddings
  - detection_alerts: Logs every detection/recognition event
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# =============================================================================
# Base Class
# =============================================================================

class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# =============================================================================
# Criminal Model
# =============================================================================

class Criminal(Base):
    """
    Represents a criminal record in the database.

    The 'embedding' column stores a JSON-serialized array of 512 floats
    (ArcFace face embedding). The 'lbp_histogram' column stores LBP
    feature histograms as a JSON string for disguise-based partial matching.
    """
    __tablename__ = "criminals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    crime = Column(Text, nullable=False)
    embedding = Column(
        Text, nullable=False,
        comment="ArcFace face embedding (512-dim float array as JSON string)"
    )
    lbp_histogram = Column(
        Text, nullable=True,
        comment="LBP features per region for disguise matching (JSON string)"
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationship: one criminal can have many detection alerts
    alerts = relationship(
        "DetectionAlert",
        back_populates="criminal",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Criminal(id={self.id}, name='{self.name}')>"


# =============================================================================
# Detection Alert Model
# =============================================================================

class DetectionAlert(Base):
    """
    Logs a detection event — when a face is identified (or suspected)
    during real-time monitoring.

    Stores the matched criminal (if any), confidence score, whether
    the person appeared disguised, and an optional base64-encoded
    image snapshot for audit/review.
    """
    __tablename__ = "detection_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    criminal_id = Column(
        Integer,
        ForeignKey("criminals.id", ondelete="SET NULL"),
        nullable=True,
    )
    confidence = Column(Float, nullable=False, default=0.0)
    is_disguised = Column(Boolean, nullable=False, default=False)
    image_data = Column(
        Text, nullable=True,
        comment="Base64-encoded JPEG snapshot of the detected face"
    )
    detected_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # Relationship back to the matched criminal
    criminal = relationship("Criminal", back_populates="alerts")

    def __repr__(self) -> str:
        return (
            f"<DetectionAlert(id={self.id}, criminal_id={self.criminal_id}, "
            f"confidence={self.confidence:.4f})>"
        )
