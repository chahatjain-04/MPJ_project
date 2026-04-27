"""
Pydantic Schemas
================
Request and response models for all API endpoints.
These ensure strict validation and provide automatic OpenAPI documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# =============================================================================
# Shared Models
# =============================================================================

class BoundingBox(BaseModel):
    """Represents a detected face's bounding box location and confidence."""
    x: int = Field(..., description="Top-left x coordinate")
    y: int = Field(..., description="Top-left y coordinate")
    width: int = Field(..., description="Box width in pixels")
    height: int = Field(..., description="Box height in pixels")
    confidence: float = Field(..., description="Detection confidence (0-1)")


# =============================================================================
# /detect Endpoint
# =============================================================================

class ImageRequest(BaseModel):
    """Request containing a base64-encoded image."""
    image: str = Field(..., description="Base64-encoded JPEG image")


class DetectionResponse(BaseModel):
    """Response from the /detect endpoint."""
    faces: List[BoundingBox] = Field(default_factory=list, description="Detected faces")
    count: int = Field(..., description="Number of faces detected")


# =============================================================================
# /recognize Endpoint
# =============================================================================

class RecognitionResult(BaseModel):
    """Result for a single recognized face."""
    name: str = Field(..., description="Matched criminal name or 'Unknown'")
    label: str = Field(
        "unknown",
        description="'criminal' if matched in the database, 'unknown' otherwise"
    )
    confidence: float = Field(..., description="Cosine similarity score (0-1)")
    is_disguised: bool = Field(False, description="Whether disguise was detected")
    bounding_box: BoundingBox = Field(..., description="Face location in the image")


class RecognitionResponse(BaseModel):
    """Response from the /recognize endpoint."""
    results: List[RecognitionResult] = Field(
        default_factory=list, description="Recognition results for each face"
    )


# =============================================================================
# /add-criminal Endpoint
# =============================================================================

class AddCriminalRequest(BaseModel):
    """Request to add a new criminal to the database."""
    name: str = Field(..., min_length=1, description="Criminal's name")
    crime: str = Field(..., min_length=1, description="Description of the crime")
    image: str = Field(..., description="Base64-encoded face image")


class AddCriminalResponse(BaseModel):
    """Response after adding a criminal."""
    success: bool
    message: str
    criminal_id: Optional[int] = None


# =============================================================================
# /alerts Endpoint
# =============================================================================

class AlertResponse(BaseModel):
    """A single detection alert record."""
    id: int
    criminal_name: Optional[str] = Field(None, description="Name if matched, null if unknown")
    confidence: float
    is_disguised: bool
    detected_at: datetime


class AlertsListResponse(BaseModel):
    """Response from the /alerts endpoint."""
    alerts: List[AlertResponse] = Field(default_factory=list)
    total: int = Field(..., description="Total number of alerts returned")
