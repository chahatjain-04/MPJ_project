"""
Criminal Face Detection & Identification System — FastAPI Backend
=================================================================
Main application entry point. Defines all REST API endpoints and
orchestrates the face detection, recognition, and database services.

Endpoints:
  POST /detect      → Detect faces in an image, return bounding boxes
  POST /recognize   → Identify faces against the criminal database
  POST /add-criminal → Register a new criminal with face image
  GET  /alerts      → Retrieve recent detection alerts

Run with:
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import logging
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    CORS_ORIGINS,
    RECOGNITION_SIMILARITY_THRESHOLD,
    DISGUISE_RECOGNITION_THRESHOLD,
    STRONG_MATCH_THRESHOLD,
    MAX_ALERTS_LIMIT,
)
from backend.schemas import (
    ImageRequest,
    DetectionResponse,
    BoundingBox,
    RecognitionResponse,
    RecognitionResult,
    AddCriminalRequest,
    AddCriminalResponse,
    AlertsListResponse,
    AlertResponse,
)
from backend.services.face_detector import FaceDetector
from backend.services.face_recognizer import FaceRecognizer
from backend.services.disguise_handler import DisguiseHandler
from backend.services.database import DatabaseService

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Service Instances (initialized at startup)
# =============================================================================
face_detector: FaceDetector = None
face_recognizer: FaceRecognizer = None
disguise_handler: DisguiseHandler = None
db_service: DatabaseService = None


# =============================================================================
# Application Lifespan (startup + shutdown)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle:
      - Startup: Load all AI models and connect to the database.
      - Shutdown: Close the database connection pool.
    """
    global face_detector, face_recognizer, disguise_handler, db_service

    # --- Startup ---
    logger.info("=" * 60)
    logger.info("Starting Criminal Face Detection Backend...")
    logger.info("=" * 60)

    # Initialize AI models
    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()
    disguise_handler = DisguiseHandler()

    # Connect to PostgreSQL
    db_service = DatabaseService()
    await db_service.connect()

    logger.info("All services initialized. Server is ready.")
    logger.info("=" * 60)

    yield  # Application runs here

    # --- Shutdown ---
    logger.info("Shutting down...")
    await db_service.disconnect()
    logger.info("Shutdown complete.")


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="Criminal Face Detection & Identification API",
    description=(
        "Real-time criminal face detection using OpenCV DNN, "
        "ArcFace R100 recognition, and PostgreSQL-backed similarity search."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware to allow requests from the Java frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================

def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64-encoded image string into an OpenCV image (numpy array).

    Supports images with or without the data URI prefix
    (e.g., 'data:image/jpeg;base64,...').

    Args:
        base64_string: Base64-encoded image data.

    Returns:
        BGR image as a numpy array.

    Raises:
        HTTPException: If the image cannot be decoded.
    """
    try:
        # Strip data URI prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        # Decode base64 → bytes → numpy array → OpenCV image
        image_bytes = base64.b64decode(base64_string)
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("cv2.imdecode returned None")

        return image

    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise HTTPException(
            status_code=400, detail=f"Invalid image data: {str(e)}"
        )


# =============================================================================
# API Endpoints
# =============================================================================


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_faces(request: ImageRequest):
    """
    Detect faces in an image.

    Accepts a base64-encoded image and returns bounding boxes for all
    detected faces with their confidence scores.

    This endpoint only performs detection (locating faces), not
    identification (recognizing who they are). Use /recognize for that.
    """
    # Decode the base64 image
    image = decode_base64_image(request.image)

    # Run face detection using OpenCV DNN
    detections = face_detector.detect_faces(image)

    # Build response (exclude face_crop from the response)
    faces = [
        BoundingBox(
            x=d["x"],
            y=d["y"],
            width=d["width"],
            height=d["height"],
            confidence=d["confidence"],
        )
        for d in detections
    ]

    return DetectionResponse(faces=faces, count=len(faces))


@app.post("/recognize", response_model=RecognitionResponse, tags=["Recognition"])
async def recognize_faces(request: ImageRequest):
    """
    Detect and identify faces in an image against the criminal database.

    Pipeline for each detected face:
      1. Generate ArcFace embedding (512-dim vector)
      2. Search the database using in-memory cosine similarity
      3. If low confidence → check for disguise
      4. If disguised → attempt LBP-based partial matching
      5. Log alert if a criminal is identified
      6. Return identity, confidence, and disguise status

    This is the primary endpoint used by the Java frontend for
    real-time monitoring.
    """
    image = decode_base64_image(request.image)

    # Step 1: Detect all faces in the image
    detections = face_detector.detect_faces(image)

    results = []

    for det in detections:
        face_crop = det["face_crop"]
        name = "Unknown"
        label = "unknown"   # Will become "criminal" on any confirmed match
        confidence = 0.0
        is_disguised = False
        criminal_id = None

        # -----------------------------------------------------------------
        # Step 2: Generate ArcFace embedding for this face
        # -----------------------------------------------------------------
        try:
            embedding = face_recognizer.get_embedding(face_crop)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            results.append(
                RecognitionResult(
                    name="Unknown",
                    label="unknown",
                    confidence=0.0,
                    is_disguised=False,
                    bounding_box=BoundingBox(
                        x=det["x"],
                        y=det["y"],
                        width=det["width"],
                        height=det["height"],
                        confidence=det["confidence"],
                    ),
                )
            )
            continue

        # -----------------------------------------------------------------
        # Step 3: ALWAYS check for disguise (independent of match result)
        # This runs on every face so that even matched criminals show
        # the correct disguise status.
        # -----------------------------------------------------------------
        is_disguised = disguise_handler.is_disguised(face_crop)

        # -----------------------------------------------------------------
        # Step 4: Search the database for matching embeddings
        # First pass: use the standard (stricter) threshold.
        # -----------------------------------------------------------------
        matches = await db_service.search_similar(
            embedding, threshold=RECOGNITION_SIMILARITY_THRESHOLD
        )

        if matches:
            # Best match found via ArcFace embeddings (strict threshold)
            best = matches[0]
            name = best["name"]
            confidence = best["similarity"]
            criminal_id = best["id"]
            label = "criminal"
            logger.info(
                f"MATCH: '{name}' (confidence={confidence:.4f}, "
                f"disguised={is_disguised})"
            )
        elif is_disguised:
            # ---------------------------------------------------------
            # Step 4b: Strict ArcFace failed AND face is disguised.
            # Give ArcFace a second chance with the lower disguise
            # threshold before falling back to LBP.
            # ---------------------------------------------------------
            logger.info(
                "Face DISGUISED, retrying ArcFace with lower threshold "
                f"({DISGUISE_RECOGNITION_THRESHOLD})..."
            )
            disguise_matches = await db_service.search_similar(
                embedding, threshold=DISGUISE_RECOGNITION_THRESHOLD
            )

            if disguise_matches:
                best = disguise_matches[0]
                name = best["name"] + " (disguised)"
                confidence = best["similarity"]
                criminal_id = best["id"]
                label = "criminal"
                logger.info(
                    f"DISGUISE-THRESHOLD MATCH: '{name}' "
                    f"(confidence={confidence:.4f})"
                )
            else:
                # ---------------------------------------------------------
                # Step 4c: Both ArcFace attempts failed → try LBP
                # ---------------------------------------------------------
                logger.info("ArcFace exhausted. Attempting LBP matching...")

                lbp_features = disguise_handler.extract_lbp_features(face_crop)

                if lbp_features:
                    stored_lbp = await db_service.get_all_lbp_features()
                    lbp_matches = disguise_handler.partial_match(
                        lbp_features, stored_lbp
                    )

                    if lbp_matches:
                        best_lbp = lbp_matches[0]
                        name = best_lbp["name"] + " (partial)"
                        confidence = best_lbp["similarity"]
                        criminal_id = best_lbp["id"]
                        label = "criminal"
                        logger.info(
                            f"LBP MATCH: '{name}' "
                            f"(similarity={confidence:.4f}, "
                            f"regions={best_lbp['matched_regions']})"
                        )

        # -----------------------------------------------------------------
        # Step 5: Log alert if criminal identified
        # -----------------------------------------------------------------
        if criminal_id is not None:
            try:
                _, buffer = cv2.imencode(".jpg", face_crop)
                face_b64 = base64.b64encode(buffer).decode("utf-8")

                await db_service.log_alert(
                    criminal_id=criminal_id,
                    confidence=confidence,
                    is_disguised=is_disguised,
                    image_data=face_b64,
                )
            except Exception as e:
                logger.error(f"Failed to log alert: {e}")

        # -----------------------------------------------------------------
        # Step 6: Build result for this face
        # label = "criminal" → frontend displays "Yes"
        # label = "unknown"  → frontend displays "No"
        #
        # Disguise detection uses TWO signals combined with OR:
        #   1. Skin-color occlusion (works for masks, scarves, cloth)
        #   2. Confidence drop below STRONG_MATCH_THRESHOLD
        #      (works for ANY material including hands, which are
        #       skin-colored and defeat the skin detector)
        # An unknown face always gets is_disguised=False.
        # -----------------------------------------------------------------
        if label == "criminal":
            confidence_drop_disguise = confidence < STRONG_MATCH_THRESHOLD
            report_disguised = is_disguised or confidence_drop_disguise
            if confidence_drop_disguise and not is_disguised:
                logger.info(
                    f"Disguise detected via confidence drop: {confidence:.4f} "
                    f"< {STRONG_MATCH_THRESHOLD} (skin detector missed it)"
                )
        else:
            report_disguised = False

        results.append(
            RecognitionResult(
                name=name,
                label=label,
                confidence=round(confidence, 4),
                is_disguised=report_disguised,
                bounding_box=BoundingBox(
                    x=det["x"],
                    y=det["y"],
                    width=det["width"],
                    height=det["height"],
                    confidence=det["confidence"],
                ),
            )
        )

    return RecognitionResponse(results=results)


@app.post("/add-criminal", response_model=AddCriminalResponse, tags=["Manage"])
async def add_criminal(request: AddCriminalRequest):
    """
    Add a new criminal to the database.

    Accepts a name, crime description, and a face image. The system:
      1. Detects the face in the image
      2. Generates the ArcFace embedding
      3. Extracts LBP features for future disguise matching
      4. Stores everything in the database

    Only one face should be in the image; if multiple are detected,
    the largest face is used.
    """
    # Decode and validate the image
    image = decode_base64_image(request.image)

    # Detect faces
    detections = face_detector.detect_faces(image)

    if not detections:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the provided image. "
                   "Please provide a clear frontal face photo.",
        )

    # Use the largest face if multiple are detected
    if len(detections) > 1:
        detections.sort(
            key=lambda d: d["width"] * d["height"], reverse=True
        )
        logger.warning(
            f"Multiple faces detected ({len(detections)}). "
            f"Using the largest face."
        )

    face_crop = detections[0]["face_crop"]

    # Generate ArcFace embedding
    try:
        embedding = face_recognizer.get_embedding(face_crop)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate face embedding: {str(e)}",
        )

    # Extract LBP features for disguise matching
    lbp_features = disguise_handler.extract_lbp_features(face_crop)

    # Store in database
    try:
        criminal_id = await db_service.add_criminal(
            name=request.name,
            crime=request.crime,
            embedding=embedding,
            lbp_histogram=lbp_features if lbp_features else None,
        )
    except Exception as e:
        logger.error(f"Database insert failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store criminal record: {str(e)}",
        )

    return AddCriminalResponse(
        success=True,
        message=f"Criminal '{request.name}' added successfully.",
        criminal_id=criminal_id,
    )


@app.get("/alerts", response_model=AlertsListResponse, tags=["Alerts"])
async def get_alerts(limit: int = 20):
    """
    Retrieve recent detection alerts.

    Returns the most recent detection events, including:
      - Matched criminal name (if identified)
      - Confidence score
      - Whether disguise was detected
      - Timestamp of detection

    Use the 'limit' query parameter to control how many alerts to return
    (default: 20, max: configured in MAX_ALERTS_LIMIT).
    """
    # Clamp limit to configured maximum
    limit = min(limit, MAX_ALERTS_LIMIT)

    try:
        alerts_data = await db_service.get_recent_alerts(limit=limit)
    except Exception as e:
        logger.error(f"Failed to fetch alerts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve alerts: {str(e)}",
        )

    alerts = [
        AlertResponse(
            id=a["id"],
            criminal_name=a["criminal_name"],
            confidence=a["confidence"],
            is_disguised=a["is_disguised"],
            detected_at=a["detected_at"],
        )
        for a in alerts_data
    ]

    return AlertsListResponse(alerts=alerts, total=len(alerts))


@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "Criminal Face Detection API"}
