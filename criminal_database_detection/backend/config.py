"""
Configuration Module
====================
Central configuration for the Criminal Face Detection backend.
All settings can be overridden via environment variables.
"""

import os

# =============================================================================
# Database Configuration (PostgreSQL)
# =============================================================================
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "criminal_detection")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "0704")

# SQLAlchemy async connection URL for asyncpg
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)

# =============================================================================
# Model File Paths
# =============================================================================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# OpenCV DNN Face Detector (SSD-based)
FACE_DETECTOR_PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_DETECTOR_CAFFEMODEL = os.path.join(
    MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel"
)

# ArcFace R100 recognition model (ONNX format)
ARCFACE_MODEL_PATH = os.path.join(MODEL_DIR, "arcface_r100.onnx")

# =============================================================================
# Detection & Recognition Thresholds
# =============================================================================

# Minimum confidence score for the SSD face detector to consider a detection valid.
# Faces with confidence below this value are discarded.
FACE_DETECTION_CONFIDENCE = float(os.getenv("FACE_DETECTION_CONFIDENCE", "0.5"))

# Minimum cosine similarity between embeddings to consider a match.
# Lowered from 0.6 → 0.45 to handle slight appearance variations such as
# lighting changes, mild disguises (glasses, hat, etc.).
RECOGNITION_SIMILARITY_THRESHOLD = float(
    os.getenv("RECOGNITION_SIMILARITY_THRESHOLD", "0.45")
)

# Even lower threshold used when the face is already flagged as disguised.
# Allows ArcFace to still match criminals with heavier occlusion (mask, scarf).
# Set higher than 0.30 to avoid too many false positives on unknowns.
DISGUISE_RECOGNITION_THRESHOLD = float(
    os.getenv("DISGUISE_RECOGNITION_THRESHOLD", "0.35")
)

# Confidence below this value means the criminal was matched but the face was
# PARTIALLY COVERED (disguised) — hand, mask, cloth, etc.
# Typical clear-face ArcFace scores: 0.70–0.95.  Covered face: 0.45–0.69.
STRONG_MATCH_THRESHOLD = float(
    os.getenv("STRONG_MATCH_THRESHOLD", "0.70")
)


# =============================================================================
# OpenCV DNN Parameters
# =============================================================================

# Expected input size for the SSD face detection network
DNN_INPUT_SIZE = (300, 300)

# Mean pixel values subtracted from the input image (BGR order)
DNN_MEAN_VALUES = (104.0, 177.0, 123.0)

# Scale factor applied to pixel values after mean subtraction
DNN_SCALE_FACTOR = 1.0

# =============================================================================
# ArcFace Model Parameters
# =============================================================================

# Input face crop size expected by ArcFace R100
ARCFACE_INPUT_SIZE = (112, 112)

# Embedding dimensionality produced by ArcFace R100
EMBEDDING_DIM = 512

# =============================================================================
# LBP (Local Binary Patterns) Parameters for Disguise Detection
# =============================================================================

# Radius of the circular LBP operator (larger = captures coarser texture)
LBP_RADIUS = 3

# Number of sampling points on the circle (should be 8 * radius for good coverage)
LBP_N_POINTS = 24

# Threshold for LBP-based partial matching (chi-squared distance)
LBP_MATCH_THRESHOLD = float(os.getenv("LBP_MATCH_THRESHOLD", "0.3"))

# =============================================================================
# API Settings
# =============================================================================

# Maximum number of alerts returned by the /alerts endpoint
MAX_ALERTS_LIMIT = 50

# CORS origins allowed to access the API (Java frontend typically runs on localhost)
CORS_ORIGINS = ["*"]
