# Criminal Face Detection & Identification System

> Real-time criminal face detection and identification system using OpenCV DNN, ArcFace R100, and PostgreSQL.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Step 1: Install PostgreSQL](#step-1-install-postgresql)
- [Step 2: Create the Database](#step-2-create-the-database)
- [Step 3: Download AI Models](#step-3-download-ai-models)
- [Step 4: Install Python Dependencies](#step-4-install-python-dependencies)
- [Step 5: Start the Backend](#step-5-start-the-backend)
- [Step 6: Test the System](#step-6-test-the-system)
- [API Endpoints](#api-endpoints)
- [Batch Import](#batch-import)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

This system detects and identifies criminal faces in real-time using:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Face Detection | OpenCV DNN (SSD ResNet-10) | Locate faces in images/video frames |
| Face Recognition | ArcFace R100 (ONNX) | Generate 512-dim face embeddings |
| Disguise Detection | LBP (Local Binary Patterns) | Partial matching for occluded faces |
| Database | PostgreSQL + SQLAlchemy ORM | Store criminal records & detection alerts |
| Backend API | FastAPI + asyncpg | REST API with async PostgreSQL access |
| Frontend | Java Swing + OpenCV | Webcam capture & GUI interface |

---

## Architecture

```
┌─────────────────────┐     HTTP/REST      ┌─────────────────────────────┐
│   Java Frontend     │ ──────────────────▶ │   FastAPI Backend           │
│   (Swing + Webcam)  │                     │                             │
└─────────────────────┘                     │  ┌─────────────────────┐    │
                                            │  │ Face Detector (SSD)  │    │
                                            │  └──────────┬──────────┘    │
                                            │  ┌──────────▼──────────┐    │
                                            │  │ Face Recognizer      │    │
                                            │  │ (ArcFace R100)       │    │
                                            │  └──────────┬──────────┘    │
                                            │  ┌──────────▼──────────┐    │
                                            │  │ Disguise Handler     │    │
                                            │  │ (LBP Features)       │    │
                                            │  └──────────┬──────────┘    │
                                            │  ┌──────────▼──────────┐    │
                                            │  │ Database Service     │    │
                                            │  │ (SQLAlchemy + asyncpg)│   │
                                            │  └──────────┬──────────┘    │
                                            └─────────────┼───────────────┘
                                                          │
                                            ┌─────────────▼───────────────┐
                                            │   PostgreSQL Database       │
                                            │   (criminals, alerts)       │
                                            └─────────────────────────────┘
```

---

## Prerequisites

Make sure you have these installed before proceeding:

| Tool | Version | Check Command |
|------|---------|---------------|
| **Python** | 3.9+ | `python3 --version` |
| **PostgreSQL** | 14+ | `psql --version` |
| **pip** | Latest | `pip --version` |
| **Git** | Any | `git --version` |
| Java JDK (for frontend) | 17+ | `java --version` |
| Maven (for frontend) | 3.8+ | `mvn --version` |

---

## Step 1: Install PostgreSQL

### macOS (Homebrew)

```bash
# Install PostgreSQL
brew install postgresql@16

# Start the PostgreSQL service
brew services start postgresql@16

# Verify it's running
psql -U $(whoami) -d postgres -c "SELECT version();"
```

> **Note**: On macOS, Homebrew PostgreSQL runs under your current username by default (no password needed for local connections). You may need to create the `postgres` role:
> ```bash
> createuser -s postgres
> ```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### Windows

1. Download from https://www.postgresql.org/download/windows/
2. Run the installer (set password to `postgres` during setup)
3. PostgreSQL will run as a Windows service automatically

---

## Step 2: Create the Database

### Option A: Using the SQL schema file

```bash
# Connect to PostgreSQL
psql -U postgres

# Inside the psql shell:
CREATE DATABASE criminal_detection;
\c criminal_detection
\i database/schema.sql
```

### Option B: Let SQLAlchemy auto-create tables (recommended)

Just create the database — the backend will create tables automatically on startup:

```bash
# Create the database only
psql -U postgres -c "CREATE DATABASE criminal_detection;"
```

That's it! When you start the backend, SQLAlchemy will auto-create the `criminals` and `detection_alerts` tables from the ORM models.

### Verify the database exists

```bash
psql -U postgres -c "\l" | grep criminal_detection
```

### Configure credentials (if non-default)

If your PostgreSQL uses different credentials than `postgres`/`postgres`, set environment variables:

```bash
# Linux/macOS:
export DB_USER=your_username
export DB_PASSWORD=your_password
export DB_HOST=localhost
export DB_PORT=5432

# Windows (PowerShell):
$env:DB_USER = "your_username"
$env:DB_PASSWORD = "your_password"
```

Or edit `backend/config.py` directly.

---

## Step 3: Download AI Models

The system needs 3 pretrained model files. Run the automated downloader for the first 2:

```bash
# From the project root directory:
python scripts/download_models.py
```

This downloads:
- ✅ `deploy.prototxt` — SSD face detector architecture (~28 KB)
- ✅ `res10_300x300_ssd_iter_140000.caffemodel` — SSD weights (~10.7 MB)

### Download ArcFace R100 manually (~248 MB)

The ArcFace model is too large for automated download. Download it from one of these sources:

1. **InsightFace releases**: https://github.com/deepinsight/insightface/releases
2. **ONNX Model Zoo**: https://github.com/onnx/models

Save the file as:
```
backend/models/arcface_r100.onnx
```

### Verify models directory

After downloading, your `backend/models/` should look like:

```
backend/models/
├── deploy.prototxt                         (~28 KB)
├── res10_300x300_ssd_iter_140000.caffemodel (~10.7 MB)
├── arcface_r100.onnx                       (~248 MB)
└── README.md
```

---

## Step 4: Install Python Dependencies

```bash
# Navigate to the project root
cd criminal_database_detection

# Create a virtual environment
python3 -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Install all dependencies
pip install -r backend/requirements.txt
```

### Key dependencies installed

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework for the REST API |
| `uvicorn` | ASGI server to run FastAPI |
| `asyncpg` | Async PostgreSQL driver |
| `sqlalchemy[asyncio]` | ORM with async support |
| `opencv-python` | Face detection (DNN module) |
| `onnxruntime` | ArcFace model inference |
| `numpy` | Numerical computing (embeddings) |
| `scikit-image` | LBP feature extraction |
| `Pillow` | Image processing utilities |

---

## Step 5: Start the Backend

Make sure PostgreSQL is running and the database is created, then:

```bash
# From the project root directory (with venv activated):

# Development mode (with auto-reload):
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode:
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Expected startup output

```
2026-04-15 11:00:00 | INFO    | Starting Criminal Face Detection Backend...
2026-04-15 11:00:00 | INFO    | Loading face detection model...
2026-04-15 11:00:00 | INFO    | Face detection model loaded successfully.
2026-04-15 11:00:01 | INFO    | Loading ArcFace R100 model...
2026-04-15 11:00:02 | INFO    | ArcFace model loaded.
2026-04-15 11:00:02 | INFO    | Disguise handler initialized (LBP radius=3, points=24)
2026-04-15 11:00:02 | INFO    | Connecting to PostgreSQL database...
2026-04-15 11:00:02 | INFO    | Initializing database tables...
2026-04-15 11:00:02 | INFO    | Database tables initialized successfully.
2026-04-15 11:00:02 | INFO    | Embedding cache loaded: 0 criminal(s)
2026-04-15 11:00:02 | INFO    | All services initialized. Server is ready.
```

### Verify it's running

Open your browser and go to:

- **Health check**: http://localhost:8000/health
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

---

## Step 6: Test the System

### Quick health check

```bash
curl http://localhost:8000/health
# → {"status":"healthy","service":"Criminal Face Detection API"}
```

### Run the automated test suite

```bash
# With the backend running in another terminal:
python -m backend.test_api
```

### Test via Swagger UI

1. Open http://localhost:8000/docs
2. Try the endpoints interactively:
   - `POST /detect` — upload a base64 image to detect faces
   - `POST /recognize` — identify faces against the database
   - `POST /add-criminal` — register a new criminal with a face photo
   - `GET /alerts` — view recent detection alerts

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/detect` | Detect faces in an image, return bounding boxes |
| `POST` | `/recognize` | Identify faces against the criminal database |
| `POST` | `/add-criminal` | Register a new criminal with face image |
| `GET` | `/alerts` | Retrieve recent detection alerts |
| `GET` | `/health` | System health check |
| `GET` | `/docs` | Interactive Swagger API documentation |

### Example: Add a criminal

```bash
# Encode a face image to base64
BASE64_IMAGE=$(base64 -i path/to/face.jpg)

# Add the criminal
curl -X POST http://localhost:8000/add-criminal \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"John Doe\",
    \"crime\": \"Robbery\",
    \"image\": \"$BASE64_IMAGE\"
  }"
```

### Example: Recognize faces

```bash
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$BASE64_IMAGE\"}"
```

---

## Batch Import

To bulk-import criminal records from a folder of face images:

```bash
# Image filenames must follow: Name__Crime.ext
# Example: John_Doe__robbery.jpg

python -m backend.batch_import ./criminals_data
```

See `backend/batch_import.py` for details on the filename format.

---

## Project Structure

```
criminal_database_detection/
├── backend/
│   ├── __init__.py              # Package initializer
│   ├── main.py                  # FastAPI app & API endpoints
│   ├── config.py                # Configuration (DB, models, thresholds)
│   ├── schemas.py               # Pydantic request/response models
│   ├── db_models.py             # SQLAlchemy ORM models (Criminal, DetectionAlert)
│   ├── database_engine.py       # Async SQLAlchemy engine & session factory
│   ├── requirements.txt         # Python dependencies
│   ├── batch_import.py          # Bulk criminal import script
│   ├── test_api.py              # API test suite
│   ├── models/                  # AI model files (download required)
│   │   ├── deploy.prototxt
│   │   ├── res10_300x300_ssd_iter_140000.caffemodel
│   │   ├── arcface_r100.onnx
│   │   └── README.md
│   └── services/
│       ├── __init__.py
│       ├── database.py          # PostgreSQL database service (SQLAlchemy ORM)
│       ├── face_detector.py     # OpenCV DNN face detection
│       ├── face_recognizer.py   # ArcFace embedding generation
│       └── disguise_handler.py  # LBP-based disguise detection
├── database/
│   └── schema.sql               # PostgreSQL schema (optional, tables auto-created)
├── frontend/                    # Java Swing GUI (separate setup)
├── scripts/
│   └── download_models.py       # Automated model downloader
├── README.md                    # ← You are here
└── .gitignore
```

---

## Configuration

All settings are in `backend/config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `criminal_detection` | Database name |
| `DB_USER` | `postgres` | Database username |
| `DB_PASSWORD` | `postgres` | Database password |
| `DATABASE_URL` | (auto-built) | Full SQLAlchemy connection URL |
| `FACE_DETECTION_CONFIDENCE` | `0.5` | Min confidence for face detection |
| `RECOGNITION_SIMILARITY_THRESHOLD` | `0.6` | Min cosine similarity for a match |
| `LBP_MATCH_THRESHOLD` | `0.4` | Min similarity for LBP partial matching |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `psql: connection refused` | Ensure PostgreSQL is running: `brew services start postgresql@16` (macOS) or `sudo systemctl start postgresql` (Linux) |
| `FATAL: database "criminal_detection" does not exist` | Create it: `psql -U postgres -c "CREATE DATABASE criminal_detection;"` |
| `FATAL: role "postgres" does not exist` | Create the role: `createuser -s postgres` (common on macOS Homebrew installs) |
| `FATAL: password authentication failed` | Check `DB_USER`/`DB_PASSWORD` in `config.py` or set env vars |
| `ModuleNotFoundError: No module named 'asyncpg'` | Run `pip install -r backend/requirements.txt` inside your venv |
| `FileNotFoundError: arcface_r100.onnx` | Download the ArcFace model — see [Step 3](#step-3-download-ai-models) |
| `FileNotFoundError: deploy.prototxt` | Run `python scripts/download_models.py` |
| `cv2.error: Can't open face detection model` | Verify model files exist in `backend/models/` |
| `CORS error from frontend` | Backend allows all origins by default (`CORS_ORIGINS = ["*"]`) |
| `Slow recognition` | Use GPU (set CUDA execution provider) or reduce webcam resolution |
| `Address already in use (port 8000)` | Kill the existing process: `lsof -ti:8000 \| xargs kill` |

---

## Quick Start (TL;DR)

```bash
# 1. Install & start PostgreSQL
brew install postgresql@16 && brew services start postgresql@16

# 2. Create database
psql -U postgres -c "CREATE DATABASE criminal_detection;"

# 3. Download AI models
python scripts/download_models.py
# + manually download arcface_r100.onnx → backend/models/

# 4. Install Python deps
python3 -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt

# 5. Run the server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Open API docs
open http://localhost:8000/docs
```
