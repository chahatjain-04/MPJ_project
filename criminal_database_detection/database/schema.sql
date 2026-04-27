-- =============================================================================
-- Criminal Face Detection & Identification System
-- Database Schema — PostgreSQL
-- =============================================================================

-- =============================================================================
-- Table: criminals
-- Stores criminal records with their face embeddings for identification.
-- The 'embedding' column holds a JSON string of 512 floats (ArcFace embedding).
-- The 'lbp_histogram' column stores LBP feature histograms (JSON string) for
-- partial matching when faces are disguised or occluded.
-- =============================================================================
CREATE TABLE IF NOT EXISTS criminals (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,
    crime           TEXT NOT NULL,
    embedding       TEXT NOT NULL,              -- ArcFace face embedding (512-dim float array as JSON)
    lbp_histogram   TEXT DEFAULT NULL,          -- LBP features per region for disguise matching (JSON)
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- Table: detection_alerts
-- Logs every detection event. Stores the matched criminal (if any),
-- confidence score, whether the person appeared disguised, and a
-- base64-encoded image snapshot for audit/review purposes.
-- =============================================================================
CREATE TABLE IF NOT EXISTS detection_alerts (
    id              SERIAL PRIMARY KEY,
    criminal_id     INTEGER DEFAULT NULL,
    confidence      REAL NOT NULL DEFAULT 0.0,
    is_disguised    BOOLEAN NOT NULL DEFAULT FALSE,
    image_data      TEXT DEFAULT NULL,          -- base64-encoded JPEG snapshot
    detected_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (criminal_id) REFERENCES criminals(id) ON DELETE SET NULL
);

-- =============================================================================
-- Indexes for performance
-- =============================================================================

-- Index on criminal name for text searches
CREATE INDEX IF NOT EXISTS idx_criminals_name ON criminals (name);

-- Index on detection timestamp for efficient recent-alerts queries
CREATE INDEX IF NOT EXISTS idx_alerts_detected_at ON detection_alerts (detected_at DESC);

-- Index on criminal_id in alerts for JOIN performance
CREATE INDEX IF NOT EXISTS idx_alerts_criminal_id ON detection_alerts (criminal_id);

-- =============================================================================
-- Verify setup
-- =============================================================================
-- You can verify the tables were created by running:
-- \dt
-- \d criminals
-- \d detection_alerts
