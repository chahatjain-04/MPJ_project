"""
Disguise Detection & Partial Matching Service
==============================================
Handles cases where a criminal's face is partially occluded (sunglasses,
mask, scarf, hat, etc.). When the standard ArcFace embedding-based
recognition fails or returns low confidence, this module:

  1. Analyzes the face for signs of occlusion/disguise
  2. Extracts LBP (Local Binary Patterns) features from visible regions
  3. Performs partial matching against stored LBP histograms

LBP is a powerful texture descriptor that encodes local pixel patterns.
It's robust to monotonic illumination changes and computationally efficient.
By computing LBP on sub-regions of the face (forehead, eyes, jawline),
we can match criminals even when parts of their face are covered.

Face Regions:
  ┌─────────────────┐
  │    Forehead      │  Region 0: top 30%
  ├─────────────────┤
  │   Eye Region     │  Region 1: 30-55%
  ├─────────────────┤
  │   Nose/Cheek     │  Region 2: 55-75%
  ├─────────────────┤
  │   Mouth/Chin     │  Region 3: bottom 25%
  └─────────────────┘
"""

import cv2
import numpy as np
import logging
from skimage.feature import local_binary_pattern
from backend.config import LBP_RADIUS, LBP_N_POINTS, LBP_MATCH_THRESHOLD

logger = logging.getLogger(__name__)

# Number of histogram bins for 'uniform' LBP
# Uniform LBP has (n_points + 2) distinct patterns
LBP_HIST_BINS = LBP_N_POINTS + 2

# Face regions as (start_ratio, end_ratio) of face height
FACE_REGIONS = {
    "forehead": (0.0, 0.30),
    "eyes": (0.25, 0.55),
    "nose_cheek": (0.50, 0.75),
    "mouth_chin": (0.70, 1.0),
}


class DisguiseHandler:
    """
    Detects facial disguises and performs LBP-based partial matching
    on visible face regions.
    """

    def __init__(self):
        """Initialize the disguise handler with LBP parameters."""
        self.radius = LBP_RADIUS
        self.n_points = LBP_N_POINTS
        logger.info(
            f"Disguise handler initialized (LBP radius={self.radius}, "
            f"points={self.n_points})"
        )

    def is_disguised(self, face_image: np.ndarray) -> bool:
        """
        Analyze a face image to determine if it's partially disguised/occluded.

        Detection strategy:
          1. Convert face to grayscale
          2. Split into horizontal regions (forehead, eyes, nose, mouth)
          3. Calculate skin-color coverage in each region
          4. If any region has significantly low skin coverage, the face
             is likely occluded in that area → disguised

        Skin detection uses the YCrCb color space, which separates
        luminance from chrominance, making it robust to lighting changes.

        Args:
            face_image: BGR face crop (numpy array).

        Returns:
            True if the face appears to be disguised/occluded.
        """
        if face_image is None or face_image.size == 0:
            return True  # No face = assume disguised

        h, w = face_image.shape[:2]
        if h < 20 or w < 20:
            return True  # Too small to analyze

        # -----------------------------------------------------------------
        # Step 1: Detect skin-colored pixels using YCrCb color space
        # Typical skin color ranges in YCrCb:
        #   Y:  0-255 (any brightness)
        #   Cr: 133-173 (red chrominance)
        #   Cb: 77-127  (blue chrominance)
        # -----------------------------------------------------------------
        ycrcb = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, (0, 125, 70), (255, 185, 135))

        # -----------------------------------------------------------------
        # Step 2: Check skin coverage in each face region
        # If a region has < 15% skin coverage, it's likely occluded
        # -----------------------------------------------------------------
        occluded_regions = 0
        total_regions = len(FACE_REGIONS)

        for region_name, (start_ratio, end_ratio) in FACE_REGIONS.items():
            y_start = int(h * start_ratio)
            y_end = int(h * end_ratio)
            region_mask = skin_mask[y_start:y_end, :]

            # Calculate fraction of skin-colored pixels in this region
            if region_mask.size > 0:
                skin_fraction = np.count_nonzero(region_mask) / region_mask.size
            else:
                skin_fraction = 0.0

            # If skin coverage is low, this region is likely occluded.
            # Threshold raised to 0.18 so sunglasses/masks reliably trigger.
            if skin_fraction < 0.18:
                occluded_regions += 1
                logger.debug(
                    f"Region '{region_name}' occluded: "
                    f"skin coverage = {skin_fraction:.2%}"
                )

        # Face is considered disguised if 2+ regions are occluded.
        # 2 catches typical real-world disguises:
        #   - Sunglasses alone cover the eye region (1) + often nose_cheek (2)
        #   - A face mask covers nose_cheek (2) + mouth_chin (3)
        # Was 3 before — too strict: most disguises were missed.
        is_disguised = occluded_regions >= 2
        if is_disguised:
            logger.info(
                f"Face classified as DISGUISED ({occluded_regions}/{total_regions} "
                f"regions occluded)"
            )

        return is_disguised

    def extract_lbp_features(self, face_image: np.ndarray) -> dict:
        """
        Extract LBP histograms from each visible region of the face.

        LBP (Local Binary Patterns) works by:
          1. For each pixel, sample N points on a circle of radius R
          2. Compare each sample to the center pixel (0 if less, 1 if greater)
          3. Form a binary number from these comparisons
          4. 'uniform' patterns (≤2 bit transitions) get unique bins;
             all non-uniform patterns go into a single bin

        Using 'uniform' LBP reduces the histogram from 2^N bins to (N+2) bins,
        while capturing the most important texture patterns (edges, corners, etc.).

        Args:
            face_image: BGR face crop (numpy array).

        Returns:
            Dict mapping region names to their LBP histogram (list of floats).
            Only visible (non-occluded) regions are included.
        """
        # Convert to grayscale for LBP computation
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Detect skin to determine which regions are visible
        ycrcb = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, (0, 125, 70), (255, 185, 135))

        features = {}

        for region_name, (start_ratio, end_ratio) in FACE_REGIONS.items():
            y_start = int(h * start_ratio)
            y_end = int(h * end_ratio)

            # Check if this region is visible (has enough skin pixels)
            region_skin = skin_mask[y_start:y_end, :]
            if region_skin.size > 0:
                skin_fraction = np.count_nonzero(region_skin) / region_skin.size
            else:
                skin_fraction = 0.0

            # Only extract LBP from sufficiently visible regions
            if skin_fraction >= 0.10:
                region_gray = gray[y_start:y_end, :]

                # Compute LBP for this region
                lbp = local_binary_pattern(
                    region_gray, self.n_points, self.radius, method="uniform"
                )

                # Build a normalized histogram of LBP patterns
                hist, _ = np.histogram(
                    lbp.ravel(),
                    bins=np.arange(0, LBP_HIST_BINS + 1),
                    density=True,
                )

                features[region_name] = hist.tolist()
            else:
                logger.debug(
                    f"Skipping occluded region '{region_name}' "
                    f"(skin={skin_fraction:.2%})"
                )

        return features

    def partial_match(
        self,
        query_features: dict,
        stored_features_list: list,
        threshold: float = None,
    ) -> list:
        """
        Compare LBP features from visible regions against stored entries
        using chi-squared distance.

        For each stored entry, we:
          1. Find overlapping regions (both query and stored have features)
          2. Compute chi-squared distance for each overlapping region
          3. Average the distances across all overlapping regions
          4. Convert distance to a similarity score: 1 / (1 + distance)

        Chi-squared distance is ideal for histogram comparison:
          χ²(h1, h2) = Σ (h1[i] - h2[i])² / (h1[i] + h2[i])

        Args:
            query_features: LBP features from the query face {region: histogram}.
            stored_features_list: List of dicts with 'id', 'name', 'lbp_histogram'.
            threshold: Min similarity to consider a match (default from config).

        Returns:
            List of matching entries with 'id', 'name', 'similarity', sorted
            by similarity descending. Empty list if no matches.
        """
        if threshold is None:
            threshold = LBP_MATCH_THRESHOLD

        if not query_features:
            logger.debug("No visible regions for LBP matching.")
            return []

        matches = []

        for entry in stored_features_list:
            stored_features = entry.get("lbp_histogram")
            if not stored_features:
                continue

            # Find regions present in both query and stored features
            common_regions = set(query_features.keys()) & set(stored_features.keys())

            if not common_regions:
                continue  # No overlapping visible regions

            # Compute chi-squared distance for each common region
            total_distance = 0.0
            for region in common_regions:
                query_hist = np.array(query_features[region], dtype=np.float64)
                stored_hist = np.array(stored_features[region], dtype=np.float64)

                # Chi-squared distance (add epsilon to avoid division by zero)
                denominator = query_hist + stored_hist + 1e-10
                chi_sq = np.sum((query_hist - stored_hist) ** 2 / denominator)
                total_distance += chi_sq

            # Average distance across overlapping regions
            avg_distance = total_distance / len(common_regions)

            # Convert distance to similarity (0 = infinite distance, 1 = identical)
            similarity = 1.0 / (1.0 + avg_distance)

            if similarity >= threshold:
                matches.append(
                    {
                        "id": entry["id"],
                        "name": entry["name"],
                        "similarity": round(similarity, 4),
                        "matched_regions": list(common_regions),
                    }
                )

        # Sort by similarity descending (best match first)
        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return matches
