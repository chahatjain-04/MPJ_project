"""
Face Detection Service
======================
Uses OpenCV's DNN module with a pretrained SSD (Single Shot Detector) model
to detect faces in images. The model used is:
  - Architecture: deploy.prototxt (ResNet-10 based SSD)
  - Weights: res10_300x300_ssd_iter_140000.caffemodel

This detector is fast (~30ms per frame on CPU) and works well for frontal
and slightly angled faces. It outputs bounding boxes with confidence scores.
"""

import cv2
import numpy as np
import logging
from backend.config import (
    FACE_DETECTOR_PROTOTXT,
    FACE_DETECTOR_CAFFEMODEL,
    FACE_DETECTION_CONFIDENCE,
    DNN_INPUT_SIZE,
    DNN_MEAN_VALUES,
    DNN_SCALE_FACTOR,
)

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detects faces in images using OpenCV's DNN module with a pre-trained
    SSD face detection model.
    """

    def __init__(self):
        """
        Load the SSD face detection model from disk.
        The .prototxt file defines the network architecture.
        The .caffemodel file contains the pretrained weights.
        """
        logger.info("Loading face detection model...")
        self.net = cv2.dnn.readNetFromCaffe(
            FACE_DETECTOR_PROTOTXT, FACE_DETECTOR_CAFFEMODEL
        )
        # Use CPU backend (switch to cv2.dnn.DNN_BACKEND_CUDA for GPU)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info("Face detection model loaded successfully.")

    def detect_faces(self, image: np.ndarray, confidence_threshold: float = None):
        """
        Detect faces in the given image.

        Args:
            image: BGR image as a numpy array (from cv2.imread or decoded).
            confidence_threshold: Minimum confidence to keep a detection.
                                  Defaults to FACE_DETECTION_CONFIDENCE from config.

        Returns:
            List of dicts, each containing:
                - 'x', 'y': top-left corner of the bounding box
                - 'width', 'height': dimensions of the bounding box
                - 'confidence': detection confidence score (0.0 - 1.0)
                - 'face_crop': cropped face image (numpy array)
        """
        if confidence_threshold is None:
            confidence_threshold = FACE_DETECTION_CONFIDENCE

        # Get image dimensions for scaling bounding boxes back to original size
        (h, w) = image.shape[:2]

        # =====================================================================
        # Step 1: Create a blob from the image
        # cv2.dnn.blobFromImage performs:
        #   1. Resize to DNN_INPUT_SIZE (300x300)
        #   2. Apply scale factor (1.0 = no scaling)
        #   3. Subtract mean values (104.0, 177.0, 123.0) to normalize
        #   4. Optionally swap R and B channels (not needed for this model)
        # =====================================================================
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, DNN_INPUT_SIZE),
            scalefactor=DNN_SCALE_FACTOR,
            size=DNN_INPUT_SIZE,
            mean=DNN_MEAN_VALUES,
            swapRB=False,
            crop=False,
        )

        # =====================================================================
        # Step 2: Forward pass through the network
        # The output is a 4D tensor: [1, 1, N, 7]
        # where N = number of detections, and each detection has 7 values:
        #   [batch_id, class_id, confidence, x1, y1, x2, y2]
        # x1/y1/x2/y2 are normalized coordinates (0.0 - 1.0)
        # =====================================================================
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []

        # =====================================================================
        # Step 3: Filter detections by confidence and extract bounding boxes
        # =====================================================================
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])

            # Skip low-confidence detections
            if confidence < confidence_threshold:
                continue

            # Scale bounding box from normalized [0,1] to pixel coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Clamp coordinates to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Skip invalid bounding boxes (too small or zero area)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            # Crop the face region from the original (full-resolution) image
            face_crop = image[y1:y2, x1:x2].copy()

            faces.append(
                {
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                    "confidence": round(confidence, 4),
                    "face_crop": face_crop,
                }
            )

        logger.debug(f"Detected {len(faces)} face(s) in image ({w}x{h})")
        return faces
