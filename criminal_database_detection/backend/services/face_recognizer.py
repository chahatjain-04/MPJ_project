"""
Face Recognition Service
========================
Uses the ArcFace R100 model (ONNX format) to generate 512-dimensional
face embeddings. These embeddings are compact numerical representations
of a face that can be compared using cosine similarity.

ArcFace (Additive Angular Margin Loss) is one of the most accurate
face recognition models. The R100 variant uses a ResNet-100 backbone
trained on the MS1MV2 dataset (~5.8M images, ~85K identities).

Pipeline:
  1. Receive a cropped face image
  2. Resize to 112x112 (ArcFace input size)
  3. Normalize pixel values: (pixel - 127.5) / 128.0
  4. Convert to NCHW format (batch, channels, height, width)
  5. Run inference through ONNX Runtime
  6. L2-normalize the output embedding
"""

import cv2
import numpy as np
import onnxruntime as ort
import logging
from backend.config import ARCFACE_MODEL_PATH, ARCFACE_INPUT_SIZE, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Generates face embeddings using ArcFace R100 ONNX model and
    provides cosine similarity comparison between embeddings.
    """

    def __init__(self):
        """
        Load the ArcFace ONNX model using ONNX Runtime.
        ONNX Runtime automatically selects the best execution provider
        (CUDA if available, otherwise CPU).
        """
        logger.info("Loading ArcFace R100 model...")

        # Configure ONNX Runtime session options for optimal performance
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # Use 4 threads for CPU inference
        session_options.intra_op_num_threads = 4

        # Create inference session with available providers
        # Prioritize CUDA > CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            ARCFACE_MODEL_PATH, sess_options=session_options, providers=providers
        )

        # Cache input/output names for inference
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(
            f"ArcFace model loaded. Input: {self.input_name}, "
            f"Provider: {self.session.get_providers()}"
        )

    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess a face image for ArcFace input.

        Steps:
          1. Resize to 112x112 pixels
          2. Convert BGR to RGB (ArcFace expects RGB)
          3. Normalize: (pixel - 127.5) / 128.0  → range roughly [-1, 1]
          4. Transpose from HWC (height, width, channels) to CHW
          5. Add batch dimension → shape (1, 3, 112, 112)

        Args:
            face_image: Cropped face image in BGR format (numpy array).

        Returns:
            Preprocessed image tensor (1, 3, 112, 112) as float32.
        """
        # Resize to the expected input size
        face = cv2.resize(face_image, ARCFACE_INPUT_SIZE)

        # Convert from BGR (OpenCV default) to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize pixel values to approximately [-1, 1]
        face = (face.astype(np.float32) - 127.5) / 128.0

        # Transpose from HWC to CHW format (model expects channels-first)
        face = np.transpose(face, (2, 0, 1))

        # Add batch dimension: (3, 112, 112) → (1, 3, 112, 112)
        face = np.expand_dims(face, axis=0)

        return face

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate a 512-dimensional face embedding for the given face image.

        The embedding is L2-normalized so that cosine similarity between
        two embeddings equals their dot product, simplifying comparisons.

        Args:
            face_image: Cropped face image in BGR format (numpy array).

        Returns:
            L2-normalized embedding vector of shape (512,) as float32.
        """
        # Preprocess the face image
        input_tensor = self.preprocess_face(face_image)

        # Run inference
        result = self.session.run(
            [self.output_name], {self.input_name: input_tensor}
        )

        # Extract the embedding (remove batch dimension)
        embedding = result[0][0]

        # L2-normalize the embedding
        # This ensures ||embedding|| = 1, so dot product = cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Since embeddings are L2-normalized, cosine similarity = dot product.

        Cosine similarity ranges from -1 to 1:
          - 1.0: identical faces
          - 0.0: completely unrelated
          - < 0: very different (rare for face embeddings)

        Args:
            embedding1: First face embedding (512-dim).
            embedding2: Second face embedding (512-dim).

        Returns:
            Cosine similarity score as a float.
        """
        # For L2-normalized vectors: cos_sim = dot(a, b) / (||a|| * ||b||) = dot(a, b)
        similarity = float(np.dot(embedding1, embedding2))

        # Clamp to valid range (floating point errors can cause slight overflow)
        return max(-1.0, min(1.0, similarity))
