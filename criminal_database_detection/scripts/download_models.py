"""
Model Downloader Script
=======================
Downloads the required AI model files for the Criminal Face Detection system.

Models downloaded:
  1. deploy.prototxt — OpenCV SSD face detector architecture (~28 KB)
  2. res10_300x300_ssd_iter_140000.caffemodel — SSD weights (~10.7 MB)
  3. arcface_r100.onnx — ArcFace recognition model (must be downloaded manually)

Usage:
    python scripts/download_models.py

Note: The ArcFace ONNX model is large (~248 MB) and hosted on Google Drive,
so it cannot be downloaded automatically. This script will print instructions
for manual download.
"""

import os
import sys
import urllib.request
import hashlib

# Target directory for models
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backend", "models"
)

# Model URLs and metadata
MODELS = [
    {
        "name": "deploy.prototxt",
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "size_desc": "~28 KB",
        "description": "OpenCV SSD face detector architecture",
    },
    {
        "name": "res10_300x300_ssd_iter_140000.caffemodel",
        "url": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "size_desc": "~10.7 MB",
        "description": "SSD face detector pretrained weights",
    },
]


def download_file(url: str, dest: str, name: str, size_desc: str) -> bool:
    """
    Download a file with a progress indicator.

    Args:
        url: URL to download from.
        dest: Destination file path.
        name: Display name for the file.
        size_desc: Human-readable size description.

    Returns:
        True if download succeeded, False otherwise.
    """
    if os.path.exists(dest):
        file_size = os.path.getsize(dest)
        print(f"  ✅ {name} already exists ({file_size:,} bytes)")
        return True

    print(f"  ⬇️  Downloading {name} ({size_desc})...")
    print(f"     URL: {url[:80]}...")

    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                bar_len = 30
                filled = int(bar_len * downloaded // total_size)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r     [{bar}] {percent:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
        file_size = os.path.getsize(dest)
        print(f"\n  ✅ Downloaded: {name} ({file_size:,} bytes)")
        return True

    except Exception as e:
        print(f"\n  ❌ Failed to download {name}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Criminal Face Detection — Model Downloader             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Target directory: {MODELS_DIR}\n")

    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Download each model
    success = 0
    total = len(MODELS)

    for model in MODELS:
        dest = os.path.join(MODELS_DIR, model["name"])
        print(f"\n  [{success + 1}/{total}] {model['description']}")

        if download_file(model["url"], dest, model["name"], model["size_desc"]):
            success += 1

    # Check for ArcFace model
    arcface_path = os.path.join(MODELS_DIR, "arcface_r100.onnx")
    print(f"\n  [3/3] ArcFace R100 ONNX Model")

    if os.path.exists(arcface_path):
        file_size = os.path.getsize(arcface_path)
        print(f"  ✅ arcface_r100.onnx already exists ({file_size:,} bytes)")
        success += 1
    else:
        print(f"  ⚠️  arcface_r100.onnx must be downloaded manually.")
        print(f"")
        print(f"     This model is too large (~248 MB) for automated download.")
        print(f"     Please download it from one of these sources:")
        print(f"")
        print(f"     1. InsightFace releases:")
        print(f"        https://github.com/deepinsight/insightface/releases")
        print(f"")
        print(f"     2. ONNX Model Zoo:")
        print(f"        https://github.com/onnx/models")
        print(f"")
        print(f"     Save the file as:")
        print(f"        {arcface_path}")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"  Download Summary: {success}/3 models ready")

    if success >= 2:
        print(f"  SSD face detector models: ✅ Ready")
    if os.path.exists(arcface_path):
        print(f"  ArcFace recognition model: ✅ Ready")
    else:
        print(f"  ArcFace recognition model: ⚠️  Manual download needed")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
