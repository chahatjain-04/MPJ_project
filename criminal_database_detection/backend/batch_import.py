"""
Batch Import Script
===================
Bulk-import criminal records from a folder of face images.

Expected folder structure:
    criminals_data/
    ├── John_Doe__robbery.jpg
    ├── Jane_Smith__fraud.jpg
    └── Mike_Brown__assault.png

Filename format:  Name__Crime.ext
    - Name: Criminal's name (underscores become spaces)
    - Crime: Crime description (underscores become spaces)
    - Separated by double underscore (__)

Usage:
    python -m backend.batch_import ./criminals_data
    python -m backend.batch_import ./criminals_data --url http://localhost:8000

This script reads each image, sends it to the /add-criminal endpoint,
and reports success/failure for each file.
"""

import os
import sys
import base64
import argparse
import requests
import time


def parse_filename(filename: str) -> tuple:
    """
    Parse criminal details from the filename.

    Format: Name__Crime.ext
    Example: John_Doe__robbery.jpg → ("John Doe", "robbery")

    Args:
        filename: Image filename (without directory path).

    Returns:
        Tuple of (name, crime) with underscores replaced by spaces.
        Returns (None, None) if the format is invalid.
    """
    # Remove file extension
    name_part = os.path.splitext(filename)[0]

    # Split by double underscore
    parts = name_part.split("__")

    if len(parts) != 2:
        return None, None

    name = parts[0].replace("_", " ").strip()
    crime = parts[1].replace("_", " ").strip()

    return name, crime


def import_criminal(base_url: str, name: str, crime: str, image_path: str) -> bool:
    """
    Send a single criminal record to the API.

    Args:
        base_url: Backend URL (e.g., "http://localhost:8000").
        name: Criminal's name.
        crime: Crime description.
        image_path: Path to the face image.

    Returns:
        True if successfully added, False otherwise.
    """
    # Read and encode the image
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Send to API
    payload = {
        "name": name,
        "crime": crime,
        "image": base64_image,
    }

    try:
        resp = requests.post(
            f"{base_url}/add-criminal",
            json=payload,
            timeout=60,
        )

        if resp.status_code == 200:
            data = resp.json()
            return data.get("success", False)
        else:
            print(f"    Error: HTTP {resp.status_code} — {resp.text[:200]}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"    Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Bulk-import criminal records from a folder of face images."
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing criminal face images.",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Backend URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    base_url = args.url.rstrip("/")

    # Validate folder
    if not os.path.isdir(folder):
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)

    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Collect image files
    image_files = [
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    if not image_files:
        print(f"No image files found in: {folder}")
        print(f"Supported formats: {', '.join(valid_extensions)}")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Criminal Face Detection — Batch Import                 ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Source folder: {folder}")
    print(f"  Backend URL:   {base_url}")
    print(f"  Images found:  {len(image_files)}")
    print(f"\n  Filename format: Name__Crime.ext")
    print(f"  Example: John_Doe__robbery.jpg\n")

    # Check backend connectivity
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code != 200:
            print(f"  ⚠️  Backend returned status {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Cannot connect to backend at {base_url}")
        print(f"     Make sure the server is running.")
        sys.exit(1)

    # Process each image
    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, filename in enumerate(sorted(image_files), 1):
        image_path = os.path.join(folder, filename)
        name, crime = parse_filename(filename)

        if name is None:
            print(f"  [{i}/{len(image_files)}] SKIP  {filename}")
            print(f"    Reason: Invalid format. Use Name__Crime.ext")
            skip_count += 1
            continue

        print(f"  [{i}/{len(image_files)}] Importing: {name} — {crime}")

        if import_criminal(base_url, name, crime, image_path):
            print(f"    ✅ Added successfully")
            success_count += 1
        else:
            print(f"    ❌ Failed to add")
            fail_count += 1

        # Small delay to avoid overwhelming the server
        time.sleep(0.5)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"  Import Complete!")
    print(f"  ✅ Success: {success_count}")
    print(f"  ❌ Failed:  {fail_count}")
    print(f"  ⏭️  Skipped: {skip_count}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
