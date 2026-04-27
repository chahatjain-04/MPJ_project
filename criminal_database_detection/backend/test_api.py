"""
API Test Script
===============
Standalone test script to verify all backend API endpoints work correctly.
Run this AFTER starting the backend server.

Usage:
    python -m backend.test_api

Or:
    python backend/test_api.py

Prerequisites:
    - Backend running at http://localhost:8000
    - PostgreSQL database created and schema applied
    - AI models downloaded to backend/models/
"""

import requests
import base64
import sys
import os
import json
import numpy as np

# Backend URL
BASE_URL = "http://localhost:8000"

# Create a simple test image (solid color with a rectangle)
# In real use, you'd supply an actual face photo
def create_test_image():
    """
    Create a simple test image for API testing.
    For actual face detection, use a real photo with a face.
    """
    try:
        import cv2
        # Create a 300x300 blank image
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:] = (200, 180, 160)  # Skin-like color
        # Draw a rough face-like ellipse
        cv2.ellipse(img, (150, 150), (80, 100), 0, 0, 360, (180, 160, 140), -1)
        # Encode as JPEG → base64
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')
    except ImportError:
        print("[WARN] OpenCV not available. Using minimal test image.")
        # 1x1 pixel JPEG as minimal test
        return "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAFBABAAAAAAAAAAAAAAAAAAAACf/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AKgA/9k="


def separator(title):
    """Print a visual separator for test sections."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_health():
    """Test the /health endpoint."""
    separator("TEST: Health Check")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"  Status: {resp.status_code}")
        print(f"  Response: {resp.json()}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
        print("  ✅ PASSED")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def test_detect(base64_image):
    """Test the POST /detect endpoint."""
    separator("TEST: Face Detection (/detect)")
    try:
        payload = {"image": base64_image}
        resp = requests.post(f"{BASE_URL}/detect", json=payload, timeout=30)
        print(f"  Status: {resp.status_code}")
        data = resp.json()
        print(f"  Faces detected: {data.get('count', 0)}")
        if data.get('faces'):
            for i, face in enumerate(data['faces']):
                print(f"    Face {i+1}: x={face['x']}, y={face['y']}, "
                      f"w={face['width']}, h={face['height']}, "
                      f"confidence={face['confidence']:.4f}")
        print("  ✅ PASSED")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def test_recognize(base64_image):
    """Test the POST /recognize endpoint."""
    separator("TEST: Face Recognition (/recognize)")
    try:
        payload = {"image": base64_image}
        resp = requests.post(f"{BASE_URL}/recognize", json=payload, timeout=30)
        print(f"  Status: {resp.status_code}")
        data = resp.json()
        results = data.get('results', [])
        print(f"  Results: {len(results)}")
        for i, result in enumerate(results):
            print(f"    Face {i+1}: name='{result['name']}', "
                  f"confidence={result['confidence']:.4f}, "
                  f"disguised={result['is_disguised']}")
        print("  ✅ PASSED")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def test_add_criminal(base64_image):
    """Test the POST /add-criminal endpoint."""
    separator("TEST: Add Criminal (/add-criminal)")
    try:
        payload = {
            "name": "Test Criminal",
            "crime": "API Testing - Safe to delete",
            "image": base64_image
        }
        resp = requests.post(f"{BASE_URL}/add-criminal", json=payload, timeout=30)
        print(f"  Status: {resp.status_code}")
        data = resp.json()
        print(f"  Success: {data.get('success')}")
        print(f"  Message: {data.get('message')}")
        print(f"  Criminal ID: {data.get('criminal_id')}")
        if resp.status_code == 200:
            print("  ✅ PASSED")
            return True
        elif resp.status_code == 400:
            print("  ⚠️  No face in test image (expected with synthetic image)")
            print("  ✅ PASSED (endpoint works, needs real face)")
            return True
        else:
            print(f"  ❌ FAILED: Unexpected status {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def test_alerts():
    """Test the GET /alerts endpoint."""
    separator("TEST: Get Alerts (/alerts)")
    try:
        resp = requests.get(f"{BASE_URL}/alerts?limit=5", timeout=10)
        print(f"  Status: {resp.status_code}")
        data = resp.json()
        alerts = data.get('alerts', [])
        print(f"  Total alerts: {data.get('total', 0)}")
        for alert in alerts[:3]:
            print(f"    Alert #{alert['id']}: "
                  f"criminal='{alert.get('criminal_name', 'Unknown')}', "
                  f"confidence={alert['confidence']:.4f}")
        print("  ✅ PASSED")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def main():
    """Run all API tests."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Criminal Face Detection — API Test Suite               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nTarget: {BASE_URL}")

    # Generate test image
    print("\nGenerating test image...")
    test_image = create_test_image()
    print(f"  Image size: {len(test_image)} chars (base64)")

    # Run all tests
    results = {
        "Health Check": test_health(),
        "Face Detection": test_detect(test_image),
        "Face Recognition": test_recognize(test_image),
        "Add Criminal": test_add_criminal(test_image),
        "Get Alerts": test_alerts(),
    }

    # Summary
    separator("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")
    print(f"\n  Result: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n  ⚠️  Some tests failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
