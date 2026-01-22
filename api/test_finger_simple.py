"""
Simple Finger Detection Tester (Convex Hull Method)
Tests the new convex hull area difference method

Usage: python test_finger_simple.py <image_path>
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from services.yolo_service import YOLOService
from checkers.finger_checker import FingerChecker
import config

def test_finger_detection(image_path: str):
    """Test finger detection on a single image"""
    
    print(f"\n{'='*60}")
    print(f"Testing Finger Detection: {Path(image_path).name}")
    print(f"{'='*60}\n")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return
    
    print(f"✅ Image loaded: {img.shape[1]}x{img.shape[0]}")
    
    # Initialize services
    print("\n📦 Initializing YOLO service...")
    yolo = YOLOService(config.YOLO_MODEL_PATH, config.YOLO_CONF_THRESH)
    
    print("📦 Initializing Finger Checker (Convex Hull Method)...")
    finger_checker = FingerChecker(
        overlap_area_threshold=config.OVERLAP_AREA_THRESHOLD,
        min_contour_area=config.MIN_CONTOUR_AREA
    )
    
    # Run YOLO detection
    print("\n🔍 Running YOLO detection...")
    detected, polygon, conf, bbox = yolo.detect_card(img)
    
    if not detected:
        print("❌ No card detected by YOLO")
        return
    
    print(f"✅ Card detected with confidence: {conf:.3f}")
    print(f"   Polygon points: {len(polygon)}")
    
    # Run finger detection
    print(f"\n🖐️  Running Finger Detection...")
    print(f"   Method: Convex Hull Area Difference")
    print(f"   Threshold: {config.OVERLAP_AREA_THRESHOLD*100:.2f}%")
    print(f"   Min Contour Area: {config.MIN_CONTOUR_AREA} pixels")
    
    passed, area_diff_ratio, message, metrics = finger_checker.check(polygon, img.shape)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"Status: {'✅ PASSED' if passed else '❌ FAILED (Finger Detected)'}")
    print(f"Area Difference Ratio: {area_diff_ratio*100:.2f}%")
    print(f"Threshold: {config.OVERLAP_AREA_THRESHOLD*100:.2f}%")
    print(f"\nMessage: {message}")
    
    if metrics:
        print(f"\n📊 Detailed Metrics:")
        print(f"   Contour Area: {metrics.get('contour_area', 0):.2f} pixels")
        print(f"   Hull Area: {metrics.get('hull_area', 0):.2f} pixels")
        print(f"   Area Difference: {metrics.get('area_diff_percentage', 0):.2f}%")
        print(f"   Vertices: {metrics.get('vertex_count', 0)}")
    
    # Explain the result
    print(f"\n💡 Explanation:")
    if passed:
        print(f"   ✅ The card appears clean.")
        print(f"   ✅ Area difference ({area_diff_ratio*100:.2f}%) is below threshold ({config.OVERLAP_AREA_THRESHOLD*100:.2f}%).")
        print(f"   ✅ No significant finger overlap detected.")
    else:
        print(f"   ❌ Finger overlap detected!")
        print(f"   ❌ Area difference ({area_diff_ratio*100:.2f}%) exceeds threshold ({config.OVERLAP_AREA_THRESHOLD*100:.2f}%).")
        print(f"   ❌ The convex hull is significantly larger than the actual contour,")
        print(f"      indicating a concavity caused by finger or object overlap.")
    
    # Visualize
    print(f"\n🎨 Creating visualization...")
    annotated = finger_checker.visualize(img, polygon, area_diff_ratio, passed)
    
    # Save result
    output_dir = Path("outputs/finger_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(image_path).stem}_finger_test.jpg"
    cv2.imwrite(str(output_path), annotated)
    print(f"✅ Saved annotated image to: {output_path}")
    
    # Display
    cv2.imshow("Finger Detection Test", annotated)
    print(f"\n👁️  Displaying result (Press any key to close)...")
    print(f"   🟢 Green: Actual contour (card boundary)")
    print(f"   🔴 Red: Convex hull (shows finger intrusion)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_finger_simple.py <image_path>")
        print("Example: python test_finger_simple.py test_images/card.jpg")
        sys.exit(1)
    
    test_finger_detection(sys.argv[1])
