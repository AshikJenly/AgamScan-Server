"""
Test finger detection in isolation
Helps debug finger detection without running full API
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from checkers.finger_checker import FingerChecker
from services.yolo_service import YOLOService
import config


def test_finger_detection(image_path: str):
    """Test finger detection on a single image"""
    
    print(f"\n{'='*70}")
    print(f"Testing Finger Detection")
    print(f"{'='*70}")
    
    # Initialize services
    print("\n1. Initializing YOLO service...")
    yolo_service = YOLOService()
    if not yolo_service.load_model():
        print("❌ Failed to load YOLO model")
        return
    
    print("\n2. Initializing Finger Checker...")
    finger_checker = FingerChecker()
    print(f"   Settings:")
    print(f"   - Padding: {finger_checker.padding_frac*100:.1f}%")
    print(f"   - Pixel Tolerance: {finger_checker.pixel_tolerance}px")
    print(f"   - Min Overlap Points: {finger_checker.min_overlap_points}")
    print(f"   - Overlap Threshold: {finger_checker.overlap_thresh*100:.1f}%")
    print(f"   - Score Threshold: {finger_checker.score_thresh}")
    
    # Load image
    print(f"\n3. Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Detect card
    print("\n4. Detecting card with YOLO...")
    success, detection, message = yolo_service.detect_card(image)
    print(f"   {message}")
    
    if not success or detection is None:
        print("❌ Card detection failed")
        return
    
    polygon = detection.get("polygon")
    if polygon is None:
        print("❌ No polygon found in detection")
        return
    
    print(f"   Polygon points: {len(polygon)}")
    print(f"   Confidence: {detection['confidence']:.2f}")
    
    # Test finger detection
    print("\n5. Testing finger detection...")
    passed, score, message, metrics = finger_checker.check(polygon, image.shape)
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"\nStatus: {'✅ PASSED' if passed else '❌ FAILED (Finger Detected)'}")
    print(f"Score: {score:.3f} (threshold: {finger_checker.score_thresh})")
    print(f"Message: {message}")
    
    print(f"\nMetrics:")
    for key, value in metrics.items():
        if key not in ['inner_box', 'outer_box']:
            print(f"  {key}: {value}")
    
    # Visualize
    print(f"\n6. Creating visualization...")
    annotated = finger_checker.visualize(image, polygon, score, passed)
    
    # Save output
    output_path = f"finger_test_output_{Path(image_path).stem}.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"   Saved to: {output_path}")
    
    # Show interpretation
    print(f"\n{'='*70}")
    print(f"INTERPRETATION")
    print(f"{'='*70}")
    
    overlap_count = metrics.get('overlap_points_count', 0)
    overlap_ratio = metrics.get('overlap_ratio', 0)
    total_points = metrics.get('total_polygon_points', 0)
    
    print(f"\nOverlap Analysis:")
    print(f"  Total polygon points: {total_points}")
    print(f"  Points outside safe zone: {overlap_count}")
    print(f"  Overlap ratio: {overlap_ratio*100:.1f}%")
    
    if overlap_count == 0:
        print(f"\n  ℹ️ No overlap detected - card is fully within safe zone")
    elif overlap_count < finger_checker.min_overlap_points:
        print(f"\n  ℹ️ Overlap detected but below minimum ({overlap_count} < {finger_checker.min_overlap_points})")
        print(f"     This is likely edge noise, not a finger")
    else:
        print(f"\n  ⚠️ Significant overlap detected ({overlap_count} points)")
        print(f"     This indicates possible finger or object intrusion")
    
    print(f"\nShape Analysis:")
    print(f"  Vertices: {metrics.get('vertex_count', 0)} (ideal: 4-6)")
    print(f"  Curvature: {metrics.get('curvature_deviation', 0):.1f}° (threshold: {finger_checker.curvature_threshold}°)")
    
    if metrics.get('vertex_count', 0) > 7:
        print(f"  ⚠️ High vertex count suggests irregular shape")
    
    if metrics.get('curvature_deviation', 0) > finger_checker.curvature_threshold:
        print(f"  ⚠️ High curvature suggests curved/bent edges")
    
    print(f"\n{'='*70}\n")
    
    # Display image (optional)
    print("Press 'q' to close the image window...")
    resized = cv2.resize(annotated, (min(1200, annotated.shape[1]), 
                                     min(900, annotated.shape[0])))
    cv2.imshow("Finger Detection Test", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_finger_detection.py <image_path>")
        print("Example: python test_finger_detection.py /path/to/card.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"❌ File not found: {image_path}")
        sys.exit(1)
    
    test_finger_detection(image_path)
