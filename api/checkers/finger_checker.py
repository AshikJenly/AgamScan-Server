"""
Finger detection checker
Detects finger overlap on card using:
1. Convex hull area difference method (for obvious bends/irregularities)
2. MediaPipe hand landmark detection (for subtle finger presence)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path

from config import (
    OVERLAP_AREA_THRESHOLD  # 2.5% area difference indicates overlap (stricter detection)
)

# Convex hull area difference method settings
MIN_CONTOUR_AREA = 5000  # Minimum contour area to process

# MediaPipe lazy loading (only imported if hand detection is needed)
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not available. Finger detection will use only area difference method.")

# Fingertip landmark indices (MediaPipe standard)
FINGERTIPS = [4, 8, 12, 16, 20]


class FingerChecker:
    """Check for finger overlap on card using hybrid detection methods"""
    
    def __init__(
        self,
        overlap_area_threshold: float = OVERLAP_AREA_THRESHOLD,
        min_contour_area: float = MIN_CONTOUR_AREA,
        hand_model_path: str = "models/hand_landmarker.task",
        enable_hand_detection: bool = True
    ):
        """
        Initialize finger checker with hybrid detection
        
        Args:
            overlap_area_threshold: Area difference ratio threshold (default: 0.025 = 2.5%)
            min_contour_area: Minimum contour area to process (default: 5000 pixels)
            hand_model_path: Path to MediaPipe hand landmark model
            enable_hand_detection: Whether to use MediaPipe hand detection (default: True)
        """
        self.overlap_area_threshold = overlap_area_threshold
        self.min_contour_area = min_contour_area
        self.enable_hand_detection = enable_hand_detection
        self.hand_landmarker = None
        
        # Initialize MediaPipe hand detector if available and enabled
        if enable_hand_detection and MEDIAPIPE_AVAILABLE:
            try:
                model_path = Path(hand_model_path)
                if model_path.exists():
                    base_options = python.BaseOptions(model_asset_path=str(model_path))
                    hand_options = vision.HandLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.IMAGE,
                        num_hands=2,
                        min_hand_detection_confidence=0.3,
                        min_hand_presence_confidence=0.3,
                        min_tracking_confidence=0.3
                    )
                    self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
                    print("✅ MediaPipe hand detector initialized")
                else:
                    print(f"⚠️ Hand model not found at {model_path}. Using area-only detection.")
            except Exception as e:
                print(f"⚠️ Failed to initialize MediaPipe: {e}. Using area-only detection.")
        else:
            print("ℹ️ MediaPipe hand detection disabled. Using area difference method only.")
    
    def detect_finger_by_polygon_shape(
        self,
        poly_points: np.ndarray,
        image_shape: tuple
    ) -> Tuple[bool, float, Optional[np.ndarray], List, dict]:
        """
        Detect if card polygon has finger overlap using convex hull area difference method
        
        This method compares the actual contour area with its convex hull area.
        If a finger overlaps the card, the difference will be significant.
        
        Args:
            poly_points: Polygon points from segmentation mask
            image_shape: Shape of the image (height, width, channels)
            
        Returns:
            Tuple of (overlap_detected, area_diff_ratio, contour, hull, metrics)
        """
        try:
            # Create mask from polygon
            mask = np.zeros(image_shape[:2], np.uint8)
            cv2.fillPoly(mask, [poly_points], 255)
            
            # Find contours
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return False, 0.0, None, [], {}
            
            # Get largest contour
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            
            # Check if contour area is too small
            if area < self.min_contour_area:
                return False, 0.0, contour, [], {
                    "contour_area": area,
                    "reason": "Contour area too small"
                }
            
            # Approximate contour for smoother edges
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Compute convex hull
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            # Calculate area difference ratio
            # If finger overlaps, hull will be larger than actual contour
            area_diff_ratio = (hull_area - area) / hull_area if hull_area > 0 else 0.0
            
            # Detect overlap based on area difference threshold
            overlap_detected = area_diff_ratio > self.overlap_area_threshold
            
            # Collect metrics for reporting
            metrics = {
                "contour_area": round(area, 2),
                "hull_area": round(hull_area, 2),
                "area_diff_ratio": round(area_diff_ratio, 4),
                "area_diff_percentage": round(area_diff_ratio * 100, 2),
                "threshold": self.overlap_area_threshold,
                "threshold_percentage": round(self.overlap_area_threshold * 100, 2),
                "vertex_count": len(approx)
            }
            
            # Debug logging
            print(f"    [Finger Check] Area: {area:.2f}, Hull: {hull_area:.2f}, "
                  f"Diff Ratio: {area_diff_ratio:.4f} ({area_diff_ratio*100:.2f}%), "
                  f"Overlap: {'✅ YES' if overlap_detected else '❌ NO'}")
            
            return overlap_detected, area_diff_ratio, contour, hull, metrics
            
        except Exception as e:
            print(f"Warning: Finger detection failed - {str(e)}")
            return False, 0.0, None, [], {}
    
    def detect_fingertips_in_polygon(
        self,
        image: np.ndarray,
        poly_points: np.ndarray
    ) -> Tuple[bool, int, List[Tuple[int, int]]]:
        """
        Detect actual fingertips within the card polygon using MediaPipe
        
        Args:
            image: BGR image
            poly_points: Card polygon points
            
        Returns:
            Tuple of (fingers_detected, fingertip_count, fingertip_coords)
        """
        if not self.hand_landmarker:
            return False, 0, []
        
        try:
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
            
            # Detect hands
            result = self.hand_landmarker.detect(mp_image)
            
            if not result.hand_landmarks:
                return False, 0, []
            
            # Create polygon mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly_points], 255)
            
            # Check if any fingertips are inside the polygon
            fingertips_in_polygon = []
            
            for hand_landmarks in result.hand_landmarks:
                for tip_idx in FINGERTIPS:
                    lm = hand_landmarks[tip_idx]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    
                    # Check if fingertip is within image bounds
                    if 0 <= x < w and 0 <= y < h:
                        # Check if fingertip is inside the card polygon
                        if mask[y, x] > 0:
                            fingertips_in_polygon.append((x, y))
            
            fingers_detected = len(fingertips_in_polygon) > 0
            
            if fingers_detected:
                print(f"    [Hand Detection] Found {len(fingertips_in_polygon)} fingertip(s) on card")
            
            return fingers_detected, len(fingertips_in_polygon), fingertips_in_polygon
            
        except Exception as e:
            print(f"    [Hand Detection] Failed: {str(e)}")
            return False, 0, []
    
    def check(
        self,
        polygon_points: np.ndarray,
        image_shape: tuple,
        image: Optional[np.ndarray] = None,
        is_final: bool = False
    ) -> Tuple[bool, float, str, dict]:
        """
        Check if finger is detected on card using hybrid detection:
        1. Convex hull area difference (for obvious bends)
        2. MediaPipe hand detection (for subtle finger presence)
        
        Logic:
        - If is_final=True: ONLY use MediaPipe hand detection (skip contour-based check)
        - If is_final=False (default): Use hybrid detection
          - If area difference is significant (>2.5%), always flag as finger
          - If area difference is small (<2.5%), check for actual fingertips using MediaPipe
          - Only flag as finger if fingertips are actually detected on card
        
        Args:
            polygon_points: Polygon points from YOLO segmentation
            image_shape: Image shape
            image: Optional BGR image for hand detection
            is_final: If True, only use MediaPipe detection (skip contour-based check)
            
        Returns:
            Tuple of (passed, area_diff_ratio, message, metrics)
        """
        try:
            # Initialize variables
            overlap_detected = False
            area_diff_ratio = 0.0
            metrics = {}
            fingertips_detected = False
            fingertip_count = 0
            fingertip_coords = []
            
            # SPECIAL MODE: If is_final=True, ONLY use MediaPipe detection
            if is_final:
                print("    [Finger Check] is_final=True: Using ONLY MediaPipe detection")
                
                # Only perform MediaPipe hand detection
                if self.hand_landmarker and image is not None:
                    fingertips_detected, fingertip_count, fingertip_coords = self.detect_fingertips_in_polygon(
                        image,
                        polygon_points
                    )
                    
                    metrics = {
                        "hand_detection_enabled": True,
                        "fingertips_detected": fingertips_detected,
                        "fingertip_count": fingertip_count,
                        "area_diff_ratio": 0.0,
                        "area_diff_percentage": 0.0,
                        "detection_mode": "mediapipe_only"
                    }
                else:
                    # MediaPipe not available, cannot perform check
                    metrics = {
                        "hand_detection_enabled": False,
                        "detection_mode": "mediapipe_only",
                        "error": "MediaPipe not available for is_final check"
                    }
                
                # Decision: Pass only if no fingertips detected
                passed = not fingertips_detected
                detection_method = "hand_landmarks_only" if fingertips_detected else "no_fingers_detected"
                
            else:
                # NORMAL MODE: Use hybrid detection (area difference + MediaPipe)
                # Step 1: Check area difference
                overlap_detected, area_diff_ratio, _, _, metrics = self.detect_finger_by_polygon_shape(
                    polygon_points,
                    image_shape
                )
                
                # Step 2: If area difference is small but non-zero, verify with hand detection
                if self.hand_landmarker and image is not None:
                    if area_diff_ratio > 0.005:  # Only check if there's some irregularity (>0.5%)
                        fingertips_detected, fingertip_count, fingertip_coords = self.detect_fingertips_in_polygon(
                            image,
                            polygon_points
                        )
                        
                        # Update metrics with hand detection results
                        metrics.update({
                            "hand_detection_enabled": True,
                            "fingertips_detected": fingertips_detected,
                            "fingertip_count": fingertip_count
                        })
                else:
                    metrics["hand_detection_enabled"] = False
                
                # Decision logic:
                # 1. If area difference is significant (>threshold), flag as finger
                # 2. If area difference is small but fingertips are detected, flag as finger
                # 3. Otherwise, pass
                
                if overlap_detected:
                    # Obvious bend/irregularity detected
                    passed = False
                    detection_method = "area_difference"
                elif fingertips_detected:
                    # Small irregularity + actual fingertips detected
                    passed = False
                    detection_method = "hand_landmarks"
                else:
                    # Clean card or small irregularity without fingertips
                    passed = True
                    detection_method = "both_checks_passed"
            
            # Generate detailed message
            if passed:
                if is_final:
                    message = (f"✅ No finger detected "
                              f"(MediaPipe only: no fingertips found)")
                elif self.hand_landmarker and image is not None:
                    message = (f"✅ No finger detected "
                              f"(area: {area_diff_ratio*100:.2f}%, no fingertips found)")
                else:
                    message = (f"✅ No finger detected "
                              f"(area diff: {area_diff_ratio*100:.2f}%, threshold: {self.overlap_area_threshold*100:.2f}%)")
            else:
                if is_final:
                    message = (f"🖐️ Finger detected via MediaPipe only "
                              f"({fingertip_count} fingertip(s) on card)")
                elif detection_method == "area_difference":
                    message = (f"🖐️ Finger detected via area difference "
                              f"({area_diff_ratio*100:.2f}% > {self.overlap_area_threshold*100:.2f}%)")
                else:
                    message = (f"🖐️ Finger detected via hand landmarks "
                              f"({fingertip_count} fingertip(s) on card, area: {area_diff_ratio*100:.2f}%)")
            
            metrics["detection_method"] = detection_method
            
            return passed, area_diff_ratio, message, metrics
            
        except Exception as e:
            return False, 0.0, f"❌ Finger check failed: {str(e)}", {}
    
    def visualize(
        self,
        image: np.ndarray,
        polygon_points: np.ndarray,
        score: float,
        passed: bool
    ) -> np.ndarray:
        """
        Annotate image with finger detection result showing both methods
        
        Args:
            image: Original BGR image
            polygon_points: Polygon points from segmentation
            score: Area difference ratio
            passed: Whether check passed
            
        Returns:
            Annotated image with contour, convex hull, and fingertips highlighted
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]
        
        # Get detection details
        overlap_detected, area_diff_ratio, contour, hull, metrics = self.detect_finger_by_polygon_shape(
            polygon_points,
            image.shape
        )
        
        # Get hand detection results
        fingertips_detected = False
        fingertip_coords = []
        if self.hand_landmarker:
            fingertips_detected, fingertip_count, fingertip_coords = self.detect_fingertips_in_polygon(
                image,
                polygon_points
            )
        
        # Draw contour (actual card boundary) in GREEN
        if contour is not None:
            cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
        
        # Draw convex hull in RED (shows where finger might be)
        if hull is not None:
            cv2.drawContours(annotated, [hull], -1, (0, 0, 255), 2)
        
        # Draw fingertips if detected
        if fingertip_coords:
            for x, y in fingertip_coords:
                cv2.circle(annotated, (x, y), 8, (255, 0, 255), -1)  # Magenta circles
                cv2.circle(annotated, (x, y), 10, (255, 255, 255), 2)  # White outline
        
        # Add status overlay
        color = (0, 255, 0) if passed else (0, 0, 255)
        status = "CLEAN" if passed else "FINGER DETECTED"
        
        # Calculate overlay height based on content
        overlay_height = 170 if fingertip_coords else 140
        
        # Semi-transparent background for text
        cv2.rectangle(annotated, (10, 10), (min(w - 10, 650), overlay_height), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (min(w - 10, 650), overlay_height), color, 2)
        
        # Main status
        cv2.putText(
            annotated,
            f"Finger Check: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )
        
        # Area difference ratio
        cv2.putText(
            annotated,
            f"Area Diff: {area_diff_ratio*100:.2f}% (Threshold: {self.overlap_area_threshold*100:.2f}%)",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        # Hand detection status
        if self.hand_landmarker:
            hand_status = f"Fingertips: {len(fingertip_coords)} detected" if fingertip_coords else "Fingertips: None"
            hand_color = (255, 0, 255) if fingertip_coords else (255, 255, 255)
            cv2.putText(
                annotated,
                hand_status,
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                hand_color,
                1,
                cv2.LINE_AA
            )
            y_offset = 120
        else:
            y_offset = 95
        
        # Contour and Hull areas
        if metrics:
            cv2.putText(
                annotated,
                f"Contour: {metrics.get('contour_area', 0):.0f}px | Hull: {metrics.get('hull_area', 0):.0f}px",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        # Legend
        legend_y = h - 20
        if fingertip_coords:
            cv2.putText(
                annotated,
                "Green: Contour | Red: Hull | Magenta: Fingertips",
                (20, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )
        else:
            cv2.putText(
                annotated,
                "Green: Contour | Red: Hull",
                (20, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )
        
        return annotated


if __name__ == "__main__":
    # Test finger checker with convex hull area difference method
    checker = FingerChecker()
    
    # Create test image and polygon
    test_img = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
    
    # Clean rectangular polygon (should have low area difference)
    clean_poly = np.array([
        [100, 100],
        [500, 100],
        [500, 400],
        [100, 400]
    ], dtype=np.int32)
    
    print("Testing Clean Rectangular Card:")
    passed, ratio, msg, metrics = checker.check(clean_poly, test_img.shape)
    print(f"  {msg}")
    print(f"  Metrics: {metrics}")
    
    # Irregular polygon (simulating finger - creates area difference)
    # When finger overlaps, the convex hull will be larger than actual contour
    irregular_poly = np.array([
        [100, 100],
        [250, 80],   # Indent (simulates finger creating concavity)
        [350, 80],
        [500, 100],
        [500, 400],
        [350, 420],  # Another indent
        [250, 420],
        [100, 400]
    ], dtype=np.int32)
    
    print("\nTesting Card with Finger Overlap:")
    passed, ratio, msg, metrics = checker.check(irregular_poly, test_img.shape)
    print(f"  {msg}")
    print(f"  Metrics: {metrics}")
