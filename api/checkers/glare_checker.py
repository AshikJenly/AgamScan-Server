"""
Glare detection checker
Detects bright glare/reflection regions in images
"""

import cv2
import numpy as np
from typing import Tuple

from config import (
    GLARE_RATIO_THRESH,
    GLARE_BRIGHTNESS_THRESH,
    GLARE_SATURATION_THRESH
)


class GlareChecker:
    """Check for glare/bright reflections in image"""
    
    def __init__(
        self,
        ratio_threshold: float = GLARE_RATIO_THRESH,
        brightness_threshold: int = GLARE_BRIGHTNESS_THRESH,
        saturation_threshold: int = GLARE_SATURATION_THRESH
    ):
        """
        Initialize glare checker
        
        Args:
            ratio_threshold: Maximum acceptable glare ratio (%)
            brightness_threshold: Minimum brightness for glare detection (0-255)
            saturation_threshold: Maximum saturation for glare (low = white/glare)
        """
        self.ratio_threshold = ratio_threshold
        self.brightness_threshold = brightness_threshold
        self.saturation_threshold = saturation_threshold
    
    def detect_glare(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Detect glare regions in image
        
        Args:
            image: BGR image
            
        Returns:
            Tuple of (glare_ratio_percent, glare_mask)
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Create masks for bright regions
        glare_mask = cv2.inRange(v, self.brightness_threshold, 255)
        
        # Create mask for low saturation (whitish areas)
        low_sat_mask = cv2.inRange(s, 0, self.saturation_threshold)
        
        # Combine both: bright + low saturation = glare
        combined_mask = cv2.bitwise_and(glare_mask, low_sat_mask)
        
        # Morphological operations to clean noise
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate glare ratio
        total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
        glare_pixels = np.sum(combined_mask > 0)
        glare_ratio = (glare_pixels / total_pixels) * 100
        
        return glare_ratio, combined_mask
    
    def check(self, image: np.ndarray) -> Tuple[bool, float, str]:
        """
        Check if image has acceptable glare levels
        
        Args:
            image: BGR image
            
        Returns:
            Tuple of (passed, glare_ratio, message)
        """
        try:
            glare_ratio, _ = self.detect_glare(image)
            
            # Check if passed
            passed = glare_ratio <= self.ratio_threshold
            
            # Generate message
            if passed:
                message = f"✅ Acceptable lighting (glare: {glare_ratio:.2f}%)"
            else:
                message = f"❌ Too much glare detected ({glare_ratio:.2f}%, threshold: {self.ratio_threshold}%)"
            
            return passed, glare_ratio, message
            
        except Exception as e:
            return False, 0.0, f"❌ Glare check failed: {str(e)}"
    
    def visualize(self, image: np.ndarray, glare_ratio: float, passed: bool) -> np.ndarray:
        """
        Annotate image with glare regions highlighted
        
        Args:
            image: Original BGR image
            glare_ratio: Calculated glare ratio
            passed: Whether check passed
            
        Returns:
            Annotated image with glare regions highlighted in red
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]
        
        # Detect glare regions
        _, glare_mask = self.detect_glare(image)
        
        # Overlay glare regions in red
        overlay = annotated.copy()
        overlay[glare_mask > 0] = (0, 0, 255)  # Red for glare areas
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
        
        # Add text overlay
        color = (0, 255, 0) if passed else (0, 0, 255)
        status = "OK" if passed else "GLARE DETECTED"
        
        cv2.rectangle(annotated, (10, 10), (w - 10, 80), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (w - 10, 80), color, 2)
        
        cv2.putText(
            annotated,
            f"Glare Check: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )
        
        cv2.putText(
            annotated,
            f"Glare: {glare_ratio:.2f}% (Threshold: {self.ratio_threshold}%)",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        return annotated


if __name__ == "__main__":
    # Test glare checker
    checker = GlareChecker()
    
    # Create test image with artificial glare
    test_img = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
    # Add bright white region (glare)
    test_img[100:200, 300:450] = 255
    
    print("Testing Image with Glare:")
    passed, ratio, msg = checker.check(test_img)
    print(f"  {msg}")
    
    # Test normal image
    normal_img = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
    print("\nTesting Normal Image:")
    passed, ratio, msg = checker.check(normal_img)
    print(f"  {msg}")
