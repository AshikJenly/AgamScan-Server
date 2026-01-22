"""
Blur detection checker
Detects blurry images using Laplacian variance
"""

import cv2
import numpy as np
from typing import Tuple

from config import BLUR_VARIANCE_THRESH


class BlurChecker:
    """Check if image is too blurry"""
    
    def __init__(self, threshold: float = BLUR_VARIANCE_THRESH):
        """
        Initialize blur checker
        
        Args:
            threshold: Minimum variance threshold for acceptable sharpness
        """
        self.threshold = threshold
    
    def variance_of_laplacian(self, gray: np.ndarray) -> float:
        """
        Calculate variance of Laplacian to measure blur
        
        Args:
            gray: Grayscale image
            
        Returns:
            Variance value (higher = sharper)
        """
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def check(self, image: np.ndarray) -> Tuple[bool, float, str]:
        """
        Check if image is acceptably sharp
        
        Args:
            image: BGR image (from cv2)
            
        Returns:
            Tuple of (passed, score, message)
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate blur score
            blur_score = self.variance_of_laplacian(gray)
            
            # Check if passed
            passed = blur_score >= self.threshold
            
            # Generate message
            if passed:
                message = f"✅ Image is sharp (score: {blur_score:.1f})"
            else:
                message = f"❌ Image is blurry (score: {blur_score:.1f}, threshold: {self.threshold})"
            
            return passed, blur_score, message
            
        except Exception as e:
            return False, 0.0, f"❌ Blur check failed: {str(e)}"
    
    def visualize(self, image: np.ndarray, blur_score: float, passed: bool) -> np.ndarray:
        """
        Annotate image with blur detection result
        
        Args:
            image: Original BGR image
            blur_score: Calculated blur score
            passed: Whether check passed
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]
        
        # Add text overlay
        color = (0, 255, 0) if passed else (0, 0, 255)
        status = "SHARP" if passed else "BLURRY"
        
        cv2.rectangle(annotated, (10, 10), (w - 10, 80), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (w - 10, 80), color, 2)
        
        cv2.putText(
            annotated, 
            f"Blur Check: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )
        
        cv2.putText(
            annotated,
            f"Score: {blur_score:.1f} (Threshold: {self.threshold})",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        return annotated


if __name__ == "__main__":
    # Test blur checker
    checker = BlurChecker()
    
    # Create test images
    sharp_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    blurry_img = cv2.GaussianBlur(sharp_img, (21, 21), 0)
    
    print("Testing Sharp Image:")
    passed, score, msg = checker.check(sharp_img)
    print(f"  {msg}")
    
    print("\nTesting Blurry Image:")
    passed, score, msg = checker.check(blurry_img)
    print(f"  {msg}")
