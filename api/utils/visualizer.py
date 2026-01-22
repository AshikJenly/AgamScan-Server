"""
Visualization utilities for drawing bounding boxes and annotations
"""

import cv2
import numpy as np
import base64
from typing import Dict, List, Any, Optional


class Visualizer:
    """Utility class for visualizing detection and OCR results"""
    
    @staticmethod
    def encode_image_to_base64(image: np.ndarray) -> str:
        """
        Encode image to base64 string
        
        Args:
            image: BGR image
            
        Returns:
            Base64 encoded string
        """
        success, encoded = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(encoded).decode('utf-8')
    
    @staticmethod
    def decode_base64_to_image(base64_str: str) -> np.ndarray:
        """
        Decode base64 string to image
        
        Args:
            base64_str: Base64 encoded image string
            
        Returns:
            BGR image
        """
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    @staticmethod
    def draw_bbox(
        image: np.ndarray,
        bbox: List[int],
        color: tuple = (0, 255, 0),
        thickness: int = 2,
        label: Optional[str] = None
    ) -> np.ndarray:
        """
        Draw bounding box on image
        
        Args:
            image: BGR image
            bbox: [x1, y1, x2, y2] or [x, y, width, height]
            color: BGR color tuple
            thickness: Line thickness
            label: Optional label text
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Handle both formats
        if len(bbox) == 4:
            x1, y1 = bbox[0], bbox[1]
            # Check if it's [x, y, w, h] or [x1, y1, x2, y2]
            if bbox[2] < image.shape[1] and bbox[3] < image.shape[0]:
                # Likely [x, y, w, h]
                x2, y2 = x1 + bbox[2], y1 + bbox[3]
            else:
                # Likely [x1, y1, x2, y2]
                x2, y2 = bbox[2], bbox[3]
        else:
            raise ValueError("bbox must be [x1, y1, x2, y2] or [x, y, width, height]")
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        if label:
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        return annotated
    
    @staticmethod
    def draw_polygon(
        image: np.ndarray,
        polygon: np.ndarray,
        color: tuple = (255, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw polygon on image
        
        Args:
            image: BGR image
            polygon: Array of polygon points
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        cv2.polylines(annotated, [polygon], True, color, thickness)
        return annotated
    
    @staticmethod
    def draw_ocr_boxes(
        image: np.ndarray,
        ocr_data: Dict[str, Any],
        draw_lines: bool = True,
        draw_words: bool = False
    ) -> np.ndarray:
        """
        Draw OCR bounding boxes on image
        
        Args:
            image: BGR image
            ocr_data: OCR data from OCR service
            draw_lines: Whether to draw line boxes
            draw_words: Whether to draw word boxes
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Draw line boxes
        if draw_lines:
            for line in ocr_data.get("lines", []):
                bbox = line.get("bounding_box")
                if bbox:
                    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                    cv2.rectangle(
                        annotated,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )
        
        # Draw word boxes
        if draw_words:
            for word in ocr_data.get("words", []):
                bbox = word.get("bounding_box")
                if bbox:
                    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                    conf = word.get("confidence", 0)
                    color = (0, 255, 0) if conf > 0.8 else (0, 165, 255)
                    cv2.rectangle(
                        annotated,
                        (x, y),
                        (x + w, y + h),
                        color,
                        1
                    )
        
        return annotated
    
    @staticmethod
    def draw_quality_status(
        image: np.ndarray,
        quality_checks: Dict[str, Any],
        y_offset: int = 30
    ) -> np.ndarray:
        """
        Draw quality check results on image
        
        Args:
            image: BGR image
            quality_checks: Quality check results
            y_offset: Y position offset
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]
        
        # Semi-transparent overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, y_offset + 100), (0, 0, 0), -1)
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
        
        y = y_offset
        for check_name, check_result in quality_checks.items():
            passed = check_result.get("passed", False)
            score = check_result.get("score", 0)
            threshold = check_result.get("threshold", 0)
            
            color = (0, 255, 0) if passed else (0, 0, 255)
            status = "✓" if passed else "✗"
            
            text = f"{status} {check_name.upper()}: {score:.1f} (thresh: {threshold:.1f})"
            cv2.putText(
                annotated,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )
            y += 30
        
        return annotated
    
    @staticmethod
    def create_error_visualization(
        image: np.ndarray,
        error_stage: str,
        error_message: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Create visualization for errors (blur, glare, finger detection)
        
        Args:
            image: Original image
            error_stage: Stage where error occurred
            error_message: Error message
            additional_data: Additional data (detection, mask, etc.)
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]
        
        # Draw error banner
        cv2.rectangle(annotated, (0, 0), (w, 80), (0, 0, 255), -1)
        cv2.putText(
            annotated,
            f"FAILED: {error_stage.upper()}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            annotated,
            error_message,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        return annotated


if __name__ == "__main__":
    # Test visualizer
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test bbox drawing
    annotated = Visualizer.draw_bbox(
        test_img,
        [100, 100, 400, 300],
        color=(0, 255, 0),
        label="Test Card"
    )
    print("✅ Bbox drawing test passed")
    
    # Test base64 encoding/decoding
    encoded = Visualizer.encode_image_to_base64(test_img)
    decoded = Visualizer.decode_base64_to_image(encoded)
    print(f"✅ Base64 encoding test passed (length: {len(encoded)})")
    
    # Test quality status
    quality_checks = {
        "blur": {"passed": True, "score": 150.5, "threshold": 100.0},
        "glare": {"passed": False, "score": 8.2, "threshold": 5.0}
    }
    annotated = Visualizer.draw_quality_status(test_img, quality_checks)
    print("✅ Quality status drawing test passed")
