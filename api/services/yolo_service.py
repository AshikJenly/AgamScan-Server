"""
YOLO service for card detection
Handles loading YOLO model and performing card detection
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from ultralytics import YOLO

from config import MODEL_PATH, YOLO_CONFIDENCE_THRESH, MIN_CARD_AREA, MASK_COVERAGE_THRESH


class YOLOService:
    """Service for YOLO-based card detection"""
    
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        confidence_threshold: float = YOLO_CONFIDENCE_THRESH
    ):
        """
        Initialize YOLO service
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detection
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        
    def load_model(self):
        """Load YOLO model"""
        try:
            print(f"🚀 Loading YOLO model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            print("✅ YOLO model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def detect_card(
        self,
        image: np.ndarray
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Detect card in image
        
        Args:
            image: BGR image from cv2
            
        Returns:
            Tuple of (success, detection_data, message)
            detection_data contains:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - mask: binary mask
                - polygon: polygon points from segmentation
                - mask_area: area of mask
                - coverage: mask coverage ratio
        """
        if not self.is_loaded():
            return False, None, "❌ YOLO model not loaded"
        
        try:
            h, w = image.shape[:2]
            
            # Run YOLO prediction
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                show=False,
                verbose=False
            )
            
            r = results[0]
            
            # Find best card detection
            best_det = None
            
            if r.boxes is not None:
                for i, box in enumerate(r.boxes):
                    cls_name = self.model.names[int(box.cls)].lower()
                    
                    # Look for "card" class
                    if "card" not in cls_name:
                        continue
                    
                    conf = float(box.conf)
                    
                    # Keep highest confidence detection
                    if best_det is None or conf > best_det["confidence"]:
                        bbox = [int(v) for v in box.xyxy[0].cpu().numpy()]
                        
                        # Get mask if available
                        mask = None
                        polygon = None
                        if r.masks is not None:
                            mask_data = r.masks.data[i].cpu().numpy()
                            mask = cv2.resize(
                                mask_data,
                                (w, h),
                                interpolation=cv2.INTER_NEAREST
                            )
                            mask = (mask > 0.5).astype(np.uint8)
                            
                            # Get polygon points
                            polygon = np.int32(r.masks.xy[i])
                        
                        best_det = {
                            "bbox": bbox,
                            "confidence": conf,
                            "mask": mask,
                            "polygon": polygon,
                            "mask_index": i
                        }
            
            # Check if card detected
            if best_det is None:
                return False, None, "❌ No card detected in image"
            
            # Validate detection quality
            x1, y1, x2, y2 = best_det["bbox"]
            bbox_area = max(1, (x2 - x1) * (y2 - y1))
            
            if best_det["mask"] is not None:
                mask_area = np.sum(best_det["mask"])
                coverage = np.sum(best_det["mask"][y1:y2, x1:x2]) / bbox_area
                
                best_det["mask_area"] = int(mask_area)
                best_det["coverage"] = float(coverage)
                
                # Check minimum area
                if mask_area < MIN_CARD_AREA:
                    return False, best_det, f"❌ Card too small (area: {mask_area}, min: {MIN_CARD_AREA})"
                
                # Check mask coverage
                if coverage < MASK_COVERAGE_THRESH:
                    return False, best_det, f"❌ Poor mask coverage ({coverage:.2f}, min: {MASK_COVERAGE_THRESH})"
            
            return True, best_det, f"✅ Card detected (confidence: {best_det['confidence']:.2f})"
            
        except Exception as e:
            return False, None, f"❌ Detection failed: {str(e)}"
    
    def crop_card(
        self,
        image: np.ndarray,
        detection: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        Crop card region from image using mask
        
        Args:
            image: Original image
            detection: Detection data from detect_card
            
        Returns:
            Cropped card image or None
        """
        try:
            if detection["mask"] is None:
                # Fallback to bbox crop
                x1, y1, x2, y2 = detection["bbox"]
                return image[y1:y2, x1:x2]
            
            # Crop using mask
            mask = detection["mask"]
            ys, xs = np.where(mask > 0)
            
            if len(xs) == 0:
                return None
            
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            
            return image[y1:y2, x1:x2]
            
        except Exception as e:
            print(f"Warning: Crop failed - {str(e)}")
            return None


if __name__ == "__main__":
    # Test YOLO service
    service = YOLOService()
    
    if service.load_model():
        print("\n✅ YOLO service initialized successfully")
        
        # Create test image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("\nTesting detection on dummy image...")
        success, detection, message = service.detect_card(test_img)
        print(f"  {message}")
        
        if success:
            print(f"  Bbox: {detection['bbox']}")
            print(f"  Confidence: {detection['confidence']:.2f}")
            if detection['mask'] is not None:
                print(f"  Mask area: {detection['mask_area']}")
                print(f"  Coverage: {detection['coverage']:.2f}")
    else:
        print("\n❌ Failed to initialize YOLO service")
