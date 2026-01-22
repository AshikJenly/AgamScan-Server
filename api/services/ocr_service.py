"""
OCR service using Azure Document Intelligence (Vision API)
Extracts text from images
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

from config import AZURE_VISION_KEY, AZURE_VISION_ENDPOINT


class OCRService:
    """Service for OCR using Azure Document Intelligence"""
    
    def __init__(
        self,
        endpoint: str = AZURE_VISION_ENDPOINT,
        key: str = AZURE_VISION_KEY
    ):
        """
        Initialize OCR service
        
        Args:
            endpoint: Azure Vision endpoint
            key: Azure Vision API key
        """
        self.endpoint = endpoint
        self.key = key
        self.client = None
    
    def initialize(self) -> bool:
        """Initialize Azure client"""
        try:
            if not self.endpoint or not self.key:
                print("❌ Azure Vision credentials not configured")
                return False
            
            self.client = ImageAnalysisClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.key)
            )
            print("✅ OCR service initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize OCR service: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self.client is not None
    
    def extract_text(
        self,
        image: np.ndarray
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Extract text from image using Azure Vision
        
        Args:
            image: BGR image from cv2
            
        Returns:
            Tuple of (success, ocr_data, message)
            ocr_data contains:
                - lines: list of text lines with bounding boxes
                - words: list of words with confidence scores
                - full_text: concatenated text
                - raw_result: raw Azure response
        """
        if not self.is_initialized():
            return False, None, "❌ OCR service not initialized"
        
        try:
            # Convert image to bytes
            success, encoded_image = cv2.imencode('.jpg', image)
            if not success:
                return False, None, "❌ Failed to encode image"
            
            image_bytes = encoded_image.tobytes()
            
            # Analyze image
            result = self.client.analyze(
                image_data=image_bytes,
                visual_features=[VisualFeatures.READ],
                gender_neutral_caption=True
            )
            
            # Parse results
            lines_data = []
            words_data = []
            all_text = []
            
            if result.read is not None:
                for block in result.read.blocks:
                    for line in block.lines:
                        line_text = line.text.strip()
                        if not line_text:
                            continue
                        
                        all_text.append(line_text)
                        
                        # Parse line bounding box
                        line_bbox = None
                        if line.bounding_polygon:
                            points = [(p.x, p.y) for p in line.bounding_polygon]
                            if points:
                                xs = [p[0] for p in points]
                                ys = [p[1] for p in points]
                                line_bbox = {
                                    "x": int(min(xs)),
                                    "y": int(min(ys)),
                                    "width": int(max(xs) - min(xs)),
                                    "height": int(max(ys) - min(ys))
                                }
                        
                        # Parse words
                        line_words = []
                        for word in line.words:
                            word_text = word.text.strip()
                            word_conf = float(word.confidence) if hasattr(word, 'confidence') else 1.0
                            
                            word_bbox = None
                            if word.bounding_polygon:
                                points = [(p.x, p.y) for p in word.bounding_polygon]
                                if points:
                                    xs = [p[0] for p in points]
                                    ys = [p[1] for p in points]
                                    word_bbox = {
                                        "x": int(min(xs)),
                                        "y": int(min(ys)),
                                        "width": int(max(xs) - min(xs)),
                                        "height": int(max(ys) - min(ys))
                                    }
                            
                            word_data = {
                                "text": word_text,
                                "confidence": word_conf,
                                "bounding_box": word_bbox
                            }
                            
                            line_words.append(word_data)
                            words_data.append(word_data)
                        
                        lines_data.append({
                            "text": line_text,
                            "words": line_words,
                            "bounding_box": line_bbox
                        })
            
            if not lines_data:
                return False, None, "❌ No text detected in image"
            
            ocr_data = {
                "lines": lines_data,
                "words": words_data,
                "full_text": "\n".join(all_text),
                "raw_result": result
            }
            
            return True, ocr_data, f"✅ Extracted {len(lines_data)} lines, {len(words_data)} words"
            
        except Exception as e:
            return False, None, f"❌ OCR failed: {str(e)}"
    
    def get_text_confidence(self, ocr_data: Dict[str, Any]) -> float:
        """
        Calculate average confidence across all words
        
        Args:
            ocr_data: OCR data from extract_text
            
        Returns:
            Average confidence score
        """
        words = ocr_data.get("words", [])
        if not words:
            return 0.0
        
        confidences = [w["confidence"] for w in words]
        return sum(confidences) / len(confidences)


if __name__ == "__main__":
    # Test OCR service
    service = OCRService()
    
    if service.initialize():
        print("\n✅ OCR service initialized successfully")
        
        # Create test image with text
        test_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(
            test_img,
            "TEST DOCUMENT",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )
        
        print("\nTesting OCR on test image...")
        success, data, message = service.extract_text(test_img)
        print(f"  {message}")
        
        if success:
            print(f"  Full text: {data['full_text']}")
            print(f"  Avg confidence: {service.get_text_confidence(data):.2f}")
    else:
        print("\n❌ Failed to initialize OCR service")
