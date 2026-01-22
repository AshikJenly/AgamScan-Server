"""
Pydantic models for request and response schemas
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x: int
    y: int
    width: int
    height: int


class QualityCheckResult(BaseModel):
    """Result of a single quality check"""
    passed: bool
    score: float
    threshold: float
    message: str


class QualityChecks(BaseModel):
    """All quality check results"""
    blur: QualityCheckResult
    glare: QualityCheckResult
    finger: QualityCheckResult


class OCRWord(BaseModel):
    """Individual OCR word with confidence"""
    text: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None


class OCRLine(BaseModel):
    """OCR line with words"""
    text: str
    words: List[OCRWord]
    bounding_box: Optional[BoundingBox] = None


class OCRResult(BaseModel):
    """Complete OCR result"""
    lines: List[OCRLine]
    full_text: str


class ExtractedField(BaseModel):
    """Extracted field with value and confidence"""
    value: str
    confidence: float


class NERResult(BaseModel):
    """Named Entity Recognition results"""
    fields: Dict[str, ExtractedField]


class ProcessingError(BaseModel):
    """Error information"""
    stage: str
    error: str
    details: Optional[str] = None


class ProcessResponse(BaseModel):
    """Main API response"""
    success: bool
    stage_completed: str = Field(
        description="Last successfully completed stage: detection, quality_check, ocr, ner, complete"
    )
    
    # Detection results
    card_detected: bool = False
    detection_confidence: Optional[float] = None
    
    # Quality check results
    quality_checks: Optional[QualityChecks] = None
    quality_passed: bool = False
    
    # OCR results
    ocr_result: Optional[OCRResult] = None
    
    # NER results
    ner_result: Optional[NERResult] = None
    
    # Visualization
    annotated_image_base64: Optional[str] = Field(
        None,
        description="Base64 encoded annotated image with boxes"
    )
    
    # Error information
    error: Optional[ProcessingError] = None
    
    # Processing metadata
    processing_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    yolo_model_loaded: bool
    azure_configured: bool
    version: str


class ConfigResponse(BaseModel):
    """Configuration information response"""
    thresholds: Dict[str, Any]
    model_info: Dict[str, str]
