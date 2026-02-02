"""
AgamScan API - Main FastAPI Application
Document card processing pipeline with YOLO detection, quality checks, OCR, and NER
"""

import time
import traceback
from io import BytesIO
from typing import Optional
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import models
from models.schemas import (
    ProcessResponse,
    QualityCheckResult,
    QualityChecks,
    OCRResult,
    OCRLine,
    OCRWord,
    NERResult,
    ExtractedField,
    ProcessingError,
    HealthResponse,
    ConfigResponse,
    BoundingBox
)

# Import services and checkers
from services.yolo_service import YOLOService
from services.ocr_service import OCRService
from services.ner_service import NERService
from checkers.blur_checker import BlurChecker
from checkers.glare_checker import GlareChecker
from checkers.finger_checker import FingerChecker
from utils.visualizer import Visualizer

# Import config
import config

# Initialize FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (will be loaded on startup)
yolo_service: Optional[YOLOService] = None
ocr_service: Optional[OCRService] = None
ner_service: Optional[NERService] = None
blur_checker: Optional[BlurChecker] = None
glare_checker: Optional[GlareChecker] = None
finger_checker: Optional[FingerChecker] = None
visualizer = Visualizer()


def save_processing_result(
    image: np.ndarray,
    filename: str,
    success: bool,
    stage: str,
    details: dict = None
):
    """
    Save processing result to outputs directory
    
    Args:
        image: Image to save
        filename: Original filename
        success: Whether processing succeeded
        stage: Processing stage completed
        details: Additional details to log
    """
    try:
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(filename).stem
        
        # Determine output directory
        if success:
            output_dir = config.PASSED_DIR
            prefix = "passed"
        else:
            output_dir = config.FAILED_DIR
            prefix = "failed"
        
        # Create filename with stage info
        output_filename = f"{prefix}_{stage}_{base_name}_{timestamp}.jpg"
        output_path = output_dir / output_filename
        
        # Save image
        cv2.imwrite(str(output_path), image)
        
        # Save details as JSON (with proper serialization)
        if details:
            json_path = output_dir / f"{prefix}_{stage}_{base_name}_{timestamp}.json"
            
            # Convert numpy types to Python native types for JSON serialization
            def convert_to_serializable(obj):
                """Recursively convert numpy types to Python native types"""
                if isinstance(obj, dict):
                    return {key: convert_to_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                else:
                    return obj
            
            serializable_details = convert_to_serializable(details)
            
            with open(json_path, 'w') as f:
                json.dump(serializable_details, f, indent=2)
        
        print(f"💾 Saved: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"⚠️ Failed to save result: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global yolo_service, ocr_service, ner_service
    global blur_checker, glare_checker, finger_checker
    
    print("\n" + "="*60)
    print("🚀 Starting AgamScan API...")
    print("="*60)
    
    try:
        # Validate configuration
        config.validate_config()
        print("✅ Configuration validated")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("⚠️ API will start but may not function properly")
    
    # Initialize YOLO service
    print("\n📦 Initializing YOLO service...")
    yolo_service = YOLOService()
    if yolo_service.load_model():
        print("✅ YOLO service ready")
    else:
        print("❌ YOLO service failed to initialize")
    
    # Initialize OCR service
    print("\n📦 Initializing OCR service...")
    ocr_service = OCRService()
    if ocr_service.initialize():
        print("✅ OCR service ready")
    else:
        print("❌ OCR service failed to initialize")
    
    # Initialize NER service
    print("\n📦 Initializing NER service...")
    ner_service = NERService()
    if ner_service.initialize():
        print("✅ NER service ready")
    else:
        print("❌ NER service failed to initialize")
    
    # Initialize checkers
    print("\n📦 Initializing quality checkers...")
    blur_checker = BlurChecker()
    glare_checker = GlareChecker()
    finger_checker = FingerChecker()
    print("✅ Quality checkers ready")
    
    print("\n" + "="*60)
    print("✅ AgamScan API is running!")
    print(f"📖 API docs: http://{config.API_HOST}:{config.API_PORT}/docs")
    print("="*60 + "\n")


def read_image_from_upload(file: UploadFile) -> np.ndarray:
    """
    Read image from uploaded file
    
    Args:
        file: Uploaded file
        
    Returns:
        BGR image
    """
    # Read file bytes
    contents = file.file.read()
    
    # Convert to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image")
    
    return image


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "AgamScan API is running",
        "version": config.API_VERSION,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        yolo_model_loaded=yolo_service.is_loaded() if yolo_service else False,
        azure_configured=bool(config.AZURE_VISION_KEY and config.AZURE_AI_KEY),
        version=config.API_VERSION
    )


@app.get("/config", response_model=ConfigResponse, tags=["Configuration"])
async def get_config():
    """Get current configuration"""
    return ConfigResponse(
        thresholds={
            "yolo_confidence": config.YOLO_CONFIDENCE_THRESH,
            "blur_variance": config.BLUR_VARIANCE_THRESH,
            "glare_ratio": config.GLARE_RATIO_THRESH,
            "finger_overlap_area": config.OVERLAP_AREA_THRESHOLD,
            "min_card_area": config.MIN_CARD_AREA,
            "mask_coverage": config.MASK_COVERAGE_THRESH
        },
        model_info={
            "yolo_model": config.MODEL_PATH,
            "azure_ai_model": config.AZURE_AI_MODEL_NAME
        }
    )


@app.post("/process", response_model=ProcessResponse, tags=["Processing"])
async def process_document(
    file: UploadFile = File(..., description="Image file to process"),
    is_final: bool = False
):
    """
    Process document image through complete pipeline
    
    Parameters:
    - file: Image file to process
    - is_final: If False, only performs YOLO detection and quality checks (blur, glare, finger).
                If True, performs complete pipeline including OCR and NER.
    
    Pipeline stages:
    1. YOLO Detection - Detect card in image
    2. Quality Checks - Blur, Glare (on cropped card), Finger detection
    3. OCR - Extract text using Azure Document Intelligence (only if is_final=True)
    4. NER - Extract structured fields using Azure AI (only if is_final=True)
    5. Visualization - Annotate image with results
    
    Returns JSON with extracted data and annotated image
    """
    start_time = time.time()
    
    try:
        # Read image
        image = read_image_from_upload(file)
        print(f"\n📸 Processing image: {file.filename} ({image.shape})")
        
        # ========== STAGE 1: YOLO Detection ==========
        print("🔍 Stage 1: YOLO Detection...")
        success, detection, message = yolo_service.detect_card(image)
        print(f"   {message}")
        
        if not success:
            # Failed to detect card
            annotated = visualizer.create_error_visualization(
                image,
                "Card Detection",
                message,
                detection
            )
            
            return ProcessResponse(
                success=False,
                stage_completed="detection",
                card_detected=False,
                annotated_image_base64=visualizer.encode_image_to_base64(annotated),
                error=ProcessingError(
                    stage="detection",
                    error=message
                ),
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Card detected successfully
        detection_confidence = detection["confidence"]
        polygon = detection["polygon"]
        
        # Crop card region
        card_image = yolo_service.crop_card(image, detection)
        if card_image is None:
            card_image = image
        
        # ========== STAGE 2: Quality Checks ==========
        print("✓ Stage 2: Quality Checks...")
        
        # Blur check - on cropped card
        blur_passed, blur_score, blur_msg = blur_checker.check(card_image)
        print(f"   {blur_msg}")
        
        # Glare check - on cropped card (not full image)
        glare_passed, glare_score, glare_msg = glare_checker.check(card_image)
        print(f"   {glare_msg} (checked on cropped card)")
        
        # Finger check - on original polygon with full image for hand detection
        # Pass is_final parameter to control detection method
        finger_passed, finger_score, finger_msg, finger_metrics = finger_checker.check(
            polygon,
            image.shape,
            image,  # Pass full image for MediaPipe hand detection
            is_final=is_final  # If True, only use MediaPipe; if False, use hybrid detection
        )
        print(f"   {finger_msg}")
        if finger_metrics:
            print(f"      Metrics: {finger_metrics}")
        
        # Create quality checks result
        quality_checks = QualityChecks(
            blur=QualityCheckResult(
                passed=blur_passed,
                score=blur_score,
                threshold=config.BLUR_VARIANCE_THRESH,
                message=blur_msg
            ),
            glare=QualityCheckResult(
                passed=glare_passed,
                score=glare_score,
                threshold=config.GLARE_RATIO_THRESH,
                message=glare_msg
            ),
            finger=QualityCheckResult(
                passed=finger_passed,
                score=finger_score,
                threshold=config.OVERLAP_AREA_THRESHOLD,
                message=finger_msg
            )
        )
        
        quality_passed = blur_passed and glare_passed and finger_passed
        
        # If quality checks failed, return annotated image
        if not quality_passed:
            print("❌ Quality checks failed")
            
            # Visualize the failed check
            annotated = image.copy()
            
            if not blur_passed:
                annotated = blur_checker.visualize(annotated, blur_score, blur_passed)
            elif not glare_passed:
                annotated = glare_checker.visualize(annotated, glare_score, glare_passed)
            elif not finger_passed:
                annotated = finger_checker.visualize(
                    annotated,
                    polygon,
                    finger_score,
                    finger_passed
                )
            
            # Add card detection box
            annotated = visualizer.draw_polygon(annotated, polygon, (0, 0, 255), 3)
            
            # Save failed result
            save_details = {
                "filename": file.filename,
                "stage": "quality_check",
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "detection_confidence": detection_confidence,
                "is_final": is_final,
                "quality_checks": {
                    "blur": {"passed": blur_passed, "score": blur_score, "threshold": config.BLUR_VARIANCE_THRESH},
                    "glare": {"passed": glare_passed, "score": glare_score, "threshold": config.GLARE_RATIO_THRESH},
                    "finger": {"passed": finger_passed, "score": finger_score, "threshold": config.OVERLAP_AREA_THRESHOLD, "metrics": finger_metrics}
                }
            }
            save_processing_result(annotated, file.filename, False, "quality_check", save_details)
            
            return ProcessResponse(
                success=False,
                stage_completed="quality_check",
                card_detected=True,
                detection_confidence=detection_confidence,
                quality_checks=quality_checks,
                quality_passed=False,
                annotated_image_base64=visualizer.encode_image_to_base64(annotated),
                error=ProcessingError(
                    stage="quality_check",
                    error="Quality checks failed",
                    details=f"Blur: {blur_passed}, Glare: {glare_passed}, Finger: {finger_passed}"
                ),
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        print("✅ Quality checks passed")
        
        # If is_final is False, stop here and return quality check results
        if not is_final:
            print("⏸️  is_final=False, stopping after quality checks")
            
            # Create simple visualization with card detection box
            annotated = image.copy()
            annotated = visualizer.draw_polygon(annotated, polygon, (0, 255, 0), 3)
            
            # Add success indicator
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0, 0), (w, 50), (0, 255, 0), -1)
            cv2.putText(
                annotated,
                "✓ QUALITY CHECKS PASSED",
                (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Save preliminary result
            save_details = {
                "filename": file.filename,
                "stage": "quality_check_passed",
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": processing_time,
                "detection_confidence": detection_confidence,
                "is_final": False,
                "quality_checks": {
                    "blur": {"passed": blur_passed, "score": blur_score, "threshold": config.BLUR_VARIANCE_THRESH},
                    "glare": {"passed": glare_passed, "score": glare_score, "threshold": config.GLARE_RATIO_THRESH},
                    "finger": {"passed": finger_passed, "score": finger_score, "threshold": config.OVERLAP_AREA_THRESHOLD, "metrics": finger_metrics}
                }
            }
            save_processing_result(annotated, file.filename, True, "quality_check_passed", save_details)
            
            return ProcessResponse(
                success=True,
                stage_completed="quality_check",
                card_detected=True,
                detection_confidence=detection_confidence,
                quality_checks=quality_checks,
                quality_passed=True,
                annotated_image_base64=visualizer.encode_image_to_base64(annotated),
                processing_time_ms=processing_time
            )
        
        # ========== STAGE 3: OCR ==========
        print("📝 Stage 3: OCR...")
        ocr_success, ocr_data, ocr_msg = ocr_service.extract_text(card_image)
        print(f"   {ocr_msg}")
        
        if not ocr_success:
            annotated = visualizer.create_error_visualization(
                image,
                "OCR",
                ocr_msg
            )
            
            return ProcessResponse(
                success=False,
                stage_completed="ocr",
                card_detected=True,
                detection_confidence=detection_confidence,
                quality_checks=quality_checks,
                quality_passed=True,
                annotated_image_base64=visualizer.encode_image_to_base64(annotated),
                error=ProcessingError(
                    stage="ocr",
                    error=ocr_msg
                ),
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Parse OCR result
        ocr_lines = []
        for line_data in ocr_data["lines"]:
            words = []
            for word_data in line_data["words"]:
                bbox = word_data.get("bounding_box")
                words.append(OCRWord(
                    text=word_data["text"],
                    confidence=word_data["confidence"],
                    bounding_box=BoundingBox(**bbox) if bbox else None
                ))
            
            bbox = line_data.get("bounding_box")
            ocr_lines.append(OCRLine(
                text=line_data["text"],
                words=words,
                bounding_box=BoundingBox(**bbox) if bbox else None
            ))
        
        ocr_result = OCRResult(
            lines=ocr_lines,
            full_text=ocr_data["full_text"]
        )
        
        # ========== STAGE 4: NER ==========
        print("🧠 Stage 4: NER (Field Extraction)...")
        ner_success, ner_fields, ner_msg = ner_service.extract_fields(ocr_data)
        print(f"   {ner_msg}")
        
        if not ner_success:
            # NER failed, but we have OCR results
            # Return with OCR visualization
            annotated = card_image.copy()
            annotated = visualizer.draw_ocr_boxes(annotated, ocr_data, draw_lines=True)
            
            # Save failed result
            save_details = {
                "filename": file.filename,
                "stage": "ner",
                "ocr_lines_count": len(ocr_data["lines"]),
                "error": ner_msg
            }
            save_processing_result(annotated, file.filename, False, "ner", save_details)
            
            return ProcessResponse(
                success=False,
                stage_completed="ner",
                card_detected=True,
                detection_confidence=detection_confidence,
                quality_checks=quality_checks,
                quality_passed=True,
                ocr_result=ocr_result,
                annotated_image_base64=visualizer.encode_image_to_base64(annotated),
                error=ProcessingError(
                    stage="ner",
                    error=ner_msg
                ),
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Parse NER result
        ner_result = NERResult(
            fields={
                field_name: ExtractedField(**field_data)
                for field_name, field_data in ner_fields.items()
            }
        )
        
        # ========== STAGE 5: Visualization ==========
        print("🎨 Stage 5: Creating visualization...")
        annotated = card_image.copy()
        annotated = visualizer.draw_ocr_boxes(annotated, ocr_data, draw_lines=True)
        
        # Add success indicator
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 50), (0, 255, 0), -1)
        cv2.putText(
            annotated,
            "✓ PROCESSING COMPLETE",
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        processing_time = (time.time() - start_time) * 1000
        print(f"✅ Processing complete ({processing_time:.0f}ms)")
        
        # Save successful result
        filled_fields = sum(1 for f in ner_fields.values() if f["value"])
        save_details = {
            "filename": file.filename,
            "stage": "complete",
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": processing_time,
            "detection_confidence": detection_confidence,
            "is_final": is_final,
            "quality_checks": {
                "blur": {"passed": blur_passed, "score": blur_score, "threshold": config.BLUR_VARIANCE_THRESH},
                "glare": {"passed": glare_passed, "score": glare_score, "threshold": config.GLARE_RATIO_THRESH},
                "finger": {"passed": finger_passed, "score": finger_score, "threshold": config.OVERLAP_AREA_THRESHOLD, "metrics": finger_metrics}
            },
            "ocr_result": {
                "lines_count": len(ocr_data["lines"]),
                "full_text": ocr_data["full_text"],
                "lines": ocr_data["lines"]
            },
            "ner_result": {
                "fields_extracted": filled_fields,
                "total_fields": len(ner_fields),
                "fields": ner_fields
            }
        }
        save_processing_result(annotated, file.filename, True, "complete", save_details)
        
        # Return complete result
        return ProcessResponse(
            success=True,
            stage_completed="complete",
            card_detected=True,
            detection_confidence=detection_confidence,
            quality_checks=quality_checks,
            quality_passed=True,
            ocr_result=ocr_result,
            ner_result=ner_result,
            annotated_image_base64=visualizer.encode_image_to_base64(annotated),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    print(f"\n🚀 Starting AgamScan API server...")
    print(f"📍 Host: {config.API_HOST}")
    print(f"📍 Port: {config.API_PORT}")
    print(f"📖 Docs: http://{config.API_HOST}:{config.API_PORT}/docs\n")
    
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )
