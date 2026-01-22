"""
Configuration file for AgamScan environment variables and sets default values
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== Azure Credentials ====================
AZURE_VISION_KEY = os.getenv("AZURE_VISION_KEY", "")
AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT", "")
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT", "")
AZURE_AI_KEY = os.getenv("AZURE_AI_KEY", "")
AZURE_AI_MODEL_NAME = os.getenv("AZURE_AI_MODEL_NAME", "Phi-4-mini-instruct")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-05-01-preview")

# ==================== YOLO Model Configuration ====================
MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    "/home/ashikjenly/__/DocScan-App/notebooks/datas/card_dataset_annotated_v1/runs/segment/train3/weights/best.pt"
)
YOLO_CONFIDENCE_THRESH = float(os.getenv("YOLO_CONFIDENCE_THRESH", "0.90"))

# ==================== Quality Check Thresholds ====================
# Blur Detection
BLUR_VARIANCE_THRESH = float(os.getenv("BLUR_VARIANCE_THRESH", "100.0"))

# Glare Detection
GLARE_RATIO_THRESH = float(os.getenv("GLARE_RATIO_THRESH", "5.0"))
GLARE_BRIGHTNESS_THRESH = int(os.getenv("GLARE_BRIGHTNESS_THRESH", "220"))
GLARE_SATURATION_THRESH = int(os.getenv("GLARE_SATURATION_THRESH", "50"))

# Finger Detection (Convex Hull Area Difference Method)
OVERLAP_AREA_THRESHOLD = float(os.getenv("OVERLAP_AREA_THRESHOLD", "0.025"))  # 2.5% area difference threshold

# Card Quality Checks
MIN_CARD_AREA = int(os.getenv("MIN_CARD_AREA", "5000"))
MASK_COVERAGE_THRESH = float(os.getenv("MASK_COVERAGE_THRESH", "0.85"))

# ==================== Output Directories ====================
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
FAILED_DIR = OUTPUT_DIR / "failed"
PASSED_DIR = OUTPUT_DIR / "passed"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
FAILED_DIR.mkdir(exist_ok=True)
PASSED_DIR.mkdir(exist_ok=True)

# ==================== NER Configuration ====================
REQUIRED_FIELDS = {
    "Issuing Authority": "",
    "First Name": "",
    "Last Name": "",
    "Licence Number": "",
    "Date Of Birth": "",
    "Date Of Expiry": "",
    "Gender": "",
    "Address": "",
    "Country": ""
}

# ==================== API Configuration ====================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_TITLE = "AgamScan - Document Processing API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
AgamScan API for document card detection, quality checks, OCR, and field extraction.

Pipeline:
1. YOLO Detection - Detect card in image
2. Quality Checks - Blur, Glare, Finger detection
3. OCR - Extract text using Azure Document Intelligence
4. NER - Extract structured fields using Azure AI
5. Return - JSON response with extracted data and annotated image
"""

# ==================== Validation ====================
def validate_config():
    """Validate that required configuration is present"""
    missing = []
    
    if not AZURE_VISION_KEY:
        missing.append("AZURE_VISION_KEY")
    if not AZURE_VISION_ENDPOINT:
        missing.append("AZURE_VISION_ENDPOINT")
    if not AZURE_AI_ENDPOINT:
        missing.append("AZURE_AI_ENDPOINT")
    if not AZURE_AI_KEY:
        missing.append("AZURE_AI_KEY")
    if not Path(MODEL_PATH).exists():
        missing.append(f"YOLO_MODEL_PATH (file not found: {MODEL_PATH})")
    
    if missing:
        raise ValueError(
            f"Missing required configuration: {', '.join(missing)}\n"
            "Please check your .env file or environment variables."
        )

if __name__ == "__main__":
    # Test configuration
    print("=== AgamScan Configuration ===")
    print(f"YOLO Model: {MODEL_PATH}")
    print(f"Azure Vision Endpoint: {AZURE_VISION_ENDPOINT}")
    print(f"Azure AI Endpoint: {AZURE_AI_ENDPOINT}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("\n=== Quality Thresholds ===")
    print(f"Blur Threshold: {BLUR_VARIANCE_THRESH}")
    print(f"Glare Threshold: {GLARE_RATIO_THRESH}%")
    print(f"Finger Overlap Threshold: {OVERLAP_AREA_THRESHOLD * 100}%")
    try:
        validate_config()
        print("\n✅ Configuration is valid")
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
