# AgamScan API

**Document Card Processing API** with YOLO detection, quality checks (blur, glare, finger detection), OCR, and NER field extraction.

## 🌟 Features

- **YOLO Card Detection**: Automatically detect and segment card/document in images
- **Quality Checks**:
  - ✅ Blur detection (Laplacian variance)
  - ✅ Glare detection (HSV-based bright region detection)
  - ✅ Finger detection (Polygon curvature analysis)
- **OCR**: Text extraction using Azure Document Intelligence
- **NER**: Structured field extraction using Azure AI LLM
- **Visualization**: Annotated images with bounding boxes and quality indicators
- **FastAPI**: Modern, fast API with automatic OpenAPI/Swagger documentation

## 📋 Pipeline Flow

```
Image Upload
    ↓
1. YOLO Detection (Card detection)
    ↓ [If failed] → Return error with visualization
2. Quality Checks (Blur, Glare, Finger)
    ↓ [If failed] → Return error with annotated problem area
3. OCR (Text extraction)
    ↓ [If failed] → Return error
4. NER (Field extraction)
    ↓ [If failed] → Return with OCR results
5. Success → Return complete data + annotated image
```

## 🚀 Setup

### Prerequisites

- Python 3.8+
- Azure Account with:
  - Azure Vision API (for OCR)
  - Azure AI Services (for NER)
- Trained YOLO segmentation model

### Installation

1. **Clone the repository**
   ```bash
   cd /home/ashikjenly/__/AgamScan-App/api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials and paths
   ```

5. **Set up your `.env` file**
   ```env
   # Azure Credentials
   AZURE_VISION_KEY=your_vision_key_here
   AZURE_VISION_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
   AZURE_AI_ENDPOINT=https://your-ai-resource.services.ai.azure.com/models
   AZURE_AI_KEY=your_ai_key_here
   
   # YOLO Model
   YOLO_MODEL_PATH=/path/to/your/yolo/model/best.pt
   
   # Adjust thresholds as needed
   BLUR_VARIANCE_THRESH=100.0
   GLARE_RATIO_THRESH=5.0
   FINGER_CURVATURE_THRESH=18.0
   ```

## 🎯 Usage

### Start the API Server

```bash
cd /home/ashikjenly/__/AgamScan-App/api
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### 1. **POST /process** - Process Document Image

Upload an image and process through the complete pipeline.

**Request:**
```bash
curl -X POST "http://localhost:8000/process" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

**Response (Success):**
```json
{
  "success": true,
  "stage_completed": "complete",
  "card_detected": true,
  "detection_confidence": 0.95,
  "quality_checks": {
    "blur": {
      "passed": true,
      "score": 150.5,
      "threshold": 100.0,
      "message": "✅ Image is sharp"
    },
    "glare": {
      "passed": true,
      "score": 3.2,
      "threshold": 5.0,
      "message": "✅ Acceptable lighting"
    },
    "finger": {
      "passed": true,
      "score": 12.5,
      "threshold": 18.0,
      "message": "✅ No finger detected"
    }
  },
  "quality_passed": true,
  "ocr_result": {
    "lines": [...],
    "full_text": "..."
  },
  "ner_result": {
    "fields": {
      "First Name": {
        "value": "JOHN",
        "confidence": 0.98
      },
      "Last Name": {
        "value": "DOE",
        "confidence": 0.97
      },
      ...
    }
  },
  "annotated_image_base64": "base64_encoded_image_with_boxes",
  "processing_time_ms": 1234.5
}
```

**Response (Failed - Quality Check):**
```json
{
  "success": false,
  "stage_completed": "quality_check",
  "card_detected": true,
  "detection_confidence": 0.92,
  "quality_checks": {
    "blur": {
      "passed": false,
      "score": 75.3,
      "threshold": 100.0,
      "message": "❌ Image is blurry"
    },
    ...
  },
  "quality_passed": false,
  "annotated_image_base64": "base64_image_showing_blur_issue",
  "error": {
    "stage": "quality_check",
    "error": "Quality checks failed",
    "details": "Blur: False, Glare: True, Finger: True"
  }
}
```

#### 2. **GET /health** - Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "yolo_model_loaded": true,
  "azure_configured": true,
  "version": "1.0.0"
}
```

#### 3. **GET /config** - Get Configuration

```bash
curl http://localhost:8000/config
```

### Python Client Example

```python
import requests

# Process an image
with open("card_image.jpg", "rb") as f:
    files = {"file": ("card.jpg", f, "image/jpeg")}
    response = requests.post("http://localhost:8000/process", files=files)
    
result = response.json()

if result["success"]:
    print("Processing successful!")
    print(f"Extracted fields: {result['ner_result']['fields']}")
    
    # Decode and save annotated image
    import base64
    img_data = base64.b64decode(result["annotated_image_base64"])
    with open("annotated.jpg", "wb") as f:
        f.write(img_data)
else:
    print(f"Failed at stage: {result['stage_completed']}")
    print(f"Error: {result['error']['error']}")
```

## 📁 Project Structure

```
api/
├── app.py                      # Main FastAPI application
├── config.py                   # Configuration management
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── .env                       # Your environment variables (git-ignored)
│
├── models/
│   └── schemas.py             # Pydantic models for API
│
├── services/
│   ├── yolo_service.py        # YOLO detection service
│   ├── ocr_service.py         # Azure OCR service
│   └── ner_service.py         # Azure NER service
│
├── checkers/
│   ├── blur_checker.py        # Blur detection
│   ├── glare_checker.py       # Glare detection
│   └── finger_checker.py      # Finger detection
│
└── utils/
    └── visualizer.py          # Image annotation utilities
```

## 🔧 Customization

### Adding New Quality Checkers

1. Create a new checker in `checkers/` directory:

```python
# checkers/my_checker.py
class MyChecker:
    def __init__(self, threshold):
        self.threshold = threshold
    
    def check(self, image):
        # Your logic here
        score = ...
        passed = score >= self.threshold
        message = "..." 
        return passed, score, message
    
    def visualize(self, image, score, passed):
        # Annotate image
        return annotated_image
```

2. Import and use in `app.py`:

```python
from checkers.my_checker import MyChecker

# In startup_event()
my_checker = MyChecker(threshold=10.0)

# In process_document()
my_passed, my_score, my_msg = my_checker.check(image)
```

### Adjusting Thresholds

Edit your `.env` file or modify `config.py`:

```env
# Lower = more strict, Higher = more lenient
BLUR_VARIANCE_THRESH=100.0
GLARE_RATIO_THRESH=5.0
FINGER_CURVATURE_THRESH=18.0
```

## 🐛 Troubleshooting

### YOLO Model Not Loading
- Verify `YOLO_MODEL_PATH` in `.env` points to a valid `.pt` file
- Ensure the model is a segmentation model (not detection only)

### Azure Services Failing
- Check your Azure credentials in `.env`
- Verify your subscription has access to the services
- Check quota limits in Azure portal

### Image Processing Errors
- Supported formats: JPEG, PNG, BMP, TIFF
- Recommended: Good lighting, minimal glare, card fully visible
- Image resolution: At least 640x480 for good results

## 📊 Performance Tips

- **GPU Acceleration**: YOLO runs faster on GPU. Install CUDA-enabled PyTorch:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

- **Image Size**: Resize large images before processing to speed up:
  ```python
  # In preprocessing
  max_size = 1920
  if image.shape[1] > max_size:
      scale = max_size / image.shape[1]
      image = cv2.resize(image, None, fx=scale, fy=scale)
  ```

## 📝 License

This project is part of AgamScan-App.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📧 Support

For issues and questions, please open an issue on the repository.

---

**Built with ❤️ using FastAPI, YOLO, and Azure AI**
