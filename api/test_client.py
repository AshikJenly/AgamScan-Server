"""
Test client for AgamScan API
Simple script to test the /process endpoint
"""

import requests
import sys
import json
import base64
from pathlib import Path


def test_api(image_path: str, api_url: str = "http://localhost:8000"):
    """
    Test the API with an image file
    
    Args:
        image_path: Path to image file
        api_url: Base URL of the API
    """
    print(f"\n{'='*60}")
    print(f"Testing AgamScan API")
    print(f"{'='*60}")
    
    # Check if image exists
    img_file = Path(image_path)
    if not img_file.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return
    
    print(f"\n📸 Image: {img_file.name}")
    print(f"📍 API: {api_url}")
    
    # Test health endpoint
    print(f"\n🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"   Status: {health['status']}")
            print(f"   YOLO Model Loaded: {health['yolo_model_loaded']}")
            print(f"   Azure Configured: {health['azure_configured']}")
            print(f"   Version: {health['version']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"   Make sure the API server is running!")
        return
    
    # Test process endpoint
    print(f"\n🔄 Processing image...")
    try:
        with open(image_path, "rb") as f:
            files = {"file": (img_file.name, f, "image/jpeg")}
            response = requests.post(
                f"{api_url}/process",
                files=files,
                timeout=60
            )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n{'='*60}")
            print(f"RESULTS")
            print(f"{'='*60}")
            
            print(f"\n✅ Success: {result['success']}")
            print(f"📊 Stage Completed: {result['stage_completed']}")
            print(f"⏱️  Processing Time: {result.get('processing_time_ms', 0):.0f}ms")
            
            # Detection results
            if result.get('card_detected'):
                print(f"\n🔍 Detection:")
                print(f"   Card Detected: Yes")
                print(f"   Confidence: {result.get('detection_confidence', 0):.2f}")
            
            # Quality checks
            if result.get('quality_checks'):
                print(f"\n✓ Quality Checks:")
                for check_name, check_data in result['quality_checks'].items():
                    status = "✅" if check_data['passed'] else "❌"
                    print(f"   {status} {check_name.upper()}: {check_data['score']:.2f} (threshold: {check_data['threshold']:.2f})")
                    print(f"      {check_data['message']}")
            
            # OCR results
            if result.get('ocr_result'):
                ocr = result['ocr_result']
                print(f"\n📝 OCR Results:")
                print(f"   Lines: {len(ocr['lines'])}")
                print(f"   Text Preview:")
                print(f"   ---")
                for line in ocr['lines'][:5]:  # Show first 5 lines
                    print(f"   {line['text']}")
                if len(ocr['lines']) > 5:
                    print(f"   ... ({len(ocr['lines']) - 5} more lines)")
                print(f"   ---")
            
            # NER results
            if result.get('ner_result'):
                ner = result['ner_result']
                print(f"\n🧠 Extracted Fields:")
                for field_name, field_data in ner['fields'].items():
                    if field_data['value']:
                        print(f"   {field_name}: {field_data['value']} (conf: {field_data['confidence']:.2f})")
            
            # Error information
            if result.get('error'):
                error = result['error']
                print(f"\n❌ Error:")
                print(f"   Stage: {error['stage']}")
                print(f"   Error: {error['error']}")
                if error.get('details'):
                    print(f"   Details: {error['details']}")
            
            # Save annotated image
            if result.get('annotated_image_base64'):
                output_path = f"output_{img_file.stem}_annotated.jpg"
                img_data = base64.b64decode(result['annotated_image_base64'])
                with open(output_path, "wb") as f:
                    f.write(img_data)
                print(f"\n💾 Annotated image saved: {output_path}")
            
            # Save JSON response
            json_path = f"output_{img_file.stem}_result.json"
            # Remove base64 image from JSON to make it readable
            result_copy = result.copy()
            if 'annotated_image_base64' in result_copy:
                result_copy['annotated_image_base64'] = f"<base64_image_{len(result['annotated_image_base64'])}bytes>"
            
            with open(json_path, "w") as f:
                json.dump(result_copy, f, indent=2)
            print(f"💾 JSON result saved: {json_path}")
            
        else:
            print(f"\n❌ API Error: {response.status_code}")
            print(f"   {response.text}")
            
    except requests.exceptions.Timeout:
        print(f"\n❌ Request timeout (60s)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <image_path> [api_url]")
        print("Example: python test_client.py card.jpg")
        print("Example: python test_client.py card.jpg http://localhost:8000")
        sys.exit(1)
    
    image_path = sys.argv[1]
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    test_api(image_path, api_url)
