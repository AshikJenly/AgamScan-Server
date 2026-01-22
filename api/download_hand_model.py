"""
Download MediaPipe Hand Landmarker model
Run this once to download the model file needed for finger detection
"""

import urllib.request
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

def download_hand_model():
    """Download MediaPipe hand landmarker model"""
    
    # Create models directory if it doesn't exist
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Check if model already exists
    if MODEL_PATH.exists():
        print(f"✅ Model already exists at: {MODEL_PATH}")
        print(f"   File size: {MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    
    print(f"📥 Downloading MediaPipe hand landmarker model...")
    print(f"   URL: {MODEL_URL}")
    print(f"   Destination: {MODEL_PATH}")
    
    try:
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            mb_downloaded = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            print(f"\r   Progress: {percent:.1f}% ({mb_downloaded:.2f}/{mb_total:.2f} MB)", end="")
        
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=show_progress)
        print()  # New line after progress
        
        print(f"✅ Model downloaded successfully!")
        print(f"   File size: {MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download model: {e}")
        print("\nYou can manually download the model from:")
        print(f"   {MODEL_URL}")
        print(f"And place it at:")
        print(f"   {MODEL_PATH}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("MediaPipe Hand Landmarker Model Download")
    print("="*60)
    download_hand_model()
    print("="*60)
