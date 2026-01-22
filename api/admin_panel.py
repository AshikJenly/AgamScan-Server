"""
Admin Panel for AgamScan
Monitors all request logs, images, and pipeline outputs
"""

from flask import Flask, render_template, jsonify, send_file, request
from pathlib import Path
import json
from datetime import datetime
import os
from typing import List, Dict, Any
import base64

app = Flask(__name__)

# Configuration
OUTPUTS_DIR = Path("outputs")
LOGS_DIR = OUTPUTS_DIR / "logs"
PASSED_DIR = OUTPUTS_DIR / "passed"
FAILED_DIR = OUTPUTS_DIR / "failed"

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class AdminMonitor:
    """Monitor and retrieve processing logs"""
    
    @staticmethod
    def get_all_requests(limit: int = 100, status: str = None) -> List[Dict[str, Any]]:
        """
        Get all request logs
        
        Args:
            limit: Maximum number of requests to return
            status: Filter by status ('passed', 'failed', or None for all)
            
        Returns:
            List of request logs with metadata
        """
        requests = []
        
        # Scan both passed and failed directories
        dirs_to_scan = []
        if status == 'passed' or status is None:
            dirs_to_scan.append(('passed', PASSED_DIR))
        if status == 'failed' or status is None:
            dirs_to_scan.append(('failed', FAILED_DIR))
        
        for status_label, directory in dirs_to_scan:
            if not directory.exists():
                continue
            
            # Get all JSON files (sorted by modification time, newest first)
            json_files = sorted(
                [f for f in directory.iterdir() if f.suffix == '.json'],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        log_data = json.load(f)
                    
                    # Generate request ID from filename (remove .json extension)
                    request_id = json_file.stem
                    
                    # Add metadata
                    log_data['request_id'] = request_id
                    log_data['status'] = status_label
                    log_data['directory'] = str(directory)
                    log_data['timestamp'] = log_data.get('timestamp', datetime.fromtimestamp(json_file.stat().st_mtime).isoformat())
                    
                    # Get corresponding image file (same name but .jpg)
                    image_file = json_file.with_suffix('.jpg')
                    
                    # Get file paths
                    log_data['files'] = {
                        'original': str(image_file) if image_file.exists() else None,
                        'yolo_detected': None,  # Not stored separately in current structure
                        'final': str(image_file) if image_file.exists() else None,
                    }
                    
                    requests.append(log_data)
                except Exception as e:
                    print(f"Error reading log {json_file}: {e}")
        
        # Sort by timestamp
        requests.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return requests[:limit]
    
    @staticmethod
    def get_request_details(request_id: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific request
        
        Args:
            request_id: Request ID (filename without extension)
            
        Returns:
            Detailed request information
        """
        # Search in both passed and failed directories
        for directory in [PASSED_DIR, FAILED_DIR]:
            json_file = directory / f"{request_id}.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        log_data = json.load(f)
                    
                    # Get corresponding image
                    image_file = json_file.with_suffix('.jpg')
                    
                    # Add file contents as base64
                    files = {}
                    if image_file.exists():
                        with open(image_file, 'rb') as f:
                            img_b64 = base64.b64encode(f.read()).decode('utf-8')
                            files['original'] = img_b64
                            files['final'] = img_b64  # Same image for now
                    
                    log_data['files_base64'] = files
                    log_data['request_id'] = request_id
                    log_data['directory'] = str(directory)
                    log_data['timestamp'] = log_data.get('timestamp', datetime.fromtimestamp(json_file.stat().st_mtime).isoformat())
                    
                    # Extract OCR and NER results if available
                    if 'ocr_result' in log_data:
                        log_data['has_ocr'] = True
                    if 'ner_result' in log_data:
                        log_data['has_ner'] = True
                    
                    # Determine success status
                    if 'success' not in log_data:
                        # Infer from directory or stage
                        log_data['success'] = 'passed' in str(directory)
                    
                    # Add error message if failed
                    if not log_data.get('success'):
                        stage = log_data.get('stage', 'unknown')
                        if stage == 'quality_check':
                            # Check which quality check failed
                            qc = log_data.get('quality_checks', {})
                            failed_checks = [k for k, v in qc.items() if not v.get('passed', True)]
                            if failed_checks:
                                log_data['error'] = f"{', '.join(failed_checks).title()} check(s) failed"
                            else:
                                log_data['error'] = "Quality check failed"
                        elif stage == 'yolo_detection':
                            log_data['error'] = "No card detected"
                        else:
                            log_data['error'] = f"Failed at {stage} stage"
                    
                    return log_data
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")
                    return None
        
        return None
    
    @staticmethod
    def get_statistics() -> Dict[str, Any]:
        """Get overall statistics"""
        # Count JSON files in each directory
        total_passed = len(list(PASSED_DIR.glob("*.json"))) if PASSED_DIR.exists() else 0
        total_failed = len(list(FAILED_DIR.glob("*.json"))) if FAILED_DIR.exists() else 0
        
        # Get failure reasons breakdown
        failure_reasons = {}
        if FAILED_DIR.exists():
            for json_file in FAILED_DIR.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        log_data = json.load(f)
                    
                    # Determine failure reason
                    stage = log_data.get('stage', 'Unknown')
                    if stage == 'quality_check':
                        qc = log_data.get('quality_checks', {})
                        failed_checks = [k for k, v in qc.items() if not v.get('passed', True)]
                        if failed_checks:
                            reason = f"{', '.join(failed_checks).title()} failed"
                        else:
                            reason = "Quality check failed"
                    elif stage == 'yolo_detection':
                        reason = "No card detected"
                    else:
                        reason = f"Failed at {stage}"
                    
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")
        
        return {
            'total_requests': total_passed + total_failed,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'success_rate': round((total_passed / max(1, total_passed + total_failed)) * 100, 2),
            'failure_reasons': failure_reasons
        }


# ============= API Routes =============

@app.route('/')
def index():
    """Main admin panel page"""
    return render_template('admin.html')


@app.route('/api/requests')
def get_requests():
    """Get all requests (with optional filtering)"""
    limit = int(request.args.get('limit', 100))
    status = request.args.get('status')  # 'passed', 'failed', or None
    
    requests = AdminMonitor.get_all_requests(limit=limit, status=status)
    return jsonify(requests)


@app.route('/api/request/<request_id>')
def get_request_detail(request_id):
    """Get detailed information for a specific request"""
    details = AdminMonitor.get_request_details(request_id)
    if details:
        return jsonify(details)
    return jsonify({'error': 'Request not found'}), 404


@app.route('/api/statistics')
def get_statistics():
    """Get overall statistics"""
    stats = AdminMonitor.get_statistics()
    return jsonify(stats)


@app.route('/api/image/<path:filepath>')
def get_image(filepath):
    """Serve image files"""
    try:
        return send_file(filepath, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/download/<request_id>/<filename>')
def download_file(request_id, filename):
    """Download a specific file from a request"""
    for directory in [PASSED_DIR, FAILED_DIR]:
        # Try with request_id as the full filename
        file_path = directory / request_id
        if file_path.exists():
            return send_file(file_path, as_attachment=True)
        
        # Try with request_id + filename
        file_path = directory / f"{request_id}.{filename}"
        if file_path.exists():
            return send_file(file_path, as_attachment=True)
    
    return jsonify({'error': 'File not found'}), 404


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'admin-panel'})


if __name__ == '__main__':
    print("=" * 60)
    print("🎛️  AgamScan Admin Panel")
    print("=" * 60)
    print(f"📊 Monitoring directory: {OUTPUTS_DIR.absolute()}")
    print(f"🌐 Starting server on http://localhost:5001")
    print(f"📝 Access admin panel at http://localhost:5001")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8006, debug=True)
