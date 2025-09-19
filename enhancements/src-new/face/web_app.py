"""
🚀 Y.M.I.R Web-based Emotion Detection Flask App
===============================================
Web interface for the enhanced facial emotion recognition system.
"""

from flask import Flask, render_template_string, jsonify, request, Response
import cv2
import json
import threading
import time
import numpy as np
from datetime import datetime
import base64
import io
from PIL import Image

# Import our enhanced emotion detection system
from fer_enhanced_v3 import EnhancedEmotionDetector, EnhancedEmotionConfig

app = Flask(__name__)

class WebEmotionSystem:
    """Web-based emotion detection system"""
    
    def __init__(self):
        self.detector = None
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.current_emotions = {}
        self.current_objects = []
        self.current_environment = {}
        self.session_analytics = {
            'total_readings': 0,
            'confidence_sum': 0,
            'quality_sum': 0,
            'stability_sum': 0,
            'start_time': None
        }
        self.frame_lock = threading.Lock()
        
        # Initialize the enhanced detector
        self._init_detector()
    
    def _init_detector(self):
        """Initialize the enhanced emotion detector"""
        try:
            config = EnhancedEmotionConfig(
                camera_width=640,
                camera_height=480,
                emotion_analysis_interval=30,
                require_user_consent=False,  # Web handles permissions
                use_firebase=True
            )
            
            self.detector = EnhancedEmotionDetector(config)
            print("✅ Enhanced Emotion Detector initialized for web")
            
        except Exception as e:
            print(f"❌ Detector initialization error: {e}")
            self.detector = None
    
    def start_camera(self):
        """Start camera capture"""
        if self.is_running:
            return True
            
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return False
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            self.session_analytics['start_time'] = time.time()
            
            # Start processing thread
            threading.Thread(target=self._process_frames, daemon=True).start()
            
            print("✅ Web camera started")
            return True
            
        except Exception as e:
            print(f"❌ Camera start error: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        print("🛑 Web camera stopped")
    
    def _process_frames(self):
        """Process camera frames with emotion detection"""
        frame_count = 0
        
        while self.is_running and self.cap:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process with enhanced detector if available
                if self.detector:
                    # Store original frame for web display
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                    
                    # Process with our enhanced system (simplified for web)
                    self._process_with_enhanced_detector(frame, frame_count)
                else:
                    # Fallback: just store the frame
                    with self.frame_lock:
                        self.current_frame = frame
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"⚠️ Frame processing error: {e}")
                continue
    
    def _process_with_enhanced_detector(self, frame, frame_count):
        """Process frame with the enhanced detector (simplified for web)"""
        try:
            # Simulate the enhanced processing workflow
            # MediaPipe processing
            if hasattr(self.detector, 'mediapipe_processor'):
                mediapipe_results = self.detector.mediapipe_processor.process_frame(frame)
                faces = mediapipe_results.get('faces', [])
            else:
                faces = []
            
            # YOLO processing (every 15 frames)
            if frame_count % 15 == 0 and hasattr(self.detector, 'yolo_processor'):
                try:
                    objects, environment_context = self.detector.yolo_processor.detect_objects_with_emotion_context(frame)
                    self.current_objects = objects[:10]  # Limit for web display
                    self.current_environment = environment_context
                except Exception as e:
                    print(f"⚠️ YOLO processing error: {e}")
            
            # Emotion analysis (every 30 frames)
            if frame_count % 30 == 0 and faces and hasattr(self.detector, 'deepface_ensemble'):
                try:
                    face_info = faces[0]  # Process first face
                    if face_info['roi'].size > 0:
                        emotion_result = self.detector.deepface_ensemble.analyze_face_with_context(
                            face_info['id'], face_info['roi'], self.current_environment
                        )
                        
                        if emotion_result:
                            # Update current emotions for web display
                            self.current_emotions = {
                                'dominant': max(emotion_result['emotions'].items(), key=lambda x: x[1]),
                                'all_emotions': emotion_result['emotions'],
                                'confidence': emotion_result['confidence'],
                                'quality': face_info.get('quality_score', 0.8),
                                'stability': emotion_result.get('stability', 0.0),
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Update analytics
                            self._update_analytics(emotion_result, face_info.get('quality_score', 0.8))
                            
                except Exception as e:
                    print(f"⚠️ Emotion analysis error: {e}")
            
        except Exception as e:
            print(f"⚠️ Enhanced detector processing error: {e}")
    
    def _update_analytics(self, emotion_result, quality_score):
        """Update session analytics"""
        self.session_analytics['total_readings'] += 1
        self.session_analytics['confidence_sum'] += emotion_result['confidence']
        self.session_analytics['quality_sum'] += quality_score
        self.session_analytics['stability_sum'] += emotion_result.get('stability', 0.0)
    
    def get_current_frame_jpeg(self):
        """Get current frame as JPEG bytes"""
        with self.frame_lock:
            if self.current_frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    return buffer.tobytes()
        return None
    
    def get_emotion_data(self):
        """Get current emotion data for API"""
        if not self.current_emotions:
            return {
                'status': 'waiting',
                'message': 'Waiting for emotion detection...'
            }
        
        return {
            'status': 'active',
            'emotions': self.current_emotions,
            'objects': [{'class': obj.get('class', 'unknown'), 'confidence': obj.get('confidence', 0)} 
                       for obj in self.current_objects],
            'environment': self.current_environment,
            'analytics': self._get_analytics_summary()
        }
    
    def _get_analytics_summary(self):
        """Get analytics summary"""
        total = self.session_analytics['total_readings']
        if total == 0:
            return {
                'total_readings': 0,
                'avg_confidence': 0,
                'avg_quality': 0,
                'avg_stability': 0,
                'session_duration': 0
            }
        
        session_duration = 0
        if self.session_analytics['start_time']:
            session_duration = time.time() - self.session_analytics['start_time']
        
        return {
            'total_readings': total,
            'avg_confidence': self.session_analytics['confidence_sum'] / total,
            'avg_quality': self.session_analytics['quality_sum'] / total,
            'avg_stability': self.session_analytics['stability_sum'] / total,
            'session_duration': session_duration
        }

# Global web emotion system
web_system = WebEmotionSystem()

def get_embedded_html():
    """Return embedded HTML as fallback"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Y.M.I.R Web Emotion Detection</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
            color: #f0f0f0;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            text-align: center;
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 20px;
            background: linear-gradient(45deg, #4070ff, #00d4ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .card {
            background: rgba(30, 30, 30, 0.9);
            border-radius: 16px;
            padding: 30px;
            margin: 20px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(100, 100, 255, 0.2);
        }
        .video-container {
            margin: 20px 0;
        }
        #video-feed {
            width: 100%;
            max-width: 640px;
            height: auto;
            border-radius: 12px;
            border: 3px solid #4070ff;
            box-shadow: 0 0 20px rgba(64, 112, 255, 0.3);
        }
        .btn {
            padding: 12px 24px;
            margin: 10px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .btn-primary {
            background: linear-gradient(45deg, #4070ff, #00d4ff);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(64, 112, 255, 0.4);
        }
        .status {
            margin: 20px 0;
            padding: 15px;
            background: rgba(50, 50, 50, 0.5);
            border-radius: 8px;
        }
        .emotion-display {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric {
            padding: 20px;
            background: rgba(64, 112, 255, 0.1);
            border-radius: 8px;
            border-left: 4px solid #4070ff;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #4070ff;
        }
        .metric-label {
            color: #b0b0b0;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.9rem;
        }
        #console {
            background: #000;
            color: #00ff88;
            padding: 20px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            text-align: left;
            height: 200px;
            overflow-y: auto;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Y.M.I.R Web Emotion Detection</h1>
        
        <div class="card">
            <h2>📹 Live Camera Feed</h2>
            <div class="video-container">
                <img id="video-feed" src="/video_feed" alt="Live Camera Feed" onerror="this.style.display='none'">
            </div>
            
            <div>
                <button class="btn btn-primary" onclick="startCamera()">📹 Start Camera</button>
                <button class="btn btn-primary" onclick="stopCamera()">⏹️ Stop Camera</button>
                <button class="btn btn-primary" onclick="refreshData()">🔄 Refresh Data</button>
            </div>
            
            <div class="status" id="status">
                System Status: Ready
            </div>
        </div>
        
        <div class="card">
            <h2>🎭 Emotion Analysis</h2>
            <div class="emotion-display">
                <div class="metric">
                    <div class="metric-value" id="dominant-emotion">-</div>
                    <div class="metric-label">Dominant Emotion</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="confidence">0%</div>
                    <div class="metric-label">Confidence</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="objects-count">0</div>
                    <div class="metric-label">Objects Detected</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="environment">Unknown</div>
                    <div class="metric-label">Environment</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>🖥️ System Console</h2>
            <div id="console">
                🚀 Y.M.I.R Web Emotion Detection System<br>
                💡 Click "Start Camera" to begin emotion analysis<br>
                🔧 System ready for initialization...<br>
            </div>
        </div>
    </div>

    <script>
        let isRunning = false;
        
        function log(message) {
            const console = document.getElementById('console');
            const timestamp = new Date().toLocaleTimeString();
            console.innerHTML += `[${timestamp}] ${message}<br>`;
            console.scrollTop = console.scrollHeight;
        }
        
        async function startCamera() {
            try {
                const response = await fetch('/api/start_camera', { method: 'POST' });
                const result = await response.json();
                
                if (result.success) {
                    isRunning = true;
                    document.getElementById('status').textContent = 'System Status: Camera Active';
                    document.getElementById('video-feed').style.display = 'block';
                    log('✅ Camera started successfully');
                    startDataUpdates();
                } else {
                    log('❌ Failed to start camera: ' + result.message);
                }
            } catch (error) {
                log('❌ Error starting camera: ' + error.message);
            }
        }
        
        async function stopCamera() {
            try {
                const response = await fetch('/api/stop_camera', { method: 'POST' });
                const result = await response.json();
                
                isRunning = false;
                document.getElementById('status').textContent = 'System Status: Camera Stopped';
                log('🛑 Camera stopped');
            } catch (error) {
                log('❌ Error stopping camera: ' + error.message);
            }
        }
        
        async function refreshData() {
            try {
                const response = await fetch('/api/emotions');
                const data = await response.json();
                
                if (data.status === 'active' && data.emotions) {
                    const emotions = data.emotions;
                    document.getElementById('dominant-emotion').textContent = 
                        emotions.dominant ? emotions.dominant[0].toUpperCase() : '-';
                    document.getElementById('confidence').textContent = 
                        emotions.confidence ? Math.round(emotions.confidence * 100) + '%' : '0%';
                }
                
                if (data.objects) {
                    document.getElementById('objects-count').textContent = data.objects.length;
                }
                
                if (data.environment && data.environment.type) {
                    document.getElementById('environment').textContent = 
                        data.environment.type.replace('_', ' ');
                }
                
                log('🔄 Data refreshed');
            } catch (error) {
                log('❌ Error refreshing data: ' + error.message);
            }
        }
        
        function startDataUpdates() {
            setInterval(() => {
                if (isRunning) {
                    refreshData();
                }
            }, 2000); // Update every 2 seconds
        }
        
        // Auto-refresh data every 5 seconds
        setInterval(refreshData, 5000);
        
        log('🌐 Web interface loaded');
        log('📱 Ready to start emotion detection');
    </script>
</body>
</html>'''

@app.route('/')
def index():
    """Serve the main page"""
    import os
    
    # Try multiple possible locations for the HTML file
    possible_paths = [
        'web_emotion_detection.html',
        './web_emotion_detection.html',
        os.path.join(os.path.dirname(__file__), 'web_emotion_detection.html'),
        os.path.join(os.getcwd(), 'web_emotion_detection.html')
    ]
    
    for html_path in possible_paths:
        try:
            if os.path.exists(html_path):
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                print(f"✅ HTML file loaded from: {html_path}")
                return html_content
        except Exception as e:
            print(f"⚠️ Error reading {html_path}: {e}")
            continue
    
    # If no HTML file found, return the embedded version
    print("⚠️ HTML file not found, serving embedded version")
    return get_embedded_html()

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            frame_bytes = web_system.get_current_frame_jpeg()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Send a black frame if no camera
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', black_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_camera', methods=['POST'])
def api_start_camera():
    """API endpoint to start camera"""
    success = web_system.start_camera()
    return jsonify({
        'success': success,
        'message': 'Camera started successfully' if success else 'Failed to start camera'
    })

@app.route('/api/stop_camera', methods=['POST'])
def api_stop_camera():
    """API endpoint to stop camera"""
    web_system.stop_camera()
    return jsonify({
        'success': True,
        'message': 'Camera stopped successfully'
    })

@app.route('/api/emotions')
def api_emotions():
    """API endpoint to get current emotion data"""
    return jsonify(web_system.get_emotion_data())

@app.route('/api/analytics')
def api_analytics():
    """API endpoint to get session analytics"""
    return jsonify(web_system._get_analytics_summary())

@app.route('/api/status')
def api_status():
    """API endpoint to get system status"""
    return jsonify({
        'camera_active': web_system.is_running,
        'detector_available': web_system.detector is not None,
        'has_current_frame': web_system.current_frame is not None,
        'objects_detected': len(web_system.current_objects),
        'emotions_detected': bool(web_system.current_emotions),
        'session_duration': time.time() - web_system.session_analytics['start_time'] if web_system.session_analytics['start_time'] else 0
    })

@app.route('/start_camera')
def start_camera_page():
    """Simple page to start camera"""
    success = web_system.start_camera()
    return f"""
    <h1>Y.M.I.R Camera Control</h1>
    <p>Camera Status: {'✅ Started' if success else '❌ Failed'}</p>
    <p><a href="/">← Back to Main Page</a></p>
    <p><a href="/video_feed">Direct Video Feed</a></p>
    <p><a href="/api/emotions">Emotion API</a></p>
    """

if __name__ == '__main__':
    print("🚀 Starting Y.M.I.R Web Emotion Detection System")
    print("=" * 60)
    print("🌐 Open browser and go to: http://localhost:5000")
    print("📱 The web app will handle camera permissions automatically")
    print("🎯 Enhanced YOLO and emotion detection will run in background")
    print("=" * 60)
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)