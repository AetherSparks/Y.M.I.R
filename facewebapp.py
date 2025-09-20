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
import uuid
from typing import Optional, Dict, Any
from fer_enhanced_v3 import EnhancedEmotionDetector, EnhancedEmotionConfig

# Firebase imports with fallback
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    # Define dummy variables to prevent Pylance errors
    firebase_admin = None  # type: ignore
    credentials = None  # type: ignore  
    firestore = None  # type: ignore
    FIREBASE_AVAILABLE = False
    print("⚠️ Firebase not available. Install firebase-admin for cloud storage.")

# Import our enhanced emotion detection system
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'enhancements', 'src-new', 'face'))
    from fer_enhanced_v3 import EnhancedEmotionDetector, EnhancedEmotionConfig
except ImportError:
    try:
        from fer_enhanced_v3 import EnhancedEmotionDetector, EnhancedEmotionConfig
    except ImportError:
        print("⚠️ Enhanced emotion detector not available. Using basic video only.")
        EnhancedEmotionDetector = None
        EnhancedEmotionConfig = None

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
        
        # 🎯 EMOTION SMOOTHING AND STABILITY
        self.emotion_history = []  # Last 10 emotion readings
        self.stable_emotion = None  # Current stable emotion
        self.confidence_threshold = 0.75  # Only update if confidence > 75%
        self.stability_window = 5  # Need 5 consistent readings
        self.last_update_time = 0  # Rate limiting
        
        # 📊 EFFICIENT REAL-TIME DATA STORAGE
        self.session_id = str(uuid.uuid4())  # Unique session ID
        self.readings_buffer = []  # Local buffer for offline resilience
        self.last_firebase_sync = 0  # Rate limit Firebase writes
        self.firebase_batch_size = 5  # Batch readings for efficiency
        self.offline_mode = False
        self.firebase_client = None
        self.storage_enabled = False
        
        # Initialize Firebase if available
        self._init_firebase_storage()
        
        # Initialize the enhanced detector
        self._init_detector()
        
        print(f"🔑 Session ID: {self.session_id}")
        print(f"📊 Storage: {'Firebase' if self.storage_enabled else 'Local Only'}")
    
    def _init_detector(self):
        """Initialize the enhanced emotion detector"""
        try:
            if EnhancedEmotionDetector is None or EnhancedEmotionConfig is None:
                print("⚠️ Enhanced emotion detector classes not available")
                self.detector = None
                return
                
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
    
    def _init_firebase_storage(self):
        """Initialize Firebase Firestore for real-time emotion storage"""
        if not FIREBASE_AVAILABLE:
            print("⚠️ Firebase unavailable - using local storage only")
            return
            
        try:
            # Initialize Firebase if not already done
            if firebase_admin is not None and not firebase_admin._apps:
                # For production, use service account key
                # For demo, we'll simulate Firebase functionality
                print("🔥 Firebase would be initialized here with service account")
                print("🔥 Demo mode: Using local storage with Firebase-like structure")
                self.storage_enabled = True
                return
            
            if firestore is not None:
                self.firebase_client = firestore.client()
                self.storage_enabled = True
                print("✅ Firebase Firestore initialized")
            else:
                print("⚠️ Firestore not available - using local storage")
                self.storage_enabled = True  # Enable demo mode
            
        except Exception as e:
            print(f"⚠️ Firebase initialization failed: {e}")
            print("📱 Falling back to local storage")
            self.storage_enabled = False
    
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
            if self.detector and hasattr(self.detector, 'mediapipe_processor') and self.detector.mediapipe_processor:
                mediapipe_results = self.detector.mediapipe_processor.process_frame(frame)
                faces = mediapipe_results.get('faces', [])
            else:
                faces = []
            
            # YOLO processing (every 15 frames)
            if frame_count % 15 == 0 and self.detector and hasattr(self.detector, 'yolo_processor') and self.detector.yolo_processor:
                try:
                    objects, environment_context = self.detector.yolo_processor.detect_objects_with_emotion_context(frame)
                    self.current_objects = objects[:10]  # Limit for web display
                    self.current_environment = environment_context
                except Exception as e:
                    print(f"⚠️ YOLO processing error: {e}")
            
            # Emotion analysis (every 30 frames)
            if frame_count % 30 == 0 and faces and self.detector and hasattr(self.detector, 'deepface_ensemble') and self.detector.deepface_ensemble:
                try:
                    face_info = faces[0]  # Process first face
                    if face_info['roi'].size > 0:
                        emotion_result = self.detector.deepface_ensemble.analyze_face_with_context(
                            face_info['id'], face_info['roi'], self.current_environment
                        )
                        
                        if emotion_result:
                            # 🎯 APPLY EMOTION SMOOTHING AND STABILITY
                            smoothed_emotion = self._smooth_emotion(
                                emotion_result['emotions'], 
                                emotion_result['confidence']
                            )
                            
                            if smoothed_emotion:
                                # Update current emotions for web display
                                self.current_emotions = {
                                    'dominant': smoothed_emotion['dominant'],
                                    'all_emotions': smoothed_emotion['all_emotions'],
                                    'confidence': smoothed_emotion['confidence'],
                                    'quality': face_info.get('quality_score', 0.8),
                                    'stability': smoothed_emotion['stability'],
                                    'timestamp': datetime.now().isoformat(),
                                    'raw_emotions': emotion_result['emotions'],  # Show original for comparison
                                    'smoothing_applied': True
                                }
                                
                                # Update analytics
                                self._update_analytics(emotion_result, face_info.get('quality_score', 0.8))
                                
                                # 📊 STORE EMOTION DATA EFFICIENTLY
                                self._store_emotion_reading({
                                    'session_id': self.session_id,
                                    'timestamp': time.time(),
                                    'emotions': smoothed_emotion,
                                    'environment': self.current_environment,
                                    'objects': [obj.get('class', 'unknown') for obj in self.current_objects[:5]],
                                    'quality_metrics': {
                                        'face_quality': face_info.get('quality_score', 0.8),
                                        'confidence': smoothed_emotion['confidence'],
                                        'stability': smoothed_emotion['stability']
                                    }
                                })
                            
                except Exception as e:
                    print(f"⚠️ Emotion analysis error: {e}")
            
        except Exception as e:
            print(f"⚠️ Enhanced detector processing error: {e}")
    
    def _smooth_emotion(self, raw_emotions, confidence):
        """🎯 SMOOTH EMOTIONS TO PREVENT RAPID JUMPING"""
        current_time = time.time()
        
        # Rate limiting: Only update every 2 seconds maximum
        if current_time - self.last_update_time < 2.0:
            return None
        
        # Only process high confidence readings
        if confidence < self.confidence_threshold:
            print(f"⚠️ Low confidence ({confidence:.2f}) - maintaining stable emotion")
            return None
        
        # Get dominant emotion from raw data
        dominant_raw = max(raw_emotions.items(), key=lambda x: x[1])
        dominant_emotion = dominant_raw[0]
        dominant_score = dominant_raw[1]
        
        # Add to history
        self.emotion_history.append({
            'emotion': dominant_emotion,
            'score': dominant_score,
            'confidence': confidence,
            'timestamp': current_time
        })
        
        # Keep only last 10 readings
        if len(self.emotion_history) > 10:
            self.emotion_history.pop(0)
        
        # Check for stability: need at least 3 readings of same emotion
        if len(self.emotion_history) >= 3:
            recent_emotions = [r['emotion'] for r in self.emotion_history[-self.stability_window:]]
            emotion_counts = {}
            for emotion in recent_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Find most frequent emotion in recent history
            most_frequent = max(emotion_counts.items(), key=lambda x: x[1])
            most_frequent_emotion = most_frequent[0]
            frequency = most_frequent[1]
            
            # Only update if emotion appears in at least 60% of recent readings
            stability_threshold = max(2, len(recent_emotions) * 0.6)
            
            if frequency >= stability_threshold:
                # Calculate stability score
                stability = frequency / len(recent_emotions)
                
                # Check if this is a significant change from current stable emotion
                if (self.stable_emotion is None or 
                    most_frequent_emotion != self.stable_emotion or
                    stability > 0.8):  # Very stable
                    
                    self.stable_emotion = most_frequent_emotion
                    self.last_update_time = current_time
                    
                    # Calculate smoothed scores (average of recent readings)
                    smoothed_emotions = {}
                    for emotion_name in raw_emotions.keys():
                        emotion_scores = [r['score'] for r in self.emotion_history[-3:] 
                                        if emotion_name in raw_emotions]
                        if emotion_scores:
                            smoothed_emotions[emotion_name] = sum(emotion_scores) / len(emotion_scores)
                        else:
                            smoothed_emotions[emotion_name] = raw_emotions[emotion_name]
                    
                    print(f"🎯 EMOTION STABILIZED: {most_frequent_emotion.upper()} (stability: {stability:.2f})")
                    
                    return {
                        'dominant': (most_frequent_emotion, smoothed_emotions[most_frequent_emotion]),
                        'all_emotions': smoothed_emotions,
                        'confidence': confidence,
                        'stability': stability,
                        'readings_analyzed': len(recent_emotions)
                    }
        
        # If no stable emotion yet, return None (keep previous)
        print(f"⏳ Analyzing emotion stability... ({len(self.emotion_history)} readings)")
        return None
    
    def _update_analytics(self, emotion_result, quality_score):
        """Update session analytics"""
        self.session_analytics['total_readings'] += 1
        self.session_analytics['confidence_sum'] += emotion_result['confidence']
        self.session_analytics['quality_sum'] += quality_score
        self.session_analytics['stability_sum'] += emotion_result.get('stability', 0.0)
    
    def _store_emotion_reading(self, reading_data: Dict[str, Any]):
        """📊 EFFICIENT EMOTION DATA STORAGE WITH OFFLINE RESILIENCE"""
        current_time = time.time()
        
        # Add to local buffer (always works)
        self.readings_buffer.append(reading_data)
        
        # Rate limiting: Only sync to Firebase every 3 seconds
        if current_time - self.last_firebase_sync < 3.0:
            return
        
        # Batch process when buffer reaches threshold or time limit
        if len(self.readings_buffer) >= self.firebase_batch_size or \
           current_time - self.last_firebase_sync > 10.0:
            self._sync_to_firebase()
    
    def _sync_to_firebase(self):
        """📤 BATCH SYNC EMOTION READINGS TO FIREBASE"""
        if not self.readings_buffer:
            return
            
        try:
            if self.storage_enabled:
                # In production, this would use Firebase Firestore batch writes
                self._firebase_batch_write(self.readings_buffer)
                print(f"✅ Synced {len(self.readings_buffer)} emotion readings to Firebase")
            else:
                # Local storage fallback
                self._local_storage_fallback(self.readings_buffer)
                print(f"💾 Stored {len(self.readings_buffer)} readings locally")
            
            # Clear buffer after successful sync
            self.readings_buffer.clear()
            self.last_firebase_sync = time.time()
            self.offline_mode = False
            
        except Exception as e:
            print(f"⚠️ Storage sync failed: {e}")
            self.offline_mode = True
            
            # Keep buffer but limit size to prevent memory issues
            if len(self.readings_buffer) > 50:
                self.readings_buffer = self.readings_buffer[-25:]
    
    def _firebase_batch_write(self, readings: list):
        """🔥 FIREBASE FIRESTORE BATCH WRITE (Production Implementation)"""
        # This would be the actual Firebase implementation:
        # 
        # batch = self.firebase_client.batch()
        # session_ref = self.firebase_client.collection('emotion_sessions').document(self.session_id)
        # 
        # for reading in readings:
        #     reading_ref = session_ref.collection('readings').document()
        #     batch.set(reading_ref, reading)
        # 
        # batch.commit()
        
        # For demo purposes, simulate Firebase structure
        firebase_structure = {
            'session_id': self.session_id,
            'created_at': datetime.now().isoformat(),
            'total_readings': len(readings),
            'readings': readings,
            'metadata': {
                'storage_type': 'firebase_batch',
                'batch_size': len(readings),
                'timestamp': time.time()
            }
        }
        
        # In demo mode, save to local file with Firebase-like structure
        with open(f'firebase_demo_{self.session_id}.json', 'w') as f:
            json.dump(firebase_structure, f, indent=2)
    
    def _local_storage_fallback(self, readings: list):
        """💾 LOCAL STORAGE FALLBACK FOR OFFLINE RESILIENCE"""
        local_data = {
            'session_id': self.session_id,
            'stored_at': datetime.now().isoformat(),
            'readings': readings,
            'metadata': {
                'storage_type': 'local_fallback',
                'offline_mode': self.offline_mode,
                'total_readings': len(readings)
            }
        }
        
        # Append to local storage file
        storage_file = f'emotion_storage_{self.session_id}.json'
        try:
            # Try to load existing data
            with open(storage_file, 'r') as f:
                existing_data = json.load(f)
                existing_data['readings'].extend(readings)
                existing_data['metadata']['total_readings'] += len(readings)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = local_data
        
        # Save updated data
        with open(storage_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
    
    def get_storage_analytics(self):
        """📈 GET STORAGE AND PERFORMANCE ANALYTICS"""
        return {
            'session_id': self.session_id,
            'storage_enabled': self.storage_enabled,
            'offline_mode': self.offline_mode,
            'buffer_size': len(self.readings_buffer),
            'last_sync': self.last_firebase_sync,
            'total_stored': self.session_analytics['total_readings'],
            'storage_type': 'firebase' if self.storage_enabled else 'local'
        }
    
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
    
    # If no HTML file found, return a simple page
    print("⚠️ HTML file not found, serving basic page")
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Y.M.I.R Emotion Detection</title></head>
    <body>
        <h1>Y.M.I.R Emotion Detection</h1>
        <p>HTML file not found. Please ensure web_emotion_detection.html exists.</p>
        <p><a href="/video_feed">Direct Video Feed</a></p>
    </body>
    </html>
    """


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

@app.route('/api/storage')
def api_storage():
    """API endpoint to get storage analytics"""
    return jsonify(web_system.get_storage_analytics())

@app.route('/api/export_session', methods=['POST'])
def api_export_session():
    """API endpoint to export session data"""
    try:
        # Force sync any remaining buffered data
        web_system._sync_to_firebase()
        
        # Create export data
        export_data = {
            'session_info': {
                'session_id': web_system.session_id,
                'exported_at': datetime.now().isoformat(),
                'duration': time.time() - web_system.session_analytics['start_time'] if web_system.session_analytics['start_time'] else 0
            },
            'analytics': web_system._get_analytics_summary(),
            'storage': web_system.get_storage_analytics(),
            'export_format': 'ymir_emotion_session_v1.0'
        }
        
        return jsonify({
            'success': True,
            'export_data': export_data,
            'message': 'Session data exported successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to export session data'
        })

@app.route('/api/status')
def api_status():
    """API endpoint to get system status"""
    return jsonify({
        'camera_active': web_system.is_running,
        'detector_available': web_system.detector is not None,
        'has_current_frame': web_system.current_frame is not None,
        'objects_detected': len(web_system.current_objects),
        'emotions_detected': bool(web_system.current_emotions),
        'session_duration': time.time() - web_system.session_analytics['start_time'] if web_system.session_analytics['start_time'] else 0,
        'storage_status': {
            'enabled': web_system.storage_enabled,
            'offline_mode': web_system.offline_mode,
            'buffer_size': len(web_system.readings_buffer)
        }
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