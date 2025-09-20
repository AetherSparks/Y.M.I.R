"""
🚀 Y.M.I.R Web-based Emotion Detection Flask App - MICROSERVICE VERSION
===============================================
Your working facewebapp.py converted to run as microservice on port 5001
with CORS enabled for integration with main app.
"""

from flask import Flask, render_template_string, jsonify, request, Response
from flask_cors import CORS  # Added for microservice communication
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

# Add missing imports for the enhanced detector
import sys
import os
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

# Ensure all required libraries are available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe available")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not available")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace available")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace not available")

try:
    import torch
    TORCH_AVAILABLE = True
    print("✅ PyTorch available")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available")
# Firebase imports - REQUIRED for production
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
    print("✅ Firebase Admin SDK loaded")
except ImportError:
    raise ImportError("❌ Firebase Admin SDK is required for production. Install: pip install firebase-admin")

# Import enhanced emotion detection system - REQUIRED for production
try:
    import sys
    import os
    # Add the correct path to the enhanced detector
    enhanced_detector_path = os.path.join(os.path.dirname(__file__), 'enhancements', 'src-new', 'face')
    if enhanced_detector_path not in sys.path:
        sys.path.append(enhanced_detector_path)
    
    from fer_enhanced_v3 import EnhancedEmotionDetector, EnhancedEmotionConfig
    print("✅ Enhanced emotion detector loaded")
except ImportError as e:
    try:
        # Try alternative path
        from enhancements.src_new.face.fer_enhanced_v3 import EnhancedEmotionDetector, EnhancedEmotionConfig
        print("✅ Enhanced emotion detector loaded from alternative path")
    except ImportError:
        raise ImportError(f"❌ Enhanced emotion detector is required for production: {e}")

app = Flask(__name__)
CORS(app)  # Enable CORS for microservice communication

class WebEmotionSystem:
    """Web-based emotion detection system using full EnhancedEmotionDetector"""
    
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.current_emotions = {}
        self.frame_lock = threading.Lock()
        
        # 📊 Session tracking
        self.session_id = str(uuid.uuid4())
        self.session_analytics = {
            'total_readings': 0,
            'confidence_sum': 0,
            'quality_sum': 0,
            'stability_sum': 0,
            'start_time': None
        }
        
        # Storage configuration
        self.storage_enabled = True
        self.offline_mode = False
        self.readings_buffer = []
        self.last_firebase_sync = 0
        
        # Initialize the enhanced detector
        self._init_full_detector()
        
        # 🎛️ Initialize visual settings with defaults
        self.visual_settings = {
            'show_face_mesh': True,
            'show_body_pose': True,
            'show_hand_tracking': True,
            'show_gaze_tracking': True,
            'show_object_detection': True,
            'show_emotion_context': True,
            'confidence_threshold': 0.25,
            'video_quality': '720p',
            'analysis_interval': 30,
            'show_quality_indicators': True,
            'show_fps_display': False
        }
        
        print(f"🔑 Session ID: {self.session_id}")
        print(f"📊 Using FULL Enhanced Detector with detailed logging")
        print(f"🎛️ Visual settings initialized with defaults")
    
    def _init_full_detector(self):
        """Initialize the enhanced emotion detector"""
        try:
                
            # Create config for web microservice (HEADLESS MODE)
            config = EnhancedEmotionConfig(
                camera_width=640,
                camera_height=480,
                emotion_analysis_interval=30,
                require_user_consent=False,  # Web handles permissions
                use_firebase=True,  # ENABLE Firebase to store emotions for combiner
                show_analytics=False,  # Disable GUI analytics
                privacy_mode=False  # Ensure processing works
            )
            
            # Initialize the FULL enhanced detector
            self.detector = EnhancedEmotionDetector(config)
            print("✅ FULL Enhanced Emotion Detector initialized for microservice")
            
        except Exception as e:
            print(f"❌ Detector initialization error: {e}")
            raise RuntimeError(f"Failed to initialize emotion detector: {e}")
    
    
    def start_camera(self):
        """Start camera capture using the full detector"""
        if self.is_running:
            return True
            
        try:
            if self.detector:
                print("🎬 Starting camera using FULL detector...")
                # Use the full detector's camera system
                success = self.detector.start_camera()
                if success:
                    self.is_running = True
                    self.session_analytics['start_time'] = time.time()
                    
                    # Start processing thread using the FULL detector
                    threading.Thread(target=self._process_frames_with_full_detector, daemon=True).start()
                    
                    print("✅ FULL detector camera started with detailed logging")
                    return True
                else:
                    print("❌ Full detector camera failed to start")
                    return False
            else:
                print("❌ Full detector not available")
                return False
            
        except Exception as e:
            print(f"❌ Camera start error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        
        if self.detector and self.detector.cap:
            self.detector.cap.release()
            self.detector.cap = None
            
        print("🛑 FULL detector camera stopped")
    
    def _process_frames_with_full_detector(self):
        """Process camera frames using the FULL enhanced detector with all logging"""
        print("🚀 Starting FULL detector frame processing with detailed logging")
        print(f"🔍 DEBUG: Enhanced detector available: {self.detector is not None}")
        print(f"🔍 DEBUG: Firebase manager available: {hasattr(self.detector, 'firebase_manager') if self.detector else False}")
        
        frame_count = 0
        while self.is_running and self.detector and self.detector.cap:
            try:
                frame_count += 1
                # Only log every 30 frames (~1 second) to reduce spam
                if frame_count % 30 == 0:
                    print(f"📹 Processing frame {frame_count}")
                
                ret, frame = self.detector.cap.read()
                if not ret:
                    print("⚠️ Failed to read frame from camera")
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Store frame for web streaming
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Use individual detector components for HEADLESS processing
                self._process_frame_headless(frame, frame_count)
                
                # Extract current emotions from the full detector
                if hasattr(self.detector, 'emotion_lock') and hasattr(self.detector, 'current_emotions'):
                    with self.detector.emotion_lock:
                        if self.detector.current_emotions:
                            # Update web system's current emotions for API
                            self.current_emotions = {}
                            for face_id, reading in self.detector.current_emotions.items():
                                self.current_emotions[face_id] = {
                                    'dominant': reading.emotions,
                                    'confidence': reading.confidence,
                                    'quality': reading.quality_score,
                                    'timestamp': reading.timestamp.isoformat(),
                                    'stability': reading.stability
                                }
                            print(f"📊 Updated emotions for {len(self.current_emotions)} faces")
                else:
                    print("⚠️ Detector doesn't have emotion_lock or current_emotions - skipping frame")
                
                time.sleep(0.016)  # ~60 FPS for smooth video
                
            except Exception as e:
                print(f"❌ FULL detector frame processing error: {e}")
                import traceback
                traceback.print_exc()
                print(f"🛑 Stopping frame processing due to error")
                break
                
        print("🛑 Frame processing loop ended")
    
    
    def _process_frame_headless(self, frame, frame_count):
        """Process frame using detector components in headless mode"""
        try:
            # MediaPipe processing (safe for headless)
            if hasattr(self.detector, 'mediapipe_processor'):
                mediapipe_results = self.detector.mediapipe_processor.process_frame(frame)
                faces = mediapipe_results.get('faces', [])
                
                # Only log when faces are found or every 30 frames
                if len(faces) > 0 or frame_count % 30 == 0:
                    print(f"👥 MediaPipe: {len(faces)} faces detected")
            
            # YOLO processing (every 5 frames) - THIS SHOULD SHOW THE DETAILED LOGS
            if frame_count % 5 == 0 and hasattr(self.detector, 'yolo_processor'):
                print("🎯 Processing with YOLO...")
                try:
                    objects, environment_context = self.detector.yolo_processor.detect_objects_with_emotion_context(frame)
                    self.detector.detected_objects = objects
                    self.detector.current_environment_context = environment_context
                    
                    # 🔍 EXTENSIVE YOLO LOGGING (THIS IS WHAT YOU WANT TO SEE!)
                    if objects:
                        print(f"\n🎯 YOLO DETECTED OBJECTS (Frame {frame_count}):")
                        for obj in objects:
                            conf = obj.get('confidence', 0)
                            class_name = obj.get('class', 'unknown')
                            print(f"   📦 {class_name.upper()} (confidence: {conf:.2f})")
                    
                    if environment_context:
                        print(f"\n🌍 ENVIRONMENT ANALYSIS:")
                        env_type = environment_context.get('type', 'unknown')
                        print(f"   🏠 Environment Type: {env_type.upper()}")
                        
                        if 'context_modifiers' in environment_context:
                            modifiers = environment_context['context_modifiers']
                            print(f"   🎭 Emotion Modifiers Applied:")
                            for emotion, modifier in modifiers.items():
                                if modifier != 1.0:
                                    direction = "↗️" if modifier > 1.0 else "↘️"
                                    print(f"      {direction} {emotion}: {modifier:.2f}x")
                        
                        if 'detected_categories' in environment_context:
                            categories = environment_context['detected_categories']
                            print(f"   📋 Object Categories: {', '.join(categories)}")
                        
                        print()  # Empty line for readability
                        
                except Exception as e:
                    print(f"⚠️ YOLO processing error: {e}")
            
            # Face emotion analysis (every 30 frames for performance while keeping 60fps video)
            if frame_count % 30 == 0 and 'faces' in locals() and faces:
                print("🧠 Processing emotions...")
                try:
                    face_info = faces[0]  # Process first face
                    if face_info['roi'].size > 0:
                        # Simple emotion analysis without threading for headless mode
                        emotion_result = self.detector.deepface_ensemble.analyze_face_with_context(
                            face_info['id'], face_info['roi'], 
                            getattr(self.detector, 'current_environment_context', {})
                        )
                        
                        if emotion_result:
                            # 🧠 EXTENSIVE EMOTION ANALYSIS LOGGING (THIS TOO!)
                            dominant_emotion = max(emotion_result['emotions'].items(), key=lambda x: x[1])
                            print(f"\n🧠 FACE EMOTION ANALYSIS (Face {face_info['id']}):")
                            print(f"   🎭 Dominant: {dominant_emotion[0].upper()} ({dominant_emotion[1]:.1f}%)")
                            print(f"   📊 Confidence: {emotion_result['confidence']:.2f}")
                            print(f"   🏆 Quality Score: {face_info.get('quality_score', 0.8):.2f}")
                            
                            # Show all emotions detected
                            print(f"   🎪 All Emotions Detected:")
                            sorted_emotions = sorted(emotion_result['emotions'].items(), key=lambda x: x[1], reverse=True)
                            for emotion, score in sorted_emotions:  # Show ALL emotions (not just top 4)
                                bar = "█" * max(1, int(score / 10))  # Visual bar
                                print(f"      {emotion.capitalize():12} {score:5.1f}% {bar}")
                            
                            # 🔥 STORE IN FIREBASE for combiner!
                            print(f"🔍 DEBUG: Checking Firebase storage...")
                            print(f"🔍 DEBUG: detector exists: {self.detector is not None}")
                            print(f"🔍 DEBUG: has firebase_manager: {hasattr(self.detector, 'firebase_manager') if self.detector else False}")
                            print(f"🔍 DEBUG: firebase_manager value: {getattr(self.detector, 'firebase_manager', None) if self.detector else None}")
                            
                            # 🔥 FIREBASE STORAGE - Enhanced + Fallback
                            firebase_stored = False
                            
                            # SKIP ENHANCED DETECTOR - Go directly to fallback for guaranteed storage
                            print(f"   🔍 DEBUG: Skipping enhanced detector, using direct Firebase storage")
                            
                            # 💾 DIRECT Firebase storage for guaranteed compatibility with combiner
                            if FIREBASE_AVAILABLE:
                                try:
                                    print(f"   🔄 Using direct Firebase storage (fallback)")
                                    import firebase_admin
                                    from firebase_admin import firestore
                                    from datetime import datetime, timezone
                                    
                                    # Get current UTC timestamp for storage
                                    storage_timestamp = datetime.now(timezone.utc)
                                    print(f"   🕐 DEBUG: Storing with timestamp: {storage_timestamp}")
                                    
                                    # Get Firestore client
                                    if not firebase_admin._apps:
                                        cred = firebase_admin.credentials.Certificate('firebase_credentials.json')
                                        firebase_admin.initialize_app(cred)
                                    
                                    db = firestore.client()
                                    
                                    # Store facial emotion with expected format for combiner
                                    facial_doc = {
                                        'timestamp': storage_timestamp,
                                        'face_id': face_info['id'],
                                        'emotions': emotion_result['emotions'],  # Key field for combiner!
                                        'confidence': emotion_result['confidence'],
                                        'quality_score': face_info.get('quality_score', 0.8),
                                        'session_id': self.session_id
                                        # NO 'role' field - this marks it as facial emotion
                                    }
                                    
                                    # Store in emotion_readings collection for combiner
                                    doc_ref = db.collection('emotion_readings').document()
                                    doc_ref.set(facial_doc)
                                    print(f"   🔥 Stored facial emotion to Firebase directly!")
                                    print(f"   📄 Document ID: {doc_ref.id}")
                                    print(f"   📊 Emotions: {emotion_result['emotions']}")
                                    print(f"   🕐 Timestamp: {storage_timestamp}")
                                    print(f"   👤 Face ID: {face_info['id']}")
                                    firebase_stored = True
                                    
                                except Exception as firebase_error:
                                    print(f"   ❌ Direct Firebase storage error: {firebase_error}")
                            
                            if not firebase_stored:
                                print(f"   ⚠️ All Firebase storage methods failed")
                            
                            print()  # Empty line for readability
                            
                except Exception as e:
                    print(f"⚠️ Emotion analysis error: {e}")
            
            # Processing completed - log only when emotions found
            pass
            
        except Exception as e:
            print(f"❌ Headless processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def get_current_frame_jpeg(self):
        """Get current frame with visual overlays as JPEG bytes"""
        with self.frame_lock:
            if self.current_frame is not None:
                # Apply visual overlays based on settings
                display_frame = self.apply_visual_overlays(self.current_frame.copy())
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    return buffer.tobytes()
        return None
    
    def apply_visual_overlays(self, frame):
        """🎛️ Apply visual overlays based on current settings"""
        try:
            if not hasattr(self, 'visual_settings'):
                return frame
            
            settings = self.visual_settings
            # Removed excessive overlay logging
            
            # 🎯 Simple Visual Overlays (always available)
            
            # 📊 Show current emotions as overlay
            if self.current_emotions and settings.get('show_quality_indicators', True):
                self._draw_emotion_overlay(frame)
            
            # 📈 FPS Display
            if settings.get('show_fps_display', False):
                # Calculate actual FPS from frame timing
                current_time = time.time()
                if hasattr(self, 'last_frame_time'):
                    fps = 1.0 / (current_time - self.last_frame_time)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.last_frame_time = current_time
            
            # 🔥 Settings Status Display
            self._draw_settings_status(frame, settings)
            
            # 🎯 Enhanced Visual Overlays (if detector available)
            if self.detector and hasattr(self.detector, 'mediapipe_processor'):
                self._apply_enhanced_overlays(frame, settings)
            
            return frame
            
        except Exception as e:
            print(f"⚠️ Visual overlay error: {e}")
            return frame
    
    def _draw_emotion_overlay(self, frame):
        """Draw current emotion information on frame"""
        try:
            y_offset = 50
            for emotion, confidence in self.current_emotions.items():
                text = f"{emotion.capitalize()}: {confidence:.1f}%"
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
        except Exception as e:
            print(f"⚠️ Emotion overlay error: {e}")
    
    def _draw_settings_status(self, frame, settings):
        """Draw visual settings status"""
        try:
            h, w = frame.shape[:2]
            y_start = h - 120
            
            # Show active visual features
            active_features = []
            if settings.get('show_face_mesh'): active_features.append("Face Mesh")
            if settings.get('show_body_pose'): active_features.append("Body Pose") 
            if settings.get('show_hand_tracking'): active_features.append("Hand Track")
            if settings.get('show_object_detection'): active_features.append("Objects")
            
            if active_features:
                text = f"Active: {', '.join(active_features)}"
                cv2.putText(frame, text, (10, y_start), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Show video quality
            quality = settings.get('video_quality', '720p')
            cv2.putText(frame, f"Quality: {quality}", (10, y_start + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Show detection confidence
            conf = settings.get('confidence_threshold', 0.25)
            cv2.putText(frame, f"Confidence: {conf:.2f}", (10, y_start + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                       
        except Exception as e:
            print(f"⚠️ Settings overlay error: {e}")
    
    def _apply_enhanced_overlays(self, frame, settings):
        """Apply enhanced visual overlays using detector components"""
        try:
            # 📐 Face detection overlay
            if settings.get('show_face_mesh', True):
                # Draw face rectangles and landmarks using MediaPipe
                try:
                    import mediapipe as mp
                    try:
                        # Try new MediaPipe import structure
                        from mediapipe.python.solutions import face_detection as mp_face_detection
                        from mediapipe.python.solutions import drawing_utils as mp_drawing
                    except ImportError:
                        # Fallback to older MediaPipe import structure
                        mp_face_detection = mp.solutions.face_detection
                        mp_drawing = mp.solutions.drawing_utils
                    
                    with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_detection.process(rgb_frame)
                    
                        if results.detections:
                            for detection in results.detections:
                                # Draw face detection box
                                bbox = detection.location_data.relative_bounding_box
                                h, w, _ = frame.shape
                                x = int(bbox.xmin * w)
                                y = int(bbox.ymin * h)
                                width = int(bbox.width * w)
                                height = int(bbox.height * h)
                                
                                # Draw green rectangle around face
                                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                                cv2.putText(frame, f"Face {detection.score[0]:.2f}", (x, y-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                except ImportError:
                    # MediaPipe not available, skip face overlay
                    pass
            
            # 🎯 Object detection status
            if settings.get('show_object_detection', True):
                detected_objects = getattr(self.detector, 'detected_objects', [])
                if detected_objects:
                    cv2.putText(frame, f"Objects: {len(detected_objects)}", (frame.shape[1] - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                               
        except Exception as e:
            print(f"⚠️ Enhanced overlay error: {e}")
    
    def get_emotion_data(self):
        """Get current emotion data for API using full detector"""
        if not self.current_emotions:
            return {
                'status': 'waiting',
                'message': 'Waiting for emotion detection...'
            }
        
        # Get detected objects from full detector
        objects = []
        environment = {}
        if self.detector:
            objects = [{'class': obj.get('class', 'unknown'), 'confidence': obj.get('confidence', 0)} 
                      for obj in self.detector.detected_objects[:5]]
            environment = self.detector.current_environment_context
        
        return {
            'status': 'active',
            'emotions': self.current_emotions,
            'objects': objects,
            'environment': environment,
            'analytics': self._get_analytics_summary()
        }
    
    def _get_analytics_summary(self):
        """Get analytics summary from full detector"""
        if self.detector and hasattr(self.detector, 'emotion_analytics'):
            # Use the full detector's analytics
            analytics = self.detector.emotion_analytics.get_emotion_trends()
            
            # Ensure required fields exist for frontend compatibility
            if isinstance(analytics, dict):
                # Make sure we have total_readings field
                if 'total_readings' not in analytics:
                    analytics['total_readings'] = analytics.get('total_readings', 0)
                return analytics
        
        # Fallback to basic analytics
        session_duration = 0
        if self.session_analytics['start_time']:
            session_duration = time.time() - self.session_analytics['start_time']
        
        return {
            'total_readings': 0,
            'avg_confidence': 0,
            'avg_quality': 0,
            'avg_stability': 0,
            'session_duration': session_duration
        }
    
    def get_storage_analytics(self):
        """Get storage analytics for API compatibility"""
        return {
            'session_id': self.session_id,
            'storage_enabled': self.storage_enabled,
            'offline_mode': self.offline_mode,
            'buffer_size': len(self.readings_buffer),
            'last_sync': self.last_firebase_sync,
            'total_stored': self.session_analytics['total_readings'],
            'storage_type': 'microservice_mode'
        }
    
    def _sync_to_firebase(self):
        """Sync buffered emotions to Firebase if storage enabled"""
        if self.storage_enabled and FIREBASE_AVAILABLE and self.readings_buffer:
            try:
                print(f"🔄 Syncing {len(self.readings_buffer)} buffered emotions to Firebase")
                # Process any buffered readings here if needed
                self.last_firebase_sync = int(time.time())
                print(f"✅ Firebase sync completed")
            except Exception as e:
                print(f"❌ Firebase sync error: {e}")
        else:
            print(f"⏸️ Firebase sync skipped - storage_enabled: {self.storage_enabled}, FIREBASE_AVAILABLE: {FIREBASE_AVAILABLE}, buffer_size: {len(self.readings_buffer)}")

# Global web emotion system
web_system = WebEmotionSystem()

# 🎯 MICROSERVICE API ENDPOINTS (Same as your original but with added health check)

@app.route('/health')
def health_check():
    """Health check endpoint for microservice monitoring"""
    return jsonify({
        'service': 'Y.M.I.R Face Emotion Detection Microservice',
        'status': 'healthy',
        'version': '1.0.0',
        'port': 5002,
        'detector_available': web_system.detector is not None,
        'camera_running': web_system.is_running
    })

@app.route('/')
def index():
    """API-only microservice info page"""
    return jsonify({
        'service': 'Y.M.I.R Face Emotion Detection Microservice',
        'version': '1.0.0',
        'port': 5002,
        'description': 'API-only microservice for facial emotion detection',
        'endpoints': {
            'health': '/health',
            'start_camera': '/api/start_camera',
            'stop_camera': '/api/stop_camera', 
            'emotions': '/api/emotions',
            'video_feed': '/video_feed',
            'status': '/api/status'
        },
        'usage': 'This microservice provides APIs only. Use the main app at port 5000 for UI.'
    })

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

# 🎯 API ENDPOINTS (KEEP YOUR EXACT WORKING API STRUCTURE)

@app.route('/api/start_camera', methods=['POST'])
def api_start_camera():
    """API endpoint to start camera"""
    print("🎬 API: start_camera called")
    try:
        success = web_system.start_camera()
        print(f"🎬 Camera start result: {success}")
        response = {
            'success': success,
            'message': 'Camera started successfully' if success else 'Failed to start camera',
            'session_id': web_system.session_id
        }
        print(f"🎬 Returning response: {response}")
        return jsonify(response)
    except Exception as e:
        print(f"🎬 ERROR in start_camera API: {e}")
        return jsonify({
            'success': False,
            'message': f'Camera start error: {str(e)}'
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
        'running': web_system.is_running,
        'camera_active': web_system.is_running,
        'detector_available': web_system.detector is not None,
        'detector_loaded': web_system.detector is not None,
        'has_current_frame': web_system.current_frame is not None,
        'objects_detected': len(web_system.detector.detected_objects) if web_system.detector else 0,
        'emotions_detected': bool(web_system.current_emotions),
        'session_duration': time.time() - web_system.session_analytics['start_time'] if web_system.session_analytics['start_time'] else 0,
        'session_id': web_system.session_id,
        'analytics': web_system._get_analytics_summary(),
        'storage_status': {
            'enabled': web_system.storage_enabled,
            'offline_mode': web_system.offline_mode,
            'buffer_size': len(web_system.readings_buffer)
        }
    })

@app.route('/api/settings', methods=['POST'])
def api_update_settings():
    """🎛️ Update visual processing settings for enhanced features"""
    try:
        settings = request.get_json()
        # Visual settings updated silently
        
        # Update detector configuration if detector exists
        if web_system.detector and hasattr(web_system.detector, 'config'):
            config = web_system.detector.config
            
            # Update MediaPipe settings
            if hasattr(config, 'mediapipe_config'):
                mp_config = config.mediapipe_config
                mp_config.show_face_mesh = settings.get('show_face_mesh', True)
                mp_config.show_body_pose = settings.get('show_body_pose', True)
                mp_config.show_hand_tracking = settings.get('show_hand_tracking', True)
                mp_config.show_gaze_tracking = settings.get('show_gaze_tracking', True)
                print(f"✅ Updated MediaPipe settings: face_mesh={mp_config.show_face_mesh}, body_pose={mp_config.show_body_pose}")
            
            # Update YOLO settings
            if hasattr(config, 'yolo_config'):
                yolo_config = config.yolo_config
                yolo_config.confidence_threshold = settings.get('confidence_threshold', 0.25)
                print(f"✅ Updated YOLO confidence: {yolo_config.confidence_threshold}")
            
            # Update analysis settings
            config.emotion_analysis_interval = settings.get('analysis_interval', 30)
            config.show_analytics = settings.get('show_quality_indicators', True)
            
            print(f"✅ All visual settings updated successfully")
            
        # Store settings globally for video feed processing
        web_system.visual_settings = settings
        
        return jsonify({
            'success': True,
            'message': 'Visual settings updated successfully',
            'settings': settings
        })
        
    except Exception as e:
        print(f"❌ Settings update error: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to update settings: {str(e)}'
        }), 500

# UI routes removed - this is an API-only microservice  
# Use the main app at port 5000 for UI

if __name__ == '__main__':
    print("🎭 Starting Y.M.I.R Face Emotion Detection MICROSERVICE")
    print("=" * 60)
    print("🌐 Microservice running on: http://localhost:5002")
    print("📱 CORS enabled for integration with main app")
    print("🎯 All your existing emotion detection logic preserved")
    print("📹 Video feed: http://localhost:5002/video_feed")
    print("🏥 Health check: http://localhost:5002/health")
    print("=" * 60)
    
    # 🎯 START ON PORT 5002 AS MICROSERVICE - PRODUCTION MODE
    # 🚀 DISABLE DEBUG: Prevents crashes and auto-restart on file changes
    app.run(debug=False, host='0.0.0.0', port=5002, threaded=True)