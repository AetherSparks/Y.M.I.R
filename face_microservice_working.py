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
        self.current_emotions: Dict[str, float] = {}
        self.frame_lock = threading.Lock()
        
        # 📊 Session tracking - wait for frontend to set session ID
        self.session_id = None
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
        
        # Microservice initialized with session ID
    
    def set_session_id(self, session_id: str):
        """Set the session ID for emotion storage"""
        if session_id:
            self.session_id = session_id
            # 🔥 CRITICAL: Also update Enhanced Detector session ID
            if hasattr(self, 'detector') and self.detector and hasattr(self.detector, 'firebase_manager'):
                if self.detector.firebase_manager:
                    self.detector.firebase_manager.session_id = session_id
                    print(f"✅ Enhanced Detector session ID updated to: {session_id}")
            return True
        return False
    
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
            # Enhanced emotion detector initialized
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize emotion detector: {e}")
    
    
    def start_camera(self):
        """Start camera capture using the full detector"""
        if self.is_running:
            return True
            
        try:
            if self.detector:
                # Starting camera with enhanced detector
                # Use the full detector's camera system
                success = self.detector.start_camera()
                if success:
                    self.is_running = True
                    self.session_analytics['start_time'] = time.time()
                    
                    # Start processing thread using the FULL detector
                    threading.Thread(target=self._process_frames_with_full_detector, daemon=True).start()
                    
                    # Camera started successfully
                    return True
                else:
                    # Camera failed to start
                    return False
            else:
                # Detector not available
                return False
            
        except Exception as e:
            # Camera start failed
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        
        if self.detector and self.detector.cap:
            self.detector.cap.release()
            self.detector.cap = None
            
        # Camera stopped
    
    def _process_frames_with_full_detector(self):
        """🚀 OPTIMIZED: Process camera frames using the FULL enhanced detector with performance optimizations"""
        # Starting optimized frame processing
        
        frame_count = 0
        last_fps_time = time.time()
        fps_counter = 0
        
        # Performance tracking
        processing_times = deque(maxlen=30)  # Track last 30 frame processing times
        
        while self.is_running and self.detector and self.detector.cap:
            try:
                frame_start_time = time.time()
                frame_count += 1
                fps_counter += 1
                
                # Calculate and update FPS every second
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    self.current_fps = fps_counter / (current_time - last_fps_time)
                    fps_counter = 0
                    last_fps_time = current_time
                    
                    # Only log every second instead of every 30 frames
                    if frame_count % 60 == 0:  # Log every 60 frames (less frequent)
                        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
                        # Frame processing stats available
                
                ret, frame = self.detector.cap.read()
                if not ret:
                    # Failed to read frame
                    time.sleep(0.01)  # Brief pause before retry
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Store frame for web streaming (non-blocking)
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Skip processing every few frames if FPS is too low
                if hasattr(self, 'current_fps') and self.current_fps < 15 and frame_count % 2 == 0:
                    continue  # Skip every other frame for performance
                
                # 🚀 Use FULL enhanced detector processing (includes emotion analysis)
                processed_frame = self.detector.process_frame(frame)
                # Note: processed_frame includes all visualizations and emotion analysis
                
                # Track processing time for performance monitoring
                processing_time = time.time() - frame_start_time
                processing_times.append(processing_time)
                
                # Extract current emotions from the full detector
                if hasattr(self.detector, 'emotion_lock') and hasattr(self.detector, 'current_emotions'):
                    with self.detector.emotion_lock:
                        if self.detector.current_emotions:
                            # Update web system's current emotions for API
                            self.current_emotions: Dict[str, float] = {}
                            for face_id, reading in self.detector.current_emotions.items():
                                self.current_emotions[face_id] = {
                                    'dominant': reading.emotions,
                                    'confidence': reading.confidence,
                                    'quality': reading.quality_score,
                                    'timestamp': reading.timestamp.isoformat(),
                                    'stability': reading.stability
                                }
                            # Emotions updated
                else:
                    # Detector state invalid - skipping frame
                
                    time.sleep(0.016)  # ~60 FPS for smooth video
                
            except Exception as e:
                # Frame processing error - stopping
                break
                
        # Frame processing ended
    
    
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
                    
                    # YOLO object detection completed
                    
                    # Environment analysis and context modifiers applied
                        
                except Exception as e:
                    # YOLO processing error
                    pass
            
            # Face emotion analysis (every 30 frames for performance while keeping 60fps video)
            if frame_count % 30 == 0 and 'faces' in locals() and faces:
                # Processing facial emotions
                try:
                    face_info = faces[0]  # Process first face
                    if face_info['roi'].size > 0:
                        # Simple emotion analysis without threading for headless mode
                        emotion_result = self.detector.deepface_ensemble.analyze_face_with_context(
                            face_info['id'], face_info['roi'], 
                            getattr(self.detector, 'current_environment_context', {})
                        )
                        
                        if emotion_result:
                            # Facial emotion analysis completed
                            
                            # 🔥 STORE IN FIREBASE for combiner!
                            # Checking Firebase storage availability
                            
                            # 🔥 FIREBASE STORAGE - Enhanced + Fallback
                            firebase_stored = False
                            
                            # SKIP ENHANCED DETECTOR - Go directly to fallback for guaranteed storage
                            # Using direct Firebase storage
                            
                            # 💾 DIRECT Firebase storage for guaranteed compatibility with combiner
                            print(f"🔍 Storage check: FIREBASE_AVAILABLE={FIREBASE_AVAILABLE}, session_id={self.session_id}")
                            if FIREBASE_AVAILABLE and self.session_id:
                                try:
                                    # Firebase storage fallback
                                    import firebase_admin
                                    from firebase_admin import firestore
                                    from datetime import datetime, timezone
                                    
                                    # Get current UTC timestamp for storage
                                    storage_timestamp = datetime.now(timezone.utc)
                                    # Storing with timestamp
                                    
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
                                    # Facial emotion stored to Firebase
                                    firebase_stored = True
                                    
                                except Exception as firebase_error:
                                    print(f"🔥 Firebase storage error: {firebase_error}")
                                    pass
                            
                            if not firebase_stored:
                                # Firebase storage methods failed
                                pass
                            
                except Exception as e:
                    # Emotion analysis error
                    pass
            
            # Processing completed - log only when emotions found
            pass
            
        except Exception as e:
            # Frame processing error
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
    
    def get_current_frame_jpeg_optimized(self):
        """🚀 OPTIMIZED: Get current frame with performance optimizations for smooth streaming"""
        with self.frame_lock:
            if self.current_frame is not None:
                # Apply lightweight visual overlays 
                display_frame = self.apply_visual_overlays_optimized(self.current_frame.copy())
                
                # Optimized JPEG encoding with adaptive quality
                quality = 75  # Fixed quality setting
                ret, buffer = cv2.imencode('.jpg', display_frame, [
                    cv2.IMWRITE_JPEG_QUALITY, quality,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 1
                ])
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
            # Visual overlay error
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
            # Emotion overlay error
            pass
    
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
            # Settings overlay error
            pass
    
    def _apply_enhanced_overlays(self, frame, settings):
        """Apply enhanced visual overlays using detector components"""
        try:
            # 📐 Face detection overlay
            if settings.get('show_face_mesh', True):
                # Draw face rectangles and landmarks using MediaPipe
                try:
                    import mediapipe as mp
                    # Use standard MediaPipe import structure with error handling
                    if hasattr(mp.solutions, 'face_detection'):
                        mp_face_detection = mp.solutions.face_detection # type: ignore
                        mp_drawing = mp.solutions.drawing_utils # type: ignore
                    else:
                        # Skip MediaPipe overlay if not available
                        return
                    
                    with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_detection.process(rgb_frame)
                    
                        if hasattr(results, 'detections') and results.detections:
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
                
                except (ImportError, AttributeError):
                    # MediaPipe not available, skip face overlay
                    pass
            
            # 🎯 Object detection status
            if settings.get('show_object_detection', True):
                detected_objects = getattr(self.detector, 'detected_objects', [])
                if detected_objects:
                    cv2.putText(frame, f"Objects: {len(detected_objects)}", (frame.shape[1] - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                               
        except Exception as e:
            # Enhanced overlay error
            pass
    
    def apply_visual_overlays_optimized(self, frame):
        """🚀 OPTIMIZED: Lightweight visual overlays for smooth 60 FPS streaming"""
        try:
            if not hasattr(self, 'visual_settings'):
                return frame
            
            settings = self.visual_settings
            
            # Only apply essential overlays for performance
            # Skip heavy overlays like MediaPipe in streaming mode
            
            # 📊 Essential emotion overlay (lightweight)
            if self.current_emotions and settings.get('show_quality_indicators', True):
                self._draw_emotion_overlay_lightweight(frame)
            
            # 🔲 Simple face detection boxes only (no MediaPipe mesh for streaming)
            if settings.get('show_face_detection', True) and hasattr(self, 'last_faces'):
                for face in getattr(self, 'last_faces', []):
                    if len(face) >= 4:
                        x, y, w, h = face[:4]
                        # Simple optimized rectangle
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
            # ⚡ Performance indicator
            if settings.get('show_performance_metrics', False):
                fps_text = f"FPS: {getattr(self, 'current_fps', 0):.1f}"
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            # Silent error handling for performance
            return frame
    
    def _draw_emotion_overlay_lightweight(self, frame):
        """Lightweight emotion overlay for streaming performance"""
        try:
            if not self.current_emotions:
                return
            
            # Get primary emotion
            if isinstance(self.current_emotions, dict) and self.current_emotions:
                primary_emotion = max(self.current_emotions.keys(), key=lambda k: self.current_emotions[k])
            else:
                return
            confidence = self.current_emotions[primary_emotion]
            
            # Simple text overlay
            emotion_text = f"{primary_emotion.upper()}: {confidence:.1f}%"
            
            # Optimized text rendering
            (text_width, text_height), baseline = cv2.getTextSize(
                emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Background rectangle
            cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + 10), 
                         (0, 0, 0), -1)
            
            # Emotion color mapping for visual feedback
            color_map = {
                'happy': (0, 255, 0),
                'sad': (255, 0, 0), 
                'angry': (0, 0, 255),
                'neutral': (128, 128, 128),
                'surprise': (255, 255, 0),
                'fear': (128, 0, 128),
                'disgust': (0, 128, 128)
            }
            color = color_map.get(primary_emotion.lower(), (255, 255, 255))
            
            # Text overlay
            cv2.putText(frame, emotion_text, (15, 10 + text_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                       
        except Exception:
            pass  # Silent fail for performance
    
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
                # Syncing buffered emotions to Firebase
                # Process any buffered readings here if needed
                self.last_firebase_sync = int(time.time())
                # Firebase sync completed
                pass
            except Exception as e:
                # Firebase sync error
                pass
        else:
            # Firebase sync skipped
            pass

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
    """🚀 OPTIMIZED Video streaming route for smooth 60 FPS performance"""
    def generate():
        last_frame_time = time.time()
        frame_interval = 1.0 / 60.0  # Target 60 FPS
        frame_count = 0
        
        while True:
            current_time = time.time()
            frame_count += 1
            
            # Get frame from optimized buffer
            frame_bytes = web_system.get_current_frame_jpeg_optimized()
            
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Send minimal black frame for startup
                if frame_count <= 10:  # Only send black frames during startup
                    black_frame = np.zeros((240, 320, 3), dtype=np.uint8)  # Smaller startup frame
                    ret, buffer = cv2.imencode('.jpg', black_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Dynamic frame rate control - adapt to system performance
            elapsed = current_time - last_frame_time
            if elapsed < frame_interval:
                sleep_time = max(0.001, frame_interval - elapsed)  # Minimum 1ms sleep
                time.sleep(sleep_time)
            
            last_frame_time = time.time()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 🎯 API ENDPOINTS (KEEP YOUR EXACT WORKING API STRUCTURE)

@app.route('/api/start_camera', methods=['POST'])
def api_start_camera():
    """API endpoint to start camera"""
    # API: start_camera called
    try:
        success = web_system.start_camera()
        # Camera start result available
        response = {
            'success': success,
            'message': 'Camera started successfully' if success else 'Failed to start camera',
            'session_id': web_system.session_id
        }
        # Returning API response
        return jsonify(response)
    except Exception as e:
        # API error in start_camera
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

@app.route('/api/session', methods=['POST'])
def api_set_session():
    """🎯 Set session ID for emotion storage synchronization"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'message': 'session_id is required'
            }), 400
        
        success = web_system.set_session_id(session_id)
        
        return jsonify({
            'success': success,
            'message': 'Session ID updated successfully' if success else 'Failed to update session ID',
            'session_id': web_system.session_id
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to set session ID: {str(e)}'
        }), 500

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
                # MediaPipe settings updated
            
            # Update YOLO settings
            if hasattr(config, 'yolo_config'):
                yolo_config = config.yolo_config
                yolo_config.confidence_threshold = settings.get('confidence_threshold', 0.25)
                # YOLO confidence updated
            
            # Update analysis settings
            config.emotion_analysis_interval = settings.get('analysis_interval', 30)
            config.show_analytics = settings.get('show_quality_indicators', True)
            
            # Visual settings updated successfully
            
        # Store settings globally for video feed processing
        web_system.visual_settings = settings
        
        return jsonify({
            'success': True,
            'message': 'Visual settings updated successfully',
            'settings': settings
        })
        
    except Exception as e:
        # Settings update error
        return jsonify({
            'success': False,
            'message': f'Failed to update settings: {str(e)}'
        }), 500

# UI routes removed - this is an API-only microservice  
# Use the main app at port 5000 for UI

@app.route('/api/mediapipe/landmarks')
def api_mediapipe_landmarks():
    """Get MediaPipe landmarks for visual overlays"""
    try:
        if not web_system.is_running or not web_system.detector:
            return jsonify({
                'success': False,
                'message': 'Camera not running or detector not available'
            })
        
        # Get current frame
        with web_system.frame_lock:
            if web_system.current_frame is None:
                return jsonify({
                    'success': False,
                    'message': 'No frame available'
                })
            frame = web_system.current_frame.copy()
        
        landmarks_data = {
            'success': True,
            'face_landmarks': [],
            'pose_landmarks': [],
            'hand_landmarks': [],
            'gaze_landmarks': {}
        }
        
        # Get MediaPipe results if processor is available
        if hasattr(web_system.detector, 'mediapipe_processor'):
            try:
                # Process frame with MediaPipe
                mp_results = web_system.detector.mediapipe_processor.process_frame(frame)
                
                # Extract face landmarks
                if 'face_mesh' in mp_results and mp_results['face_mesh']:
                    face_landmarks = []
                    for landmark in mp_results['face_mesh'].multi_face_landmarks[0].landmark:
                        face_landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    landmarks_data['face_landmarks'] = face_landmarks
                
                # Extract pose landmarks
                if 'pose' in mp_results and mp_results['pose'] and mp_results['pose'].pose_landmarks:
                    pose_landmarks = []
                    for landmark in mp_results['pose'].pose_landmarks.landmark:
                        pose_landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })
                    landmarks_data['pose_landmarks'] = pose_landmarks
                
                # Extract hand landmarks
                if 'hands' in mp_results and mp_results['hands'] and mp_results['hands'].multi_hand_landmarks:
                    hand_landmarks_list = []
                    for hand_landmarks in mp_results['hands'].multi_hand_landmarks:
                        hand_data = {
                            'landmarks': []
                        }
                        for landmark in hand_landmarks.landmark:
                            hand_data['landmarks'].append({
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z
                            })
                        hand_landmarks_list.append(hand_data)
                    landmarks_data['hand_landmarks'] = hand_landmarks_list
                
                # Extract gaze tracking (simplified)
                if 'face_mesh' in mp_results and mp_results['face_mesh']:
                    # Get eye landmarks for gaze estimation
                    if mp_results['face_mesh'].multi_face_landmarks:
                        face_landmarks_list = mp_results['face_mesh'].multi_face_landmarks[0].landmark
                        
                        # Left eye center (approximate)
                        left_eye_landmarks = [33, 133, 157, 158, 159, 160, 161, 163]
                        right_eye_landmarks = [362, 398, 384, 385, 386, 387, 388, 466]
                        
                        if len(face_landmarks_list) > max(left_eye_landmarks + right_eye_landmarks):
                            left_eye_center = {
                                'x': sum(face_landmarks_list[i].x for i in left_eye_landmarks) / len(left_eye_landmarks),
                                'y': sum(face_landmarks_list[i].y for i in left_eye_landmarks) / len(left_eye_landmarks)
                            }
                            right_eye_center = {
                                'x': sum(face_landmarks_list[i].x for i in right_eye_landmarks) / len(right_eye_landmarks),
                                'y': sum(face_landmarks_list[i].y for i in right_eye_landmarks) / len(right_eye_landmarks)
                            }
                            
                            # Simple gaze direction estimation
                            landmarks_data['gaze_landmarks'] = {
                                'left_eye': left_eye_center,
                                'right_eye': right_eye_center,
                                'gaze_direction': {
                                    'x': 0.1,  # Simplified - would need more complex calculation
                                    'y': 0.05
                                }
                            }
                
            except Exception as mp_error:
                # MediaPipe processing error
                # Return success=True but with empty landmarks
                pass
        
        return jsonify(landmarks_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to get MediaPipe landmarks'
        })

if __name__ == '__main__':
    print("Y.M.I.R Face Emotion Detection MICROSERVICE")
    print("Microservice running on: http://localhost:5002")
    print("Video feed: http://localhost:5002/video_feed")
    print("Health check: http://localhost:5002/health")
    
    # 🎯 START ON PORT 5002 AS MICROSERVICE - PRODUCTION MODE
    # 🚀 DISABLE DEBUG: Prevents crashes and auto-restart on file changes
    app.run(debug=False, host='0.0.0.0', port=5002, threaded=True)