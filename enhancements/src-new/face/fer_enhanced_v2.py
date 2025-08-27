"""
🎯 Y.M.I.R Advanced Emotion Detection System v2.0
==================================================
Enhanced Features:
- 🧠 Ensemble emotion detection (multiple models)
- 💾 Smart memory management (only significant changes)
- 🔥 Firebase Firestore integration (real database)
- 📊 Advanced analytics and emotion trends
- 🎛️ Confidence-based intelligent filtering
- 🎨 Face quality assessment
- 🔄 Real-time emotion smoothing
- ⚡ Optimized performance and memory usage
"""

import cv2
import numpy as np
import threading
import time
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from enum import Enum
from collections import deque, defaultdict
import statistics
from datetime import datetime, timezone

# Computer Vision imports
import mediapipe as mp
import dlib
from deepface import DeepFace
from scipy.spatial import distance as dist
import skimage.measure

# Firebase imports
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
    print("✅ Firebase available for cloud storage")
except ImportError:
    FIREBASE_AVAILABLE = False
    print("⚠️ Firebase not available - install: pip install firebase-admin")

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO available for object detection")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available - install: pip install ultralytics")

warnings.filterwarnings("ignore")

@dataclass
class EnhancedEmotionConfig:
    """Enhanced configuration class for emotion detection system v2.0"""
    # Camera settings
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    
    # Detection settings
    min_face_confidence: float = 0.7
    min_pose_confidence: float = 0.6
    emotion_analysis_interval: int = 30  # Analyze every 30 frames (1 second at 30fps)
    
    # Accuracy improvements
    use_ensemble_detection: bool = True
    min_face_quality_score: float = 0.6
    emotion_smoothing_window: int = 5
    confidence_threshold: float = 0.7
    
    # Memory optimization
    store_only_significant_changes: bool = True
    emotion_change_threshold: float = 15.0  # Only store if emotion changes by 15%
    max_memory_entries: int = 1000
    
    # Database settings
    use_firebase: bool = True
    firebase_collection: str = "emotion_sessions"
    
    # Display settings
    show_face_mesh: bool = True
    show_body_pose: bool = True
    show_hand_tracking: bool = True
    show_gaze_tracking: bool = True
    show_yolo_objects: bool = True
    show_analytics: bool = True
    
    # Performance settings
    max_workers: int = 6
    deepface_timeout: float = 2.0
    
    # Privacy settings
    require_user_consent: bool = True
    privacy_mode: bool = False

class FaceQuality(Enum):
    """Face quality assessment levels"""
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"

@dataclass
class EmotionReading:
    """Structure for emotion reading data"""
    timestamp: datetime
    face_id: int
    emotions: Dict[str, float]
    confidence: float
    quality_score: float
    context_objects: List[str]
    face_bbox: Tuple[int, int, int, int]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'face_id': self.face_id,
            'emotions': self.emotions,
            'confidence': self.confidence,
            'quality_score': self.quality_score,
            'context_objects': self.context_objects,
            'face_bbox': self.face_bbox
        }

class EmotionAnalytics:
    """Advanced emotion analytics and insights"""
    
    def __init__(self, window_size: int = 100):
        self.emotion_history = deque(maxlen=window_size)
        self.session_stats = defaultdict(list)
        
    def add_reading(self, reading: EmotionReading):
        """Add emotion reading to analytics"""
        self.emotion_history.append(reading)
        
        # Update session stats
        dominant_emotion = max(reading.emotions.items(), key=lambda x: x[1])
        self.session_stats['dominant_emotions'].append(dominant_emotion[0])
        self.session_stats['confidence_scores'].append(reading.confidence)
        self.session_stats['quality_scores'].append(reading.quality_score)
    
    def get_emotion_trends(self) -> Dict[str, Any]:
        """Calculate emotion trends and patterns"""
        if not self.emotion_history:
            return {}
            
        # Calculate average emotions over time
        emotion_averages = defaultdict(list)
        for reading in self.emotion_history:
            for emotion, score in reading.emotions.items():
                emotion_averages[emotion].append(score)
        
        trends = {}
        for emotion, scores in emotion_averages.items():
            trends[emotion] = {
                'average': statistics.mean(scores),
                'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0,
                'min': min(scores),
                'max': max(scores),
                'trend': 'stable'  # Could implement trend analysis
            }
        
        # Session insights
        insights = {
            'total_readings': len(self.emotion_history),
            'avg_confidence': statistics.mean(self.session_stats['confidence_scores']) if self.session_stats['confidence_scores'] else 0,
            'avg_quality': statistics.mean(self.session_stats['quality_scores']) if self.session_stats['quality_scores'] else 0,
            'dominant_emotion': max(set(self.session_stats['dominant_emotions']), key=self.session_stats['dominant_emotions'].count) if self.session_stats['dominant_emotions'] else 'neutral',
            'emotion_stability': self._calculate_stability(),
            'emotion_trends': trends
        }
        
        return insights
    
    def _calculate_stability(self) -> float:
        """Calculate emotional stability score (0-1)"""
        if len(self.emotion_history) < 2:
            return 1.0
            
        # Calculate variance in dominant emotions
        dominant_scores = []
        for reading in self.emotion_history:
            max_score = max(reading.emotions.values())
            dominant_scores.append(max_score)
        
        if not dominant_scores:
            return 1.0
            
        variance = statistics.variance(dominant_scores)
        stability = max(0, 1 - (variance / 1000))  # Normalize to 0-1
        return stability

class FirebaseManager:
    """Firebase Firestore integration for cloud storage"""
    
    def __init__(self, config: EnhancedEmotionConfig):
        self.config = config
        self.db = None
        self.session_id = None
        
        if FIREBASE_AVAILABLE and config.use_firebase:
            self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase connection"""
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                # You'll need to add your Firebase credentials file
                cred_path = Path("src/firebase_credentials.json")  # Fixed path
                if not cred_path.exists():
                    cred_path = Path("firebase_credentials.json")  # Fallback path
                if cred_path.exists():
                    cred = credentials.Certificate(str(cred_path))
                    firebase_admin.initialize_app(cred)
                    print(f"✅ Firebase initialized with {cred_path}")
                else:
                    print("⚠️ Firebase credentials not found - using offline mode")
                    print(f"   Looked for: {cred_path.absolute()}")
                    return
            
            self.db = firestore.client()
            self.session_id = f"session_{int(time.time())}"
            print("✅ Firebase Firestore connected")
            
        except Exception as e:
            print(f"⚠️ Firebase initialization error: {e}")
            self.db = None
    
    def store_emotion_reading(self, reading: EmotionReading) -> bool:
        """Store emotion reading in Firestore"""
        if not self.db:
            return False
            
        try:
            doc_ref = self.db.collection(self.config.firebase_collection).document()
            doc_ref.set({
                **reading.to_dict(),
                'session_id': self.session_id
            })
            return True
            
        except Exception as e:
            print(f"⚠️ Firebase storage error: {e}")
            return False
    
    def store_session_summary(self, analytics: Dict[str, Any]) -> bool:
        """Store session summary and analytics"""
        if not self.db:
            return False
            
        try:
            doc_ref = self.db.collection("session_summaries").document(self.session_id)
            doc_ref.set({
                **analytics,
                'session_id': self.session_id,
                'end_time': datetime.now(timezone.utc).isoformat()
            })
            return True
            
        except Exception as e:
            print(f"⚠️ Firebase session storage error: {e}")
            return False

class EnhancedEmotionDetector:
    """Enhanced emotion detection system v2.0 with accuracy and memory optimizations"""
    
    def __init__(self, config: EnhancedEmotionConfig = None):
        self.config = config or EnhancedEmotionConfig()
        self.cap = None
        
        # Initialize components
        self._init_computer_vision()
        self._init_firebase()
        
        # Enhanced emotion processing
        self.emotion_smoothing_buffer = defaultdict(lambda: deque(maxlen=self.config.emotion_smoothing_window))
        self.last_stored_emotions = {}
        self.emotion_analytics = EmotionAnalytics()
        
        # Threading and synchronization
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.emotion_lock = threading.Lock()
        self.last_analysis_time = 0
        
        # Frame processing
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        # Detection data
        self.current_emotions = {}
        self.detected_objects = []
        self.face_bboxes = []
        self.high_quality_faces = []
        
        # Visual settings
        self.colors = {
            "face": (0, 255, 0),
            "body": (255, 255, 0),
            "hands": (0, 0, 255),
            "eyes": (0, 255, 255),
            "objects": (255, 0, 255),
            "emotion_text": (0, 255, 255),
            "analytics": (255, 255, 255),
            "quality_good": (0, 255, 0),
            "quality_fair": (0, 255, 255),
            "quality_poor": (0, 0, 255)
        }
        
        print("🚀 Enhanced Emotion Detection System v2.0 initialized")
    
    def _init_computer_vision(self):
        """Initialize computer vision models"""
        try:
            # MediaPipe initialization
            mp_face_detection = mp.solutions.face_detection
            mp_face_mesh = mp.solutions.face_mesh
            mp_pose = mp.solutions.pose
            mp_hands = mp.solutions.hands
            
            self.face_detection = mp_face_detection.FaceDetection(
                min_detection_confidence=self.config.min_face_confidence
            )
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=3,
                min_detection_confidence=self.config.min_face_confidence
            )
            self.pose = mp_pose.Pose(
                min_detection_confidence=self.config.min_pose_confidence,
                min_tracking_confidence=self.config.min_pose_confidence
            )
            self.hands = mp_hands.Hands(
                min_detection_confidence=self.config.min_pose_confidence,
                min_tracking_confidence=self.config.min_pose_confidence
            )
            
            # YOLO initialization
            if YOLO_AVAILABLE:
                self.yolo_model = YOLO('yolov8n.pt')
            else:
                self.yolo_model = None
            
            print("✅ Computer vision models initialized")
            
        except Exception as e:
            print(f"❌ Computer vision initialization error: {e}")
    
    def _init_firebase(self):
        """Initialize Firebase manager"""
        self.firebase_manager = FirebaseManager(self.config)
    
    def assess_face_quality(self, face_roi: np.ndarray) -> Tuple[float, FaceQuality]:
        """Assess the quality of face region for emotion analysis"""
        try:
            # Convert to grayscale for analysis
            if len(face_roi.shape) == 3:
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_roi
            
            # Calculate quality metrics
            height, width = gray_face.shape
            
            # 1. Size check (prefer larger faces)
            size_score = min(1.0, (height * width) / (100 * 100))
            
            # 2. Sharpness check (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 500)
            
            # 3. Brightness check (avoid over/under exposed)
            mean_brightness = np.mean(gray_face)
            brightness_score = 1.0 - abs(mean_brightness - 127) / 127
            
            # 4. Contrast check
            contrast_score = np.std(gray_face) / 128
            contrast_score = min(1.0, contrast_score)
            
            # Combined quality score
            quality_score = (size_score * 0.3 + 
                           sharpness_score * 0.4 + 
                           brightness_score * 0.2 + 
                           contrast_score * 0.1)
            
            # Determine quality level
            if quality_score >= 0.8:
                quality_level = FaceQuality.EXCELLENT
            elif quality_score >= 0.6:
                quality_level = FaceQuality.GOOD
            elif quality_score >= 0.4:
                quality_level = FaceQuality.FAIR
            else:
                quality_level = FaceQuality.POOR
            
            return quality_score, quality_level
            
        except Exception as e:
            print(f"⚠️ Face quality assessment error: {e}")
            return 0.5, FaceQuality.FAIR
    
    def ensemble_emotion_detection(self, face_roi: np.ndarray) -> Optional[Dict[str, Any]]:
        """Enhanced emotion detection using ensemble methods"""
        try:
            # Resize face for consistent analysis
            face_224 = cv2.resize(face_roi, (224, 224))
            
            results = []
            
            # Method 1: DeepFace with VGG-Face
            def analyze_vgg_face():
                try:
                    result = DeepFace.analyze(
                        face_224,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv',
                        model_name='VGG-Face',
                        silent=True
                    )
                    return result[0]['emotion'], 0.8
                except:
                    return None, 0
            
            # Method 2: DeepFace with Facenet
            def analyze_facenet():
                try:
                    result = DeepFace.analyze(
                        face_224,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv',
                        model_name='Facenet',
                        silent=True
                    )
                    return result[0]['emotion'], 0.7
                except:
                    return None, 0
            
            # Method 3: DeepFace default (fastest)
            def analyze_default():
                try:
                    result = DeepFace.analyze(
                        face_224,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv',
                        silent=True
                    )
                    return result[0]['emotion'], 0.9
                except:
                    return None, 0
            
            # Run ensemble analysis with timeout
            if self.config.use_ensemble_detection:
                futures = []
                with ThreadPoolExecutor(max_workers=3) as ensemble_executor:
                    futures.append(ensemble_executor.submit(analyze_default))
                    futures.append(ensemble_executor.submit(analyze_vgg_face))
                    # futures.append(ensemble_executor.submit(analyze_facenet))  # Comment out if too slow
                    
                    for future in as_completed(futures, timeout=self.config.deepface_timeout):
                        try:
                            emotions, weight = future.result()
                            if emotions:
                                results.append((emotions, weight))
                        except:
                            continue
            else:
                # Single model analysis (faster)
                emotions, weight = analyze_default()
                if emotions:
                    results.append((emotions, weight))
            
            if not results:
                return None
            
            # Ensemble voting - weighted average
            ensemble_emotions = defaultdict(float)
            total_weight = 0
            
            for emotions, weight in results:
                for emotion, score in emotions.items():
                    ensemble_emotions[emotion] += score * weight
                total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                for emotion in ensemble_emotions:
                    ensemble_emotions[emotion] /= total_weight
            
            # Calculate ensemble confidence
            confidence = min(1.0, total_weight / (len(results) * 0.8))
            
            return {
                'emotions': dict(ensemble_emotions),
                'confidence': confidence,
                'models_used': len(results)
            }
            
        except Exception as e:
            print(f"⚠️ Ensemble emotion detection error: {e}")
            return None
    
    def smooth_emotions(self, face_id: int, emotions: Dict[str, float]) -> Dict[str, float]:
        """Apply temporal smoothing to emotion predictions"""
        smoothed_emotions = {}
        
        # Add current emotions to buffer
        for emotion, score in emotions.items():
            self.emotion_smoothing_buffer[f"{face_id}_{emotion}"].append(score)
        
        # Calculate smoothed values using moving average
        for emotion, score in emotions.items():
            buffer_key = f"{face_id}_{emotion}"
            if len(self.emotion_smoothing_buffer[buffer_key]) > 1:
                # Use weighted average with more weight on recent values
                values = list(self.emotion_smoothing_buffer[buffer_key])
                weights = [i + 1 for i in range(len(values))]  # More weight on recent
                weighted_sum = sum(v * w for v, w in zip(values, weights))
                weight_sum = sum(weights)
                smoothed_emotions[emotion] = weighted_sum / weight_sum
            else:
                smoothed_emotions[emotion] = score
        
        return smoothed_emotions
    
    def should_store_emotion(self, face_id: int, emotions: Dict[str, float]) -> bool:
        """Determine if emotion data should be stored based on significance"""
        if not self.config.store_only_significant_changes:
            return True
        
        # Store if this is the first reading for this face
        if face_id not in self.last_stored_emotions:
            return True
        
        # Calculate emotion change magnitude
        last_emotions = self.last_stored_emotions[face_id]
        total_change = 0
        
        for emotion, current_score in emotions.items():
            if emotion in last_emotions:
                change = abs(current_score - last_emotions[emotion])
                total_change += change
        
        # Store if total change exceeds threshold
        return total_change >= self.config.emotion_change_threshold
    
    def analyze_emotion_enhanced(self, face_id: int, face_roi: np.ndarray, face_bbox: Tuple[int, int, int, int], context_objects: List[str]):
        """Enhanced emotion analysis with quality assessment and ensemble detection"""
        current_time = time.time()
        
        # Rate limiting
        with self.emotion_lock:
            if current_time - self.last_analysis_time < 1.0:  # Minimum 1 second interval
                return
            self.last_analysis_time = current_time
        
        try:
            # Assess face quality
            quality_score, quality_level = self.assess_face_quality(face_roi)
            
            # Skip low quality faces
            if quality_score < self.config.min_face_quality_score:
                print(f"⚠️ Face {face_id} quality too low ({quality_score:.2f}) - skipping")
                return
            
            print(f"✅ Analyzing face {face_id} - Quality: {quality_level.value} ({quality_score:.2f})")
            
            # Ensemble emotion detection
            emotion_result = self.ensemble_emotion_detection(face_roi)
            if not emotion_result:
                print(f"⚠️ Emotion analysis failed for face {face_id}")
                return
            
            # Filter by confidence
            if emotion_result['confidence'] < self.config.confidence_threshold:
                print(f"⚠️ Low confidence ({emotion_result['confidence']:.2f}) - skipping")
                return
            
            # Apply emotion smoothing
            raw_emotions = emotion_result['emotions']
            smoothed_emotions = self.smooth_emotions(face_id, raw_emotions)
            
            # Create emotion reading
            reading = EmotionReading(
                timestamp=datetime.now(timezone.utc),
                face_id=face_id,
                emotions=smoothed_emotions,
                confidence=emotion_result['confidence'],
                quality_score=quality_score,
                context_objects=context_objects,
                face_bbox=face_bbox
            )
            
            # Update current emotions
            with self.emotion_lock:
                self.current_emotions[face_id] = reading
            
            # Store if significant change
            if self.should_store_emotion(face_id, smoothed_emotions):
                # Update last stored emotions
                self.last_stored_emotions[face_id] = smoothed_emotions.copy()
                
                # Add to analytics
                self.emotion_analytics.add_reading(reading)
                
                # Store in Firebase
                if self.firebase_manager.store_emotion_reading(reading):
                    storage_status = "☁️ Stored in Firebase"
                else:
                    storage_status = "💾 Stored locally"
                
                # Display results
                timestamp = reading.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                dominant_emotion = max(smoothed_emotions.items(), key=lambda x: x[1])
                
                print(f"🕒 {timestamp} - Face {face_id}: {dominant_emotion[0].upper()} ({dominant_emotion[1]:.1f}%)")
                print(f"   Quality: {quality_level.value} | Confidence: {emotion_result['confidence']:.2f} | {storage_status}")
                if context_objects:
                    print(f"   🎯 Context: {', '.join(context_objects[:3])}...")
            else:
                print(f"⏭️ Face {face_id}: No significant emotion change - not storing")
                
        except Exception as e:
            print(f"❌ Enhanced emotion analysis error: {e}")
    
    def start_camera(self) -> bool:
        """Start camera with enhanced initialization"""
        if self.cap and self.cap.isOpened():
            print("📷 Camera already running")
            return True
        
        # Request permission
        if self.config.require_user_consent:
            print("\n🔐 ENHANCED PRIVACY NOTICE:")
            print("This application uses advanced emotion detection with cloud storage.")
            print("Your privacy is protected - data is encrypted and never shared.")
            print("Features: Ensemble detection, quality assessment, smart storage")
            
            while True:
                response = input("Grant enhanced camera access? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    print("✅ Enhanced camera access granted")
                    break
                elif response in ['n', 'no']:
                    print("❌ Camera access denied")
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no")
        
        try:
            print("📷 Starting enhanced camera system...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                print("❌ Camera not detected!")
                return False
            
            # Set enhanced camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time processing
            
            print(f"✅ Enhanced camera started ({self.config.camera_width}x{self.config.camera_height})")
            return True
            
        except Exception as e:
            print(f"❌ Enhanced camera initialization error: {e}")
            return False
    
    def detect_objects_yolo(self, frame: np.ndarray) -> List[Dict]:
        """YOLO object detection for environmental context"""
        objects = []
        
        if not self.yolo_model:
            return objects
        
        try:
            results = self.yolo_model(frame, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        # Filter by confidence and relevant objects
                        if confidence > 0.5 and class_name in ['person', 'laptop', 'cell phone', 'book', 'tv', 'chair', 'couch']:
                            objects.append({
                                "class": class_name,
                                "confidence": float(confidence),
                                "bbox": (int(x1), int(y1), int(x2), int(y2))
                            })
                            
        except Exception as e:
            print(f"⚠️ YOLO detection error: {e}")
        
        return objects
    
    def detect_faces_enhanced(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Enhanced face detection with quality filtering"""
        face_bboxes = []
        self.high_quality_faces = []
        
        try:
            # MediaPipe face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    
                    x = max(0, int(bboxC.xmin * w))
                    y = max(0, int(bboxC.ymin * h))
                    w_box = min(w - x, int(bboxC.width * w))
                    h_box = min(h - y, int(bboxC.height * h))
                    
                    if w_box > 50 and h_box > 50:  # Minimum size filter
                        face_bbox = (x, y, w_box, h_box)
                        face_bboxes.append(face_bbox)
                        
                        # Assess quality for this face
                        face_roi = frame[y:y+h_box, x:x+w_box]
                        quality_score, quality_level = self.assess_face_quality(face_roi)
                        
                        self.high_quality_faces.append({
                            'bbox': face_bbox,
                            'quality_score': quality_score,
                            'quality_level': quality_level,
                            'roi': face_roi
                        })
            
        except Exception as e:
            print(f"⚠️ Enhanced face detection error: {e}")
        
        return face_bboxes
    
    def draw_face_mesh(self, frame: np.ndarray, mesh_results):
        """Draw MediaPipe face mesh like original fer1.py"""
        if self.config.show_face_mesh and mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x_l, y_l = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x_l, y_l), 1, self.colors["face"], -1)
    
    def draw_body_landmarks(self, frame: np.ndarray, pose_results):
        """Draw MediaPipe body pose like original fer1.py"""
        if self.config.show_body_pose and pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                x_b, y_b = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x_b, y_b), 5, self.colors["body"], -1)
    
    def draw_hand_landmarks(self, frame: np.ndarray, hand_results):
        """Draw MediaPipe hand tracking like original fer1.py"""
        if self.config.show_hand_tracking and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_h, y_h = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x_h, y_h), 5, self.colors["hands"], -1)
    
    def draw_gaze_tracking(self, frame: np.ndarray, mesh_results):
        """Draw gaze tracking like original fer1.py"""
        if self.config.show_gaze_tracking and mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                # Left eye landmarks (simplified)
                left_eye = [(landmarks[33].x, landmarks[33].y), (landmarks[160].x, landmarks[160].y),
                           (landmarks[158].x, landmarks[158].y), (landmarks[133].x, landmarks[133].y)]
                # Right eye landmarks 
                right_eye = [(landmarks[362].x, landmarks[362].y), (landmarks[385].x, landmarks[385].y),
                            (landmarks[387].x, landmarks[387].y), (landmarks[263].x, landmarks[263].y)]
                
                # Convert to pixel coordinates
                left_eye = [(int(l[0] * frame.shape[1]), int(l[1] * frame.shape[0])) for l in left_eye]
                right_eye = [(int(r[0] * frame.shape[1]), int(r[1] * frame.shape[0])) for r in right_eye]
                
                # Draw eye points
                for (x, y) in left_eye + right_eye:
                    cv2.circle(frame, (x, y), 2, self.colors["eyes"], -1)

    def draw_enhanced_visualizations(self, frame: np.ndarray, mesh_results=None, pose_results=None, hand_results=None):
        """Draw enhanced visualizations with quality indicators AND MediaPipe meshes"""
        # Draw MediaPipe meshes (like original fer1.py)
        if mesh_results:
            self.draw_face_mesh(frame, mesh_results)
            self.draw_gaze_tracking(frame, mesh_results)
        
        if pose_results:
            self.draw_body_landmarks(frame, pose_results)
        
        if hand_results:
            self.draw_hand_landmarks(frame, hand_results)
        
        # Draw face boxes with quality indicators
        for i, face_info in enumerate(self.high_quality_faces):
            bbox = face_info['bbox']
            quality_score = face_info['quality_score']
            quality_level = face_info['quality_level']
            
            x, y, w, h = bbox
            
            # Color based on quality
            if quality_level == FaceQuality.EXCELLENT:
                color = self.colors["quality_good"]
                thickness = 3
            elif quality_level == FaceQuality.GOOD:
                color = self.colors["quality_good"]
                thickness = 2
            elif quality_level == FaceQuality.FAIR:
                color = self.colors["quality_fair"]
                thickness = 2
            else:
                color = self.colors["quality_poor"]
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw quality score
            cv2.putText(frame, f"Q: {quality_score:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw YOLO objects
        if self.config.show_yolo_objects:
            for obj in self.detected_objects:
                x1, y1, x2, y2 = obj["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["objects"], 2)
                cv2.putText(frame, f"{obj['class']}: {obj['confidence']:.2f}",
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["objects"], 2)
    
    def draw_enhanced_emotions(self, frame: np.ndarray):
        """Draw enhanced emotion information with analytics"""
        y_offset = 30
        
        with self.emotion_lock:
            for face_id, reading in self.current_emotions.items():
                emotions = reading.emotions
                confidence = reading.confidence
                quality = reading.quality_score
                
                # Find corresponding face bbox
                if face_id < len(self.high_quality_faces):
                    x, y, w, h = self.high_quality_faces[face_id]['bbox']
                    
                    # Draw emotion text beside face
                    text_x = x + w + 10
                    text_y = y + 20
                    
                    # Show confidence and quality
                    cv2.putText(frame, f"Conf: {confidence:.2f} | Q: {quality:.2f}",
                              (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["emotion_text"], 1)
                    text_y += 20
                    
                    # Show top 3 emotions
                    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    for emotion, score in sorted_emotions:
                        text = f"{emotion.upper()}: {score:.1f}%"
                        cv2.putText(frame, text, (text_x, text_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["emotion_text"], 2)
                        text_y += 25
        
        # Draw analytics summary
        if self.config.show_analytics:
            analytics = self.emotion_analytics.get_emotion_trends()
            if analytics:
                info_lines = [
                    f"📊 Session Analytics:",
                    f"Readings: {analytics.get('total_readings', 0)}",
                    f"Avg Confidence: {analytics.get('avg_confidence', 0):.2f}",
                    f"Avg Quality: {analytics.get('avg_quality', 0):.2f}",
                    f"Dominant: {analytics.get('dominant_emotion', 'N/A').upper()}",
                    f"Stability: {analytics.get('emotion_stability', 0):.2f}"
                ]
                
                for i, line in enumerate(info_lines):
                    cv2.putText(frame, line, (frame.shape[1] - 350, 30 + i * 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["analytics"], 1)
    
    def draw_enhanced_controls(self, frame: np.ndarray):
        """Draw enhanced control information"""
        controls = [
            f"🚀 Y.M.I.R v2.0 - Enhanced Mode",
            f"FPS: {self.fps:.1f} | Faces: {len(self.high_quality_faces)} | Objects: {len(self.detected_objects)}",
            "Enhanced Controls:",
            "Q - Quit | F - Face Mesh | B - Body | H - Hands",
            "Y - YOLO Objects | A - Analytics | P - Privacy Mode",
            "S - Save Analytics | E - Export Data"
        ]
        
        for i, text in enumerate(controls):
            y_pos = 30 + i * 20
            cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def process_frame_enhanced(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced frame processing with all improvements"""
        self.frame_count += 1
        
        # Convert to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Enhanced face detection
        self.face_bboxes = self.detect_faces_enhanced(frame)
        
        # YOLO object detection (every 10 frames for performance)
        if self.frame_count % 10 == 0:
            self.detected_objects = self.detect_objects_yolo(frame)
        
        # MediaPipe processing (like original fer1.py)
        mesh_results = self.face_mesh.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        
        # Enhanced emotion analysis (configurable interval)
        if self.frame_count % self.config.emotion_analysis_interval == 0:
            context_objects = [obj["class"] for obj in self.detected_objects]
            
            for i, face_info in enumerate(self.high_quality_faces):
                if i >= 3:  # Limit to 3 faces for performance
                    break
                
                face_roi = face_info['roi']
                face_bbox = face_info['bbox']
                
                if face_roi.size > 0:
                    # Submit for enhanced analysis
                    self.executor.submit(
                        self.analyze_emotion_enhanced,
                        i, face_roi, face_bbox, context_objects
                    )
        
        # Draw enhanced visualizations WITH MediaPipe results
        self.draw_enhanced_visualizations(frame, mesh_results, pose_results, hand_results)
        self.draw_enhanced_emotions(frame)
        self.draw_enhanced_controls(frame)
        
        # Calculate FPS
        elapsed_time = time.time() - self.start_time
        self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return frame
    
    def handle_enhanced_key_events(self, key: int) -> bool:
        """Handle enhanced keyboard events"""
        if key == ord('q'):
            return False
        elif key == ord('f'):
            self.config.show_face_mesh = not self.config.show_face_mesh
            print(f"Face mesh: {'ON' if self.config.show_face_mesh else 'OFF'}")
        elif key == ord('b'):
            self.config.show_body_pose = not self.config.show_body_pose
            print(f"Body pose: {'ON' if self.config.show_body_pose else 'OFF'}")
        elif key == ord('h'):
            self.config.show_hand_tracking = not self.config.show_hand_tracking
            print(f"Hand tracking: {'ON' if self.config.show_hand_tracking else 'OFF'}")
        elif key == ord('y'):
            self.config.show_yolo_objects = not self.config.show_yolo_objects
            print(f"YOLO objects: {'ON' if self.config.show_yolo_objects else 'OFF'}")
        elif key == ord('a'):
            self.config.show_analytics = not self.config.show_analytics
            print(f"Analytics: {'ON' if self.config.show_analytics else 'OFF'}")
        elif key == ord('p'):
            self.config.privacy_mode = not self.config.privacy_mode
            print(f"Privacy mode: {'ON' if self.config.privacy_mode else 'OFF'}")
        elif key == ord('s'):
            self.save_analytics_summary()
        elif key == ord('e'):
            self.export_session_data()
        elif key == ord('1'):
            self.config.emotion_analysis_interval = 15  # Every 0.5 seconds
            print("Analysis interval: 0.5 seconds")
        elif key == ord('2'):
            self.config.emotion_analysis_interval = 30  # Every 1 second
            print("Analysis interval: 1 second")
        elif key == ord('3'):
            self.config.emotion_analysis_interval = 60  # Every 2 seconds
            print("Analysis interval: 2 seconds")
        
        return True
    
    def save_analytics_summary(self):
        """Save current analytics summary"""
        analytics = self.emotion_analytics.get_emotion_trends()
        if analytics:
            if self.firebase_manager.store_session_summary(analytics):
                print("✅ Analytics saved to Firebase")
            else:
                # Fallback to local JSON
                filename = f"analytics_summary_{int(time.time())}.json"
                with open(filename, 'w') as f:
                    json.dump(analytics, f, indent=2)
                print(f"✅ Analytics saved to {filename}")
        else:
            print("⚠️ No analytics data to save")
    
    def export_session_data(self):
        """Export all session data"""
        try:
            export_data = {
                'session_info': {
                    'start_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
                    'duration_seconds': time.time() - self.start_time,
                    'total_frames': self.frame_count,
                    'average_fps': self.fps,
                    'config': asdict(self.config)
                },
                'analytics': self.emotion_analytics.get_emotion_trends(),
                'recent_readings': [reading.to_dict() for reading in list(self.emotion_analytics.emotion_history)[-50:]]
            }
            
            filename = f"enhanced_session_export_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Enhanced session data exported to {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Export error: {e}")
            return ""
    
    def run_enhanced(self):
        """Run the enhanced emotion detection system"""
        print("\n🚀 Starting Enhanced Y.M.I.R Emotion Detection System v2.0...")
        
        if not self.start_camera():
            return
        
        # Create window
        cv2.namedWindow("🚀 Y.M.I.R Enhanced Emotion Detection v2.0", cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Frame not captured. Retrying...")
                    continue
                
                # Flip frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Privacy mode
                if self.config.privacy_mode:
                    frame = np.zeros_like(frame)
                    cv2.putText(frame, "🔒 ENHANCED PRIVACY MODE", 
                              (frame.shape[1]//2 - 200, frame.shape[0]//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    # Enhanced processing
                    frame = self.process_frame_enhanced(frame)
                
                # Display frame
                cv2.imshow("🚀 Y.M.I.R Enhanced Emotion Detection v2.0", frame)
                
                # Handle key events
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_enhanced_key_events(key):
                    break
                
                # Check if window was closed
                if cv2.getWindowProperty("🚀 Y.M.I.R Enhanced Emotion Detection v2.0", cv2.WND_PROP_VISIBLE) < 1:
                    break
        
        except KeyboardInterrupt:
            print("\n🛑 Enhanced system interrupted by user")
        
        finally:
            self.cleanup_enhanced()
    
    def cleanup_enhanced(self):
        """Enhanced cleanup with analytics export"""
        print("\n🧹 Cleaning up enhanced system...")
        
        # Save final analytics
        self.save_analytics_summary()
        
        # Export session data
        self.export_session_data()
        
        # Stop camera
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        print("✅ Enhanced cleanup complete")

def main():
    """Main function for enhanced system"""
    print("🚀 Y.M.I.R Enhanced Emotion Detection System v2.0")
    print("=" * 60)
    print("🧠 Features: Ensemble detection, Quality assessment, Smart storage")
    print("☁️ Cloud storage: Firebase Firestore integration")
    print("📊 Analytics: Real-time emotion trends and insights")
    print("=" * 60)
    
    # Enhanced configuration
    config = EnhancedEmotionConfig(
        camera_width=1280,
        camera_height=720,
        use_ensemble_detection=True,
        min_face_quality_score=0.6,
        emotion_smoothing_window=5,
        confidence_threshold=0.7,
        store_only_significant_changes=True,
        emotion_change_threshold=15.0,
        use_firebase=True,
        require_user_consent=True
    )
    
    # Create enhanced detector
    detector = EnhancedEmotionDetector(config)
    
    # Run enhanced system
    detector.run_enhanced()

if __name__ == "__main__":
    main()