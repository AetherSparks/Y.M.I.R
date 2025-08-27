"""
🎯 Y.M.I.R Advanced Emotion Detection System
==========================================
Features:
- YOLO object detection for environmental context
- SQLite production-ready database storage  
- User consent & privacy controls
- Dynamic mesh control system
- Optimized DeepFace with timeout protection
- Background context analysis
- Multi-threaded performance optimization
- GPU acceleration support
"""

import sqlite3
import cv2
import numpy as np
import threading
import time
import json
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from enum import Enum

# Computer Vision imports
import mediapipe as mp
import dlib
from deepface import DeepFace
from scipy.spatial import distance as dist

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO available for object detection")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available - install ultralytics: pip install ultralytics")

warnings.filterwarnings("ignore")

@dataclass
class EmotionConfig:
    """Configuration class for emotion detection system"""
    # Camera settings
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    
    # Detection settings
    min_face_confidence: float = 0.6
    min_pose_confidence: float = 0.6
    emotion_analysis_interval: int = 5  # frames between analysis
    
    # Display settings
    show_face_mesh: bool = True
    show_body_pose: bool = True
    show_hand_tracking: bool = True
    show_gaze_tracking: bool = True
    show_yolo_objects: bool = True
    
    # Performance settings
    max_workers: int = 4
    deepface_timeout: float = 3.0
    
    # Privacy settings
    require_user_consent: bool = True
    privacy_mode: bool = False

class CameraState(Enum):
    """Camera state enumeration"""
    STOPPED = "stopped"
    STARTING = "starting" 
    RUNNING = "running"
    ERROR = "error"

class AdvancedEmotionDetector:
    """Advanced emotion detection system with YOLO, database storage, and privacy controls"""
    
    def __init__(self, config: EmotionConfig = None):
        self.config = config or EmotionConfig()
        self.camera_state = CameraState.STOPPED
        self.cap = None
        
        # Initialize computer vision models
        self._init_mediapipe()
        self._init_yolo()
        self._init_dlib()
        self._init_database()
        
        # Threading and synchronization
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.emotion_lock = threading.Lock()
        self.last_emotion_time = 0
        
        # Frame processing
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        # Detection data
        self.current_emotions = {}
        self.detected_objects = []
        self.face_bboxes = []
        self.last_known_faces = []
        self.no_face_counter = 0
        
        # Visual settings
        self.colors = {
            "face": (0, 255, 0),
            "body": (255, 255, 0), 
            "hands": (0, 0, 255),
            "eyes": (0, 255, 255),
            "objects": (255, 0, 255),
            "emotion_text": (0, 255, 255)
        }
        
        print("🎯 Advanced Emotion Detection System initialized")
        
    def _init_mediapipe(self):
        """Initialize MediaPipe models"""
        try:
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
            
            print("✅ MediaPipe models initialized")
        except Exception as e:
            print(f"❌ MediaPipe initialization error: {e}")
    
    def _init_yolo(self):
        """Initialize YOLO model for object detection"""
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                # Use YOLOv8n (nano) for speed, or YOLOv8s/m/l for accuracy
                self.yolo_model = YOLO('yolov8n.pt')  
                print("✅ YOLO model initialized")
            except Exception as e:
                print(f"⚠️ YOLO initialization error: {e}")
                
    def _init_dlib(self):
        """Initialize Dlib for advanced face analysis"""
        self.dlib_detector = None
        self.dlib_predictor = None
        
        dlib_path = Path("shape_predictor_68_face_landmarks.dat")
        if dlib_path.exists():
            try:
                self.dlib_detector = dlib.get_frontal_face_detector()
                self.dlib_predictor = dlib.shape_predictor(str(dlib_path))
                print("✅ Dlib initialized")
            except Exception as e:
                print(f"⚠️ Dlib initialization error: {e}")
        else:
            print("⚠️ Dlib landmarks file not found - advanced face analysis disabled")
    
    def _init_database(self):
        """Initialize SQLite database for production storage"""
        self.db_path = Path("emotion_data.db")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS emotions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        face_id INTEGER,
                        angry REAL,
                        disgust REAL,
                        fear REAL,
                        happy REAL,
                        sad REAL,
                        surprise REAL,
                        neutral REAL,
                        confidence REAL,
                        context_objects TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                        end_time DATETIME,
                        total_frames INTEGER,
                        avg_fps REAL
                    )
                """)
                
            print("✅ SQLite database initialized")
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def request_camera_permission(self) -> bool:
        """Request user permission to access camera"""
        if not self.config.require_user_consent:
            return True
            
        print("\n🔐 PRIVACY NOTICE:")
        print("This application requires camera access for emotion detection.")
        print("Your privacy is important - no data is shared externally.")
        
        while True:
            response = input("Grant camera access? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                print("✅ Camera access granted")
                return True
            elif response in ['n', 'no']:
                print("❌ Camera access denied")
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")
    
    def start_camera(self) -> bool:
        """Start camera with user consent"""
        if self.camera_state == CameraState.RUNNING:
            print("📷 Camera already running")
            return True
            
        if not self.request_camera_permission():
            print("🔒 Camera access required for emotion detection")
            return False
            
        try:
            self.camera_state = CameraState.STARTING
            print("📷 Starting camera...")
            
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.camera_state = CameraState.ERROR
                print("❌ Camera not detected!")
                return False
                
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height) 
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            
            self.camera_state = CameraState.RUNNING
            print(f"✅ Camera started ({self.config.camera_width}x{self.config.camera_height})")
            return True
            
        except Exception as e:
            self.camera_state = CameraState.ERROR
            print(f"❌ Camera initialization error: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera and release resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_state = CameraState.STOPPED
        print("🛑 Camera stopped")
    
    def detect_objects_yolo(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using YOLO for environmental context"""
        objects = []
        
        if self.yolo_model is None:
            return objects
            
        try:
            results = self.yolo_model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.yolo_model.names[class_id]
                        
                        objects.append({
                            "class": class_name,
                            "confidence": float(confidence),
                            "bbox": (int(x1), int(y1), int(x2), int(y2))
                        })
                        
        except Exception as e:
            print(f"⚠️ YOLO detection error: {e}")
            
        return objects
    
    def analyze_emotion_optimized(self, face_id: int, face_roi: np.ndarray, context_objects: List[str]):
        """Optimized emotion analysis with timeout protection and context"""
        current_time = time.time()
        
        # Rate limiting to prevent overload
        with self.emotion_lock:
            if current_time - self.last_emotion_time < 2.0:  # 2 second minimum interval
                return
            self.last_emotion_time = current_time
        
        try:
            # Validate face ROI
            if face_roi is None or face_roi.size == 0:
                return
                
            # Resize for DeepFace
            resized_face = cv2.resize(face_roi, (224, 224))
            
            # DeepFace analysis with timeout protection
            emotion_result = None
            analysis_error = None
            
            def deepface_worker():
                nonlocal emotion_result, analysis_error
                try:
                    emotion_result = DeepFace.analyze(
                        resized_face,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv',
                        silent=True
                    )
                except Exception as e:
                    analysis_error = e
            
            # Run with timeout
            analysis_thread = threading.Thread(target=deepface_worker, daemon=True)
            analysis_thread.start()
            analysis_thread.join(timeout=self.config.deepface_timeout)
            
            if analysis_thread.is_alive():
                print(f"⚠️ DeepFace timeout for face {face_id}")
                return
                
            if analysis_error:
                print(f"⚠️ DeepFace error: {analysis_error}")
                return
                
            if emotion_result is None:
                return
                
            # Process results
            emotions = emotion_result[0]['emotion']
            
            # Calculate confidence (dominance of top emotion)
            max_emotion = max(emotions.values())
            confidence = max_emotion / 100.0
            
            # Update current emotions
            with self.emotion_lock:
                self.current_emotions[face_id] = {
                    'emotions': emotions,
                    'confidence': confidence,
                    'timestamp': current_time
                }
            
            # Store in database
            self.save_emotion_to_db(face_id, emotions, confidence, context_objects)
            
            # Display results
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"🕒 {timestamp} - Face {face_id}: {emotions}")
            if context_objects:
                print(f"🎯 Context: {', '.join(context_objects)}")
                
        except Exception as e:
            print(f"❌ Emotion analysis error: {e}")
    
    def save_emotion_to_db(self, face_id: int, emotions: Dict[str, float], confidence: float, context_objects: List[str]):
        """Save emotion data to SQLite database"""
        try:
            with self.get_db_connection() as conn:
                conn.execute("""
                    INSERT INTO emotions (
                        face_id, angry, disgust, fear, happy, sad, surprise, neutral, 
                        confidence, context_objects
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    face_id,
                    emotions.get('angry', 0),
                    emotions.get('disgust', 0), 
                    emotions.get('fear', 0),
                    emotions.get('happy', 0),
                    emotions.get('sad', 0),
                    emotions.get('surprise', 0),
                    emotions.get('neutral', 0),
                    confidence,
                    json.dumps(context_objects)
                ))
                conn.commit()
        except Exception as e:
            print(f"⚠️ Database save error: {e}")
    
    def detect_faces(self, frame: np.ndarray, gray_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Advanced face detection using multiple methods"""
        face_bboxes = []
        
        # Primary: MediaPipe face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h) 
                w_box = int(bboxC.width * w)
                h_box = int(bboxC.height * h)
                face_bboxes.append((x, y, w_box, h_box))
        
        # Fallback: OpenCV Haar Cascades
        if not face_bboxes:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
            face_bboxes = [(x, y, w, h) for x, y, w, h in faces]
        
        # Face retention mechanism
        if face_bboxes:
            self.last_known_faces = face_bboxes
            self.no_face_counter = 0
        else:
            self.no_face_counter += 1
            if self.no_face_counter < 30:  # Retain faces for 30 frames
                face_bboxes = self.last_known_faces
        
        return face_bboxes
    
    def draw_visualizations(self, frame: np.ndarray, mesh_results, pose_results, hand_results):
        """Draw all visualization overlays"""
        # Face mesh
        if self.config.show_face_mesh and mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, self.colors["face"], -1)
        
        # Body pose
        if self.config.show_body_pose and pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, self.colors["body"], -1)
        
        # Hand tracking
        if self.config.show_hand_tracking and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, self.colors["hands"], -1)
        
        # YOLO objects
        if self.config.show_yolo_objects:
            for obj in self.detected_objects:
                x1, y1, x2, y2 = obj["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["objects"], 2)
                cv2.putText(frame, f"{obj['class']}: {obj['confidence']:.2f}", 
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["objects"], 2)
    
    def draw_emotions(self, frame: np.ndarray):
        """Draw emotion information on frame"""
        with self.emotion_lock:
            for face_id, data in self.current_emotions.items():
                if face_id < len(self.face_bboxes):
                    x, y, w, h = self.face_bboxes[face_id]
                    
                    # Draw face bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors["face"], 2)
                    
                    # Draw emotions text
                    emotions = data['emotions']
                    confidence = data['confidence']
                    
                    text_x = x + w + 10
                    text_y = y + 20
                    
                    # Show confidence
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                              (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["emotion_text"], 2)
                    text_y += 25
                    
                    # Show top 3 emotions
                    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    for emotion, score in sorted_emotions:
                        text = f"{emotion.upper()}: {score:.1f}%"
                        cv2.putText(frame, text, (text_x, text_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["emotion_text"], 2)
                        text_y += 25
    
    def draw_controls_info(self, frame: np.ndarray):
        """Draw control information"""
        info_text = [
            f"FPS: {self.fps:.1f}",
            f"Faces: {len(self.face_bboxes)}",
            f"Objects: {len(self.detected_objects)}",
            "Controls:",
            "Q - Quit",
            "F - Toggle Face Mesh", 
            "B - Toggle Body Pose",
            "H - Toggle Hand Tracking",
            "Y - Toggle YOLO Objects",
            "P - Privacy Mode"
        ]
        
        for i, text in enumerate(info_text):
            y_pos = 30 + i * 25
            cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame with all detection and analysis"""
        self.frame_count += 1
        
        # Convert color spaces
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        self.face_bboxes = self.detect_faces(frame, gray_frame)
        
        # Detect objects for context
        if self.frame_count % 5 == 0:  # Every 5 frames
            self.detected_objects = self.detect_objects_yolo(frame)
        
        # Get MediaPipe results
        mesh_results = self.face_mesh.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        
        # Analyze emotions for detected faces
        if self.frame_count % self.config.emotion_analysis_interval == 0:
            context_objects = [obj["class"] for obj in self.detected_objects]
            
            for i, (x, y, w, h) in enumerate(self.face_bboxes):
                if i >= 3:  # Limit to 3 faces for performance
                    break
                    
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    self.executor.submit(self.analyze_emotion_optimized, i, face_roi, context_objects)
        
        # Draw visualizations
        self.draw_visualizations(frame, mesh_results, pose_results, hand_results)
        self.draw_emotions(frame)
        self.draw_controls_info(frame)
        
        # Calculate FPS
        elapsed_time = time.time() - self.start_time
        self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return frame
    
    def handle_key_events(self, key: int) -> bool:
        """Handle keyboard events for dynamic control"""
        if key == ord('q'):
            return False  # Quit
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
        elif key == ord('p'):
            self.config.privacy_mode = not self.config.privacy_mode
            print(f"Privacy mode: {'ON' if self.config.privacy_mode else 'OFF'}")
        
        return True
    
    def run(self):
        """Main execution loop"""
        print("\n🎯 Starting Advanced Emotion Detection System...")
        
        if not self.start_camera():
            return
        
        # Create window
        cv2.namedWindow("🎯 Y.M.I.R Advanced Emotion Detection", cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Frame not captured. Retrying...")
                    continue
                
                # Flip frame horizontally for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Privacy mode
                if self.config.privacy_mode:
                    frame = np.zeros_like(frame)
                    cv2.putText(frame, "PRIVACY MODE", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    # Process frame
                    frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow("🎯 Y.M.I.R Advanced Emotion Detection", frame)
                
                # Handle key events
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key_events(key):
                    break
                    
                # Check if window was closed
                if cv2.getWindowProperty("🎯 Y.M.I.R Advanced Emotion Detection", cv2.WND_PROP_VISIBLE) < 1:
                    break
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up resources...")
        self.stop_camera()
        cv2.destroyAllWindows()
        self.executor.shutdown(wait=True)
        print("✅ Cleanup complete")
    
    def export_emotion_data(self, format: str = "json") -> str:
        """Export emotion data from database"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.execute("SELECT * FROM emotions ORDER BY timestamp DESC LIMIT 1000")
                data = cursor.fetchall()
                
            if format.lower() == "json":
                export_data = []
                for row in data:
                    export_data.append({
                        "id": row[0],
                        "timestamp": row[1], 
                        "face_id": row[2],
                        "emotions": {
                            "angry": row[3],
                            "disgust": row[4],
                            "fear": row[5],
                            "happy": row[6],
                            "sad": row[7],
                            "surprise": row[8],
                            "neutral": row[9]
                        },
                        "confidence": row[10],
                        "context_objects": json.loads(row[11]) if row[11] else []
                    })
                
                filename = f"emotion_export_{int(time.time())}.json"
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                print(f"✅ Data exported to {filename}")
                return filename
                
        except Exception as e:
            print(f"❌ Export error: {e}")
            return ""

def main():
    """Main function"""
    print("🎯 Y.M.I.R Advanced Emotion Detection System")
    print("=" * 50)
    
    # Create configuration
    config = EmotionConfig(
        camera_width=1280,
        camera_height=720,
        require_user_consent=True,
        show_face_mesh=True,
        show_body_pose=True,
        show_hand_tracking=True,
        show_yolo_objects=True
    )
    
    # Create detector
    detector = AdvancedEmotionDetector(config)
    
    # Run detection
    detector.run()
    
    # Export data
    print("\nExporting session data...")
    detector.export_emotion_data("json")

if __name__ == "__main__":
    main()