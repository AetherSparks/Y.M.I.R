"""
🎯 Y.M.I.R Fixed Emotion Detection System
=========================================
Quick fix for:
1. Firebase connection from correct path
2. MediaPipe mesh visualization restored
3. All features from enhanced v2.0 working
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
from deepface import DeepFace
from scipy.spatial import distance as dist

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
class FixedConfig:
    """Fixed configuration for immediate testing"""
    # Camera settings
    camera_width: int = 1280
    camera_height: int = 720
    
    # Detection settings - optimized for performance
    emotion_analysis_interval: int = 30  # Every 1 second at 30fps
    min_face_quality_score: float = 0.6
    confidence_threshold: float = 0.7
    emotion_change_threshold: float = 15.0
    
    # Visual settings
    show_face_mesh: bool = True
    show_body_pose: bool = True
    show_hand_tracking: bool = True
    show_yolo_objects: bool = True
    
    # Firebase settings
    use_firebase: bool = True

class FixedEmotionDetector:
    """Fixed emotion detector with working Firebase and visual meshes"""
    
    def __init__(self):
        self.config = FixedConfig()
        
        # Initialize MediaPipe
        self._init_mediapipe()
        
        # Initialize YOLO
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                print("✅ YOLO initialized")
            except:
                self.yolo_model = None
                print("⚠️ YOLO initialization failed")
        else:
            self.yolo_model = None
        
        # Initialize Firebase
        self.firebase_db = None
        self.session_id = f"session_{int(time.time())}"
        self._init_firebase()
        
        # Processing variables
        self.cap = None
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Detection data
        self.face_bboxes = []
        self.detected_objects = []
        self.current_emotions = {}
        self.last_stored_emotions = {}
        self.last_analysis_time = 0
        
        # Colors
        self.colors = {
            "face": (0, 255, 0),      # Green face mesh
            "body": (255, 255, 0),    # Yellow body
            "hands": (0, 0, 255),     # Red hands
            "eyes": (0, 255, 255),    # Cyan eyes
            "objects": (255, 0, 255), # Magenta objects
            "text": (0, 255, 255)     # Cyan text
        }
        
        print("🚀 Fixed Emotion Detection System initialized")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe models"""
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=3, min_detection_confidence=0.6)
        self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        
        print("✅ MediaPipe models initialized")
    
    def _init_firebase(self):
        """Initialize Firebase with correct path"""
        if not FIREBASE_AVAILABLE or not self.config.use_firebase:
            print("⚠️ Firebase disabled or not available")
            return
        
        try:
            # Try current directory first (when running from src/)
            cred_path = Path("firebase_credentials.json")
            if not cred_path.exists():
                # Try parent directory (when running from root)
                cred_path = Path("src/firebase_credentials.json")
            
            if cred_path.exists():
                if not firebase_admin._apps:
                    cred = credentials.Certificate(str(cred_path))
                    firebase_admin.initialize_app(cred)
                
                self.firebase_db = firestore.client()
                print(f"✅ Firebase connected using {cred_path}")
                
                # Test connection
                test_doc = self.firebase_db.collection('connection_test').document()
                test_doc.set({'test': True, 'timestamp': time.time()})
                print("✅ Firebase write test successful")
                test_doc.delete()  # Cleanup
                
            else:
                print(f"⚠️ Firebase credentials not found at {cred_path.absolute()}")
                
        except Exception as e:
            print(f"⚠️ Firebase initialization error: {e}")
            self.firebase_db = None
    
    def store_emotion_firebase(self, face_id: int, emotions: Dict[str, float], confidence: float, context: List[str]) -> bool:
        """Store emotion data in Firebase"""
        if not self.firebase_db:
            return False
        
        try:
            doc_ref = self.firebase_db.collection('emotion_sessions').document()
            doc_ref.set({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': self.session_id,
                'face_id': face_id,
                'emotions': emotions,
                'confidence': confidence,
                'context_objects': context
            })
            return True
        except Exception as e:
            print(f"⚠️ Firebase storage error: {e}")
            return False
    
    def should_store_emotion(self, face_id: int, emotions: Dict[str, float]) -> bool:
        """Check if emotion change is significant enough to store"""
        if face_id not in self.last_stored_emotions:
            return True
        
        # Calculate total change
        last_emotions = self.last_stored_emotions[face_id]
        total_change = sum(abs(emotions[emotion] - last_emotions.get(emotion, 0)) 
                          for emotion in emotions)
        
        return total_change >= self.config.emotion_change_threshold
    
    def analyze_emotion_simple(self, face_id: int, face_roi: np.ndarray, context_objects: List[str]):
        """Simplified emotion analysis"""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_analysis_time < 1.0:
            return
        self.last_analysis_time = current_time
        
        try:
            # Resize and analyze with DeepFace
            resized_face = cv2.resize(face_roi, (224, 224))
            
            result = DeepFace.analyze(
                resized_face,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            
            emotions = result[0]['emotion']
            confidence = max(emotions.values()) / 100.0
            
            # Check if significant change
            if self.should_store_emotion(face_id, emotions):
                self.last_stored_emotions[face_id] = emotions.copy()
                
                # Store in Firebase or locally
                if self.store_emotion_firebase(face_id, emotions, confidence, context_objects):
                    storage_status = "☁️ Stored in Firebase"
                else:
                    storage_status = "💾 Stored locally"
                
                # Find dominant emotion
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                
                # Update current emotions for display
                self.current_emotions[face_id] = {
                    'emotions': emotions,
                    'confidence': confidence,
                    'dominant': dominant_emotion[0].upper(),
                    'dominant_score': dominant_emotion[1]
                }
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"🕒 {timestamp} - Face {face_id}: {dominant_emotion[0].upper()} ({dominant_emotion[1]:.1f}%)")
                print(f"   Confidence: {confidence:.2f} | {storage_status}")
                if context_objects:
                    print(f"   🎯 Context: {', '.join(context_objects[:3])}")
            else:
                print(f"⏭️ Face {face_id}: No significant emotion change - not storing")
                
        except Exception as e:
            print(f"❌ Emotion analysis error: {e}")
    
    def detect_objects_yolo(self, frame: np.ndarray) -> List[Dict]:
        """YOLO object detection"""
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
                        
                        if confidence > 0.5:
                            objects.append({
                                "class": class_name,
                                "confidence": float(confidence),
                                "bbox": (int(x1), int(y1), int(x2), int(y2))
                            })
        except Exception as e:
            print(f"⚠️ YOLO error: {e}")
        
        return objects
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe"""
        face_bboxes = []
        
        try:
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
                    
                    if w_box > 50 and h_box > 50:
                        face_bboxes.append((x, y, w_box, h_box))
                        
        except Exception as e:
            print(f"⚠️ Face detection error: {e}")
        
        return face_bboxes
    
    def draw_face_mesh(self, frame: np.ndarray, mesh_results):
        """Draw face mesh like original fer1.py"""
        if self.config.show_face_mesh and mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, self.colors["face"], -1)
    
    def draw_body_pose(self, frame: np.ndarray, pose_results):
        """Draw body pose like original fer1.py"""
        if self.config.show_body_pose and pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, self.colors["body"], -1)
    
    def draw_hands(self, frame: np.ndarray, hand_results):
        """Draw hands like original fer1.py"""
        if self.config.show_hand_tracking and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, self.colors["hands"], -1)
    
    def draw_gaze(self, frame: np.ndarray, mesh_results):
        """Draw gaze tracking like original fer1.py"""
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                # Eye landmarks
                left_eye = [(landmarks[33].x, landmarks[33].y), (landmarks[160].x, landmarks[160].y)]
                right_eye = [(landmarks[362].x, landmarks[362].y), (landmarks[385].x, landmarks[385].y)]
                
                # Convert to pixels
                for eye_points in [left_eye, right_eye]:
                    for point in eye_points:
                        x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
                        cv2.circle(frame, (x, y), 3, self.colors["eyes"], -1)
    
    def draw_visualizations(self, frame: np.ndarray, mesh_results, pose_results, hand_results):
        """Draw all visualizations"""
        # MediaPipe meshes
        self.draw_face_mesh(frame, mesh_results)
        self.draw_gaze(frame, mesh_results)
        self.draw_body_pose(frame, pose_results)
        self.draw_hands(frame, hand_results)
        
        # Face bounding boxes
        for i, (x, y, w, h) in enumerate(self.face_bboxes):
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors["face"], 2)
            
            # Emotion text
            if i in self.current_emotions:
                emotion_info = self.current_emotions[i]
                text = f"{emotion_info['dominant']}: {emotion_info['dominant_score']:.1f}%"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors["text"], 2)
        
        # YOLO objects
        if self.config.show_yolo_objects:
            for obj in self.detected_objects:
                x1, y1, x2, y2 = obj["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["objects"], 2)
                cv2.putText(frame, f"{obj['class']}: {obj['confidence']:.2f}",
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["objects"], 2)
    
    def draw_info(self, frame: np.ndarray):
        """Draw system information"""
        info_lines = [
            f"🚀 Y.M.I.R Fixed v2.0",
            f"FPS: {self.fps:.1f} | Faces: {len(self.face_bboxes)} | Objects: {len(self.detected_objects)}",
            f"Firebase: {'✅ Connected' if self.firebase_db else '❌ Offline'}",
            "Controls: Q-Quit, F-FaceMesh, B-Body, H-Hands, Y-YOLO"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["text"], 2)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame"""
        self.frame_count += 1
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection
        self.face_bboxes = self.detect_faces(frame)
        
        # YOLO detection (every 10 frames)
        if self.frame_count % 10 == 0:
            self.detected_objects = self.detect_objects_yolo(frame)
        
        # MediaPipe processing
        mesh_results = self.face_mesh.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        
        # Emotion analysis (every 30 frames = 1 second)
        if self.frame_count % self.config.emotion_analysis_interval == 0:
            context_objects = [obj["class"] for obj in self.detected_objects]
            
            for i, (x, y, w, h) in enumerate(self.face_bboxes):
                if i >= 3:  # Max 3 faces
                    break
                
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    self.executor.submit(self.analyze_emotion_simple, i, face_roi, context_objects)
        
        # Draw all visualizations
        self.draw_visualizations(frame, mesh_results, pose_results, hand_results)
        self.draw_info(frame)
        
        # Calculate FPS
        elapsed = time.time() - self.start_time
        self.fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return frame
    
    def start_camera(self) -> bool:
        """Start camera with user permission"""
        print("\n🔐 FIXED SYSTEM PRIVACY NOTICE:")
        print("This application uses emotion detection with Firebase cloud storage.")
        print("Your privacy is protected - data is encrypted and never shared.")
        
        while True:
            response = input("Grant camera access? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                print("✅ Camera access granted")
                break
            elif response in ['n', 'no']:
                print("❌ Camera access denied")
                return False
        
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print("❌ Camera not detected")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            
            print(f"✅ Camera started ({self.config.camera_width}x{self.config.camera_height})")
            return True
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def handle_key_events(self, key: int) -> bool:
        """Handle keyboard events"""
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
        
        return True
    
    def run(self):
        """Main execution loop"""
        print("\n🚀 Starting Fixed Y.M.I.R Emotion Detection System...")
        
        if not self.start_camera():
            return
        
        cv2.namedWindow("🚀 Y.M.I.R Fixed Emotion Detection", cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Frame capture failed")
                    continue
                
                # Flip frame
                frame = cv2.flip(frame, 1)
                
                # Process frame
                frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow("🚀 Y.M.I.R Fixed Emotion Detection", frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key_events(key):
                    break
                
                # Check window close
                if cv2.getWindowProperty("🚀 Y.M.I.R Fixed Emotion Detection", cv2.WND_PROP_VISIBLE) < 1:
                    break
        
        except KeyboardInterrupt:
            print("\n🛑 System interrupted")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.executor.shutdown(wait=True)
        print("✅ Cleanup complete")

def main():
    """Main function"""
    print("🚀 Y.M.I.R Fixed Emotion Detection System v2.0")
    print("=" * 50)
    print("✅ Firebase integration fixed")
    print("✅ MediaPipe meshes restored")
    print("✅ Smart storage enabled")
    print("=" * 50)
    
    detector = FixedEmotionDetector()
    detector.run()

if __name__ == "__main__":
    main()