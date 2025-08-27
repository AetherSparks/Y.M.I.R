"""
🎯 Y.M.I.R Emotion Detection - No Firebase Version
==================================================
This version works WITHOUT Firebase to avoid protobuf conflicts.
Still includes all enhanced features:
- MediaPipe meshes (face, body, hands, gaze)
- YOLO object detection
- Smart emotion storage
- Quality assessment
- Local JSON export
"""

import cv2
import numpy as np
import threading
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict
from datetime import datetime

# Computer Vision imports
import mediapipe as mp
from deepface import DeepFace

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO available for object detection")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available - install: pip install ultralytics")

warnings.filterwarnings("ignore")

class NoFirebaseEmotionDetector:
    """Emotion detector without Firebase - no protobuf conflicts"""
    
    def __init__(self):
        # Configuration
        self.config = {
            'camera_width': 1280,
            'camera_height': 720,
            'emotion_analysis_interval': 30,  # Every 1 second at 30fps
            'emotion_change_threshold': 15.0,  # Store only significant changes
            'confidence_threshold': 0.7,
            'show_face_mesh': True,
            'show_body_pose': True,
            'show_hand_tracking': True,
            'show_yolo_objects': True,
            'show_gaze': True
        }
        
        # Initialize MediaPipe
        self._init_mediapipe()
        
        # Initialize YOLO
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                print("✅ YOLO model loaded")
            except Exception as e:
                print(f"⚠️ YOLO failed to load: {e}")
        
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
        self.emotion_history = []
        self.last_analysis_time = 0
        self.session_id = f"session_{int(time.time())}"
        
        # Colors for visualization
        self.colors = {
            "face": (0, 255, 0),      # Green
            "body": (255, 255, 0),    # Yellow
            "hands": (0, 0, 255),     # Red
            "eyes": (0, 255, 255),    # Cyan
            "objects": (255, 0, 255), # Magenta
            "text": (255, 255, 255),  # White
            "quality_good": (0, 255, 0),  # Green
            "quality_fair": (0, 255, 255), # Yellow
            "quality_poor": (0, 0, 255)    # Red
        }
        
        print("🚀 No-Firebase Emotion Detection System initialized")
    
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
    
    def assess_face_quality(self, face_roi: np.ndarray) -> float:
        """Assess face quality for better emotion analysis"""
        try:
            if len(face_roi.shape) == 3:
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_roi
            
            height, width = gray_face.shape
            
            # Size score
            size_score = min(1.0, (height * width) / (100 * 100))
            
            # Sharpness score (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 500)
            
            # Brightness score
            mean_brightness = np.mean(gray_face)
            brightness_score = 1.0 - abs(mean_brightness - 127) / 127
            
            # Combined quality score
            quality_score = (size_score * 0.4 + sharpness_score * 0.4 + brightness_score * 0.2)
            
            return quality_score
            
        except Exception as e:
            print(f"⚠️ Quality assessment error: {e}")
            return 0.5
    
    def should_store_emotion(self, face_id: int, emotions: Dict[str, float]) -> bool:
        """Check if emotion change is significant enough to store"""
        if face_id not in self.last_stored_emotions:
            return True
        
        last_emotions = self.last_stored_emotions[face_id]
        total_change = sum(abs(emotions[emotion] - last_emotions.get(emotion, 0)) 
                          for emotion in emotions)
        
        return total_change >= self.config['emotion_change_threshold']
    
    def store_emotion_local(self, face_id: int, emotions: Dict[str, float], confidence: float, 
                           quality: float, context: List[str]) -> bool:
        """Store emotion data locally with smart filtering"""
        try:
            emotion_entry = {
                'timestamp': datetime.now().isoformat(),
                'session_id': self.session_id,
                'face_id': face_id,
                'emotions': emotions,
                'confidence': confidence,
                'quality_score': quality,
                'context_objects': context
            }
            
            self.emotion_history.append(emotion_entry)
            
            # Keep only recent entries (memory management)
            if len(self.emotion_history) > 1000:
                self.emotion_history = self.emotion_history[-500:]  # Keep last 500
            
            return True
            
        except Exception as e:
            print(f"⚠️ Local storage error: {e}")
            return False
    
    def analyze_emotion_enhanced(self, face_id: int, face_roi: np.ndarray, context_objects: List[str]):
        """Enhanced emotion analysis with quality assessment"""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_analysis_time < 1.0:
            return
        self.last_analysis_time = current_time
        
        try:
            # Assess face quality first
            quality_score = self.assess_face_quality(face_roi)
            
            if quality_score < 0.5:  # Skip very low quality faces
                print(f"⚠️ Face {face_id} quality too low ({quality_score:.2f}) - skipping")
                return
            
            # Resize for DeepFace
            resized_face = cv2.resize(face_roi, (224, 224))
            
            # DeepFace analysis with timeout protection
            def analyze_with_timeout():
                return DeepFace.analyze(
                    resized_face,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv',
                    silent=True
                )
            
            # Run analysis with timeout
            analysis_thread = threading.Thread(target=analyze_with_timeout)
            analysis_thread.daemon = True
            analysis_thread.start()
            analysis_thread.join(timeout=3.0)  # 3 second timeout
            
            if analysis_thread.is_alive():
                print(f"⚠️ DeepFace timeout for face {face_id}")
                return
            
            # Get results
            result = analyze_with_timeout()
            emotions = result[0]['emotion']
            confidence = max(emotions.values()) / 100.0
            
            # Filter by confidence
            if confidence < self.config['confidence_threshold']:
                print(f"⚠️ Low confidence ({confidence:.2f}) - skipping face {face_id}")
                return
            
            # Check if significant change
            if self.should_store_emotion(face_id, emotions):
                self.last_stored_emotions[face_id] = emotions.copy()
                
                # Store locally
                if self.store_emotion_local(face_id, emotions, confidence, quality_score, context_objects):
                    storage_status = "💾 Stored locally"
                else:
                    storage_status = "❌ Storage failed"
                
                # Update current emotions for display
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                self.current_emotions[face_id] = {
                    'emotions': emotions,
                    'confidence': confidence,
                    'quality': quality_score,
                    'dominant': dominant_emotion[0].upper(),
                    'dominant_score': dominant_emotion[1]
                }
                
                # Display results
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                quality_level = "excellent" if quality_score >= 0.8 else "good" if quality_score >= 0.6 else "fair"
                
                print(f"✅ Analyzing face {face_id} - Quality: {quality_level} ({quality_score:.2f})")
                print(f"🕒 {timestamp} - Face {face_id}: {dominant_emotion[0].upper()} ({dominant_emotion[1]:.1f}%)")
                print(f"   Quality: {quality_level} | Confidence: {confidence:.2f} | {storage_status}")
                if context_objects:
                    print(f"   🎯 Context: {', '.join(context_objects[:3])}")
            else:
                print(f"⏭️ Face {face_id}: No significant emotion change - not storing")
                
        except Exception as e:
            print(f"❌ Emotion analysis error: {e}")
    
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
    
    def detect_objects_yolo(self, frame: np.ndarray) -> List[Dict]:
        """YOLO object detection for context"""
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
                        
                        if confidence > 0.5 and class_name in ['person', 'laptop', 'cell phone', 'book', 'tv', 'chair']:
                            objects.append({
                                "class": class_name,
                                "confidence": float(confidence),
                                "bbox": (int(x1), int(y1), int(x2), int(y2))
                            })
        except Exception as e:
            print(f"⚠️ YOLO error: {e}")
        
        return objects
    
    def draw_face_mesh(self, frame: np.ndarray, mesh_results):
        """Draw MediaPipe face mesh"""
        if self.config['show_face_mesh'] and mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, self.colors["face"], -1)
    
    def draw_body_pose(self, frame: np.ndarray, pose_results):
        """Draw MediaPipe body pose"""
        if self.config['show_body_pose'] and pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, self.colors["body"], -1)
    
    def draw_hands(self, frame: np.ndarray, hand_results):
        """Draw MediaPipe hand tracking"""
        if self.config['show_hand_tracking'] and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, self.colors["hands"], -1)
    
    def draw_gaze(self, frame: np.ndarray, mesh_results):
        """Draw gaze tracking"""
        if self.config['show_gaze'] and mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                # Simplified eye landmarks
                eye_points = [
                    (landmarks[33].x, landmarks[33].y),   # Left eye
                    (landmarks[133].x, landmarks[133].y), # Left eye
                    (landmarks[362].x, landmarks[362].y), # Right eye
                    (landmarks[263].x, landmarks[263].y)  # Right eye
                ]
                
                for point in eye_points:
                    x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, self.colors["eyes"], -1)
    
    def draw_visualizations(self, frame: np.ndarray, mesh_results, pose_results, hand_results):
        """Draw all visual elements"""
        # MediaPipe visualizations
        self.draw_face_mesh(frame, mesh_results)
        self.draw_gaze(frame, mesh_results)
        self.draw_body_pose(frame, pose_results)
        self.draw_hands(frame, hand_results)
        
        # Face bounding boxes with quality indicators
        for i, (x, y, w, h) in enumerate(self.face_bboxes):
            # Determine quality color
            if i in self.current_emotions:
                quality = self.current_emotions[i]['quality']
                if quality >= 0.8:
                    color = self.colors["quality_good"]
                    thickness = 3
                elif quality >= 0.6:
                    color = self.colors["quality_fair"]
                    thickness = 2
                else:
                    color = self.colors["quality_poor"]
                    thickness = 1
            else:
                color = self.colors["face"]
                thickness = 2
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Emotion text
            if i in self.current_emotions:
                emotion_info = self.current_emotions[i]
                text = f"{emotion_info['dominant']}: {emotion_info['dominant_score']:.1f}%"
                cv2.putText(frame, text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors["text"], 2)
                
                # Quality score
                cv2.putText(frame, f"Q: {emotion_info['quality']:.2f}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # YOLO objects
        if self.config['show_yolo_objects']:
            for obj in self.detected_objects:
                x1, y1, x2, y2 = obj["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["objects"], 2)
                cv2.putText(frame, f"{obj['class']}: {obj['confidence']:.2f}",
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["objects"], 2)
    
    def draw_info(self, frame: np.ndarray):
        """Draw system information"""
        info_lines = [
            f"🚀 Y.M.I.R No-Firebase v2.0 - NO PROTOBUF CONFLICTS!",
            f"FPS: {self.fps:.1f} | Faces: {len(self.face_bboxes)} | Objects: {len(self.detected_objects)}",
            f"Stored entries: {len(self.emotion_history)} | Session: {self.session_id}",
            "Controls: Q-Quit | F-FaceMesh | B-Body | H-Hands | Y-YOLO | S-Save"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text"], 1)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame with all enhancements"""
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
        
        # Emotion analysis (configurable interval)
        if self.frame_count % self.config['emotion_analysis_interval'] == 0:
            context_objects = [obj["class"] for obj in self.detected_objects]
            
            for i, (x, y, w, h) in enumerate(self.face_bboxes):
                if i >= 3:  # Max 3 faces
                    break
                
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    self.executor.submit(self.analyze_emotion_enhanced, i, face_roi, context_objects)
        
        # Draw all visualizations
        self.draw_visualizations(frame, mesh_results, pose_results, hand_results)
        self.draw_info(frame)
        
        # Calculate FPS
        elapsed = time.time() - self.start_time
        self.fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return frame
    
    def export_data(self):
        """Export emotion data to JSON file"""
        try:
            filename = f"emotion_session_{self.session_id}.json"
            
            export_data = {
                'session_info': {
                    'session_id': self.session_id,
                    'start_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
                    'duration_seconds': time.time() - self.start_time,
                    'total_frames': self.frame_count,
                    'average_fps': self.fps,
                    'total_emotions_stored': len(self.emotion_history)
                },
                'emotion_data': self.emotion_history,
                'statistics': self._calculate_statistics()
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"✅ Data exported to {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Export error: {e}")
            return ""
    
    def _calculate_statistics(self) -> Dict:
        """Calculate session statistics"""
        if not self.emotion_history:
            return {}
        
        # Calculate dominant emotions
        emotion_counts = defaultdict(int)
        confidence_scores = []
        quality_scores = []
        
        for entry in self.emotion_history:
            emotions = entry['emotions']
            dominant = max(emotions.items(), key=lambda x: x[1])[0]
            emotion_counts[dominant] += 1
            confidence_scores.append(entry['confidence'])
            quality_scores.append(entry['quality_score'])
        
        return {
            'most_common_emotion': max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'none',
            'emotion_distribution': dict(emotion_counts),
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            'average_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0
        }
    
    def start_camera(self) -> bool:
        """Start camera with user permission"""
        print("\n🔐 NO-FIREBASE PRIVACY NOTICE:")
        print("This application uses emotion detection with local storage only.")
        print("No data is sent to the cloud - everything stays on your device.")
        print("Features: Enhanced accuracy, MediaPipe meshes, YOLO objects")
        
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
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera_width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera_height'])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            print(f"✅ Camera started ({self.config['camera_width']}x{self.config['camera_height']})")
            return True
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def handle_key_events(self, key: int) -> bool:
        """Handle keyboard events"""
        if key == ord('q'):
            return False
        elif key == ord('f'):
            self.config['show_face_mesh'] = not self.config['show_face_mesh']
            print(f"Face mesh: {'ON' if self.config['show_face_mesh'] else 'OFF'}")
        elif key == ord('b'):
            self.config['show_body_pose'] = not self.config['show_body_pose']
            print(f"Body pose: {'ON' if self.config['show_body_pose'] else 'OFF'}")
        elif key == ord('h'):
            self.config['show_hand_tracking'] = not self.config['show_hand_tracking']
            print(f"Hand tracking: {'ON' if self.config['show_hand_tracking'] else 'OFF'}")
        elif key == ord('y'):
            self.config['show_yolo_objects'] = not self.config['show_yolo_objects']
            print(f"YOLO objects: {'ON' if self.config['show_yolo_objects'] else 'OFF'}")
        elif key == ord('s'):
            self.export_data()
        
        return True
    
    def run(self):
        """Main execution loop"""
        print("\n🚀 Starting No-Firebase Y.M.I.R Emotion Detection System...")
        
        if not self.start_camera():
            return
        
        cv2.namedWindow("🚀 Y.M.I.R No-Firebase Emotion Detection", cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Frame capture failed")
                    continue
                
                # Flip frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Process frame
                frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow("🚀 Y.M.I.R No-Firebase Emotion Detection", frame)
                
                # Handle keyboard events
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key_events(key):
                    break
                
                # Check if window was closed
                if cv2.getWindowProperty("🚀 Y.M.I.R No-Firebase Emotion Detection", cv2.WND_PROP_VISIBLE) < 1:
                    break
        
        except KeyboardInterrupt:
            print("\n🛑 System interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources and export final data"""
        print("\n🧹 Cleaning up...")
        
        # Export final session data
        self.export_data()
        
        # Release resources
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.executor.shutdown(wait=True)
        
        print("✅ Cleanup complete")
        print(f"Final session stats: {len(self.emotion_history)} emotions stored")

def main():
    """Main function"""
    print("🚀 Y.M.I.R No-Firebase Emotion Detection System")
    print("=" * 60)
    print("✅ NO PROTOBUF CONFLICTS - Firebase disabled")
    print("✅ All MediaPipe meshes enabled")
    print("✅ YOLO object detection")
    print("✅ Enhanced emotion accuracy")
    print("✅ Smart local storage")
    print("=" * 60)
    
    detector = NoFirebaseEmotionDetector()
    detector.run()

if __name__ == "__main__":
    main()