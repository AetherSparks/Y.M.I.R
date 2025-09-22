"""
🔬 MediaPipe Model Component for Y.M.I.R
========================================
Handles face mesh, pose estimation, hand tracking, and face detection using MediaPipe.
Provides detailed anatomical feature extraction for emotion analysis.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class FaceQuality(Enum):
    """Face quality assessment levels"""
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"

@dataclass
class MediaPipeConfig:
    """Configuration for MediaPipe components"""
    min_face_confidence: float = 0.7
    min_pose_confidence: float = 0.6
    max_num_faces: int = 3
    min_face_size: int = 50
    show_face_mesh: bool = True
    show_body_pose: bool = True
    show_hand_tracking: bool = True
    show_gaze_tracking: bool = True

class MediaPipeProcessor:
    """Enhanced MediaPipe processing for face, pose, and hand detection"""
    
    def __init__(self, config: Optional[MediaPipeConfig] = None):
        self.config = config or MediaPipeConfig()
        self._init_mediapipe_models()
        
        # Timestamp management for MediaPipe
        self.last_timestamp_ms = 0
        self.frame_count = 0
        
        # Visual settings
        self.colors = {
            "face": (0, 255, 0),
            "body": (255, 255, 0),
            "hands": (0, 0, 255),
            "eyes": (0, 255, 255),
            "quality_good": (0, 255, 0),
            "quality_fair": (0, 255, 255),
            "quality_poor": (0, 0, 255)
        }
        
        print("✅ MediaPipe processor initialized")
    
    def _get_next_timestamp_ms(self) -> int:
        """Generate monotonically increasing timestamps for MediaPipe"""
        self.frame_count += 1
        # Generate timestamp that always increases
        current_ms = int(time.time() * 1000)
        if current_ms <= self.last_timestamp_ms:
            # Ensure monotonic increase
            current_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = current_ms
        return current_ms
    
    def _init_mediapipe_models(self):
        """Initialize all MediaPipe models"""
        try:
            # MediaPipe solutions
            mp_face_detection = mp.solutions.face_detection
            mp_face_mesh = mp.solutions.face_mesh
            mp_pose = mp.solutions.pose
            mp_hands = mp.solutions.hands
            
            # Initialize models
            self.face_detection = mp_face_detection.FaceDetection(
                min_detection_confidence=self.config.min_face_confidence
            )
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=self.config.max_num_faces,
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
            
            print("✅ MediaPipe models loaded successfully")
            
        except Exception as e:
            print(f"❌ MediaPipe initialization error: {e}")
            raise
    
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
    
    def detect_faces_enhanced(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Enhanced face detection with quality assessment"""
        faces = []
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if results.detections:
                for i, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    
                    # Calculate absolute coordinates
                    x = max(0, int(bboxC.xmin * w))
                    y = max(0, int(bboxC.ymin * h))
                    w_box = min(w - x, int(bboxC.width * w))
                    h_box = min(h - y, int(bboxC.height * h))
                    
                    # Apply minimum size filter
                    if w_box > self.config.min_face_size and h_box > self.config.min_face_size:
                        face_bbox = (x, y, w_box, h_box)
                        face_roi = frame[y:y+h_box, x:x+w_box]
                        
                        # Assess face quality
                        quality_score, quality_level = self.assess_face_quality(face_roi)
                        
                        faces.append({
                            'id': i,
                            'bbox': face_bbox,
                            'quality_score': quality_score,
                            'quality_level': quality_level,
                            'roi': face_roi,
                            'confidence': detection.score[0] if detection.score else 0.0,
                            'center': (x + w_box // 2, y + h_box // 2)
                        })
            
        except Exception as e:
            print(f"⚠️ Enhanced face detection error: {e}")
        
        return faces
    
    def process_frame(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> Dict[str, Any]:
        """Process frame with MediaPipe models (timestamp-free for reliability)"""
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process directly with numpy array (no timestamp management)
            # This eliminates timestamp synchronization errors completely
            face_results = self.face_mesh.process(rgb_frame)
            pose_results = self.pose.process(rgb_frame)
            hand_results = self.hands.process(rgb_frame)
            
            # Detect faces with quality assessment
            faces = self.detect_faces_enhanced(frame)
            
            return {
                'faces': faces,
                'face_mesh': face_results,
                'pose': pose_results,
                'hands': hand_results,
                'frame_processed': True
            }
            
        except Exception as e:
            print(f"❌ MediaPipe frame processing error: {e}")
            return {
                'faces': [],
                'face_mesh': None,
                'pose': None,
                'hands': None,
                'frame_processed': False
            }
    
    def extract_facial_features(self, mesh_results) -> Dict[str, Any]:
        """Extract detailed facial features for emotion analysis"""
        features = {
            'landmarks': [],
            'eye_landmarks': [],
            'mouth_landmarks': [],
            'eyebrow_landmarks': [],
            'face_contour': []
        }
        
        if not mesh_results or not mesh_results.multi_face_landmarks:
            return features
        
        try:
            for face_landmarks in mesh_results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Extract specific feature groups
                # Eyes (simplified - key points)
                left_eye = [landmarks[33], landmarks[160], landmarks[158], landmarks[133]]
                right_eye = [landmarks[362], landmarks[385], landmarks[387], landmarks[263]]
                features['eye_landmarks'].extend([(l.x, l.y, l.z) for l in left_eye + right_eye])
                
                # Mouth region
                mouth_points = [landmarks[i] for i in [61, 84, 17, 314, 405, 320, 308, 324, 318]]
                features['mouth_landmarks'].extend([(l.x, l.y, l.z) for l in mouth_points])
                
                # Eyebrows
                left_eyebrow = [landmarks[i] for i in [70, 63, 105, 66, 107]]
                right_eyebrow = [landmarks[i] for i in [296, 334, 293, 300, 276]]
                features['eyebrow_landmarks'].extend([(l.x, l.y, l.z) for l in left_eyebrow + right_eyebrow])
                
                # Face contour
                contour_points = [landmarks[i] for i in [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288]]
                features['face_contour'].extend([(l.x, l.y, l.z) for l in contour_points])
                
                # All landmarks
                features['landmarks'].extend([(l.x, l.y, l.z) for l in landmarks])
                
        except Exception as e:
            print(f"⚠️ Feature extraction error: {e}")
        
        return features
    
    def draw_face_mesh(self, frame: np.ndarray, mesh_results):
        """Draw MediaPipe face mesh"""
        if self.config.show_face_mesh and mesh_results and mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x_l, y_l = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x_l, y_l), 1, self.colors["face"], -1)
    
    def draw_body_landmarks(self, frame: np.ndarray, pose_results):
        """Draw MediaPipe body pose"""
        if self.config.show_body_pose and pose_results and pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                x_b, y_b = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x_b, y_b), 5, self.colors["body"], -1)
    
    def draw_hand_landmarks(self, frame: np.ndarray, hand_results):
        """Draw MediaPipe hand tracking"""
        if self.config.show_hand_tracking and hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_h, y_h = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x_h, y_h), 5, self.colors["hands"], -1)
    
    def draw_gaze_tracking(self, frame: np.ndarray, mesh_results):
        """Draw gaze tracking indicators"""
        if self.config.show_gaze_tracking and mesh_results and mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Left eye landmarks
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
    
    def draw_face_quality_indicators(self, frame: np.ndarray, faces: List[Dict]):
        """Draw quality indicators for detected faces"""
        for face in faces:
            bbox = face['bbox']
            quality_score = face['quality_score']
            quality_level = face['quality_level']
            
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
            
            # Draw face ID
            cv2.putText(frame, f"Face {face['id']}", (x, y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_all_visualizations(self, frame: np.ndarray, results: Dict[str, Any]):
        """Draw all MediaPipe visualizations on frame"""
        # Draw face mesh components
        self.draw_face_mesh(frame, results.get('face_mesh'))
        self.draw_gaze_tracking(frame, results.get('face_mesh'))
        self.draw_body_landmarks(frame, results.get('pose'))
        self.draw_hand_landmarks(frame, results.get('hands'))
        
        # Draw face quality indicators
        self.draw_face_quality_indicators(frame, results.get('faces', []))
    
    def get_pose_features(self, pose_results) -> Dict[str, Any]:
        """Extract pose features that might influence emotion"""
        pose_features = {
            'shoulder_slope': 0.0,
            'head_tilt': 0.0,
            'posture_confidence': 0.0,
            'body_openness': 0.0
        }
        
        if not pose_results or not pose_results.pose_landmarks:
            return pose_features
        
        try:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Calculate shoulder slope (affects confidence/mood)
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
            pose_features['shoulder_slope'] = shoulder_slope
            
            # Calculate head tilt
            nose = landmarks[0]
            left_ear = landmarks[7]
            right_ear = landmarks[8]
            head_center_y = (left_ear.y + right_ear.y) / 2
            head_tilt = abs(nose.y - head_center_y)
            pose_features['head_tilt'] = head_tilt
            
            # Overall posture confidence (based on visibility)
            visible_landmarks = sum(1 for lm in landmarks if lm.visibility > 0.5)
            pose_features['posture_confidence'] = visible_landmarks / len(landmarks)
            
            # Body openness (arms position)
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            elbow_spread = abs(left_elbow.x - right_elbow.x)
            if shoulder_width > 0:
                pose_features['body_openness'] = elbow_spread / shoulder_width
            
        except Exception as e:
            print(f"⚠️ Pose feature extraction error: {e}")
        
        return pose_features
    
    def cleanup(self):
        """Cleanup MediaPipe resources"""
        try:
            if hasattr(self, 'face_detection'):
                self.face_detection.close()
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
            if hasattr(self, 'pose'):
                self.pose.close()
            if hasattr(self, 'hands'):
                self.hands.close()
            print("✅ MediaPipe resources cleaned up")
        except Exception as e:
            print(f"⚠️ MediaPipe cleanup error: {e}")