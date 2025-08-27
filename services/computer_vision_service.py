"""
COMPUTER VISION MICROSERVICE
Extracted from app.py lines 678-798
Contains all video processing, face detection, and MediaPipe drawing functions
"""
import cv2
import time
import threading
import numpy as np
from scipy.spatial import distance as dist
from concurrent.futures import ThreadPoolExecutor
import mediapipe as mp

class ComputerVisionService:
    def __init__(self):
        # Initialize MediaPipe (from app.py:587-596)
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands

        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.6)
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Colors for drawing (from app.py:637)
        self.COLORS = {
            "face": (0, 255, 0),      # Green for face mesh
            "eyes": (255, 0, 0),      # Blue for eyes  
            "body": (0, 255, 255),    # Yellow for body landmarks
            "hands": (0, 0, 255)      # Red for hands
        }
        
        # Eye aspect ratio threshold (from app.py:640)
        self.EYE_AR_THRESH = 0.30
        
        # Camera and frame management
        self.cap = None
        self.FRAME_COUNT = 0
        self._camera_active = True  # Track camera state - DEFAULT: ON like app.py
        print(f"🔍 DEBUG: Camera initialized with active = {self._camera_active}")
        
        print("✅ Computer Vision Service initialized")
        
    @property
    def camera_active(self):
        return self._camera_active
        
    @camera_active.setter  
    def camera_active(self, value):
        if self._camera_active != value:
            import traceback
            print(f"🔍 DEBUG: Camera active changing from {self._camera_active} to {value}")
            print("🔍 DEBUG: Call stack:")
            traceback.print_stack()
        self._camera_active = value

    # EXTRACTED FROM app.py:678-686 (EXACT COPY)
    def eye_aspect_ratio(self, eye):
        """Compute the eye aspect ratio"""
        # Compute the Euclidean distances between the two sets of vertical eye landmarks
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # Compute the Euclidean distance between the horizontal eye landmarks
        C = dist.euclidean(eye[0], eye[3])
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    # EXTRACTED FROM app.py:688-713 (EXACT COPY)
    def draw_gaze(self, frame, mesh_results):
        """Draw gaze tracking on the frame"""
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                # Extract left and right eye landmarks
                left_eye = [(landmarks[33].x, landmarks[33].y), (landmarks[160].x, landmarks[160].y),
                            (landmarks[158].x, landmarks[158].y), (landmarks[133].x, landmarks[133].y),
                            (landmarks[153].x, landmarks[153].y), (landmarks[144].x, landmarks[144].y)]
                right_eye = [(landmarks[362].x, landmarks[362].y), (landmarks[385].x, landmarks[385].y),
                             (landmarks[387].x, landmarks[387].y), (landmarks[263].x, landmarks[263].y),
                             (landmarks[373].x, landmarks[373].y), (landmarks[380].x, landmarks[380].y)]
                # Convert to pixel coordinates
                left_eye = [(int(l[0] * frame.shape[1]), int(l[1] * frame.shape[0])) for l in left_eye]
                right_eye = [(int(r[0] * frame.shape[1]), int(r[1] * frame.shape[0])) for r in right_eye]
                # Draw eyes
                for (x, y) in left_eye + right_eye:
                    cv2.circle(frame, (x, y), 2, self.COLORS["eyes"], -1)
                # Calculate eye aspect ratio
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                # Display gaze direction
                if ear < self.EYE_AR_THRESH:
                    cv2.putText(frame, "Looking forward", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Looking away", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # EXTRACTED FROM app.py:716-721 (EXACT COPY)
    def draw_face_mesh(self, frame, mesh_results):
        """Draw face mesh landmarks"""
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x_l, y_l = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x_l, y_l), 1, self.COLORS["face"], -1)

    # EXTRACTED FROM app.py:724-728 (EXACT COPY)
    def draw_body_landmarks(self, frame, pose_results):
        """Draw body pose landmarks"""
        if pose_results.pose_landmarks:
            for landmark in pose_results.pose_landmarks.landmark:
                x_b, y_b = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x_b, y_b), 5, self.COLORS["body"], -1)

    # EXTRACTED FROM app.py:731-736 (EXACT COPY)
    def draw_hand_landmarks(self, frame, hand_results):
        """Draw hand landmarks"""
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_h, y_h = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x_h, y_h), 5, self.COLORS["hands"], -1)

    # EXTRACTED FROM app.py:742-798 (EXACT COPY)
    def generate_frames(self, emotion_service=None):
        """Generate video frames with all visual processing - EXACTLY like app.py"""
        last_frame_time = time.time()

        # Simple camera initialization like app.py (lines 747-748)
        if self.cap is None or not self.cap.isOpened():
            print("🔄 Starting camera like app.py...")
            self.cap = cv2.VideoCapture(0)
            self.camera_active = True
            print(f"✅ Camera initialized and active: {self.camera_active}")
        
        while True:
            # If camera is deactivated (privacy mode), show black frame
            if not self.camera_active:
                # Create black frame for privacy
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "Camera Stopped", (250, 240), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', black_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.1)  # Small delay
                continue
            
            # Ensure camera is always open (like app.py lines 751-754)
            if self.cap is None or not self.cap.isOpened():
                print("⚠️ Camera is closed! Restarting like app.py...")
                self.cap = cv2.VideoCapture(0)  # Simple restart like app.py
                self.camera_active = True  # Mark as active when restarted
                print(f"✅ Camera restarted and active: {self.camera_active}")
                time.sleep(1)  # Small delay to allow initialization
                
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Frame not captured! Retrying like app.py...")
                continue  # Simple continue like app.py (line 759)  

            frame = cv2.flip(frame, 1)  # Flip horizontally
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Limit FPS to 30
            if time.time() - last_frame_time < 1 / 30:  
                continue
            last_frame_time = time.time()

            # Process face detection & emotion analysis every 10 frames (more frequent)
            if self.FRAME_COUNT % 10 == 0:
                results = self.face_detection.process(rgb_frame)
                face_found = False
                
                # Debug: Print detection info every 50 frames
                if self.FRAME_COUNT % 50 == 0:
                    detection_count = len(results.detections) if results.detections else 0
                    print(f"🔍 Frame {self.FRAME_COUNT}: {detection_count} faces detected")
                
                if results.detections and emotion_service:
                    print(f"📸 Frame {self.FRAME_COUNT}: Found {len(results.detections)} faces, starting emotion analysis...")
                    for idx, detection in enumerate(results.detections):
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                        
                        print(f"🔲 Face {idx} bounding box: x={x}, y={y}, w={w_box}, h={h_box}")
                        
                        # Ensure bounding box is valid
                        if x >= 0 and y >= 0 and w_box > 0 and h_box > 0 and (x + w_box) <= w and (y + h_box) <= h:
                            face_roi = frame[y:y + h_box, x:x + w_box]
                            if face_roi.size > 0:
                                print(f"✅ Valid face ROI for face {idx}, starting emotion analysis...")
                                # Start emotion analysis in background thread
                                threading.Thread(target=emotion_service.analyze_emotion, args=(idx, face_roi), daemon=True).start()
                                face_found = True
                            else:
                                print(f"❌ Empty face ROI for face {idx}")
                            break
                        else:
                            print(f"❌ Invalid bounding box for face {idx}")
                elif results.detections and not emotion_service:
                    print(f"⚠️ Frame {self.FRAME_COUNT}: Faces detected but no emotion service!")
                elif not results.detections:
                    print(f"🔍 Frame {self.FRAME_COUNT}: No faces detected in this frame")
                
                # Debug: Print face detection status
                if not face_found and self.FRAME_COUNT % 100 == 0:  # Every 100 frames
                    print(f"🔍 Frame {self.FRAME_COUNT}: No faces detected")  

            # Process pose, hands, and face mesh in parallel threads
            with ThreadPoolExecutor() as executor:
                mesh_future = executor.submit(self.face_mesh.process, rgb_frame)
                pose_future = executor.submit(self.pose.process, rgb_frame)
                hands_future = executor.submit(self.hands.process, rgb_frame)
                
                # Get results and draw
                mesh_results = mesh_future.result()
                pose_results = pose_future.result()
                hands_results = hands_future.result()
                
                self.draw_face_mesh(frame, mesh_results)
                self.draw_gaze(frame, mesh_results)
                self.draw_body_landmarks(frame, pose_results)
                self.draw_hand_landmarks(frame, hands_results)

            self.FRAME_COUNT += 1
            
            # Encode and yield frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def initialize_camera(self):
        """Simple camera initialization like app.py"""
        try:
            print("📷 Starting camera...")
            self.cap = cv2.VideoCapture(0)  # Simple like app.py line 611
            if self.cap.isOpened():
                self.camera_active = True
                print("✅ Camera started")
                return True
            else:
                print("❌ Camera not available")
                return False
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False

    def stop_camera(self):
        """Stop and release camera"""
        if self.cap:
            self.cap.release()
            self.cap = None
            self.camera_active = False
        print("🛑 Camera stopped")

    def cleanup(self):
        """Cleanup resources"""
        self.stop_camera()
        cv2.destroyAllWindows()
        print("🧹 Computer vision service cleanup complete")