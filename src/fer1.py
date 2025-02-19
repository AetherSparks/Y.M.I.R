import json
import cv2
import numpy as np
import threading
import time
import mediapipe as mp
import dlib
import warnings
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import distance as dist

# Suppress Warnings
warnings.filterwarnings("ignore")

# Initialize Mediapipe Modules
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.6)
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# OpenCV Haar Cascade (Backup)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

# Dlib for High Precision Face Landmark Detection
DLIB_LANDMARK_PATH = "shape_predictor_68_face_landmarks.dat"
try:
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor(DLIB_LANDMARK_PATH)
except Exception as e:
    print(f"⚠️ Dlib Error: {e}")
    dlib_detector = None
    dlib_predictor = None


# Open Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Camera not detected!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# If No Faces Are Detected, Keep Last Known Faces for a While
NO_FACE_LIMIT = 30  # Number of frames to retain last known faces
no_face_counter = 0  # Tracks how many frames we’ve had no face detected
fps = 0
frame_count = 0
start_time = time.time()
frame_skip = 3  # Optimized FPS

# Create FPS Slider
cv2.namedWindow("Advanced Emotion Detection & Multimodal Analysis")
cv2.createTrackbar("FPS Control", "Advanced Emotion Detection & Multimodal Analysis", 27, 29, lambda val: None)


# Store emotions per face in a thread-safe dictionary
emotion_data = {"faces": {}, "lock": threading.Lock()}
last_known_faces = []  # Stores last detected faces

# Colors for different landmarks
COLORS = {
    "eyes": (0, 255, 255),
    "mouth": (255, 0, 0),
    "face": (0, 255, 0),
    "hands": (0, 0, 255),
    "body": (255, 255, 0)
}

# Thread Pool for Emotion Analysis
executor = ThreadPoolExecutor(max_workers=4)

# Gaze Tracking Variables
EYE_AR_THRESH = 0.30  # Eye aspect ratio threshold for gaze detection

# Multi-threaded Emotion Analysis with a time delay
import time
import json
import cv2
from deepface import DeepFace

def analyze_emotion(face_id, face_roi):
    current_time = time.time()

    with emotion_data["lock"]:
        # Ensure log list exists
        if "log" not in emotion_data:
            emotion_data["log"] = []

        emotion_data["last_update"] = current_time

    try:
        resized_face = cv2.resize(face_roi, (224, 224))
        emotion_result = DeepFace.analyze(resized_face, actions=['emotion'], enforce_detection=False, detector_backend='opencv')

        with emotion_data["lock"]:
            emotions = emotion_result[0]['emotion']
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))  # Format timestamp

            # Store emotions and timestamp
            emotion_data["faces"][face_id] = emotions
            emotion_data["log"].append({"timestamp": timestamp, "emotions": emotions})

            # Print emotions with timestamp
            print(f"🕒 {timestamp} - Emotions: {emotions}")

        # Save to JSON file (thread-safe)
        with emotion_data["lock"]:
            with open("emotion_log.json", "w") as f:
                json.dump(emotion_data["log"], f, indent=4)

    except Exception as e:
        print(f"⚠️ Emotion detection error: {e}")



# Gaze Tracking
def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the Euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def draw_gaze(frame, mesh_results):
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
                cv2.circle(frame, (x, y), 2, COLORS["eyes"], -1)
            # Calculate eye aspect ratio
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            # Display gaze direction
            if ear < EYE_AR_THRESH:
                cv2.putText(frame, "Looking forward", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Looking away", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Draw Face Mesh
def draw_face_mesh(frame, mesh_results):
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x_l, y_l = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x_l, y_l), 1, COLORS["face"], -1)

# Draw Body Landmarks
def draw_body_landmarks(frame, pose_results):
    if pose_results.pose_landmarks:
        for landmark in pose_results.pose_landmarks.landmark:
            x_b, y_b = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x_b, y_b), 5, COLORS["body"], -1)

# Draw Hand Landmarks
def draw_hand_landmarks(frame, hand_results):
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x_h, y_h = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x_h, y_h), 5, COLORS["hands"], -1)

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
      print("❌ Frame not captured. Retrying...")
      continue  # Keep retrying instead of exiting

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face Detection
    results = face_detection.process(rgb_frame)
    face_bboxes = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            face_bboxes.append((x, y, w_box, h_box))
            break  # Only process the first detected face

    # If Mediapipe Fails, Use Haar Cascade
    if not face_bboxes:
        face_bboxes = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))

    # If Still No Face, Use Last Known Position
    if not face_bboxes:
        face_bboxes = last_known_faces

    # Store Last Known Face Positions
    if face_bboxes:
     last_known_faces = face_bboxes  # Update last known faces
     no_face_counter = 0  # Reset counter
    else:
       no_face_counter += 1  # Increase the counteri
       if no_face_counter < NO_FACE_LIMIT:
          face_bboxes = last_known_faces  # Use last known positions

    # Face Mesh
    mesh_results = face_mesh.process(rgb_frame)

    # Body & Hand Detection
    pose_results = pose.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    # Process Only the First Face
    if frame_count % frame_skip == 0 and face_bboxes:
        x, y, w_box, h_box = face_bboxes[0]
        face_roi = frame[y:y + h_box, x:x + w_box]
        if face_roi.size > 0:
            executor.submit(analyze_emotion, 0, face_roi)
    
    # Draw Face Boxes & Emotions
    with emotion_data["lock"]:
        detected_faces = emotion_data["faces"]

        for i, (x, y, w_box, h_box) in enumerate(face_bboxes):
            if i == 0:  # Only process the first face
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), COLORS["face"], 2)

                # Draw Face Mesh
                draw_face_mesh(frame, mesh_results)

                # Display Emotions
                if i in detected_faces:
                    emotions = detected_faces[i]
                    text_x = x + w_box + 10
                    text_y = y + 20
                    for emotion, confidence in emotions.items():
                        print(f"  {emotion.upper()}: {round(confidence, 2)}%")
                        text = f"{emotion.upper()}: {round(confidence, 2)}%"
                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        text_y += 25

    # Draw Gaze Direction
    draw_gaze(frame, mesh_results)

    # Draw Body Landmarks
    draw_body_landmarks(frame, pose_results)

    # Draw Hand Landmarks
    draw_hand_landmarks(frame, hand_results)

    # Display FPS
    elapsed_time = time.time() - start_time
    fps = round(frame_count / elapsed_time, 2)
    cv2.putText(frame, f"FPS: {fps}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show Output
    cv2.imshow("Advanced Emotion Detection & Multimodal Analysis", frame)

    # Exit Condition
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("Advanced Emotion Detection & Multimodal Analysis", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
executor.shutdown()
# Compute Average Emotion at End
with open("emotion_log.json", "r") as f:
    log_data = json.load(f)

if log_data:
    emotion_sums = {}
    emotion_counts = {}
    
    for entry in log_data:
        for emotion, confidence in entry["emotions"].items():
            if emotion in emotion_sums:
                emotion_sums[emotion] += confidence
                emotion_counts[emotion] += 1
            else:
                emotion_sums[emotion] = confidence
                emotion_counts[emotion] = 1

    average_emotions = {emotion: round(emotion_sums[emotion] / emotion_counts[emotion], 2) for emotion in emotion_sums}
    print("\n📊 **Average Detected Emotions:**")
    for emotion, avg_confidence in average_emotions.items():
        print(f"  {emotion.upper()}: {avg_confidence}%")
else:
    print("⚠️ No emotion data recorded.")