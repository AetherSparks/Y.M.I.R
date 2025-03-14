#ALL IMPORTS
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
import atexit
import json
import pickle
import cv2
import numpy as np
import threading
import time
import mediapipe as mp
import dlib
import warnings
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from scipy.spatial import distance as dist
from collections import deque
from transformers import pipeline
from rich.console import Console
import pandas as pd
import torch
import requests
import time
import re
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

#══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════





#Flask App════════════════════════════════════════════════════════Initialised══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)



























#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


# === 📌 Load Required Models & Dataset ===
MODEL_PATH = "models/ensemble_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
SCALER_PATH = "models/scaler.pkl"
DATASET_PATH = "datasets/therapeutic_music_enriched.csv"

# ✅ Load Model, Label Encoder, and Scaler
with open(MODEL_PATH, "rb") as f:
    ensemble_model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ✅ Load Song Dataset
df = pd.read_csv(DATASET_PATH)

# === 📌 Emotion to Audio Mapping ===
EMOTION_TO_AUDIO = {
    "angry":       [0.4, 0.9, 5, -5.0, 0.3, 0.1, 0.0, 0.6, 0.2, 120],
    "disgust":     [0.3, 0.7, 6, -7.0, 0.5, 0.2, 0.0, 0.5, 0.3, 100],
    "fear":        [0.2, 0.6, 7, -10.0, 0.6, 0.3, 0.1, 0.4, 0.1, 80],
    "happy":       [0.8, 0.9, 8, -3.0, 0.2, 0.4, 0.0, 0.5, 0.9, 130],
    "sad":         [0.3, 0.4, 4, -12.0, 0.4, 0.6, 0.1, 0.3, 0.1, 70],
    "surprise":    [0.7, 0.8, 9, -6.0, 0.4, 0.3, 0.0, 0.6, 0.7, 125],
    "neutral":     [0.5, 0.5, 5, -8.0, 0.3, 0.4, 0.0, 0.4, 0.5, 110],
    "boredom":     [0.2, 0.3, 4, -15.0, 0.2, 0.8, 0.1, 0.2, 0.1, 60],
    "excitement":  [0.9, 1.0, 9, -2.0, 0.1, 0.2, 0.0, 0.7, 1.0, 140],
    "relaxation":  [0.6, 0.3, 5, -10.0, 0.2, 0.9, 0.5, 0.3, 0.7, 80]
}

# === 📌 Emotion to Mood Mapping ===
EMOTION_TO_MOOD = {
    "angry":       ["Relaxation", "Serenity"],
    "disgust":     ["Calm", "Neutral"],
    "fear":        ["Reassurance", "Serenity"],
    "happy":       ["Excitement", "Optimism"],
    "sad":         ["Optimism", "Upliftment"],
    "surprise":    ["Excitement", "Joy"],
    "neutral":     ["Serenity", "Neutral"],
    "boredom":     ["Upliftment", "Optimism"],
    "excitement":  ["Joy", "Happy"],
    "relaxation":  ["Calm", "Peaceful"]
}

# === 📌 Process Emotion Scores & Compute Features ===
def process_emotions(emotion_file):
    """Reads JSON emotion file, extracts values, and converts to audio features."""

    # ✅ Load Emotion Data
    with open(emotion_file, "r") as file:
        emotions = json.load(file)
    
    # print(f"\n📂 Loaded JSON Content:\n{json.dumps(emotions, indent=4)}\n")

    # ✅ Fix: Extract the correct dictionary
    emotions = emotions["final_average_emotions"]

    # ✅ Debugging Print (to verify)
    # print(f"DEBUG: extracted emotions -> {emotions}")
    # print(f"DEBUG: type of each value -> {[type(v) for v in emotions.values()]}")

    # ✅ Now, this should work fine
    emotion_scores = {emotion: float(score) for emotion, score in emotions.items()}


    
    # print(f"\n📊 Extracted Emotion Scores:\n{emotion_scores}\n")

    weighted_audio_features = np.zeros(len(list(EMOTION_TO_AUDIO.values())[0]))  




    print("\n🛠 Debugging Weighted Audio Features Calculation:")
    for emotion, weight in emotion_scores.items():
        if emotion in EMOTION_TO_AUDIO:
            contribution = np.array(EMOTION_TO_AUDIO[emotion]) * weight
            weighted_audio_features += contribution
            # print(f"🔹 {emotion} ({weight}): {contribution}")

    # ✅ Normalize Features Before Model Input
    weighted_audio_features = scaler.transform([weighted_audio_features])[0]

    # print(f"\n🎵 Final Normalized Audio Features (Input to Model):\n{weighted_audio_features}\n")

    return weighted_audio_features.reshape(1, -1), emotion_scores


# === 📌 Mood Prediction & Song Recommendation ===
def recommend_songs(emotion_file):
    """Predicts mood based on emotions and recommends matching songs."""

    # ✅ Get Emotion-Based Features
    emotion_vector, emotion_scores = process_emotions(emotion_file)
    mood_probs = ensemble_model.predict_proba(emotion_vector)
    # print(f"\n🔍 Model Confidence Scores for Moods: {dict(zip(le.classes_, mood_probs[0]))}\n")

    # ✅ Predict Mood
    predicted_mood_index = ensemble_model.predict(emotion_vector)[0]
    predicted_mood = le.inverse_transform([predicted_mood_index])[0]

    # print(f"\n🎯 Initial Predicted Mood (Model Output): {predicted_mood}\n")

    # ✅ Find Top 2 Dominant Emotions
    dominant_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    mapped_moods = set()
    for emotion, _ in dominant_emotions:
        mapped_moods.update(EMOTION_TO_MOOD.get(emotion, ["Neutral"]))

    # print(f"\n🎭 Dominant Emotions: {dominant_emotions} → Adjusted Moods: {mapped_moods}\n")

    # ✅ Adjust Mood If Necessary
    if predicted_mood not in mapped_moods:
        predicted_mood = list(mapped_moods)[0]  # Take the first mapped mood

    # print(f"\n🎯 Final Adjusted Mood: {predicted_mood}\n")

    # ✅ Filter Songs Based on Mood
    filtered_songs = df[df["Mood_Label"] == predicted_mood]

    # ✅ Use Fallback Moods If No Songs Found
    if filtered_songs.empty:
        filtered_songs = df[df["Mood_Label"].isin(mapped_moods)]

    # ✅ Final Fallback to Neutral if Still Empty
    if filtered_songs.empty:
        filtered_songs = df[df["Mood_Label"] == "Neutral"]

    # ✅ Select Up to 10 Songs
    recommended_songs = filtered_songs.drop_duplicates(subset=["Track Name", "Artist Name"]).sample(min(10, len(filtered_songs)))


    song_list = [
        {
            "track": row["Track Name"],
            "artist": row["Artist Name"],
            "mood": row["Mood_Label"],
        }
        for _, row in recommended_songs.iterrows()
    ]

    return song_list

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════




















































######################################################################################################################################

console = Console()

# Groq API Configuration
GROQ_API_KEY = "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"  # Replace with actual key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Load emotion classifiers
device = "cuda" if torch.cuda.is_available() else "cpu"
emotion_models = [
    pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", device=0 if device == "cuda" else -1),
    pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", device=0 if device == "cuda" else -1),
    pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=0 if device == "cuda" else -1)
]

# Load sentiment analysis model
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Emotion mapping
emotion_map = {
    "joy": "happy", "happiness": "happy", "excitement": "happy",
    "anger": "angry", "annoyance": "angry",
    "sadness": "sad", "grief": "sad",
    "fear": "fearful", "surprise": "surprised",
    "disgust": "disgusted", "neutral": "neutral",
}

# Rolling emotion tracking
previous_emotions = []

# Chat session storage
chat_session = []

# Function to handle negations
def handle_negations(text):
    """Detects negations and flips associated emotions."""
    negation_patterns = [
        r"\b(not|never|no)\s+(happy|joyful|excited)\b",
        r"\b(not|never|no)\s+(sad|depressed|unhappy)\b",
        r"\b(not|never|no)\s+(angry|mad|furious)\b"
    ]
    
    for pattern in negation_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            emotion = match.group(2).lower()
            if emotion in ["happy", "joyful", "excited"]:
                return "sad"
            elif emotion in ["sad", "depressed", "unhappy"]:
                return "happy"
            elif emotion in ["angry", "mad", "furious"]:
                return "calm"
    return None

# Function to analyze sentiment
def detect_sentiment(text):
    """Detects sentiment polarity (positive, neutral, negative)."""
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = sentiment_model(**inputs)
    sentiment_scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    sentiment_labels = ["negative", "neutral", "positive"]
    
    return sentiment_labels[torch.argmax(sentiment_scores).item()]

# Function to detect conversation emotions with improved weighting
def detect_conversation_emotions(chat_history):
    """Analyzes chat history, considers recent messages more, and balances emotion scores."""
    emotion_scores = {}
    emotion_counts = {}
    model_emotions = []
    
    # More weight to recent messages
    recent_weight = 1.5  
    messages = chat_history[-5:]  # Use last 5 messages for better context
    full_chat_text = " ".join([entry["user"] for entry in messages])

    # Check for negation handling
    negated_emotion = handle_negations(full_chat_text)
    if negated_emotion:
        return negated_emotion, {}, []

    for model in emotion_models:
        results = model(full_chat_text)
        top_predictions = sorted(results, key=lambda x: x["score"], reverse=True)[:2]

        for pred in top_predictions:
            model_label = pred["label"].lower()
            model_score = pred["score"]
            mapped_emotion = emotion_map.get(model_label, "neutral")
            model_emotions.append(f"{model_label} ({model_score:.2f}) → {mapped_emotion}")

            if model_score < 0.4:  # Ignore weak emotions
                continue  

            # Apply weight to recent messages
            weighted_score = model_score * (recent_weight if messages[-1]["user"] == full_chat_text else 1.0)

            if mapped_emotion not in emotion_scores:
                emotion_scores[mapped_emotion] = weighted_score
                emotion_counts[mapped_emotion] = 1
            else:
                emotion_scores[mapped_emotion] += weighted_score
                emotion_counts[mapped_emotion] += 1

    # Compute weighted average
    avg_emotion_scores = {label: emotion_scores[label] / emotion_counts[label] for label in emotion_scores}

    # Consider sentiment analysis
    sentiment = detect_sentiment(full_chat_text)
    if sentiment == "negative" and "sad" in avg_emotion_scores:
        avg_emotion_scores["sad"] += 0.1  # Boost sadness slightly if sentiment is negative

    # Rolling emotion tracking
    if len(previous_emotions) > 5:
        previous_emotions.pop(0)
    previous_emotions.append(avg_emotion_scores)

    # Compute final dominant emotion
    if avg_emotion_scores:
        dominant_emotion = max(avg_emotion_scores, key=avg_emotion_scores.get)
    else:
        dominant_emotion = "neutral"

    return dominant_emotion, avg_emotion_scores, model_emotions

# Function to generate chatbot response
def generate_chatbot_response(user_input):
    """Generates chatbot response using Groq API."""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama3-70b-8192", "messages": [{"role": "user", "content": user_input}]}

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response_json = response.json()

        if response.status_code == 200 and "choices" in response_json:
            return response_json["choices"][0]["message"]["content"].strip()
        else:
            console.print(f"[bold red][ERROR] Groq API request failed: {response_json}[/bold red]")
            return "I'm sorry, but I couldn't process your request."
    except Exception as e:
        console.print(f"[bold red][ERROR] Groq API request failed: {e}[/bold red]")
        return "I'm facing a technical issue. Please try again later."


######################################################################################################################################









































######################################################################################################################################
# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for API calls

# Mediapipe Modules
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.6)  # Multi-face support
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# OpenCV Haar Cascade (Backup)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

# Dlib Face Landmarks
DLIB_LANDMARK_PATH = "shape_predictor_68_face_landmarks.dat"
try:
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor(DLIB_LANDMARK_PATH)
except Exception as e:
    print(f"⚠️ Dlib Error: {e}")
    dlib_detector, dlib_predictor = None, None

# Open Webcam
cap = cv2.VideoCapture(0)
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

# Emotion Data Storage (Rolling Average)
emotion_data = {"faces": {}, "lock": threading.Lock(),  "log":[],"average_emotions" :{}}
emotion_history = deque(maxlen=5)  # Store last 5 emotion results for smoothing
emotion_log = deque(maxlen=10)  # Store last 10 logs

# Store emotions per face in a thread-safe dictionary
emotion_data = {"faces": {}, "lock": threading.Lock()}
last_known_faces = []  # Stores last detected faces

# Colors for Visualization
COLORS = {"face": (0, 255, 0), "eyes": (255, 0, 0), "body": (0, 255, 255), "hands": (0, 0, 255)}

# Eye Aspect Ratio Threshold
EYE_AR_THRESH = 0.30
FRAME_COUNT = 0  # Frame counter for controlling emotion updates

# Function to Analyze Emotions (Runs in Background)
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
                
                        

# Function for Video Streaming
def generate_frames():
    global FRAME_COUNT, cap
    last_frame_time = time.time()

    # Ensure the camera is always open
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

    while True:
        if cap is None or not cap.isOpened():
            print("⚠️ Camera is closed! Restarting...")
            cap = cv2.VideoCapture(0)  # Restart camera
            time.sleep(1)  # Small delay to allow initialization

        ret, frame = cap.read()
        if not ret:
            print("❌ Frame not captured! Retrying...")
            continue  

        frame = cv2.flip(frame, 1)  # Flip horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Limit FPS to 30
        if time.time() - last_frame_time < 1 / 30:  
            continue
        last_frame_time = time.time()

        # Process face detection & emotion analysis every 20 frames
        if FRAME_COUNT % 20 == 0:
            results = face_detection.process(rgb_frame)
            if results.detections:
                for idx, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    
                    face_roi = frame[y:y + h_box, x:x + w_box]
                    if face_roi.size > 0:
                        threading.Thread(target=analyze_emotion, args=(idx, face_roi)).start()
                    break  

        # Process Pose and Hands in separate threads
        with ThreadPoolExecutor() as executor:
            executor.submit(lambda: draw_face_mesh(frame, face_mesh.process(rgb_frame)))
            executor.submit(lambda: draw_gaze(frame, face_mesh.process(rgb_frame)))
            executor.submit(lambda: draw_body_landmarks(frame, pose.process(rgb_frame)))
            executor.submit(lambda: draw_hand_landmarks(frame, hands.process(rgb_frame)))

        FRAME_COUNT += 1
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        
        

import json
import json
import threading
def calculate_and_store_average_emotions():
    with emotion_data["lock"]:
        if "log" not in emotion_data:
            emotion_data["log"] = []  # Ensure log exists

        log_data = emotion_data["log"]
        if log_data:  # Ensure log_data is not empty
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

            # ✅ FIX: Remove extra `/ 100`
            average_emotions = {
                emotion: round(emotion_sums[emotion] / emotion_counts[emotion], 2)  # Correct averaging
                for emotion in emotion_sums
            }
            emotion_data["average_emotions"] = average_emotions

            # ✅ Save to JSON immediately
            try:
                with open("emotion_log.json", "w") as f:
                    json.dump({"log": log_data, "average_emotions": average_emotions}, f, indent=4)
                    f.flush()  # Ensure data is written to disk
            except Exception as e:
                print(f"❌ Error writing to emotion_log.json: {e}")

            # ✅ Debugging: Print updated average emotions
            print("\n📊 **Updated Average Emotions:**")
            for emotion, avg_confidence in average_emotions.items():
                print(f"  {emotion.upper()}: {avg_confidence}")

        else:
            print("⚠️ No emotion data recorded.")


            
            
            
            
            

######################################################################################################################################

def calculate_final_emotions():
    try:
        # Load JSON files
        with open("chat_results.json", "r") as f1, open("emotion_log.json", "r") as f2:
            chat_data = json.load(f1)
            emotion_log_data = json.load(f2)

        # Extract dominant emotion from chat_results.json
        dominant_emotion = chat_data["dominant_emotion"]

        # Handle possible list structure in emotion_log.json
        if isinstance(emotion_log_data["average_emotions"], list):
            average_emotions = emotion_log_data["average_emotions"][0]  # Take the first entry if it's a list
        else:
            average_emotions = emotion_log_data["average_emotions"]

        # Convert percentages to decimal
        average_emotions = {emotion: confidence / 100 for emotion, confidence in average_emotions.items()}

        # Assign 100% confidence to the dominant emotion from chat
        dominant_emotion_dict = {emotion: 0.0 for emotion in average_emotions}
        dominant_emotion_dict[dominant_emotion] = 1.0  # 100% as 1.0

        # Convert both to DataFrames
        df1 = pd.DataFrame([dominant_emotion_dict])  # From chat_results.json
        df2 = pd.DataFrame([average_emotions])  # From emotion_log.json

        # Combine and compute the average
        final_average_df = pd.concat([df1, df2], ignore_index=True)
        final_average_emotions = final_average_df.mean().round(4)  # Keep two decimal places

        # Convert to dictionary
        final_emotion_result = final_average_emotions.to_dict()

        # Save to JSON
        with open("final_average_emotions.json", "w") as f:
            json.dump({"final_average_emotions": final_emotion_result}, f, indent=4)

        return final_emotion_result

    except Exception as e:
        return {"error": str(e)}

######################################################################################################################################
import threading
import time
import json

latest_songs = []  # Store recommended songs
latest_final_emotion = {}  # Store latest emotion data
running = True  # Flag to control background process

def update_all_in_background(interval=5):
    """Continuously updates emotions, calculates final averages, and recommends songs."""
    global latest_songs, latest_final_emotion

    while running:
        try:
            print("\n🔄 Processing Emotions & Recommendations...")
            
            # Step 1: Process Video  Emotions
            calculate_and_store_average_emotions()  

            # Step 2: Calculate Final Emotion from Both Sources
            final_emotions = calculate_final_emotions()  
            latest_final_emotion = final_emotions  # Store it globally
            
            # Step 3: Recommend Songs Based on Final Emotion
            latest_songs = recommend_songs("final_average_emotions.json")
            
            print("✅ Updated Final Emotion:", latest_final_emotion)  # Debugging print
            print("✅ Updated Songs:", latest_songs[:3])  # Print first 3 songs as a check
            print("✅ Music recommendations updated.")

        except Exception as e:
            print(f"❌ Error updating: {e}")

        time.sleep(interval)  # Update every `interval` seconds










######################################################################################################################################

          
# Flask Route: Home
@app.route('/')

def home1():
    return render_template('home.html')

@app.route('/ai_app')
def ai_app():
    return render_template('index.html')

# About Page Route
@app.route('/about')
def about():
    return render_template('about.html')

# Contact Page Route    
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Flask Route: Get Emotions
@app.route('/get_emotions', methods=['GET'])
def get_emotions():
    with emotion_data["lock"]:
        return jsonify({
            "faces": emotion_data.get("faces", {}),
            "average_emotions": emotion_data.get("average_emotions", {})
        })


# Flask Route: Get Logs
@app.route('/get_logs')
def get_logs():
    return jsonify(list(emotion_log))

# Flask Route: Video Feed
@app.route('/video_feed')
def video_feed():
    """Starts the video feed and reinitializes the camera if necessary."""
    global cap

    if cap is None or not cap.isOpened():
        print("🔄 Restarting camera...")
        cap = cv2.VideoCapture(0)  # Restart camera

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



########################################################################### 📸 CAMERA CONTROL ROUTES ############################################################################## 

running = True  # To control video streaming


@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    """Pauses or resumes the camera feed based on user activity."""
    global cap
    action = request.json.get("action")  # Expecting 'pause' or 'resume'

    try:
        if action == "pause":
            if cap is not None and cap.isOpened():
                print("⏸️ Camera paused.")
            return jsonify({"status": "Camera paused"}), 200

        elif action == "resume":
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)  # Restart if needed
            print("▶️ Camera resumed.")
            return jsonify({"status": "Camera resumed"}), 200

        else:
            return jsonify({"error": "Invalid action"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

###🛑 CLEANUP AND EXIT ###

def cleanup():
    """Stops the camera, saves emotions, and releases resources on exit."""
    global cap, running
    try:
        running = False  # Stop video stream loop

        if cap is not None and cap.isOpened():
            cap.release()  # Release webcam
            cap = None  # Ensure it's fully removed
            print("🎥 Camera released successfully.")

        cv2.destroyAllWindows()
        print("✅ Cleanup complete: Camera released, and emotions saved.")

    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

# Ensure cleanup runs when Flask stops
atexit.register(cleanup)  # Runs cleanup when Flask app stops

@app.route('/exit', methods=['POST'])
def stop_server():
    """Stops Flask when the user closes the tab."""
    print("🛑 Received exit request. Shutting down server...")
    cleanup()  # Perform cleanup actions

    # Try shutting down Flask server cleanly


#####################################################################################################################################





############################################################CHAtbotttttt flasskk#######################################################
@app.route('/chat', methods=['POST'])
def chat():
    """Handles chatbot interaction and continuously updates detected emotions."""
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "No message provided."}), 400

    chatbot_response = generate_chatbot_response(user_input)

    # Log conversation
    chat_session.append({"user": user_input, "chatbot": chatbot_response})

    # Continuously detect dominant emotion
    dominant_emotion, emotion_scores, model_emotions = detect_conversation_emotions(chat_session)

    response_data = {
        "response": chatbot_response,
        "dominant_emotion": dominant_emotion,  # Always update detected emotion
        "model_emotions": model_emotions
    }

    # Save chat results **after every message**
    save_chat_results()

    # If the user is ending the conversation
    if user_input.lower() in ["quit", "bye", "exit", "goodbye", "end"]:
        response_data["end_chat"] = True  # Notify frontend to stop input

    return jsonify(response_data)

@app.route('/detect_emotion', methods=['GET'])
def detect_emotion():
    """Detects the dominant emotion from the chat session (for real-time updates)."""
    if not chat_session:
        return jsonify({"dominant_emotion": "neutral", "model_emotions": []})  # Default if empty chat

    dominant_emotion, emotion_scores, model_emotions = detect_conversation_emotions(chat_session)
    return jsonify({"dominant_emotion": dominant_emotion, "model_emotions": model_emotions})

@app.route('/save_chat', methods=['POST'])
def save_chat():
    """Saves chat conversation and detected emotion."""
    if not chat_session:
        return jsonify({"error": "No chat data to save."}), 400

    save_chat_results()  # Save after every message
    return jsonify({"message": "Chat saved successfully."})

def save_chat_results():
    """Saves chatbot results (full conversation + updated emotions) to `chat_results.json`."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    dominant_emotion, emotion_scores, model_emotions = detect_conversation_emotions(chat_session)

    chat_data = {
        "timestamp": timestamp,
        "conversation": chat_session,
        "dominant_emotion": dominant_emotion,
        "emotion_scores": emotion_scores,
        "model_emotions": model_emotions
    }

    with open("chat_results.json", "w") as f:
        json.dump(chat_data, f, indent=4)


######################################################################################################################################                              
@app.route('/process_results', methods=['POST'])
def process_results():
    final_emotions = calculate_final_emotions()
    if "error" in final_emotions:
        return jsonify({"error": final_emotions["error"]})

    songs = recommend_songs("final_averaged_emotions.json")  # Ensure correct file is passed
    return jsonify({"final_emotions": final_emotions, "recommended_songs": songs})
#########################################################################################################################################



# Flask App Run
if __name__ == '__main__':
    # Start background processing thread
    background_thread = threading.Thread(target=update_all_in_background, daemon=True)
    background_thread.start()
    try:
        app.run(debug=True, host='127.0.0.1', port=5000)

    except KeyboardInterrupt:
        print("\n🔴 Server stopped manually.")

        

        
