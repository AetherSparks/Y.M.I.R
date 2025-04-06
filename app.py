#ALL IMPORTS
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
from email.message import EmailMessage
import random
import smtplib
from dotenv import load_dotenv
load_dotenv()
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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
from flask import Flask, render_template, Response, jsonify, render_template_string, request, send_from_directory, session, url_for, redirect ,flash
from flask_mail import Mail , Message
from flask_session import Session
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
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

#══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════




#Flask App════════════════════════════════════════════════════════Initialised══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY' , 'fallbackkey123')

#══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════



# Email Config
# Email Config
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USE_SSL = False,
    MAIL_USERNAME=os.environ.get('EMAIL_USER'),
    MAIL_PASSWORD=os.environ.get('EMAIL_PASS'),
    MAIL_DEFAULT_SENDER=os.environ.get('EMAIL_USER')
)

# Initialize mail with app explicitly
mail = Mail(app)


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///emotion_ai.db'  # or use PostgreSQL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# print("Mail config:", {
#     "server": app.config['MAIL_SERVER'],
#     "port": app.config['MAIL_PORT'],
#     "username": bool(app.config['MAIL_USERNAME']),  # Just print if it exists
#     "password": bool(app.config['MAIL_PASSWORD']),  # Jus   t print if it exists
#     "use_tls": app.config['MAIL_USE_TLS']
# })


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    favorites = db.relationship('FavoriteSong', backref='user', lazy=True)

class FavoriteSong(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), nullable=False)
    artist = db.Column(db.String(120), nullable=False)
    link = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)















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




    # print("\n🛠 Debugging Weighted Audio Features Calculation:")
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
    # Load pre-trained transformations
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/pca.pkl", "rb") as f:
        pca = pickle.load(f)

    # Apply transformations
    emotion_vector = scaler.transform(emotion_vector.reshape(1, -1))  # Standardize
    emotion_vector = pca.transform(emotion_vector)  # Reduce dimensions

    # Predict using ensemble model
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




















































#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
console = Console()

# Groq API Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
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
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════









































#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
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
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════






            
            
            
            
            

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
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
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


import yt_dlp
import os

def search_youtube(song, artist):
    query = f"{song} {artist} audio"
    search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
    # You can use YouTube API or scrape first result link
    # For now, use yt-dlp to search:
    ydl_opts = {
        'quiet': True,
        'default_search': 'ytsearch1',  # get top result
        'skip_download': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(query, download=False)
        if 'entries' in result and result['entries']:
            return result['entries'][0]['webpage_url']  # Top video URL
    return None

def download_youtube_async(song, artist, filename):
    # Search & download audio from YouTube using yt-dlp
    query = f"{artist} {song} audio"

    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(MUSIC_DIR, filename.replace('.mp3', '')),
            'quiet': True,
            'noplaylist': True,
            'logger': None,
            'no_warnings': True,
            'progress_hooks': [],
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'default_search': 'ytsearch1',
        }


        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([query])

        return jsonify({
            'audio_link': url_for('static', filename=f'music/{filename}'),
            'source': 'youtube'
        })

    except Exception as e:
        print(f"Error downloading from YouTube: {e}")
        return jsonify({'error': 'Song not found'}), 404



#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
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
            # print("\n🔄 Processing Emotions & Recommendations...")
            
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
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
import re

def clean_filename(text):
    return re.sub(r'[^\w\-_\. ]', '_', text)



#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
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

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap
    print("🛑 Stopping camera...")
    if cap is not None:
        cap.release()
        cap = None
    return ('', 204)

analyzer = SentimentIntensityAnalyzer()

app.secret_key = "your_secret_key"

DATA_FILE = 'goals.json'
POSTS_FILE = 'data/posts.json'

# Mood-based meditation scripts (more human-like and varied)
SCRIPTS = {
    "anxious": [
        "Take a deep breath in... and out. Imagine a calm ocean. Let the waves carry your anxiety away.",
        "Inhale peace. Exhale worry. Picture a peaceful forest with birds gently chirping.",
        "You are safe. You are grounded. With every breath, let go of anxious thoughts."
    ],
    "stressed": [
        "Let go of tension with every breath. Relax your shoulders. You are safe.",
        "You are not your stress. You are strength, you are calm. Let each exhale ground you.",
        "Breathe deeply. Picture your thoughts floating like clouds, drifting far away."
    ],
    "tired": [
        "Close your eyes. Imagine a soft, glowing light recharging your body and mind.",
        "Sink into stillness. Every breath is a wave of renewal flowing through you.",
        "Let your body rest. Let your mind slow down. You deserve peace and rest."
    ],
    "happy": [
        "Let’s deepen your joy. Smile softly and be present with the happiness within.",
        "Breathe in gratitude. Breathe out love. Stay with this beautiful feeling.",
        "Feel the warmth inside you. Your happiness is a gift — cherish this moment."
    ]
}

# Optional: Motivational quotes to display with the meditation
QUOTES = [
    "You are enough. Just as you are.",
    "Breathe. You’ve got this.",
    "Peace begins with a single breath.",
    "Today is a fresh start.",
    "Inner calm is your superpower."
]

@app.route("/meditation")
def meditation():
    return render_template("meditation.html")

@app.route("/meditation/result", methods=["POST"])
def meditation_result():
    feeling = request.form.get("feeling", "").lower()

    # Find matching script list based on mood keyword
    for mood, scripts in SCRIPTS.items():
        if mood in feeling:
            script = random.choice(scripts)
            break
    else:
        # Default script if mood not found
        script = f"Let’s take a few moments to be still. You mentioned feeling '{feeling}'. Breathe deeply and allow peace to fill your body."

    quote = random.choice(QUOTES)

    return render_template("meditation_result.html", script=script, quote=quote)

@app.route('/journal', methods=['GET', 'POST'])
def journal():
    if request.method == 'POST':
        entry = request.form['entry']
        sentiment, suggestion = analyze_journal(entry)
        return render_template('journal.html', sentiment=sentiment, suggestion=suggestion, entry=entry)
    return render_template('journal.html')

def analyze_journal(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.05:
        sentiment = 'positive'
    elif compound <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    suggestions = {
        "positive": "Keep up the positive energy! 😊",
        "negative": "Try writing about what made you feel this way. 💬",
        "neutral": "Explore your thoughts more deeply next time. ✍️"
    }

    return sentiment, suggestions.get(sentiment)

@app.route('/breathing', methods=['GET', 'POST'])
def breathing():
    suggestion = None
    if request.method == 'POST':
        mood = request.form['mood'].lower()
        suggestion = suggest_breathing(mood)
    return render_template('breathing.html', suggestion=suggestion)

def suggest_breathing(mood):
    techniques = {
        "anxious": "Box Breathing (4-4-4-4) – Inhale, hold, exhale, hold for 4 seconds each.",
        "stressed": "4-7-8 Breathing – Inhale 4s, hold 7s, exhale 8s. Great for calming nerves.",
        "tired": "Diaphragmatic Breathing – Deep belly breaths to refresh energy.",
        "distracted": "Alternate Nostril Breathing – Helps center your focus.",
        "neutral": "Guided Breath Awareness – Simply observe your breath."
    }
    return techniques.get(mood, "Try Box Breathing to get started.")

def load_goals():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def save_goals(goals):
    with open(DATA_FILE, 'w') as f:
        json.dump(goals, f, indent=4)

@app.route('/goals', methods=['GET', 'POST'])
def goals():
    if request.method == 'POST':
        new_goal = request.form.get('goal')
        if new_goal:
            goals = load_goals()
            goals.append({
                "goal": new_goal,
                "created": datetime.today().strftime('%Y-%m-%d'),
                "streak": 0,
                "last_checked": ""
            })
            save_goals(goals)
            return redirect(url_for('goals'))
    
    goals = load_goals()
    return render_template('goals.html', goals=goals)

@app.route('/check_goal/<int:goal_index>')
def check_goal(goal_index):
    goals = load_goals()
    today = datetime.today().strftime('%Y-%m-%d')

    if goals[goal_index]["last_checked"] != today:
        goals[goal_index]["last_checked"] = today
        goals[goal_index]["streak"] += 1
        save_goals(goals)

    return redirect(url_for('goals'))

@app.route('/sound-therapy', methods=['GET', 'POST'])
def sound_therapy():
    mood = request.form.get('mood') if request.method == 'POST' else None

    mood_to_sound = {
        "relaxed": {
            "title": "Sunset Landscape",
            "file": "Sunset-Landscape(chosic.com).mp3"
        },
        "anxious": {
            "title": "White Petals",
            "file": "keys-of-moon-white-petals(chosic.com).mp3"
        },
        "sad": {
            "title": "Rainforest Sounds",
            "file": "Rain-Sound-and-Rainforest(chosic.com).mp3"
        },
        "tired": {
            "title": "Meditation",
            "file": "meditation.mp3"
        },
        "focus": {
            "title": "Magical Moments",
            "file": "Magical-Moments-chosic.com_.mp3"
        }
    }

    recommended = mood_to_sound.get(mood, None)

    # All available sounds (for browsing below)
    all_sounds = list(mood_to_sound.values())

    return render_template('sound_therapy.html', recommended=recommended, all_sounds=all_sounds)

def load_posts():
    if os.path.exists(POSTS_FILE):
        with open(POSTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_posts(posts):
    with open(POSTS_FILE, 'w') as f:
        json.dump(posts, f, indent=4)

@app.route('/community', methods=['GET', 'POST'])
def community_support():
    posts = load_posts()

    if request.method == 'POST':
        username = request.form['username']
        message = request.form['message']
        # Very basic AI reply simulation (you can plug in sentiment/local AI later)
        ai_response = "Thanks for sharing. You're not alone on this journey 🌟"

        posts.insert(0, {
            'username': username,
            'message': message,
            'reply': ai_response
        })

        save_posts(posts)
        return redirect(url_for('community_support'))

    return render_template('community_support.html', posts=posts)

# Sample movie list
movie_data = [
    {"title": "Inception", "genres": "Action|Sci-Fi|Thriller"},
    {"title": "The Dark Knight", "genres": "Action|Crime|Drama"},
    {"title": "Titanic", "genres": "Drama|Romance"},
    {"title": "The Shawshank Redemption", "genres": "Drama"},
    {"title": "Avatar", "genres": "Action|Adventure|Fantasy"}
]

@app.route('/recommend', methods=['GET', 'POST'])
def home():
    mood = None
    recommendations = None
    
    if request.method == 'POST':
        mood = request.form['mood']
        recommendations = get_movie_recommendations(mood)

    return render_template('recommendations.html', mood=mood, recommendations=recommendations)

def get_movie_recommendations(mood):
    # Filter movies based on mood, for simplicity we just return all movies here
    # You can customize this logic to filter movies based on the mood
    return movie_data

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        try:
            print("======= CONTACT FORM SUBMISSION =======")
            print("Form data:", request.form)

            # Get form data
            name = request.form.get('name', '')
            email = request.form.get('email', '')
            subject = request.form.get('subject', 'No Subject')
            message = request.form.get('message', '')
            phone = request.form.get('phone', 'Not provided')

            # Timestamp for submission
            submission_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Create the email message
            print("Creating message...")
            msg = Message(
                subject=f"New Contact Inquiry: {subject}",
                sender=("Your Website Contact Form", app.config.get('MAIL_DEFAULT_SENDER') or os.getenv('EMAIL_USER')),
                recipients=[app.config.get('MAIL_USERNAME') or os.getenv('EMAIL_USER')],
                reply_to=email
            )

            # Plain text body (ASCII-safe fallback)
            msg.body = f"""Hello Admin,

You have received a new contact form submission on your website.

Submitted On: {submission_time}

Name: {name}
Email: {email}
Phone: {phone}
Subject: {subject}
Message:
{message}

Please respond promptly.
"""

            # HTML body (UTF-8 + emoji support)
            html_body = render_template_string("""
<html>
  <body style="font-family: Arial, sans-serif; color: #333;">
    <h2> New Contact Form Submission</h2>
    <p><strong> Submitted On:</strong> {{ submission_time }}</p>
    <p><strong> Name:</strong> {{ name }}</p>
    <p><strong> Email:</strong> {{ email }}</p>
    <p><strong> Phone:</strong> {{ phone }}</p>
    <p><strong> Subject:</strong> {{ subject }}</p>
    <p><strong> Message:</strong><br>{{ message }}</p>
    <hr>
    <p>Regards,<br><strong>Your Website Bot</strong></p>
  </body>
</html>
""", submission_time=submission_time, name=name, email=email, phone=phone, subject=subject, message=message.replace('\n', '<br>'))

            msg.html = html_body  # Attach HTML email

            # Optional: Handle attachments
            if 'attachment' in request.files:
                file = request.files['attachment']
                if file and file.filename != '':
                    print(f"Attaching file: {file.filename}")
                    file_content = file.read()
                    msg.attach(file.filename, file.content_type, file_content)
                    print("Attachment added.")

            # Send email
            print("Sending email...")
            mail.send(msg)
            print("Email sent!")

            return jsonify({"success": True, "message": "Thank you! Your message has been sent."})

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print("======= CONTACT FORM ERROR =======")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Traceback:\n{error_details}")

            return jsonify({
                "success": False,
                "message": "Oops! Something went wrong. Please try again later."
            }), 500

    # GET request → render contact form
    return render_template('contact.html')

from flask import jsonify

@app.route('/book-appointment', methods=['POST'])
def book_appointment():
    user_name = request.form['user_name']
    user_email = request.form['user_email']
    appointment_date = request.form['appointment_date']
    time_slot = request.form['time_slot']
    duration = request.form['meeting_duration']
    timezone = request.form['timezone']
    notes = request.form['appointment_notes']

    subject = 'New appointment Booking!'
    body = f"""
    New Appointment Booked!

    Name: {user_name}
    Email: {user_email}
    Appointment Date: {appointment_date}
    Time Slot: {time_slot}
    Duration: {duration} minutes
    Timezone: {timezone}
    Notes: {notes}
    """

    sender_email = os.environ.get('EMAIL_USER')
    receiver_email = os.environ.get('EMAIL_USER')
    password = os.environ.get('EMAIL_PASS')

    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg.set_content(body)

        # Send Email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, password)
            smtp.send_message(msg)

        # ✅ Return JSON success message (instead of flash)
        return jsonify({"success": True, "message": "Appointment booked successfully!"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "message": "Failed to send email!"}), 500


@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/cookies')
def cookies():
    return render_template('cookiepolicy.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/wellness')
def wellness():
    return render_template('wellness_tools.html')

@app.route('/gaming')
def gaming():
    return render_template('gaming.html')

@app.route('/emotion_history')
def emotion_history():
    return render_template('emotion_timeline.html')

@app.route('/mood_transition')
def mood_transition():
    return render_template('mood_transition.html')


@app.route('/save_favorite', methods=['POST'])
def save_favorite():
    data = request.json
    song = {
        'title': data['title'],
        'artist': data['artist'],
        'link': data['link']
    }

    # Load current list
    if os.path.exists('favorites.json'):
        with open('favorites.json', 'r') as f:
            favorites = json.load(f)
    else:
        favorites = []

    # Avoid duplicates
    if not any(f['title'] == song['title'] and f['artist'] == song['artist'] for f in favorites):
        favorites.append(song)
        with open('favorites.json', 'w') as f:
            json.dump(favorites, f, indent=4)
        return jsonify({'status': 'saved'})
    else:
        return jsonify({'status': 'duplicate'})

# Get favorites
@app.route('/get_favorites')
def get_favorites():
    if os.path.exists('favorites.json'):
        with open('favorites.json', 'r') as f:
            return jsonify(json.load(f))
    return jsonify([])

@app.route('/get_neutral_songs', methods=['GET'])
def get_neutral_songs():
    neutral_songs = [
        {"track": "On Top Of The World", "artist": "Imagine Dragons", "mood": "Neutral"},
        {"track": "Counting Stars", "artist": "OneRepublic", "mood": "Neutral"},
        {"track": "Let Her Go", "artist": "Passenger", "mood": "Neutral"},
        {"track": "Photograph", "artist": "Ed Sheeran", "mood": "Neutral"},
        {"track": "Paradise", "artist": "Coldplay", "mood": "Neutral"},
        {"track": "Stay", "artist": "Zedd & Alessia Cara", "mood": "Neutral"},
        {"track": "Happier", "artist": "Marshmello & Bastille", "mood": "Neutral"},
        {"track": "Closer", "artist": "The Chainsmokers & Halsey", "mood": "Neutral"},
        {"track": "Waves", "artist": "Dean Lewis", "mood": "Neutral"},
        {"track": "Memories", "artist": "Maroon 5", "mood": "Neutral"}
    ]
    
    return jsonify({"songs": neutral_songs})


@app.route('/remove_favorite', methods=['POST'])
def remove_favorite():
    data = request.get_json()
    title = data.get('title')
    artist = data.get('artist')
    
    # Load existing favorites
    if os.path.exists('favorites.json'):
        with open('favorites.json', 'r') as f:
            favorites = json.load(f)
    else:
        favorites = []

    # Remove the song
    updated = [song for song in favorites if not (song['title'] == title and song['artist'] == artist)]

    with open('favorites.json', 'w') as f:
        json.dump(updated, f, indent=4)

    return jsonify({'success': True})



def fetch_soundcloud(song, artist):
    """Try to fetch from SoundCloud"""
    filename = f"{clean_filename(artist)}_{clean_filename(song)}.mp3"
    output_path = os.path.join(MUSIC_DIR, filename)
    
    query = f"scsearch:{artist} - {song}"
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(MUSIC_DIR, os.path.splitext(filename)[0] + '.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([query])
        
        if os.path.exists(output_path):
            return output_path
    except Exception as e:
        print(f"SoundCloud error: {e}")
        return None
    
def fetch_youtube_playlist_search(song, artist):
    """Search for the song in popular music playlists"""
    filename = f"{clean_filename(artist)}_{clean_filename(song)}.mp3"
    output_path = os.path.join(MUSIC_DIR, filename)
    
    # Try to find the song in popular playlists
    playlist_queries = [
        f"top hits {artist}",
        f"{artist} essentials",
        f"best of {artist}"
    ]
    
    for query in playlist_queries:
        try:
            ydl_opts = {
                'quiet': True,
                'extract_flat': True,
                'force_generic_extractor': False
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                results = ydl.extract_info(f"ytsearch1:{query} playlist", download=False)
                
                if not results or not results.get('entries'):
                    continue
                    
                playlist_url = results['entries'][0].get('url')
                if not playlist_url:
                    continue
                
                # Now get the playlist contents
                playlist_opts = {
                    'quiet': True,
                    'extract_flat': True,
                    'ignoreerrors': True,
                }
                
                with yt_dlp.YoutubeDL(playlist_opts) as ydl_playlist:
                    playlist_results = ydl_playlist.extract_info(playlist_url, download=False)
                    
                    if not playlist_results or not playlist_results.get('entries'):
                        continue
                    
                    # Look for our song in the playlist
                    song_lower = song.lower()
                    for entry in playlist_results['entries']:
                        if not entry:
                            continue
                            
                        entry_title = entry.get('title', '').lower()
                        if song_lower in entry_title:
                            # Found the song! Download it
                            song_url = entry.get('url')
                            if not song_url:
                                continue
                                
                            if not song_url.startswith('http'):
                                song_url = f"https://www.youtube.com/watch?v={song_url}"
                            
                            download_opts = {
                                'format': 'bestaudio/best',
                                'outtmpl': os.path.join(MUSIC_DIR, os.path.splitext(filename)[0]),
                                'postprocessors': [{
                                    'key': 'FFmpegExtractAudio',
                                    'preferredcodec': 'mp3',
                                    'preferredquality': '192',
                                }],
                                'quiet': True,
                            }
                            
                            with yt_dlp.YoutubeDL(download_opts) as ydl_download:
                                ydl_download.download([song_url])
                                
                            if os.path.exists(output_path):
                                return output_path
        except Exception as e:
            print(f"Playlist search error: {e}")
            continue
    
    return None





def filter_music_videos(videos, song, artist):
    """Filter videos to prioritize official music content"""
    if not videos:
        return []
    
    scored_videos = []
    song_lower = song.lower()
    artist_lower = artist.lower()
    
    for video in videos:
        if not video:
            continue
            
        title = video.get('title', '').lower()
        channel = video.get('channel', '').lower()
        
        score = 0
        
        # Check if it's a music video
        if song_lower in title and artist_lower in title:
            score += 10
        elif song_lower in title:
            score += 5
        
        # Prefer official artist channels
        if artist_lower in channel:
            score += 8
        
        # Prefer videos with "official" or "audio" in the title
        if "official" in title:
            score += 5
        if "audio" in title:
            score += 3
            
        # Avoid instrumental or cover versions
        if "instrumental" in title or "karaoke" in title:
            score -= 10
        if "cover" in title and artist_lower not in title:
            score -= 5
        if 'trailer' in title or 'teaser' in title or 'preview' in title:
            continue

        # Prefer videos with appropriate duration (3-8 minutes typically)
        duration = video.get('duration')
        if not duration or duration < 60:  # Less than 1 min
            continue  # skip short/incomplete videos

            
        if score > 0:
            scored_videos.append((video, score))
    
    # Sort by score
    scored_videos.sort(key=lambda x: x[1], reverse=True)
    return [v[0] for v in scored_videos]


def get_youtube_info(query, max_results=5):
    """Get info about YouTube videos without downloading"""
    ydl_opts = {
        'quiet': True,
        'extract_flat': False,
        'force_generic_extractor': False,
        'ignoreerrors': True,
        'verbose': True,
        'no_warnings': False,
        'noplaylist': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            return results.get('entries', [])
    except Exception as e:
        print(f"YouTube search error: {e}")
        return []



def fetch_youtube_smart(song, artist):
    """Enhanced YouTube downloader with smarter search and filtering"""
    filename = f"{clean_filename(artist)}_{clean_filename(song)}.mp3"
    output_path = os.path.join(MUSIC_DIR, filename)
    
    # Generate multiple search queries
    queries = [
        f"{artist} - {song} official audio",
        f"{artist} - {song} official",
        f"{artist} {song} official audio",
        f"{song} by {artist} audio"
    ]
    
    for query in queries:
        # First get video info for better filtering
        videos = get_youtube_info(query)
        filtered_videos = filter_music_videos(videos, song, artist)
        
        if not filtered_videos:
            continue
            
        # Get the best video URL
        video_url = filtered_videos[0].get('url') or filtered_videos[0].get('id')
        if not video_url:
            continue
            
        if not video_url.startswith('http'):
            video_url = f"https://www.youtube.com/watch?v={video_url}"
        
        # Download the best match
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(MUSIC_DIR, os.path.splitext(filename)[0]),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            if os.path.exists(output_path):
                return output_path
        except Exception as e:
            print(f"Download error: {e}")
            continue
    
    return None

def fetch_with_retries(song, artist, max_retries=3):
    """Try multiple methods with retries and backoff"""
    methods = [
        fetch_youtube_smart,
        fetch_soundcloud,
        fetch_youtube_playlist_search
    ]
    
    for method in methods:
        for attempt in range(max_retries):
            try:
                result = method(song, artist)
                if result:
                    return result
                
                # Add a small delay between retries
                time.sleep(1 + random.random())
            except Exception as e:
                print(f"Error in {method.__name__}: {e}")
                continue
    
    return None

MUSIC_DIR = os.path.join('static', 'music')

# Ensure music folder exists
os.makedirs(MUSIC_DIR, exist_ok=True)

@app.route('/get_audio')
def get_audio():
    song = request.args.get('song')
    artist = request.args.get('artist')
    
    if not song or not artist:
        return jsonify({
            'error': 'Missing song or artist parameter'
        }), 400
    
    song_file_name = f"{clean_filename(artist)}_{clean_filename(song)}.mp3"

    # 1. Check if already downloaded locally
    local_path = os.path.join(MUSIC_DIR, song_file_name)
    if os.path.exists(local_path):
        return jsonify({
            'audio_link': url_for('static', filename=f'music/{song_file_name}'),
            'source': 'local'
        })

    
    # 3. Fallback to YouTube using the methods from the first document
    try:
        # Try multiple methods to fetch the song
        result_path = fetch_with_retries(song, artist, max_retries=2)
        
        if result_path and os.path.exists(result_path):
            filename = os.path.basename(result_path)
            return jsonify({
                'track': song,
                'artist': artist,
                'audio_link': url_for('static', filename=f'music/{filename}'),
                'source': 'youtube'
            })
        else:
            return jsonify({
                'error': 'Failed to fetch audio from all sources'
            }), 404
    except Exception as e:
        print(f"YouTube fetch error: {e}")
        return jsonify({
            'error': f'Error fetching audio: {str(e)}'
        }), 500



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
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
########################################################################### 📸 CAMERA CONTROL ROUTES ############################################################################## 
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

running = True  # To control video streaming


# @app.route('/toggle_camera', methods=['POST'])
# def toggle_camera():
#     """Pauses or resumes the camera feed based on user activity."""
#     global cap
#     action = request.json.get("action")  # Expecting 'pause' or 'resume'

#     try:
#         if action == "pause":
#             if cap is not None and cap.isOpened():
#                 print("⏸️ Camera paused.")
#             return jsonify({"status": "Camera paused"}), 200

#         elif action == "resume":
#             if cap is None or not cap.isOpened():
#                 cap = cv2.VideoCapture(0)  # Restart if needed
#             print("▶️ Camera resumed.")
#             return jsonify({"status": "Camera resumed"}), 200

#         else:
#             return jsonify({"error": "Invalid action"}), 400

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

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
    cleanup()  # Perform cleanup action
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
    

@app.route('/emotion_timeline', methods=['GET'])
def emotion_timeline():
    """Returns emotion data over time for visualization"""
    try:
        # Get saved emotion records from the log
        timeline_data = list(emotion_log)
        
        # If we have chat data, add that too
        if os.path.exists("chat_results.json"):
            with open("chat_results.json", "r") as f:
                chat_data = json.load(f)
                if "emotion_scores" in chat_data:
                    # Add timestamps to chat emotions
                    chat_emotions = {
                        "timestamp": chat_data["timestamp"],
                        "emotions": chat_data["emotion_scores"]
                    }
                    timeline_data.append(chat_emotions)
        
        return jsonify({"timeline": timeline_data})
    except Exception as e:
        print(f"Error generating emotion timeline: {str(e)}")
        return jsonify({"error": "Failed to generate emotion timeline"}), 500


@app.route('/transition_playlist', methods=['POST'])
def transition_playlist():
    """Creates a playlist that transitions from current emotion to target emotion"""
    try:
        data = request.json
        start_emotion = data.get('start_emotion', '')
        target_emotion = data.get('target_emotion', '')
        
        if not start_emotion or not target_emotion:
            return jsonify({"error": "Start and target emotions are required"}), 400
            
        # Get transition playlist
        playlist = generate_transition_playlist(start_emotion, target_emotion)
        
        return jsonify({"playlist": playlist})
    except Exception as e:
        print(f"Error generating transition playlist: {str(e)}")
        return jsonify({"error": "Failed to generate transition playlist"}), 500

def generate_transition_playlist(start_emotion, target_emotion):
    """
    Generate a playlist that gradually transitions from one emotion to another.
    
    Args:
        start_emotion: The starting emotion
        target_emotion: The target emotion
    
    Returns:
        A list of song recommendations for the transition
    """
    # Map emotions to arousal and valence values (psychological dimensions of emotion)
    emotion_mapping = {
        "angry": {"arousal": 0.9, "valence": 0.2},
        "disgusted": {"arousal": 0.6, "valence": 0.2},
        "fearful": {"arousal": 0.7, "valence": 0.1},
        "happy": {"arousal": 0.7, "valence": 0.9},
        "neutral": {"arousal": 0.5, "valence": 0.5},
        "sad": {"arousal": 0.3, "valence": 0.2},
        "surprised": {"arousal": 0.8, "valence": 0.7},
        "calm": {"arousal": 0.2, "valence": 0.7}
    }
    
    # Define song database with arousal and valence values
    # In a real app, this would come from a database or music API
    song_database = [
        {"track": "Thunderstruck", "artist": "AC/DC", "arousal": 0.9, "valence": 0.7},
        {"track": "Someone Like You", "artist": "Adele", "arousal": 0.3, "valence": 0.3},
        {"track": "Happy", "artist": "Pharrell Williams", "arousal": 0.8, "valence": 0.9},
        {"track": "Relaxing Piano", "artist": "Various Artists", "arousal": 0.2, "valence": 0.7},
        {"track": "Seven Nation Army", "artist": "The White Stripes", "arousal": 0.7, "valence": 0.6},
        {"track": "Sweet Child O' Mine", "artist": "Guns N' Roses", "arousal": 0.8, "valence": 0.7},
        {"track": "Bohemian Rhapsody", "artist": "Queen", "arousal": 0.6, "valence": 0.6},
        {"track": "Let It Go", "artist": "Idina Menzel", "arousal": 0.7, "valence": 0.8},
        {"track": "Hello", "artist": "Adele", "arousal": 0.4, "valence": 0.3},
        {"track": "Dancing Queen", "artist": "ABBA", "arousal": 0.7, "valence": 0.9},
        {"track": "Piano Sonata No. 14", "artist": "Beethoven", "arousal": 0.3, "valence": 0.4},
        {"track": "In Da Club", "artist": "50 Cent", "arousal": 0.8, "valence": 0.7},
        {"track": "Boulevard of Broken Dreams", "artist": "Green Day", "arousal": 0.6, "valence": 0.4},
        {"track": "Eye of the Tiger", "artist": "Survivor", "arousal": 0.8, "valence": 0.8},
        {"track": "Hallelujah", "artist": "Leonard Cohen", "arousal": 0.2, "valence": 0.4},
        {"track": "Shape of You", "artist": "Ed Sheeran", "arousal": 0.7, "valence": 0.8},
        {"track": "Fix You", "artist": "Coldplay", "arousal": 0.5, "valence": 0.5},
        {"track": "We Will Rock You", "artist": "Queen", "arousal": 0.8, "valence": 0.7},
        {"track": "Numb", "artist": "Linkin Park", "arousal": 0.7, "valence": 0.3},
        {"track": "All of Me", "artist": "John Legend", "arousal": 0.4, "valence": 0.6}
    ]
    
    # Get start and target emotion values
    start_values = emotion_mapping.get(start_emotion.lower(), {"arousal": 0.5, "valence": 0.5})
    target_values = emotion_mapping.get(target_emotion.lower(), {"arousal": 0.5, "valence": 0.5})
    
    # Calculate number of steps for transition (5 songs)
    steps = 5
    
    # Calculate step sizes
    arousal_step = (target_values["arousal"] - start_values["arousal"]) / steps
    valence_step = (target_values["valence"] - start_values["valence"]) / steps
    
    transition_playlist = []
    
    # Generate playlist with songs that gradually transition
    for i in range(steps + 1):
        current_arousal = start_values["arousal"] + (arousal_step * i)
        current_valence = start_values["valence"] + (valence_step * i)
        
        # Find song with closest match to current values
        best_match = None
        best_distance = float('inf')
        
        for song in song_database:
            # Calculate Euclidean distance in arousal-valence space
            distance = ((song["arousal"] - current_arousal) ** 2 + 
                        (song["valence"] - current_valence) ** 2) ** 0.5
            
            if distance < best_distance:
                best_distance = distance
                best_match = song
        
        if best_match and best_match not in transition_playlist:
            # Add emotion info to the song
            song_with_info = {
                "track": best_match["track"],
                "artist": best_match["artist"],
                "arousal": best_match["arousal"],
                "valence": best_match["valence"],
                "transition_step": i,
                "arousal_target": current_arousal,
                "valence_target": current_valence
            }
            transition_playlist.append(song_with_info)
    
    return transition_playlist

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
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
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════







#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
@app.route('/process_results', methods=['GET', 'POST'])
def process_results():
    try:
        # Calculate final averaged emotions
        final_emotions = calculate_final_emotions()
        
        # Check for errors in emotion calculationk
        if not isinstance(final_emotions, dict):
            return jsonify({"error": "Emotion processing failed, invalid data received."}), 500
        
        if "error" in final_emotions:
            return jsonify({"error": final_emotions["error"]}), 400

        # Fetch recommended songs
        songs = recommend_songs("final_average_emotions.json")

        # Ensure songs is a list
        if not isinstance(songs, list):
            songs = []  # Fallback to empty list if invalid

        # Log response for debugging
        # print("✅ Processed Emotions:", final_emotions)
        # print("✅ Recommended Songs:", songs)

        return jsonify({"final_emotions": final_emotions, "recommended_songs": songs})

    except Exception as e:
        print(f"❌ Error in /process_results: {str(e)}")
        return jsonify({"error": "An unexpected error occurred while processing results."}), 500

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════



#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Only start background thread in development
    if os.environ.get('FLASK_ENV') != 'production':
        background_thread = threading.Thread(target=update_all_in_background, daemon=True)
        background_thread.start()
    
    try:
        port = int(os.environ.get('PORT', 10000))
        # Use gunicorn in production (configured in Render.yaml)
        if os.environ.get('FLASK_ENV') == 'production':
            app.run(host='0.0.0.0', port=port)
        else:
            # Debug mode for local development
            app.run(host='0.0.0.0', port=port, debug=True)
    except KeyboardInterrupt:
        print("\n🔴 Server stopped manually.")
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

        

