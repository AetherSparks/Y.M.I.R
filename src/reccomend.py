import os
import pandas as pd
import numpy as np
import json
import re
import pickle

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
    
    print(f"\n📂 Loaded JSON Content:\n{json.dumps(emotions, indent=4)}\n")

    # ✅ Fix: Extract the correct dictionary
    emotions = emotions["final_average_emotions"]

    # ✅ Debugging Print (to verify)
    print(f"DEBUG: extracted emotions -> {emotions}")
    print(f"DEBUG: type of each value -> {[type(v) for v in emotions.values()]}")

    # ✅ Now, this should work fine
    emotion_scores = {emotion: float(score) for emotion, score in emotions.items()}



    
    print(f"\n📊 Extracted Emotion Scores:\n{emotion_scores}\n")

    weighted_audio_features = np.zeros(len(list(EMOTION_TO_AUDIO.values())[0]))  




    print("\n🛠 Debugging Weighted Audio Features Calculation:")
    for emotion, weight in emotion_scores.items():
        if emotion in EMOTION_TO_AUDIO:
            contribution = np.array(EMOTION_TO_AUDIO[emotion]) * weight
            weighted_audio_features += contribution
            print(f"🔹 {emotion} ({weight}): {contribution}")

    # ✅ Normalize Features Before Model Input
    weighted_audio_features = scaler.transform([weighted_audio_features])[0]

    print(f"\n🎵 Final Normalized Audio Features (Input to Model):\n{weighted_audio_features}\n")

    return weighted_audio_features.reshape(1, -1), emotion_scores


# === 📌 Mood Prediction & Song Recommendation ===
def recommend_songs(emotion_file):
    """Predicts mood based on emotions and recommends matching songs."""

    # ✅ Get Emotion-Based Features
    emotion_vector, emotion_scores = process_emotions(emotion_file)
    mood_probs = ensemble_model.predict_proba(emotion_vector)
    print(f"\n🔍 Model Confidence Scores for Moods: {dict(zip(le.classes_, mood_probs[0]))}\n")

    # ✅ Predict Mood
    predicted_mood_index = ensemble_model.predict(emotion_vector)[0]
    predicted_mood = le.inverse_transform([predicted_mood_index])[0]

    print(f"\n🎯 Initial Predicted Mood (Model Output): {predicted_mood}\n")

    # ✅ Find Top 2 Dominant Emotions
    dominant_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    mapped_moods = set()
    for emotion, _ in dominant_emotions:
        mapped_moods.update(EMOTION_TO_MOOD.get(emotion, ["Neutral"]))

    print(f"\n🎭 Dominant Emotions: {dominant_emotions} → Adjusted Moods: {mapped_moods}\n")

    # ✅ Adjust Mood If Necessary
    if predicted_mood not in mapped_moods:
        predicted_mood = list(mapped_moods)[0]  # Take the first mapped mood

    print(f"\n🎯 Final Adjusted Mood: {predicted_mood}\n")

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


    song_list = "\n".join(
        [f"🎵 {row['Track Name']} by {row['Artist Name']} ({row['Mood_Label']})" for _, row in recommended_songs.iterrows()]
    )

    return f"\n🎶 Recommended Songs:\n{song_list}"

# === 🚀 Run Recommendation System ===
emotion_file_path = "final_average_emotions.json"
print(recommend_songs(emotion_file_path))
