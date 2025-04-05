import os
import pandas as pd
import numpy as np
import json
import re
import pickle

MODEL_PATH = "models/ensemble_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
SCALER_PATH = "models/scaler.pkl"
DATASET_PATH = "datasets/therapeutic_music_enriched.csv"

with open(MODEL_PATH, "rb") as f:
    ensemble_model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv(DATASET_PATH)

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


def process_emotions(emotion_file):
    """Reads JSON emotion file, extracts values, and converts to audio features."""

    with open(emotion_file, "r") as file:
        emotions = json.load(file)
    
    print(f"\n📂 Loaded JSON Content:\n{json.dumps(emotions, indent=4)}\n")

    emotions = emotions["final_average_emotions"]

    print(f"DEBUG: extracted emotions -> {emotions}")
    print(f"DEBUG: type of each value -> {[type(v) for v in emotions.values()]}")

    emotion_scores = {emotion: float(score) for emotion, score in emotions.items()}



    
    print(f"\n📊 Extracted Emotion Scores:\n{emotion_scores}\n")

    weighted_audio_features = np.zeros(len(list(EMOTION_TO_AUDIO.values())[0]))  




    print("\n🛠 Debugging Weighted Audio Features Calculation:")
    for emotion, weight in emotion_scores.items():
        if emotion in EMOTION_TO_AUDIO:
            contribution = np.array(EMOTION_TO_AUDIO[emotion]) * weight
            weighted_audio_features += contribution
            print(f"🔹 {emotion} ({weight}): {contribution}")

    weighted_audio_features = scaler.transform([weighted_audio_features])[0]

    print(f"\n🎵 Final Normalized Audio Features (Input to Model):\n{weighted_audio_features}\n")

    return weighted_audio_features.reshape(1, -1), emotion_scores


# === 📌 Mood Prediction & Song Recommendation ===
def recommend_songs(emotion_file):
    """Predicts mood based on emotions and recommends matching songs."""

    emotion_vector, emotion_scores = process_emotions(emotion_file)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/pca.pkl", "rb") as f:
        pca = pickle.load(f)

    emotion_vector = scaler.transform(emotion_vector.reshape(1, -1))  # Standardize
    emotion_vector = pca.transform(emotion_vector)  # Reduce dimensions

    mood_probs = ensemble_model.predict_proba(emotion_vector)

    print(f"\n🔍 Model Confidence Scores for Moods: {dict(zip(le.classes_, mood_probs[0]))}\n")

    predicted_mood_index = ensemble_model.predict(emotion_vector)[0]
    predicted_mood = le.inverse_transform([predicted_mood_index])[0]

    print(f"\n🎯 Initial Predicted Mood (Model Output): {predicted_mood}\n")

    dominant_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    mapped_moods = set()
    for emotion, _ in dominant_emotions:
        mapped_moods.update(EMOTION_TO_MOOD.get(emotion, ["Neutral"]))

    print(f"\n🎭 Dominant Emotions: {dominant_emotions} → Adjusted Moods: {mapped_moods}\n")

    if predicted_mood not in mapped_moods:
        predicted_mood = list(mapped_moods)[0]  # Take the first mapped mood

    print(f"\n🎯 Final Adjusted Mood: {predicted_mood}\n")

    filtered_songs = df[df["Mood_Label"] == predicted_mood]

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

        average_emotions = {emotion: confidence / 100 for emotion, confidence in average_emotions.items()}

        dominant_emotion_dict = {emotion: 0.0 for emotion in average_emotions}
        dominant_emotion_dict[dominant_emotion] = 1.0  # 100% as 1.0

        # Convert both to DataFrames
        df1 = pd.DataFrame([dominant_emotion_dict])  # From chat_results.json
        df2 = pd.DataFrame([average_emotions])  # From emotion_log.json

        final_average_df = pd.concat([df1, df2], ignore_index=True)
        final_average_emotions = final_average_df.mean().round(4)  # Keep two decimal places

        final_emotion_result = final_average_emotions.to_dict()

        with open("final_average_emotions.json", "w") as f:
            json.dump({"final_average_emotions": final_emotion_result}, f, indent=4)

        return final_emotion_result

    except Exception as e:
        return {"error": str(e)}

calculate_final_emotions()
emotion_file_path = "final_average_emotions.json"
print(recommend_songs(emotion_file_path))
