"""
RECOMMENDATION MICROSERVICE  
Extracted from app.py lines 161-338
Contains music recommendation logic and emotion processing
"""
import pickle
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional

class RecommendationService:
    def __init__(self):
        # Load models (from app.py:114-130)
        try:
            with open("models/ensemble_model.pkl", "rb") as f:
                self.ensemble_model = pickle.load(f)
            with open("models/label_encoder.pkl", "rb") as f:
                self.le = pickle.load(f)
            with open("models/scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open("models/pca.pkl", "rb") as f:
                self.pca = pickle.load(f)
            
            # Load music dataset
            self.df = pd.read_csv("datasets/therapeutic_music_enriched.csv")
            print("✅ Models and dataset loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            # Initialize with None for graceful degradation
            self.ensemble_model = None
            self.le = None
            self.scaler = None  
            self.pca = None
            self.df = None
            
        # Emotion to audio feature mapping (from app.py:133-144)
        self.EMOTION_TO_AUDIO = {
            "angry":       [0.4, 0.9, 5, -5.0, 0.3, 0.1, 0.0, 0.6, 0.2, 120],
            "disgust":     [0.3, 0.7, 6, -7.0, 0.5, 0.2, 0.0, 0.5, 0.3, 100],
            "fear":        [0.2, 0.6, 7, -10.0, 0.6, 0.3, 0.1, 0.4, 0.1, 80],
            "happy":       [0.8, 0.9, 8, -3.0, 0.2, 0.4, 0.0, 0.5, 0.9, 130],
            "sad":         [0.3, 0.4, 4, -12.0, 0.4, 0.6, 0.1, 0.3, 0.1, 70],
            "surprise":    [0.7, 0.8, 9, -6.0, 0.4, 0.3, 0.0, 0.6, 0.7, 125],
            "neutral":     [0.5, 0.5, 5, -8.0, 0.3, 0.4, 0.0, 0.4, 0.5, 110],
        }
        
        # Emotion to mood mapping (from app.py:146-158)  
        self.EMOTION_TO_MOOD = {
            "angry":       ["Relaxation", "Serenity"],
            "disgust":     ["Calm", "Neutral"], 
            "fear":        ["Reassurance", "Serenity"],
            "happy":       ["Excitement", "Optimism"],
            "sad":         ["Optimism", "Upliftment"],
            "surprise":    ["Excitement", "Joy"],
            "neutral":     ["Serenity", "Neutral"],
        }
        
        print("✅ Recommendation Service initialized")

    # EXTRACTED FROM app.py:161-202 (EXACT COPY)
    def process_emotions(self, emotion_file):
        """Process emotion data from file and convert to feature vector"""
        try:
            if os.path.exists(emotion_file):
                with open(emotion_file, "r") as file:
                    emotions = json.load(file)
                    
                # Handle nested structure
                if "final_average_emotions" in emotions:
                    emotions = emotions["final_average_emotions"]
                elif "average_emotions" in emotions:
                    emotions = emotions["average_emotions"]
                else:
                    emotions = emotions
                    
                # Ensure emotions is a dictionary (handle list case)
                if isinstance(emotions, list):
                    if emotions:
                        emotions = emotions[0]  # Take first item if it's a list
                    else:
                        emotions = {}  # Empty list becomes empty dict

                # Normalize emotion scores
                emotion_scores = {emotion: float(score) / 100.0 for emotion, score in emotions.items()}
                
                # Calculate weighted audio features based on emotions
                weighted_audio_features = np.zeros(len(list(self.EMOTION_TO_AUDIO.values())[0]))
                
                for emotion, weight in emotion_scores.items():
                    if emotion in self.EMOTION_TO_AUDIO:
                        # Weighted contribution of each emotion to audio features
                        contribution = np.array(self.EMOTION_TO_AUDIO[emotion]) * weight
                        weighted_audio_features += contribution
                
                return weighted_audio_features, emotion_scores
                
            else:
                print(f"Emotion file not found: {emotion_file}")
                return None, None
                
        except Exception as e:
            print(f"Error processing emotions: {e}")
            return None, None

    # EXTRACTED FROM app.py:206-338 (EXACT COPY)
    def recommend_songs(self, emotion_file):
        """Predicts mood based on emotions and recommends matching songs"""
        import logging
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        
        try:
            # Check if models are loaded
            if self.ensemble_model is None:
                logger.error("Models not loaded properly")
                return self._get_fallback_songs()
                
            # Get Emotion-Based Features
            emotion_vector, emotion_scores = self.process_emotions(emotion_file)
            
            if emotion_vector is None:
                logger.warning("Could not process emotions, using fallback")
                return self._get_fallback_songs()
            
            # Apply transformations with validation
            if emotion_vector.ndim == 1:
                emotion_vector = emotion_vector.reshape(1, -1)
                
            # Standardize
            emotion_vector_scaled = self.scaler.transform(emotion_vector)
            # Reduce dimensions
            emotion_vector_pca = self.pca.transform(emotion_vector_scaled)

            # Debug confidence scores if needed
            mood_probs = self.ensemble_model.predict_proba(emotion_vector_pca)
            logger.debug(f"Model Confidence Scores: {dict(zip(self.le.classes_, mood_probs[0]))}")

            # Predict Mood
            predicted_mood_index = self.ensemble_model.predict(emotion_vector_pca)[0]
            predicted_mood = self.le.inverse_transform([predicted_mood_index])[0]
            logger.info(f"Initial Predicted Mood: {predicted_mood}")

            # Find Top 2 Dominant Emotions with validation
            if not emotion_scores:
                logger.warning("No emotion scores found, using default neutral mood")
                dominant_emotions = [("neutral", 1.0)]
            else:
                dominant_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:2]
                
            mapped_moods = set()
            for emotion, score in dominant_emotions:
                mapped_moods.update(self.EMOTION_TO_MOOD.get(emotion, ["Neutral"]))
            
            logger.info(f"Dominant Emotions: {dominant_emotions} → Adjusted Moods: {mapped_moods}")

            # Adjust Mood If Necessary
            if predicted_mood not in mapped_moods and mapped_moods:
                predicted_mood = list(mapped_moods)[0]  # Take the first mapped mood
                logger.info(f"Adjusted mood to: {predicted_mood}")

            # Filter Songs Based on Mood with validation
            try:
                filtered_songs = self.df[self.df["Mood_Label"] == predicted_mood].copy()
            except KeyError:
                logger.error("Missing required column 'Mood_Label' in dataframe")
                return self._get_fallback_songs()

            # Use Fallback Moods If No Songs Found
            if filtered_songs.empty and mapped_moods:
                logger.info(f"No songs found for {predicted_mood}, trying mapped moods: {mapped_moods}")
                filtered_songs = self.df[self.df["Mood_Label"].isin(mapped_moods)].copy()

            # Final Fallback to Neutral if Still Empty
            if filtered_songs.empty:
                logger.warning("No songs found for any mapped mood, falling back to Neutral")
                filtered_songs = self.df[self.df["Mood_Label"] == "Neutral"].copy()
                
                # If still empty, return fallback songs
                if filtered_songs.empty:
                    logger.error("No songs found for any mood category, including Neutral")
                    return self._get_fallback_songs()

            # Select Up to 10 Songs with duplicate handling
            required_columns = ["Track Name", "Artist Name", "Mood_Label"]
            if not all(col in filtered_songs.columns for col in required_columns):
                logger.error(f"Missing required columns in dataframe. Required: {required_columns}")
                return self._get_fallback_songs()
                
            try:
                # Remove duplicates based on Track Name + Artist Name
                filtered_songs = filtered_songs.drop_duplicates(subset=["Track Name", "Artist Name"])
                
                # Select up to 10 songs randomly
                if len(filtered_songs) > 10:
                    recommended_songs = filtered_songs.sample(10)
                else:
                    recommended_songs = filtered_songs
                
                # Convert to list of dictionaries
                song_list = []
                for _, row in recommended_songs.iterrows():
                    song_dict = {
                        "track": str(row["Track Name"]),
                        "artist": str(row["Artist Name"]), 
                        "mood": str(row["Mood_Label"])
                    }
                    song_list.append(song_dict)
                
                logger.info(f"Successfully recommended {len(song_list)} songs for mood: {predicted_mood}")
                return song_list
                
            except Exception as e:
                logger.error(f"Error processing song selection: {e}")
                return self._get_fallback_songs()
                
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return self._get_fallback_songs()

    def _get_fallback_songs(self):
        """Get fallback songs when recommendation fails"""
        return [
            {"track": "On Top Of The World", "artist": "Imagine Dragons", "mood": "Neutral"},
            {"track": "Counting Stars", "artist": "OneRepublic", "mood": "Neutral"},
            {"track": "Let Her Go", "artist": "Passenger", "mood": "Neutral"}, 
            {"track": "Photograph", "artist": "Ed Sheeran", "mood": "Neutral"},
            {"track": "Paradise", "artist": "Coldplay", "mood": "Neutral"},
            {"track": "Viva La Vida", "artist": "Coldplay", "mood": "Neutral"},
            {"track": "Stressed Out", "artist": "Twenty One Pilots", "mood": "Neutral"},
            {"track": "Believer", "artist": "Imagine Dragons", "mood": "Neutral"},
            {"track": "Thunder", "artist": "Imagine Dragons", "mood": "Neutral"},
            {"track": "Shape of You", "artist": "Ed Sheeran", "mood": "Neutral"},
        ]
        
    def calculate_final_emotions(self, chat_file: str = "chat_results.json", 
                                emotion_file: str = "emotion_log.json") -> Dict:
        """Calculate final emotions from chat and emotion data - compatible with emotion service"""
        try:
            # Load JSON files
            with open(chat_file, "r") as f1, open(emotion_file, "r") as f2:
                chat_data = json.load(f1)
                emotion_log_data = json.load(f2)

            # Extract dominant emotion from chat_results.json
            dominant_emotion = chat_data["dominant_emotion"]

            # Handle possible list structure in emotion_log.json
            if isinstance(emotion_log_data["average_emotions"], list):
                average_emotions = emotion_log_data["average_emotions"][0]
            else:
                average_emotions = emotion_log_data["average_emotions"]

            # Convert percentages to decimal
            average_emotions = {emotion: confidence / 100 for emotion, confidence in average_emotions.items()}

            # Assign 100% confidence to the dominant emotion from chat
            dominant_emotion_dict = {emotion: 0.0 for emotion in average_emotions}
            dominant_emotion_dict[dominant_emotion] = 1.0

            # Convert both to DataFrames
            df1 = pd.DataFrame([dominant_emotion_dict])
            df2 = pd.DataFrame([average_emotions])

            # Combine and compute the average
            final_average_df = pd.concat([df1, df2], ignore_index=True)
            final_average_emotions = final_average_df.mean().round(4)

            # Convert to dictionary
            final_emotion_result = final_average_emotions.to_dict()

            # Save to JSON
            with open("final_average_emotions.json", "w") as f:
                json.dump({"final_average_emotions": final_emotion_result}, f, indent=4)

            return final_emotion_result

        except Exception as e:
            print(f"Error calculating final emotions: {e}")
            return {"error": str(e)}

    def cleanup(self):
        """Cleanup service resources"""
        print("🧹 Recommendation service cleanup complete")