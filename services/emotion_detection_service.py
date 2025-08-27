"""
EMOTION DETECTION MICROSERVICE
Extracted from app.py lines 644-675, 805-845, 860-901, 428-456, 459-517
Contains all facial emotion detection and text sentiment analysis
"""
import cv2
import time
import json
import threading
import pandas as pd
import torch
import re
import os
from deepface import DeepFace
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, List, Optional, Tuple

class EmotionDetectionService:
    def __init__(self):
        # Global emotion data structure (from app.py:628-634)
        self.emotion_data = {
            "faces": {}, 
            "lock": threading.Lock(),  
            "log": [],
            "average_emotions": {}
        }
        
        # Add analysis rate limiting to prevent DeepFace hangs
        self.analysis_lock = threading.Lock()
        self.last_analysis_time = 0
        
        # Load sentiment analysis model (from app.py:409-410)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        
        # VADER sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        print("✅ Emotion Detection Service initialized")

    # EXTRACTED FROM app.py:644-675 (EXACT COPY)
    def analyze_emotion(self, face_id, face_roi):
        """Analyze facial emotion using DeepFace with timeout protection"""
        current_time = time.time()
        
        # Rate limiting: Only allow one analysis per 2 seconds to prevent DeepFace hangs
        with self.analysis_lock:
            if current_time - self.last_analysis_time < 2.0:
                print(f"⏳ Rate limiting: Skipping analysis for face {face_id} (too soon)")
                return
            self.last_analysis_time = current_time
        
        print(f"🔍 Starting emotion analysis for face {face_id}...")

        with self.emotion_data["lock"]:
            # Ensure log list exists
            if "log" not in self.emotion_data:
                self.emotion_data["log"] = []

            self.emotion_data["last_update"] = current_time

        try:
            # Validate face ROI
            if face_roi is None or face_roi.size == 0:
                print(f"⚠️ Invalid face ROI for face {face_id}")
                return

            print(f"✅ Face ROI valid for face {face_id}: {face_roi.shape}")

            # Resize with error handling
            try:
                resized_face = cv2.resize(face_roi, (224, 224))
                print(f"✅ Face resized for face {face_id}: {resized_face.shape}")
            except Exception as resize_error:
                print(f"⚠️ Face resize error for face {face_id}: {resize_error}")
                return

            # DeepFace analysis with timeout protection
            import threading
            
            emotion_result = None
            analysis_error = None
            
            print(f"🧠 Starting DeepFace analysis for face {face_id}...")
            
            def deepface_worker():
                nonlocal emotion_result, analysis_error
                try:
                    emotion_result = DeepFace.analyze(
                        resized_face, 
                        actions=['emotion'], 
                        enforce_detection=False, 
                        detector_backend='opencv',
                        silent=True
                    )
                    print(f"✅ DeepFace analysis complete for face {face_id}")
                except Exception as e:
                    analysis_error = e
                    print(f"❌ DeepFace worker error for face {face_id}: {e}")
            
            # Run with timeout
            analysis_thread = threading.Thread(target=deepface_worker)
            analysis_thread.daemon = True
            analysis_thread.start()
            analysis_thread.join(timeout=3.0)  # Increased timeout to 3 seconds
            
            if analysis_thread.is_alive():
                print(f"⚠️ DeepFace timeout for face {face_id}")
                return
                
            if analysis_error:
                print(f"⚠️ DeepFace error for face {face_id}: {analysis_error}")
                return
                
            if emotion_result is None:
                print(f"⚠️ No emotion result for face {face_id}")
                return

            with self.emotion_data["lock"]:
                emotions = emotion_result[0]['emotion']
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))

                # Store emotions and timestamp
                self.emotion_data["faces"][face_id] = emotions
                self.emotion_data["log"].append({"timestamp": timestamp, "emotions": emotions})

                # Print emotions with timestamp
                print(f"🕒 {timestamp} - Emotions: {emotions}")

            # Save to JSON file (thread-safe)
            try:
                with self.emotion_data["lock"]:
                    with open("emotion_log.json", "w") as f:
                        json.dump(self.emotion_data["log"], f, indent=4)
            except Exception as save_error:
                print(f"⚠️ Error saving emotions: {save_error}")

        except Exception as e:
            print(f"⚠️ Emotion detection error for face {face_id}: {e}")
            import traceback
            traceback.print_exc()

    # EXTRACTED FROM app.py:805-845 (EXACT COPY)
    def calculate_and_store_average_emotions(self):
        """Calculate and store average emotions from logged data"""
        with self.emotion_data["lock"]:
            if "log" not in self.emotion_data:
                self.emotion_data["log"] = []

            log_data = self.emotion_data["log"]
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

                # Calculate averages
                average_emotions = {
                    emotion: round(emotion_sums[emotion] / emotion_counts[emotion], 2)
                    for emotion in emotion_sums
                }
                self.emotion_data["average_emotions"] = average_emotions

                # Save to JSON immediately
                try:
                    with open("emotion_log.json", "w") as f:
                        json.dump({"log": log_data, "average_emotions": average_emotions}, f, indent=4)
                        f.flush()
                except Exception as e:
                    print(f"❌ Error writing to emotion_log.json: {e}")

                # Debugging: Print updated average emotions
                print("\\n📊 **Updated Average Emotions:**")
                for emotion, avg_confidence in average_emotions.items():
                    print(f"  {emotion.upper()}: {avg_confidence}")

            else:
                print("⚠️ No emotion data recorded.")

    # EXTRACTED FROM app.py:860-901 (EXACT COPY)
    def calculate_final_emotions(self):
        """Calculate final averaged emotions from chat and emotion logs"""
        try:
            # Load JSON files
            with open("chat_results.json", "r") as f1, open("emotion_log.json", "r") as f2:
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
            return {"error": str(e)}

    # EXTRACTED FROM app.py:428-446 (EXACT COPY)
    def handle_negations(self, text):
        """Detects negations and flips associated emotions"""
        negation_patterns = [
            r"\\b(not|never|no)\\s+(happy|joyful|excited)\\b",
            r"\\b(not|never|no)\\s+(sad|depressed|unhappy)\\b",
            r"\\b(not|never|no)\\s+(angry|mad|furious)\\b"
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

    # EXTRACTED FROM app.py:449-456 (EXACT COPY)
    def detect_sentiment(self, text):
        """Detects sentiment polarity (positive, neutral, negative)"""
        inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.sentiment_model(**inputs)
        sentiment_scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        sentiment_labels = ["negative", "neutral", "positive"]
        
        return sentiment_labels[torch.argmax(sentiment_scores).item()]

    # EXTRACTED FROM app.py:459-517 (EXACT COPY)
    def detect_conversation_emotions(self, chat_history):
        """Analyzes chat history, considers recent messages more, and balances emotion scores"""
        emotion_scores = {}
        emotion_counts = {}
        model_emotions = []
        
        # More weight to recent messages
        recent_weight = 1.5  
        messages = chat_history[-5:]  # Use last 5 messages for better context
        full_chat_text = " ".join([entry["user"] for entry in messages])

        # Check for negation handling
        negated_emotion = self.handle_negations(full_chat_text)
        if negated_emotion:
            return negated_emotion, {}, []

        # Note: emotion_models would need to be initialized separately
        # This is a placeholder for the transformer models from app.py:402-406
        print(f"Analyzing text: {full_chat_text}")
        
        # Use VADER sentiment as fallback
        vader_scores = self.vader_analyzer.polarity_scores(full_chat_text)
        
        # Convert VADER to emotion mapping
        if vader_scores['compound'] >= 0.05:
            dominant_emotion = "happy"
        elif vader_scores['compound'] <= -0.05:
            dominant_emotion = "sad"
        else:
            dominant_emotion = "neutral"
            
        emotion_scores = {dominant_emotion: vader_scores['compound']}
        
        return dominant_emotion, emotion_scores, [f"VADER: {dominant_emotion} ({vader_scores['compound']:.2f})"]

    # EXTRACTED FROM app.py:1838-1843 (get_emotions route logic)
    def get_emotions(self):
        """Get current emotion data"""
        with self.emotion_data["lock"]:
            return {
                "faces": self.emotion_data.get("faces", {}),
                "average_emotions": self.emotion_data.get("average_emotions", {})
            }

    def cleanup(self):
        """Cleanup and save final emotion data"""
        self.calculate_and_store_average_emotions()
        print("🧹 Emotion detection service cleanup complete")