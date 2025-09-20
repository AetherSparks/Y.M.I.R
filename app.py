#ALL IMPORTS
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
from email.message import EmailMessage
import random
from signal import signal
import smtplib
import sys
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
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
import statistics

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
    """Predicts mood based on emotions and recommends matching songs.
    
    Args:
        emotion_file: Path to file containing emotion data
        
    Returns:
        list: List of dictionaries containing recommended songs with track, artist, and mood
    """
    import pickle
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # ✅ Get Emotion-Based Features
        emotion_vector, emotion_scores = process_emotions(emotion_file)
        
        # Load pre-trained transformations with error handling
        try:
            with open("models/scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
        except (FileNotFoundError, pickle.PickleError) as e:
            logger.error(f"Error loading scaler model: {e}")
            raise RuntimeError("Failed to load scaler model")

        try:
            with open("models/pca.pkl", "rb") as f:
                pca = pickle.load(f)
        except (FileNotFoundError, pickle.PickleError) as e:
            logger.error(f"Error loading PCA model: {e}")
            raise RuntimeError("Failed to load PCA model")

        # Apply transformations with validation
        if emotion_vector.ndim == 1:
            emotion_vector = emotion_vector.reshape(1, -1)
            
        # Standardize
        emotion_vector_scaled = scaler.transform(emotion_vector)
        # Reduce dimensions
        emotion_vector_pca = pca.transform(emotion_vector_scaled)

        # Debug confidence scores if needed
        mood_probs = ensemble_model.predict_proba(emotion_vector_pca)
        logger.debug(f"Model Confidence Scores: {dict(zip(le.classes_, mood_probs[0]))}")

        # ✅ Predict Mood
        predicted_mood_index = ensemble_model.predict(emotion_vector_pca)[0]
        predicted_mood = le.inverse_transform([predicted_mood_index])[0]
        logger.info(f"Initial Predicted Mood: {predicted_mood}")

        # ✅ Find Top 2 Dominant Emotions with validation
        if not emotion_scores:
            logger.warning("No emotion scores found, using default neutral mood")
            dominant_emotions = [("Neutral", 1.0)]
        else:
            dominant_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            
        mapped_moods = set()
        for emotion, score in dominant_emotions:
            mapped_moods.update(EMOTION_TO_MOOD.get(emotion, ["Neutral"]))
        
        logger.info(f"Dominant Emotions: {dominant_emotions} → Adjusted Moods: {mapped_moods}")

        # ✅ Adjust Mood If Necessary
        if predicted_mood not in mapped_moods and mapped_moods:
            predicted_mood = list(mapped_moods)[0]  # Take the first mapped mood
            logger.info(f"Adjusted mood to: {predicted_mood}")

        # ✅ Filter Songs Based on Mood with validation
        try:
            filtered_songs = df[df["Mood_Label"] == predicted_mood].copy()
        except KeyError:
            logger.error("Missing required column 'Mood_Label' in dataframe")
            raise ValueError("Dataset missing required columns")

        # ✅ Use Fallback Moods If No Songs Found
        if filtered_songs.empty and mapped_moods:
            logger.info(f"No songs found for {predicted_mood}, trying mapped moods: {mapped_moods}")
            filtered_songs = df[df["Mood_Label"].isin(mapped_moods)].copy()

        # ✅ Final Fallback to Neutral if Still Empty
        if filtered_songs.empty:
            logger.warning("No songs found for any mapped mood, falling back to Neutral")
            filtered_songs = df[df["Mood_Label"] == "Neutral"].copy()
            
            # If still empty, return empty list with warning
            if filtered_songs.empty:
                logger.error("No songs found for any mood category, including Neutral")
                return []

        # ✅ Select Up to 10 Songs with duplicate handling
        required_columns = ["Track Name", "Artist Name", "Mood_Label"]
        if not all(col in filtered_songs.columns for col in required_columns):
            logger.error(f"Missing required columns in dataframe. Required: {required_columns}")
            raise ValueError("Dataset missing required columns")
            
        try:
            # Drop duplicates more efficiently
            filtered_songs.drop_duplicates(subset=["Track Name", "Artist Name"], inplace=True)
            sample_size = min(10, len(filtered_songs))
            
            if sample_size == 0:
                logger.warning("No songs available after filtering duplicates")
                return []
                
            recommended_songs = filtered_songs.sample(sample_size)
        except Exception as e:
            logger.error(f"Error during sampling: {e}")
            # Fallback to first N records if sampling fails
            recommended_songs = filtered_songs.head(min(10, len(filtered_songs)))

        # ✅ Create song list with error handling for missing values
        song_list = []
        for _, row in recommended_songs.iterrows():
            try:
                song_list.append({
                    "track": row["Track Name"],
                    "artist": row["Artist Name"],
                    "mood": row["Mood_Label"],
                })
            except KeyError as e:
                logger.warning(f"Skipping song due to missing data: {e}")
                
        logger.info(f"Successfully recommended {len(song_list)} songs")
        return song_list
        
    except Exception as e:
        logger.error(f"Unexpected error in recommend_songs: {e}")
        # Return empty list rather than crashing
        return []

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════




















































#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
console = Console()

# Enhanced Chatbot with Gemini API Support
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("✅ Google Gemini API available")
except ImportError:
    GEMINI_AVAILABLE = False
    print("❌ Google Gemini API not available")

# Production-grade ML emotion detection
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    ML_AVAILABLE = True
    print("✅ Transformers available for production ML emotion detection")
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ Transformers not available")

@dataclass
class ChatMessage:
    """Structured chat message with metadata"""
    role: str
    content: str
    timestamp: datetime
    emotion: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'emotion': self.emotion,
            'confidence': self.confidence,
            'metadata': self.metadata or {}
        }

@dataclass
class UserProfile:
    """User profile with preferences and history"""
    user_id: str
    name: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    conversation_style: str = "balanced"
    emotion_history: Optional[List[str]] = None
    topics_of_interest: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}
        if self.emotion_history is None:
            self.emotion_history = []
        if self.topics_of_interest is None:
            self.topics_of_interest = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_active is None:
            self.last_active = datetime.now()

class ProductionEmotionAnalyzer:
    """Production-grade emotion analysis with ensemble of SOTA models"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.model_configs = [
            {
                'name': 'roberta_emotion',
                'model_id': 'j-hartmann/emotion-english-distilroberta-base',
                'weight': 0.4,
                'emotions': ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            },
            {
                'name': 'twitter_roberta',
                'model_id': 'cardiffnlp/twitter-roberta-base-emotion-multilabel-latest', 
                'weight': 0.3,
                'emotions': ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
            },
            {
                'name': 'bertweet_emotion',
                'model_id': 'finiteautomata/bertweet-base-emotion-analysis',
                'weight': 0.3,
                'emotions': ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
            }
        ]
        self.fallback_available = False
        
        # Initialize models
        self._initialize_production_models()
        
        # Standard emotion mapping for consistency
        self.emotion_standardization = {
            # Map all model outputs to standard emotions
            'anger': 'angry', 'angry': 'angry',
            'sadness': 'sad', 'sad': 'sad',
            'joy': 'happy', 'happiness': 'happy', 'happy': 'happy',
            'fear': 'anxious', 'anxious': 'anxious', 'nervous': 'anxious',
            'surprise': 'surprised', 'surprised': 'surprised',
            'disgust': 'disgusted', 'disgusted': 'disgusted',
            'love': 'loving', 'loving': 'loving',
            'neutral': 'neutral',
            'anticipation': 'excited', 'excitement': 'excited', 'excited': 'excited',
            'optimism': 'hopeful', 'hope': 'hopeful', 'hopeful': 'hopeful',
            'pessimism': 'worried', 'worry': 'worried', 'worried': 'worried',
            'trust': 'confident', 'confident': 'confident'
        }
    
    def _initialize_production_models(self):
        """Initialize multiple production-grade emotion models"""
        successful_models = 0
        
        for config in self.model_configs:
            try:
                print(f"🔄 Loading {config['name']}...")
                
                # Load model and tokenizer
                model = AutoModelForSequenceClassification.from_pretrained(config['model_id'])
                tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
                
                # Create pipeline - handle different model requirements
                device = 0 if torch.cuda.is_available() else -1
                
                # Some models don't support return_all_scores
                try:
                    pipeline_model = pipeline(
                        "text-classification",
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        return_all_scores=True
                    )
                except Exception:
                    # Fallback without return_all_scores
                    pipeline_model = pipeline(
                        "text-classification",
                        model=model,
                        tokenizer=tokenizer,
                        device=device
                    )
                
                self.models[config['name']] = {
                    'pipeline': pipeline_model,
                    'weight': config['weight'],
                    'emotions': config['emotions']
                }
                
                successful_models += 1
                print(f"✅ {config['name']} loaded successfully")
                
            except Exception as e:
                print(f"⚠️ Failed to load {config['name']}: {e}")
                continue
        
        if successful_models == 0:
            print("❌ No emotion models loaded, using fallback")
            self._setup_fallback_model()
        else:
            print(f"✅ {successful_models}/{len(self.model_configs)} emotion models loaded")
    
    def _setup_fallback_model(self):
        """Setup lightweight fallback model"""
        try:
            # Use a simple, reliable model as fallback
            pipeline_model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )
            
            self.models['fallback'] = {
                'pipeline': pipeline_model,
                'weight': 1.0,
                'emotions': ['negative', 'neutral', 'positive']
            }
            self.fallback_available = True
            print("✅ Fallback sentiment model loaded")
            
        except Exception as e:
            print(f"❌ Even fallback model failed: {e}")
    
    def _analyze_with_ensemble(self, text: str) -> Dict[str, Any]:
        """Use ensemble of models for robust emotion detection"""
        if not self.models:
            return self._get_neutral_result()
        
        model_results = {}
        
        # Run all models in parallel for speed
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            future_to_model = {
                executor.submit(self._run_single_model, name, config, text): name 
                for name, config in self.models.items()
            }
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result(timeout=5)  # 5 second timeout per model
                    if result:
                        model_results[model_name] = result
                except Exception as e:
                    print(f"⚠️ Model {model_name} failed: {e}")
        
        if not model_results:
            return self._get_neutral_result()
        
        # Combine results using weighted voting
        return self._ensemble_vote(model_results)
    
    def _run_single_model(self, model_name: str, model_config: Dict, text: str) -> Optional[Dict]:
        """Run a single model and return standardized results"""
        try:
            pipeline_model = model_config['pipeline']
            results = pipeline_model(text)
            
            # Handle different result formats from different models
            standardized_emotions = {}
            
            # Flatten results if they're wrapped in extra lists
            if isinstance(results, list) and len(results) > 0:
                # Check if it's wrapped in an extra list: [[{...}]]
                if isinstance(results[0], list):
                    results = results[0]  # Unwrap the outer list
            
            # Now process the actual results
            if isinstance(results, list) and len(results) > 0:
                for result in results:
                    if isinstance(result, dict) and 'label' in result and 'score' in result:
                        emotion = result['label'].lower()
                        score = result['score']
                        
                        # Skip 'others' category from some models
                        if emotion == 'others':
                            continue
                        
                        # Map to standard emotion
                        std_emotion = self.emotion_standardization.get(emotion, emotion)
                        
                        if std_emotion in standardized_emotions:
                            standardized_emotions[std_emotion] = max(standardized_emotions[std_emotion], score)
                        else:
                            standardized_emotions[std_emotion] = score
            
            # Case 2: Results is a single dictionary
            elif isinstance(results, dict) and 'label' in results and 'score' in results:
                emotion = results['label'].lower()
                score = results['score']
                if emotion != 'others':  # Skip 'others' category
                    std_emotion = self.emotion_standardization.get(emotion, emotion)
                    standardized_emotions[std_emotion] = score
            
            # Case 3: Results is in different format - try to handle gracefully
            else:
                print(f"⚠️ Unknown result format from {model_name}: {type(results)}")
                return None
            
            if not standardized_emotions:
                print(f"⚠️ No emotions extracted from {model_name}")
                return None
            
            return {
                'emotions': standardized_emotions,
                'weight': model_config['weight'],
                'model': model_name
            }
            
        except Exception as e:
            print(f"⚠️ Single model {model_name} error: {e}")
            return None
    
    def _ensemble_vote(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine multiple model results using weighted voting"""
        
        # Collect all emotions and their weighted scores
        emotion_scores = {}
        total_weight = 0
        model_count = len(model_results)
        
        for model_name, result in model_results.items():
            weight = result['weight']
            total_weight += weight
            
            for emotion, score in result['emotions'].items():
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = []
                emotion_scores[emotion].append(score * weight)
        
        # Calculate final scores
        final_emotions = {}
        for emotion, scores in emotion_scores.items():
            # Use mean of weighted scores
            final_emotions[emotion] = statistics.mean(scores)
        
        # Find dominant emotion
        if final_emotions:
            dominant_emotion = max(final_emotions.items(), key=lambda x: x[1])
            
            # Check for mixed emotions (multiple high-confidence emotions)
            high_confidence = {k: v for k, v in final_emotions.items() if v > 0.4}
            mixed = len(high_confidence) > 1
            
            return {
                'dominant_emotion': dominant_emotion[0],
                'confidence': dominant_emotion[1],
                'all_emotions': final_emotions,
                'mixed_emotions': mixed,
                'method': f'ensemble_{model_count}_models',
                'models_used': list(model_results.keys())
            }
        
        return self._get_neutral_result()
    
    def _get_neutral_result(self) -> Dict[str, Any]:
        """Return neutral result as fallback"""
        return {
            'dominant_emotion': 'neutral',
            'confidence': 0.6,
            'all_emotions': {'neutral': 0.6},
            'mixed_emotions': False,
            'method': 'fallback_neutral'
        }
    
    def analyze_text_emotion(self, text: str) -> Dict[str, Any]:
        """Main emotion analysis method"""
        if not text.strip():
            return self._get_neutral_result()
        
        try:
            # Use ensemble approach - no rule-based preprocessing
            result = self._analyze_with_ensemble(text)
            
            # Add original text for debugging
            result['original_text'] = text
            
            return result
            
        except Exception as e:
            print(f"❌ Emotion analysis failed: {e}")
            return self._get_neutral_result()

class FunctionCalling:
    """Production-grade function calling with error handling"""
    
    def __init__(self):
        self.available_functions = {
            'web_search': self.web_search,
            'get_weather': self.get_weather,
            'calculate': self.calculate,
            'get_time': self.get_time,
            'get_date': self.get_date
        }
    
    def web_search(self, query: str, num_results: int = 3) -> str:
        """Search the web for information"""
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if data.get('AbstractText'):
                return f"Search result: {data['AbstractText']}"
            elif data.get('RelatedTopics'):
                topics = data['RelatedTopics'][:num_results]
                results = []
                for topic in topics:
                    if 'Text' in topic:
                        results.append(topic['Text'])
                return "Search results:\n" + "\n".join(f"• {result}" for result in results)
            else:
                return f"No specific results found for '{query}'"
                
        except Exception as e:
            return f"Web search error: {e}"
    
    def get_weather(self, location: str = "current") -> str:
        """Get weather information"""
        return f"Weather functionality requires API key setup. Location requested: {location}"
    
    def calculate(self, expression: str) -> str:
        """Perform mathematical calculations"""
        try:
            # Safe evaluation
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return f"Calculation: {expression} = {result}"
            else:
                return "Invalid mathematical expression"
        except Exception as e:
            return f"Calculation error: {e}"
    
    def get_time(self) -> str:
        """Get current time"""
        return f"Current time: {datetime.now().strftime('%H:%M:%S')}"
    
    def get_date(self) -> str:
        """Get current date"""
        return f"Current date: {datetime.now().strftime('%Y-%m-%d (%A)')}"
    
    def detect_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Detect function calls using NLP, not regex rules"""
        # This could be enhanced with a small NLP model for intent detection
        # For now, simple pattern matching but easily replaceable
        function_calls = []
        
        patterns = {
            'web_search': [r'search for (.+)', r'look up (.+)', r'find (.+)'],
            'get_weather': [r'weather', r'temperature'],
            'calculate': [r'calculate (.+)', r'what is (.+[\+\-\*/].+)'],
            'get_time': [r'time'],
            'get_date': [r'date']
        }
        
        for function_name, regex_patterns in patterns.items():
            for pattern in regex_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    function_calls.append({
                        'function': function_name,
                        'args': match.groups() if match.groups() else [],
                        'confidence': 0.8
                    })
        
        return function_calls
    
    def execute_function(self, function_name: str, args: List[str]) -> str:
        """Execute a function with error handling"""
        if function_name in self.available_functions:
            try:
                if args:
                    return self.available_functions[function_name](*args)
                else:
                    return self.available_functions[function_name]()
            except Exception as e:
                return f"Function execution error: {e}"
        else:
            return f"Function '{function_name}' not available"
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

def update_all_in_background(interval=1):
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


#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
class GeminiChatbot:
    """Production-ready Gemini chatbot with ensemble emotion detection"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self.chat_session = None
        
        # Initialize components
        self.emotion_analyzer = ProductionEmotionAnalyzer()
        self.function_calling = FunctionCalling()
        self.conversation_history = deque(maxlen=100)
        self.user_profile = None
        
        # Configuration
        self.config = {
            'model_name': 'gemini-2.0-flash-exp',
            'temperature': 0.7,
            'top_k': 40,
            'top_p': 0.95,
            'max_tokens': 2048,
            'safety_settings': [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        }
        
        # System context
        self.system_context = """You are Y.M.I.R (Your Mental Intelligence and Recovery), a production-grade AI assistant specializing in mental health, emotional support, and intelligent conversation.

Your capabilities:
- Advanced ensemble emotion detection using multiple SOTA models
- Contextual understanding without rule-based preprocessing
- Function calling for web search, calculations, etc.
- Conversation memory and emotional intelligence
- Mental health and wellness guidance
- Technical assistance and problem-solving

Always be helpful, accurate, and emotionally intelligent in your responses."""
        
        self._initialize_gemini()
        self._load_user_profile()
    
    def _initialize_gemini(self):
        """Initialize Gemini API"""
        try:
            if not GEMINI_AVAILABLE:
                print("⚠️ Gemini API not available")
                return
            
            genai.configure(api_key=self.api_key)  # type: ignore
            
            generation_config = {
                'temperature': self.config['temperature'],
                'top_k': self.config['top_k'], 
                'top_p': self.config['top_p'],
                'max_output_tokens': self.config['max_tokens']
            }
            
            self.model = genai.GenerativeModel(  # type: ignore
                model_name=self.config['model_name'],
                generation_config=generation_config,  # type: ignore
                safety_settings=self.config['safety_settings']
            )
            
            self.chat_session = self.model.start_chat(history=[])
            print("✅ Gemini API initialized successfully")
            
        except Exception as e:
            print(f"❌ Gemini initialization error: {e}")
            self.model = None
    
    def _load_user_profile(self):
        """Load or create user profile"""
        profile_path = Path("user_profile.json")
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r') as f:
                    data = json.load(f)
                    
                    # Handle datetime fields safely
                    if 'created_at' in data and isinstance(data['created_at'], str):
                        try:
                            data['created_at'] = datetime.fromisoformat(data['created_at'])
                        except:
                            data['created_at'] = datetime.now()
                    
                    if 'last_active' in data and isinstance(data['last_active'], str):
                        try:
                            data['last_active'] = datetime.fromisoformat(data['last_active'])
                        except:
                            data['last_active'] = datetime.now()
                    
                    self.user_profile = UserProfile(**data)
                    self.user_profile.last_active = datetime.now()
                print("✅ User profile loaded")
            except Exception as e:
                print(f"⚠️ Profile loading error: {e}")
                self._create_new_profile()
        else:
            self._create_new_profile()
    
    def _create_new_profile(self):
        """Create new user profile"""
        self.user_profile = UserProfile(
            user_id=f"user_{int(time.time())}",
            conversation_style="balanced"
        )
        self._save_user_profile()
        print("✅ New user profile created")
    
    def _save_user_profile(self):
        """Save user profile"""
        try:
            if self.user_profile is None:
                return
            profile_data = asdict(self.user_profile)
            
            # Handle datetime conversion safely
            if self.user_profile.created_at is not None:
                profile_data['created_at'] = self.user_profile.created_at.isoformat()
            else:
                profile_data['created_at'] = datetime.now().isoformat()
                
            if self.user_profile.last_active is not None:
                profile_data['last_active'] = self.user_profile.last_active.isoformat()
            else:
                profile_data['last_active'] = datetime.now().isoformat()
            
            with open("user_profile.json", 'w') as f:
                json.dump(profile_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Profile saving error: {e}")
    
    def _build_context_prompt(self, user_input: str, emotion_data: Dict[str, Any]) -> str:
        """Build context prompt with emotion and history"""
        context_parts = [self.system_context]
        
        # Add emotion context
        if emotion_data['dominant_emotion'] != 'neutral':
            emotion_context = f"""
Current user emotion: {emotion_data['dominant_emotion']} (confidence: {emotion_data['confidence']:.2f})
Detection method: {emotion_data.get('method', 'unknown')}
Models used: {', '.join(emotion_data.get('models_used', []))}

Adapt your response to be supportive and appropriate for someone feeling {emotion_data['dominant_emotion']}.
"""
            context_parts.append(emotion_context)
        
        # Add conversation history
        if self.conversation_history:
            recent_messages = list(self.conversation_history)[-6:]
            history_context = "Recent conversation:\n"
            for msg in recent_messages:
                emotion_info = f" [felt: {msg.emotion}]" if msg.emotion and msg.emotion != 'neutral' else ""
                history_context += f"{msg.role}: {msg.content}{emotion_info}\n"
            context_parts.append(history_context)
        
        return "\n".join(context_parts) + f"\n\nUser: {user_input}"
    
    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """Generate response with production-grade emotion analysis"""
        try:
            # Analyze emotions using ensemble
            emotion_data = self.emotion_analyzer.analyze_text_emotion(user_input)
            
            # Check for function calls
            function_calls = self.function_calling.detect_function_calls(user_input)
            function_results = []
            
            # Execute functions
            for call in function_calls:
                result = self.function_calling.execute_function(call['function'], call['args'])
                function_results.append(result)
            
            # Build context
            enhanced_prompt = self._build_context_prompt(user_input, emotion_data)
            
            if function_results:
                enhanced_prompt += f"\n\nFunction results: {'; '.join(function_results)}"
            
            if not self.model or not self.chat_session:
                return {
                    'response': "I'm having trouble connecting to my AI system.",
                    'emotion': emotion_data,
                    'functions_used': function_calls,
                    'streaming': False
                }
            
            # Generate response
            response = self.chat_session.send_message(enhanced_prompt)
            response_text = response.text
            
            # Create messages
            user_message = ChatMessage(
                role='user',
                content=user_input,
                timestamp=datetime.now(),
                emotion=emotion_data['dominant_emotion'],
                confidence=emotion_data['confidence'],
                metadata={'emotion_method': emotion_data.get('method')}
            )
            
            assistant_message = ChatMessage(
                role='assistant', 
                content=response_text,
                timestamp=datetime.now(),
                metadata={'functions_used': function_calls}
            )
            
            # Add to history
            self.conversation_history.append(user_message)
            self.conversation_history.append(assistant_message)
            
            # Update profile
            if self.user_profile:
                self.user_profile.last_active = datetime.now()
                if emotion_data['dominant_emotion'] != 'neutral' and self.user_profile.emotion_history is not None:
                    self.user_profile.emotion_history.append(emotion_data['dominant_emotion'])
                    self.user_profile.emotion_history = self.user_profile.emotion_history[-20:]
                self._save_user_profile()
            
            return {
                'response': response_text,
                'emotion': emotion_data,
                'functions_used': function_calls,
                'streaming': True,
                'user_message': user_message,
                'assistant_message': assistant_message
            }
            
        except Exception as e:
            print(f"❌ Response generation error: {e}")
            return {
                'response': f"I apologize, but I encountered an error: {str(e)}",
                'emotion': {'dominant_emotion': 'neutral', 'confidence': 0.5},
                'functions_used': [],
                'streaming': False
            }
    
    def get_response(self, message: str) -> Dict[str, Any]:
        """Get response for web interface (wrapper around generate_response)"""
        try:
            result = self.generate_response(message)
            
            # Format for web interface
            emotion_data = result.get('emotion', {})
            emotion_context = "No emotion detected"
            
            if emotion_data and emotion_data.get('dominant_emotion'):
                emotion = emotion_data['dominant_emotion']
                confidence = emotion_data.get('confidence', 0.0)
                emotion_context = f"Detected emotion: {emotion} ({confidence:.1%} confidence)"
            
            return {
                'response': result['response'],
                'emotion_analysis': emotion_data,
                'emotion_context': emotion_context,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Gemini get_response error: {e}")
            return {
                'response': "I apologize, but I encountered an error. Please try again.",
                'emotion_analysis': None,
                'emotion_context': "Error in emotion analysis",
                'timestamp': datetime.now().isoformat()
            }
    
    def save_conversation(self):
        """Save conversation to file"""
        try:
            # Prepare user profile data safely
            user_profile_data = None
            if self.user_profile:
                user_profile_data = asdict(self.user_profile)
                # Convert datetimes to strings safely
                if hasattr(self.user_profile.created_at, 'isoformat'):
                    user_profile_data['created_at'] = self.user_profile.created_at.isoformat()  # type: ignore
                else:
                    user_profile_data['created_at'] = str(self.user_profile.created_at)
                    
                if hasattr(self.user_profile.last_active, 'isoformat'):
                    user_profile_data['last_active'] = self.user_profile.last_active.isoformat()  # type: ignore
                else:
                    user_profile_data['last_active'] = str(self.user_profile.last_active)
            
            # Get conversation data
            conversation_list = []
            emotions_in_session = []
            
            for msg in self.conversation_history:
                try:
                    msg_dict = msg.to_dict()
                    conversation_list.append(msg_dict)
                    
                    # Track emotions
                    if msg.emotion and msg.emotion != 'neutral':
                        emotions_in_session.append(msg.emotion)
                        
                except Exception as e:
                    print(f"⚠️ Error processing message: {e}")
                    continue
            
            conversation_data = {
                'timestamp': datetime.now().isoformat(),
                'user_profile': user_profile_data,
                'conversation': conversation_list,
                'session_stats': {
                    'total_messages': len(conversation_list),
                    'user_messages': len([msg for msg in conversation_list if msg.get('role') == 'user']),
                    'assistant_messages': len([msg for msg in conversation_list if msg.get('role') == 'assistant']),
                    'emotions_detected': list(set(emotions_in_session)),
                    'session_duration_minutes': 0,  # Could calculate this later
                    'models_used': list(set(msg.get('metadata', {}).get('emotion_method', '') for msg in conversation_list if msg.get('metadata')))
                }
            }
            
            filename = f"chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(conversation_data, f, indent=2, default=str, ensure_ascii=False)
            
            print(f"✅ Conversation saved to {filename}")
            print(f"   📊 {conversation_data['session_stats']['total_messages']} messages, {len(conversation_data['session_stats']['emotions_detected'])} emotions detected")
            return filename
            
        except Exception as e:
            print(f"❌ Conversation save error: {e}")
            return None

class SimpleFallbackChatbot:
    """Simple fallback chatbot when Gemini is not available"""
    
    def __init__(self):
        self.emotion_analyzer = ProductionEmotionAnalyzer() if ML_AVAILABLE else None
        self.conversation_history = deque(maxlen=50)
        
    def get_response(self, message: str) -> Dict[str, Any]:
        """Get response with emotion analysis"""
        
        # Analyze emotion
        emotion_result = None
        if self.emotion_analyzer:
            try:
                emotion_result = self.emotion_analyzer.analyze_text_emotion(message)
            except Exception as e:
                print(f"Emotion analysis failed: {e}")
        
        # Generate simple response based on emotion
        if emotion_result and emotion_result['dominant_emotion']:
            emotion = emotion_result['dominant_emotion']
            confidence = emotion_result['confidence']
            
            responses = {
                'happy': f"I can sense your happiness! That's wonderful. Tell me more about what's making you feel so positive.",
                'sad': f"I notice you might be feeling sad. I'm here to listen and support you. What's on your mind?",
                'angry': f"I sense some frustration. Let's work through this together. What's bothering you?",
                'anxious': f"I understand you might be feeling anxious. You're safe here. What's concerning you?",
                'surprised': f"You seem surprised! What's caught your attention?",
                'neutral': f"Thank you for sharing that. How are you feeling right now?"
            }
            
            response = responses.get(emotion, responses['neutral'])
            emotion_context = f"Detected emotion: {emotion} ({confidence:.1%} confidence)"
        else:
            response = "Hello! I'm Y.M.I.R, your AI companion. How can I help you today?"
            emotion_context = "No emotion detected"
        
        # Store in conversation history
        self.conversation_history.append({
            'user': message,
            'assistant': response,
            'emotion': emotion_result,
            'timestamp': datetime.now().isoformat()
        })
        
        return {
            'response': response,
            'emotion_analysis': emotion_result,
            'emotion_context': emotion_context,
            'timestamp': datetime.now().isoformat()
        }

# Global chatbot instance
web_chatbot = None

def init_web_chatbot():
    """Initialize chatbot for web interface"""
    global web_chatbot
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key and GEMINI_AVAILABLE:
        try:
            web_chatbot = GeminiChatbot(api_key)
            print("✅ Web chatbot initialized with Gemini API")
        except Exception as e:
            print(f"⚠️ Gemini chatbot failed, using fallback: {e}")
            web_chatbot = SimpleFallbackChatbot()
    else:
        print("⚠️ Using fallback chatbot (no Gemini API)")
        web_chatbot = SimpleFallbackChatbot()

@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with production-grade emotion analysis"""
    try:
        user_input = request.json.get("message", "").strip()
        if not user_input:
            return jsonify({"error": "No message provided."}), 400
        
        if not web_chatbot:
            return jsonify({"error": "Chatbot not initialized."}), 500
        
        # Get response from enhanced chatbot
        result = web_chatbot.get_response(user_input)
        
        # Save for backwards compatibility with existing system
        chat_session = []
        if os.path.exists("chat_results.json"):
            with open("chat_results.json", "r") as f:
                try:
                    data = json.load(f)
                    chat_session = data.get("conversation", [])
                except:
                    chat_session = []
        
        # Add to session
        chat_session.append({"user": user_input, "chatbot": result['response']})
        
        # Extract emotion for backwards compatibility
        emotion_analysis = result.get('emotion_analysis', {})
        dominant_emotion = emotion_analysis.get('dominant_emotion', 'neutral')
        
        # Save chat results for compatibility
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        chat_data = {
            "timestamp": timestamp,
            "conversation": chat_session,
            "dominant_emotion": dominant_emotion,
            "emotion_scores": emotion_analysis.get('all_emotions', {dominant_emotion: 0.6}),
            "model_emotions": [f"Enhanced ensemble analysis: {dominant_emotion}"]
        }
        
        with open("chat_results.json", "w") as f:
            json.dump(chat_data, f, indent=4)
        
        response_data = {
            "response": result['response'],
            "dominant_emotion": dominant_emotion,
            "model_emotions": [f"Enhanced ensemble analysis: {dominant_emotion}"],
            "emotion_context": result['emotion_context']
        }
        
        # Check for conversation end
        if user_input.lower() in ["quit", "bye", "exit", "goodbye", "end"]:
            response_data["end_chat"] = True
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": "An error occurred while processing your message."}), 500

@app.route('/detect_emotion', methods=['GET'])
def detect_emotion():
    """Detects the dominant emotion from the chat session (for real-time updates)."""
    try:
        if os.path.exists("chat_results.json"):
            with open("chat_results.json", "r") as f:
                data = json.load(f)
                return jsonify({
                    "dominant_emotion": data.get("dominant_emotion", "neutral"),
                    "model_emotions": data.get("model_emotions", [])
                })
        else:
            return jsonify({"dominant_emotion": "neutral", "model_emotions": []})
    except Exception as e:
        print(f"Detect emotion error: {e}")
        return jsonify({"dominant_emotion": "neutral", "model_emotions": []})

@app.route('/save_chat', methods=['POST'])
def save_chat():
    """Saves chat conversation and detected emotion."""
    try:
        if web_chatbot and hasattr(web_chatbot, 'save_conversation'):
            filename = web_chatbot.save_conversation()
            if filename:
                return jsonify({"message": f"Chat saved to {filename}"})
        return jsonify({"message": "Chat saved successfully."})
    except Exception as e:
        print(f"Save chat error: {e}")
        return jsonify({"error": "Failed to save chat."}), 500
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
    # Initialize enhanced chatbot system
    print("🤖 Initializing Y.M.I.R Enhanced Chatbot System...")
    init_web_chatbot()
    
    # Start background processing thread
    background_thread = threading.Thread(target=update_all_in_background, daemon=True)
    background_thread.start()
    
    try:
        print("🚀 Starting Y.M.I.R Enhanced Flask Application")
        print("="*60)
        print("🌐 Web Interface: http://127.0.0.1:10000")
        print("🤖 Enhanced Chatbot: Production-grade emotion analysis")
        print("📹 Video Emotion Detection: Real-time processing")
        print("🎵 Music Recommendations: Emotion-based suggestions")
        print("="*60)
        app.run(debug=True, host='127.0.0.1', port=10000)

    except KeyboardInterrupt:
        print("\n🔴 Server stopped manually.")
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

        

