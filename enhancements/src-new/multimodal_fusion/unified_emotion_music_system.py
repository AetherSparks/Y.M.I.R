#!/usr/bin/env python3
"""
🎯 Y.M.I.R Unified Emotion-Music System
======================================
Professional production-ready system that combines:
- Real emotion combiner (Firebase + JSON data)
- Advanced multimodal fusion strategies  
- Music recommendation engine
- Real-time processing pipeline

This is the MAIN system that your app.py should use!

Features:
✅ Reads actual emotions from Firebase + JSON files
✅ 5 advanced fusion strategies (simple, adaptive, confidence_based, etc.)
✅ Integrates with music recommendation system
✅ Session management and analytics
✅ Production-ready error handling
✅ Silent mode for production use

Usage in app.py:
    from enhancements.src-new.multimodal_fusion.unified_emotion_music_system import get_emotion_and_music
    
    result = get_emotion_and_music(session_id="user123")
    if result:
        emotion = result['emotion']
        music_recommendations = result['music_recommendations']
"""

import os
import sys
import json
import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed - .env file won't be loaded automatically")
    print("   Install with: pip install python-dotenv")

# 🤖 AI-POWERED THERAPEUTIC RECOMMENDATIONS
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("✅ Gemini AI available for therapeutic recommendations")
except ImportError:
    GEMINI_AVAILABLE = False
    print("❌ Gemini AI not available - falling back to rule-based recommendations")

# Firebase imports (optional)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    firebase_admin = None
    credentials = None
    firestore = None

@dataclass
class UnifiedEmotionResult:
    """Complete emotion and music result"""
    # Emotion data
    emotion: str
    confidence: float
    source: str  # 'facial_only', 'text_only', 'combined_*'
    strategy: str  # fusion strategy used
    
    # Source data
    facial_data: Optional[Dict[str, Any]] = None
    text_data: Optional[Dict[str, Any]] = None
    
    # Music recommendations  
    music_recommendations: Optional[List[Dict[str, Any]]] = None
    recommendation_strategy: str = "adaptive"
    
    # Metadata
    session_id: str = ""
    timestamp: datetime = None
    processing_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class AdvancedEmotionFusionEngine:
    """🧠 Advanced emotion fusion with 5 strategies"""
    
    def __init__(self):
        self.emotion_mappings = {
            'joy': ['happy', 'joy', 'excitement', 'pleasure', 'delight'],
            'sadness': ['sad', 'sadness', 'sorrow', 'grief', 'melancholy'],
            'anger': ['angry', 'anger', 'rage', 'fury', 'irritation'],
            'fear': ['fear', 'afraid', 'scared', 'anxiety', 'worry'],
            'surprise': ['surprise', 'surprised', 'astonishment', 'amazement'],
            'disgust': ['disgust', 'disgusted', 'revulsion', 'contempt'],
            'neutral': ['neutral', 'calm', 'peaceful', 'relaxed'],
            'excitement': ['excited', 'enthusiasm', 'energy', 'vigorous']
        }
        
        self.strategies = {
            'simple': self._simple_fusion,
            'adaptive': self._adaptive_fusion,
            'confidence_based': self._confidence_based_fusion,
            'temporal_weighted': self._temporal_weighted_fusion,
            'weighted_average': self._weighted_average_fusion
        }
    
    def normalize_emotion(self, emotion: str) -> str:
        """Normalize emotion to standard vocabulary"""
        emotion_lower = emotion.lower()
        for standard_emotion, variants in self.emotion_mappings.items():
            if emotion_lower in variants:
                return standard_emotion
        return emotion_lower
    
    def fuse_emotions(self, facial_data: Dict[str, Any], text_data: Dict[str, Any], 
                     strategy: str = 'adaptive') -> Tuple[str, float, str]:
        """Main fusion method"""
        fusion_func = self.strategies.get(strategy, self._adaptive_fusion)
        return fusion_func(facial_data, text_data)
    
    def _simple_fusion(self, facial_data: Dict[str, Any], text_data: Dict[str, Any]) -> Tuple[str, float, str]:
        """Simple: highest confidence wins"""
        facial_confidence = facial_data.get('confidence', 0.0) if facial_data else 0.0
        text_confidence = text_data.get('confidence', 0.0) if text_data else 0.0
        
        if facial_confidence >= text_confidence and facial_data:
            return facial_data.get('emotion', 'neutral'), facial_confidence, "facial_higher_confidence"
        elif text_data:
            return text_data.get('emotion', 'neutral'), text_confidence, "text_higher_confidence"
        else:
            return 'neutral', 0.0, "no_data"
    
    def _adaptive_fusion(self, facial_data: Dict[str, Any], text_data: Dict[str, Any]) -> Tuple[str, float, str]:
        """Adaptive: smart context-aware fusion"""
        facial_confidence = facial_data.get('confidence', 0.0) if facial_data else 0.0
        text_confidence = text_data.get('confidence', 0.0) if text_data else 0.0
        
        # Smart confidence-based adjustment
        confidence_ratio = facial_confidence / (text_confidence + 0.01)
        
        if confidence_ratio > 2.0:
            # Strongly favor facial
            weight = 0.7
            emotion = facial_data.get('emotion', 'neutral') if facial_data else 'neutral'
            confidence = facial_confidence * weight + text_confidence * (1-weight)
            return self.normalize_emotion(emotion), confidence, f"adaptive_facial_favored"
        elif confidence_ratio < 0.5:
            # Strongly favor text
            weight = 0.3
            emotion = text_data.get('emotion', 'neutral') if text_data else 'neutral'
            confidence = facial_confidence * weight + text_confidence * (1-weight)
            return self.normalize_emotion(emotion), confidence, f"adaptive_text_favored"
        else:
            # Balanced fusion
            if facial_data and text_data:
                avg_confidence = (facial_confidence + text_confidence) / 2
                # Use higher confidence emotion
                if facial_confidence >= text_confidence:
                    emotion = facial_data.get('emotion', 'neutral')
                else:
                    emotion = text_data.get('emotion', 'neutral')
                return self.normalize_emotion(emotion), avg_confidence, "adaptive_balanced"
            elif facial_data:
                return self.normalize_emotion(facial_data.get('emotion', 'neutral')), facial_confidence, "adaptive_facial_only"
            elif text_data:
                return self.normalize_emotion(text_data.get('emotion', 'neutral')), text_confidence, "adaptive_text_only"
            else:
                return 'neutral', 0.0, "adaptive_no_data"
    
    def _confidence_based_fusion(self, facial_data: Dict[str, Any], text_data: Dict[str, Any]) -> Tuple[str, float, str]:
        """Confidence-based: pure confidence competition"""
        facial_confidence = facial_data.get('confidence', 0.0) if facial_data else 0.0
        text_confidence = text_data.get('confidence', 0.0) if text_data else 0.0
        
        if facial_confidence > text_confidence * 1.5 and facial_data:
            return self.normalize_emotion(facial_data.get('emotion', 'neutral')), facial_confidence, "confidence_facial_wins"
        elif text_confidence > facial_confidence * 1.5 and text_data:
            return self.normalize_emotion(text_data.get('emotion', 'neutral')), text_confidence, "confidence_text_wins"
        elif facial_confidence >= text_confidence and facial_data:
            return self.normalize_emotion(facial_data.get('emotion', 'neutral')), facial_confidence, "confidence_facial_slight"
        elif text_data:
            return self.normalize_emotion(text_data.get('emotion', 'neutral')), text_confidence, "confidence_text_slight"
        else:
            return 'neutral', 0.0, "confidence_no_data"
    
    def _temporal_weighted_fusion(self, facial_data: Dict[str, Any], text_data: Dict[str, Any]) -> Tuple[str, float, str]:
        """Temporal: newer emotions get higher weight"""
        now = datetime.now()
        facial_age = 999
        text_age = 999
        
        if facial_data and facial_data.get('timestamp'):
            try:
                if isinstance(facial_data['timestamp'], datetime):
                    facial_age = (now - facial_data['timestamp']).total_seconds()
                else:
                    facial_age = 30  # Default recent
            except:
                facial_age = 30
        
        if text_data and text_data.get('timestamp'):
            try:
                if isinstance(text_data['timestamp'], datetime):
                    text_age = (now - text_data['timestamp']).total_seconds()
                else:
                    text_age = 30  # Default recent
            except:
                text_age = 30
        
        # Calculate temporal weights (newer = higher weight)
        max_age = 300  # 5 minutes max relevance
        facial_weight = max(0.1, 1.0 - (facial_age / max_age))
        text_weight = max(0.1, 1.0 - (text_age / max_age))
        
        # Normalize weights
        total_weight = facial_weight + text_weight
        if total_weight > 0:
            facial_weight /= total_weight
            text_weight /= total_weight
        
        # Apply weights
        facial_confidence = facial_data.get('confidence', 0.0) if facial_data else 0.0
        text_confidence = text_data.get('confidence', 0.0) if text_data else 0.0
        
        weighted_facial = facial_confidence * facial_weight
        weighted_text = text_confidence * text_weight
        
        if weighted_facial >= weighted_text and facial_data:
            return self.normalize_emotion(facial_data.get('emotion', 'neutral')), weighted_facial, f"temporal_facial_w{facial_weight:.2f}"
        elif text_data:
            return self.normalize_emotion(text_data.get('emotion', 'neutral')), weighted_text, f"temporal_text_w{text_weight:.2f}"
        else:
            return 'neutral', 0.0, "temporal_no_data"
    
    def _weighted_average_fusion(self, facial_data: Dict[str, Any], text_data: Dict[str, Any]) -> Tuple[str, float, str]:
        """Weighted average: mathematical fusion of all emotions"""
        # Extract all emotions
        facial_emotions = facial_data.get('emotions', {}) if facial_data else {}
        text_emotion = text_data.get('emotion', 'neutral') if text_data else 'neutral'
        text_confidence = text_data.get('confidence', 0.0) if text_data else 0.0
        
        # Convert text emotion to emotions dict
        text_emotions = {text_emotion: text_confidence}
        
        # Combine all emotion scores with equal weights
        all_emotions = set(facial_emotions.keys()) | set(text_emotions.keys())
        combined_scores = {}
        
        for emotion in all_emotions:
            facial_score = facial_emotions.get(emotion, 0.0)
            text_score = text_emotions.get(emotion, 0.0)
            
            # Convert percentages to 0-1 if needed
            if isinstance(facial_score, (int, float)) and facial_score > 1.0:
                facial_score = facial_score / 100.0
            
            # Average the scores
            avg_score = (facial_score + text_score) / 2
            combined_scores[emotion] = avg_score
        
        if combined_scores:
            # Get dominant emotion
            dominant_emotion = max(combined_scores.items(), key=lambda x: x[1])
            return self.normalize_emotion(dominant_emotion[0]), dominant_emotion[1], "weighted_average_combined"
        else:
            return 'neutral', 0.0, "weighted_average_no_data"

@dataclass
class MusicRecommendation:
    """Individual music recommendation with full metadata"""
    track_name: str
    artist_name: str
    emotion_target: str
    confidence_score: float
    therapeutic_benefit: str
    audio_features: Dict[str, float]
    recommendation_reason: str
    timestamp: datetime
    session_id: str
    recommendation_strategy: str = "adaptive"
    
    # 🎵 ENHANCED DATASET FIELDS
    album: str = "Unknown Album"
    track_popularity: int = 0
    artist_popularity: int = 0
    musical_features: str = "Unknown"
    artist_genres: str = "Unknown"
    mental_health_benefit: str = "General Wellness"
    duration_ms: int = 0
    danceability: float = 0.0
    energy: float = 0.0
    valence: float = 0.0
    tempo: float = 0.0
    # Additional audio features from dataset
    key: int = 0
    loudness: float = 0.0
    mode: int = 0
    speechiness: float = 0.0
    acousticness: float = 0.0
    instrumentalness: float = 0.0
    liveness: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass  
class RecommendationSet:
    """Complete set of recommendations with analytics"""
    session_id: str
    target_emotion: str
    emotion_confidence: float
    modality_sources: Dict[str, bool]  # facial, textual availability
    recommendations: List[MusicRecommendation]
    generation_time: datetime
    recommendation_strategy: str
    context_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['generation_time'] = self.generation_time.isoformat()
        result['recommendations'] = [r.to_dict() for r in self.recommendations]
        return result

class AITherapeuticAdvisor:
    """🧠 AI-powered therapeutic music recommendation advisor using Gemini"""
    
    def __init__(self):
        self.gemini_available = GEMINI_AVAILABLE
        self.model = None
        
        if self.gemini_available:
            self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini AI model"""
        try:
            # Check for API key in environment
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_AI_API_KEY')
            
            print(f"🔑 DEBUG: Looking for API keys...")
            print(f"   GEMINI_API_KEY: {'✅ Found' if os.getenv('GEMINI_API_KEY') else '❌ Not found'}")
            print(f"   GOOGLE_API_KEY: {'✅ Found' if os.getenv('GOOGLE_API_KEY') else '❌ Not found'}")
            print(f"   GOOGLE_AI_API_KEY: {'✅ Found' if os.getenv('GOOGLE_AI_API_KEY') else '❌ Not found'}")
            
            if not api_key:
                print("⚠️ No Gemini API key found in environment variables")
                print("   Please set one of: GEMINI_API_KEY, GOOGLE_API_KEY, or GOOGLE_AI_API_KEY")
                self.gemini_available = False
                return
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ Gemini AI model initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            self.gemini_available = False
            self.model = None
    
    def get_therapeutic_strategy(self, emotion: str, confidence: float, 
                               available_moods: List[str], track_samples: List[Dict]) -> Dict[str, Any]:
        """Use AI to determine the best therapeutic music strategy"""
        
        if not self.gemini_available or not self.model:
            return self._fallback_therapeutic_strategy(emotion, available_moods)
        
        try:
            # Create context for AI
            prompt = f"""
You are a music therapy expert cross-checking our therapeutic music recommendations.

USER EMOTION: "{emotion}" (confidence: {confidence:.1%})

OUR DATASET MOODS ONLY: {', '.join(available_moods)}

SAMPLE TRACKS FROM OUR DATASET:
{self._format_track_samples(track_samples[:5])}

TASK: Cross-check and select the BEST mood from OUR DATASET ONLY for therapeutic relief.

STRICT REQUIREMENTS:
1. ONLY choose from our available moods: {', '.join(available_moods)}
2. NO external suggestions - work with what we have
3. For negative emotions (angry, sad, fear): recommend calming/uplifting moods from our dataset
4. For positive emotions: maintain/enhance using our available moods
5. Provide therapeutic reasoning using our dataset context

Respond in this EXACT JSON format:
{{
  "target_mood": "MUST be exactly one from: {', '.join(available_moods)}",
  "reasoning": "why this mood from our dataset will help therapeutically",
  "strategy": "immediate_relief" or "gradual_transition" or "mood_enhancement", 
  "therapeutic_benefit": "specific benefit using our dataset's mood",
  "confidence": 0.0-1.0
}}
"""

            response = self.model.generate_content(prompt)
            result = self._parse_ai_response(response.text)
            
            if result and self._validate_ai_result(result, available_moods):
                print(f"🤖 AI cross-check approved: {result['target_mood']} - {result['reasoning']}")
                return result
            else:
                print("⚠️ AI response invalid or not from our dataset, using fallback")
                return self._fallback_therapeutic_strategy(emotion, available_moods)
                
        except Exception as e:
            print(f"❌ AI therapeutic strategy failed: {e}")
            return self._fallback_therapeutic_strategy(emotion, available_moods)
    
    def _format_track_samples(self, tracks: List[Dict]) -> str:
        """Format track samples for AI context"""
        formatted = []
        for track in tracks:
            track_info = f"'{track.get('Track Name', 'Unknown')}' by {track.get('Artist Name', 'Unknown')} (Mood: {track.get('Mood_Label', 'Unknown')})"
            formatted.append(track_info)
        return '\n'.join(formatted)
    
    def _parse_ai_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse AI response and extract JSON"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                return result
            return None
        except Exception as e:
            print(f"Failed to parse AI response: {e}")
            return None
    
    def _validate_ai_result(self, result: Dict[str, Any], available_moods: List[str]) -> bool:
        """Validate that AI only selected from our dataset moods"""
        if 'target_mood' not in result:
            print("❌ AI response missing target_mood")
            return False
        
        target_mood = result['target_mood']
        if target_mood not in available_moods:
            print(f"❌ AI suggested '{target_mood}' which is not in our dataset: {available_moods}")
            return False
        
        required_fields = ['reasoning', 'strategy', 'therapeutic_benefit', 'confidence']
        for field in required_fields:
            if field not in result:
                print(f"❌ AI response missing required field: {field}")
                return False
        
        print(f"✅ AI response validated - using mood '{target_mood}' from our dataset")
        return True
    
    def _fallback_therapeutic_strategy(self, emotion: str, available_moods: List[str]) -> Dict[str, Any]:
        """Fallback therapeutic strategy using ONLY our dataset moods"""
        emotion_lower = emotion.lower()
        
        # Find best therapeutic mood from OUR DATASET ONLY
        def find_best_mood(preferred_moods):
            for mood in preferred_moods:
                if mood in available_moods:
                    return mood
            return available_moods[0] if available_moods else 'Neutral'
        
        if emotion_lower in ['anger', 'angry', 'rage', 'frustrated']:
            target = find_best_mood(['Calm', 'Neutral', 'Joy'])
            return {
                'target_mood': target,
                'reasoning': f'Using {target} from our dataset to reduce anger and promote peace',
                'strategy': 'immediate_relief',
                'therapeutic_benefit': f'{target} music for anger relief and emotional regulation',
                'confidence': 0.8
            }
        elif emotion_lower in ['sadness', 'sad', 'depressed', 'melancholy']:
            target = find_best_mood(['Joy', 'Excitement', 'Calm', 'Neutral'])
            return {
                'target_mood': target,
                'reasoning': f'Using {target} from our dataset to help overcome sadness',
                'strategy': 'gradual_transition',
                'therapeutic_benefit': f'{target} music for mood uplift and emotional support',
                'confidence': 0.8
            }
        elif emotion_lower in ['fear', 'anxious', 'worried', 'stressed']:
            target = find_best_mood(['Calm', 'Neutral', 'Joy'])
            return {
                'target_mood': target,
                'reasoning': f'Using {target} from our dataset to reduce anxiety and stress',
                'strategy': 'immediate_relief',
                'therapeutic_benefit': f'{target} music for anxiety relief and relaxation',
                'confidence': 0.8
            }
        else:
            # Positive or neutral emotions
            target = find_best_mood(['Joy', 'Excitement', 'Calm', 'Neutral'])
            return {
                'target_mood': target,
                'reasoning': f'Using {target} from our dataset to maintain positive emotional state',
                'strategy': 'mood_enhancement',
                'therapeutic_benefit': f'{target} music for emotional balance and well-being',
                'confidence': 0.7
            }

class AdvancedMusicRecommendationEngine:
    """🎵 ADVANCED Music recommendation engine with 4 strategies + ML models"""
    
    def __init__(self, dataset_path: str = None, model_dir: str = "models"):
        # Use absolute path to your real CSV dataset
        if dataset_path is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent  # Go up to project root
            dataset_path = project_root / "datasets" / "therapeutic_music_enriched.csv"
        
        self.dataset_path = Path(dataset_path)
        print(f"🎵 Looking for dataset at: {self.dataset_path}")
        print(f"🔍 Dataset exists: {self.dataset_path.exists()}")
        self.model_dir = Path(model_dir)
        self.music_df = None
        
        # ML Model components
        self.recommendation_model = None
        self.feature_scaler = None
        self.label_encoder = None
        
        # Session tracking and analytics
        self.recommendation_history = deque(maxlen=1000)
        self.session_recommendations = defaultdict(list)
        
        # 🤖 AI-POWERED THERAPEUTIC ADVISOR
        self.ai_advisor = AITherapeuticAdvisor()
        print(f"🧠 AI Therapeutic Advisor: {'✅ Active' if self.ai_advisor.gemini_available else '❌ Fallback mode'}")
        
        # 🎯 4 ADVANCED MUSIC STRATEGIES
        self.emotion_music_strategies = {
            'therapeutic': self._therapeutic_strategy,
            'mood_matching': self._mood_matching_strategy,
            'mood_regulation': self._mood_regulation_strategy,
            'adaptive': self._adaptive_strategy
        }
        
        # 🧠 ADVANCED THERAPEUTIC BENEFITS MAPPING
        self.therapeutic_benefits = {
            'joy': ['Mood Enhancement', 'Energy Boost', 'Social Connection'],
            'sadness': ['Emotional Processing', 'Comfort', 'Catharsis'],
            'anger': ['Tension Release', 'Calming', 'Emotional Regulation'],
            'fear': ['Anxiety Relief', 'Reassurance', 'Confidence Building'],
            'neutral': ['General Wellness', 'Relaxation', 'Focus'],
            'excitement': ['Energy Channeling', 'Celebration', 'Motivation'],
            'calm': ['Stress Relief', 'Meditation', 'Sleep Aid'],
            'surprise': ['Excitement', 'Joy', 'Energy'],
            'disgust': ['Calming', 'Neutral', 'Relaxation']
        }
        
        # 🎼 AUDIO FEATURE PREFERENCES BY EMOTION
        self.emotion_audio_preferences = {
            'joy': {'valence': (0.6, 1.0), 'energy': (0.5, 1.0), 'tempo': (100, 180)},
            'sadness': {'valence': (0.0, 0.4), 'energy': (0.0, 0.5), 'tempo': (60, 100)},
            'anger': {'valence': (0.0, 0.6), 'energy': (0.6, 1.0), 'tempo': (120, 200)},
            'fear': {'valence': (0.3, 0.7), 'energy': (0.2, 0.6), 'tempo': (70, 120)},
            'neutral': {'valence': (0.4, 0.7), 'energy': (0.3, 0.7), 'tempo': (80, 130)},
            'excitement': {'valence': (0.7, 1.0), 'energy': (0.7, 1.0), 'tempo': (120, 200)},
            'calm': {'valence': (0.4, 0.8), 'energy': (0.1, 0.4), 'tempo': (50, 90)}
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the complete recommendation system"""
        try:
            # Load music dataset
            if self.dataset_path.exists():
                self.music_df = pd.read_csv(self.dataset_path)
                print(f"✅ Loaded music dataset: {len(self.music_df)} tracks")
                
                # Ensure required columns exist
                required_columns = ['Track Name', 'Artist Name', 'Mood_Label']
                missing_columns = [col for col in required_columns if col not in self.music_df.columns]
                if missing_columns:
                    print(f"⚠️ Missing columns in dataset: {missing_columns}")
                else:
                    # Show available moods for debugging
                    available_moods = self.music_df['Mood_Label'].unique()
                    print(f"📊 Available moods in dataset: {list(available_moods)}")
                    
                    # Show mood distribution
                    mood_counts = self.music_df['Mood_Label'].value_counts()
                    print(f"📈 Mood distribution:")
                    for mood, count in mood_counts.head(10).items():
                        print(f"   {mood}: {count} songs")
            else:
                print(f"❌ Music dataset not found: {self.dataset_path}")
                print(f"❌ CANNOT PROCEED WITHOUT REAL DATASET")
                self.music_df = None
                return
            
            # Load trained ML models if available
            self._load_trained_model()
            
        except Exception as e:
            print(f"❌ Error initializing recommendation system: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_trained_model(self):
        """Load trained ML recommendation models"""
        try:
            if not self.model_dir.exists():
                return
            
            # Find the most recent model
            model_files = list(self.model_dir.glob("music_recommender_*.pkl"))
            if not model_files:
                return
            
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_model, 'rb') as f:
                model_data = pickle.load(f)
            
            self.recommendation_model = model_data.get('model')
            self.feature_scaler = model_data.get('scaler')
            self.label_encoder = model_data.get('label_encoder')
            
            logging.info(f"Loaded trained model: {latest_model}")
            
        except Exception as e:
            logging.error(f"Error loading trained model: {e}")
    
    def get_recommendations_for_emotion(self, session_id: str, emotion: str, confidence: float,
                                      strategy: str = 'adaptive', num_tracks: int = 10) -> RecommendationSet:
        """🎯 MAIN: Get advanced recommendations using 4 strategies"""
        
        # Select recommendation strategy
        strategy_func = self.emotion_music_strategies.get(strategy, self._adaptive_strategy)
        
        # Generate recommendations using selected strategy
        recommendations = strategy_func(emotion, confidence, session_id, num_tracks)
        
        # Create recommendation set with metadata
        rec_set = RecommendationSet(
            session_id=session_id,
            target_emotion=emotion,
            emotion_confidence=confidence,
            modality_sources={'facial': True, 'textual': True},  # Assume both available
            recommendations=recommendations,
            generation_time=datetime.now(),
            recommendation_strategy=strategy,
            context_metadata={'advanced_engine': True, 'strategy_count': len(self.emotion_music_strategies)}
        )
        
        # Store in history for analytics
        self.recommendation_history.append(rec_set)
        self.session_recommendations[session_id].append(rec_set)
        
        return rec_set
    
    def _therapeutic_strategy(self, emotion: str, confidence: float, 
                            session_id: str, num_recommendations: int) -> List[MusicRecommendation]:
        """🏥 Therapeutic approach - music for emotional healing"""
        
        if self.music_df is None or self.music_df.empty:
            print(f"❌ No music dataset available - cannot generate recommendations")
            return []
        
        # Get therapeutic benefits for this emotion
        benefits = self.therapeutic_benefits.get(emotion, ['General Wellness'])
        
        # Filter music by therapeutic benefits if column exists
        therapeutic_tracks = self.music_df.copy()
        if 'Mental_Health_Benefit' in self.music_df.columns:
            therapeutic_tracks = self.music_df[
                self.music_df['Mental_Health_Benefit'].isin(benefits)
            ].copy()
        
        # Also filter by emotion if available
        if 'Mood_Label' in therapeutic_tracks.columns:
            emotion_tracks = therapeutic_tracks[therapeutic_tracks['Mood_Label'] == emotion]
            if len(emotion_tracks) > 0:
                therapeutic_tracks = emotion_tracks
        
        # Score based on audio features preference
        therapeutic_tracks = self._score_tracks_by_audio_features(therapeutic_tracks, emotion)
        
        # Select top tracks
        top_tracks = therapeutic_tracks.head(num_recommendations)
        
        recommendations = []
        for _, track in top_tracks.iterrows():
            rec = MusicRecommendation(
                track_name=track.get('Track Name', 'Unknown Track'),
                artist_name=track.get('Artist Name', 'Unknown Artist'),
                emotion_target=emotion,
                confidence_score=confidence * 0.9,  # Slight confidence reduction for therapeutic
                therapeutic_benefit=track.get('Mental_Health_Benefit', benefits[0]),
                audio_features=self._extract_audio_features(track),
                recommendation_reason=f"Therapeutic support for {emotion}",
                timestamp=datetime.now(),
                session_id=session_id,
                recommendation_strategy="therapeutic",
                # 🎵 POPULATE DATASET FIELDS
                album=track.get('Album', 'Unknown Album'),
                track_popularity=int(track.get('Track Popularity', 0)),
                artist_popularity=int(track.get('Artist Popularity', 0)),
                musical_features=track.get('Musical_Features', 'Unknown'),
                artist_genres=track.get('Artist Genres', 'Unknown'),
                mental_health_benefit=track.get('Mental_Health_Benefit', 'General Wellness'),
                duration_ms=int(track.get('Duration (ms)', 0)),
                danceability=float(track.get('Danceability', 0.0)),
                energy=float(track.get('Energy', 0.0)),
                valence=float(track.get('Valence', 0.0)),
                tempo=float(track.get('Tempo', 0.0)),
                # Additional audio features
                key=int(track.get('Key', 0)),
                loudness=float(track.get('Loudness', 0.0)),
                mode=int(track.get('Mode', 0)),
                speechiness=float(track.get('Speechiness', 0.0)),
                acousticness=float(track.get('Acousticness', 0.0)),
                instrumentalness=float(track.get('Instrumentalness', 0.0)),
                liveness=float(track.get('Liveness', 0.0))
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _mood_matching_strategy(self, emotion: str, confidence: float,
                              session_id: str, num_recommendations: int) -> List[MusicRecommendation]:
        """🎭 Mood matching - music that matches current emotional state"""
        
        if self.music_df is None or self.music_df.empty:
            print(f"❌ No music dataset available - cannot generate recommendations")
            return []
        
        print(f"🎭 Mood matching for emotion: {emotion}")
        print(f"📊 Dataset has {len(self.music_df)} total tracks")
        
        # Filter by emotion match (case-insensitive and flexible)
        matching_tracks = pd.DataFrame()
        if 'Mood_Label' in self.music_df.columns:
            # Try exact match first
            exact_match = self.music_df[self.music_df['Mood_Label'].str.lower() == emotion.lower()].copy()
            
            if not exact_match.empty:
                matching_tracks = exact_match
                print(f"✅ Found {len(matching_tracks)} exact matches for '{emotion}'")
            else:
                # Try partial matching for similar emotions
                emotion_mapping = {
                    'angry': ['anger', 'rage', 'mad', 'irritated'],
                    'anger': ['angry', 'rage', 'mad', 'irritated'],
                    'sad': ['sadness', 'melancholy', 'sorrow', 'grief'],
                    'sadness': ['sad', 'melancholy', 'sorrow', 'grief'],
                    'happy': ['joy', 'happiness', 'cheerful', 'upbeat'],
                    'joy': ['happy', 'happiness', 'cheerful', 'upbeat'],
                    'excited': ['excitement', 'energetic', 'thrilled'],
                    'excitement': ['excited', 'energetic', 'thrilled'],
                    'calm': ['peaceful', 'relaxed', 'serene', 'neutral'],
                    'neutral': ['calm', 'peaceful', 'balanced']
                }
                
                related_emotions = emotion_mapping.get(emotion.lower(), [emotion.lower()])
                for related in related_emotions:
                    partial_match = self.music_df[self.music_df['Mood_Label'].str.lower().str.contains(related, na=False)].copy()
                    if not partial_match.empty:
                        matching_tracks = partial_match
                        print(f"✅ Found {len(matching_tracks)} partial matches for '{related}'")
                        break
        
        if matching_tracks.empty:
            print(f"⚠️ No mood matches found, using audio feature matching...")
            # Fallback to audio feature matching
            matching_tracks = self._find_tracks_by_audio_features(emotion, num_recommendations * 2)
            if matching_tracks.empty:
                print(f"❌ No matches found for emotion '{emotion}' in real dataset")
                return []
        
        # Score and rank by popularity if available
        matching_tracks = self._score_tracks_by_popularity(matching_tracks)
        top_tracks = matching_tracks.head(num_recommendations)
        
        print(f"🎵 Generating {len(top_tracks)} recommendations from matching tracks")
        
        recommendations = []
        for _, track in top_tracks.iterrows():
            rec = MusicRecommendation(
                track_name=track.get('Track Name', 'Unknown Track'),
                artist_name=track.get('Artist Name', 'Unknown Artist'),
                emotion_target=emotion,
                confidence_score=confidence,
                therapeutic_benefit=track.get('Mental_Health_Benefit', 'Mood Matching'),
                audio_features=self._extract_audio_features(track),
                recommendation_reason=f"Matches current {emotion} mood",
                timestamp=datetime.now(),
                session_id=session_id,
                recommendation_strategy="mood_matching",
                # 🎵 POPULATE DATASET FIELDS
                album=track.get('Album', 'Unknown Album'),
                track_popularity=int(track.get('Track Popularity', 0)),
                artist_popularity=int(track.get('Artist Popularity', 0)),
                musical_features=track.get('Musical_Features', 'Unknown'),
                artist_genres=track.get('Artist Genres', 'Unknown'),
                mental_health_benefit=track.get('Mental_Health_Benefit', 'General Wellness'),
                duration_ms=int(track.get('Duration (ms)', 0)),
                danceability=float(track.get('Danceability', 0.0)),
                energy=float(track.get('Energy', 0.0)),
                valence=float(track.get('Valence', 0.0)),
                tempo=float(track.get('Tempo', 0.0)),
                # Additional audio features
                key=int(track.get('Key', 0)),
                loudness=float(track.get('Loudness', 0.0)),
                mode=int(track.get('Mode', 0)),
                speechiness=float(track.get('Speechiness', 0.0)),
                acousticness=float(track.get('Acousticness', 0.0)),
                instrumentalness=float(track.get('Instrumentalness', 0.0)),
                liveness=float(track.get('Liveness', 0.0))
            )
            recommendations.append(rec)
        
        print(f"✅ Created {len(recommendations)} music recommendations")
        return recommendations
    
    def _mood_regulation_strategy(self, emotion: str, confidence: float,
                                session_id: str, num_recommendations: int) -> List[MusicRecommendation]:
        """🤖 AI-powered mood regulation - guide toward desired emotional state for therapeutic relief"""
        
        print(f"🧘 AI MOOD REGULATION: Providing therapeutic relief for '{emotion}'")
        
        if self.music_df is None or self.music_df.empty:
            print(f"❌ No music dataset available - cannot generate recommendations")
            return []
        
        # 🤖 GET AI THERAPEUTIC STRATEGY
        available_moods = self.music_df['Mood_Label'].unique().tolist() if 'Mood_Label' in self.music_df.columns else []
        track_samples = self.music_df.head(10).to_dict('records') if not self.music_df.empty else []
        
        ai_strategy = self.ai_advisor.get_therapeutic_strategy(
            emotion=emotion,
            confidence=confidence,
            available_moods=available_moods,
            track_samples=track_samples
        )
        
        target_emotion = ai_strategy['target_mood']
        reasoning = ai_strategy['reasoning']
        therapeutic_benefit = ai_strategy['therapeutic_benefit']
        
        print(f"🎯 AI recommends '{target_emotion}' music")
        print(f"💡 Reasoning: {reasoning}")
        
        # 🎵 ENHANCED THERAPEUTIC MUSIC SELECTION
        regulating_tracks = pd.DataFrame()
        
        # First priority: Find tracks with exact mood match for therapeutic target
        if 'Mood_Label' in self.music_df.columns:
            regulating_tracks = self.music_df[self.music_df['Mood_Label'] == target_emotion].copy()
            print(f"Found {len(regulating_tracks)} tracks with mood label '{target_emotion}'")
        
        # Second priority: Use audio features to find therapeutic tracks
        if regulating_tracks.empty or len(regulating_tracks) < num_recommendations:
            print(f"Expanding search using audio features for '{target_emotion}'")
            audio_feature_tracks = self._find_tracks_by_audio_features(target_emotion, num_recommendations * 3)
            
            if regulating_tracks.empty:
                regulating_tracks = audio_feature_tracks
            else:
                # Combine and remove duplicates
                regulating_tracks = pd.concat([regulating_tracks, audio_feature_tracks]).drop_duplicates()
        
        # 🏥 THERAPEUTIC FILTERING - Remove tracks that might worsen the condition
        if target_emotion == 'calm' and not regulating_tracks.empty:
            # For calming: avoid high-energy, aggressive music
            if 'Energy' in regulating_tracks.columns:
                regulating_tracks = regulating_tracks[regulating_tracks['Energy'] <= 0.7]
            if 'Valence' in regulating_tracks.columns:
                regulating_tracks = regulating_tracks[regulating_tracks['Valence'] >= 0.3]
                
        elif target_emotion == 'joy' and not regulating_tracks.empty:
            # For joy/uplift: favor positive, energetic music
            if 'Valence' in regulating_tracks.columns:
                regulating_tracks = regulating_tracks[regulating_tracks['Valence'] >= 0.5]
            if 'Energy' in regulating_tracks.columns:
                regulating_tracks = regulating_tracks[regulating_tracks['Energy'] >= 0.4]
        
        print(f"After therapeutic filtering: {len(regulating_tracks)} suitable tracks")
        
        # Score and select
        regulating_tracks = self._score_tracks_by_audio_features(regulating_tracks, target_emotion)
        top_tracks = regulating_tracks.head(num_recommendations)
        
        # 🎯 CREATE AI-GUIDED THERAPEUTIC RECOMMENDATIONS
        recommendations = []
        for _, track in top_tracks.iterrows():
            rec = MusicRecommendation(
                track_name=track.get('Track Name', 'Unknown Track'),
                artist_name=track.get('Artist Name', 'Unknown Artist'),
                emotion_target=target_emotion,
                confidence_score=confidence * ai_strategy['confidence'],  # Use AI confidence
                therapeutic_benefit=therapeutic_benefit,  # From AI
                audio_features=self._extract_audio_features(track),
                recommendation_reason=reasoning,  # From AI
                timestamp=datetime.now(),
                session_id=session_id,
                recommendation_strategy=f"ai_therapeutic_{ai_strategy['strategy']}",
                # 🎵 POPULATE DATASET FIELDS
                album=track.get('Album', 'Unknown Album'),
                track_popularity=int(track.get('Track Popularity', 0)),
                artist_popularity=int(track.get('Artist Popularity', 0)),
                musical_features=track.get('Musical_Features', 'Unknown'),
                artist_genres=track.get('Artist Genres', 'Unknown'),
                mental_health_benefit=track.get('Mental_Health_Benefit', 'General Wellness'),
                duration_ms=int(track.get('Duration (ms)', 0)),
                danceability=float(track.get('Danceability', 0.0)),
                energy=float(track.get('Energy', 0.0)),
                valence=float(track.get('Valence', 0.0)),
                tempo=float(track.get('Tempo', 0.0)),
                # Additional audio features
                key=int(track.get('Key', 0)),
                loudness=float(track.get('Loudness', 0.0)),
                mode=int(track.get('Mode', 0)),
                speechiness=float(track.get('Speechiness', 0.0)),
                acousticness=float(track.get('Acousticness', 0.0)),
                instrumentalness=float(track.get('Instrumentalness', 0.0)),
                liveness=float(track.get('Liveness', 0.0))
            )
            recommendations.append(rec)
        
        print(f"✅ Generated {len(recommendations)} AI-guided therapeutic recommendations")
        
        return recommendations
    
    def _adaptive_strategy(self, emotion: str, confidence: float,
                         session_id: str, num_recommendations: int) -> List[MusicRecommendation]:
        """🧠 Adaptive strategy - ALWAYS provides therapeutic relief, never mood matching"""
        
        print(f"🎯 ADAPTIVE STRATEGY: Providing therapeutic relief for emotion '{emotion}'")
        
        # 🎭 ALWAYS use mood regulation for therapeutic relief
        # If someone is angry, sad, fearful, etc. - give them relief!
        negative_emotions = ['anger', 'angry', 'sadness', 'sad', 'fear', 'afraid', 'disgust', 'worried', 'anxious', 'stressed']
        
        if emotion.lower() in negative_emotions:
            print(f"🏥 Detected negative emotion '{emotion}' - providing therapeutic relief")
            # Always use mood regulation to help the user feel better
            return self._mood_regulation_strategy(emotion, confidence, session_id, num_recommendations)
        elif emotion.lower() in ['neutral', 'calm']:
            print(f"😌 Neutral/calm emotion '{emotion}' - maintaining peaceful state")
            # For neutral/calm, provide peaceful/relaxing music to maintain the state
            return self._therapeutic_strategy(emotion, confidence, session_id, num_recommendations)
        else:
            print(f"😊 Positive emotion '{emotion}' - enhancing positive mood")
            # For positive emotions (joy, excitement), enhance but also balance
            positive_recs = self._therapeutic_strategy(emotion, confidence, session_id, num_recommendations // 2)
            calming_recs = self._mood_regulation_strategy(emotion, confidence, session_id, num_recommendations - len(positive_recs))
            return positive_recs + calming_recs
    
    def _find_tracks_by_audio_features(self, emotion: str, limit: int) -> pd.DataFrame:
        """Find tracks matching audio feature preferences for emotion"""
        if self.music_df is None or self.music_df.empty:
            return pd.DataFrame()
        
        preferences = self.emotion_audio_preferences.get(emotion, {})
        filtered_df = self.music_df.copy()
        
        # Apply audio feature filters
        for feature, (min_val, max_val) in preferences.items():
            if feature in filtered_df.columns:
                feature_col = feature.title()  # Handle capitalization
                if feature_col in filtered_df.columns:
                    filtered_df = filtered_df[
                        (filtered_df[feature_col] >= min_val) & 
                        (filtered_df[feature_col] <= max_val)
                    ]
        
        return filtered_df.head(limit)
    
    def _score_tracks_by_audio_features(self, tracks_df: pd.DataFrame, emotion: str) -> pd.DataFrame:
        """Score tracks based on how well they match audio feature preferences"""
        if tracks_df.empty:
            return tracks_df
        
        preferences = self.emotion_audio_preferences.get(emotion, {})
        tracks_df = tracks_df.copy()
        tracks_df['feature_score'] = 0.0
        
        for feature, (min_val, max_val) in preferences.items():
            feature_col = feature.title()
            if feature_col in tracks_df.columns:
                # Score based on how well the feature falls within preferred range
                mid_point = (min_val + max_val) / 2
                range_size = max_val - min_val
                
                # Distance from ideal (mid-point), normalized by range
                distances = np.abs(tracks_df[feature_col] - mid_point) / (range_size / 2)
                feature_scores = np.maximum(0, 1 - distances)  # Closer = higher score
                
                tracks_df['feature_score'] += feature_scores
        
        # Normalize by number of features scored
        if len(preferences) > 0:
            tracks_df['feature_score'] /= len(preferences)
        
        return tracks_df.sort_values('feature_score', ascending=False)
    
    def _score_tracks_by_popularity(self, tracks_df: pd.DataFrame) -> pd.DataFrame:
        """Score tracks by popularity metrics"""
        if tracks_df.empty:
            return tracks_df
        
        tracks_df = tracks_df.copy()
        
        # Use popularity columns if available
        popularity_columns = ['Track Popularity', 'Artist Popularity']
        available_pop_cols = [col for col in popularity_columns if col in tracks_df.columns]
        
        if available_pop_cols:
            # Average popularity score
            tracks_df['popularity_score'] = tracks_df[available_pop_cols].mean(axis=1)
            return tracks_df.sort_values('popularity_score', ascending=False)
        else:
            # Random shuffle if no popularity data
            return tracks_df.sample(frac=1.0) if len(tracks_df) > 1 else tracks_df
    
    def _get_therapeutic_benefit_description(self, from_emotion: str, to_emotion: str) -> str:
        """Generate specific therapeutic benefit description"""
        benefit_map = {
            ('anger', 'calm'): 'Anger relief through calming harmonies and peaceful rhythms',
            ('angry', 'calm'): 'Anger relief through calming harmonies and peaceful rhythms',
            ('sadness', 'joy'): 'Mood uplift from sadness to joy through positive energy music',
            ('sad', 'joy'): 'Mood uplift from sadness to joy through positive energy music',
            ('fear', 'calm'): 'Anxiety relief and confidence building through soothing melodies',
            ('afraid', 'calm'): 'Anxiety relief and confidence building through soothing melodies',
            ('anxious', 'calm'): 'Anxiety relief and stress reduction through calming music',
            ('worried', 'calm'): 'Worry relief and mental peace through therapeutic sounds',
            ('stressed', 'calm'): 'Stress relief and relaxation through peaceful music',
            ('disgust', 'neutral'): 'Emotional cleansing and neutralizing through balanced music',
            ('neutral', 'joy'): 'Mood enhancement from neutral to positive state',
        }
        
        key = (from_emotion.lower(), to_emotion.lower())
        return benefit_map.get(key, f'Therapeutic transition from {from_emotion} to {to_emotion}')
    
    def _get_therapeutic_reason(self, from_emotion: str, to_emotion: str) -> str:
        """Generate specific therapeutic reasoning"""
        reason_map = {
            ('anger', 'calm'): 'Providing calming music to reduce anger and promote inner peace',
            ('angry', 'calm'): 'Providing calming music to reduce anger and promote inner peace',
            ('sadness', 'joy'): 'Uplifting music to help overcome sadness and restore happiness',
            ('sad', 'joy'): 'Uplifting music to help overcome sadness and restore happiness',
            ('fear', 'calm'): 'Comforting music to alleviate fear and build confidence',
            ('afraid', 'calm'): 'Comforting music to alleviate fear and build confidence',
            ('anxious', 'calm'): 'Soothing music to reduce anxiety and promote relaxation',
            ('worried', 'calm'): 'Peaceful music to ease worries and calm the mind',
            ('stressed', 'calm'): 'Relaxing music to relieve stress and restore balance',
            ('disgust', 'neutral'): 'Neutral music to cleanse negative feelings',
            ('neutral', 'joy'): 'Positive music to enhance mood and bring joy',
        }
        
        key = (from_emotion.lower(), to_emotion.lower())
        return reason_map.get(key, f'Therapeutic music to guide from {from_emotion} toward {to_emotion}')
    
    def _extract_audio_features(self, track_row) -> Dict[str, float]:
        """Extract audio features from track data"""
        feature_columns = [
            'Danceability', 'Energy', 'Valence', 'Tempo', 'Loudness',
            'Acousticness', 'Instrumentalness', 'Liveness', 'Speechiness'
        ]
        
        features = {}
        for col in feature_columns:
            if col in track_row:
                try:
                    features[col.lower()] = float(track_row[col])
                except (ValueError, TypeError):
                    features[col.lower()] = 0.0
        
        return features
    
    # REMOVED: No fallback recommendations - use REAL data only
    
    def get_session_recommendation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get recommendation history for a session"""
        session_recs = self.session_recommendations.get(session_id, [])
        return [rec.to_dict() for rec in session_recs]
    
    def get_recommendation_analytics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics about recommendations"""
        if session_id:
            # Session-specific analytics
            session_recs = self.session_recommendations.get(session_id, [])
            
            if not session_recs:
                return {'error': 'No recommendations found for session'}
            
            emotions = [rec.target_emotion for rec in session_recs]
            strategies = [rec.recommendation_strategy for rec in session_recs]
            confidences = [rec.emotion_confidence for rec in session_recs]
            
            return {
                'session_id': session_id,
                'total_recommendations': len(session_recs),
                'emotion_distribution': {emotion: emotions.count(emotion) for emotion in set(emotions)},
                'strategy_usage': {strategy: strategies.count(strategy) for strategy in set(strategies)},
                'average_confidence': np.mean(confidences) if confidences else 0.0,
                'recommendation_timeline': [
                    {
                        'timestamp': rec.generation_time.isoformat(),
                        'emotion': rec.target_emotion,
                        'strategy': rec.recommendation_strategy,
                        'track_count': len(rec.recommendations)
                    }
                    for rec in session_recs
                ]
            }
        else:
            # Global analytics
            all_recs = list(self.recommendation_history)
            
            if not all_recs:
                return {'error': 'No recommendations found'}
            
            emotions = [rec.target_emotion for rec in all_recs]
            strategies = [rec.recommendation_strategy for rec in all_recs]
            
            return {
                'total_recommendation_sets': len(all_recs),
                'unique_sessions': len(self.session_recommendations),
                'emotion_distribution': {emotion: emotions.count(emotion) for emotion in set(emotions)},
                'strategy_usage': {strategy: strategies.count(strategy) for strategy in set(strategies)},
                'most_common_emotion': max(set(emotions), key=emotions.count) if emotions else None
            }
    
    def clear_session_data(self, session_id: str):
        """Clear recommendation data for a session"""
        if session_id in self.session_recommendations:
            del self.session_recommendations[session_id]
            logging.info(f"Cleared recommendation history for session {session_id}")

class UnifiedEmotionMusicSystem:
    """🎯 Main unified system combining everything"""
    
    def __init__(self, silent: bool = True):
        self.silent = silent
        self.firebase_client = None
        self.project_root = self._find_project_root()
        
        # 🎭 NEW: Import real emotion combiner
        try:
            from real_emotion_combiner import get_combined_emotion
            self.real_emotion_combiner = get_combined_emotion
            self.combiner_available = True
            if not silent:
                print("✅ Real emotion combiner integrated")
        except ImportError as e:
            self.real_emotion_combiner = None
            self.combiner_available = False
            if not silent:
                print(f"⚠️ Real emotion combiner not available: {e}")
        
        # Initialize components
        self.fusion_engine = AdvancedEmotionFusionEngine()
        self.music_engine = AdvancedMusicRecommendationEngine()
        
        # Session management
        self.session_data = {}
        
        # Initialize Firebase
        if FIREBASE_AVAILABLE:
            self._init_firebase()
        
        if not silent:
            print("🎯 Y.M.I.R Unified Emotion-Music System initialized")
            print(f"🔥 Firebase: {'✅ Connected' if self.firebase_client else '❌ Unavailable'}")
            print(f"🧠 Fusion strategies: {list(self.fusion_engine.strategies.keys())}")
            print(f"🎭 Multi-emotion support: {'✅ Available' if self.combiner_available else '❌ Unavailable'}")
    
    def _find_project_root(self) -> Path:
        """Find project root"""
        current = Path(__file__).parent
        for _ in range(5):
            if (current / "app.py").exists() or (current / "firebase_credentials.json").exists():
                return current
            current = current.parent
        return Path("../../../")
    
    def _init_firebase(self):
        """Initialize Firebase"""
        try:
            if firebase_admin._apps:
                self.firebase_client = firestore.client()
                return
            
            cred_paths = [
                self.project_root / "firebase_credentials.json",
                Path("firebase_credentials.json")
            ]
            
            for cred_path in cred_paths:
                if cred_path.exists():
                    cred = credentials.Certificate(str(cred_path))
                    firebase_admin.initialize_app(cred)
                    self.firebase_client = firestore.client()
                    return
        except Exception:
            pass
    
    def _get_multi_emotion_music_recommendations(self, session_id: str, dominant_emotion: str, 
                                               confidence: float, top_emotions: List[Tuple[str, float]], 
                                               is_multi_emotion: bool, num_tracks: int) -> List[Dict[str, Any]]:
        """🎵 Generate music recommendations using multi-emotion data"""
        try:
            if is_multi_emotion and len(top_emotions) > 1:
                # Multi-emotion blending: 60% primary, 30% secondary, 10% tertiary
                primary_emotion, primary_confidence = top_emotions[0]
                secondary_emotion, secondary_confidence = top_emotions[1] if len(top_emotions) > 1 else (primary_emotion, 0)
                tertiary_emotion, tertiary_confidence = top_emotions[2] if len(top_emotions) > 2 else (primary_emotion, 0)
                
                if not self.silent:
                    print(f"🎵 Multi-emotion music blending:")
                    print(f"   Primary (60%): {primary_emotion} ({primary_confidence:.2f})")
                    print(f"   Secondary (30%): {secondary_emotion} ({secondary_confidence:.2f})")
                    print(f"   Tertiary (10%): {tertiary_emotion} ({tertiary_confidence:.2f})")
                
                # Get recommendations for each emotion with proportional tracks
                primary_tracks = int(num_tracks * 0.6)
                secondary_tracks = int(num_tracks * 0.3)
                tertiary_tracks = num_tracks - primary_tracks - secondary_tracks
                
                all_recommendations = []
                
                # Primary emotion recommendations
                if primary_tracks > 0:
                    primary_rec_set = self.music_engine.get_recommendations_for_emotion(
                        session_id, primary_emotion, primary_confidence, "mood_matching", primary_tracks
                    )
                    if primary_rec_set and primary_rec_set.recommendations:
                        for rec in primary_rec_set.recommendations:
                            rec_dict = rec.to_dict()
                            rec_dict['emotion_source'] = 'primary'
                            rec_dict['emotion_weight'] = 0.6
                            rec_dict['source_emotion'] = primary_emotion
                            all_recommendations.append(rec_dict)
                
                # Secondary emotion recommendations
                if secondary_tracks > 0 and secondary_emotion != primary_emotion:
                    secondary_rec_set = self.music_engine.get_recommendations_for_emotion(
                        session_id, secondary_emotion, secondary_confidence, "mood_matching", secondary_tracks
                    )
                    if secondary_rec_set and secondary_rec_set.recommendations:
                        for rec in secondary_rec_set.recommendations:
                            rec_dict = rec.to_dict()
                            rec_dict['emotion_source'] = 'secondary'
                            rec_dict['emotion_weight'] = 0.3
                            rec_dict['source_emotion'] = secondary_emotion
                            all_recommendations.append(rec_dict)
                
                # Tertiary emotion recommendations
                if tertiary_tracks > 0 and tertiary_emotion != primary_emotion and tertiary_emotion != secondary_emotion:
                    tertiary_rec_set = self.music_engine.get_recommendations_for_emotion(
                        session_id, tertiary_emotion, tertiary_confidence, "therapeutic", tertiary_tracks
                    )
                    if tertiary_rec_set and tertiary_rec_set.recommendations:
                        for rec in tertiary_rec_set.recommendations:
                            rec_dict = rec.to_dict()
                            rec_dict['emotion_source'] = 'tertiary'
                            rec_dict['emotion_weight'] = 0.1
                            rec_dict['source_emotion'] = tertiary_emotion
                            all_recommendations.append(rec_dict)
                
                # Shuffle to mix emotions throughout the playlist
                import random
                random.shuffle(all_recommendations)
                
                if not self.silent:
                    print(f"🎵 Generated {len(all_recommendations)} multi-emotion recommendations")
                
                return all_recommendations[:num_tracks]
            
            else:
                # Single emotion - use existing logic
                if not self.silent:
                    print(f"🎵 Single emotion recommendations for: {dominant_emotion}")
                
                rec_set = self.music_engine.get_recommendations_for_emotion(
                    session_id, dominant_emotion, confidence, "adaptive", num_tracks
                )
                
                if rec_set and rec_set.recommendations:
                    recommendations = []
                    for rec in rec_set.recommendations:
                        rec_dict = rec.to_dict()
                        rec_dict['emotion_source'] = 'single'
                        rec_dict['emotion_weight'] = 1.0
                        rec_dict['source_emotion'] = dominant_emotion
                        recommendations.append(rec_dict)
                    return recommendations
                
                return []
                
        except Exception as e:
            if not self.silent:
                print(f"❌ Error generating multi-emotion music recommendations: {e}")
            
            # Return empty if no dataset available
            print(f"❌ Cannot generate recommendations without real dataset")
            return []
    
    def get_latest_facial_emotions(self, minutes_back: int = 10) -> Optional[Dict[str, Any]]:
        """Get latest facial emotions from Firebase"""
        if not self.firebase_client:
            return None
        
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes_back)
            emotions_ref = self.firebase_client.collection('emotion_readings')
            query = emotions_ref.where('timestamp', '>=', cutoff_time).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1)
            
            docs = query.stream()
            for doc in docs:
                data = doc.to_dict()
                emotions_dict = data.get('emotions', {})
                if emotions_dict:
                    dominant = max(emotions_dict.items(), key=lambda x: float(x[1]))
                    emotion = dominant[0]
                    score = float(dominant[1])
                    confidence = score / 100.0 if score > 1.0 else score
                    
                    return {
                        'emotion': emotion,
                        'confidence': confidence,
                        'emotions': emotions_dict,
                        'timestamp': data.get('timestamp')
                    }
            return None
        except Exception:
            return None
    
    def get_latest_text_emotions(self, minutes_back: int = 10) -> Optional[Dict[str, Any]]:
        """Get latest text emotions from JSON files"""
        try:
            chat_patterns = [
                self.project_root / "chat_session_*.json",
                Path("chat_session_*.json")
            ]
            
            latest_file = None
            latest_time = None
            
            for pattern in chat_patterns:
                files = glob.glob(str(pattern))
                for file_path in files:
                    if 'chat_session_' in file_path:
                        try:
                            filename = Path(file_path).name
                            time_part = filename.replace('chat_session_', '').replace('.json', '')
                            file_time = datetime.strptime(time_part, '%Y%m%d_%H%M%S')
                            
                            if latest_time is None or file_time > latest_time:
                                latest_time = file_time
                                latest_file = file_path
                        except ValueError:
                            continue
            
            if not latest_file or not latest_time:
                return None
            
            # Check if recent enough
            if (datetime.now() - latest_time).total_seconds() > (minutes_back * 60):
                return None
            
            # Read file
            with open(latest_file, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
            
            # Find latest emotion
            conversation = chat_data.get('conversation', [])
            for message in reversed(conversation):
                if (message.get('role') == 'user' and 
                    message.get('emotion') and 
                    message.get('emotion') != 'neutral'):
                    
                    return {
                        'emotion': message.get('emotion'),
                        'confidence': message.get('confidence', 0.5),
                        'timestamp': message.get('timestamp')
                    }
            
            return None
        except Exception:
            return None
    
    def get_emotion_and_music(self, session_id: str = "default", minutes_back: int = 10, 
                            strategy: str = 'adaptive', num_tracks: int = 10) -> Optional[UnifiedEmotionResult]:
        """🎯 MAIN FUNCTION: Get current emotion and music recommendations with MULTI-EMOTION support"""
        start_time = datetime.now()
        
        try:
            # 🎭 NEW: Use real emotion combiner for multi-emotion support
            if self.combiner_available:
                if not self.silent:
                    print(f"🎭 Using REAL emotion combiner for multi-emotion analysis...")
                
                emotion_result = self.real_emotion_combiner(minutes_back=minutes_back, strategy=strategy)
                
                if not emotion_result:
                    if not self.silent:
                        print("❌ No emotion detected by real combiner")
                    return None
                
                # Extract emotion data
                emotion = emotion_result['emotion']
                confidence = emotion_result['confidence']
                method = emotion_result['source']
                
                # 🎭 NEW: Multi-emotion data
                is_multi_emotion = emotion_result.get('is_multi_emotion', False)
                top_emotions = emotion_result.get('top_emotions', [(emotion, confidence)])
                fusion_weights = emotion_result.get('fusion_weights', {'facial': 0.5, 'text': 0.5})
                
                if not self.silent:
                    print(f"🎭 Detected: {emotion} (confidence: {confidence:.2f})")
                    print(f"🎭 Multi-emotion: {is_multi_emotion}")
                    if is_multi_emotion and len(top_emotions) > 1:
                        print(f"🎪 Top emotions: {top_emotions[:3]}")
                
                # 🎵 NEW: Enhanced music recommendations using multi-emotion data
                music_recommendations = self._get_multi_emotion_music_recommendations(
                    session_id, emotion, confidence, top_emotions, is_multi_emotion, num_tracks
                )
                
                # Create enhanced result with multi-emotion support
                result = UnifiedEmotionResult(
                    emotion=emotion,
                    confidence=confidence,
                    source=method,
                    strategy=strategy,
                    facial_data=emotion_result.get('facial_data'),
                    text_data=emotion_result.get('text_data'),
                    music_recommendations=music_recommendations,
                    recommendation_strategy="multi_emotion_adaptive",
                    session_id=session_id,
                    timestamp=start_time,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
                
                # Add multi-emotion metadata to result
                result.is_multi_emotion = is_multi_emotion
                result.top_emotions = top_emotions
                result.fusion_weights = fusion_weights
                
            else:
                # Fallback to old method if real combiner not available
                if not self.silent:
                    print("⚠️ Using fallback emotion detection...")
                
                facial_data = self.get_latest_facial_emotions(minutes_back)
                text_data = self.get_latest_text_emotions(minutes_back)
                
                if not facial_data and not text_data:
                    return None
                
                emotion, confidence, method = self.fusion_engine.fuse_emotions(
                    facial_data, text_data, strategy
                )
                
                rec_set = self.music_engine.get_recommendations_for_emotion(
                    session_id, emotion, confidence, "adaptive", num_tracks
                )
                
                music_recommendations = [rec.to_dict() for rec in rec_set.recommendations] if rec_set else []
                
                result = UnifiedEmotionResult(
                    emotion=emotion,
                    confidence=confidence,
                    source=method,
                    strategy=strategy,
                    facial_data=facial_data,
                    text_data=text_data,
                    music_recommendations=music_recommendations,
                    recommendation_strategy="adaptive",
                    session_id=session_id,
                    timestamp=start_time,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            
            # Store in session
            self.session_data[session_id] = result
            
            return result
            
        except Exception as e:
            if not self.silent:
                print(f"Error in get_emotion_and_music: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def get_session_history(self, session_id: str) -> Optional[UnifiedEmotionResult]:
        """Get latest result for session"""
        return self.session_data.get(session_id)

# Global instance
_unified_system = None

def get_emotion_and_music(session_id: str = "default", minutes_back: int = 10, 
                         strategy: str = 'adaptive', num_tracks: int = 100) -> Optional[Dict[str, Any]]:
    """
    🎯 MAIN FUNCTION FOR APP.PY - Enhanced with Multi-Emotion Support
    
    Get current user emotion and music recommendations in one call!
    
    Args:
        session_id: User session identifier
        minutes_back: How many minutes back to look for emotions
        strategy: Fusion strategy ('simple', 'adaptive', 'confidence_based', 'temporal_weighted', 'weighted_average')
        num_tracks: Number of music tracks to recommend (default 100 for carousel)
    
    Returns:
        Dict with emotion and music data or None if no emotions found
        {
            'emotion': str,                    # Current dominant emotion
            'confidence': float,               # 0-1 confidence score
            'source': str,                     # How emotion was determined
            'strategy': str,                   # Fusion strategy used
            'is_multi_emotion': bool,          # 🎭 NEW: Whether multiple emotions detected
            'top_emotions': List[Tuple],       # 🎭 NEW: Top 3 emotions with scores
            'fusion_weights': Dict,            # 🎭 NEW: Facial vs text weights used
            'music_recommendations': [         # List of recommended tracks
                {
                    'track_name': str,         # Track name
                    'artist_name': str,        # Artist name  
                    'album': str,              # Album name
                    'emotion_target': str,     # Target emotion for this track
                    'therapeutic_benefit': str, # Therapeutic benefit
                    'confidence_score': float, # How well it matches emotion
                    'recommendation_reason': str,
                    'emotion_source': str,     # 🎭 NEW: 'primary', 'secondary', 'tertiary', or 'single'
                    'emotion_weight': float,   # 🎭 NEW: Weight in multi-emotion blend
                    'source_emotion': str,     # 🎭 NEW: Which emotion this track addresses
                    'audio_features': Dict,    # Audio features for UI display
                    'musical_features': str,   # Musical description
                    'track_popularity': int,   # Popularity score
                    'artist_popularity': int   # Artist popularity
                }
            ],
            'processing_time_ms': float,       # Processing time
            'timestamp': str                   # When processed
        }
    """
    global _unified_system
    
    if _unified_system is None:
        _unified_system = UnifiedEmotionMusicSystem(silent=True)
    
    result = _unified_system.get_emotion_and_music(session_id, minutes_back, strategy, num_tracks)
    
    if result:
        result_dict = result.to_dict()
        
        # 🎭 Add multi-emotion data to the response
        if hasattr(result, 'is_multi_emotion'):
            result_dict['is_multi_emotion'] = result.is_multi_emotion
        if hasattr(result, 'top_emotions'):
            result_dict['top_emotions'] = result.top_emotions
        if hasattr(result, 'fusion_weights'):
            result_dict['fusion_weights'] = result.fusion_weights
        
        return result_dict
    
    return None

def get_emotion_simple(session_id: str = "default", strategy: str = 'adaptive') -> Optional[str]:
    """
    Simple function - just returns the emotion name
    
    Returns:
        str: emotion name or None
    """
    result = get_emotion_and_music(session_id, strategy=strategy)
    return result['emotion'] if result else None

def get_music_for_emotion(emotion: str, confidence: float = 0.8, num_tracks: int = 10) -> List[Dict[str, Any]]:
    """
    Get music recommendations for a specific emotion
    
    Args:
        emotion: Emotion name
        confidence: Confidence score
        num_tracks: Number of tracks to recommend
    
    Returns:
        List of music recommendations
    """
    global _unified_system
    
    if _unified_system is None:
        _unified_system = UnifiedEmotionMusicSystem(silent=True)
    
    # Use advanced music engine
    rec_set = _unified_system.music_engine.get_recommendations_for_emotion(
        "temp_session", emotion, confidence, "adaptive", num_tracks
    )
    
    return [rec.to_dict() for rec in rec_set.recommendations] if rec_set else []

def get_session_analytics(session_id: str) -> Dict[str, Any]:
    """
    Get analytics for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        Analytics data for the session
    """
    global _unified_system
    
    if _unified_system is None:
        _unified_system = UnifiedEmotionMusicSystem(silent=True)
    
    return _unified_system.music_engine.get_recommendation_analytics(session_id)

def get_music_strategies() -> List[str]:
    """
    Get available music recommendation strategies
    
    Returns:
        List of available strategies
    """
    return ['therapeutic', 'mood_matching', 'mood_regulation', 'adaptive']

def get_emotion_fusion_strategies() -> List[str]:
    """
    Get available emotion fusion strategies
    
    Returns:
        List of available fusion strategies
    """
    return ['simple', 'adaptive', 'confidence_based', 'temporal_weighted', 'weighted_average']

# Test function
def test_unified_system():
    """Test the unified system"""
    print("🧪 Testing Y.M.I.R Unified Emotion-Music System")
    print("=" * 60)
    
    # Test main function
    result = get_emotion_and_music("test_session", strategy='adaptive', num_tracks=10)
    
    if result:
        print(f"✅ EMOTION DETECTED: {result['emotion']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Source: {result['source']}")
        print(f"   Strategy: {result['strategy']}")
        print(f"   Multi-emotion: {result.get('is_multi_emotion', False)}")
        print(f"   Music tracks: {len(result['music_recommendations'])}")
        print(f"   Processing time: {result['processing_time_ms']:.1f}ms")
        
        # Show first 5 tracks with full details
        if result['music_recommendations']:
            print(f"\n🎵 TOP 5 RECOMMENDED SONGS:")
            print("-" * 80)
            for i, track in enumerate(result['music_recommendations'][:30], 1):
                print(f"{i}. 🎵 {track.get('track_name', 'Unknown Track')}")
                print(f"   🎤 Artist: {track.get('artist_name', 'Unknown Artist')}")
                print(f"   💿 Album: {track.get('album', 'Unknown Album')}")
                print(f"   🎭 Mood: {track.get('emotion_target', 'Unknown')}")
                print(f"   💚 Benefit: {track.get('therapeutic_benefit', 'General Wellness')}")
                print(f"   📊 Popularity: Track={track.get('track_popularity', 0)}, Artist={track.get('artist_popularity', 0)}")
                print(f"   🎼 Features: {track.get('musical_features', 'Unknown')}")
                if track.get('emotion_source'):
                    print(f"   🎯 Emotion Source: {track.get('emotion_source')} ({track.get('emotion_weight', 1.0):.1f})")
                print()
    else:
        print("❌ No emotions detected")
    
    # Test simple function
    emotion = get_emotion_simple("test_session")
    print(f"Simple emotion: {emotion}")
    
    # Test music-only function with detailed output
    print(f"\n🎵 TESTING DIRECT MUSIC RECOMMENDATION:")
    print("-" * 50)
    music = get_music_for_emotion("joy", 0.9, 5)
    print(f"Music for joy: {len(music)} tracks")
    
    if music:
        print(f"\n🎵 JOY MUSIC RECOMMENDATIONS:")
        print("-" * 50)
        for i, track in enumerate(music[:3], 1):
            print(f"{i}. 🎵 {track.get('track_name', 'Unknown Track')}")
            print(f"   🎤 Artist: {track.get('artist_name', 'Unknown Artist')}")
            print(f"   💚 Benefit: {track.get('therapeutic_benefit', 'General Wellness')}")
            print(f"   📊 Confidence: {track.get('confidence_score', 0):.2f}")
            print()
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    test_unified_system()