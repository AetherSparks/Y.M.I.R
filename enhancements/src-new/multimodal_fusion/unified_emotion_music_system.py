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

class AdvancedMusicRecommendationEngine:
    """🎵 ADVANCED Music recommendation engine with 4 strategies + ML models"""
    
    def __init__(self, dataset_path: str = "datasets/therapeutic_music_enriched.csv", model_dir: str = "models"):
        self.dataset_path = Path(dataset_path)
        self.model_dir = Path(model_dir)
        self.music_df = None
        
        # ML Model components
        self.recommendation_model = None
        self.feature_scaler = None
        self.label_encoder = None
        
        # Session tracking and analytics
        self.recommendation_history = deque(maxlen=1000)
        self.session_recommendations = defaultdict(list)
        
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
                logging.info(f"Loaded music dataset: {len(self.music_df)} tracks")
                
                # Ensure required columns exist
                required_columns = ['Track Name', 'Artist Name', 'Mood_Label']
                missing_columns = [col for col in required_columns if col not in self.music_df.columns]
                if missing_columns:
                    logging.warning(f"Missing columns in dataset: {missing_columns}")
            else:
                logging.warning(f"Music dataset not found: {self.dataset_path}")
                # Create fallback data
                self.music_df = pd.DataFrame({
                    'Track Name': ['On Top Of The World', 'Counting Stars', 'Let Her Go', 'Photograph', 'Paradise'],
                    'Artist Name': ['Imagine Dragons', 'OneRepublic', 'Passenger', 'Ed Sheeran', 'Coldplay'],
                    'Mood_Label': ['joy', 'joy', 'sadness', 'neutral', 'neutral']
                })
            
            # Load trained ML models if available
            self._load_trained_model()
            
        except Exception as e:
            logging.error(f"Error initializing recommendation system: {e}")
    
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
            return self._create_fallback_recommendations(emotion, confidence, session_id, num_recommendations, "therapeutic")
        
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
                recommendation_strategy="therapeutic"
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _mood_matching_strategy(self, emotion: str, confidence: float,
                              session_id: str, num_recommendations: int) -> List[MusicRecommendation]:
        """🎭 Mood matching - music that matches current emotional state"""
        
        if self.music_df is None or self.music_df.empty:
            return self._create_fallback_recommendations(emotion, confidence, session_id, num_recommendations, "mood_matching")
        
        # Filter by exact emotion match
        matching_tracks = pd.DataFrame()
        if 'Mood_Label' in self.music_df.columns:
            matching_tracks = self.music_df[self.music_df['Mood_Label'] == emotion].copy()
        
        if matching_tracks.empty:
            # Fallback to audio feature matching
            matching_tracks = self._find_tracks_by_audio_features(emotion, num_recommendations * 2)
        
        # Score and rank by popularity if available
        matching_tracks = self._score_tracks_by_popularity(matching_tracks)
        top_tracks = matching_tracks.head(num_recommendations)
        
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
                recommendation_strategy="mood_matching"
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _mood_regulation_strategy(self, emotion: str, confidence: float,
                                session_id: str, num_recommendations: int) -> List[MusicRecommendation]:
        """⚖️ Mood regulation - guide toward desired emotional state"""
        
        # Define target emotions for regulation
        regulation_map = {
            'sadness': 'neutral',
            'anger': 'calm', 
            'fear': 'calm',
            'neutral': 'joy',
            'joy': 'joy',  # Maintain positive state
            'excitement': 'calm',  # Calm down high energy
            'calm': 'calm'  # Maintain calm state
        }
        
        target_emotion = regulation_map.get(emotion, 'neutral')
        
        if self.music_df is None or self.music_df.empty:
            return self._create_fallback_recommendations(target_emotion, confidence, session_id, num_recommendations, "mood_regulation")
        
        # Get tracks for target emotion
        regulating_tracks = pd.DataFrame()
        if 'Mood_Label' in self.music_df.columns:
            regulating_tracks = self.music_df[self.music_df['Mood_Label'] == target_emotion].copy()
        
        if regulating_tracks.empty:
            regulating_tracks = self._find_tracks_by_audio_features(target_emotion, num_recommendations * 2)
        
        # Score and select
        regulating_tracks = self._score_tracks_by_audio_features(regulating_tracks, target_emotion)
        top_tracks = regulating_tracks.head(num_recommendations)
        
        recommendations = []
        for _, track in top_tracks.iterrows():
            rec = MusicRecommendation(
                track_name=track.get('Track Name', 'Unknown Track'),
                artist_name=track.get('Artist Name', 'Unknown Artist'),
                emotion_target=target_emotion,
                confidence_score=confidence * 0.8,  # Confidence reduction for regulation
                therapeutic_benefit=f"Emotional regulation: {emotion} → {target_emotion}",
                audio_features=self._extract_audio_features(track),
                recommendation_reason=f"Guide from {emotion} toward {target_emotion}",
                timestamp=datetime.now(),
                session_id=session_id,
                recommendation_strategy="mood_regulation"
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _adaptive_strategy(self, emotion: str, confidence: float,
                         session_id: str, num_recommendations: int) -> List[MusicRecommendation]:
        """🧠 Adaptive strategy - combines approaches based on context"""
        
        # Get session history for context
        session_history = self.session_recommendations.get(session_id, [])
        
        # Determine strategy based on confidence and history
        if confidence > 0.8:
            # High confidence - trust the emotion and match it
            return self._mood_matching_strategy(emotion, confidence, session_id, num_recommendations)
        elif len(session_history) > 0:
            # Has history - check if we need mood regulation
            recent_emotions = [rec.target_emotion for rec in session_history[-3:]]
            if recent_emotions.count(emotion) > 1 and emotion in ['sadness', 'anger', 'fear']:
                # Persistent negative emotion - try regulation
                return self._mood_regulation_strategy(emotion, confidence, session_id, num_recommendations)
            else:
                # Mix of matching and therapeutic
                matching_recs = self._mood_matching_strategy(emotion, confidence, session_id, num_recommendations // 2)
                therapeutic_recs = self._therapeutic_strategy(emotion, confidence, session_id, num_recommendations - len(matching_recs))
                return matching_recs + therapeutic_recs
        else:
            # No history, moderate confidence - use therapeutic approach
            return self._therapeutic_strategy(emotion, confidence, session_id, num_recommendations)
    
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
    
    def _create_fallback_recommendations(self, emotion: str, confidence: float, session_id: str, 
                                       num_tracks: int, strategy: str) -> List[MusicRecommendation]:
        """Create fallback recommendations when no dataset available"""
        fallback_tracks = {
            'joy': [
                {'track': 'Happy', 'artist': 'Pharrell Williams'},
                {'track': 'Can\'t Stop the Feeling', 'artist': 'Justin Timberlake'},
                {'track': 'Good as Hell', 'artist': 'Lizzo'}
            ],
            'sadness': [
                {'track': 'Someone Like You', 'artist': 'Adele'},
                {'track': 'Hurt', 'artist': 'Johnny Cash'},
                {'track': 'Mad World', 'artist': 'Gary Jules'}
            ],
            'neutral': [
                {'track': 'Weightless', 'artist': 'Marconi Union'},
                {'track': 'Clair de Lune', 'artist': 'Claude Debussy'},
                {'track': 'River', 'artist': 'Joni Mitchell'}
            ],
            'anger': [
                {'track': 'Breathe Me', 'artist': 'Sia'},
                {'track': 'Calm Down', 'artist': 'Rema'},
                {'track': 'Peace of Mind', 'artist': 'Boston'}
            ]
        }
        
        tracks = fallback_tracks.get(emotion, fallback_tracks['neutral'])[:num_tracks]
        
        recommendations = []
        for track_data in tracks:
            rec = MusicRecommendation(
                track_name=track_data['track'],
                artist_name=track_data['artist'],
                emotion_target=emotion,
                confidence_score=confidence * 0.7,  # Lower confidence for fallback
                therapeutic_benefit=self.therapeutic_benefits.get(emotion, ['General Wellness'])[0],
                audio_features={'valence': 0.5, 'energy': 0.5, 'tempo': 120},  # Default values
                recommendation_reason=f"Fallback {strategy} selection for {emotion}",
                timestamp=datetime.now(),
                session_id=session_id,
                recommendation_strategy=f"{strategy}_fallback"
            )
            recommendations.append(rec)
        
        return recommendations
    
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
        """🎯 MAIN FUNCTION: Get current emotion and music recommendations"""
        start_time = datetime.now()
        
        try:
            # Get emotion data from both sources
            facial_data = self.get_latest_facial_emotions(minutes_back)
            text_data = self.get_latest_text_emotions(minutes_back)
            
            # Fuse emotions
            if not facial_data and not text_data:
                return None
            
            emotion, confidence, method = self.fusion_engine.fuse_emotions(
                facial_data, text_data, strategy
            )
            
            # Get music recommendations using advanced engine
            rec_set = self.music_engine.get_recommendations_for_emotion(
                session_id, emotion, confidence, "adaptive", num_tracks
            )
            
            # Extract just the recommendations for simple return format
            music_recommendations = [rec.to_dict() for rec in rec_set.recommendations] if rec_set else []
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
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
                processing_time_ms=processing_time
            )
            
            # Store in session
            self.session_data[session_id] = result
            
            return result
            
        except Exception as e:
            if not self.silent:
                print(f"Error in get_emotion_and_music: {e}")
            return None
    
    def get_session_history(self, session_id: str) -> Optional[UnifiedEmotionResult]:
        """Get latest result for session"""
        return self.session_data.get(session_id)

# Global instance
_unified_system = None

def get_emotion_and_music(session_id: str = "default", minutes_back: int = 10, 
                         strategy: str = 'adaptive', num_tracks: int = 10) -> Optional[Dict[str, Any]]:
    """
    🎯 MAIN FUNCTION FOR APP.PY
    
    Get current user emotion and music recommendations in one call!
    
    Args:
        session_id: User session identifier
        minutes_back: How many minutes back to look for emotions
        strategy: Fusion strategy ('simple', 'adaptive', 'confidence_based', 'temporal_weighted', 'weighted_average')
        num_tracks: Number of music tracks to recommend
    
    Returns:
        Dict with emotion and music data or None if no emotions found
        {
            'emotion': str,                    # Current emotion
            'confidence': float,               # 0-1 confidence score
            'source': str,                     # How emotion was determined
            'strategy': str,                   # Fusion strategy used
            'music_recommendations': [         # List of recommended tracks
                {
                    'track': str,              # Track name
                    'artist': str,             # Artist name
                    'mood': str,               # Mood category
                    'therapeutic_benefit': str, # Therapeutic benefit
                    'confidence_match': float, # How well it matches emotion
                    'recommendation_reason': str
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
    return result.to_dict() if result else None

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
    result = get_emotion_and_music("test_session", strategy='adaptive')
    
    if result:
        print(f"✅ EMOTION DETECTED: {result['emotion']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Source: {result['source']}")
        print(f"   Strategy: {result['strategy']}")
        print(f"   Music tracks: {len(result['music_recommendations'])}")
        print(f"   Processing time: {result['processing_time_ms']:.1f}ms")
        
        # Show first track
        if result['music_recommendations']:
            track = result['music_recommendations'][0]
            print(f"   First recommendation: {track['track']} by {track['artist']}")
    else:
        print("❌ No emotions detected")
    
    # Test simple function
    emotion = get_emotion_simple("test_session")
    print(f"\nSimple emotion: {emotion}")
    
    # Test music-only function
    music = get_music_for_emotion("joy", 0.9, 3)
    print(f"Music for joy: {len(music)} tracks")
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    test_unified_system()