"""
🎵 Y.M.I.R Multimodal Music Recommendation Service
================================================
Enhanced music recommendation system that integrates with multimodal emotion fusion
to provide personalized music recommendations based on combined facial and text emotions.

Features:
- Integration with multimodal emotion fusion service
- Real-time emotion-based music recommendations
- Therapeutic music selection algorithms
- Confidence-weighted recommendation scoring
- Session-based recommendation history
- Context-aware music selection
"""

import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict, deque

# Import the multimodal emotion service
try:
    from multimodal_emotion_service import get_multimodal_processor, get_fused_emotion_for_music
    MULTIMODAL_EMOTION_AVAILABLE = True
except ImportError as e:
    MULTIMODAL_EMOTION_AVAILABLE = False
    logging.warning(f"Multimodal emotion service not available: {e}")
    
    # Create mock functions
    def get_multimodal_processor():
        return None
    def get_fused_emotion_for_music(session_id):
        return None

@dataclass
class MusicRecommendation:
    """Individual music recommendation with metadata"""
    track_name: str
    artist_name: str
    emotion_target: str
    confidence_score: float
    therapeutic_benefit: str
    audio_features: Dict[str, float]
    recommendation_reason: str
    timestamp: datetime
    session_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass  
class RecommendationSet:
    """Set of recommendations for a specific emotional state"""
    session_id: str
    target_emotion: str
    emotion_confidence: float
    modality_sources: Dict[str, bool]  # facial, textual availability
    recommendations: List[MusicRecommendation]
    generation_time: datetime
    recommendation_strategy: str
    context_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['generation_time'] = self.generation_time.isoformat()
        result['recommendations'] = [r.to_dict() for r in self.recommendations]
        return result

class MultimodalMusicRecommendationEngine:
    """Enhanced recommendation engine for multimodal emotion inputs"""
    
    def __init__(self, dataset_path: str = "datasets/therapeutic_music_enriched.csv",
                 model_dir: str = "models"):
        self.dataset_path = Path(dataset_path)
        self.model_dir = Path(model_dir)
        self.multimodal_processor = get_multimodal_processor()
        
        # Load music dataset
        self.music_df = None
        self.recommendation_model = None
        self.feature_scaler = None
        self.label_encoder = None
        
        # Recommendation history
        self.recommendation_history = deque(maxlen=1000)
        self.session_recommendations = defaultdict(list)
        
        # Emotion-music mapping strategies
        self.emotion_music_strategies = {
            'therapeutic': self._therapeutic_strategy,
            'mood_matching': self._mood_matching_strategy,
            'mood_regulation': self._mood_regulation_strategy,
            'adaptive': self._adaptive_strategy
        }
        
        # Therapeutic benefits mapping
        self.therapeutic_benefits = {
            'joy': ['Mood Enhancement', 'Energy Boost', 'Social Connection'],
            'sadness': ['Emotional Processing', 'Comfort', 'Catharsis'],
            'anger': ['Tension Release', 'Calming', 'Emotional Regulation'],
            'fear': ['Anxiety Relief', 'Reassurance', 'Confidence Building'],
            'neutral': ['General Wellness', 'Relaxation', 'Focus'],
            'excitement': ['Energy Channeling', 'Celebration', 'Motivation'],
            'calm': ['Stress Relief', 'Meditation', 'Sleep Aid']
        }
        
        # Audio feature preferences by emotion
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
        logging.info("🎵 Multimodal Music Recommendation Engine initialized")
    
    def _initialize_system(self):
        """Initialize the recommendation system"""
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
            
            # Load trained recommendation model if available
            self._load_trained_model()
            
        except Exception as e:
            logging.error(f"Error initializing recommendation system: {e}")
    
    def _load_trained_model(self):
        """Load the most recent trained recommendation model"""
        try:
            if not self.model_dir.exists():
                logging.warning("Models directory not found")
                return
            
            # Find the most recent model
            model_files = list(self.model_dir.glob("music_recommender_*.pkl"))
            if not model_files:
                logging.warning("No trained recommendation models found")
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
    
    def get_recommendations_for_emotion(self, session_id: str, 
                                      strategy: str = 'adaptive',
                                      num_recommendations: int = 10) -> Optional[RecommendationSet]:
        """Get music recommendations based on current multimodal emotion state"""
        
        # Get current fused emotion state
        emotion_data = get_fused_emotion_for_music(session_id)
        if not emotion_data:
            logging.warning(f"No emotion data available for session {session_id}")
            return None
        
        target_emotion = emotion_data['emotion']
        emotion_confidence = emotion_data['confidence']
        
        logging.info(f"Generating recommendations for emotion: {target_emotion} (confidence: {emotion_confidence:.2f})")
        
        # Select recommendation strategy
        strategy_func = self.emotion_music_strategies.get(strategy, self._adaptive_strategy)
        
        # Generate recommendations
        recommendations = strategy_func(
            target_emotion, emotion_confidence, session_id, num_recommendations
        )
        
        # Create recommendation set
        rec_set = RecommendationSet(
            session_id=session_id,
            target_emotion=target_emotion,
            emotion_confidence=emotion_confidence,
            modality_sources={
                'facial': emotion_data.get('context', {}).get('facial_available', False),
                'textual': emotion_data.get('context', {}).get('text_available', False)
            },
            recommendations=recommendations,
            generation_time=datetime.now(),
            recommendation_strategy=strategy,
            context_metadata=emotion_data.get('context', {})
        )
        
        # Store in history
        self.recommendation_history.append(rec_set)
        self.session_recommendations[session_id].append(rec_set)
        
        return rec_set
    
    def _therapeutic_strategy(self, emotion: str, confidence: float, 
                            session_id: str, num_recommendations: int) -> List[MusicRecommendation]:
        """Therapeutic approach - select music to support emotional healing"""
        
        if self.music_df is None:
            return []
        
        # Get therapeutic benefits for this emotion
        benefits = self.therapeutic_benefits.get(emotion, ['General Wellness'])
        
        # Filter music by therapeutic benefits
        therapeutic_tracks = self.music_df[
            self.music_df['Mental_Health_Benefit'].isin(benefits)
        ].copy() if 'Mental_Health_Benefit' in self.music_df.columns else self.music_df.copy()
        
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
                therapeutic_benefit=track.get('Mental_Health_Benefit', 'General Wellness'),
                audio_features=self._extract_audio_features(track),
                recommendation_reason=f"Therapeutic support for {emotion}",
                timestamp=datetime.now(),
                session_id=session_id
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _mood_matching_strategy(self, emotion: str, confidence: float,
                              session_id: str, num_recommendations: int) -> List[MusicRecommendation]:
        """Mood matching - select music that matches current emotional state"""
        
        if self.music_df is None:
            return []
        
        # Filter by exact emotion match
        matching_tracks = self.music_df[
            self.music_df['Mood_Label'] == emotion
        ].copy() if 'Mood_Label' in self.music_df.columns else pd.DataFrame()
        
        if matching_tracks.empty:
            # Fallback to audio feature matching
            matching_tracks = self._find_tracks_by_audio_features(emotion, num_recommendations * 2)
        
        # Score and rank
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
                session_id=session_id
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _mood_regulation_strategy(self, emotion: str, confidence: float,
                                session_id: str, num_recommendations: int) -> List[MusicRecommendation]:
        """Mood regulation - select music to guide toward desired emotional state"""
        
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
        
        # Get tracks for target emotion
        if self.music_df is None:
            return []
        
        regulating_tracks = self.music_df[
            self.music_df['Mood_Label'] == target_emotion
        ].copy() if 'Mood_Label' in self.music_df.columns else pd.DataFrame()
        
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
                session_id=session_id
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _adaptive_strategy(self, emotion: str, confidence: float,
                         session_id: str, num_recommendations: int) -> List[MusicRecommendation]:
        """Adaptive strategy that combines multiple approaches based on context"""
        
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
        if self.music_df is None:
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
            return tracks_df.sample(frac=1.0)
    
    def _extract_audio_features(self, track_row) -> Dict[str, float]:
        """Extract audio features from track data"""
        feature_columns = [
            'Danceability', 'Energy', 'Valence', 'Tempo', 'Loudness',
            'Acousticness', 'Instrumentalness', 'Liveness', 'Speechiness'
        ]
        
        features = {}
        for col in feature_columns:
            if col in track_row:
                features[col.lower()] = float(track_row[col])
        
        return features
    
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

# Singleton instance
multimodal_music_engine = MultimodalMusicRecommendationEngine()

def get_music_recommendation_engine() -> MultimodalMusicRecommendationEngine:
    """Get the global multimodal music recommendation engine"""
    return multimodal_music_engine

# Convenience functions for external use
def recommend_music_for_session(session_id: str, strategy: str = 'adaptive', 
                              num_tracks: int = 10) -> Optional[Dict[str, Any]]:
    """Get music recommendations for a session"""
    engine = get_music_recommendation_engine()
    rec_set = engine.get_recommendations_for_emotion(session_id, strategy, num_tracks)
    return rec_set.to_dict() if rec_set else None

def get_current_recommendations(session_id: str) -> Optional[Dict[str, Any]]:
    """Get the most recent recommendations for a session"""
    engine = get_music_recommendation_engine()
    session_recs = engine.session_recommendations.get(session_id, [])
    
    if session_recs:
        return session_recs[-1].to_dict()
    return None