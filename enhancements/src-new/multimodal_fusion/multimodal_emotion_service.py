"""
🧠 Y.M.I.R Multimodal Emotion Fusion Service
============================================
Combines facial emotion recognition and text sentiment analysis
to create unified emotion profiles for music recommendations.

Architecture:
- Facial emotions from fer_enhanced_v2.py (Firebase + EmotionReading)
- Text emotions from chatbot_production_ready.py (JSON + ChatMessage)  
- Unified fusion with confidence weighting
- Real-time processing pipeline
- Integration with music recommendation system
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import threading
import time

# Import emotion detection systems - make them optional for now
import sys
import os

# Add paths to find existing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
enhancements_dir = os.path.join(current_dir, '..')
sys.path.append(enhancements_dir)

try:
    from face.fer_enhanced_v2 import EmotionReading, FacialEmotionDetector
    FACIAL_AVAILABLE = True
    logging.info("✅ Facial emotion detection available")
except ImportError as e:
    FACIAL_AVAILABLE = False
    logging.warning(f"Facial emotion detection not available: {e}")
    
    # Create mock classes for compatibility
    class EmotionReading:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class FacialEmotionDetector:
        pass

try:
    from textual.chatbot_production_ready import ChatMessage, EmotionAnalyzer  
    TEXTUAL_AVAILABLE = True
    logging.info("✅ Text emotion detection available")
except ImportError as e:
    TEXTUAL_AVAILABLE = False
    logging.warning(f"Text emotion detection not available: {e}")
    
    # Create mock classes for compatibility
    class ChatMessage:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class EmotionAnalyzer:
        pass

@dataclass
class MultimodalEmotionReading:
    """Unified emotion reading combining facial and textual sources"""
    timestamp: datetime
    session_id: str
    
    # Facial emotion data
    facial_emotion: Optional[str] = None
    facial_confidence: float = 0.0
    facial_emotions_raw: Optional[Dict[str, float]] = None
    face_quality_score: float = 0.0
    
    # Text emotion data  
    text_emotion: Optional[str] = None
    text_confidence: float = 0.0
    text_emotions_raw: Optional[Dict[str, float]] = None
    text_content: Optional[str] = None
    
    # Fused emotion result
    fused_emotion: Optional[str] = None
    fused_confidence: float = 0.0
    fusion_weights: Optional[Dict[str, float]] = None
    
    # Context and metadata
    context: Optional[str] = None
    processing_time_ms: float = 0.0
    source_priority: str = "balanced"  # facial, textual, balanced
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultimodalEmotionReading':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class EmotionFusionEngine:
    """Advanced emotion fusion algorithms with multiple strategies"""
    
    def __init__(self):
        self.emotion_mappings = self._create_emotion_mappings()
        self.fusion_strategies = {
            'weighted_average': self._weighted_average_fusion,
            'confidence_based': self._confidence_based_fusion,
            'temporal_weighted': self._temporal_weighted_fusion,
            'adaptive': self._adaptive_fusion
        }
        
    def _create_emotion_mappings(self) -> Dict[str, List[str]]:
        """Map different emotion vocabularies to unified categories"""
        return {
            'joy': ['happy', 'joy', 'excitement', 'pleasure', 'delight'],
            'sadness': ['sad', 'sadness', 'sorrow', 'grief', 'melancholy'],
            'anger': ['angry', 'anger', 'rage', 'fury', 'irritation'],
            'fear': ['fear', 'afraid', 'scared', 'anxiety', 'worry'],
            'surprise': ['surprise', 'surprised', 'astonishment', 'amazement'],
            'disgust': ['disgust', 'disgusted', 'revulsion', 'contempt'],
            'neutral': ['neutral', 'calm', 'peaceful', 'relaxed'],
            'excitement': ['excited', 'enthusiasm', 'energy', 'vigorous']
        }
    
    def normalize_emotion(self, emotion: str) -> str:
        """Normalize emotion to unified vocabulary"""
        emotion_lower = emotion.lower()
        for standard_emotion, variants in self.emotion_mappings.items():
            if emotion_lower in variants:
                return standard_emotion
        return emotion_lower  # Return as-is if no mapping found
    
    def _weighted_average_fusion(self, facial_emotions: Dict[str, float], 
                                text_emotions: Dict[str, float],
                                facial_weight: float, text_weight: float) -> Tuple[str, float, Dict[str, float]]:
        """Weighted average of emotion scores"""
        all_emotions = set(facial_emotions.keys()) | set(text_emotions.keys())
        fused_scores = {}
        
        for emotion in all_emotions:
            facial_score = facial_emotions.get(emotion, 0.0)
            text_score = text_emotions.get(emotion, 0.0)
            
            # Weighted combination
            fused_score = (facial_score * facial_weight + text_score * text_weight)
            if facial_weight + text_weight > 0:
                fused_score /= (facial_weight + text_weight)
            
            fused_scores[emotion] = fused_score
        
        # Get top emotion
        top_emotion = max(fused_scores.items(), key=lambda x: x[1])
        return top_emotion[0], top_emotion[1], fused_scores
    
    def _confidence_based_fusion(self, facial_emotions: Dict[str, float],
                                text_emotions: Dict[str, float],
                                facial_confidence: float, text_confidence: float) -> Tuple[str, float, Dict[str, float]]:
        """Fusion based on confidence levels"""
        if facial_confidence > text_confidence * 1.5:
            # Strongly favor facial
            return self._weighted_average_fusion(facial_emotions, text_emotions, 0.8, 0.2)
        elif text_confidence > facial_confidence * 1.5:
            # Strongly favor textual
            return self._weighted_average_fusion(facial_emotions, text_emotions, 0.2, 0.8)
        else:
            # Balanced fusion
            return self._weighted_average_fusion(facial_emotions, text_emotions, 0.5, 0.5)
    
    def _temporal_weighted_fusion(self, facial_emotions: Dict[str, float],
                                 text_emotions: Dict[str, float],
                                 facial_age_seconds: float, text_age_seconds: float) -> Tuple[str, float, Dict[str, float]]:
        """Fusion with temporal decay - newer emotions have higher weight"""
        max_age = 300  # 5 minutes max relevance
        
        facial_temporal_weight = max(0.1, 1.0 - (facial_age_seconds / max_age))
        text_temporal_weight = max(0.1, 1.0 - (text_age_seconds / max_age))
        
        return self._weighted_average_fusion(facial_emotions, text_emotions, 
                                           facial_temporal_weight, text_temporal_weight)
    
    def _adaptive_fusion(self, facial_emotions: Dict[str, float],
                        text_emotions: Dict[str, float],
                        facial_confidence: float, text_confidence: float,
                        context: str = None) -> Tuple[str, float, Dict[str, float]]:
        """Adaptive fusion based on context and modality reliability"""
        # Base weights
        facial_weight = 0.5
        text_weight = 0.5
        
        # Adjust based on confidence differential
        confidence_ratio = facial_confidence / (text_confidence + 0.01)
        if confidence_ratio > 2.0:
            facial_weight = 0.7
            text_weight = 0.3
        elif confidence_ratio < 0.5:
            facial_weight = 0.3
            text_weight = 0.7
        
        # Context-based adjustments
        if context and 'conversation' in context.lower():
            text_weight *= 1.2  # Boost text in conversational contexts
        elif context and any(word in context.lower() for word in ['video', 'camera', 'face']):
            facial_weight *= 1.2  # Boost facial in visual contexts
        
        # Normalize weights
        total_weight = facial_weight + text_weight
        facial_weight /= total_weight
        text_weight /= total_weight
        
        return self._weighted_average_fusion(facial_emotions, text_emotions, 
                                           facial_weight, text_weight)
    
    def fuse_emotions(self, facial_data: Dict[str, Any], text_data: Dict[str, Any],
                     strategy: str = 'adaptive', context: str = None) -> Tuple[str, float, Dict[str, float], Dict[str, float]]:
        """Main fusion method"""
        # Normalize emotion vocabularies
        facial_emotions = {}
        if facial_data.get('emotions'):
            for emotion, score in facial_data['emotions'].items():
                normalized = self.normalize_emotion(emotion)
                facial_emotions[normalized] = score
        
        text_emotions = {}
        if text_data.get('emotions'):
            for emotion, score in text_data['emotions'].items():
                normalized = self.normalize_emotion(emotion)
                text_emotions[normalized] = score
        
        # Get confidences
        facial_confidence = facial_data.get('confidence', 0.0)
        text_confidence = text_data.get('confidence', 0.0)
        
        # Apply fusion strategy
        fusion_func = self.fusion_strategies.get(strategy, self._adaptive_fusion)
        
        if strategy == 'adaptive':
            fused_emotion, fused_confidence, fused_scores = fusion_func(
                facial_emotions, text_emotions, facial_confidence, text_confidence, context
            )
        else:
            fused_emotion, fused_confidence, fused_scores = fusion_func(
                facial_emotions, text_emotions, facial_confidence, text_confidence
            )
        
        # Calculate fusion weights used
        total_confidence = facial_confidence + text_confidence
        if total_confidence > 0:
            fusion_weights = {
                'facial': facial_confidence / total_confidence,
                'textual': text_confidence / total_confidence
            }
        else:
            fusion_weights = {'facial': 0.5, 'textual': 0.5}
        
        return fused_emotion, fused_confidence, fused_scores, fusion_weights

class MultimodalEmotionProcessor:
    """Main service for processing multimodal emotion data"""
    
    def __init__(self, session_duration_minutes: int = 30):
        self.session_duration = timedelta(minutes=session_duration_minutes)
        self.fusion_engine = EmotionFusionEngine()
        
        # Storage for recent emotion readings
        self.facial_readings = deque(maxlen=100)
        self.text_readings = deque(maxlen=100) 
        self.fused_readings = deque(maxlen=100)
        
        # Session management
        self.current_sessions = {}
        self.session_lock = threading.Lock()
        
        # Storage paths
        self.data_dir = Path("data/multimodal_emotions")
        self.data_dir.mkdir(exist_ok=True)
        
        # Processing stats
        self.processing_stats = {
            'total_processed': 0,
            'facial_count': 0,
            'text_count': 0,
            'fusion_count': 0,
            'avg_processing_time_ms': 0.0
        }
        
        logging.info("🧠 Multimodal Emotion Processor initialized")
    
    def create_session(self, session_id: str) -> str:
        """Create a new emotion processing session"""
        with self.session_lock:
            self.current_sessions[session_id] = {
                'created': datetime.now(),
                'last_activity': datetime.now(),
                'facial_count': 0,
                'text_count': 0,
                'fusion_count': 0
            }
        logging.info(f"Created emotion session: {session_id}")
        return session_id
    
    def process_facial_emotion(self, session_id: str, emotion_reading: EmotionReading) -> MultimodalEmotionReading:
        """Process facial emotion input"""
        start_time = time.time()
        
        # Update session activity
        self._update_session_activity(session_id)
        
        # Convert facial emotion reading
        facial_data = {
            'emotion': max(emotion_reading.emotions.items(), key=lambda x: x[1])[0],
            'confidence': emotion_reading.confidence,
            'emotions': emotion_reading.emotions,
            'quality_score': emotion_reading.quality_score,
            'timestamp': emotion_reading.timestamp
        }
        
        # Create multimodal reading
        multimodal_reading = MultimodalEmotionReading(
            timestamp=datetime.now(),
            session_id=session_id,
            facial_emotion=facial_data['emotion'],
            facial_confidence=facial_data['confidence'],
            facial_emotions_raw=facial_data['emotions'],
            face_quality_score=facial_data['quality_score'],
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # Store reading
        self.facial_readings.append(multimodal_reading)
        
        # Update stats
        self.processing_stats['facial_count'] += 1
        self.processing_stats['total_processed'] += 1
        
        logging.debug(f"Processed facial emotion: {facial_data['emotion']} (conf: {facial_data['confidence']:.2f})")
        return multimodal_reading
    
    def process_text_emotion(self, session_id: str, chat_message: ChatMessage) -> MultimodalEmotionReading:
        """Process text emotion input"""
        start_time = time.time()
        
        # Update session activity
        self._update_session_activity(session_id)
        
        # Create multimodal reading
        multimodal_reading = MultimodalEmotionReading(
            timestamp=datetime.now(),
            session_id=session_id,
            text_emotion=chat_message.emotion,
            text_confidence=chat_message.confidence or 0.0,
            text_content=chat_message.content,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # Store reading
        self.text_readings.append(multimodal_reading)
        
        # Update stats
        self.processing_stats['text_count'] += 1
        self.processing_stats['total_processed'] += 1
        
        logging.debug(f"Processed text emotion: {chat_message.emotion} (conf: {chat_message.confidence:.2f})")
        return multimodal_reading
    
    def fuse_recent_emotions(self, session_id: str, time_window_seconds: int = 30,
                           strategy: str = 'adaptive') -> Optional[MultimodalEmotionReading]:
        """Fuse recent facial and text emotions within time window"""
        start_time = time.time()
        
        # Get recent readings within time window
        cutoff_time = datetime.now() - timedelta(seconds=time_window_seconds)
        
        recent_facial = [r for r in self.facial_readings 
                        if r.session_id == session_id and r.timestamp >= cutoff_time]
        recent_text = [r for r in self.text_readings 
                      if r.session_id == session_id and r.timestamp >= cutoff_time]
        
        if not recent_facial and not recent_text:
            logging.debug(f"No recent emotions found for session {session_id}")
            return None
        
        # Get most recent/highest confidence readings
        facial_data = {}
        if recent_facial:
            best_facial = max(recent_facial, key=lambda x: x.facial_confidence)
            facial_data = {
                'emotion': best_facial.facial_emotion,
                'confidence': best_facial.facial_confidence,
                'emotions': best_facial.facial_emotions_raw or {},
                'timestamp': best_facial.timestamp
            }
        
        text_data = {}
        if recent_text:
            best_text = max(recent_text, key=lambda x: x.text_confidence)
            text_data = {
                'emotion': best_text.text_emotion,
                'confidence': best_text.text_confidence,
                'emotions': {best_text.text_emotion: best_text.text_confidence} if best_text.text_emotion else {},
                'timestamp': best_text.timestamp
            }
        
        # Perform fusion
        if facial_data and text_data:
            fused_emotion, fused_confidence, fused_scores, fusion_weights = self.fusion_engine.fuse_emotions(
                facial_data, text_data, strategy=strategy, context=f"session_{session_id}"
            )
        elif facial_data:
            # Only facial data available
            fused_emotion = facial_data['emotion']
            fused_confidence = facial_data['confidence']
            fused_scores = facial_data['emotions']
            fusion_weights = {'facial': 1.0, 'textual': 0.0}
        elif text_data:
            # Only text data available  
            fused_emotion = text_data['emotion']
            fused_confidence = text_data['confidence']
            fused_scores = text_data['emotions']
            fusion_weights = {'facial': 0.0, 'textual': 1.0}
        else:
            return None
        
        # Create fused reading
        fused_reading = MultimodalEmotionReading(
            timestamp=datetime.now(),
            session_id=session_id,
            facial_emotion=facial_data.get('emotion'),
            facial_confidence=facial_data.get('confidence', 0.0),
            facial_emotions_raw=facial_data.get('emotions'),
            text_emotion=text_data.get('emotion'),
            text_confidence=text_data.get('confidence', 0.0),
            text_content=recent_text[-1].text_content if recent_text else None,
            fused_emotion=fused_emotion,
            fused_confidence=fused_confidence,
            fusion_weights=fusion_weights,
            processing_time_ms=(time.time() - start_time) * 1000,
            source_priority=strategy
        )
        
        # Store fused reading
        self.fused_readings.append(fused_reading)
        
        # Update stats
        self.processing_stats['fusion_count'] += 1
        self._update_processing_time_stats(fused_reading.processing_time_ms)
        
        logging.info(f"Fused emotions: {fused_emotion} (conf: {fused_confidence:.2f}, strategy: {strategy})")
        return fused_reading
    
    def get_current_emotion_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current emotional state for a session"""
        # Try to get recent fused reading first
        recent_fused = [r for r in self.fused_readings 
                       if r.session_id == session_id]
        
        if recent_fused:
            latest = max(recent_fused, key=lambda x: x.timestamp)
            return {
                'primary_emotion': latest.fused_emotion,
                'confidence': latest.fused_confidence,
                'timestamp': latest.timestamp,
                'sources': {
                    'facial': {
                        'emotion': latest.facial_emotion,
                        'confidence': latest.facial_confidence
                    },
                    'textual': {
                        'emotion': latest.text_emotion, 
                        'confidence': latest.text_confidence
                    }
                },
                'fusion_weights': latest.fusion_weights,
                'type': 'fused'
            }
        
        # Fallback to individual readings
        recent_facial = [r for r in self.facial_readings if r.session_id == session_id]
        recent_text = [r for r in self.text_readings if r.session_id == session_id]
        
        if recent_facial:
            latest = max(recent_facial, key=lambda x: x.timestamp)
            return {
                'primary_emotion': latest.facial_emotion,
                'confidence': latest.facial_confidence,
                'timestamp': latest.timestamp,
                'type': 'facial_only'
            }
        elif recent_text:
            latest = max(recent_text, key=lambda x: x.timestamp)
            return {
                'primary_emotion': latest.text_emotion,
                'confidence': latest.text_confidence, 
                'timestamp': latest.timestamp,
                'type': 'text_only'
            }
        
        return None
    
    def save_session_data(self, session_id: str):
        """Save session emotion data to file"""
        session_readings = [r for r in self.fused_readings if r.session_id == session_id]
        
        if not session_readings:
            logging.warning(f"No fused readings found for session {session_id}")
            return
        
        # Save to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"session_{session_id}_{timestamp}.json"
        
        session_data = {
            'session_id': session_id,
            'created': datetime.now().isoformat(),
            'readings': [r.to_dict() for r in session_readings],
            'stats': self.current_sessions.get(session_id, {})
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logging.info(f"Saved session data: {filename}")
    
    def _update_session_activity(self, session_id: str):
        """Update session last activity time"""
        with self.session_lock:
            if session_id in self.current_sessions:
                self.current_sessions[session_id]['last_activity'] = datetime.now()
    
    def _update_processing_time_stats(self, processing_time_ms: float):
        """Update rolling average of processing times"""
        current_avg = self.processing_stats['avg_processing_time_ms']
        total_processed = self.processing_stats['total_processed']
        
        # Simple moving average
        if total_processed > 0:
            self.processing_stats['avg_processing_time_ms'] = (
                (current_avg * (total_processed - 1) + processing_time_ms) / total_processed
            )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            **self.processing_stats,
            'active_sessions': len(self.current_sessions),
            'facial_readings_stored': len(self.facial_readings),
            'text_readings_stored': len(self.text_readings),
            'fused_readings_stored': len(self.fused_readings)
        }
    
    def cleanup_old_sessions(self):
        """Clean up expired sessions"""
        cutoff_time = datetime.now() - self.session_duration
        expired_sessions = []
        
        with self.session_lock:
            for session_id, session_data in self.current_sessions.items():
                if session_data['last_activity'] < cutoff_time:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                # Save session data before cleanup
                self.save_session_data(session_id)
                del self.current_sessions[session_id]
        
        if expired_sessions:
            logging.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# Singleton instance for global access
multimodal_processor = MultimodalEmotionProcessor()

def get_multimodal_processor() -> MultimodalEmotionProcessor:
    """Get the global multimodal emotion processor instance"""
    return multimodal_processor

# Convenience functions for external services
def process_facial_emotion_input(session_id: str, emotion_data: Dict[str, Any]) -> MultimodalEmotionReading:
    """Convenience function to process facial emotion from external services"""
    processor = get_multimodal_processor()
    
    # Convert dict to EmotionReading-like structure
    emotion_reading = type('EmotionReading', (), {
        'emotions': emotion_data.get('emotions', {}),
        'confidence': emotion_data.get('confidence', 0.0),
        'quality_score': emotion_data.get('quality_score', 0.0),
        'timestamp': datetime.now()
    })()
    
    return processor.process_facial_emotion(session_id, emotion_reading)

def process_text_emotion_input(session_id: str, text: str, emotion: str, confidence: float) -> MultimodalEmotionReading:
    """Convenience function to process text emotion from external services"""
    processor = get_multimodal_processor()
    
    # Convert to ChatMessage-like structure
    chat_message = type('ChatMessage', (), {
        'content': text,
        'emotion': emotion,
        'confidence': confidence,
        'timestamp': datetime.now()
    })()
    
    return processor.process_text_emotion(session_id, chat_message)

def get_fused_emotion_for_music(session_id: str) -> Optional[Dict[str, Any]]:
    """Get current fused emotion state formatted for music recommendation"""
    processor = get_multimodal_processor()
    
    # Try to create fresh fusion
    fused_reading = processor.fuse_recent_emotions(session_id, time_window_seconds=60)
    
    if fused_reading:
        return {
            'emotion': fused_reading.fused_emotion,
            'confidence': fused_reading.fused_confidence,
            'timestamp': fused_reading.timestamp.isoformat(),
            'modality_weights': fused_reading.fusion_weights,
            'context': {
                'facial_available': fused_reading.facial_emotion is not None,
                'text_available': fused_reading.text_emotion is not None,
                'fusion_strategy': fused_reading.source_priority
            }
        }
    
    # Fallback to current state
    current_state = processor.get_current_emotion_state(session_id)
    if current_state:
        return {
            'emotion': current_state['primary_emotion'],
            'confidence': current_state['confidence'],
            'timestamp': current_state['timestamp'].isoformat(),
            'context': {'type': current_state['type']}
        }
    
    return None