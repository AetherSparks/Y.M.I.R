#!/usr/bin/env python3
"""
🔗 REAL Emotion Combiner - Reads from ACTUAL Storage
==================================================
Reads emotions from the ACTUAL storage locations:
- FACIAL: Firebase Firestore database (fer_enhanced_v3.py)
- TEXT: JSON chat session files (chatbot_production_ready.py)

NO fake data - only real stored emotions!
"""

import os
import sys
import json
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
# import numpy as np  # Not needed for basic fusion

# Firebase imports (same as fer_enhanced_v3.py)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
    print("✅ Firebase available for reading facial emotions")
except ImportError:
    FIREBASE_AVAILABLE = False
    firebase_admin = None
    credentials = None
    firestore = None
    print("❌ Firebase not available - install: pip install firebase-admin")

@dataclass
class RealCombinedEmotion:
    """Real combined emotion from actual storage"""
    dominant_emotion: str
    confidence: float
    facial_source: Optional[Dict[str, Any]] = None
    text_source: Optional[Dict[str, Any]] = None
    combination_method: str = "confidence_based"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class AdvancedEmotionFusionEngine:
    """🧠 ADVANCED Emotion Fusion with Multiple Strategies"""
    
    def __init__(self):
        self.emotion_mappings = self._create_emotion_mappings()
        self.fusion_strategies = {
            'simple': self._simple_fusion,
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
    
    def _simple_fusion(self, facial_data: Dict[str, Any], text_data: Dict[str, Any]) -> Tuple[str, float, str]:
        """Simple fusion - highest confidence wins (original logic)"""
        facial_confidence = facial_data.get('confidence', 0.0) if facial_data else 0.0
        text_confidence = text_data.get('confidence', 0.0) if text_data else 0.0
        
        if facial_confidence >= text_confidence and facial_data:
            return facial_data.get('emotion', 'neutral'), facial_confidence, "facial_higher_confidence"
        elif text_data:
            return text_data.get('emotion', 'neutral'), text_confidence, "text_higher_confidence"
        else:
            return 'neutral', 0.0, "no_data"
    
    def _weighted_average_fusion(self, facial_emotions: Dict[str, float], 
                                text_emotions: Dict[str, float],
                                facial_weight: float, text_weight: float) -> Tuple[str, float]:
        """Advanced weighted average of emotion scores"""
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
        if fused_scores:
            top_emotion = max(fused_scores.items(), key=lambda x: x[1])
            return top_emotion[0], top_emotion[1]
        else:
            return 'neutral', 0.0
    
    def _confidence_based_fusion(self, facial_data: Dict[str, Any], text_data: Dict[str, Any]) -> Tuple[str, float, str]:
        """Advanced fusion based on confidence levels"""
        facial_confidence = facial_data.get('confidence', 0.0) if facial_data else 0.0
        text_confidence = text_data.get('confidence', 0.0) if text_data else 0.0
        
        facial_emotions = facial_data.get('emotions', {}) if facial_data else {}
        text_emotions = {text_data.get('emotion', 'neutral'): text_confidence} if text_data else {}
        
        if facial_confidence > text_confidence * 1.5:
            # Strongly favor facial
            emotion, confidence = self._weighted_average_fusion(facial_emotions, text_emotions, 0.8, 0.2)
            return emotion, confidence, "facial_strongly_favored"
        elif text_confidence > facial_confidence * 1.5:
            # Strongly favor textual
            emotion, confidence = self._weighted_average_fusion(facial_emotions, text_emotions, 0.2, 0.8)
            return emotion, confidence, "text_strongly_favored"
        else:
            # Balanced fusion
            emotion, confidence = self._weighted_average_fusion(facial_emotions, text_emotions, 0.5, 0.5)
            return emotion, confidence, "balanced_fusion"
    
    def _temporal_weighted_fusion(self, facial_data: Dict[str, Any], text_data: Dict[str, Any], 
                                 facial_age_seconds: float, text_age_seconds: float) -> Tuple[str, float, str]:
        """Fusion with temporal decay - newer emotions have higher weight"""
        max_age = 300  # 5 minutes max relevance
        
        facial_temporal_weight = max(0.1, 1.0 - (facial_age_seconds / max_age))
        text_temporal_weight = max(0.1, 1.0 - (text_age_seconds / max_age))
        
        facial_emotions = facial_data.get('emotions', {}) if facial_data else {}
        text_emotions = {text_data.get('emotion', 'neutral'): text_data.get('confidence', 0.0)} if text_data else {}
        
        emotion, confidence = self._weighted_average_fusion(facial_emotions, text_emotions, 
                                       facial_temporal_weight, text_temporal_weight)
        
        return emotion, confidence, f"temporal_weighted_f{facial_temporal_weight:.2f}_t{text_temporal_weight:.2f}"
    
    def _adaptive_fusion(self, facial_data: Dict[str, Any], text_data: Dict[str, Any]) -> Tuple[str, float, str]:
        """🎯 SMART Adaptive fusion with MULTI-EMOTION support"""
        facial_confidence = facial_data.get('confidence', 0.0) if facial_data else 0.0
        text_confidence = text_data.get('confidence', 0.0) if text_data else 0.0
        
        # Base weights
        facial_weight = 0.5
        text_weight = 0.5
        
        # 🧠 Smart confidence-based adjustment
        confidence_ratio = facial_confidence / (text_confidence + 0.01)
        if confidence_ratio > 2.0:
            facial_weight = 0.7
            text_weight = 0.3
        elif confidence_ratio < 0.5:
            facial_weight = 0.3
            text_weight = 0.7
        
        # 🔍 Quality-based adjustment (if facial has quality score)
        if facial_data and facial_data.get('quality_score', 0) > 0.8:
            facial_weight *= 1.2  # Boost high-quality facial detection
        
        # 🎭 MULTI-EMOTION BONUS: If facial has diverse emotions, boost its weight
        if facial_data and facial_data.get('top_emotions'):
            top_emotions = facial_data['top_emotions']
            if len(top_emotions) >= 2:
                # Check if secondary emotion is significant (>10%)
                secondary_score = top_emotions[1][1] if len(top_emotions) > 1 else 0
                if secondary_score > 0.1:  # 10% threshold
                    facial_weight *= 1.15  # Boost for emotional complexity
                    print(f"🎭 Facial multi-emotion bonus: {top_emotions[1][0]} = {secondary_score:.3f}")
        
        # 🎭 MULTI-EMOTION BONUS: If text has diverse emotions, boost its weight
        if text_data and text_data.get('top_emotions'):
            top_text_emotions = text_data['top_emotions']
            if len(top_text_emotions) >= 2:
                # Check if secondary emotion is significant (>20% for text as it's usually noisier)
                secondary_score = top_text_emotions[1][1] if len(top_text_emotions) > 1 else 0
                if secondary_score > 0.2:  # 20% threshold for text
                    text_weight *= 1.15  # Boost for emotional complexity
                    print(f"🎭 Text multi-emotion bonus: {top_text_emotions[1][0]} = {secondary_score:.3f}")
        
        # 🕐 Freshness boost - newer data gets slight preference
        from datetime import timezone
        now = datetime.now(timezone.utc)  # Use UTC timezone-aware datetime
        if facial_data and facial_data.get('timestamp'):
            facial_timestamp = facial_data['timestamp']
            # Ensure facial timestamp is timezone-aware
            if facial_timestamp.tzinfo is None:
                facial_timestamp = facial_timestamp.replace(tzinfo=timezone.utc)
            facial_age = (now - facial_timestamp).total_seconds()
            if facial_age < 30:  # Less than 30 seconds old
                facial_weight *= 1.1
        
        if text_data and text_data.get('timestamp'):
            text_timestamp = text_data['timestamp']
            # Ensure text timestamp is timezone-aware
            if text_timestamp.tzinfo is None:
                text_timestamp = text_timestamp.replace(tzinfo=timezone.utc)
            text_age = (now - text_timestamp).total_seconds()
            if text_age < 30:  # Less than 30 seconds old
                text_weight *= 1.1
        
        # Normalize weights
        total_weight = facial_weight + text_weight
        facial_weight /= total_weight
        text_weight /= total_weight
        
        # 🎭 ENHANCED: Use full emotion spectrum for fusion
        facial_emotions = facial_data.get('emotions', {}) if facial_data else {}
        text_emotions = {text_data.get('emotion', 'neutral'): text_confidence} if text_data else {}
        
        emotion, confidence = self._weighted_average_fusion(facial_emotions, text_emotions, 
                                       facial_weight, text_weight)
        
        return emotion, confidence, f"adaptive_multiemo_f{facial_weight:.2f}_t{text_weight:.2f}"
    
    def fuse_emotions(self, facial_data: Dict[str, Any], text_data: Dict[str, Any], 
                     strategy: str = 'adaptive') -> Tuple[str, float, str]:
        """🎯 Main fusion method with multiple strategies"""
        
        # Normalize emotions if using advanced strategies
        if strategy != 'simple' and facial_data and facial_data.get('emotions'):
            normalized_facial_emotions = {}
            for emotion, score in facial_data['emotions'].items():
                normalized = self.normalize_emotion(emotion)
                normalized_facial_emotions[normalized] = score
            facial_data = {**facial_data, 'emotions': normalized_facial_emotions}
        
        if strategy != 'simple' and text_data and text_data.get('emotion'):
            normalized_emotion = self.normalize_emotion(text_data['emotion'])
            text_data = {**text_data, 'emotion': normalized_emotion}
        
        # Apply fusion strategy
        fusion_func = self.fusion_strategies.get(strategy, self._adaptive_fusion)
        
        if strategy == 'temporal_weighted':
            # Calculate ages for temporal weighting
            now = datetime.now()
            facial_age = (now - facial_data.get('timestamp', now)).total_seconds() if facial_data else 999
            text_age = (now - text_data.get('timestamp', now)).total_seconds() if text_data else 999
            return fusion_func(facial_data, text_data, facial_age, text_age)
        elif strategy == 'weighted_average':
            # Use equal weights for weighted average
            facial_emotions = facial_data.get('emotions', {}) if facial_data else {}
            text_emotions = {text_data.get('emotion', 'neutral'): text_data.get('confidence', 0.0)} if text_data else {}
            emotion, confidence = self._weighted_average_fusion(facial_emotions, text_emotions, 0.5, 0.5)
            return emotion, confidence, "weighted_average_equal"
        else:
            return fusion_func(facial_data, text_data)

class RealEmotionCombiner:
    """🎯 ADVANCED Emotion Combiner with Multiple Fusion Strategies"""
    
    def __init__(self, silent: bool = False):
        self.firebase_client = None
        self.project_root = self._find_project_root()
        self.fusion_engine = AdvancedEmotionFusionEngine()
        self.silent = silent
        
        # Initialize Firebase if available
        if FIREBASE_AVAILABLE:
            self._init_firebase()
        
        if not silent:
            print("🔗 ADVANCED REAL Emotion Combiner initialized")
            print(f"📂 Project root: {self.project_root}")
            print(f"🔥 Firebase: {'✅ Connected' if self.firebase_client else '❌ Unavailable'}")
            print(f"🧠 Fusion strategies: {list(self.fusion_engine.fusion_strategies.keys())}")
    
    def _find_project_root(self) -> Path:
        """Find the project root directory"""
        current = Path(__file__).parent
        
        # Look for project indicators
        for _ in range(5):  # Max 5 levels up
            if (current / "app.py").exists() or (current / "firebase_credentials.json").exists():
                return current
            current = current.parent
        
        # Default to relative path
        return Path("../../../")
    
    def _init_firebase(self):
        """Initialize Firebase connection (same logic as fer_enhanced_v3.py)"""
        try:
            if firebase_admin._apps:
                # Already initialized
                self.firebase_client = firestore.client()
                print("✅ Using existing Firebase connection")
                return
            
            # Look for credentials file
            cred_paths = [
                self.project_root / "firebase_credentials.json",
                self.project_root / "src" / "firebase_credentials.json",
                Path("firebase_credentials.json")
            ]
            
            for cred_path in cred_paths:
                if cred_path.exists():
                    cred = credentials.Certificate(str(cred_path))
                    firebase_admin.initialize_app(cred)
                    self.firebase_client = firestore.client()
                    print(f"✅ Firebase initialized with {cred_path}")
                    return
            
            print("⚠️ Firebase credentials not found")
            
        except Exception as e:
            print(f"❌ Firebase initialization error: {e}")
    
    def get_latest_facial_emotions(self, minutes_back: int = 10) -> Optional[Dict[str, Any]]:
        """Get latest facial emotions from Firebase (last X minutes)"""
        if not self.firebase_client:
            if not self.silent:
                print("❌ Firebase not available for facial emotions")
            return None
        
        try:
            # Calculate time range using UTC to match Firebase storage
            from datetime import timezone
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes_back)
            current_time = datetime.now(timezone.utc)
            
            # 🔍 DEBUG: Show what we're looking for
            print(f"🔍 DEBUG: Looking for facial emotions after {cutoff_time} (UTC)")
            print(f"🔍 DEBUG: Current time: {current_time} (UTC)")
            
            # Query Firebase for recent emotion readings (FACIAL ONLY - NO role field)
            emotions_ref = self.firebase_client.collection('emotion_readings')
            
            # Get recent readings and filter for facial emotions (no role field = facial)
            query = emotions_ref.where('timestamp', '>=', cutoff_time).limit(20)  # Get more to filter
            
            docs = query.stream()
            
            # 🔍 DEBUG: Check all documents we get
            docs_list = list(docs)
            print(f"🔍 DEBUG: Found {len(docs_list)} total documents in range")
            
            for i, doc in enumerate(docs_list):
                data = doc.to_dict()
                print(f"🔍 DEBUG Doc {i}: timestamp={data.get('timestamp')}, has_role={'role' in data}, has_emotions={'emotions' in data}, has_face_id={'face_id' in data}")
                if 'role' not in data:
                    print(f"    📹 Potential facial: {list(data.keys())}")
                else:
                    print(f"    💬 Text emotion: role={data.get('role')}")
            
            # Filter for facial emotions only (strict filtering)
            for doc in docs_list:
                data = doc.to_dict()
                # FACIAL emotions have: 'face_id', 'emotions' (plural), no 'role' field
                # TEXT emotions have: 'role': 'user', 'emotion' (singular), 'message_id'
                if ('role' not in data and 
                    'emotions' in data and 
                    'face_id' in data and 
                    'message_id' not in data):  # This is definitely a facial emotion
                    print(f"📹 Found facial emotion: {data}")
                    return {
                        'source': 'firebase',
                        'timestamp': data.get('timestamp'),
                        'emotions': data.get('emotions', {}),
                        'confidence': data.get('confidence', 0.5),
                        'face_id': data.get('face_id'),
                        'doc_id': doc.id
                    }
            
            print(f"📹 No facial emotions found in last {minutes_back} minutes")
            return None
            
        except Exception as e:
            print(f"❌ Error reading facial emotions from Firebase: {e}")
            return None
    
    def get_latest_text_emotions(self, minutes_back: int = 10) -> Optional[Dict[str, Any]]:
        """🔥 Get latest text emotions from Firebase (like ChatGPT)"""
        if not self.firebase_client:
            if not self.silent:
                print("❌ Firebase not available for text emotions")
            return None
        
        try:
            # Calculate time range using UTC to match Firebase storage
            from datetime import timezone
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes_back)
            
            # Query Firebase for recent text emotion readings (NO INDEXES!)
            emotions_ref = self.firebase_client.collection('emotion_readings')
            
            # Query ordered by timestamp (newest first) to get latest emotions
            query = emotions_ref.where('timestamp', '>=', cutoff_time).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(20)
            
            docs = query.stream()
            
            # Filter in code to avoid indexes - STRICT text emotion filtering
            text_emotions = []
            for doc in docs:
                data = doc.to_dict()
                # TEXT emotions have: 'role': 'user', 'emotion' (singular), 'message_id'
                # FACIAL emotions have: 'face_id', 'emotions' (plural), no 'role' field
                if (data.get('role') == 'user' and 
                    'emotion' in data and 
                    'message_id' in data and 
                    'face_id' not in data):  # Allow ALL emotions including neutral for testing
                    text_emotions.append({
                        'doc': doc,
                        'data': data,
                        'timestamp': data.get('timestamp')
                    })
                    print(f"🔍 DEBUG: Found text emotion - session: {data.get('session_id')}, emotion: {data.get('emotion')}, time: {data.get('timestamp')}")
            
            # Get latest (already ordered by timestamp descending)
            if text_emotions:
                latest = text_emotions[0]  # First item is the most recent
                data = latest['data']
                
                print(f"💬 Found Firebase text emotion: {data}")
                return {
                    'source': 'firebase_chat',
                    'timestamp': data.get('timestamp'),
                    'emotion': data.get('emotion'),
                    'confidence': data.get('confidence', 0.5),
                    'content': data.get('content_preview', ''),
                    'session_id': data.get('session_id'),
                    'message_id': data.get('message_id'),
                    'doc_id': latest['doc'].id
                }
            
            print(f"💬 No Firebase text emotions found in last {minutes_back} minutes")
            return None
            
        except Exception as e:
            print(f"❌ Error reading text emotions from Firebase: {e}")
            return None
    
    def combine_real_emotions(self, minutes_back: int = 10, strategy: str = 'adaptive') -> Optional[RealCombinedEmotion]:
        """🎯 ADVANCED Combine emotions using multiple fusion strategies"""
        if not self.silent:
            print(f"\\n🔗 COMBINING REAL EMOTIONS (last {minutes_back} minutes, strategy: {strategy})")
            print("=" * 70)
        
        # Get facial emotions from Firebase
        facial_data = self.get_latest_facial_emotions(minutes_back)
        
        # Get text emotions from JSON files
        text_data = self.get_latest_text_emotions(minutes_back)
        
        # Prepare data for advanced fusion with NORMALIZED SCALES
        facial_fusion_data = None
        if facial_data and facial_data.get('emotions'):
            emotions_dict = facial_data['emotions']
            if emotions_dict:
                dominant = max(emotions_dict.items(), key=lambda x: float(x[1]))
                facial_emotion = dominant[0]
                
                # 🔧 FIXED: Properly normalize facial confidence (percentage -> decimal)
                score = float(dominant[1])
                facial_confidence = score / 100.0 if score > 1.0 else score
                
                # 🔧 FIXED: Normalize ALL facial emotion scores to 0-1 scale
                normalized_emotions = {}
                for emotion, score in emotions_dict.items():
                    score_float = float(score)
                    normalized_score = score_float / 100.0 if score_float > 1.0 else score_float
                    normalized_emotions[emotion] = normalized_score
                
                # 🎭 MULTI-EMOTION: Get top 3 emotions for richer fusion
                sorted_emotions = sorted(normalized_emotions.items(), key=lambda x: x[1], reverse=True)
                top_emotions = sorted_emotions[:3]  # Top 3 emotions
                
                facial_fusion_data = {
                    'emotion': facial_emotion,  # Dominant emotion
                    'confidence': facial_confidence,  # Dominant emotion score
                    'emotions': normalized_emotions,  # All emotions normalized to 0-1
                    'top_emotions': top_emotions,  # Top 3 for multi-emotion fusion
                    'timestamp': facial_data.get('timestamp'),
                    'quality_score': facial_data.get('confidence', 0.5)  # Keep original confidence for quality
                }
                
                print(f"🔧 DEBUG: Facial emotions (top 3):")
                for emotion, score in top_emotions:
                    print(f"   {emotion}: {score:.3f}")
        
        text_fusion_data = None
        if text_data:
            # 🎭 MULTI-EMOTION: Check if text has multiple emotions too
            text_all_emotions = text_data.get('emotions', {})
            text_emotion_score = text_data.get('confidence', 0.5)
            
            if text_all_emotions:
                # Text has multiple emotions! Use them for richer fusion
                sorted_text_emotions = sorted(text_all_emotions.items(), key=lambda x: x[1], reverse=True)
                top_text_emotions = sorted_text_emotions[:3]  # Top 3 text emotions
                
                text_fusion_data = {
                    'emotion': text_data.get('emotion'),  # Dominant emotion
                    'confidence': text_emotion_score,  # Dominant emotion score
                    'emotions': text_all_emotions,  # ALL text emotions
                    'top_emotions': top_text_emotions,  # Top 3 for multi-emotion fusion
                    'timestamp': text_data.get('timestamp')
                }
                
                print(f"🔧 DEBUG: Text emotions (top 3):")
                for emotion, score in top_text_emotions:
                    print(f"   {emotion}: {score:.3f}")
            else:
                # Single emotion text (fallback)
                text_fusion_data = {
                    'emotion': text_data.get('emotion'),
                    'confidence': text_emotion_score,
                    'timestamp': text_data.get('timestamp')
                }
                
                print(f"🔧 DEBUG: Text emotion (single): {text_data.get('emotion')} = {text_emotion_score:.3f}")
        
        # Show analysis
        if not self.silent:
            print(f"\\n📊 ADVANCED COMBINATION ANALYSIS:")
            if facial_fusion_data:
                print(f"   👤 FACIAL: {facial_fusion_data['emotion']} (confidence: {facial_fusion_data['confidence']:.2f})")
                print(f"      Full emotions: {facial_fusion_data['emotions']}")
            else:
                print(f"   👤 FACIAL: No recent data")
                
            if text_fusion_data:
                print(f"   💬 TEXT: {text_fusion_data['emotion']} (confidence: {text_fusion_data['confidence']:.2f})")
            else:
                print(f"   💬 TEXT: No recent data")
        
        # Apply ADVANCED fusion - allow single-source emotions too
        if not facial_fusion_data and not text_fusion_data:
            if not self.silent:
                print("❌ No recent emotions found in either source")
            return None
        
        # 🎯 SINGLE SOURCE: If only one source available, use it directly
        if facial_fusion_data and not text_fusion_data:
            if not self.silent:
                print("🎯 FACIAL-ONLY combination (no recent text)")
            return RealCombinedEmotion(
                dominant_emotion=facial_fusion_data['emotion'],
                confidence=facial_fusion_data['confidence'],
                combination_method='facial_only',
                facial_source=facial_fusion_data,
                text_source=None
            )
            
        if text_fusion_data and not facial_fusion_data:
            if not self.silent:
                print("🎯 TEXT-ONLY combination (no recent facial)")
            return RealCombinedEmotion(
                dominant_emotion=text_fusion_data['emotion'],
                confidence=text_fusion_data['confidence'],
                combination_method='text_only',
                facial_source=None,
                text_source=text_fusion_data
            )
        
        # Use advanced fusion engine
        winner_emotion, winner_confidence, method = self.fusion_engine.fuse_emotions(
            facial_fusion_data, text_fusion_data, strategy=strategy
        )
        
        if not self.silent:
            print(f"   🎯 ADVANCED RESULT: {winner_emotion} (confidence: {winner_confidence:.2f})")
            print(f"   🧠 Method: {method}")
            print(f"   📈 Strategy: {strategy}")
        
        # Create combined result
        combined = RealCombinedEmotion(
            dominant_emotion=winner_emotion,
            confidence=winner_confidence,
            facial_source=facial_data,
            text_source=text_data,
            combination_method=f"{strategy}_{method}"
        )
        
        return combined
    
    def get_emotion_history(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get emotion history from both sources"""
        history = []
        
        # Get facial emotion history from Firebase
        if self.firebase_client:
            try:
                from datetime import timezone
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
                emotions_ref = self.firebase_client.collection('emotion_readings')
                query = emotions_ref.where('timestamp', '>=', cutoff_time).order_by('timestamp')
                
                for doc in query.stream():
                    data = doc.to_dict()
                    # STRICT filter: only facial emotions (not text)
                    if ('role' not in data and 
                        'emotions' in data and 
                        'face_id' in data and 
                        'message_id' not in data):
                        emotions_dict = data.get('emotions', {})
                        if emotions_dict:
                            dominant = max(emotions_dict.items(), key=lambda x: float(x[1]))
                            history.append({
                                'timestamp': data.get('timestamp'),
                                'emotion': dominant[0],
                                'confidence': float(dominant[1]) / 100.0 if float(dominant[1]) > 1.0 else float(dominant[1]),
                                'source': 'facial'
                            })
            except Exception as e:
                print(f"❌ Error getting facial history: {e}")
        
        # Get text emotion history from Firebase (NO INDEXES!)
        if self.firebase_client:
            try:
                from datetime import timezone
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
                emotions_ref = self.firebase_client.collection('emotion_readings')
                # Simple query - filter in code to avoid indexes
                query = emotions_ref.where('timestamp', '>=', cutoff_time).limit(200)
                
                for doc in query.stream():
                    data = doc.to_dict()
                    # STRICT filter: only user text emotions (not facial)
                    if (data.get('role') == 'user' and 
                        'emotion' in data and 
                        'message_id' in data and 
                        'face_id' not in data and 
                        data.get('emotion') != 'neutral'):
                        history.append({
                            'timestamp': data.get('timestamp'),
                            'emotion': data.get('emotion'),
                            'confidence': data.get('confidence', 0.5),
                            'source': 'text'
                        })
            except Exception as e:
                print(f"❌ Error getting Firebase text history: {e}")
        
        # Sort by timestamp
        history.sort(key=lambda x: x['timestamp'])
        return history
    
    def monitor_emotions(self, check_interval: int = 30):
        """Monitor and combine emotions in real-time"""
        print(f"\\n🔄 MONITORING REAL EMOTIONS (checking every {check_interval}s)")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        try:
            while True:
                combined = self.combine_real_emotions(minutes_back=5)
                
                if combined:
                    print(f"\\n[{datetime.now().strftime('%H:%M:%S')}] 🎯 CURRENT EMOTION: {combined.dominant_emotion} ({combined.confidence:.2f})")
                    print(f"   Method: {combined.combination_method}")
                    
                    if combined.facial_source:
                        print(f"   📹 Facial: Available from Firebase")
                    if combined.text_source:
                        print(f"   💬 Text: Available from chat files")
                else:
                    print(f"\\n[{datetime.now().strftime('%H:%M:%S')}] ⚪ No recent emotions detected")
                
                import time
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\\n👋 Emotion monitoring stopped")

# Global instance for function-only access
_combiner_instance = None

def get_combined_emotion(minutes_back: int = 10, strategy: str = 'adaptive'):
    """
    🎯 ADVANCED function to get combined emotion using multiple fusion strategies
    
    Args:
        minutes_back: How many minutes back to look for emotions
        strategy: Fusion strategy ('simple', 'adaptive', 'confidence_based', 'temporal_weighted', 'weighted_average')
        
    Returns:
        Dict with emotion info or None if no emotions found
        {
            'emotion': str,           # The winning emotion
            'confidence': float,      # Confidence score (0-1)
            'source': str,           # Method used for combination
            'strategy': str,         # Fusion strategy used
            'facial_data': dict,     # Raw facial data (if available)
            'text_data': dict        # Raw text data (if available)
        }
    """
    global _combiner_instance
    
    if _combiner_instance is None:
        _combiner_instance = RealEmotionCombiner(silent=False)  # DEBUG: Show detailed output
    
    try:
        combined = _combiner_instance.combine_real_emotions(minutes_back=minutes_back, strategy=strategy)
        
        if combined:
            return {
                'emotion': combined.dominant_emotion,
                'confidence': combined.confidence,
                'source': combined.combination_method,
                'strategy': strategy,
                'facial_data': combined.facial_source,
                'text_data': combined.text_source,
                'timestamp': combined.timestamp
            }
        else:
            print("🔍 DEBUG: combine_real_emotions returned None")
            return None
            
    except Exception as e:
        print(f"🔍 DEBUG: Exception in get_combined_emotion: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_emotion_simple(minutes_back: int = 10):
    """
    Even simpler function - just returns the emotion name
    
    Returns:
        str: emotion name or None
    """
    result = get_combined_emotion(minutes_back)
    return result['emotion'] if result else None

def get_emotion_with_confidence(minutes_back: int = 10):
    """
    Simple function returning emotion and confidence
    
    Returns:
        tuple: (emotion, confidence) or None
    """
    result = get_combined_emotion(minutes_back)
    return (result['emotion'], result['confidence']) if result else None

def test_emotion_fusion():
    """🧪 Test function for the ADVANCED emotion fusion"""
    print("🔗 Testing ADVANCED REAL Emotion Combiner")
    print("=" * 55)
    
    # Test all fusion strategies
    strategies = ['simple', 'adaptive', 'confidence_based', 'temporal_weighted', 'weighted_average']
    
    for strategy in strategies:
        print(f"\\n🧠 Testing {strategy.upper()} strategy:")
        print("-" * 40)
        
        result = get_combined_emotion(strategy=strategy)
        
        if result:
            print(f"✅ EMOTION: {result['emotion']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Method: {result['source']}")
            print(f"   Strategy: {result['strategy']}")
            
            if result['facial_data']:
                print(f"   📹 Facial: Available")
            if result['text_data']:
                print(f"   💬 Text: Available")
        else:
            print("❌ No recent emotions found")
    
    print(f"\\n" + "="*55)
    print("🎯 SIMPLE FUNCTION TESTS:")
    print("-" * 30)
    
    # Test simple functions
    emotion = get_emotion_simple()
    print(f"Simple emotion: {emotion}")
    
    emotion_conf = get_emotion_with_confidence()
    print(f"Emotion with confidence: {emotion_conf}")
    
    print(f"\\n💡 TIP: Use different strategies in your app:")
    print(f"   - 'simple': Original logic (fastest)")
    print(f"   - 'adaptive': Smart context-aware (recommended)")
    print(f"   - 'confidence_based': Pure confidence competition")
    print(f"   - 'temporal_weighted': Newer = better")
    print(f"   - 'weighted_average': Mathematical fusion")

if __name__ == "__main__":
    test_emotion_fusion()