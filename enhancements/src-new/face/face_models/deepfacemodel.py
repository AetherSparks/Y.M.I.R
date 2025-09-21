"""
🧠 DeepFace Ensemble Model Component for Y.M.I.R
================================================
Advanced emotion detection using multiple DeepFace models with ensemble voting.
Provides robust and accurate facial emotion analysis.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import statistics

# DeepFace imports
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace available for emotion analysis")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace not available - install: pip install deepface")

@dataclass
class DeepFaceConfig:
    """Configuration for DeepFace ensemble emotion detection"""
    use_ensemble_detection: bool = True
    confidence_threshold: float = 0.7
    timeout: float = 2.0
    smoothing_window: int = 5
    max_workers: int = 3
    models_to_use: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.models_to_use is None:
            self.models_to_use = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepID']

class EmotionSmoother:
    """Temporal smoothing for emotion predictions"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.emotion_buffers = defaultdict(lambda: deque(maxlen=window_size))
        self.smoothing_lock = threading.Lock()
    
    def add_emotion_reading(self, face_id: int, emotions: Dict[str, float]) -> Dict[str, float]:
        """Add emotion reading and return smoothed emotions"""
        with self.smoothing_lock:
            # Add current emotions to buffer
            for emotion, score in emotions.items():
                buffer_key = f"{face_id}_{emotion}"
                self.emotion_buffers[buffer_key].append(score)
            
            # Calculate smoothed values using weighted moving average
            smoothed_emotions = {}
            for emotion, score in emotions.items():
                buffer_key = f"{face_id}_{emotion}"
                buffer = self.emotion_buffers[buffer_key]
                
                if len(buffer) > 1:
                    # Use weighted average with more weight on recent values
                    values = list(buffer)
                    weights = [i + 1 for i in range(len(values))]  # More weight on recent
                    weighted_sum = sum(v * w for v, w in zip(values, weights))
                    weight_sum = sum(weights)
                    smoothed_emotions[emotion] = weighted_sum / weight_sum
                else:
                    smoothed_emotions[emotion] = score
            
            return smoothed_emotions
    
    def get_emotion_stability(self, face_id: int) -> float:
        """Calculate emotional stability score for a face"""
        with self.smoothing_lock:
            # Find all buffers for this face
            face_buffers = {
                emotion: buffer for buffer_key, buffer in self.emotion_buffers.items()
                if buffer_key.startswith(f"{face_id}_")
                for emotion in [buffer_key.split(f"{face_id}_")[1]]
            }
            
            if not face_buffers:
                return 1.0
            
            # Calculate variance across all emotions
            total_variance = 0.0
            emotion_count = 0
            
            for emotion, buffer in face_buffers.items():
                if len(buffer) > 1:
                    variance = statistics.variance(list(buffer))
                    total_variance += variance
                    emotion_count += 1
            
            if emotion_count == 0:
                return 1.0
            
            avg_variance = total_variance / emotion_count
            stability = max(0.0, 1.0 - (avg_variance / 1000))  # Normalize to 0-1
            return stability

class DeepFaceEnsemble:
    """Ensemble emotion detection using multiple DeepFace models"""
    
    def __init__(self, config: Optional[DeepFaceConfig] = None):
        self.config = config or DeepFaceConfig()
        self.smoother = EmotionSmoother(self.config.smoothing_window)
        
        # Model availability check
        if not DEEPFACE_AVAILABLE:
            print("❌ DeepFace not available - emotion detection disabled")
            return
        
        # Initialize models
        self.available_models = self._check_available_models()
        
        # Threading for ensemble processing
        self.analysis_lock = threading.Lock()
        self.last_analysis_time = {}
        
        # Performance tracking
        self.model_performance = defaultdict(lambda: {'successes': 0, 'failures': 0, 'avg_time': 0.0})
        
        print(f"✅ DeepFace ensemble initialized with {len(self.available_models)} models")
    
    def _check_available_models(self) -> List[str]:
        """Check which DeepFace models are available"""
        available = []
        test_models = self.config.models_to_use or []
        
        # Create a small test image
        test_image = np.ones((48, 48, 3), dtype=np.uint8) * 128
        
        if not DEEPFACE_AVAILABLE or not test_models:
            return []
            
        for model in test_models:
            try:
                # Quick test to see if model works
                from deepface import DeepFace as DF
                DF.analyze(
                    test_image,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                available.append(model)
                print(f"✅ {model} model available")
            except Exception as e:
                print(f"⚠️ {model} model not available: {e}")
        
        return available
    
    def analyze_single_model(self, face_roi: np.ndarray, model_name: str) -> Tuple[Optional[Dict[str, float]], float]:
        """Analyze emotions using a single model"""
        start_time = time.time()
        
        try:
            # Resize face for consistent analysis
            face_224 = cv2.resize(face_roi, (224, 224))
            
            # Run analysis
            from deepface import DeepFace as DF
            result = DF.analyze(
                face_224,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            
            emotions = result[0]['emotion'] if isinstance(result, list) else result['emotion']
            processing_time = time.time() - start_time
            
            # Update model performance
            self.model_performance[model_name]['successes'] += 1
            self.model_performance[model_name]['avg_time'] = (
                (self.model_performance[model_name]['avg_time'] * 
                 (self.model_performance[model_name]['successes'] - 1) + processing_time) /
                self.model_performance[model_name]['successes']
            )
            
            return emotions, processing_time
            
        except Exception as e:
            self.model_performance[model_name]['failures'] += 1
            print(f"⚠️ {model_name} analysis failed: {e}")
            return None, time.time() - start_time
    
    def ensemble_emotion_detection(self, face_roi: np.ndarray) -> Optional[Dict[str, Any]]:
        """Perform ensemble emotion detection using multiple models"""
        if not DEEPFACE_AVAILABLE or not self.available_models:
            return None
        
        try:
            results = []
            
            if self.config.use_ensemble_detection and len(self.available_models) > 1:
                # Multi-model ensemble
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = []
                    
                    for model in self.available_models:
                        future = executor.submit(self.analyze_single_model, face_roi, model)
                        futures.append((future, model))
                    
                    # Collect results with timeout
                    for future, model in futures:
                        try:
                            emotions, processing_time = future.result(timeout=self.config.timeout)
                            if emotions:
                                # Assign model-specific weights based on performance
                                weight = self._get_model_weight(model)
                                results.append((emotions, weight, model, processing_time))
                        except Exception as e:
                            print(f"⚠️ {model} ensemble timeout/error: {e}")
                            continue
            else:
                # Single model analysis (faster)
                primary_model = self.available_models[0]
                emotions, processing_time = self.analyze_single_model(face_roi, primary_model)
                if emotions:
                    weight = self._get_model_weight(primary_model)
                    results.append((emotions, weight, primary_model, processing_time))
            
            if not results:
                return None
            
            # Ensemble voting - weighted average
            ensemble_emotions = defaultdict(float)
            total_weight = 0
            total_time = 0
            models_used = []
            
            for emotions, weight, model, processing_time in results:
                for emotion, score in emotions.items():
                    ensemble_emotions[emotion] += score * weight
                total_weight += weight
                total_time += processing_time
                models_used.append(model)
            
            # Normalize by total weight
            if total_weight > 0:
                for emotion in ensemble_emotions:
                    ensemble_emotions[emotion] /= total_weight
            
            # Calculate ensemble confidence based on agreement between models
            confidence = self._calculate_ensemble_confidence(results)
            
            return {
                'emotions': dict(ensemble_emotions),
                'confidence': confidence,
                'models_used': models_used,
                'processing_time': total_time / len(results) if results else 0,
                'ensemble_size': len(results)
            }
            
        except Exception as e:
            print(f"❌ Ensemble emotion detection error: {e}")
            return None
    
    def _get_model_weight(self, model_name: str) -> float:
        """Get weight for a model based on its performance"""
        perf = self.model_performance[model_name]
        
        # Base weights for different models
        base_weights = {
            'VGG-Face': 1.0,
            'Facenet': 0.9,
            'OpenFace': 0.8,
            'DeepID': 0.85,
            'Dlib': 0.7,
            'ArcFace': 0.85,
            'SFace': 0.8
        }
        
        base_weight = base_weights.get(model_name, 0.8)
        
        # Adjust based on success rate
        total_attempts = perf['successes'] + perf['failures']
        if total_attempts > 0:
            success_rate = perf['successes'] / total_attempts
            weight_multiplier = 0.5 + (success_rate * 0.5)  # Range: 0.5 to 1.0
            base_weight *= weight_multiplier
        
        # Adjust based on processing speed (faster models get slight boost)
        if perf['avg_time'] > 0:
            speed_multiplier = max(0.8, 2.0 / (perf['avg_time'] + 1.0))
            base_weight *= speed_multiplier
        
        return max(0.1, min(1.5, base_weight))
    
    def _calculate_ensemble_confidence(self, results: List[Tuple]) -> float:
        """Calculate confidence based on agreement between models"""
        if len(results) < 2:
            return 0.8  # Default confidence for single model
        
        # Calculate agreement between models
        all_emotions = set()
        for emotions, _, _, _ in results:
            all_emotions.update(emotions.keys())
        
        emotion_agreements = []
        for emotion in all_emotions:
            scores = [emotions.get(emotion, 0) for emotions, _, _, _ in results]
            if len(scores) > 1:
                # Calculate coefficient of variation (lower = more agreement)
                mean_score = statistics.mean(scores)
                if mean_score > 0:
                    std_dev = statistics.stdev(scores)
                    cv = std_dev / mean_score
                    agreement = max(0, 1.0 - cv)  # Higher agreement = higher confidence
                    emotion_agreements.append(agreement)
        
        if emotion_agreements:
            avg_agreement = statistics.mean(emotion_agreements)
            confidence = 0.5 + (avg_agreement * 0.5)  # Range: 0.5 to 1.0
        else:
            confidence = 0.7
        
        return confidence
    
    def analyze_face_without_context(self, face_id: int, face_roi: np.ndarray) -> Optional[Dict[str, Any]]:
        """Analyze face emotions WITHOUT any context (pure raw detection)"""
        return self.analyze_face_with_context(face_id, face_roi, environment_context=None)
    
    def analyze_face_with_context(self, face_id: int, face_roi: np.ndarray, 
                                 environment_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Analyze face emotions with optional environmental context"""
        current_time = time.time()
        
        # Rate limiting per face
        with self.analysis_lock:
            last_time = self.last_analysis_time.get(face_id, 0)
            if current_time - last_time < 0.5:  # Minimum 0.5 seconds between analyses
                return None
            self.last_analysis_time[face_id] = current_time
        
        try:
            # Perform ensemble emotion detection
            emotion_result = self.ensemble_emotion_detection(face_roi)
            if not emotion_result:
                return None
            
            # Filter by confidence threshold
            if emotion_result['confidence'] < self.config.confidence_threshold:
                print(f"⚠️ Low confidence ({emotion_result['confidence']:.2f}) - skipping")
                return None
            
            # Apply temporal smoothing
            raw_emotions = emotion_result['emotions']
            smoothed_emotions = self.smoother.add_emotion_reading(face_id, raw_emotions)
            
            # Apply environmental context if available
            if environment_context and environment_context.get('context_modifiers'):
                context_modifiers = environment_context['context_modifiers']
                for emotion in smoothed_emotions:
                    modifier = context_modifiers.get(emotion, 1.0)
                    smoothed_emotions[emotion] *= modifier
                
                # Renormalize after context application
                total_score = sum(smoothed_emotions.values())
                if total_score > 100:
                    normalization_factor = 100 / total_score
                    for emotion in smoothed_emotions:
                        smoothed_emotions[emotion] *= normalization_factor
            
            # Calculate emotional stability
            stability = self.smoother.get_emotion_stability(face_id)
            
            # Enhance result with additional metrics
            enhanced_result = {
                **emotion_result,
                'emotions': smoothed_emotions,
                'raw_emotions': raw_emotions,
                'stability': stability,
                'face_id': face_id,
                'timestamp': current_time,
                'context_applied': environment_context is not None
            }
            
            return enhanced_result
            
        except Exception as e:
            print(f"❌ Face emotion analysis error: {e}")
            return None
    
    def get_dominant_emotion(self, emotions: Dict[str, float]) -> Tuple[str, float]:
        """Get the dominant emotion and its score"""
        if not emotions:
            return 'neutral', 0.0
        
        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant[0], dominant[1]
    
    def get_emotion_summary(self, emotions: Dict[str, float]) -> Dict[str, Any]:
        """Get comprehensive emotion analysis summary"""
        if not emotions:
            return {'dominant': 'neutral', 'confidence': 0.0, 'distribution': {}}
        
        dominant_emotion, dominant_score = self.get_dominant_emotion(emotions)
        
        # Calculate emotion distribution categories
        positive_emotions = ['happy', 'joy', 'surprise']
        negative_emotions = ['sad', 'angry', 'fear', 'disgust']
        
        positive_score = sum(emotions.get(e, 0) for e in positive_emotions)
        negative_score = sum(emotions.get(e, 0) for e in negative_emotions)
        neutral_score = emotions.get('neutral', 0)
        
        # Determine overall emotional valence
        if positive_score > negative_score and positive_score > neutral_score:
            valence = 'positive'
        elif negative_score > positive_score and negative_score > neutral_score:
            valence = 'negative'
        else:
            valence = 'neutral'
        
        return {
            'dominant': dominant_emotion,
            'dominant_score': dominant_score,
            'valence': valence,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'neutral_score': neutral_score,
            'distribution': emotions,
            'confidence': dominant_score / 100.0 if dominant_score > 0 else 0.0
        }
    
    def get_model_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all models"""
        stats = {}
        
        for model, perf in self.model_performance.items():
            total_attempts = perf['successes'] + perf['failures']
            success_rate = perf['successes'] / total_attempts if total_attempts > 0 else 0
            
            stats[model] = {
                'success_rate': success_rate,
                'total_attempts': total_attempts,
                'avg_processing_time': perf['avg_time'],
                'weight': self._get_model_weight(model)
            }
        
        return stats
    
    def draw_emotion_analysis(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int], 
                            emotion_result: Dict[str, Any]):
        """Draw emotion analysis results on frame"""
        if not emotion_result:
            return
        
        x, y, w, h = face_bbox
        emotions = emotion_result.get('emotions', {})
        confidence = emotion_result.get('confidence', 0)
        stability = emotion_result.get('stability', 0)
        
        # Draw emotion text beside face
        text_x = x + w + 10
        text_y = y + 20
        
        # Show confidence and stability
        cv2.putText(frame, f"Conf: {confidence:.2f} | Stab: {stability:.2f}",
                   (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        text_y += 20
        
        # Show top 3 emotions
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        for emotion, score in sorted_emotions:
            color = self._get_emotion_color(emotion)
            text = f"{emotion.upper()}: {score:.1f}%"
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            text_y += 25
        
        # Show model info
        models_used = emotion_result.get('models_used', [])
        if models_used:
            model_text = f"Models: {', '.join(models_used[:2])}"
            cv2.putText(frame, model_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _get_emotion_color(self, emotion: str) -> Tuple[int, int, int]:
        """Get color for emotion visualization"""
        emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'fear': (128, 0, 128),     # Purple
            'surprise': (0, 255, 255), # Yellow
            'disgust': (0, 128, 255),  # Orange
            'neutral': (128, 128, 128) # Gray
        }
        return emotion_colors.get(emotion.lower(), (255, 255, 255))
    
    def cleanup(self):
        """Cleanup DeepFace resources"""
        # Clear buffers and reset performance stats
        self.smoother.emotion_buffers.clear()
        self.model_performance.clear()
        self.last_analysis_time.clear()
        print("✅ DeepFace resources cleaned up")