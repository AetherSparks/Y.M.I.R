"""
TRUE Machine Learning Emotion Context System
===========================================
NO RULES, NO HARDCODED CATEGORIES, PURE DATA-DRIVEN LEARNING
Uses unsupervised learning to discover patterns in emotional responses to environments
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


class TrueMLEmotionContext:
    """🧠 TRUE ML: Learns emotion-environment patterns WITHOUT any predefined rules"""
    
    def __init__(self, firebase_manager=None):
        self.firebase_manager = firebase_manager
        
        # Raw data storage - NO CATEGORIES, just pure observations
        self.raw_observations = deque(maxlen=2000)
        
        # ML Components - discover patterns, don't impose them
        self.object_vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
        self.emotion_predictor = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)
        
        # Unsupervised pattern discovery
        self.environment_clusterer = KMeans(n_clusters=8, random_state=42)
        self.emotion_pattern_clusterer = KMeans(n_clusters=6, random_state=42)
        
        # Learning state
        self.is_trained = False
        self.training_threshold = 20  # Reduced threshold for faster initial training
        self.last_retrain = None
        
        # Discovered patterns (learned, not hardcoded)
        self.discovered_patterns = {}
        self.environment_clusters = {}
        self.emotion_clusters = {}
        
        # Model persistence
        self.model_path = "enhancements/src-new/face/true_ml_models/"
        os.makedirs(self.model_path, exist_ok=True)
        
        self._load_models()
    
    def analyze_context_pure_ml(self, objects: List[Dict[str, Any]], 
                               current_emotions: Dict[str, float]) -> Dict[str, float]:
        """🎯 PURE ML: No rules, only learned patterns"""
        
        # Store raw observation for learning
        observation = {
            'timestamp': datetime.now(),
            'objects': [obj.get('class', 'unknown') for obj in objects],
            'object_confidences': [obj.get('confidence', 0.5) for obj in objects],
            'emotions': current_emotions.copy(),
            'dominant_emotion': max(current_emotions.items(), key=lambda x: x[1])[0],
            'emotion_confidence': max(current_emotions.values()),
            'context_features': self._extract_pure_features(objects)
        }
        
        self.raw_observations.append(observation)
        
        # If we have enough data and not trained, train the system
        if len(self.raw_observations) >= self.training_threshold and not self.is_trained:
            self._train_pure_ml_system()
        
        # If trained, use ML to predict context influence
        if self.is_trained:
            return self._predict_emotion_context_ml(objects, current_emotions)
        else:
            # During learning phase, return unmodified emotions
            return current_emotions
    
    def _extract_pure_features(self, objects: List[Dict[str, Any]]) -> Dict[str, float]:
        """🔍 Extract numerical features WITHOUT predefined categories"""
        
        if not objects:
            return {
                'object_count': 0,
                'avg_confidence': 0,
                'confidence_std': 0,
                'time_hour': datetime.now().hour / 24.0,
                'time_day': datetime.now().weekday() / 7.0
            }
        
        # Pure statistical features from object data
        confidences = [obj.get('confidence', 0.5) for obj in objects]
        object_names = [obj.get('class', '') for obj in objects]
        
        # Basic numerical features
        features = {
            'object_count': len(objects),
            'unique_object_count': len(set(object_names)),
            'avg_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences),
            'confidence_std': np.std(confidences),
            'confidence_range': np.max(confidences) - np.min(confidences),
            'time_hour': datetime.now().hour / 24.0,
            'time_day': datetime.now().weekday() / 7.0
        }
        
        # Object name length and character statistics (discover linguistic patterns)
        if object_names:
            name_lengths = [len(name) for name in object_names if name]
            if name_lengths:
                features.update({
                    'avg_name_length': np.mean(name_lengths),
                    'max_name_length': np.max(name_lengths),
                    'total_characters': sum(name_lengths)
                })
        
        # Spatial features if available
        if objects and 'bbox' in objects[0]:
            bboxes = [obj.get('bbox', [0, 0, 100, 100]) for obj in objects]
            areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
            centers_x = [(bbox[0] + bbox[2]) / 2 for bbox in bboxes]
            centers_y = [(bbox[1] + bbox[3]) / 2 for bbox in bboxes]
            
            if areas:
                features.update({
                    'total_area': sum(areas),
                    'avg_area': np.mean(areas),
                    'area_std': np.std(areas),
                    'center_spread_x': np.std(centers_x) if len(centers_x) > 1 else 0,
                    'center_spread_y': np.std(centers_y) if len(centers_y) > 1 else 0
                })
        
        return features
    
    def _train_pure_ml_system(self):
        """🎓 Train the system using ONLY discovered patterns"""
        
        try:
            # Prepare pure feature matrix
            feature_data = []
            emotion_data = []
            object_text_data = []
            
            for obs in self.raw_observations:
                # Numerical features
                features = list(obs['context_features'].values())
                feature_data.append(features)
                
                # Emotion response vector
                emotion_vector = [obs['emotions'].get(emotion, 0) for emotion in 
                                ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']]
                emotion_data.append(emotion_vector)
                
                # Object text for pattern discovery
                object_text = ' '.join(obs['objects'])
                object_text_data.append(object_text)
            
            # Convert to arrays
            X_numerical = np.array(feature_data)
            y_emotions = np.array(emotion_data)
            
            # Discover object patterns using TF-IDF (unsupervised)
            if object_text_data:
                X_text = self.object_vectorizer.fit_transform(object_text_data)
                X_text_dense = X_text.toarray()
                
                # Combine numerical and text features
                if X_numerical.size > 0:
                    X_combined = np.hstack([X_numerical, X_text_dense])
                else:
                    X_combined = X_text_dense
            else:
                X_combined = X_numerical
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_combined)
            
            # Dimensionality reduction
            if X_scaled.shape[1] > 50:
                X_reduced = self.pca.fit_transform(X_scaled)
            else:
                X_reduced = X_scaled
            
            # Discover environment clusters (unsupervised)
            environment_clusters = self.environment_clusterer.fit_predict(X_reduced)
            
            # Discover emotion pattern clusters
            emotion_clusters = self.emotion_pattern_clusterer.fit_predict(y_emotions)
            
            # Train neural network to predict emotion responses
            self.emotion_predictor.fit(X_reduced, y_emotions)
            
            # Store discovered patterns
            for i, obs in enumerate(self.raw_observations):
                env_cluster = environment_clusters[i]
                emotion_cluster = emotion_clusters[i]
                
                if env_cluster not in self.environment_clusters:
                    self.environment_clusters[env_cluster] = []
                self.environment_clusters[env_cluster].append(obs)
                
                if emotion_cluster not in self.emotion_clusters:
                    self.emotion_clusters[emotion_cluster] = []
                self.emotion_clusters[emotion_cluster].append(obs)
            
            self.is_trained = True
            self.last_retrain = datetime.now()
            
            # Training complete - system ready
            
            # Save the trained system
            self._save_models()
            
        except Exception as e:
            # Training failed - continuing with previous state
            pass
    
    def _predict_emotion_context_ml(self, objects: List[Dict[str, Any]], 
                                   current_emotions: Dict[str, float]) -> Dict[str, float]:
        """🎯 Use pure ML to predict emotion context influence"""
        
        try:
            # Extract features for current context
            features = self._extract_pure_features(objects)
            feature_vector = list(features.values())
            
            # Get object text
            object_text = ' '.join([obj.get('class', '') for obj in objects])
            
            # Transform features same way as training
            if hasattr(self.object_vectorizer, 'vocabulary_'):
                try:
                    text_features = self.object_vectorizer.transform([object_text]).toarray()[0]
                    combined_features = np.concatenate([feature_vector, text_features])
                except:
                    combined_features = np.array(feature_vector)
            else:
                combined_features = np.array(feature_vector)
            
            # Scale and reduce dimensions
            if combined_features.size > 0:
                scaled_features = self.scaler.transform([combined_features])
                
                if hasattr(self.pca, 'components_'):
                    reduced_features = self.pca.transform(scaled_features)
                else:
                    reduced_features = scaled_features
                
                # Predict emotion response using neural network
                predicted_emotions = self.emotion_predictor.predict(reduced_features)[0]
                
                # Convert back to emotion dictionary
                emotion_names = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
                predicted_dict = {name: max(0, value) for name, value in zip(emotion_names, predicted_emotions)}
                
                # Normalize to percentages
                total = sum(predicted_dict.values())
                if total > 0:
                    predicted_dict = {k: (v/total)*100 for k, v in predicted_dict.items()}
                
                return predicted_dict
            
        except Exception as e:
            # Prediction failed - using fallback
            pass
        
        # Fallback to current emotions if prediction fails
        return current_emotions
    
    def _save_models(self):
        """💾 Save the trained ML system"""
        try:
            model_data = {
                'emotion_predictor': self.emotion_predictor,
                'scaler': self.scaler,
                'pca': self.pca,
                'object_vectorizer': self.object_vectorizer,
                'environment_clusterer': self.environment_clusterer,
                'emotion_pattern_clusterer': self.emotion_pattern_clusterer,
                'environment_clusters': self.environment_clusters,
                'emotion_clusters': self.emotion_clusters,
                'discovered_patterns': self.discovered_patterns,
                'is_trained': self.is_trained,
                'last_retrain': self.last_retrain,
                'training_samples': len(self.raw_observations)
            }
            
            model_file = os.path.join(self.model_path, 'true_ml_emotion_context.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Models saved successfully
            
        except Exception as e:
            # Model saving failed
            pass
    
    def _load_models(self):
        """📁 Load trained ML system"""
        try:
            model_file = os.path.join(self.model_path, 'true_ml_emotion_context.pkl')
            
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.emotion_predictor = model_data.get('emotion_predictor', self.emotion_predictor)
                self.scaler = model_data.get('scaler', self.scaler)
                self.pca = model_data.get('pca', self.pca)
                self.object_vectorizer = model_data.get('object_vectorizer', self.object_vectorizer)
                self.environment_clusterer = model_data.get('environment_clusterer', self.environment_clusterer)
                self.emotion_pattern_clusterer = model_data.get('emotion_pattern_clusterer', self.emotion_pattern_clusterer)
                self.environment_clusters = model_data.get('environment_clusters', {})
                self.emotion_clusters = model_data.get('emotion_clusters', {})
                self.discovered_patterns = model_data.get('discovered_patterns', {})
                self.is_trained = model_data.get('is_trained', False)
                self.last_retrain = model_data.get('last_retrain')
                
                training_samples = model_data.get('training_samples', 0)
                
                # Models loaded successfully
                
        except Exception as e:
            # Model loading failed - using defaults
            pass
    
    def get_discovery_insights(self) -> Dict[str, Any]:
        """🔍 Get insights about discovered patterns"""
        insights = {
            'total_observations': len(self.raw_observations),
            'is_trained': self.is_trained,
            'training_progress': min(100, (len(self.raw_observations) / self.training_threshold) * 100),
            'environment_patterns': len(self.environment_clusters),
            'emotion_patterns': len(self.emotion_clusters),
            'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None
        }
        
        if self.is_trained:
            # Analyze discovered patterns
            pattern_analysis = {}
            for cluster_id, observations in self.environment_clusters.items():
                dominant_emotions = {}
                for obs in observations:
                    dominant = obs['dominant_emotion']
                    dominant_emotions[dominant] = dominant_emotions.get(dominant, 0) + 1
                
                pattern_analysis[f'env_pattern_{cluster_id}'] = {
                    'sample_count': len(observations),
                    'common_emotions': dict(sorted(dominant_emotions.items(), key=lambda x: x[1], reverse=True)[:3])
                }
            
            insights['discovered_patterns'] = pattern_analysis
        
        return insights
    
    def force_retrain(self):
        """🔄 Force retraining of the ML system"""
        if len(self.raw_observations) >= 5:  # Minimum for forced training (reduced)
            self.is_trained = False
            self._train_pure_ml_system()
    
    def add_manual_observation(self, objects: List[str], emotions: Dict[str, float]):
        """➕ Add manual observation for training"""
        observation = {
            'timestamp': datetime.now(),
            'objects': objects,
            'object_confidences': [1.0] * len(objects),
            'emotions': emotions,
            'dominant_emotion': max(emotions.items(), key=lambda x: x[1])[0],
            'emotion_confidence': max(emotions.values()),
            'context_features': {'manual_entry': 1.0}
        }
        
        self.raw_observations.append(observation)