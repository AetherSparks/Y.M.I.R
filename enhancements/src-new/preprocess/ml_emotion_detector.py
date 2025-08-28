"""
🧠 Y.M.I.R ML-Based Emotion Detection System
==========================================
Professional machine learning approach for emotion detection from audio features
Using supervised learning with multiple models and ensemble techniques
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
import joblib
import os
import logging
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MLEmotionDetector:
    """Professional ML-based emotion detection system"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            'Tempo', 'Energy', 'Valence', 'Danceability', 'Acousticness',
            'Loudness', 'Mode', 'Speechiness', 'Instrumentalness', 'Liveness',
            'Key', 'Track Popularity', 'Artist Popularity', 'Duration (ms)'
        ]
        self.is_trained = False
        
        # Research-based emotion categories for music
        self.emotion_categories = [
            'Joy', 'Sadness', 'Excitement', 'Calm', 'Anger', 
            'Fear', 'Surprise', 'Nostalgia', 'Love', 'Neutral'
        ]
        
        logger.info("ML Emotion Detection System initialized")
    
    def create_training_dataset(self) -> pd.DataFrame:
        """
        Create training dataset using music theory and psychological research
        This generates synthetic training data based on established research
        """
        np.random.seed(42)
        # Create more samples for diverse emotions, fewer for neutral
        emotion_samples = {
            'Joy': 250,
            'Excitement': 250, 
            'Sadness': 200,
            'Calm': 200,
            'Anger': 150,
            'Fear': 150,
            'Neutral': 100  # Significantly fewer neutral samples
        }
        
        training_data = []
        
        # Joy: High valence, high energy, major key, moderate-fast tempo
        for _ in range(emotion_samples['Joy']):
            sample = {
                'Tempo': np.random.normal(120, 25),  # BPM
                'Energy': np.random.normal(0.7, 0.15),
                'Valence': np.random.normal(0.8, 0.1),
                'Danceability': np.random.normal(0.6, 0.15),
                'Acousticness': np.random.normal(0.3, 0.2),
                'Loudness': np.random.normal(-7, 3),  # dB
                'Mode': 1,  # Major key
                'Speechiness': np.random.normal(0.05, 0.03),
                'Instrumentalness': np.random.normal(0.1, 0.15),
                'Liveness': np.random.normal(0.2, 0.1),
                'Key': np.random.randint(0, 12),
                'Track Popularity': np.random.randint(40, 90),
                'Artist Popularity': np.random.randint(30, 80),
                'Duration (ms)': np.random.normal(210000, 45000),
                'Emotion': 'Joy'
            }
            training_data.append(sample)
        
        # Sadness: Low valence, low energy, minor key, slow tempo
        for _ in range(emotion_samples['Sadness']):
            sample = {
                'Tempo': np.random.normal(75, 20),
                'Energy': np.random.normal(0.3, 0.15),
                'Valence': np.random.normal(0.2, 0.1),
                'Danceability': np.random.normal(0.3, 0.1),
                'Acousticness': np.random.normal(0.6, 0.2),
                'Loudness': np.random.normal(-12, 3),
                'Mode': 0,  # Minor key
                'Speechiness': np.random.normal(0.04, 0.02),
                'Instrumentalness': np.random.normal(0.3, 0.2),
                'Liveness': np.random.normal(0.15, 0.08),
                'Key': np.random.randint(0, 12),
                'Track Popularity': np.random.randint(20, 60),
                'Artist Popularity': np.random.randint(25, 70),
                'Duration (ms)': np.random.normal(240000, 60000),
                'Emotion': 'Sadness'
            }
            training_data.append(sample)
        
        # Excitement: High energy, high danceability, loud, fast tempo
        for _ in range(emotion_samples['Excitement']):
            sample = {
                'Tempo': np.random.normal(130, 20),
                'Energy': np.random.normal(0.85, 0.1),
                'Valence': np.random.normal(0.7, 0.15),
                'Danceability': np.random.normal(0.8, 0.1),
                'Acousticness': np.random.normal(0.1, 0.1),
                'Loudness': np.random.normal(-4, 2),
                'Mode': np.random.choice([0, 1]),
                'Speechiness': np.random.normal(0.08, 0.04),
                'Instrumentalness': np.random.normal(0.05, 0.1),
                'Liveness': np.random.normal(0.3, 0.15),
                'Key': np.random.randint(0, 12),
                'Track Popularity': np.random.randint(50, 95),
                'Artist Popularity': np.random.randint(40, 85),
                'Duration (ms)': np.random.normal(190000, 30000),
                'Emotion': 'Excitement'
            }
            training_data.append(sample)
        
        # Calm: Low energy, high acousticness, slow tempo, soft
        for _ in range(emotion_samples['Calm']):
            sample = {
                'Tempo': np.random.normal(65, 15),
                'Energy': np.random.normal(0.2, 0.1),
                'Valence': np.random.normal(0.4, 0.15),
                'Danceability': np.random.normal(0.2, 0.1),
                'Acousticness': np.random.normal(0.8, 0.15),
                'Loudness': np.random.normal(-15, 4),
                'Mode': np.random.choice([0, 1]),
                'Speechiness': np.random.normal(0.03, 0.02),
                'Instrumentalness': np.random.normal(0.6, 0.25),
                'Liveness': np.random.normal(0.1, 0.05),
                'Key': np.random.randint(0, 12),
                'Track Popularity': np.random.randint(10, 50),
                'Artist Popularity': np.random.randint(15, 60),
                'Duration (ms)': np.random.normal(300000, 90000),
                'Emotion': 'Calm'
            }
            training_data.append(sample)
        
        # Anger: High energy, loud, minor key, fast tempo, low valence
        for _ in range(emotion_samples['Anger']):
            sample = {
                'Tempo': np.random.normal(140, 30),
                'Energy': np.random.normal(0.9, 0.08),
                'Valence': np.random.normal(0.3, 0.15),
                'Danceability': np.random.normal(0.5, 0.2),
                'Acousticness': np.random.normal(0.1, 0.08),
                'Loudness': np.random.normal(-2, 1.5),
                'Mode': 0,  # Minor key predominantly
                'Speechiness': np.random.normal(0.12, 0.06),
                'Instrumentalness': np.random.normal(0.2, 0.2),
                'Liveness': np.random.normal(0.4, 0.2),
                'Key': np.random.randint(0, 12),
                'Track Popularity': np.random.randint(30, 75),
                'Artist Popularity': np.random.randint(25, 70),
                'Duration (ms)': np.random.normal(200000, 40000),
                'Emotion': 'Anger'
            }
            training_data.append(sample)
        
        # Add more emotions with similar patterns...
        # Fear: Low valence, moderate energy, minor key
        for _ in range(emotion_samples['Fear']):
            sample = {
                'Tempo': np.random.normal(90, 25),
                'Energy': np.random.normal(0.4, 0.15),
                'Valence': np.random.normal(0.25, 0.1),
                'Danceability': np.random.normal(0.3, 0.1),
                'Acousticness': np.random.normal(0.4, 0.2),
                'Loudness': np.random.normal(-10, 3),
                'Mode': 0,  # Minor key
                'Speechiness': np.random.normal(0.05, 0.03),
                'Instrumentalness': np.random.normal(0.4, 0.25),
                'Liveness': np.random.normal(0.2, 0.1),
                'Key': np.random.randint(0, 12),
                'Track Popularity': np.random.randint(20, 60),
                'Artist Popularity': np.random.randint(20, 65),
                'Duration (ms)': np.random.normal(220000, 50000),
                'Emotion': 'Fear'
            }
            training_data.append(sample)
        
        # Neutral: Balanced features
        for _ in range(emotion_samples['Neutral']):
            sample = {
                'Tempo': np.random.normal(100, 30),
                'Energy': np.random.normal(0.5, 0.2),
                'Valence': np.random.normal(0.5, 0.2),
                'Danceability': np.random.normal(0.5, 0.2),
                'Acousticness': np.random.normal(0.4, 0.25),
                'Loudness': np.random.normal(-8, 4),
                'Mode': np.random.choice([0, 1]),
                'Speechiness': np.random.normal(0.06, 0.04),
                'Instrumentalness': np.random.normal(0.3, 0.3),
                'Liveness': np.random.normal(0.2, 0.15),
                'Key': np.random.randint(0, 12),
                'Track Popularity': np.random.randint(25, 75),
                'Artist Popularity': np.random.randint(25, 75),
                'Duration (ms)': np.random.normal(220000, 60000),
                'Emotion': 'Neutral'
            }
            training_data.append(sample)
        
        df = pd.DataFrame(training_data)
        
        # Clip values to realistic ranges
        df['Tempo'] = np.clip(df['Tempo'], 40, 200)
        df['Energy'] = np.clip(df['Energy'], 0, 1)
        df['Valence'] = np.clip(df['Valence'], 0, 1)
        df['Danceability'] = np.clip(df['Danceability'], 0, 1)
        df['Acousticness'] = np.clip(df['Acousticness'], 0, 1)
        df['Loudness'] = np.clip(df['Loudness'], -25, 0)
        df['Speechiness'] = np.clip(df['Speechiness'], 0, 1)
        df['Instrumentalness'] = np.clip(df['Instrumentalness'], 0, 1)
        df['Liveness'] = np.clip(df['Liveness'], 0, 1)
        df['Track Popularity'] = np.clip(df['Track Popularity'], 0, 100)
        df['Artist Popularity'] = np.clip(df['Artist Popularity'], 0, 100)
        df['Duration (ms)'] = np.clip(df['Duration (ms)'], 30000, 600000)
        
        logger.info(f"Generated training dataset: {len(df)} samples, {len(df['Emotion'].unique())} emotions")
        return df
    
    def create_feature_combinations(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features based on music theory"""
        X_enhanced = X.copy()
        
        # Emotional combinations
        X_enhanced['energy_valence'] = X['Energy'] * X['Valence']
        X_enhanced['tempo_energy'] = X['Tempo'] * X['Energy'] / 100
        X_enhanced['acoustic_energy_ratio'] = X['Acousticness'] / (X['Energy'] + 0.01)
        
        # Musical characteristics
        X_enhanced['is_major'] = (X['Mode'] == 1).astype(int)
        X_enhanced['is_high_energy'] = (X['Energy'] > 0.7).astype(int)
        X_enhanced['is_danceable'] = (X['Danceability'] > 0.6).astype(int)
        X_enhanced['is_acoustic'] = (X['Acousticness'] > 0.5).astype(int)
        
        # Tempo categories
        X_enhanced['tempo_slow'] = (X['Tempo'] < 80).astype(int)
        X_enhanced['tempo_moderate'] = ((X['Tempo'] >= 80) & (X['Tempo'] < 120)).astype(int)
        X_enhanced['tempo_fast'] = (X['Tempo'] >= 120).astype(int)
        
        # Popularity features
        X_enhanced['popularity_ratio'] = X['Track Popularity'] / (X['Artist Popularity'] + 1)
        
        return X_enhanced
    
    def train_models(self) -> Dict:
        """Train multiple ML models with proper validation"""
        logger.info("Training ML emotion detection models...")
        
        # Create training dataset
        training_df = self.create_training_dataset()
        
        # Prepare features and labels
        X = training_df[self.feature_columns]
        y = training_df['Emotion']
        
        # Feature engineering
        X_enhanced = self.create_feature_combinations(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_enhanced, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=10,
                random_state=42
            ),
            'SVM': SVC(
                C=1.0, kernel='rbf', probability=True, random_state=42
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100, 50), learning_rate_init=0.001,
                max_iter=500, random_state=42
            )
        }
        
        # Train and evaluate models
        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            
            # Full training
            model.fit(X_train_scaled, y_train)
            
            # Test predictions
            y_pred = model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_accuracy
            }
            
            logger.info(f"{name} - CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}, Test: {test_accuracy:.3f}")
        
        # Create ensemble
        logger.info("Creating ensemble model...")
        best_models = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)[:3]
        
        ensemble_estimators = [(name, results[name]['model']) for name, _ in best_models]
        self.ensemble_model = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft'  # Use probability predictions
        )
        
        self.ensemble_model.fit(X_train_scaled, y_train)
        ensemble_pred = self.ensemble_model.predict(X_test_scaled)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        logger.info(f"Ensemble accuracy: {ensemble_accuracy:.3f}")
        
        self.models = {name: results[name]['model'] for name in results}
        self.is_trained = True
        
        return results
    
    def predict_emotion(self, features: Dict) -> Tuple[str, float]:
        """Predict emotion with confidence score"""
        if not self.is_trained:
            logger.warning("Model not trained. Training now...")
            self.train_models()
        
        # Create feature vector
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0))
        
        # Convert to DataFrame for feature engineering
        X = pd.DataFrame([feature_vector], columns=self.feature_columns)
        X_enhanced = self.create_feature_combinations(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_enhanced)
        
        # Predict with ensemble
        prediction_proba = self.ensemble_model.predict_proba(X_scaled)[0]
        prediction_class = self.ensemble_model.predict(X_scaled)[0]
        
        # Get emotion name and confidence
        emotion = self.label_encoder.inverse_transform([prediction_class])[0]
        confidence = max(prediction_proba)
        
        # Anti-neutral bias: Encourage diverse emotion predictions
        if emotion == 'Neutral' and confidence < 0.7:  # High threshold for neutral
            # Find next best non-neutral emotion
            proba_with_emotions = list(zip(self.label_encoder.classes_, prediction_proba))
            proba_with_emotions.sort(key=lambda x: x[1], reverse=True)
            
            for alt_emotion, alt_confidence in proba_with_emotions:
                if alt_emotion != 'Neutral' and alt_confidence > 0.2:  # Lower threshold for alternatives
                    emotion = alt_emotion
                    confidence = min(alt_confidence * 1.2, 1.0)  # Boost confidence
                    break
        
        # Boost confidence for non-neutral predictions
        elif emotion != 'Neutral' and confidence > 0.25:
            confidence = min(confidence * 1.1, 1.0)  # 10% boost for diversity
        
        return emotion, confidence
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            logger.error("Cannot save untrained model")
            return
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'individual_models': self.models,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'emotion_categories': self.emotion_categories
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return False
        
        try:
            model_data = joblib.load(filepath)
            self.ensemble_model = model_data['ensemble_model']
            self.models = model_data['individual_models']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data['feature_columns']
            self.emotion_categories = model_data['emotion_categories']
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


def main():
    """Test the ML emotion detection system"""
    detector = MLEmotionDetector()
    
    # Train models
    results = detector.train_models()
    
    # Test prediction
    test_features = {
        'Tempo': 120,
        'Energy': 0.8,
        'Valence': 0.7,
        'Danceability': 0.6,
        'Acousticness': 0.2,
        'Loudness': -5,
        'Mode': 1,
        'Speechiness': 0.05,
        'Instrumentalness': 0.1,
        'Liveness': 0.3,
        'Key': 5,
        'Track Popularity': 70,
        'Artist Popularity': 60,
        'Duration (ms)': 210000
    }
    
    emotion, confidence = detector.predict_emotion(test_features)
    print(f"Predicted emotion: {emotion} (confidence: {confidence:.3f})")
    
    # Save model
    detector.save_model("models/ml_emotion_detector.joblib")


if __name__ == "__main__":
    main()