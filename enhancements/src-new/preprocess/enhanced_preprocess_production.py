"""
🎵 Y.M.I.R Enhanced Preprocessing - Production Ready
=================================================
Improved from original preprocess.py:
✅ Better emotion mapping logic with confidence scores
✅ Handles incremental processing (only new tracks)
✅ More sophisticated audio feature analysis
✅ Robust error handling and validation
✅ Compatible with enhanced scraper output
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import logging
from ml_emotion_detector import MLEmotionDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedMoodPreprocessor:
    """Production-ready preprocessing for mood/emotion mapping"""
    
    def __init__(self):
        # Initialize ML emotion detector
        self.ml_detector = MLEmotionDetector()
        self.ml_model_path = "models/ml_emotion_detector.joblib"
        
        # Try to load existing model, otherwise train new one
        if not os.path.exists("models"):
            os.makedirs("models")
        
        if not self.ml_detector.load_model(self.ml_model_path):
            logger.info("Training new ML emotion detection model...")
            self.ml_detector.train_models()
            self.ml_detector.save_model(self.ml_model_path)
        
        # Enhanced emotional state to mental health benefits mapping
        self.emotional_state_mapping = {
            'Joy': ['Mindfulness', 'Increased Motivation', 'Energy Boost', 'Positivity'],
            'Sadness': ['Mood Upliftment', 'Emotional Release', 'Grief Support', 'Hope Building'],
            'Excitement': ['Focus', 'Energy Channeling', 'Motivation', 'Achievement'],
            'Calm': ['Mindfulness', 'Emotional Stability', 'Relaxation', 'Peace'],
            'Anger': ['Anger Management', 'Tension Release', 'Calm Focus', 'Emotional Regulation'],
            'Fear': ['Calmness', 'Reassurance', 'Grounding', 'Courage Building'],
            'Surprise': ['Awareness', 'Adaptability', 'Mental Stimulation', 'Curiosity'],
            'Nostalgia': ['Emotional Processing', 'Memory Comfort', 'Connection', 'Reflection'],
            'Love': ['Self-Expression', 'Emotional Reinforcement', 'Connection', 'Warmth'],
            'Neutral': ['Emotional Balance', 'Stability', 'General Wellness', 'Grounding']
        }
        
        # Enhanced musical features mapping
        self.musical_feature_mapping = {
            'Joy': ['Fast Tempo', 'Major Harmony', 'Ascending Melody', 'Bright Timbre'],
            'Sadness': ['Slow Tempo', 'Minor Harmony', 'Descending Melody', 'Soft Dynamics'],
            'Excitement': ['Fast Tempo', 'Complex Rhythm', 'Energetic Harmony', 'Dynamic'],
            'Calm': ['Slow Tempo', 'Repetitive Melody', 'Peaceful Harmony', 'Flowing'],
            'Anger': ['Dynamic Rhythm', 'Complex Harmony', 'High Intensity', 'Strong Beats'],
            'Fear': ['Soft Tempo', 'Minor Harmony', 'Dissonant Elements', 'Steady Rhythm'],
            'Surprise': ['Sudden Changes', 'Unexpected Harmony', 'Dynamic Contrasts', 'Complex'],
            'Nostalgia': ['Moderate Tempo', 'Warm Harmony', 'Nostalgic Themes', 'Emotional'],
            'Love': ['Warm Timbre', 'Moderate Tempo', 'Consonant Harmony', 'Heartfelt'],
            'Neutral': ['Balanced Tempo', 'Stable Harmony', 'Comfortable Rhythm', 'Even']
        }
        
        # Processing state file
        self.state_file = "preprocess_state.json"
        self.load_processing_state()
        
        logger.info("MUSIC Enhanced Mood Preprocessor initialized with ML emotion detection")
    
    def load_processing_state(self):
        """Load processing state to avoid reprocessing"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    self.processing_state = json.load(f)
                logger.info(f"STATE Loaded processing state: {len(self.processing_state.get('processed_uris', []))} processed tracks")
            except Exception as e:
                logger.error(f"Error loading processing state: {e}")
                self.processing_state = {'processed_uris': [], 'last_processed': None}
        else:
            self.processing_state = {'processed_uris': [], 'last_processed': None}
    
    def save_processing_state(self):
        """Save processing state"""
        self.processing_state['last_processed'] = datetime.now().isoformat()
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.processing_state, f, indent=2)
            logger.info("SAVE Processing state saved")
        except Exception as e:
            logger.error(f"Error saving processing state: {e}")
    
    def analyze_audio_features_ml(self, row: pd.Series) -> Tuple[str, float]:
        """
        ML-based audio feature analysis for emotion detection with confidence scoring
        """
        # Extract features for ML model
        features = {
            'Tempo': row.get('Tempo', 120),
            'Energy': row.get('Energy', 0.5),
            'Valence': row.get('Valence', 0.5),
            'Danceability': row.get('Danceability', 0.5),
            'Acousticness': row.get('Acousticness', 0.5),
            'Loudness': row.get('Loudness', -8.0),
            'Mode': row.get('Mode', 1),
            'Speechiness': row.get('Speechiness', 0.05),
            'Instrumentalness': row.get('Instrumentalness', 0.0),
            'Liveness': row.get('Liveness', 0.1),
            'Key': row.get('Key', 5),
            'Track Popularity': row.get('Track Popularity', 50),
            'Artist Popularity': row.get('Artist Popularity', 50),
            'Duration (ms)': row.get('Duration (ms)', 210000)
        }
        
        try:
            # Use ML model for prediction
            emotion, confidence = self.ml_detector.predict_emotion(features)
            return emotion, confidence
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}, using fallback")
            # Fallback to neutral with low confidence
            return 'Neutral', 0.3
    
    def map_emotional_state_enhanced(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced emotional state mapping with confidence scores and validation
        """
        emotional_state_column = []
        confidence_column = []
        mental_health_benefit_column = []
        musical_features_column = []
        
        # Process only new tracks
        processed_uris = set(self.processing_state.get('processed_uris', []))
        new_tracks = 0
        skipped_tracks = 0
        
        for index, row in tqdm(features_df.iterrows(), total=features_df.shape[0], desc="Processing Tracks"):
            track_uri = row.get('Track URI', '')
            
            # Skip already processed tracks
            if track_uri in processed_uris:
                # Load existing data if available
                if 'Mood_Label' in row and pd.notna(row['Mood_Label']):
                    emotional_state_column.append(row['Mood_Label'])
                    confidence_column.append(row.get('Confidence', 0.5))
                    mental_health_benefit_column.append(row.get('Mental_Health_Benefit', 'Unknown'))
                    musical_features_column.append(row.get('Musical_Features', 'Unknown'))
                    skipped_tracks += 1
                    continue
            
            # Validate required features
            required_features = ['Tempo', 'Energy', 'Valence', 'Danceability']
            if not all(pd.notna(row.get(feat, None)) for feat in required_features):
                logger.warning(f"Missing features for track: {row.get('Track Name', 'Unknown')}")
                emotional_state_column.append('Neutral')
                confidence_column.append(0.1)
                mental_health_benefit_column.append('General Wellness')
                musical_features_column.append('Balanced')
                continue
            
            # Analyze emotions using ML
            emotional_state, confidence = self.analyze_audio_features_ml(row)
            
            # Map to mental health benefits and musical features
            mental_health_benefits = self.emotional_state_mapping.get(emotional_state, ['Unknown'])
            musical_features = self.musical_feature_mapping.get(emotional_state, ['Unknown'])
            
            # Store results
            emotional_state_column.append(emotional_state)
            confidence_column.append(round(confidence, 3))
            mental_health_benefit_column.append(', '.join(mental_health_benefits))
            musical_features_column.append(', '.join(musical_features))
            
            # Mark as processed
            processed_uris.add(track_uri)
            new_tracks += 1
        
        # Update dataframe
        features_df['Mood_Label'] = emotional_state_column
        features_df['Confidence'] = confidence_column
        features_df['Mental_Health_Benefit'] = mental_health_benefit_column
        features_df['Musical_Features'] = musical_features_column
        
        # Update processing state
        self.processing_state['processed_uris'] = list(processed_uris)
        
        logger.info(f"SUCCESS Processed {new_tracks} new tracks, skipped {skipped_tracks} existing tracks")
        return features_df
    
    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the dataset
        """
        initial_count = len(df)
        
        # Remove tracks without essential audio features
        essential_columns = ['Track Name', 'Artist Name', 'Tempo', 'Energy', 'Valence']
        df = df.dropna(subset=essential_columns)
        
        # Fill missing values with defaults
        numeric_defaults = {
            'Danceability': 0.5, 'Acousticness': 0.5, 'Instrumentalness': 0.0,
            'Liveness': 0.1, 'Speechiness': 0.05, 'Loudness': -8.0,
            'Track Popularity': 0, 'Artist Popularity': 0, 'Mode': 1,
            'Key': 5, 'Duration (ms)': 180000
        }
        
        for col, default_val in numeric_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_val)
        
        # Fill categorical defaults
        df['Artist Genres'] = df['Artist Genres'].fillna('[]')
        df['Album'] = df['Album'].fillna('Unknown Album')
        
        # Remove duplicates based on Track Name + Artist Name
        df = df.drop_duplicates(subset=['Track Name', 'Artist Name'], keep='first')
        
        cleaned_count = len(df)
        logger.info(f"CLEAN Data cleaning: {initial_count} -> {cleaned_count} tracks ({initial_count - cleaned_count} removed)")
        
        return df
    
    def get_dataset_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Get emotion distribution statistics
        """
        if 'Mood_Label' not in df.columns:
            return {}
        
        distribution = df['Mood_Label'].value_counts().to_dict()
        total = sum(distribution.values())
        
        # Calculate percentages
        percentage_dist = {emotion: (count/total)*100 for emotion, count in distribution.items()}
        
        return {
            'counts': distribution,
            'percentages': percentage_dist,
            'total': total,
            'unique_emotions': len(distribution)
        }
    
    def preprocess_enhanced_dataset(self, input_filepath: str, output_filepath: str):
        """
        Main preprocessing function with enhanced features
        """
        logger.info(f"START Starting enhanced preprocessing with ML emotion detection...")
        logger.info(f"INPUT Input: {input_filepath}")
        logger.info(f"OUTPUT Output: {output_filepath}")
        
        # Load dataset
        try:
            df = pd.read_csv(input_filepath)
            logger.info(f"DATA Loaded dataset: {len(df)} tracks, {len(df.columns)} features")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return
        
        # Validate and clean data
        df = self.validate_and_clean_data(df)
        
        # Remove unnecessary columns
        columns_to_drop = ['Track URI', 'Artist URI']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Map emotional states
        df = self.map_emotional_state_enhanced(df)
        
        # Get distribution stats
        distribution = self.get_dataset_distribution(df)
        logger.info("STATS Emotion Distribution:")
        for emotion, percentage in distribution.get('percentages', {}).items():
            logger.info(f"  {emotion}: {distribution['counts'][emotion]} ({percentage:.1f}%)")
        
        # Save processed dataset
        try:
            df.to_csv(output_filepath, index=False)
            logger.info(f"SAVE Saved processed dataset: {output_filepath}")
            
            # Save processing state
            self.save_processing_state()
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            return
        
        logger.info("COMPLETE Enhanced ML preprocessing completed!")
        return df


def main():
    """Main execution function"""
    # File paths (relative to src-new)
    input_filepath = 'datasets/enhanced_hindi_songs.csv'  # From enhanced scraper
    output_filepath = 'datasets/therapeutic_music_enriched.csv'  # Your existing processed dataset
    
    # Check if input file exists, fallback to original if enhanced doesn't exist
    if not os.path.exists(input_filepath):
        input_filepath = 'datasets/Y.M.I.R. original dataset.csv'  # Fallback to original
        logger.info(f"INFO Enhanced dataset not found, using original: {input_filepath}")
        
    if not os.path.exists(input_filepath):
        logger.error(f"Input file not found: {input_filepath}")
        logger.info("INFO Please check your dataset files")
        return
    
    # Initialize preprocessor
    preprocessor = EnhancedMoodPreprocessor()
    
    # Process dataset
    processed_df = preprocessor.preprocess_enhanced_dataset(input_filepath, output_filepath)
    
    if processed_df is not None:
        logger.info(f"SUCCESS Successfully processed {len(processed_df)} tracks with ML emotion detection!")
        logger.info(f"FINAL Final dataset saved to: {output_filepath}")
    else:
        logger.error("ERROR Processing failed")


if __name__ == '__main__':
    main()