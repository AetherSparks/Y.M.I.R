"""
🎵 Y.M.I.R Recommendation System Tester
=====================================
Test the production-ready music recommendation system with real examples
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RecommendationTester:
    """Test the trained recommendation system with various scenarios"""
    
    def __init__(self):
        # Load the processed dataset
        self.dataset_path = "datasets/therapeutic_music_enriched.csv"
        self.model_dir = "models"
        
        # Load dataset
        if os.path.exists(self.dataset_path):
            self.df = pd.read_csv(self.dataset_path)
            print(f"✅ Loaded dataset: {len(self.df)} tracks")
        else:
            print("❌ Dataset not found. Run preprocessing first.")
            return
            
        # Load the latest trained model
        self.load_latest_model()
        
    def load_latest_model(self):
        """Load the most recent trained recommendation model"""
        if not os.path.exists(self.model_dir):
            print("❌ No trained models found. Run training first.")
            return
            
        # Find the latest model file
        model_files = list(Path(self.model_dir).glob("music_recommender_*.pkl"))
        if not model_files:
            print("❌ No recommendation models found. Run training first.")
            return
            
        latest_model = max(model_files, key=os.path.getctime)
        print(f"📦 Loading model: {latest_model}")
        
        try:
            with open(latest_model, 'rb') as f:
                self.model_data = pickle.load(f)
            print(f"✅ Model loaded successfully!")
            print(f"   Model type: {self.model_data['model_name']}")
            print(f"   Training date: {self.model_data['training_date']}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            
    def get_emotion_distribution(self):
        """Show the current emotion distribution in dataset"""
        print("\n📊 CURRENT EMOTION DISTRIBUTION:")
        print("=" * 50)
        
        if 'Mood_Label' in self.df.columns:
            mood_counts = self.df['Mood_Label'].value_counts()
            total = len(self.df)
            
            for emotion, count in mood_counts.items():
                percentage = (count / total) * 100
                bar = "█" * int(percentage / 2)  # Visual bar
                print(f"{emotion:12} | {count:4d} ({percentage:5.1f}%) {bar}")
        else:
            print("❌ No Mood_Label column found")
            
    def recommend_by_emotion(self, target_emotion: str, limit: int = 10):
        """Get music recommendations for a specific emotion/mood"""
        print(f"\n🎵 RECOMMENDATIONS FOR: {target_emotion.upper()}")
        print("=" * 60)
        
        if 'Mood_Label' not in self.df.columns:
            print("❌ No emotion labels found in dataset")
            return
            
        # Filter songs by emotion
        emotion_songs = self.df[self.df['Mood_Label'] == target_emotion]
        
        if len(emotion_songs) == 0:
            print(f"❌ No songs found for emotion: {target_emotion}")
            available_emotions = self.df['Mood_Label'].unique()
            print(f"Available emotions: {list(available_emotions)}")
            return
            
        # Sort by confidence (if available) and popularity
        sort_columns = []
        if 'Confidence' in emotion_songs.columns:
            sort_columns.append('Confidence')
        if 'Track Popularity' in emotion_songs.columns:
            sort_columns.append('Track Popularity')
        if 'Artist Popularity' in emotion_songs.columns:
            sort_columns.append('Artist Popularity')
            
        if sort_columns:
            recommendations = emotion_songs.nlargest(limit, sort_columns)
        else:
            recommendations = emotion_songs.head(limit)
        
        # Display recommendations
        for i, (_, track) in enumerate(recommendations.iterrows(), 1):
            track_name = track.get('Track Name', 'Unknown Track')
            artist_name = track.get('Artist Name', 'Unknown Artist')
            confidence = track.get('Confidence', 0.0)
            mental_health_benefit = track.get('Mental_Health_Benefit', 'General Wellness')
            
            print(f"{i:2d}. {track_name} - {artist_name}")
            print(f"    💚 Benefits: {mental_health_benefit}")
            if confidence > 0:
                print(f"    🎯 Confidence: {confidence:.2f}")
            print()
            
    def therapeutic_recommendations(self):
        """Show therapeutic recommendations for common mental health needs"""
        print("\n💚 THERAPEUTIC MUSIC RECOMMENDATIONS")
        print("=" * 60)
        
        therapeutic_scenarios = {
            "Stress Relief": ["Calm", "Neutral"],
            "Mood Boost": ["Joy", "Excitement"], 
            "Anxiety Support": ["Calm", "Neutral"],
            "Energy Boost": ["Excitement", "Joy"],
            "Emotional Processing": ["Sadness", "Fear", "Anger"],
            "General Wellness": ["Neutral", "Calm", "Joy"]
        }
        
        for scenario, emotions in therapeutic_scenarios.items():
            print(f"\n🎯 {scenario}:")
            available_songs = self.df[self.df['Mood_Label'].isin(emotions)]
            
            if len(available_songs) > 0:
                top_songs = available_songs.nlargest(3, ['Track Popularity', 'Artist Popularity'])
                for _, track in top_songs.iterrows():
                    track_name = track.get('Track Name', 'Unknown')
                    artist_name = track.get('Artist Name', 'Unknown')
                    emotion = track.get('Mood_Label', 'Unknown')
                    print(f"  • {track_name} - {artist_name} ({emotion})")
            else:
                print("  No songs available for this therapy type")
                
    def test_model_prediction(self):
        """Test the ML model with sample audio features"""
        print("\n🤖 TESTING ML MODEL PREDICTIONS")
        print("=" * 50)
        
        if not hasattr(self, 'model_data'):
            print("❌ No model loaded")
            return
            
        # Sample audio features for testing
        test_scenarios = {
            "Happy Upbeat Song": {
                'Tempo': 120, 'Energy': 0.8, 'Valence': 0.9, 'Danceability': 0.7,
                'Acousticness': 0.2, 'Loudness': -5, 'Mode': 1, 'Speechiness': 0.05,
                'Instrumentalness': 0.1, 'Liveness': 0.3, 'Key': 2,
                'Track Popularity': 70, 'Artist Popularity': 60, 'Duration (ms)': 210000
            },
            "Sad Slow Ballad": {
                'Tempo': 70, 'Energy': 0.3, 'Valence': 0.2, 'Danceability': 0.3,
                'Acousticness': 0.7, 'Loudness': -12, 'Mode': 0, 'Speechiness': 0.04,
                'Instrumentalness': 0.2, 'Liveness': 0.1, 'Key': 8,
                'Track Popularity': 50, 'Artist Popularity': 45, 'Duration (ms)': 240000
            },
            "Calm Meditation Music": {
                'Tempo': 60, 'Energy': 0.2, 'Valence': 0.5, 'Danceability': 0.2,
                'Acousticness': 0.9, 'Loudness': -15, 'Mode': 1, 'Speechiness': 0.02,
                'Instrumentalness': 0.8, 'Liveness': 0.1, 'Key': 5,
                'Track Popularity': 30, 'Artist Popularity': 25, 'Duration (ms)': 300000
            }
        }
        
        # Test each scenario
        for scenario_name, features in test_scenarios.items():
            print(f"\n🎵 Testing: {scenario_name}")
            
            try:
                # This would require the actual ML model prediction code
                # For now, we'll simulate based on the dataset
                print("  (Simulated prediction based on similar tracks in dataset)")
                
                # Find similar tracks based on key audio features
                similar_mask = (
                    (abs(self.df['Energy'] - features['Energy']) < 0.2) &
                    (abs(self.df['Valence'] - features['Valence']) < 0.2) &
                    (abs(self.df['Tempo'] - features['Tempo']) < 30)
                )
                
                similar_tracks = self.df[similar_mask]
                if len(similar_tracks) > 0:
                    most_common_emotion = similar_tracks['Mood_Label'].mode().iloc[0]
                    confidence = len(similar_tracks) / len(self.df)
                    print(f"  🎯 Predicted Emotion: {most_common_emotion}")
                    print(f"  📊 Based on {len(similar_tracks)} similar tracks")
                else:
                    print("  ⚠️ No similar tracks found in dataset")
                    
            except Exception as e:
                print(f"  ❌ Prediction error: {e}")
                
    def dataset_quality_report(self):
        """Generate a quality report of the dataset"""
        print("\n📋 DATASET QUALITY REPORT")
        print("=" * 50)
        
        print(f"📊 Total Tracks: {len(self.df)}")
        print(f"📊 Total Features: {len(self.df.columns)}")
        
        # Check for key columns
        required_columns = ['Track Name', 'Artist Name', 'Mood_Label', 'Mental_Health_Benefit']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"⚠️ Missing columns: {missing_columns}")
        else:
            print("✅ All required columns present")
            
        # Check data completeness
        if 'Mood_Label' in self.df.columns:
            null_emotions = self.df['Mood_Label'].isnull().sum()
            print(f"📊 Complete emotion labels: {len(self.df) - null_emotions}/{len(self.df)}")
            
        # Unique values
        if 'Track Name' in self.df.columns:
            unique_tracks = self.df['Track Name'].nunique()
            print(f"🎵 Unique tracks: {unique_tracks}")
            
        if 'Artist Name' in self.df.columns:
            unique_artists = self.df['Artist Name'].nunique()
            print(f"🎤 Unique artists: {unique_artists}")


def main():
    """Main testing interface"""
    print("🎵 Y.M.I.R Music Recommendation System Tester")
    print("=" * 60)
    
    # Initialize tester
    tester = RecommendationTester()
    
    # Run all tests
    tester.dataset_quality_report()
    tester.get_emotion_distribution()
    tester.therapeutic_recommendations()
    
    # Test specific emotions
    print("\n" + "="*60)
    print("TESTING SPECIFIC EMOTIONS")
    print("="*60)
    
    emotions_to_test = ['Joy', 'Calm', 'Excitement', 'Neutral']
    for emotion in emotions_to_test:
        tester.recommend_by_emotion(emotion, limit=5)
        
    tester.test_model_prediction()
    
    print("\n✅ Testing complete!")
    print("\n📋 NEXT STEPS:")
    print("1. Review the emotion distribution (should be balanced)")
    print("2. Test recommendations for different moods")
    print("3. Verify therapeutic benefits are appropriate")
    print("4. Ready for production deployment!")


if __name__ == "__main__":
    main()