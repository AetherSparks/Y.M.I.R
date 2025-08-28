# 🧠 Y.M.I.R Multimodal Emotion Detection & Music Recommendation System

## 📋 System Overview

This document describes the complete multimodal system that combines facial emotion recognition, text sentiment analysis, and music recommendation into a unified, real-time processing pipeline.

## 🏗️ Architecture

The system consists of four main components:

### 1. Multimodal Emotion Service (`multimodal_emotion_service.py`)
- **Purpose**: Combines facial and textual emotions using advanced fusion algorithms
- **Key Features**:
  - Unified emotion data structure (`MultimodalEmotionReading`)
  - Multiple fusion strategies (weighted, confidence-based, temporal, adaptive)
  - Session-based emotion tracking
  - Real-time processing capabilities

### 2. Enhanced Music Recommendation Service (`multimodal_music_recommendation_service.py`)
- **Purpose**: Provides intelligent music recommendations based on fused emotions
- **Key Features**:
  - Multiple recommendation strategies (therapeutic, mood-matching, mood-regulation, adaptive)
  - Audio feature-based matching
  - Confidence-weighted scoring
  - Session history tracking

### 3. Real-time Processing Pipeline (`realtime_multimodal_pipeline.py`)
- **Purpose**: Orchestrates the entire system with asynchronous processing
- **Key Features**:
  - Asynchronous message processing
  - Multi-worker architecture
  - Auto-fusion and auto-recommendation
  - Performance monitoring and analytics

### 4. Testing Framework (`tests/test_multimodal_system.py`)
- **Purpose**: Comprehensive testing suite for all components
- **Key Features**:
  - Unit tests for all major functions
  - Integration tests for end-to-end workflows
  - Performance benchmarking
  - Async testing support

## 🚀 Quick Start Guide

### Step 1: Import the System

```python
import asyncio
from services.realtime_multimodal_pipeline import (
    get_realtime_pipeline, start_pipeline, create_processing_session,
    process_facial_input, process_text_input, get_music_recommendations
)
```

### Step 2: Start the Pipeline

```python
async def main():
    # Start the pipeline with 3 worker threads
    await start_pipeline(num_workers=3)
    
    # Create a processing session
    session_id = create_processing_session()
    print(f"Created session: {session_id}")
```

### Step 3: Process Emotions

```python
# Process facial emotion data
facial_data = {
    'emotions': {'joy': 0.8, 'neutral': 0.2},
    'confidence': 0.8,
    'quality_score': 0.9
}
await process_facial_input(session_id, facial_data)

# Process text emotion data
await process_text_input(session_id, "I'm feeling great today!", "joy", 0.7)
```

### Step 4: Get Music Recommendations

```python
# Get music recommendations based on fused emotions
recommendations = await get_music_recommendations(session_id, strategy='adaptive', num_tracks=10)
print(f"Recommended {len(recommendations)} tracks")
```

## 🔧 Detailed Integration Guide

### Integrating with Existing Facial Emotion Recognition

The system expects facial emotion data in this format:

```python
facial_emotion_data = {
    'emotions': {
        'joy': 0.8,
        'sadness': 0.1,
        'anger': 0.05,
        'fear': 0.03,
        'surprise': 0.02
    },
    'confidence': 0.85,
    'quality_score': 0.9,  # Face detection quality
    'timestamp': datetime.now()
}
```

To integrate with `fer_enhanced_v2.py`:

```python
from enhancements.src_new.face.fer_enhanced_v2 import FacialEmotionDetector
from services.realtime_multimodal_pipeline import process_facial_input

# Initialize facial detector
facial_detector = FacialEmotionDetector()

# Process frame and get emotion reading
emotion_reading = facial_detector.detect_emotions(frame)

# Convert to pipeline format
facial_data = {
    'emotions': emotion_reading.emotions,
    'confidence': emotion_reading.confidence,
    'quality_score': emotion_reading.quality_score
}

# Send to multimodal pipeline
await process_facial_input(session_id, facial_data)
```

### Integrating with Existing Text Sentiment Analysis

The system expects text emotion data in this format:

```python
text_emotion_data = {
    'text': "User's message or chat input",
    'emotion': 'joy',  # Detected emotion
    'confidence': 0.75  # Confidence in detection
}
```

To integrate with `chatbot_production_ready.py`:

```python
from enhancements.src_new.textual.chatbot_production_ready import EmotionAnalyzer
from services.realtime_multimodal_pipeline import process_text_input

# Initialize emotion analyzer
emotion_analyzer = EmotionAnalyzer()

# Analyze text emotion
user_message = "I love this music recommendation system!"
emotion_result = emotion_analyzer.analyze_emotion(user_message)

# Send to multimodal pipeline
await process_text_input(
    session_id, 
    user_message, 
    emotion_result.emotion,
    emotion_result.confidence
)
```

### Accessing Fused Emotions

```python
from services.multimodal_emotion_service import get_fused_emotion_for_music

# Get current fused emotion state
emotion_state = get_fused_emotion_for_music(session_id)

if emotion_state:
    print(f"Fused Emotion: {emotion_state['emotion']}")
    print(f"Confidence: {emotion_state['confidence']}")
    print(f"Sources: {emotion_state['context']}")
```

## 🎵 Music Recommendation Strategies

### 1. Therapeutic Strategy
Selects music to support emotional healing and mental health benefits.

```python
recommendations = await get_music_recommendations(
    session_id, 
    strategy='therapeutic',
    num_tracks=10
)
```

**Use Case**: When user shows persistent negative emotions (sadness, anger, anxiety)

### 2. Mood Matching Strategy  
Selects music that matches the current emotional state.

```python
recommendations = await get_music_recommendations(
    session_id,
    strategy='mood_matching', 
    num_tracks=10
)
```

**Use Case**: When user wants music that reflects their current mood

### 3. Mood Regulation Strategy
Selects music to guide user toward a more positive emotional state.

```python
recommendations = await get_music_recommendations(
    session_id,
    strategy='mood_regulation',
    num_tracks=10
)
```

**Use Case**: When user needs emotional support to improve their mood

### 4. Adaptive Strategy (Recommended)
Intelligently combines strategies based on context, confidence, and history.

```python
recommendations = await get_music_recommendations(
    session_id,
    strategy='adaptive',  # Default
    num_tracks=10
)
```

**Use Case**: General usage - system adapts based on situation

## 📊 Emotion Fusion Algorithms

The system provides multiple fusion algorithms to combine facial and textual emotions:

### 1. Weighted Average Fusion
Simple weighted combination of emotion scores.
- **Best for**: Balanced scenarios with similar confidence levels

### 2. Confidence-Based Fusion  
Weights emotions based on detection confidence levels.
- **Best for**: When one modality is significantly more reliable

### 3. Temporal Weighted Fusion
Considers recency of emotion detections with temporal decay.
- **Best for**: Real-time scenarios where timing matters

### 4. Adaptive Fusion (Default)
Dynamically adjusts fusion strategy based on context and reliability.
- **Best for**: General usage - adapts to different scenarios

## 🔧 Configuration Options

### Pipeline Configuration

```python
# Create session with specific configuration
session_config = {
    'auto_fusion_enabled': True,
    'auto_fusion_interval': 10,  # seconds
    'auto_recommendation_enabled': True,
    'preferred_strategy': 'adaptive',
    'emotion_history_limit': 100
}

session_id = create_processing_session(session_config)
```

### Emotion Processor Configuration

```python
from services.multimodal_emotion_service import MultimodalEmotionProcessor

# Create processor with custom settings
processor = MultimodalEmotionProcessor(
    session_duration_minutes=30,  # Session timeout
)
```

### Music Engine Configuration

```python
from services.multimodal_music_recommendation_service import MultimodalMusicRecommendationEngine

# Create engine with custom dataset
engine = MultimodalMusicRecommendationEngine(
    dataset_path="path/to/custom/music/dataset.csv",
    model_dir="path/to/trained/models"
)
```

## 📈 Monitoring and Analytics

### Pipeline Statistics

```python
pipeline = get_realtime_pipeline()
stats = pipeline.get_stats()

print(f"Total messages processed: {stats['total_messages_processed']}")
print(f"Active sessions: {stats['active_sessions']}")
print(f"Average processing time: {stats['average_processing_time_ms']:.2f} ms")
```

### Session Analytics

```python
# Get detailed session information
session_info = pipeline.get_session_info(session_id)
print(f"Session created: {session_info['created']}")
print(f"Last activity: {session_info['last_activity']}")
print(f"Current emotion state: {session_info['emotion_state']}")
```

### Recommendation Analytics

```python
from services.multimodal_music_recommendation_service import get_music_recommendation_engine

engine = get_music_recommendation_engine()
analytics = engine.get_recommendation_analytics(session_id)

print(f"Total recommendations: {analytics['total_recommendations']}")
print(f"Emotion distribution: {analytics['emotion_distribution']}")
print(f"Strategy usage: {analytics['strategy_usage']}")
```

## 🧪 Testing the System

### Run Comprehensive Tests

```bash
cd tests
python test_multimodal_system.py
```

### Custom Testing

```python
import asyncio
from tests.test_multimodal_system import run_comprehensive_tests, run_async_integration_tests

# Run all tests
success = run_comprehensive_tests()

# Run async tests
asyncio.run(run_async_integration_tests())
```

## 🚨 Error Handling

### Common Issues and Solutions

#### 1. Import Errors
**Issue**: Cannot import emotion detection modules
**Solution**: Ensure all dependencies are installed and paths are correct

```python
try:
    from enhancements.src_new.face.fer_enhanced_v2 import FacialEmotionDetector
except ImportError as e:
    logging.warning(f"Facial emotion detection not available: {e}")
    # Implement fallback or mock
```

#### 2. No Emotion Data
**Issue**: No emotions available for fusion
**Solution**: Check that both modalities are providing data

```python
emotion_state = get_fused_emotion_for_music(session_id)
if not emotion_state:
    # Fallback to default emotion or request more input
    emotion_state = {'emotion': 'neutral', 'confidence': 0.5}
```

#### 3. Model Loading Errors  
**Issue**: Cannot load trained ML models
**Solution**: Ensure models exist and are properly formatted

```python
try:
    engine = MultimodalMusicRecommendationEngine()
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    # Use rule-based recommendations as fallback
```

## 📚 API Reference

### Core Functions

#### `create_processing_session(config=None) -> str`
Creates a new multimodal processing session.

#### `process_facial_input(session_id, emotion_data) -> str`
Processes facial emotion data through the pipeline.

#### `process_text_input(session_id, text, emotion=None, confidence=None) -> str`  
Processes text emotion data through the pipeline.

#### `get_music_recommendations(session_id, strategy='adaptive', num_tracks=10) -> Dict`
Gets music recommendations based on current emotional state.

### Data Structures

#### `MultimodalEmotionReading`
Unified structure for storing multimodal emotion data.

#### `MusicRecommendation`
Individual music recommendation with metadata.

#### `RecommendationSet`
Set of recommendations for a specific emotional state.

## 🔄 Integration with Flask App

To integrate with your main Flask application:

```python
from flask import Flask, request, jsonify
from services.realtime_multimodal_pipeline import *
import asyncio

app = Flask(__name__)

# Global pipeline instance
pipeline = None

@app.before_first_request
async def initialize_pipeline():
    global pipeline
    await start_pipeline(num_workers=3)
    pipeline = get_realtime_pipeline()

@app.route('/api/emotion/facial', methods=['POST'])
async def process_facial_emotion_endpoint():
    data = request.json
    session_id = data.get('session_id')
    emotion_data = data.get('emotion_data')
    
    if not session_id:
        session_id = create_processing_session()
    
    message_id = await process_facial_input(session_id, emotion_data)
    
    return jsonify({
        'session_id': session_id,
        'message_id': message_id,
        'status': 'processed'
    })

@app.route('/api/emotion/text', methods=['POST'])
async def process_text_emotion_endpoint():
    data = request.json
    session_id = data.get('session_id')
    text = data.get('text')
    
    if not session_id:
        session_id = create_processing_session()
    
    message_id = await process_text_input(session_id, text)
    
    return jsonify({
        'session_id': session_id,
        'message_id': message_id,
        'status': 'processed'
    })

@app.route('/api/recommendations/<session_id>', methods=['GET'])
async def get_recommendations_endpoint(session_id):
    strategy = request.args.get('strategy', 'adaptive')
    num_tracks = int(request.args.get('num_tracks', 10))
    
    recommendations = await get_music_recommendations(session_id, strategy, num_tracks)
    
    return jsonify(recommendations)
```

## 🎯 Best Practices

1. **Session Management**: Always create sessions for user interactions and clean up when done
2. **Error Handling**: Implement robust error handling for all API calls
3. **Performance**: Use appropriate number of workers based on expected load
4. **Monitoring**: Regularly check pipeline statistics and performance metrics
5. **Testing**: Run comprehensive tests before deployment
6. **Fallbacks**: Always have fallback strategies when emotions cannot be detected

## 📞 Support and Troubleshooting

For issues or questions:

1. Check the test framework for examples
2. Review error logs for detailed information
3. Ensure all dependencies are properly installed
4. Verify model files are present and accessible
5. Test individual components before full integration

## 🔮 Future Enhancements

Planned improvements:
- WebSocket support for real-time updates
- Advanced ML models for emotion fusion
- Expanded music recommendation algorithms  
- Voice emotion analysis integration
- Biometric data fusion (heart rate, etc.)
- Cloud deployment configurations

---

This multimodal system represents a significant advancement in emotion-aware music recommendation, combining state-of-the-art emotion detection with intelligent fusion algorithms and personalized recommendation strategies.