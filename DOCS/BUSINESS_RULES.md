# Y.M.I.R. Business Rules and System Specifications

## 1. EMOTION DETECTION RULES

### 1.1 Facial Emotion Analysis
- **Rule**: Facial emotion detection must run every 20 frames to optimize performance
- **Rule**: Minimum detection confidence threshold is set to 0.5 for reliable results
- **Rule**: System must support multi-face detection (max 5 faces simultaneously)
- **Rule**: If no face is detected for 30 consecutive frames, system retains last known emotion state

### 1.2 Text-based Emotion Analysis
- **Rule**: Chat emotion analysis considers the last 5 messages for context
- **Rule**: Recent messages are weighted 1.5x higher than older messages
- **Rule**: Emotion predictions below 0.4 confidence score are filtered out
- **Rule**: Negation patterns in text automatically override standard emotion detection

### 1.3 Emotion Fusion Algorithm
- **Rule**: Final emotion state is calculated by averaging facial and text-based emotions
- **Rule**: System maintains rolling emotion history (last 5 emotion results for smoothing)
- **Rule**: Dominant emotion mapping must align with therapeutic mood categories

## 2. MUSIC RECOMMENDATION RULES

### 2.1 Content-Based Filtering
- **Rule**: Recommendations are based on current emotional state, not user history
- **Rule**: System provides maximum 10 song recommendations per session
- **Rule**: Duplicate songs (same title + artist) are automatically filtered out
- **Rule**: Fallback to "Neutral" mood category if no matching songs found

### 2.2 Audio Feature Mapping
- **Rule**: Each emotion maps to specific audio features (valence, energy, tempo, etc.)
- **Rule**: Audio features are normalized using pre-trained scaler before model input
- **Rule**: Emotion-to-mood mapping prioritizes therapeutic benefit over exact emotion match

### 2.3 Music Source Management
- **Rule**: System first checks local music cache before external downloads
- **Rule**: YouTube downloads are limited to audio-only format (MP3, 192kbps)
- **Rule**: Downloaded files must use sanitized filenames (special characters removed)
- **Rule**: Maximum 2 retry attempts for failed downloads

## 3. USER INTERACTION RULES

### 3.1 Privacy and Data Protection
- **Rule**: Facial recognition data is processed locally and not stored permanently
- **Rule**: Chat conversations are saved locally in JSON format only
- **Rule**: User consent required for webcam access
- **Rule**: No personal data is transmitted to external services without encryption

### 3.2 Session Management
- **Rule**: Emotion logging continues throughout active session
- **Rule**: Background processing updates occur every 1 second
- **Rule**: Camera resources are released when user exits the application
- **Rule**: Session data persists locally until explicitly cleared

### 3.3 User Experience Rules
- **Rule**: Video feed must maintain minimum 30 FPS for smooth experience
- **Rule**: Recommendation updates happen in real-time without page refresh
- **Rule**: System provides immediate feedback for all user actions
- **Rule**: Error messages must be user-friendly and actionable

## 4. WELLNESS FEATURE RULES

### 4.1 Meditation and Breathing Exercises
- **Rule**: Mood-based script selection uses keyword matching in user input
- **Rule**: Default meditation script provided if specific mood not recognized
- **Rule**: Breathing technique recommendations based on current emotional state
- **Rule**: Session tracking includes timestamps and emotional context

### 4.2 Goal Setting and Progress Tracking
- **Rule**: Users can check off goals only once per day
- **Rule**: Streak counter resets if goal not completed for more than 1 day
- **Rule**: Goal creation requires non-empty goal description
- **Rule**: Progress data persists across sessions in local JSON storage

### 4.3 Community Support Features
- **Rule**: Anonymous posting supported for user privacy
- **Rule**: AI responses generated for community posts using sentiment analysis
- **Rule**: Post moderation through basic content filtering
- **Rule**: Community posts limited to text format only

## 5. TECHNICAL SYSTEM RULES

### 5.1 Performance Requirements
- **Rule**: Emotion detection processing limited to prevent system overload
- **Rule**: Background threads must be daemon threads to allow clean shutdown
- **Rule**: Video processing optimized with frame skipping (every 3rd frame)
- **Rule**: Maximum memory usage monitoring for large datasets

### 5.2 Error Handling and Recovery
- **Rule**: Camera initialization failure triggers automatic retry mechanism
- **Rule**: API failures fall back to cached or default responses
- **Rule**: Missing model files prevent application startup with clear error message
- **Rule**: Database connection errors logged with timestamp and context

### 5.3 Integration Rules
- **Rule**: External API keys must be stored in environment variables
- **Rule**: Model files loaded once at application startup for efficiency
- **Rule**: Third-party library versions locked to prevent compatibility issues
- **Rule**: Graceful degradation when optional features unavailable

## 6. SECURITY AND COMPLIANCE RULES

### 6.1 Data Security
- **Rule**: All file operations include error handling and validation
- **Rule**: User input sanitized before processing or storage
- **Rule**: Email functionality requires encrypted SMTP connection
- **Rule**: Session data protected from unauthorized access

### 6.2 Operational Security
- **Rule**: Debug mode disabled in production environment
- **Rule**: Regular cleanup of temporary files and cached data
- **Rule**: Monitoring and logging for security events
- **Rule**: Resource cleanup on application termination

## 7. BUSINESS LOGIC RULES

### 7.1 Application Flow
- **Rule**: Multi-modal emotion detection runs simultaneously for comprehensive analysis
- **Rule**: Music recommendations update dynamically based on emotion changes
- **Rule**: User can save/remove favorite songs with persistent storage
- **Rule**: Wellness tools accessible regardless of current emotion state

### 7.2 Content Management
- **Rule**: Therapeutic music dataset curated for mental health benefits
- **Rule**: Meditation scripts vary by mood for personalized experience
- **Rule**: Sound therapy recommendations matched to specific emotional needs
- **Rule**: Educational content maintained for user guidance and support

---

*These business rules ensure Y.M.I.R. operates effectively while prioritizing user privacy, emotional well-being, and system reliability.*