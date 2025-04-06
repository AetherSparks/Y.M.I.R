# **EmotiMuse: AI-Powered Emotion Detection & Music Recommendation System** 🎵🤖

![EmotiMuse Banner](https://github.com/user-attachments/assets/3f4b822c-71db-4360-a202-a64f098c1137)

## **Overview**  
EmotiMuse is a cutting-edge AI system that personalizes your music experience through emotional intelligence:

- **Multi-Channel Emotion Detection** - Analyzes both facial expressions and text interactions 🔍
- **Intelligent Music Matching** - Recommends music tailored to your emotional state 🎧
- **Seamless User Experience** - Real-time analysis and recommendations through an intuitive interface 💻

## **Key Features**

| Feature | Description |
|---------|-------------|
| **🎭 Multimodal Emotion Detection** | Combines DeepFace visual analysis with natural language processing for comprehensive emotional assessment |
| **💬 Interactive Emotion Chatbot** | Engages users in conversation to gather emotional context beyond facial expressions |
| **📊 Emotion Fusion Algorithm** | Integrates visual and text-based emotional signals for enhanced accuracy |
| **🎶 Personalized Music Recommendations** | Content-based recommendation engine tailored to emotional states |
| **🔄 Real-Time Processing** | Continuous emotion monitoring and dynamic recommendation updates |
| **🌐 Responsive Web Interface** | Flask-powered application accessible across devices |

## **Technical Architecture**

### Project Structure
```
emotion-music-recommendation/
├── datasets/
│   ├── therapeutic_music_enriched.csv        # Enhanced music dataset with emotional mappings
│   ├── Y.M.I.R. original dataset.csv         # Original YMIR dataset
│   └── imagesofdataset/                      # Data visualization images
│       ├── Figure_1.png
│       └── ...
├── src/
│   ├── chatbot.py                            # NLP-based emotion detection
│   ├── dataset.py                            # Dataset management utilities
│   ├── fer1.py                               # Facial emotion recognition
│   ├── modules.py                            # Core system components
│   ├── recommend.py                          # Music recommendation engine
│   └── train_music_recommendation.py         # Model training script
├── static/
│   └── styles.css                            # Application styling
├── templates/                                # Web interface templates
│   ├── about.html
│   ├── contact.html
│   └── ...
└── Website-Images/                           # UI screenshots and marketing assets
    ├── Homepage.jpg
    └── ...
```

## **Installation Guide**

### Prerequisites
- Python 3.8+
- Webcam access
- Internet connection

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/emotion-music-recommendation.git
   cd emotion-music-recommendation
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application**
   ```bash
   python app.py
   ```

5. **Access the web interface**
   - Open your browser and navigate to `http://127.0.0.1:5000/`
   - Grant camera permissions when prompted

## **How It Works**

### The EmotiMuse Pipeline
1. **Emotion Capture** - Your emotions are captured through:
   - Real-time facial analysis via webcam
   - Conversational text analysis through the chatbot interface

2. **Emotion Processing** - Our fusion algorithm combines these signals to determine your emotional state with greater accuracy than single-mode systems

3. **Music Matching** - The recommendation engine searches our curated music database to find tracks that complement or enhance your current emotional state

4. **Continuous Adaptation** - The system continuously monitors emotional changes, updating recommendations as your mood evolves

## **Development Roadmap**

### Current Development Status
- ✅ Core emotion detection system
- ✅ Basic recommendation engine
- ✅ Web interface prototype
- ✅ Initial dataset integration

### Upcoming Enhancements
- 🔄 **UI/UX Improvements** - Enhanced design and responsive layouts
- 🔄 **Advanced Recommendation Algorithm** - More nuanced emotion-music matching
- 🔄 **Camera Controls** - Improved start/stop functionality and permissions handling
- 🔄 **Favorites System** - Save and organize recommended music
- 🔄 **Extended Button Functionality** - Complete implementation of all interface controls

> **Note:** This project is under active development. Features and interfaces may change as we refine the system.

## **Contributing**

We welcome contributions to improve EmotiMuse! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## **Team**

- **Pallav Sharma** - [GitHub Profile](https://github.com/pallav)

## **Acknowledgments**

- The DeepFace library for facial emotion recognition
- OpenAI for NLP technologies
- The Flask development community
- Contributors to the YMIR dataset

## **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

⚠️ **Development Status:** EmotiMuse is currently in beta. Some features may be incomplete or subject to change.