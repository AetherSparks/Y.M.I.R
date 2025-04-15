# Y.M.I.R: Yielding Melodies for Internal Restoration 🎵🧠  
*A Mood-Based AI Emotion Detection & Music Recommendation System*

![Y.M.I.R Banner](https://github.com/pallav110/DTI-AND-AI-PROJECT/blob/main/YMIR%20thumbnail.jpg?raw=true)
<!-- Replace with your actual image path once added -->

---

## 🎥 **Watch the Demo Video**
[![Watch on YouTube](https://img.shields.io/badge/Watch%20Demo%20Video-Youtube-red?logo=youtube)](https://your-youtube-link.com)  
<!-- Replace with actual YouTube video link -->

---

## **Overview**  
**Y.M.I.R** is a cutting-edge AI-powered system that personalizes your music and wellness experience using emotional intelligence:

- **Multi-Channel Emotion Detection** – Facial expression + text-based interaction 🔍  
- **Intelligent Music Matching** – Personalized recommendations based on your current mood 🎷  
- **Wellness-First Experience** – Chat support, daily motivation, and real-time emotional assistance 💬💡  

---

## **Key Features**

| Feature | Description |
|---------|-------------|
| **🌝 Multimodal Emotion Detection** | Combines DeepFace visual analysis with natural language processing for comprehensive emotional assessment |
| **💬 Interactive Emotion Chatbot** | Engages users in conversation to gather emotional context beyond facial expressions |
| **📊 Emotion Fusion Algorithm** | Integrates visual and text-based emotional signals for enhanced accuracy |
| **🎶 Personalized Music Recommendations** | Content-based recommendation engine tailored to emotional states |
| **🔀 Real-Time Processing** | Continuous emotion monitoring and dynamic recommendation updates |
| **🌐 Responsive Web Interface** | Flask-powered application accessible across devices |

---

## **Technical Architecture**

### 📁 Project Structure
```
Y.M.I.R/
├── datasets/
│   ├── therapeutic_music_enriched.csv
│   ├── Y.M.I.R. original dataset.csv
│   └── imagesofdataset/
│       ├── Figure_1.png
├── src/
│   ├── chatbot.py
│   ├── dataset.py
│   ├── fer1.py
│   ├── modules.py
│   ├── recommend.py
│   └── train_music_recommendation.py
├── static/
│   └── styles.css
├── templates/
│   ├── about.html
│   ├── contact.html
├── Website-Images/
│   ├── Homepage.jpg
│   └── ...
```

---

## 🚀 **Installation Guide**

### Prerequisites
- Python 3.8+
- Webcam access
- Internet connection

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Y.M.I.R.git
   cd Y.M.I.R
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

5. **Access the interface**
   Open your browser at `http://127.0.0.1:5000/` and grant webcam permissions.

---

## 🧠 **How It Works**

### The Y.M.I.R Emotional Pipeline

1. **Emotion Capture**  
   - Facial recognition using webcam (DeepFace)  
   - Chatbot-based emotional interpretation (NLP)

2. **Signal Fusion**  
   - Combined analysis for deeper mood recognition

3. **Music Matching**  
   - Emotion-aware recommendation engine suggests therapeutic or uplifting tracks

4. **Continuous Support**  
   - Recommendations evolve with your emotional state

---

## 🛠️ **Development Roadmap**

### ✅ Current Progress
- Emotion detection system (visual & textual)
- Music recommendation engine
- Web UI prototype
- Dataset integration (YMIR & therapeutic music)

### 🔧 In Progress
- Improved UI & camera control
- Favorites system & user personalization
- Advanced emotion-music mapping
- Button functionality & chatbot refinement

---

## 💡 **Contributing**

We’d love your input!

1. Fork the repo  
2. Create a branch (`git checkout -b feature/awesome-idea`)  
3. Commit changes (`git commit -m 'Add cool stuff'`)  
4. Push and submit a Pull Request

---

## 👨‍💻 **Team**

- **Abhiraj Ghose** – E23CSEU0014
- [Abhiraj's GitHub Profile](https://github.com/AetherSparks) 
- **Pallav Sharma** – E23CSEU0022  
- [Pallav's GitHub Profile](https://github.com/pallav)

---

## 🙌 **Acknowledgments**

- [DeepFace](https://github.com/serengil/deepface) – Facial emotion detection  
- OpenAI – Natural Language Processing  
- Flask & Python Community  
- Contributors to the YMIR dataset  

---

## 📄 **License**

This project is licensed under the MIT License. See `LICENSE` for details.

---

⚠️ **Note:** Y.M.I.R is currently in beta. Expect ongoing changes and new features as development continues.

