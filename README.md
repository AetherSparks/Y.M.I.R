
---

# **Emotion-Based AI Music Recommendation System 🎵🤖**  

### **Overview**  
This is an AI-powered **Emotion Detection and Music Recommendation System** that:  
✅ Detects facial emotions using a webcam (DeepFace + multimodal analysis + Text based models) 📷  
✅ Analyzes text-based emotions via chatbot interaction 💬  
✅ Averages both detected emotions for better accuracy 📊  
✅ Suggests music based on the user's final emotional state 🎶  

### **Project Structure**  

```
│── Your_Project
│── datasets/
│   ├── therapeutic_music_enriched.csv
│   ├── Y.M.I.R. original dataset.csv
│   │── imagesofdataset/
│   │   ├── Figure_1.png
│   │   ├── Figure_2.png
│   │   ├── Figure_3.png
│   │   ├── Figure_4.png
│   │   ├── Figure_5.png
│   │   ├── Figure_6.png
│   │   ├── image copy.png
│   │   ├── image.png
│── src/
│   ├── chatbot.py
│   ├── dataset.py
│   ├── fer1.py
│   ├── modules.py
│   ├── reccomend.py
│   ├── train_music_reccomendation.py
│── static/
│   ├── styles.css
│   
│── templates/
│   ├── about.html
│   ├── contact.html
│   ├── cookiepolicy.html
│   ├── features.html
│   ├── footer.html
│   ├── header.html
│   ├── home.html
│   ├── index.html
│   ├── pricing.html
│   ├── privacy.html
│   ├── services.html
│   ├── wellness_tools.html
│── Website-Images/
│   ├── Aboutpage.jpg
│   ├── Contactpage.jpg
│   ├── Homepage.jpg
│   ├── MainFunctionality.jpg

---
```







### **Installation & Setup**  

#### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/yourusername/emotion-music-recommendation.git
cd emotion-music-recommendation
```

#### **2️⃣ Set Up Virtual Environment**  
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

#### **3️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

#### **4️⃣ Run the Flask App**  
```bash
python app.py
```
Then, open **`http://127.0.0.1:5000/`** in your browser. 🎉  

---

### **Features & Functionality 🚀**  

| Feature                        | Description |
|--------------------------------|-------------|
| **🎭 Facial Emotion Detection**  | Uses DeepFace + AI models to analyze facial expressions in real-time. |
| **💬 Chatbot Integration**       | Analyzes text-based emotions while chatting with the user. |
| **📊 Multimodal Emotion Analysis** | Combines facial + text emotions for better accuracy. |
| **🎶 AI Music Recommendation**   | Suggests songs based on detected emotions using a content-based model. |
| **🌐 Flask Web App**            | A user-friendly UI where all processes run seamlessly. |
| **🔄 Real-time Emotion Logging** | Continuously tracks emotions and updates song recommendations dynamically. |

---

### **How It Works?**  
1️⃣ **User starts the system** → **Camera & chatbot analyze emotions**  
2️⃣ **Emotions are logged & averaged** → **Final emotion is detected**  
3️⃣ **Music Recommendation Model suggests songs** based on emotions 🎵  
4️⃣ **User gets song recommendations in real-time**  

---

### **Customization & Future Enhancements**  
✅ **Train on a custom dataset** for better emotion detection.  
✅ **Improve chatbot intelligence** with a more advanced LLM (e.g., LLaMA, GPT4All).  
✅ **Enhance UI/UX** with animations and design improvements.  
✅ **Deploy the Flask app online** for real-world use.  

---

### **Contributors & Credits**  
🚀 Developed by **Pallav Sharma** and **Abhiraj Ghose**  
💡 Special thanks to OpenAI, DeepFace, and Flask community!  

---

### **License**  
📜 This project is **open-source** under the MIT License.  

---

https://github.com/user-attachments/assets/3f4b822c-71db-4360-a202-a64f098c1137


### **Changes**
#### Features left and their description
- **💡** Still need changes in design and responsiveness.  
- **💡** Need to make more accurate decisions for recommending music.  
- **💡** Need to handle camera functionality well with start and stop.  
- **💡** Need to develop other functions for buttons as well.  
- **💡** Need to introduce a saving method for music.  

---

**Warning ⚠️ : The website is still in building process, and the functionality may change over time.** 

