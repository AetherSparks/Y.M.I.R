"""
TEXT ANALYSIS & CHAT MICROSERVICE
Extracted from app.py lines 520-536, 1953-2015
Contains chatbot functionality and text processing
"""
import json
import time
import os
import requests
from typing import Dict, List, Optional
from transformers import pipeline

class TextAnalysisService:
    def __init__(self, groq_api_key=None):
        # API Configuration
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # Chat session storage (from app.py:425)
        self.chat_session = []
        
        # Load emotion models (from app.py:402-406) - Optional, can be initialized later
        self.emotion_models = None
        self._initialize_models()
        
        # Emotion mapping (from app.py:413-419)
        self.emotion_map = {
            "joy": "happy", "happiness": "happy", "excitement": "happy",
            "anger": "angry", "annoyance": "angry", 
            "sadness": "sad", "grief": "sad",
            "fear": "fearful", "surprise": "surprised",
            "disgust": "disgusted", "neutral": "neutral",
        }
        
        print("✅ Text Analysis & Chat Service initialized")

    def _initialize_models(self):
        """Initialize transformer models for emotion analysis"""
        try:
            # Load emotion models (from app.py:402-406)
            self.emotion_models = [
                pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion"),
                pipeline("text-classification", model="SamLowe/roberta-base-go_emotions"),
                pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
            ]
            print("✅ Emotion analysis models loaded")
        except Exception as e:
            print(f"⚠️ Could not load emotion models: {e}")
            self.emotion_models = None

    # EXTRACTED FROM app.py:520-536 (EXACT COPY)
    def generate_chatbot_response(self, user_input):
        """Generates chatbot response using Groq API"""
        headers = {"Authorization": f"Bearer {self.groq_api_key}", "Content-Type": "application/json"}
        payload = {"model": "llama3-70b-8192", "messages": [{"role": "user", "content": user_input}]}

        try:
            response = requests.post(self.groq_api_url, headers=headers, json=payload)
            response_json = response.json()

            if response.status_code == 200 and "choices" in response_json:
                return response_json["choices"][0]["message"]["content"].strip()
            else:
                print(f"[ERROR] Groq API request failed: {response_json}")
                return "I'm sorry, but I couldn't process your request."
        except Exception as e:
            print(f"[ERROR] Groq API request failed: {e}")
            return "I'm facing a technical issue. Please try again later."

    def detect_conversation_emotions(self, chat_history):
        """Analyzes chat history and detects emotions using transformer models"""
        if not self.emotion_models or not chat_history:
            # Fallback to simple keyword-based emotion detection
            return self._simple_emotion_detection(chat_history)
            
        emotion_scores = {}
        emotion_counts = {}
        model_emotions = []
        
        # More weight to recent messages
        recent_weight = 1.5  
        messages = chat_history[-5:]  # Use last 5 messages for better context
        full_chat_text = " ".join([entry["user"] for entry in messages])

        try:
            for model in self.emotion_models:
                results = model(full_chat_text)
                top_predictions = sorted(results, key=lambda x: x["score"], reverse=True)[:2]

                for pred in top_predictions:
                    model_label = pred["label"].lower()
                    model_score = pred["score"]
                    mapped_emotion = self.emotion_map.get(model_label, "neutral")
                    model_emotions.append(f"{model_label} ({model_score:.2f}) → {mapped_emotion}")

                    if model_score < 0.4:  # Ignore weak emotions
                        continue  

                    # Apply weight to recent messages
                    weighted_score = model_score * (recent_weight if len(messages) > 0 else 1.0)

                    if mapped_emotion not in emotion_scores:
                        emotion_scores[mapped_emotion] = weighted_score
                        emotion_counts[mapped_emotion] = 1
                    else:
                        emotion_scores[mapped_emotion] += weighted_score
                        emotion_counts[mapped_emotion] += 1

            # Compute weighted average
            avg_emotion_scores = {label: emotion_scores[label] / emotion_counts[label] for label in emotion_scores}

            # Compute final dominant emotion
            if avg_emotion_scores:
                dominant_emotion = max(avg_emotion_scores, key=avg_emotion_scores.get)
            else:
                dominant_emotion = "neutral"

            return dominant_emotion, avg_emotion_scores, model_emotions
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return self._simple_emotion_detection(chat_history)

    def _simple_emotion_detection(self, chat_history):
        """Simple keyword-based emotion detection as fallback"""
        if not chat_history:
            return "neutral", {}, []
            
        full_text = " ".join([entry["user"] for entry in chat_history[-5:]]).lower()
        
        emotion_keywords = {
            "happy": ["happy", "joy", "excited", "great", "awesome", "wonderful", "good"],
            "sad": ["sad", "depressed", "down", "unhappy", "terrible", "awful", "bad"],
            "angry": ["angry", "mad", "furious", "annoyed", "irritated", "frustrated"],
            "fearful": ["scared", "afraid", "worried", "anxious", "nervous", "fear"],
            "surprised": ["surprised", "shocked", "amazed", "wow", "incredible"],
            "neutral": ["okay", "fine", "normal", "average"]
        }
        
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in full_text)
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            return dominant_emotion, emotion_scores, [f"Keyword-based: {dominant_emotion}"]
        else:
            return "neutral", {"neutral": 1}, ["Keyword-based: neutral"]

    # EXTRACTED FROM app.py:1953-1980 (EXACT COPY - modified for service)
    def handle_chat_interaction(self, user_input):
        """Handles chatbot interaction and continuously updates detected emotions"""
        user_input = user_input.strip()
        if not user_input:
            return {"error": "No message provided."}

        chatbot_response = self.generate_chatbot_response(user_input)

        # Log conversation
        self.chat_session.append({"user": user_input, "chatbot": chatbot_response})

        # Continuously detect dominant emotion
        dominant_emotion, emotion_scores, model_emotions = self.detect_conversation_emotions(self.chat_session)

        response_data = {
            "response": chatbot_response,
            "dominant_emotion": dominant_emotion,  # Always update detected emotion
            "model_emotions": model_emotions
        }

        # Save chat results **after every message**
        self.save_chat_results()

        # If the user is ending the conversation
        if user_input.lower() in ["quit", "bye", "exit", "goodbye", "end"]:
            response_data["end_chat"] = True  # Notify frontend to stop input

        return response_data

    def detect_emotion_from_current_session(self):
        """Detects the dominant emotion from the current chat session"""
        if not self.chat_session:
            return {"dominant_emotion": "neutral", "model_emotions": []}  # Default if empty chat

        dominant_emotion, emotion_scores, model_emotions = self.detect_conversation_emotions(self.chat_session)
        return {"dominant_emotion": dominant_emotion, "model_emotions": model_emotions}

    # EXTRACTED FROM app.py:2000-2014 (EXACT COPY)  
    def save_chat_results(self):
        """Saves chatbot results (full conversation + updated emotions) to `chat_results.json`"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        dominant_emotion, emotion_scores, model_emotions = self.detect_conversation_emotions(self.chat_session)

        chat_data = {
            "timestamp": timestamp,
            "conversation": self.chat_session,
            "dominant_emotion": dominant_emotion,
            "emotion_scores": emotion_scores,
            "model_emotions": model_emotions
        }

        with open("chat_results.json", "w") as f:
            json.dump(chat_data, f, indent=4)

    def get_chat_session(self):
        """Get current chat session"""
        return {
            "session_length": len(self.chat_session),
            "conversation": self.chat_session,
            "last_message": self.chat_session[-1] if self.chat_session else None
        }

    def clear_chat_session(self):
        """Clear current chat session"""
        self.chat_session = []
        print("🧹 Chat session cleared")

    def save_chat_session(self, filename="chat_session.json"):
        """Save current chat session to file"""
        try:
            with open(filename, "w") as f:
                json.dump({
                    "session_data": self.chat_session,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_messages": len(self.chat_session)
                }, f, indent=4)
            return {"success": True, "message": f"Chat session saved to {filename}"}
        except Exception as e:
            return {"success": False, "message": f"Error saving chat session: {e}"}

    def load_chat_session(self, filename="chat_session.json"):
        """Load chat session from file"""
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                self.chat_session = data.get("session_data", [])
            return {"success": True, "message": f"Chat session loaded from {filename}"}
        except FileNotFoundError:
            return {"success": False, "message": "Chat session file not found"}
        except Exception as e:
            return {"success": False, "message": f"Error loading chat session: {e}"}

    def get_conversation_summary(self):
        """Get a summary of the current conversation"""
        if not self.chat_session:
            return {"summary": "No conversation yet"}
            
        user_messages = [entry["user"] for entry in self.chat_session]
        bot_responses = [entry["chatbot"] for entry in self.chat_session]
        
        dominant_emotion, emotion_scores, _ = self.detect_conversation_emotions(self.chat_session)
        
        return {
            "total_exchanges": len(self.chat_session),
            "dominant_emotion": dominant_emotion,
            "emotion_scores": emotion_scores,
            "conversation_length": sum(len(msg) for msg in user_messages),
            "last_user_message": user_messages[-1] if user_messages else None,
            "last_bot_response": bot_responses[-1] if bot_responses else None
        }

    def cleanup(self):
        """Cleanup service resources"""
        # Save current session before cleanup
        if self.chat_session:
            self.save_chat_results()
        print("🧹 Text Analysis & Chat service cleanup complete")