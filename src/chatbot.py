import os
import json
import torch
import requests
from transformers import pipeline
from rich.console import Console
from rich.panel import Panel
import time

# Rich Console for Styling
console = Console()

# Groq API Configuration
GROQ_API_KEY = "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"  # Replace with actual key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Load emotion classifiers
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    emotion_models = [
        pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", device=0 if device == "cuda" else -1),
        pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", device=0 if device == "cuda" else -1),
        pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=0 if device == "cuda" else -1)
    ]
except Exception as e:
    console.print(f"[bold red][ERROR] Failed to load emotion models: {e}[/bold red]")
    exit(1)

# Emotion mapping to unify different models' outputs
emotion_map = {
    "joy": "happy", "happiness": "happy", "excitement": "happy",
    "anger": "angry", "annoyance": "angry",
    "sadness": "sad", "grief": "sad",
    "fear": "fearful", "surprise": "surprised",
    "disgust": "disgusted", "neutral": "neutral",
}

# Store user chat session
chat_session = []

def detect_conversation_emotions(chat_history):
    """Analyzes the entire chat history to detect dominant emotions."""
    full_chat_text = " ".join([entry["user"] for entry in chat_history])  # Combine user inputs
    emotion_scores = {}
    emotion_counts = {}
    model_emotions = []

    try:
        for model in emotion_models:
            results = model(full_chat_text)
            top_predictions = sorted(results, key=lambda x: x["score"], reverse=True)[:2]

            for pred in top_predictions:
                model_label = pred["label"].lower()
                model_score = pred["score"]
                mapped_emotion = emotion_map.get(model_label, "neutral")
                model_emotions.append(f"{model_label} ({model_score:.2f}) → {mapped_emotion}")

                if model_score > 0.3:
                    if mapped_emotion not in emotion_scores:
                        emotion_scores[mapped_emotion] = model_score
                        emotion_counts[mapped_emotion] = 1
                    else:
                        emotion_scores[mapped_emotion] += model_score
                        emotion_counts[mapped_emotion] += 1

        avg_emotion_scores = {label: emotion_scores[label] / emotion_counts[label] for label in emotion_scores}
        dominant_emotion = max(avg_emotion_scores, key=avg_emotion_scores.get) if avg_emotion_scores else "neutral"
        
        console.print(f"\n[bold cyan]🧠 Detected Emotion:[/bold cyan] [bold yellow]{dominant_emotion}[/bold yellow]")
        
        return dominant_emotion, model_emotions
    except Exception as e:
        console.print(f"[bold red][ERROR] Emotion detection failed: {e}[/bold red]")
        return "neutral", []

def generate_chatbot_response(user_input):
    """Generates chatbot response using Groq API."""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama3-70b-8192", "messages": [{"role": "user", "content": user_input}]}

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response_json = response.json()

        if response.status_code == 200 and "choices" in response_json:
            return response_json["choices"][0]["message"]["content"].strip()
        else:
            console.print(f"[bold red][ERROR] Groq API request failed: {response_json}[/bold red]")
            return "I'm sorry, but I couldn't process your request."
    except Exception as e:
        console.print(f"[bold red][ERROR] Groq API request failed: {e}[/bold red]")
        return "I'm facing a technical issue. Please try again later."

def save_chat_results():
    """Saves chatbot results (full conversation) to `chat_results.json`."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Format timestamp
    dominant_emotion, model_emotions = detect_conversation_emotions(chat_session)

    chat_data = {
        "timestamp": timestamp,
        "conversation": chat_session,
        "dominant_emotion": dominant_emotion,
        "model_emotions": model_emotions
    }

    with open("chat_results.json", "w") as f:
        json.dump(chat_data, f, indent=4)

    console.print(f"\n[bold green]✅ Chat session saved! Detected emotion: {dominant_emotion}[/bold green]")

def main():
    console.print(Panel.fit("💬 [bold magenta]AI Chatbot (Groq)[/bold magenta]", subtitle="Type 'exit' to stop."))

    while True:
        user_input = console.input("[bold cyan]You: [/bold cyan]")

        if user_input.lower() == "exit":
            save_chat_results()
            console.print("[bold yellow]👋 Chatbot shutting down.[/bold yellow]")
            break

        chatbot_response = generate_chatbot_response(user_input)
        console.print(f"[bold green]🤖 Me: {chatbot_response}[/bold green]")

        chat_session.append({"user": user_input, "chatbot": chatbot_response})

if __name__ == "__main__":
    main()
