import json
import torch
import requests
import time
import re
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
import os

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

console = Console()
print(f"GROQ API Key: {GROQ_API_KEY}")  

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

device = "cuda" if torch.cuda.is_available() else "cpu"
emotion_models = [
    pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", device=0 if device == "cuda" else -1),
    pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", device=0 if device == "cuda" else -1),
    pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=0 if device == "cuda" else -1)
]

sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

emotion_map = {
    "joy": "happy", "happiness": "happy", "excitement": "happy",
    "anger": "angry", "annoyance": "angry",
    "sadness": "sad", "grief": "sad",
    "fear": "fearful", "surprise": "surprised",
    "disgust": "disgusted", "neutral": "neutral",
}

previous_emotions = []

chat_session = []

def handle_negations(text):
    """Detects negations and flips associated emotions."""
    negation_patterns = [
        r"\b(not|never|no)\s+(happy|joyful|excited)\b",
        r"\b(not|never|no)\s+(sad|depressed|unhappy)\b",
        r"\b(not|never|no)\s+(angry|mad|furious)\b"
    ]
    
    for pattern in negation_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            emotion = match.group(2).lower()
            if emotion in ["happy", "joyful", "excited"]:
                return "sad"
            elif emotion in ["sad", "depressed", "unhappy"]:
                return "happy"
            elif emotion in ["angry", "mad", "furious"]:
                return "calm"
    return None

def detect_sentiment(text):
    """Detects sentiment polarity (positive, neutral, negative)."""
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = sentiment_model(**inputs)
    sentiment_scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    sentiment_labels = ["negative", "neutral", "positive"]
    
    return sentiment_labels[torch.argmax(sentiment_scores).item()]

def detect_conversation_emotions(chat_history):
    """Analyzes chat history, considers recent messages more, and balances emotion scores."""
    emotion_scores = {}
    emotion_counts = {}
    model_emotions = []
    
    recent_weight = 1.5  
    messages = chat_history[-5:]  
    full_chat_text = " ".join([entry["user"] for entry in messages])

    negated_emotion = handle_negations(full_chat_text)
    if negated_emotion:
        return negated_emotion, {}, []

    for model in emotion_models:
        results = model(full_chat_text)
        top_predictions = sorted(results, key=lambda x: x["score"], reverse=True)[:2]

        for pred in top_predictions:
            model_label = pred["label"].lower()
            model_score = pred["score"]
            mapped_emotion = emotion_map.get(model_label, "neutral")
            model_emotions.append(f"{model_label} ({model_score:.2f}) → {mapped_emotion}")

            if model_score < 0.4:  
                continue  

            weighted_score = model_score * (recent_weight if messages[-1]["user"] == full_chat_text else 1.0)

            if mapped_emotion not in emotion_scores:
                emotion_scores[mapped_emotion] = weighted_score
                emotion_counts[mapped_emotion] = 1
            else:
                emotion_scores[mapped_emotion] += weighted_score
                emotion_counts[mapped_emotion] += 1

    avg_emotion_scores = {label: emotion_scores[label] / emotion_counts[label] for label in emotion_scores}

    sentiment = detect_sentiment(full_chat_text)
    if sentiment == "negative" and "sad" in avg_emotion_scores:
        avg_emotion_scores["sad"] += 0.1  
    if len(previous_emotions) > 5:
        previous_emotions.pop(0)
    previous_emotions.append(avg_emotion_scores)

    if avg_emotion_scores:
        dominant_emotion = max(avg_emotion_scores, key=avg_emotion_scores.get)
    else:
        dominant_emotion = "neutral"

    console.print(f"\n[bold cyan]🧠 Updated Detected Emotion:[/bold cyan] [bold yellow]{dominant_emotion}[/bold yellow]")

    return dominant_emotion, avg_emotion_scores, model_emotions

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
    """Saves chatbot results (full conversation + updated emotions) to `chat_results.json`."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    dominant_emotion, emotion_scores, model_emotions = detect_conversation_emotions(chat_session)

    chat_data = {
        "timestamp": timestamp,
        "conversation": chat_session,
        "dominant_emotion": dominant_emotion,
        "emotion_scores": emotion_scores,
        "model_emotions": model_emotions
    }

    with open("chat_results.json", "w") as f:
        json.dump(chat_data, f, indent=4)

def main():
    console.print(Panel.fit("💬 [bold magenta]AI Chatbot (Groq)[/bold magenta]", subtitle="Type 'exit' to stop."))

    while True:
        user_input = console.input("[bold cyan]You: [/bold cyan]")

        if user_input.lower() == "exit":
            save_chat_results()
            break

        chatbot_response = generate_chatbot_response(user_input)
        print("DEBUG: Chatbot response:", chatbot_response)  

        chat_session.append({"user": user_input, "chatbot": chatbot_response})
        save_chat_results()

if __name__ == "__main__":
    main()
