"""
Y.M.I.R AI Emotion Detection System - Main Flask Application
===========================================================
Basic Flask app with microservices architecture for emotion detection,
chatbot integration, and music recommendations.

Author: Y.M.I.R Development Team
Version: 1.0.0
"""

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
from flask import Flask, render_template, request, jsonify, url_for, Response
from flask_cors import CORS
import os
import requests
import json
from datetime import datetime
import sys
from pathlib import Path
from dotenv import load_dotenv
from email.message import EmailMessage
import random
from signal import signal
import smtplib
import sys
from dotenv import load_dotenv
load_dotenv()
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import atexit
import json
import pickle
import cv2
import numpy as np
import threading
import time
import mediapipe as mp
import dlib
import warnings
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, Response, jsonify, render_template_string, request, send_from_directory, session, url_for, redirect ,flash
from flask_mail import Mail , Message
from flask_session import Session
from flask_cors import CORS
from scipy.spatial import distance as dist
from collections import deque
from transformers import pipeline
from rich.console import Console
import pandas as pd
import torch
import requests
import time
import re
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

# Load environment variables
load_dotenv()

# Import Firebase Authentication
try:
    from firebase_auth_backend import firebase_auth, add_auth_routes, require_auth, optional_auth
    FIREBASE_AUTH_AVAILABLE = True
    print("✅ Firebase Authentication available")
except ImportError as e:
    FIREBASE_AUTH_AVAILABLE = False
    print(f"⚠️ Firebase Authentication not available: {e}")
    firebase_auth = None

# Import multimodal emotion combiner
try:
    combiner_path = Path(__file__).parent / 'enhancements' / 'src-new' / 'multimodal_fusion'
    sys.path.append(str(combiner_path))
    from real_emotion_combiner import RealEmotionCombiner, RealCombinedEmotion
    EMOTION_COMBINER_AVAILABLE = True
except ImportError:
    EMOTION_COMBINER_AVAILABLE = False
    RealEmotionCombiner = None
    RealCombinedEmotion = None
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Initialize Flask app
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ymir-dev-key-2024')
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════



#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Email Config
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USE_SSL = False,
    MAIL_USERNAME=os.environ.get('EMAIL_USER'),
    MAIL_PASSWORD=os.environ.get('EMAIL_PASS'),
    MAIL_DEFAULT_SENDER=os.environ.get('EMAIL_USER')
)

# Initialize mail with app explicitly
mail = Mail(app)


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///emotion_ai.db'  # or use PostgreSQL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# print("Mail config:", {
#     "server": app.config['MAIL_SERVER'],
#     "port": app.config['MAIL_PORT'],
#     "username": bool(app.config['MAIL_USERNAME']),  # Just print if it exists
#     "password": bool(app.config['MAIL_PASSWORD']),  # Jus   t print if it exists
#     "use_tls": app.config['MAIL_USE_TLS']
# })
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Enable CORS for API calls
CORS(app)

# Configure static files
app.static_folder = 'static'
app.template_folder = 'templates'

# Microservice URLs
FACE_MICROSERVICE_URL = 'http://localhost:5002'
TEXT_MICROSERVICE_URL = 'http://localhost:5003'
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


class MicroserviceClient:
    """Client to communicate with microservices"""
    
    def __init__(self):
        self.face_service_url = FACE_MICROSERVICE_URL
        self.text_service_url = TEXT_MICROSERVICE_URL
        
        # Initialize emotion combiner
        if EMOTION_COMBINER_AVAILABLE:
            self.emotion_combiner = RealEmotionCombiner()
        else:
            self.emotion_combiner = None
    
    def check_face_service_health(self):
        """Check if face microservice is running"""
        try:
            response = requests.get(f'{self.face_service_url}/health', timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def start_camera(self):
        """Start camera via microservice"""
        try:
            response = requests.post(f'{self.face_service_url}/api/start_camera', timeout=15)
            return response.json()
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Camera start timed out'}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': 'Cannot connect to face microservice'}
        except Exception as e:
            return {'success': False, 'error': f'Microservice error: {str(e)}'}
    
    def stop_camera(self):
        """Stop camera via microservice"""
        try:
            response = requests.post(f'{self.face_service_url}/api/stop_camera', timeout=10)
            return response.json()
        except Exception as e:
            return {'success': False, 'error': f'Microservice error: {str(e)}'}
    
    def get_emotions(self):
        """Get current emotions from microservice"""
        try:
            response = requests.get(f'{self.face_service_url}/api/emotions', timeout=2)
            return response.json()
        except Exception as e:
            return {'error': f'Microservice error: {str(e)}'}
    
    def get_face_service_status(self):
        """Get face service status"""
        try:
            response = requests.get(f'{self.face_service_url}/api/status', timeout=2)
            return response.json()
        except Exception as e:
            return {'error': f'Microservice error: {str(e)}'}
    
    def check_text_service_health(self):
        """Check if text microservice is running"""
        try:
            response = requests.get(f'{self.text_service_url}/health', timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def analyze_text(self, text, is_user=True):
        """Analyze text emotion via microservice"""
        try:
            response = requests.post(f'{self.text_service_url}/api/analyze_text', 
                                   json={'text': text, 'is_user': is_user}, timeout=10)
            return response.json()
        except Exception as e:
            return {'success': False, 'error': f'Text microservice error: {str(e)}'}
    
    def chat_with_bot(self, message, auth_header=None, session_id=None):
        """Chat with bot via microservice"""
        try:
            headers = {'Content-Type': 'application/json'}
            if auth_header:
                headers['Authorization'] = auth_header
            
            # Prepare payload with message and optional session_id
            payload = {'message': message}
            if session_id:
                payload['session_id'] = session_id
            
            response = requests.post(f'{self.text_service_url}/api/chat',
                                   json=payload, 
                                   headers=headers,
                                   timeout=45)
            return response.json()
        except Exception as e:
            return {'success': False, 'error': f'Chat microservice error: {str(e)}'}
    
    def get_text_conversation(self):
        """Get conversation history from text microservice"""
        try:
            response = requests.get(f'{self.text_service_url}/api/conversation', timeout=5)
            return response.json()
        except Exception as e:
            return {'success': False, 'error': f'Text microservice error: {str(e)}'}
    
    def get_text_service_status(self):
        """Get text service status"""
        try:
            response = requests.get(f'{self.text_service_url}/api/status', timeout=2)
            return response.json()
        except Exception as e:
            return {'error': f'Text microservice error: {str(e)}'}
    
    def get_combined_emotions(self):
        """Get combined emotions from both face and text microservices"""
        if not self.emotion_combiner:
            return {'error': 'Emotion combiner not available'}
        
        try:
            # Only get face emotions if camera is actually running
            face_status = self.get_face_service_status()
            face_emotions = {}
            
            if face_status.get('running') and not face_status.get('error'):
                face_emotions = self.get_emotions()
            else:
                # Camera not running, don't trigger it with API calls
                print("⚠️ Skipping face emotions check (camera not running)")
                face_emotions = {'status': 'camera_not_running'}
            
            text_status = self.get_text_service_status()
            
            # Combine emotions using the fusion engine (works with Firebase data directly)
            from real_emotion_combiner import get_combined_emotion
            combined_result = get_combined_emotion(minutes_back=5, strategy='adaptive')
            
            if combined_result:
                # Convert to expected format
                combined = RealCombinedEmotion(
                    dominant_emotion=combined_result['emotion'],
                    confidence=combined_result['confidence'],
                    combination_method=combined_result.get('strategy', 'adaptive'),
                    facial_source=combined_result.get('facial_data'),
                    text_source=combined_result.get('text_data')
                )
            else:
                combined = None
            
            if combined:
                return {
                    'success': True,
                    'combined_emotion': {
                        'dominant_emotion': combined.dominant_emotion,
                        'confidence': combined.confidence,
                        'combination_method': combined.combination_method,
                        'timestamp': combined.timestamp.isoformat(),
                        'facial_source': combined.facial_source,
                        'text_source': combined.text_source,
                        # Multi-emotion support
                        'top_emotions': getattr(combined, 'top_emotions', [(combined.dominant_emotion, combined.confidence)]),
                        'is_multi_emotion': getattr(combined, 'is_multi_emotion', False),
                        'fusion_weights': getattr(combined, 'fusion_weights', {'facial': 0.5, 'text': 0.5}),
                        'all_emotions': getattr(combined, 'all_fused_emotions', {combined.dominant_emotion: combined.confidence})
                    },
                    'face_emotions': face_emotions,
                    'text_available': not text_status.get('error')
                }
            else:
                return {
                    'success': False,
                    'error': 'No combined emotion data available'
                }
        except Exception as e:
            return {'error': f'Combined emotion error: {str(e)}'}

# Initialize microservice client
microservice_client = MicroserviceClient()

# Add emotion combiner monitoring
import threading

# Simple cache for music recommendations to prevent excessive API calls
music_recommendation_cache = {}
MUSIC_CACHE_DURATION = 20  # 20 seconds for immediate emotion response
def monitor_combined_emotions():
    """Monitor and log combined emotions every 60 seconds"""
    import time
    while True:
        try:
            time.sleep(60)  # Check every 60 seconds (reduced from 10 to save API calls)
            if EMOTION_COMBINER_AVAILABLE and microservice_client.emotion_combiner:
                # 🔍 CHECK: Only monitor if face service is actually running to avoid triggering camera
                face_status = microservice_client.get_face_service_status()
                
                if face_status.get('running') and not face_status.get('error'):
                    print(f"\n🔗 EMOTION COMBINER CHECK (camera running)")
                    print("=" * 50)
                    
                    # Get combined emotions ONLY when camera is active
                    combined_result = microservice_client.get_combined_emotions()
                    
                    if combined_result.get('success'):
                        combined = combined_result['combined_emotion']
                        print(f"🎯 COMBINED EMOTION: {combined['dominant_emotion'].upper()}")
                        print(f"   Confidence: {combined['confidence']:.2f}")
                        print(f"   Method: {combined['combination_method']}")
                        print(f"   Timestamp: {combined['timestamp']}")
                        
                        if combined['facial_source']:
                            print(f"   📹 Facial data: Available")
                        else:
                            print(f"   📹 Facial data: None")
                            
                        if combined['text_source']:
                            print(f"   💬 Text data: Available")
                        else:
                            print(f"   💬 Text data: None")
                    else:
                        print(f"❌ Combined emotion error: {combined_result.get('error', 'Unknown')}")
                    
                    print("=" * 50)
                else:
                    # 🚫 DON'T call get_combined_emotions when camera is not running to avoid auto-start
                    print(f"⏸️ Emotion combiner paused (camera not running)")
                
        except Exception as e:
            print(f"⚠️ Emotion combiner monitoring error: {e}")

# Start the monitoring thread
if EMOTION_COMBINER_AVAILABLE:
    monitor_thread = threading.Thread(target=monitor_combined_emotions, daemon=True)
    monitor_thread.start()
    print("✅ Emotion combiner monitoring started (every 60 seconds)")


#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#reya's implementation
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

DATA_FILE = 'goals.json'
POSTS_FILE = 'data/posts.json'
app.secret_key = "your_secret_key"

analyzer = SentimentIntensityAnalyzer()

# Mood-based meditation scripts (more human-like and varied)
SCRIPTS = {
    "anxious": [
        "Take a deep breath in... and out. Imagine a calm ocean. Let the waves carry your anxiety away.",
        "Inhale peace. Exhale worry. Picture a peaceful forest with birds gently chirping.",
        "You are safe. You are grounded. With every breath, let go of anxious thoughts."
    ],
    "stressed": [
        "Let go of tension with every breath. Relax your shoulders. You are safe.",
        "You are not your stress. You are strength, you are calm. Let each exhale ground you.",
        "Breathe deeply. Picture your thoughts floating like clouds, drifting far away."
    ],
    "tired": [
        "Close your eyes. Imagine a soft, glowing light recharging your body and mind.",
        "Sink into stillness. Every breath is a wave of renewal flowing through you.",
        "Let your body rest. Let your mind slow down. You deserve peace and rest."
    ],
    "happy": [
        "Let’s deepen your joy. Smile softly and be present with the happiness within.",
        "Breathe in gratitude. Breathe out love. Stay with this beautiful feeling.",
        "Feel the warmth inside you. Your happiness is a gift — cherish this moment."
    ]
}

# Optional: Motivational quotes to display with the meditation
QUOTES = [
    "You are enough. Just as you are.",
    "Breathe. You’ve got this.",
    "Peace begins with a single breath.",
    "Today is a fresh start.",
    "Inner calm is your superpower."
]

def suggest_breathing(mood):
    techniques = {
        "anxious": "Box Breathing (4-4-4-4) – Inhale, hold, exhale, hold for 4 seconds each.",
        "stressed": "4-7-8 Breathing – Inhale 4s, hold 7s, exhale 8s. Great for calming nerves.",
        "tired": "Diaphragmatic Breathing – Deep belly breaths to refresh energy.",
        "distracted": "Alternate Nostril Breathing – Helps center your focus.",
        "neutral": "Guided Breath Awareness – Simply observe your breath."
    }
    return techniques.get(mood, "Try Box Breathing to get started.")
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════



#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Flask Route: Home
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

@app.route('/')
def home1():
    return render_template('home.html')

@app.route('/ai_app')
def ai_app():
    """Main AI application dashboard"""
    # Check if microservices are running
    face_service_status = microservice_client.check_face_service_health()
    text_service_status = microservice_client.check_text_service_health()
    
    return render_template('ai_dashboard.html', 
                         face_service_available=face_service_status,
                         text_service_available=text_service_status)

@app.route('/about')
def about():
    return render_template('about.html')  

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/wellness')
def wellness():
    return render_template('wellness_tools.html')

@app.route('/gaming')
def gaming():
    return render_template('gaming.html')

@app.route('/cookies')
def cookies():
    return render_template('cookiepolicy.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        try:
            print("======= CONTACT FORM SUBMISSION =======")
            print("Form data:", request.form)

            # Get form data
            name = request.form.get('name', '')
            email = request.form.get('email', '')
            subject = request.form.get('subject', 'No Subject')
            message = request.form.get('message', '')
            phone = request.form.get('phone', 'Not provided')

            # Timestamp for submission
            submission_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Create the email message
            print("Creating message...")
            msg = Message(
                subject=f"New Contact Inquiry: {subject}",
                sender=("Your Website Contact Form", app.config.get('MAIL_DEFAULT_SENDER') or os.getenv('EMAIL_USER')),
                recipients=[app.config.get('MAIL_USERNAME') or os.getenv('EMAIL_USER')],
                reply_to=email
            )

            # Plain text body (ASCII-safe fallback)
            msg.body = f"""Hello Admin,

You have received a new contact form submission on your website.

Submitted On: {submission_time}

Name: {name}
Email: {email}
Phone: {phone}
Subject: {subject}
Message:
{message}

Please respond promptly.
"""

            # HTML body (UTF-8 + emoji support)
            html_body = render_template_string("""
<html>
  <body style="font-family: Arial, sans-serif; color: #333;">
    <h2> New Contact Form Submission</h2>
    <p><strong> Submitted On:</strong> {{ submission_time }}</p>
    <p><strong> Name:</strong> {{ name }}</p>
    <p><strong> Email:</strong> {{ email }}</p>
    <p><strong> Phone:</strong> {{ phone }}</p>
    <p><strong> Subject:</strong> {{ subject }}</p>
    <p><strong> Message:</strong><br>{{ message }}</p>
    <hr>
    <p>Regards,<br><strong>Your Website Bot</strong></p>
  </body>
</html>
""", submission_time=submission_time, name=name, email=email, phone=phone, subject=subject, message=message.replace('\n', '<br>'))

            msg.html = html_body  # Attach HTML email

            # Optional: Handle attachments
            if 'attachment' in request.files:
                file = request.files['attachment']
                if file and file.filename != '':
                    print(f"Attaching file: {file.filename}")
                    file_content = file.read()
                    msg.attach(file.filename, file.content_type, file_content)
                    print("Attachment added.")

            # Send email
            print("Sending email...")
            mail.send(msg)
            print("Email sent!")

            return jsonify({"success": True, "message": "Thank you! Your message has been sent."})

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print("======= CONTACT FORM ERROR =======")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Traceback:\n{error_details}")

            return jsonify({
                "success": False,
                "message": "Oops! Something went wrong. Please try again later."
            }), 500

    # GET request → render contact form
    return render_template('contact.html')

from flask import jsonify

@app.route('/book-appointment', methods=['POST'])
def book_appointment():
    user_name = request.form['user_name']
    user_email = request.form['user_email']
    appointment_date = request.form['appointment_date']
    time_slot = request.form['time_slot']
    duration = request.form['meeting_duration']
    timezone = request.form['timezone']
    notes = request.form['appointment_notes']

    subject = 'New appointment Booking!'
    body = f"""
    New Appointment Booked!

    Name: {user_name}
    Email: {user_email}
    Appointment Date: {appointment_date}
    Time Slot: {time_slot}
    Duration: {duration} minutes
    Timezone: {timezone}
    Notes: {notes}
    """

    sender_email = os.environ.get('EMAIL_USER')
    receiver_email = os.environ.get('EMAIL_USER')
    password = os.environ.get('EMAIL_PASS')

    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg.set_content(body)

        # Send Email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, password)
            smtp.send_message(msg)

        # ✅ Return JSON success message (instead of flash)
        return jsonify({"success": True, "message": "Appointment booked successfully!"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "message": "Failed to send email!"}), 500
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#Emotion based jorunaling and welness tool  [REYA'S IMPLEMENTATION]
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
@app.route("/meditation")
def meditation():
    return render_template("meditation.html")

@app.route("/meditation/result", methods=["POST"])
def meditation_result():
    feeling = request.form.get("feeling", "").lower()

    # Find matching script list based on mood keyword
    for mood, scripts in SCRIPTS.items():
        if mood in feeling:
            script = random.choice(scripts)
            break
    else:
        # Default script if mood not found
        script = f"Let’s take a few moments to be still. You mentioned feeling '{feeling}'. Breathe deeply and allow peace to fill your body."

    quote = random.choice(QUOTES)

    return render_template("meditation_result.html", script=script, quote=quote)

@app.route('/breathing', methods=['GET', 'POST'])
def breathing():
    suggestion = None
    if request.method == 'POST':
        mood = request.form['mood'].lower()
        suggestion = suggest_breathing(mood)
    return render_template('breathing.html', suggestion=suggestion)

@app.route('/journal', methods=['GET', 'POST'])
def journal():
    if request.method == 'POST':
        entry = request.form['entry']
        sentiment, suggestion = analyze_journal(entry)
        return render_template('journal.html', sentiment=sentiment, suggestion=suggestion, entry=entry)
    return render_template('journal.html')

def analyze_journal(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.05:
        sentiment = 'positive'
    elif compound <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    suggestions = {
        "positive": "Keep up the positive energy! 😊",
        "negative": "Try writing about what made you feel this way. 💬",
        "neutral": "Explore your thoughts more deeply next time. ✍️"
    }

    return sentiment, suggestions.get(sentiment)

def load_goals():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def save_goals(goals):
    with open(DATA_FILE, 'w') as f:
        json.dump(goals, f, indent=4)

@app.route('/goals', methods=['GET', 'POST'])
def goals():
    if request.method == 'POST':
        new_goal = request.form.get('goal')
        if new_goal:
            goals = load_goals()
            goals.append({
                "goal": new_goal,
                "created": datetime.today().strftime('%Y-%m-%d'),
                "streak": 0,
                "last_checked": ""
            })
            save_goals(goals)
            return redirect(url_for('goals'))
    
    goals = load_goals()
    return render_template('goals.html', goals=goals)

@app.route('/check_goal/<int:goal_index>')
def check_goal(goal_index):
    goals = load_goals()
    today = datetime.today().strftime('%Y-%m-%d')

    if goals[goal_index]["last_checked"] != today:
        goals[goal_index]["last_checked"] = today
        goals[goal_index]["streak"] += 1
        save_goals(goals)

    return redirect(url_for('goals'))

@app.route('/sound-therapy', methods=['GET', 'POST'])
def sound_therapy():
    mood = request.form.get('mood') if request.method == 'POST' else None

    mood_to_sound = {
        "relaxed": {
            "title": "Sunset Landscape",
            "file": "Sunset-Landscape(chosic.com).mp3"
        },
        "anxious": {
            "title": "White Petals",
            "file": "keys-of-moon-white-petals(chosic.com).mp3"
        },
        "sad": {
            "title": "Rainforest Sounds",
            "file": "Rain-Sound-and-Rainforest(chosic.com).mp3"
        },
        "tired": {
            "title": "Meditation",
            "file": "meditation.mp3"
        },
        "focus": {
            "title": "Magical Moments",
            "file": "Magical-Moments-chosic.com_.mp3"
        }
    }

    recommended = mood_to_sound.get(mood, None)

    # All available sounds (for browsing below)
    all_sounds = list(mood_to_sound.values())

    return render_template('sound_therapy.html', recommended=recommended, all_sounds=all_sounds)

def load_posts():
    if os.path.exists(POSTS_FILE):
        with open(POSTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_posts(posts):
    with open(POSTS_FILE, 'w') as f:
        json.dump(posts, f, indent=4)


@app.route('/community', methods=['GET', 'POST'])
def community_support():
    posts = load_posts()

    if request.method == 'POST':
        username = request.form['username']
        message = request.form['message']
        # Very basic AI reply simulation (you can plug in sentiment/local AI later)
        ai_response = "Thanks for sharing. You're not alone on this journey 🌟"

        posts.insert(0, {
            'username': username,
            'message': message,
            'reply': ai_response
        })

        save_posts(posts)
        return redirect(url_for('community_support'))

    return render_template('community_support.html', posts=posts)
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════



#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#Emotion based movie recommendation [SNEHA'S IMPLEMENTATION]
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Sample movie list
movie_data = [
    {"title": "Inception", "genres": "Action|Sci-Fi|Thriller"},
    {"title": "The Dark Knight", "genres": "Action|Crime|Drama"},
    {"title": "Titanic", "genres": "Drama|Romance"},
    {"title": "The Shawshank Redemption", "genres": "Drama"},
    {"title": "Avatar", "genres": "Action|Adventure|Fantasy"}
]

@app.route('/recommend', methods=['GET', 'POST'])
def home():
    mood = None
    recommendations = None
    
    if request.method == 'POST':
        mood = request.form['mood']
        recommendations = get_movie_recommendations(mood)

    return render_template('recommendations.html', mood=mood, recommendations=recommendations)

def get_movie_recommendations(mood):
    # Filter movies based on mood, for simplicity we just return all movies here
    # You can customize this logic to filter movies based on the mood
    return movie_data
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════



#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# API Routes - Proxy to microservices
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
@app.route('/api/camera/start', methods=['POST'])
def api_start_camera():
    """Proxy camera start to face microservice"""
    result = microservice_client.start_camera()
    return jsonify(result)

@app.route('/api/camera/stop', methods=['POST'])
def api_stop_camera():
    """Proxy camera stop to face microservice"""
    result = microservice_client.stop_camera()
    return jsonify(result)

@app.route('/api/camera/settings', methods=['POST'])
def api_camera_settings():
    """🎛️ Update visual settings for camera processing"""
    try:
        settings = request.get_json()
        print(f"🎛️ Updating visual settings: {settings}")
        
        # Forward settings to face microservice
        response = requests.post(f'{microservice_client.face_service_url}/api/settings',
                               json=settings, timeout=5)
        
        if response.status_code == 200:
            return jsonify({'success': True, 'message': 'Visual settings updated successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to update visual settings'}), 400
            
    except Exception as e:
        print(f"❌ Settings update error: {e}")
        return jsonify({'success': False, 'message': f'Settings error: {str(e)}'}), 500

# Also add direct API routes that match the microservice endpoints
@app.route('/api/start_camera', methods=['POST'])
def api_start_camera_direct():
    """Direct proxy to microservice start_camera"""
    result = microservice_client.start_camera()
    return jsonify(result)

@app.route('/api/stop_camera', methods=['POST'])
def api_stop_camera_direct():
    """Direct proxy to microservice stop_camera"""
    result = microservice_client.stop_camera()
    return jsonify(result)

@app.route('/api/emotions')
def api_get_emotions():
    """Proxy emotion data from face microservice"""
    result = microservice_client.get_emotions()
    return jsonify(result)

@app.route('/api/face_status')
def api_face_status():
    """Get face service status"""
    result = microservice_client.get_face_service_status()
    return jsonify(result)

@app.route('/api/mediapipe/landmarks')
def api_mediapipe_landmarks():
    """Proxy MediaPipe landmarks from face microservice"""
    try:
        response = requests.get(f'{FACE_MICROSERVICE_URL}/api/mediapipe/landmarks', timeout=5)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get landmarks: {str(e)}',
            'face_landmarks': [],
            'pose_landmarks': [],
            'hand_landmarks': [],
            'gaze_landmarks': {}
        })

# Text Microservice API Routes
@app.route('/api/text/analyze', methods=['POST'])
def api_analyze_text():
    """Proxy text analysis to text microservice"""
    data = request.get_json()
    result = microservice_client.analyze_text(data.get('text'), data.get('is_user', True))
    return jsonify(result)

@app.route('/api/text/chat', methods=['POST'])
def api_chat():
    """Proxy chat to text microservice"""
    data = request.get_json()
    # Forward Authorization header if present
    auth_header = request.headers.get('Authorization')
    # Forward session_id if present
    session_id = data.get('session_id')
    result = microservice_client.chat_with_bot(data.get('message'), auth_header, session_id)
    return jsonify(result)

@app.route('/api/text/conversation')
def api_text_conversation():
    """Get conversation history from text microservice"""
    result = microservice_client.get_text_conversation()
    return jsonify(result)

@app.route('/api/text_status')
def api_text_status():
    """Get text service status"""
    result = microservice_client.get_text_service_status()
    return jsonify(result)

# 🎓 User Learning API Endpoints
@app.route('/api/user_feedback', methods=['POST'])
def api_user_feedback():
    """🎓 Proxy user feedback to text microservice for learning"""
    try:
        data = request.get_json()
        response = requests.post(f'{microservice_client.text_service_url}/api/user_feedback',
                               json=data, timeout=10)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'User feedback error: {str(e)}'
        }), 500

@app.route('/api/learning_analytics')
def api_learning_analytics():
    """🎓 Get user learning analytics from text microservice"""
    try:
        user_id = request.args.get('user_id', 'default')
        response = requests.get(f'{microservice_client.text_service_url}/api/learning_analytics',
                              params={'user_id': user_id}, timeout=5)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Learning analytics error: {str(e)}'
        }), 500

@app.route('/api/emotion_suggestions')
def api_emotion_suggestions():
    """🎓 Get personalized emotion suggestions from text microservice"""
    try:
        text = request.args.get('text')
        predicted_emotion = request.args.get('predicted_emotion')
        user_id = request.args.get('user_id', 'default')
        
        response = requests.get(f'{microservice_client.text_service_url}/api/emotion_suggestions',
                              params={
                                  'text': text,
                                  'predicted_emotion': predicted_emotion,
                                  'user_id': user_id
                              }, timeout=5)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Emotion suggestions error: {str(e)}'
        }), 500

@app.route('/api/combined_emotions')
def api_combined_emotions():
    """Get combined emotions from both face and text analysis"""
    print(f"\n🔗 API CALL: /api/combined_emotions")
    result = microservice_client.get_combined_emotions()
    print(f"🔗 API RESULT: {result}")
    return jsonify(result)
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 🎵 EMOTION-BASED MUSIC RECOMMENDATION API
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
@app.route('/api/music/recommendations')
def api_music_recommendations():
    """🎵 Get emotion-based music recommendations for the carousel (100 songs for scrolling)"""
    try:
        session_id = request.args.get('session_id', 'default')
        limit = int(request.args.get('limit', 100))  # Default 100 for carousel
        minutes_back = int(request.args.get('minutes_back', 10))
        strategy = request.args.get('strategy', 'adaptive')
        
        # Import and use the unified emotion music system
        try:
            import sys
            sys.path.append('enhancements/src-new/multimodal_fusion')
            from unified_emotion_music_system import get_emotion_and_music
            print(f"🎵 Generating new music recommendations for {session_id}")
            result = get_emotion_and_music(session_id, minutes_back, strategy, limit)
            
            # 🚀 Get current emotion for cache key
            current_emotion = "unknown"
            if result and result.get('combined_emotion'):
                current_emotion = result['combined_emotion'].get('dominant_emotion', 'unknown')
            
            # Check cache with emotion-aware key
            cache_key = f"{session_id}_{minutes_back}_{strategy}_{limit}_{current_emotion}"
            current_time = datetime.now()
            
            if cache_key in music_recommendation_cache:
                cached_data, cached_time = music_recommendation_cache[cache_key]
                if (current_time - cached_time).total_seconds() < MUSIC_CACHE_DURATION:
                    print(f"💾 Using cached music recommendations for {current_emotion} (cached {(current_time - cached_time).total_seconds():.0f}s ago)")
                    return jsonify(cached_data)
            
            if result:
                recommendations = result.get('music_recommendations', [])
                
                # Format for API response - ensure we have all the fields the frontend needs
                formatted_recommendations = []
                for track in recommendations:
                    formatted_track = {
                        # Essential display info
                        'track_name': track.get('track_name', 'Unknown Track'),
                        'artist_name': track.get('artist_name', 'Unknown Artist'),
                        'album': track.get('album', 'Unknown Album'),
                        
                        # Metadata for UI
                        'track_popularity': track.get('track_popularity', 50),
                        'artist_popularity': track.get('artist_popularity', 50),
                        'emotion_target': track.get('emotion_target', 'neutral'),
                        'therapeutic_benefit': track.get('therapeutic_benefit', 'General Wellness'),
                        'musical_features': track.get('musical_features', 'Balanced'),
                        
                        # Audio features for advanced UI (if needed)
                        'audio_features': track.get('audio_features', {}),
                        
                        # Multi-emotion metadata
                        'emotion_source': track.get('emotion_source', 'single'),
                        'emotion_weight': track.get('emotion_weight', 1.0),
                        'source_emotion': track.get('source_emotion', result.get('emotion')),
                        'confidence_score': track.get('confidence_score', 0.5),
                        'recommendation_reason': track.get('recommendation_reason', 'Emotion-based match')
                    }
                    formatted_recommendations.append(formatted_track)
                
                api_result = {
                    'success': True,
                    'emotion': {
                        'dominant': result.get('emotion'),
                        'confidence': result.get('confidence'),
                        'is_multi_emotion': result.get('is_multi_emotion', False),
                        'top_emotions': result.get('top_emotions', []),
                        'fusion_weights': result.get('fusion_weights', {})
                    },
                    'recommendations': formatted_recommendations,
                    'metadata': {
                        'total_songs': len(formatted_recommendations),
                        'dataset_size': '3212',  # Your real dataset size
                        'session_id': session_id,
                        'processing_time_ms': result.get('processing_time_ms', 0),
                        'timestamp': result.get('timestamp'),
                        'update_interval': 30  # Tell frontend to update every 30 seconds
                    }
                }
                
                # 🚀 Cache the successful result
                music_recommendation_cache[cache_key] = (api_result, current_time)
                print(f"💾 Music recommendations cached for {MUSIC_CACHE_DURATION}s")
                
                return jsonify(api_result)
            else:
                return jsonify({
                    'success': False,
                    'error': 'No emotion detected - please ensure face/text microservices are running',
                    'recommendations': [],
                    'metadata': {
                        'session_id': session_id,
                        'timestamp': datetime.now().isoformat()
                    }
                })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'Music recommendation system not available',
                'recommendations': []
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Music recommendation error: {str(e)}',
            'recommendations': []
        }), 500
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
@app.route('/api/video_feed')
def api_video_feed():
    """Proxy video feed from face microservice"""
    try:
        # Stream video from microservice
        response = requests.get(f'{FACE_MICROSERVICE_URL}/video_feed', stream=True, timeout=30)
        return Response(
            response.iter_content(chunk_size=1024),
            content_type=response.headers.get('content-type', 'multipart/x-mixed-replace; boundary=frame')
        )
    except Exception as e:
        return jsonify({'error': f'Video feed error: {str(e)}'}), 500
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    face_service_health = microservice_client.check_face_service_health()
    text_service_health = microservice_client.check_text_service_health()
    
    return jsonify({
        'status': 'healthy',
        'service': 'Y.M.I.R AI Main App',
        'version': '1.0.0',
        'microservices': {
            'face_emotion_detection': {
                'url': FACE_MICROSERVICE_URL,
                'healthy': face_service_health
            },
            'text_emotion_analysis': {
                'url': TEXT_MICROSERVICE_URL,
                'healthy': text_service_health
            }
        }
    })
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# ===== MUSIC PLAYER API ENDPOINTS =====
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
import re
import os
import time

@app.route('/api/get_audio')
def api_get_audio():
    """Get audio URL for a song using YouTube to MP3 conversion"""
    song = request.args.get('song')
    artist = request.args.get('artist')
    
    if not song or not artist:
        return jsonify({
            'error': 'Missing song or artist parameter'
        }), 400
    
    try:
        # Method 1: Try YouTube to MP3 API first
        audio_url = get_audio_from_youtube_api(song, artist)
        if audio_url:
            return jsonify({
                'success': True,
                'audio_url': audio_url,
                'source': 'youtube_api',
                'song': song,
                'artist': artist
            })
        
        # Method 2: Try direct YouTube search with yt-dlp
        audio_url = get_audio_from_youtube_search(song, artist)
        if audio_url:
            return jsonify({
                'success': True,
                'audio_url': audio_url,
                'source': 'youtube_search',
                'song': song,
                'artist': artist
            })
        
        return jsonify({
            'success': False,
            'error': 'Unable to find audio on YouTube',
            'song': song,
            'artist': artist
        }), 404
        
    except Exception as e:
        print(f"❌ Audio fetch error: {e}")
        return jsonify({
            'error': f'Failed to fetch audio: {str(e)}'
        }), 500

def get_audio_from_youtube_api(song, artist):
    """Get audio using YouTube to MP3 API with real YouTube search"""
    try:
        # Search for the song on YouTube
        search_query = f"{artist} {song} official audio"
        youtube_url = search_youtube_url_real(search_query)
        
        if not youtube_url:
            return None
        
        print(f"🔍 Found YouTube URL: {youtube_url}")
        
        # Convert using your Railway API
        api_url = 'https://yt-mp3-server-production.up.railway.app/api/convert'
        
        response = requests.post(api_url, 
                               json={'url': youtube_url},
                               headers={'Content-Type': 'application/json'},
                               timeout=60)  # Increased timeout
        
        if response.ok:
            # Create a blob URL from the response
            blob_data = response.content
            
            # Save temporarily and serve via Flask
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_file.write(blob_data)
            temp_file.close()
            
            # Move to static directory for serving
            import shutil
            filename = f"{clean_filename(artist)}_{clean_filename(song)}.mp3"
            static_path = os.path.join('static', 'music', filename)
            os.makedirs(os.path.dirname(static_path), exist_ok=True)
            shutil.move(temp_file.name, static_path)
            
            # Return the Flask URL
            from flask import url_for
            return url_for('static', filename=f'music/{filename}', _external=True)
        
    except Exception as e:
        print(f"⚠️ YouTube API error: {e}")
        
    return None

def search_youtube_url_real(query):
    """Search YouTube and return the best video URL using yt-dlp"""
    try:
        try:
            import yt_dlp
        except ImportError:
            print("⚠️ yt-dlp not available, using fallback URLs")
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': False,
            'ignoreerrors': True,
            'noplaylist': True,
            'no_warnings': True,
            'format': 'bestaudio[ext=m4a]/bestaudio/best[height<=720]/best',  # Optimal format from debug
            'socket_timeout': 30,
            'retries': 3,
            'extractor_args': {
                'youtube': {
                    'skip': ['dash']  # Skip DASH formats that cause issues
                }
            }
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Search for top 3 results
            results = ydl.extract_info(f"ytsearch3:{query}", download=False)
            
            if results and results.get('entries'):
                for entry in results['entries']:
                    if entry and entry.get('id'):
                        title = entry.get('title', '').lower()
                        if 'official' in title or 'audio' in title:
                            return f"https://www.youtube.com/watch?v={entry['id']}"
                
                # Fallback to first result
                first_entry = results['entries'][0]
                if first_entry and first_entry.get('id'):
                    return f"https://www.youtube.com/watch?v={first_entry['id']}"
        
    except Exception as e:
        print(f"⚠️ YouTube search error: {e}")
        



def get_audio_from_youtube_search(song, artist):
    """Alternative method using yt-dlp direct download"""
    try:
        try:
            import yt_dlp
        except ImportError:
            return None
        
        filename = f"{clean_filename(artist)}_{clean_filename(song)}.mp3"
        output_path = os.path.join('static', 'music', filename)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        queries = [
            f"{artist} - {song} official audio",
            f"{artist} {song} official",
            f"{song} by {artist} audio"
        ]
        
        for query in queries:
            try:
                ydl_opts = {
                    'format': 'bestaudio[ext=m4a]/bestaudio/best[height<=720]/best',
                    'outtmpl': os.path.splitext(output_path)[0],
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '128',
                    }],
                    'quiet': True,
                    'ignoreerrors': True,
                    'no_warnings': True,
                    'retries': 3,
                    'socket_timeout': 30,
                    'extractor_args': {
                        'youtube': {
                            'skip': ['dash']
                        }
                    }
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([f"ytsearch1:{query}"])
                
                if os.path.exists(output_path):
                    from flask import url_for
                    return url_for('static', filename=f'music/{filename}', _external=True)
                    
            except Exception as e:
                print(f"⚠️ Query '{query}' failed: {e}")
                continue
        
    except Exception as e:
        print(f"⚠️ YouTube search download error: {e}")
        
    return None

def clean_filename(filename):
    """Clean filename for filesystem compatibility"""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

@app.route('/api/check_local_music')
def api_check_local_music():
    """Check if a song is already cached locally"""
    song = request.args.get('song')
    artist = request.args.get('artist')
    
    if not song or not artist:
        return jsonify({'error': 'Missing parameters'}), 400
    
    try:
        filename = f"{clean_filename(artist)}_{clean_filename(song)}.mp3"
        local_path = os.path.join('static', 'music', filename)
        
        if os.path.exists(local_path):
            from flask import url_for
            return jsonify({
                'success': True,
                'cached': True,
                'audio_url': url_for('static', filename=f'music/{filename}', _external=True),
                'source': 'local_cache'
            })
        else:
            return jsonify({
                'success': True,
                'cached': False
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_album_art')
def api_get_album_art():
    """Get album art for a song using iTunes API"""
    song = request.args.get('song')
    artist = request.args.get('artist')
    
    if not song or not artist:
        return jsonify({'error': 'Missing parameters'}), 400
    
    try:
        # Try iTunes API (free)
        search_term = f"{artist} {song}".replace(' ', '+')
        itunes_url = f"https://itunes.apple.com/search?term={search_term}&media=music&limit=1"
        
        response = requests.get(itunes_url, timeout=10)
        if response.ok:
            data = response.json()
            if data.get('results') and len(data['results']) > 0:
                # Get the largest artwork available
                artwork_url = data['results'][0].get('artworkUrl100', '')
                if artwork_url:
                    # Upgrade to higher resolution
                    artwork_url = artwork_url.replace('100x100', '500x500')
                    return jsonify({
                        'success': True,
                        'image_url': artwork_url,
                        'source': 'itunes',
                        'song': song,
                        'artist': artist
                    })
        
        # Fallback to placeholder
        return jsonify({
            'success': True,
            'image_url': 'https://via.placeholder.com/400x400/6A5ACD/FFFFFF/png?text=🎵',
            'source': 'placeholder',
            'song': song,
            'artist': artist
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════


#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Add Firebase Authentication routes
if FIREBASE_AUTH_AVAILABLE:
    add_auth_routes(app)
    print("✅ Firebase Authentication routes added")

if __name__ == '__main__':
    print("🚀 Starting Y.M.I.R AI Emotion Detection System...")
    print("📍 Home page: http://localhost:5000")
    print("🔧 AI App: http://localhost:5000/ai_app")
    
    if FIREBASE_AUTH_AVAILABLE:
        print("🔐 Authentication: Firebase Auth enabled")
    else:
        print("⚠️ Authentication: Running without Firebase Auth")
    
    # 🚀 PRODUCTION MODE: Disable debug to prevent auto-restart crashes
    app.run(
        debug=False,
        host='localhost',
        port=5000,
        threaded=True
    )
#════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════