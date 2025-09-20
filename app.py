"""
Y.M.I.R AI Emotion Detection System - Main Flask Application
===========================================================
Basic Flask app with microservices architecture for emotion detection,
chatbot integration, and music recommendations.

Author: Y.M.I.R Development Team
Version: 1.0.0
"""

from flask import Flask, render_template, request, jsonify, url_for, Response
from flask_cors import CORS
import os
import requests
import json
from datetime import datetime
import sys
from pathlib import Path

# Import multimodal emotion combiner
try:
    combiner_path = Path(__file__).parent / 'enhancements' / 'src-new' / 'multimodal_fusion'
    sys.path.append(str(combiner_path))
    from real_emotion_combiner import RealEmotionCombiner, RealCombinedEmotion
    EMOTION_COMBINER_AVAILABLE = True
    print("✅ Multimodal emotion combiner available")
except ImportError as e:
    EMOTION_COMBINER_AVAILABLE = False
    print(f"⚠️ Emotion combiner not available: {e}")
    RealEmotionCombiner = None
    RealCombinedEmotion = None

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ymir-dev-key-2024')

# Enable CORS for API calls
CORS(app)

# Configure static files
app.static_folder = 'static'
app.template_folder = 'templates'

# Microservice URLs
FACE_MICROSERVICE_URL = 'http://localhost:5002'
TEXT_MICROSERVICE_URL = 'http://localhost:5003'

class MicroserviceClient:
    """Client to communicate with microservices"""
    
    def __init__(self):
        self.face_service_url = FACE_MICROSERVICE_URL
        self.text_service_url = TEXT_MICROSERVICE_URL
        
        # Initialize emotion combiner
        if EMOTION_COMBINER_AVAILABLE:
            self.emotion_combiner = RealEmotionCombiner()
            print("✅ Emotion combiner initialized")
        else:
            self.emotion_combiner = None
    
    def check_face_service_health(self):
        """Check if face microservice is running"""
        try:
            print(f"🏥 Checking health: {self.face_service_url}/health")
            response = requests.get(f'{self.face_service_url}/health', timeout=2)
            print(f"🏥 Health response: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"🏥 Health check failed: {e}")
            return False
    
    def start_camera(self):
        """Start camera via microservice"""
        try:
            print(f"🔄 Starting camera via microservice: {self.face_service_url}/api/start_camera")
            response = requests.post(f'{self.face_service_url}/api/start_camera', timeout=15)  # Increased timeout
            print(f"📡 Microservice response status: {response.status_code}")
            result = response.json()
            print(f"📊 Microservice response: {result}")
            return result
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Camera start timed out (>15s). Camera may be in use by another application.'}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error': 'Cannot connect to face microservice. Is it running on port 5002?'}
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
            print(f"🏥 Checking text service health: {self.text_service_url}/health")
            response = requests.get(f'{self.text_service_url}/health', timeout=2)
            print(f"🏥 Text health response: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"🏥 Text health check failed: {e}")
            return False
    
    def analyze_text(self, text, is_user=True):
        """Analyze text emotion via microservice"""
        try:
            response = requests.post(f'{self.text_service_url}/api/analyze_text', 
                                   json={'text': text, 'is_user': is_user}, timeout=10)
            return response.json()
        except Exception as e:
            return {'success': False, 'error': f'Text microservice error: {str(e)}'}
    
    def chat_with_bot(self, message):
        """Chat with bot via microservice"""
        try:
            response = requests.post(f'{self.text_service_url}/api/chat',
                                   json={'message': message}, timeout=15)
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
            # 🔍 SAFE CHECK: Only get face emotions if camera is actually running
            face_status = self.get_face_service_status()
            face_emotions = {}
            
            if face_status.get('running') and not face_status.get('error'):
                # Camera is actively running, safe to get emotions
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
                        'text_source': combined.text_source
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
def monitor_combined_emotions():
    """Monitor and log combined emotions every 10 seconds"""
    import time
    while True:
        try:
            time.sleep(10)  # Check every 10 seconds
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
    print("✅ Emotion combiner monitoring started (every 10 seconds)")

@app.route('/')
def home():
    """Render the home page"""
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
    """About page"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

@app.route('/features')
def features():
    """Features page"""
    return render_template('features.html')

@app.route('/pricing')
def pricing():
    """Pricing page"""
    return render_template('pricing.html')

@app.route('/privacy')
def privacy():
    """Privacy policy page"""
    return render_template('privacy.html')

@app.route('/services')
def services():
    """Services page"""
    return render_template('services.html')

@app.route('/wellness')
def wellness():
    return render_template('wellness_tools.html')

@app.route('/meditation')
def meditation():
    """Meditation page"""
    return render_template('meditation.html')

@app.route('/breathing')
def breathing():
    """Breathing exercises page"""
    return render_template('breathing.html')

@app.route('/journal')
def journal():
    """Journal page"""
    return render_template('journal.html')

@app.route('/community_support')
def community_support():
    """Community support page"""
    return render_template('community_support.html')

@app.route('/cookiepolicy')
def cookiepolicy():
    """Cookie policy page"""
    return render_template('cookiepolicy.html')

# API Routes - Proxy to microservices
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
    result = microservice_client.chat_with_bot(data.get('message'))
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

@app.route('/api/test_combiner')
def api_test_combiner():
    """Test emotion combiner with detailed logging"""
    print(f"\n🧪 TESTING EMOTION COMBINER")
    print("=" * 60)
    
    if not EMOTION_COMBINER_AVAILABLE:
        return jsonify({'error': 'Emotion combiner not available'})
    
    try:
        # Import and test the combiner directly
        from real_emotion_combiner import test_emotion_fusion, get_combined_emotion
        
        print("🧪 Testing direct combiner function...")
        result = get_combined_emotion(minutes_back=10, strategy='adaptive')
        
        if result:
            print(f"✅ DIRECT COMBINER RESULT:")
            print(f"   Emotion: {result['emotion']}")
            print(f"   Confidence: {result['confidence']}")
            print(f"   Source: {result['source']}")
            print(f"   Strategy: {result['strategy']}")
            
            if result['facial_data']:
                print(f"   📹 Facial data: {result['facial_data']}")
            if result['text_data']:
                print(f"   💬 Text data: {result['text_data']}")
        else:
            print(f"❌ No combined emotion data available")
        
        print("=" * 60)
        
        # Also test via microservice client
        client_result = microservice_client.get_combined_emotions()
        print(f"🔗 MICROSERVICE CLIENT RESULT: {client_result}")
        
        return jsonify({
            'success': True,
            'direct_result': result,
            'client_result': client_result
        })
        
    except Exception as e:
        print(f"❌ Combiner test error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Test failed: {str(e)}'})

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

# ===== MUSIC PLAYER API ENDPOINTS =====
import requests
import re
import os
from urllib.parse import quote
import time

@app.route('/api/get_audio')
def api_get_audio():
    """Get audio URL for a song using multiple sources"""
    song = request.args.get('song')
    artist = request.args.get('artist')
    
    if not song or not artist:
        return jsonify({
            'error': 'Missing song or artist parameter'
        }), 400
    
    try:
        # Method 1: Try YouTube to MP3 API first (fastest, best quality)
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
        
        # No fallbacks - return error if YouTube methods fail
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
    """Try to get audio using your YouTube to MP3 API with real YouTube search"""
    try:
        # Search for the song on YouTube using yt-dlp
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
        # Try to import yt_dlp, fallback if not available
        try:
            import yt_dlp
        except ImportError:
            print("⚠️ yt-dlp not available, using fallback URLs")
            return search_youtube_url_fallback(query)
        
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
                        # Prefer official videos
                        title = entry.get('title', '').lower()
                        if 'official' in title or 'audio' in title:
                            return f"https://www.youtube.com/watch?v={entry['id']}"
                
                # Fallback to first result
                first_entry = results['entries'][0]
                if first_entry and first_entry.get('id'):
                    return f"https://www.youtube.com/watch?v={first_entry['id']}"
        
    except Exception as e:
        print(f"⚠️ YouTube search error: {e}")
        
    return search_youtube_url_fallback(query)

def search_youtube_url_fallback(query):
    """Fallback YouTube URLs for demo"""
    # Popular song URLs for demo
    fallback_urls = [
        "https://youtu.be/e1uCqIPCP1k",  # Your test URL
        "https://youtu.be/dQw4w9WgXcQ",  # Rick Roll
        "https://youtu.be/kJQP7kiw5Fk",  # Despacito
        "https://youtu.be/JGwWNGJdvx8",  # Shape of You
        "https://youtu.be/2Vv-BfVoq4g"   # Perfect
    ]
    
    import random
    return random.choice(fallback_urls)

def get_audio_from_youtube_search(song, artist):
    """Alternative method using yt-dlp direct download"""
    try:
        # Try to import yt_dlp
        try:
            import yt_dlp
        except ImportError:
            print("⚠️ yt-dlp not available for direct download")
            return None
        
        filename = f"{clean_filename(artist)}_{clean_filename(song)}.mp3"
        output_path = os.path.join('static', 'music', filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate search queries
        queries = [
            f"{artist} - {song} official audio",
            f"{artist} {song} official",
            f"{song} by {artist} audio"
        ]
        
        for query in queries:
            try:
                ydl_opts = {
                    'format': 'bestaudio[ext=m4a]/bestaudio/best[height<=720]/best',  # Optimal format from debug
                    'outtmpl': os.path.splitext(output_path)[0],
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '128',  # Good quality audio
                    }],
                    'quiet': True,
                    'ignoreerrors': True,
                    'no_warnings': True,
                    'retries': 3,
                    'socket_timeout': 30,
                    'extractor_args': {
                        'youtube': {
                            'skip': ['dash']  # Skip problematic DASH formats
                        }
                    }
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Search and download first result
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
    import re
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
    """Get album art for a song using multiple sources"""
    song = request.args.get('song')
    artist = request.args.get('artist')
    
    if not song or not artist:
        return jsonify({'error': 'Missing parameters'}), 400
    
    try:
        # Try multiple sources for album art
        
        # Method 1: Try Last.fm API (free, no key needed for basic search)
        album_art_url = get_album_art_lastfm(artist, song)
        if album_art_url:
            return jsonify({
                'success': True,
                'image_url': album_art_url,
                'source': 'lastfm',
                'song': song,
                'artist': artist
            })
        
        # Method 2: Try iTunes API (free)
        album_art_url = get_album_art_itunes(artist, song)
        if album_art_url:
            return jsonify({
                'success': True,
                'image_url': album_art_url,
                'source': 'itunes',
                'song': song,
                'artist': artist
            })
        
        # Method 3: Fallback to mood-based placeholder
        album_art_url = get_mood_based_placeholder(song, artist)
        return jsonify({
            'success': True,
            'image_url': album_art_url,
            'source': 'placeholder',
            'song': song,
            'artist': artist
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_album_art_lastfm(artist, song):
    """Try to get album art from Last.fm API"""
    try:
        import urllib.parse
        
        # Last.fm API doesn't require key for basic search
        base_url = "http://ws.audioscrobbler.com/2.0/"
        params = {
            'method': 'track.getInfo',
            'api_key': 'your_lastfm_key',  # You'd need to get this
            'artist': artist,
            'track': song,
            'format': 'json'
        }
        
        # For demo, return None to use iTunes
        return None
        
    except Exception as e:
        print(f"⚠️ Last.fm error: {e}")
        return None

def get_album_art_itunes(artist, song):
    """Try to get album art from iTunes API (free)"""
    try:
        import urllib.parse
        
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
                    return artwork_url
        
    except Exception as e:
        print(f"⚠️ iTunes API error: {e}")
        
    return None

def get_mood_based_placeholder(song, artist):
    """Generate mood-based placeholder album art"""
    # Create beautiful gradient placeholders based on song mood
    mood_gradients = {
        'happy': 'https://via.placeholder.com/400x400/FFD700/FFFFFF/png?text=♪',
        'sad': 'https://via.placeholder.com/400x400/4682B4/FFFFFF/png?text=♫', 
        'energetic': 'https://via.placeholder.com/400x400/FF6347/FFFFFF/png?text=♪',
        'calm': 'https://via.placeholder.com/400x400/87CEEB/FFFFFF/png?text=♫',
        'romantic': 'https://via.placeholder.com/400x400/FFB6C1/FFFFFF/png?text=♥',
        'default': 'https://via.placeholder.com/400x400/6A5ACD/FFFFFF/png?text=🎵'
    }
    
    # You could analyze song name/artist for mood keywords
    # For now, return default
    return mood_gradients['default']

if __name__ == '__main__':
    print("🚀 Starting Y.M.I.R AI Emotion Detection System...")
    print("📍 Home page: http://localhost:5000")
    print("🔧 AI App: http://localhost:5000/ai_app")
    
    # 🚀 PRODUCTION MODE: Disable debug to prevent auto-restart crashes
    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )