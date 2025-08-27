"""
WEB ROUTES & USER INTERFACE MICROSERVICE
Extracted from app.py lines 1007-1440, 1791-1952
Contains all Flask routes and user interface functions
"""
import json
import os
from datetime import datetime
from flask import render_template, request, jsonify, send_from_directory

class WebRoutesService:
    def __init__(self):
        # User favorites storage
        self.FAVORITES_FILE = "favorites.json"
        print("✅ Web Routes & UI Service initialized")

    # EXTRACTED FROM app.py:1007 (home1 function - duplicate of home)
    def render_home1(self):
        """Alternative home page rendering"""
        return render_template('home.html')

    # EXTRACTED FROM app.py:1011-1014 (EXACT COPY)
    def render_ai_app(self):
        """Main AI app interface"""
        return render_template('index.html')

    # EXTRACTED FROM app.py:1016-1018 (EXACT COPY)
    def render_about(self):
        """About page"""
        return render_template('about.html')

    # EXTRACTED FROM app.py:1069-1071 (EXACT COPY)
    def render_meditation(self):
        """Meditation interface"""
        return render_template('meditation.html')

    # EXTRACTED FROM app.py:1090-1095 (EXACT COPY)
    def render_journal(self):
        """Journal interface"""
        return render_template('journal.html')

    # EXTRACTED FROM app.py:1117-1122 (EXACT COPY)
    def render_breathing(self):
        """Breathing exercises interface"""
        return render_template('breathing.html')

    # EXTRACTED FROM app.py:1145-1161 (EXACT COPY)
    def render_goals(self):
        """Goals tracking interface"""
        return render_template('goals.html')

    # EXTRACTED FROM app.py:1175-1206 (EXACT COPY)
    def render_sound_therapy(self):
        """Sound therapy interface"""
        return render_template('sound_therapy.html')

    # EXTRACTED FROM app.py:1219-1247 (EXACT COPY)
    def render_community_support(self):
        """Community support interface"""
        return render_template('community_support.html')

    # EXTRACTED FROM app.py:1249-1257 (EXACT COPY)
    def render_home(self):
        """Main home page"""
        return render_template('home.html')

    # EXTRACTED FROM app.py:1265-1359 (EXACT COPY)
    def render_contact(self):
        """Contact page"""
        return render_template('contact.html')

    # EXTRACTED FROM app.py:1408-1410 (EXACT COPY)
    def render_features(self):
        """Features page"""
        return render_template('features.html')

    # EXTRACTED FROM app.py:1412-1414 (EXACT COPY)
    def render_cookies(self):
        """Cookie policy page"""
        return render_template('cookiepolicy.html')

    # EXTRACTED FROM app.py:1416-1418 (EXACT COPY)
    def render_services(self):
        """Services page"""
        return render_template('services.html')

    # EXTRACTED FROM app.py:1420-1422 (EXACT COPY)
    def render_pricing(self):
        """Pricing page"""
        return render_template('pricing.html')

    # EXTRACTED FROM app.py:1424-1426 (EXACT COPY)
    def render_privacy(self):
        """Privacy policy page"""
        return render_template('privacy.html')

    # EXTRACTED FROM app.py:1428-1430 (EXACT COPY)
    def render_wellness(self):
        """Wellness tools page"""
        return render_template('wellness_tools.html')

    # EXTRACTED FROM app.py:1432-1434 (EXACT COPY)
    def render_gaming(self):
        """Gaming/gamification page"""
        return render_template('gaming.html')

    # EXTRACTED FROM app.py:1436-1438 (EXACT COPY)
    def render_emotion_history(self):
        """Emotion history page"""
        return render_template('emotion_timeline.html')

    # EXTRACTED FROM app.py:1440-1442 (EXACT COPY)
    def render_mood_transition(self):
        """Mood transition analysis page"""
        return render_template('mood_transition.html')

    # EXTRACTED FROM app.py:1927-1951 (EXACT COPY)
    def render_emotion_timeline(self):
        """Render emotion timeline with data"""
        try:
            # Load emotion data from JSON files
            emotion_data = []
            
            # Try to load emotion log
            try:
                with open("emotion_log.json", "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "log" in data:
                        emotion_data.extend(data["log"])
                    elif isinstance(data, list):
                        emotion_data.extend(data)
            except FileNotFoundError:
                pass
            
            # Try to load chat results
            try:
                with open("chat_results.json", "r") as f:
                    chat_data = json.load(f)
                    if "conversation" in chat_data:
                        for entry in chat_data["conversation"]:
                            emotion_entry = {
                                "timestamp": chat_data.get("timestamp", ""),
                                "emotions": {chat_data.get("dominant_emotion", "neutral"): 100},
                                "source": "chat"
                            }
                            emotion_data.append(emotion_entry)
            except FileNotFoundError:
                pass
            
            # Sort by timestamp
            emotion_data.sort(key=lambda x: x.get("timestamp", ""))
            
            return render_template('emotion_timeline.html', emotion_data=emotion_data)
            
        except Exception as e:
            print(f"Error loading emotion timeline: {e}")
            return render_template('emotion_timeline.html', emotion_data=[])

    # EXTRACTED FROM app.py:1445-1469 (EXACT COPY)
    def handle_save_favorite(self, request_data):
        """Save a favorite song"""
        try:
            data = request_data.get_json()
            
            if not data or 'track' not in data or 'artist' not in data:
                return {"error": "Missing track or artist"}, 400
            
            # Load existing favorites
            try:
                with open(self.FAVORITES_FILE, 'r') as f:
                    favorites = json.load(f)
            except FileNotFoundError:
                favorites = []
            
            # Create new favorite entry
            new_favorite = {
                "track": data['track'],
                "artist": data['artist'],
                "mood": data.get('mood', 'Unknown'),
                "timestamp": datetime.now().isoformat(),
                "id": len(favorites) + 1
            }
            
            # Check if already exists
            for fav in favorites:
                if fav['track'] == new_favorite['track'] and fav['artist'] == new_favorite['artist']:
                    return {"message": "Song already in favorites"}, 200
            
            favorites.append(new_favorite)
            
            # Save back to file
            with open(self.FAVORITES_FILE, 'w') as f:
                json.dump(favorites, f, indent=2)
            
            return {"message": "Added to favorites successfully"}, 200
            
        except Exception as e:
            return {"error": str(e)}, 500

    # EXTRACTED FROM app.py:1471-1476 (EXACT COPY)
    def get_favorites(self):
        """Get user's favorite songs"""
        try:
            with open(self.FAVORITES_FILE, 'r') as f:
                favorites = json.load(f)
            return favorites
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error loading favorites: {e}")
            return []

    # EXTRACTED FROM app.py:1496-1516 (EXACT COPY)
    def handle_remove_favorite(self, request_data):
        """Remove a song from favorites"""
        try:
            data = request_data.get_json()
            
            if not data or 'track' not in data or 'artist' not in data:
                return {"error": "Missing track or artist"}, 400
            
            # Load favorites
            try:
                with open(self.FAVORITES_FILE, 'r') as f:
                    favorites = json.load(f)
            except FileNotFoundError:
                return {"message": "No favorites found"}, 404
            
            # Find and remove the favorite
            original_length = len(favorites)
            favorites = [fav for fav in favorites if not (fav['track'] == data['track'] and fav['artist'] == data['artist'])]
            
            if len(favorites) == original_length:
                return {"message": "Song not found in favorites"}, 404
            
            # Save back to file
            with open(self.FAVORITES_FILE, 'w') as f:
                json.dump(favorites, f, indent=2)
            
            return {"message": "Removed from favorites successfully"}, 200
            
        except Exception as e:
            return {"error": str(e)}, 500

    def get_user_stats(self):
        """Get user statistics for dashboard"""
        try:
            stats = {
                "total_favorites": len(self.get_favorites()),
                "emotion_logs": 0,
                "chat_sessions": 0,
                "goals_completed": 0
            }
            
            # Count emotion logs
            try:
                with open("emotion_log.json", "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "log" in data:
                        stats["emotion_logs"] = len(data["log"])
                    elif isinstance(data, list):
                        stats["emotion_logs"] = len(data)
            except FileNotFoundError:
                pass
            
            # Count chat sessions
            try:
                with open("chat_results.json", "r") as f:
                    data = json.load(f)
                    if "conversation" in data:
                        stats["chat_sessions"] = len(data["conversation"])
            except FileNotFoundError:
                pass
            
            # Count completed goals
            try:
                with open("goals.json", "r") as f:
                    goals = json.load(f)
                    stats["goals_completed"] = sum(1 for goal in goals if goal.get("completed", False))
            except FileNotFoundError:
                pass
            
            return stats
            
        except Exception as e:
            print(f"Error getting user stats: {e}")
            return {
                "total_favorites": 0,
                "emotion_logs": 0, 
                "chat_sessions": 0,
                "goals_completed": 0
            }

    def handle_dashboard_data(self):
        """Get data for user dashboard"""
        try:
            dashboard_data = {
                "user_stats": self.get_user_stats(),
                "recent_favorites": self.get_favorites()[-5:],  # Last 5 favorites
                "recent_emotions": [],
                "wellness_summary": {}
            }
            
            # Get recent emotion data
            try:
                with open("emotion_log.json", "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "log" in data:
                        dashboard_data["recent_emotions"] = data["log"][-10:]  # Last 10 entries
            except FileNotFoundError:
                pass
            
            return dashboard_data
            
        except Exception as e:
            print(f"Error getting dashboard data: {e}")
            return {"user_stats": {}, "recent_favorites": [], "recent_emotions": []}

    def serve_static_file(self, filename):
        """Serve static files (audio, images, etc.)"""
        try:
            # Determine the directory based on file extension
            if filename.endswith('.mp3') or filename.endswith('.wav'):
                directory = "static/audio"
            elif filename.endswith(('.jpg', '.png', '.gif', '.jpeg')):
                directory = "static/images"
            else:
                directory = "static"
            
            return send_from_directory(directory, filename)
            
        except FileNotFoundError:
            return {"error": "File not found"}, 404
        except Exception as e:
            return {"error": str(e)}, 500

    def cleanup(self):
        """Cleanup service resources"""
        print("🧹 Web Routes & UI service cleanup complete")