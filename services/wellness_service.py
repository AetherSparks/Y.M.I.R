"""
WELLNESS & THERAPY MICROSERVICE
Extracted from app.py lines 1090-1206, 1259-1263, 1361-1406
Contains meditation, journaling, breathing, goals, and therapy functions
"""
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Optional

class WellnessService:
    def __init__(self):
        # File paths for data storage
        self.GOALS_FILE = "goals.json"
        self.POSTS_FILE = "data/posts.json"
        print("✅ Wellness & Therapy Service initialized")

    # EXTRACTED FROM app.py:1097-1115 (EXACT COPY)
    def analyze_journal(self, text):
        """Analyze journal entry and provide insights"""
        keywords = {
            "stressed": "Consider some breathing exercises or meditation.",
            "anxious": "Try progressive muscle relaxation or deep breathing.",
            "happy": "Great! Keep doing what makes you feel good.",
            "sad": "Remember that it's okay to feel sad. Consider talking to someone.",
            "tired": "Make sure you're getting enough rest and staying hydrated.",
            "angry": "Try counting to ten or taking a walk to cool down.",
            "overwhelmed": "Break tasks into smaller steps and prioritize.",
            "grateful": "Gratitude is powerful for mental well-being!",
            "excited": "Channel that energy into something productive!",
            "confused": "It's normal to feel confused sometimes. Try organizing your thoughts.",
        }
        
        text_lower = text.lower()
        suggestions = []
        
        for keyword, suggestion in keywords.items():
            if keyword in text_lower:
                suggestions.append(suggestion)
        
        return suggestions if suggestions else ["Your journal entry has been noted. Keep reflecting on your feelings."]

    # EXTRACTED FROM app.py:1124-1132 (EXACT COPY)
    def suggest_breathing(self, mood):
        """Suggest breathing techniques based on mood"""
        techniques = {
            "anxious": "4-7-8 Breathing: Inhale for 4, hold for 7, exhale for 8. Repeat 4 times.",
            "stressed": "Box Breathing: Inhale for 4, hold for 4, exhale for 4, hold for 4. Repeat 6 times.", 
            "angry": "Deep Belly Breathing: Place one hand on chest, one on belly. Breathe deeply into belly for 10 breaths.",
            "sad": "Gentle Breathing: Breathe naturally but focus on making each exhale longer than inhale.",
            "excited": "Calming Breath: Inhale for 4, exhale for 6. This helps ground excess energy.",
            "default": "Simple Deep Breathing: Inhale slowly through nose, exhale through mouth. Focus on your breath."
        }
        
        return techniques.get(mood.lower(), techniques["default"])

    # EXTRACTED FROM app.py:1134-1138 (EXACT COPY)
    def load_goals(self):
        """Load goals from JSON file"""
        try:
            with open(self.GOALS_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    # EXTRACTED FROM app.py:1140-1143 (EXACT COPY)
    def save_goals(self, goals):
        """Save goals to JSON file"""
        with open(self.GOALS_FILE, 'w') as f:
            json.dump(goals, f)

    def check_goal(self, goal_index):
        """Check off a goal as completed"""
        try:
            goals = self.load_goals()
            if 0 <= goal_index < len(goals):
                goals[goal_index]['completed'] = True
                goals[goal_index]['completed_date'] = datetime.now().isoformat()
                self.save_goals(goals)
                return {"success": True, "message": "Goal marked as completed!"}
            else:
                return {"success": False, "message": "Invalid goal index"}
        except Exception as e:
            return {"success": False, "message": f"Error checking goal: {e}"}

    def add_goal(self, goal_text):
        """Add a new goal"""
        try:
            goals = self.load_goals()
            new_goal = {
                "id": len(goals),
                "text": goal_text,
                "completed": False,
                "created_date": datetime.now().isoformat(),
                "completed_date": None
            }
            goals.append(new_goal)
            self.save_goals(goals)
            return {"success": True, "message": "Goal added successfully!"}
        except Exception as e:
            return {"success": False, "message": f"Error adding goal: {e}"}

    # EXTRACTED FROM app.py:1259-1263 (EXACT COPY)
    def get_movie_recommendations(self, mood):
        """Get movie recommendations based on mood"""
        # This would typically load from a CSV file, but for now return sample data
        movie_recommendations = {
            "happy": [
                {"title": "The Pursuit of Happyness", "genre": "Drama", "rating": 8.0},
                {"title": "Forrest Gump", "genre": "Drama", "rating": 8.8},
                {"title": "Up", "genre": "Animation", "rating": 8.3}
            ],
            "sad": [
                {"title": "Inside Out", "genre": "Animation", "rating": 8.1},
                {"title": "The Shawshank Redemption", "genre": "Drama", "rating": 9.3},
                {"title": "Good Will Hunting", "genre": "Drama", "rating": 8.3}
            ],
            "anxious": [
                {"title": "Calm", "genre": "Documentary", "rating": 7.5},
                {"title": "Peaceful Warrior", "genre": "Drama", "rating": 7.2},
                {"title": "The Secret Garden", "genre": "Family", "rating": 7.3}
            ],
            "default": [
                {"title": "The Grand Budapest Hotel", "genre": "Comedy", "rating": 8.1},
                {"title": "Paddington", "genre": "Family", "rating": 7.2},
                {"title": "Chef", "genre": "Comedy", "rating": 7.3}
            ]
        }
        
        return movie_recommendations.get(mood.lower(), movie_recommendations["default"])

    def get_meditation_script(self, mood="default"):
        """Get meditation script based on mood"""
        scripts = {
            "anxious": {
                "title": "Anxiety Relief Meditation",
                "duration": "10 minutes",
                "script": """
                Find a comfortable seated position... Close your eyes gently...
                Take three deep breaths, letting each exhale release tension...
                Notice any anxious thoughts without judgment... Let them pass like clouds in the sky...
                Focus on the sensation of your breath... In and out... In and out...
                With each breath, feel your body becoming more relaxed...
                You are safe in this moment... You are exactly where you need to be...
                Continue breathing mindfully for the remaining time...
                """
            },
            "stressed": {
                "title": "Stress Relief Meditation", 
                "duration": "12 minutes",
                "script": """
                Sit comfortably and close your eyes... Take a moment to acknowledge your stress...
                Breathe in calm, breathe out tension... 
                Starting from your toes, consciously relax each part of your body...
                Move up through your legs, torso, arms, and finally your face...
                Imagine stress leaving your body with each exhale...
                You have the strength to handle whatever comes your way...
                Rest in this peaceful state for as long as you need...
                """
            },
            "default": {
                "title": "Mindfulness Meditation",
                "duration": "15 minutes", 
                "script": """
                Settle into a comfortable position... Close your eyes or soften your gaze...
                Begin by taking three deep, cleansing breaths...
                Now let your breath return to its natural rhythm...
                Simply observe each breath without trying to change it...
                When thoughts arise, acknowledge them and gently return to your breath...
                You are cultivating awareness and presence in this moment...
                Continue this practice, being kind and patient with yourself...
                """
            }
        }
        
        return scripts.get(mood.lower(), scripts["default"])

    # EXTRACTED FROM app.py:1208-1214 (EXACT COPY)  
    def load_posts(self):
        """Load community posts from JSON file"""
        try:
            with open(self.POSTS_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    # EXTRACTED FROM app.py:1214-1218 (EXACT COPY)
    def save_posts(self, posts):
        """Save posts to JSON file"""
        os.makedirs(os.path.dirname(self.POSTS_FILE), exist_ok=True)
        with open(self.POSTS_FILE, 'w') as f:
            json.dump(posts, f, indent=2)

    def add_community_post(self, content, author="Anonymous"):
        """Add a new community post"""
        try:
            posts = self.load_posts()
            new_post = {
                "id": len(posts),
                "content": content,
                "author": author,
                "timestamp": datetime.now().isoformat(),
                "likes": 0,
                "responses": []
            }
            posts.append(new_post)
            self.save_posts(posts)
            return {"success": True, "message": "Post added successfully!", "post": new_post}
        except Exception as e:
            return {"success": False, "message": f"Error adding post: {e}"}

    def get_community_posts(self, limit=10):
        """Get recent community posts"""
        try:
            posts = self.load_posts()
            # Return most recent posts first
            recent_posts = sorted(posts, key=lambda x: x.get('timestamp', ''), reverse=True)
            return recent_posts[:limit]
        except Exception as e:
            print(f"Error loading posts: {e}")
            return []

    # EXTRACTED FROM app.py:1361-1406 (EXACT COPY - adapted for service)
    def book_appointment(self, appointment_data):
        """Handle appointment booking"""
        try:
            # Validate required fields
            required_fields = ['name', 'email', 'phone', 'service', 'date', 'time']
            for field in required_fields:
                if field not in appointment_data or not appointment_data[field]:
                    return {"success": False, "message": f"Missing required field: {field}"}
            
            # Create appointment record
            appointment = {
                "id": f"apt_{int(datetime.now().timestamp())}",
                "name": appointment_data['name'],
                "email": appointment_data['email'], 
                "phone": appointment_data['phone'],
                "service": appointment_data['service'],
                "date": appointment_data['date'],
                "time": appointment_data['time'],
                "message": appointment_data.get('message', ''),
                "status": "pending",
                "created_at": datetime.now().isoformat()
            }
            
            # Save appointment (in production, this would go to a database)
            appointments_file = "appointments.json"
            try:
                with open(appointments_file, 'r') as f:
                    appointments = json.load(f)
            except FileNotFoundError:
                appointments = []
            
            appointments.append(appointment)
            
            with open(appointments_file, 'w') as f:
                json.dump(appointments, f, indent=2)
            
            return {
                "success": True, 
                "message": "Appointment booked successfully!",
                "appointment_id": appointment['id']
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error booking appointment: {e}"}

    def get_sound_therapy_options(self):
        """Get available sound therapy options"""
        return {
            "nature_sounds": [
                {"name": "Ocean Waves", "file": "ocean_waves.mp3", "description": "Calming ocean sounds"},
                {"name": "Rainfall", "file": "rainfall.mp3", "description": "Gentle rain sounds"},
                {"name": "Forest Birds", "file": "forest_birds.mp3", "description": "Peaceful bird songs"},
                {"name": "Mountain Stream", "file": "mountain_stream.mp3", "description": "Flowing water sounds"}
            ],
            "binaural_beats": [
                {"name": "Focus (40Hz)", "file": "focus_40hz.mp3", "description": "Gamma waves for concentration"},
                {"name": "Relaxation (10Hz)", "file": "relax_10hz.mp3", "description": "Alpha waves for relaxation"},
                {"name": "Deep Sleep (2Hz)", "file": "sleep_2hz.mp3", "description": "Delta waves for sleep"},
                {"name": "Meditation (6Hz)", "file": "meditation_6hz.mp3", "description": "Theta waves for meditation"}
            ],
            "white_noise": [
                {"name": "Pink Noise", "file": "pink_noise.mp3", "description": "Balanced frequency noise"},
                {"name": "Brown Noise", "file": "brown_noise.mp3", "description": "Deep, low-frequency noise"},
                {"name": "Fan Sound", "file": "fan_sound.mp3", "description": "Consistent fan noise"}
            ]
        }

    def get_wellness_tips(self, category="general"):
        """Get wellness tips based on category"""
        tips = {
            "general": [
                "Stay hydrated - drink at least 8 glasses of water daily",
                "Take breaks every hour if you work at a computer",
                "Practice gratitude by writing down 3 things you're thankful for each day",
                "Get 7-9 hours of sleep each night",
                "Take a 10-minute walk outside for fresh air and vitamin D"
            ],
            "mental_health": [
                "Practice mindfulness meditation for 10 minutes daily",
                "Limit social media usage to improve mental well-being", 
                "Connect with friends or family members regularly",
                "Keep a mood journal to track emotional patterns",
                "Seek professional help if you're struggling with persistent negative thoughts"
            ],
            "stress": [
                "Try the 4-7-8 breathing technique when feeling stressed",
                "Organize your workspace to reduce cognitive load",
                "Set boundaries between work and personal time",
                "Practice saying 'no' to commitments that drain your energy",
                "Use the Pomodoro technique for better time management"
            ],
            "sleep": [
                "Keep your bedroom cool, dark, and quiet",
                "Avoid screens 1 hour before bedtime",
                "Create a consistent bedtime routine",
                "Limit caffeine intake after 2 PM",
                "Use relaxation techniques like progressive muscle relaxation"
            ]
        }
        
        return tips.get(category.lower(), tips["general"])

    def cleanup(self):
        """Cleanup service resources"""
        print("🧹 Wellness & Therapy service cleanup complete")