"""
Y.M.I.R Firebase Authentication Backend Integration
Handles server-side authentication verification and user data management
"""

import firebase_admin
from firebase_admin import credentials, auth, firestore
import os
import json
from functools import wraps
from flask import request, jsonify, g
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YMIRFirebaseAuth:
    def __init__(self, credentials_path=None):
        """Initialize Firebase Admin SDK"""
        self.db = None
        self.auth_client = None
        
        try:
            # Initialize Firebase Admin SDK
            if not firebase_admin._apps:
                # Try multiple credential sources
                if not credentials_path:
                    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                
                if credentials_path and os.path.exists(credentials_path):
                    # Use explicit credentials file
                    cred = credentials.Certificate(credentials_path)
                    project_id = os.getenv('FIREBASE_PROJECT_ID', 'y-m-i-r')
                    firebase_admin.initialize_app(cred, {
                        'projectId': project_id
                    })
                    logger.info(f"✅ Firebase Admin SDK initialized with credentials file: {credentials_path}")
                else:
                    # Try environment variable path
                    env_cred_path = os.path.join(os.getcwd(), 'firebase_credentials.json')
                    if os.path.exists(env_cred_path):
                        cred = credentials.Certificate(env_cred_path)
                        project_id = os.getenv('FIREBASE_PROJECT_ID', 'y-m-i-r')
                        firebase_admin.initialize_app(cred, {
                            'projectId': project_id
                        })
                        logger.info(f"✅ Firebase Admin SDK initialized with credentials file: {env_cred_path}")
                    else:
                        # Last resort: try default credentials (will fail gracefully)
                        firebase_admin.initialize_app()
                        logger.info("✅ Firebase Admin SDK initialized with default credentials")
            
            # Initialize Firestore
            self.db = firestore.client()
            self.auth_client = auth
            logger.info("✅ Firebase services initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Firebase initialization failed: {e}")
            self.db = None
            self.auth_client = None

    def verify_token(self, id_token):
        """Verify Firebase ID token and return user info"""
        try:
            if not self.auth_client:
                return None
                
            decoded_token = self.auth_client.verify_id_token(id_token)
            return {
                'uid': decoded_token['uid'],
                'email': decoded_token.get('email'),
                'name': decoded_token.get('name'),
                'email_verified': decoded_token.get('email_verified', False)
            }
        except Exception as e:
            logger.error(f"❌ Token verification failed: {e}")
            return None

    def get_user_data(self, uid):
        """Get user data from Firestore"""
        try:
            if not self.db:
                return None
                
            user_ref = self.db.collection('users').document(uid)
            user_doc = user_ref.get()
            
            if user_doc.exists:
                return user_doc.to_dict()
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting user data: {e}")
            return None

    def create_or_update_user(self, uid, user_data):
        """Create or update user document in Firestore"""
        try:
            if not self.db:
                return False
                
            user_ref = self.db.collection('users').document(uid)
            user_doc = user_ref.get()
            
            if not user_doc.exists:
                # Create new user
                user_ref.set({
                    'email': user_data.get('email'),
                    'name': user_data.get('name'),
                    'created_at': firestore.SERVER_TIMESTAMP,
                    'last_active': firestore.SERVER_TIMESTAMP,
                    'preferences': {},
                    'settings': {
                        'privacy_mode': True,
                        'data_retention': 30,  # days
                        'email_notifications': False
                    }
                })
                logger.info(f"✅ Created new user: {uid}")
            else:
                # Update existing user
                user_ref.update({
                    'last_active': firestore.SERVER_TIMESTAMP
                })
                logger.info(f"✅ Updated user: {uid}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating/updating user: {e}")
            return False

    def save_user_emotion_data(self, uid, emotion_data):
        """Save emotion detection data for user"""
        try:
            if not self.db:
                return False
                
            emotions_ref = self.db.collection('users').document(uid).collection('emotions')
            emotions_ref.add({
                **emotion_data,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'source': 'y-m-i-r-dashboard'
            })
            
            logger.info(f"✅ Saved emotion data for user: {uid}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving emotion data: {e}")
            return False

    def save_user_conversation(self, uid, conversation_data):
        """Save conversation data for user"""
        try:
            if not self.db:
                return False
                
            conversations_ref = self.db.collection('users').document(uid).collection('conversations')
            conversations_ref.add({
                **conversation_data,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'source': 'y-m-i-r-chatbot'
            })
            
            logger.info(f"✅ Saved conversation data for user: {uid}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving conversation data: {e}")
            return False

    def save_user_music_history(self, uid, music_data):
        """Save music listening history for user"""
        try:
            if not self.db:
                return False
                
            music_ref = self.db.collection('users').document(uid).collection('music_history')
            music_ref.add({
                **music_data,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'source': 'y-m-i-r-music-player'
            })
            
            logger.info(f"✅ Saved music history for user: {uid}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving music history: {e}")
            return False

    def get_user_emotion_history(self, uid, limit=50):
        """Get user's emotion history"""
        try:
            if not self.db:
                return []
                
            emotions_ref = (self.db.collection('users').document(uid)
                          .collection('emotions')
                          .order_by('timestamp', direction=firestore.Query.DESCENDING)
                          .limit(limit))
            
            emotions = emotions_ref.stream()
            return [{'id': doc.id, **doc.to_dict()} for doc in emotions]
            
        except Exception as e:
            logger.error(f"❌ Error getting emotion history: {e}")
            return []

    def get_user_preferences(self, uid):
        """Get user preferences"""
        try:
            user_data = self.get_user_data(uid)
            return user_data.get('preferences', {}) if user_data else {}
        except Exception as e:
            logger.error(f"❌ Error getting user preferences: {e}")
            return {}

    def update_user_preferences(self, uid, preferences):
        """Update user preferences"""
        try:
            if not self.db:
                return False
                
            user_ref = self.db.collection('users').document(uid)
            user_ref.update({
                'preferences': preferences,
                'last_active': firestore.SERVER_TIMESTAMP
            })
            
            logger.info(f"✅ Updated preferences for user: {uid}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating preferences: {e}")
            return False

# Initialize Firebase Auth instance
firebase_auth = YMIRFirebaseAuth()

# Authentication decorator
def require_auth(f):
    """Decorator to require authentication for Flask routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get token from Authorization header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No valid authentication token provided'}), 401
        
        token = auth_header.split(' ')[1]
        user_info = firebase_auth.verify_token(token)
        
        if not user_info:
            return jsonify({'error': 'Invalid authentication token'}), 401
        
        # Add user info to Flask's g object for use in the route
        g.current_user = user_info
        return f(*args, **kwargs)
    
    return decorated_function

def optional_auth(f):
    """Decorator for routes that work with or without authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Try to get token, but don't fail if not present
        auth_header = request.headers.get('Authorization')
        
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            user_info = firebase_auth.verify_token(token)
            g.current_user = user_info if user_info else None
        else:
            g.current_user = None
        
        return f(*args, **kwargs)
    
    return decorated_function

# Flask route integrations
def add_auth_routes(app):
    """Add authentication routes to Flask app"""
    
    @app.route('/api/auth/verify', methods=['POST'])
    def verify_auth():
        """Verify authentication token"""
        data = request.get_json()
        token = data.get('token')
        
        if not token:
            return jsonify({'success': False, 'error': 'No token provided'}), 400
        
        user_info = firebase_auth.verify_token(token)
        if user_info:
            # Create or update user in database
            firebase_auth.create_or_update_user(user_info['uid'], user_info)
            
            return jsonify({
                'success': True,
                'user': user_info
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid token'}), 401

    @app.route('/api/auth/user/data', methods=['GET'])
    @require_auth
    def get_user_data():
        """Get current user's data"""
        uid = g.current_user['uid']
        user_data = firebase_auth.get_user_data(uid)
        
        if user_data:
            return jsonify({
                'success': True,
                'data': user_data
            })
        else:
            return jsonify({'success': False, 'error': 'User data not found'}), 404

    @app.route('/api/auth/user/emotions', methods=['POST'])
    @require_auth
    def save_user_emotions():
        """Save user emotion data"""
        uid = g.current_user['uid']
        emotion_data = request.get_json()
        
        success = firebase_auth.save_user_emotion_data(uid, emotion_data)
        
        if success:
            return jsonify({'success': True, 'message': 'Emotion data saved'})
        else:
            return jsonify({'success': False, 'error': 'Failed to save emotion data'}), 500

    @app.route('/api/auth/user/emotions', methods=['GET'])
    @require_auth
    def get_user_emotions():
        """Get user emotion history"""
        uid = g.current_user['uid']
        limit = request.args.get('limit', 50, type=int)
        
        emotions = firebase_auth.get_user_emotion_history(uid, limit)
        
        return jsonify({
            'success': True,
            'emotions': emotions
        })

    @app.route('/api/auth/user/conversations', methods=['POST'])
    @require_auth
    def save_user_conversation():
        """Save user conversation data"""
        uid = g.current_user['uid']
        conversation_data = request.get_json()
        
        success = firebase_auth.save_user_conversation(uid, conversation_data)
        
        if success:
            return jsonify({'success': True, 'message': 'Conversation saved'})
        else:
            return jsonify({'success': False, 'error': 'Failed to save conversation'}), 500

    @app.route('/api/auth/user/music', methods=['POST'])
    @require_auth
    def save_user_music():
        """Save user music history"""
        uid = g.current_user['uid']
        music_data = request.get_json()
        
        success = firebase_auth.save_user_music_history(uid, music_data)
        
        if success:
            return jsonify({'success': True, 'message': 'Music history saved'})
        else:
            return jsonify({'success': False, 'error': 'Failed to save music history'}), 500

    @app.route('/api/auth/user/preferences', methods=['GET'])
    @require_auth
    def get_user_preferences():
        """Get user preferences"""
        uid = g.current_user['uid']
        preferences = firebase_auth.get_user_preferences(uid)
        
        return jsonify({
            'success': True,
            'preferences': preferences
        })

    @app.route('/api/auth/user/preferences', methods=['POST'])
    @require_auth
    def update_user_preferences():
        """Update user preferences"""
        uid = g.current_user['uid']
        preferences = request.get_json()
        
        success = firebase_auth.update_user_preferences(uid, preferences)
        
        if success:
            return jsonify({'success': True, 'message': 'Preferences updated'})
        else:
            return jsonify({'success': False, 'error': 'Failed to update preferences'}), 500

    # Enhanced existing routes to support authentication
    @app.route('/api/music/recommendations')
    @optional_auth
    def enhanced_music_recommendations():
        """Enhanced music recommendations with user context"""
        # Get existing recommendation logic
        from app import api_music_recommendations as original_recommendations
        
        # If user is authenticated, enhance with user data
        if g.current_user:
            uid = g.current_user['uid']
            preferences = firebase_auth.get_user_preferences(uid)
            
            # Add user preferences to request context
            request.user_preferences = preferences
        
        # Call original function
        return original_recommendations()

    logger.info("✅ Authentication routes added to Flask app")

# Export the main components
__all__ = ['firebase_auth', 'require_auth', 'optional_auth', 'add_auth_routes']