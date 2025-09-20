"""
Y.M.I.R AI Emotion Detection System - Main Flask Application
===========================================================
Basic Flask app with microservices architecture for emotion detection,
chatbot integration, and music recommendations.

Author: Y.M.I.R Development Team
Version: 1.0.0
"""

from flask import Flask, render_template, request, jsonify, url_for
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ymir-dev-key-2024')

# Enable CORS for API calls
CORS(app)

# Configure static files
app.static_folder = 'static'
app.template_folder = 'templates'

@app.route('/')
def home():
    """Render the home page"""
    return render_template('home.html')

@app.route('/ai_app')
def ai_app():
    """Main AI application dashboard - placeholder for now"""
    return render_template('index.html')

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

@app.route('/wellness_tools')
def wellness_tools():
    """Wellness tools page"""
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

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'Y.M.I.R AI Main App',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("🚀 Starting Y.M.I.R AI Emotion Detection System...")
    print("📍 Home page: http://localhost:5000")
    print("🔧 AI App: http://localhost:5000/ai_app")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )