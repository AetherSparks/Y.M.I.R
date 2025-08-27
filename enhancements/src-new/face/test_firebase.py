"""
🔥 Firebase Connection Test for Y.M.I.R
========================================
Test script to verify your Firebase setup is working correctly
"""

import json
from pathlib import Path
from datetime import datetime

# Test Firebase connection
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    
    print("✅ Firebase libraries imported successfully")
    
    # Check if credentials file exists
    cred_path = Path("firebase_credentials.json")
    if not cred_path.exists():
        print("❌ Firebase credentials file not found!")
        print("   Make sure 'firebase_credentials.json' is in the src/ directory")
        exit(1)
    
    print("✅ Firebase credentials file found")
    
    # Initialize Firebase (check if already initialized)
    if not firebase_admin._apps:
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred)
        print("✅ Firebase app initialized")
    else:
        print("✅ Firebase app already initialized")
    
    # Get Firestore client
    db = firestore.client()
    print("✅ Firestore client created")
    
    # Test read/write operations
    print("\n🧪 Testing Firebase operations...")
    
    # Test collection: emotion_sessions
    test_doc_ref = db.collection('emotion_sessions').document()
    test_data = {
        'timestamp': datetime.now().isoformat(),
        'face_id': 0,
        'emotions': {
            'happy': 75.5,
            'neutral': 20.3,
            'sad': 4.2
        },
        'confidence': 0.87,
        'quality_score': 0.73,
        'context_objects': ['person', 'laptop'],
        'face_bbox': [100, 100, 200, 200],
        'session_id': 'test_session_123',
        'test': True
    }
    
    # Write test data
    test_doc_ref.set(test_data)
    print("✅ Test emotion data written to Firestore")
    
    # Read test data back
    doc = test_doc_ref.get()
    if doc.exists:
        retrieved_data = doc.to_dict()
        print("✅ Test emotion data read from Firestore")
        print(f"   Retrieved emotions: {retrieved_data['emotions']}")
    else:
        print("❌ Failed to read test data back")
    
    # Test collection: session_summaries
    summary_ref = db.collection('session_summaries').document('test_session_123')
    summary_data = {
        'total_readings': 1,
        'avg_confidence': 0.87,
        'avg_quality': 0.73,
        'dominant_emotion': 'happy',
        'emotion_stability': 0.95,
        'session_id': 'test_session_123',
        'end_time': datetime.now().isoformat(),
        'test': True
    }
    
    summary_ref.set(summary_data)
    print("✅ Test session summary written to Firestore")
    
    # Cleanup test data
    print("\n🧹 Cleaning up test data...")
    test_doc_ref.delete()
    summary_ref.delete()
    print("✅ Test data cleaned up")
    
    print("\n🎉 FIREBASE SETUP IS CORRECT!")
    print("   Your Firebase integration is working perfectly.")
    print("   Project ID:", json.loads(cred_path.read_text())['project_id'])
    print("   You can now run the enhanced emotion detection system!")
    
except ImportError as e:
    print("❌ Firebase libraries not installed")
    print("   Run: pip install firebase-admin")
    print(f"   Error: {e}")

except Exception as e:
    print(f"❌ Firebase setup error: {e}")
    print("\n🔧 Troubleshooting tips:")
    print("1. Make sure 'firebase_credentials.json' is in the src/ directory")
    print("2. Check that your Firebase project has Firestore enabled")
    print("3. Verify your service account has the correct permissions")
    print("4. Make sure you're connected to the internet")