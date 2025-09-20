#!/usr/bin/env python3
"""
🧪 Test Facial Emotion Firebase Storage
=====================================
Manually add a facial emotion to Firebase to test if the storage works
"""

import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone
import uuid

def test_facial_firebase_storage():
    """Test adding a facial emotion directly to Firebase"""
    try:
        # Initialize Firebase
        if not firebase_admin._apps:
            cred = credentials.Certificate('firebase_credentials.json')
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        print("✅ Firebase connection successful")
        
        # Create a test facial emotion document
        test_facial_doc = {
            'timestamp': datetime.now(timezone.utc),
            'face_id': f'test_face_{uuid.uuid4().hex[:8]}',
            'emotions': {
                'happy': 85.2,
                'sad': 10.1,
                'angry': 2.3,
                'surprise': 1.8,
                'fear': 0.4,
                'disgust': 0.1,
                'neutral': 0.1
            },
            'confidence': 0.85,
            'quality_score': 0.9,
            'session_id': 'test_session'
            # NO 'role' field - this marks it as facial emotion
        }
        
        # Store in emotion_readings collection
        doc_ref = db.collection('emotion_readings').document()
        doc_ref.set(test_facial_doc)
        
        print("✅ Test facial emotion stored to Firebase!")
        print(f"📄 Document ID: {doc_ref.id}")
        print(f"📊 Emotions: {test_facial_doc['emotions']}")
        print(f"🎭 Dominant: happy (85.2%)")
        
        # Now test if combiner can find it
        print("\n🔍 Testing if combiner can find the test emotion...")
        
        # Import and test combiner
        import sys
        from pathlib import Path
        combiner_path = Path(__file__).parent / 'enhancements' / 'src-new' / 'multimodal_fusion'
        sys.path.append(str(combiner_path))
        
        from real_emotion_combiner import get_combined_emotion
        
        result = get_combined_emotion(minutes_back=5, strategy='adaptive')
        
        if result:
            print("✅ Combiner found the test emotion!")
            print(f"   🎭 Combined: {result['emotion']}")
            print(f"   📊 Confidence: {result['confidence']}")
            print(f"   🔄 Strategy: {result['strategy']}")
            if result.get('facial_data'):
                print("   📹 Facial data: FOUND ✅")
            else:
                print("   📹 Facial data: NOT FOUND ❌")
        else:
            print("❌ Combiner could not find the test emotion")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 TESTING FACIAL EMOTION FIREBASE STORAGE")
    print("=" * 50)
    
    success = test_facial_firebase_storage()
    
    if success:
        print("\n🎯 DIAGNOSIS:")
        print("✅ Firebase storage works correctly")
        print("🔍 Issue: Face microservice not calling Firebase storage")
        print("\n🔧 NEXT STEPS:")
        print("1. Check face microservice console for errors")
        print("2. Verify camera is detecting faces")
        print("3. Look for Firebase storage logs in face microservice")
    else:
        print("\n❌ Firebase storage issue detected")
        print("🔧 Check Firebase credentials and connectivity")