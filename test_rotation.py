#!/usr/bin/env python3
"""
Test script for Gemini API rotation system
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from gemini_api_manager import get_gemini_model, get_api_status, gemini_manager
    print("✅ Successfully imported rotation manager")
    
    # Check API keys
    status = get_api_status()
    print(f"📊 Total keys: {status['total_keys']}")
    print(f"📊 Active keys: {status['active_keys']}")
    print(f"📊 Current key: {status['current_key']}")
    
    # List all keys
    for key_info in status['keys']:
        status_str = "✅" if key_info['is_active'] and not key_info['quota_exhausted'] else "❌"
        print(f"   {status_str} {key_info['key_id']}: active={key_info['is_active']}, exhausted={key_info['quota_exhausted']}")
    
    # Try to create a model
    print("\n🧪 Testing model creation...")
    try:
        # This will automatically rotate if needed
        model = get_gemini_model("gemini-2.0-flash-exp")
        print("✅ Model creation successful!")
        
        # Test a simple call
        try:
            response = model.generate_content("Say hello")
            print(f"✅ Test message successful: {response.text[:50]}...")
        except Exception as e:
            print(f"❌ Test message failed: {e}")
            
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")