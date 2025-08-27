"""
🔧 Protobuf Version Conflict Fix
================================
This script fixes the protobuf version conflict that causes:
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
"""

import subprocess
import sys

def fix_protobuf_conflict():
    """Fix protobuf version conflicts"""
    print("🔧 Fixing protobuf version conflict...")
    
    commands = [
        # Uninstall conflicting protobuf versions
        "pip uninstall protobuf -y",
        
        # Install compatible protobuf version
        "pip install protobuf==3.20.3",
        
        # Reinstall key packages with compatible versions
        "pip install --upgrade firebase-admin==6.2.0",
        "pip install --upgrade mediapipe==0.10.7",
        
        # Optional: Fix other potential conflicts
        "pip install --upgrade grpcio==1.59.0",
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Success: {cmd}")
            else:
                print(f"⚠️ Warning: {cmd} - {result.stderr.strip()}")
        except Exception as e:
            print(f"❌ Error: {cmd} - {e}")
    
    print("\n🎯 Protobuf fix complete!")
    print("Now restart your Python environment and try running the emotion detection again.")

if __name__ == "__main__":
    fix_protobuf_conflict()