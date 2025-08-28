"""
🎵 Y.M.I.R Directory Structure & Path Validation
=============================================
Validates that all paths are correctly configured
"""

import os
import pandas as pd

def check_directory_structure():
    """Check if all required files and directories exist"""
    print("🔍 Y.M.I.R Directory Structure Check")
    print("=" * 50)
    
    # Current working directory
    cwd = os.getcwd()
    print(f"📁 Current directory: {cwd}")
    
    # Files to check
    files_to_check = {
        'Original Dataset': '../../datasets/Y.M.I.R. original dataset.csv',
        'Processed Dataset': '../../datasets/therapeutic_music_enriched.csv',
        'Enhanced Scraper': 'preprocess/enhanced_scraper_production.py',
        'Enhanced Preprocessor': 'preprocess/enhanced_preprocess_production.py',
        'Usage Guide': 'preprocess/enhanced_usage_guide.py',
        'Recommendation Model': 'recommendation/music_recommendation_production.py'
    }
    
    all_good = True
    
    for name, path in files_to_check.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        
        if exists:
            try:
                size = os.path.getsize(path) / 1024  # KB
                print(f"{status} {name}: {path} ({size:.1f} KB)")
            except:
                print(f"{status} {name}: {path}")
        else:
            print(f"{status} {name}: {path} - NOT FOUND")
            all_good = False
    
    print("\n" + "=" * 50)
    
    # Check datasets specifically
    print("📊 Dataset Analysis:")
    
    original_path = '../../datasets/Y.M.I.R. original dataset.csv'
    if os.path.exists(original_path):
        try:
            df = pd.read_csv(original_path)
            print(f"✅ Original dataset: {len(df)} tracks, {len(df.columns)} columns")
            
            # Check for essential columns
            required_cols = ['Track Name', 'Artist Name', 'Tempo', 'Energy', 'Valence']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️ Missing columns: {missing_cols}")
            else:
                print("✅ All required columns present")
                
        except Exception as e:
            print(f"❌ Error reading original dataset: {e}")
    
    processed_path = '../../datasets/therapeutic_music_enriched.csv'
    if os.path.exists(processed_path):
        try:
            df = pd.read_csv(processed_path)
            print(f"✅ Processed dataset: {len(df)} tracks, {len(df.columns)} columns")
            
            if 'Mood_Label' in df.columns:
                mood_counts = df['Mood_Label'].value_counts()
                print(f"🎭 Mood classes: {len(mood_counts)}")
                print(f"🎭 Top 3 moods: {dict(mood_counts.head(3))}")
            
        except Exception as e:
            print(f"❌ Error reading processed dataset: {e}")
    else:
        print("ℹ️ Processed dataset not found - will be created on first run")
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("🎉 All files found! Ready to run the enhanced system.")
        print("\n📋 Next steps:")
        print("1. Navigate to enhancements/src-new/preprocess/")
        print("2. Run: python enhanced_usage_guide.py status")
        print("3. Run: python enhanced_usage_guide.py test")
    else:
        print("❌ Some files are missing. Please check the file structure.")
    
    return all_good

def create_missing_directories():
    """Create any missing directories"""
    dirs_to_create = [
        '../../datasets',
        'preprocess',
        'recommendation'
    ]
    
    for directory in dirs_to_create:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"📁 Created directory: {directory}")
            except Exception as e:
                print(f"❌ Failed to create {directory}: {e}")

if __name__ == "__main__":
    create_missing_directories()
    check_directory_structure()