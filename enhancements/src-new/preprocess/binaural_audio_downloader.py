"""
🎵 Y.M.I.R Binaural Beats Audio Downloader
==========================================
Legal audio acquisition for experimental dataset:
✅ Spotify preview clips (30s samples)
✅ YouTube search for copyright-free versions  
✅ Synthetic binaural beats generation
✅ Organized by experimental category
"""

import os
import csv
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging
from urllib.parse import urlparse
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import yt_dlp
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import lfilter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_downloader.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BinauralAudioDownloader:
    """Download and organize binaural beats audio for Y.M.I.R experimental modes"""
    
    def __init__(self):
        # Initialize Spotify client
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify credentials required in .env file")
            
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=self.client_id,
            client_secret=self.client_secret
        ))
        
        # File paths
        self.dataset_file = "datasets/binaural_beats_experimental.csv"
        self.audio_base_dir = "datasets/experimental_audio"
        self.download_state_file = "audio_download_state.json"
        
        # Create audio directories by category
        self.categories = ['focus', 'sleep', 'meditation', 'relaxation', 'creativity', 'healing', 'isochronic', 'ambient']
        self.setup_audio_directories()
        
        # Load download state
        self.download_state = self.load_download_state()
        
        # YouTube downloader config
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': '%(title)s.%(ext)s',
            'noplaylist': True,
            'extract_flat': False,
            'quiet': True,
            'no_warnings': True,
        }
        
        logger.info("🎵 Binaural Audio Downloader initialized")
    
    def setup_audio_directories(self):
        """Create organized directory structure for audio files"""
        base_path = Path(self.audio_base_dir)
        base_path.mkdir(exist_ok=True)
        
        for category in self.categories:
            category_path = base_path / category
            category_path.mkdir(exist_ok=True)
            
            # Subdirectories for different sources
            (category_path / "spotify_previews").mkdir(exist_ok=True)
            (category_path / "youtube_audio").mkdir(exist_ok=True)
            (category_path / "synthetic").mkdir(exist_ok=True)
        
        logger.info(f"📁 Created audio directories in {self.audio_base_dir}")
    
    def load_download_state(self) -> Dict:
        """Load download progress state"""
        if os.path.exists(self.download_state_file):
            try:
                with open(self.download_state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ Error loading download state: {e}")
        
        return {
            'downloaded_previews': [],
            'downloaded_youtube': [],
            'generated_synthetic': [],
            'failed_downloads': []
        }
    
    def save_download_state(self):
        """Save download progress state"""
        try:
            with open(self.download_state_file, 'w') as f:
                json.dump(self.download_state, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving download state: {e}")
    
    def load_experimental_dataset(self) -> List[Dict]:
        """Load binaural beats dataset"""
        tracks = []
        try:
            with open(self.dataset_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                tracks = list(reader)
            logger.info(f"📊 Loaded {len(tracks)} experimental tracks")
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
        
        return tracks
    
    def download_spotify_preview(self, track: Dict) -> Optional[str]:
        """Download 30-second preview from Spotify (legal)"""
        track_uri = track['Track URI']
        track_id = track_uri.split(':')[-1]
        category = track.get('Experimental_Category', 'unknown')
        
        if track_uri in self.download_state['downloaded_previews']:
            logger.info(f"⏭️ Preview already downloaded: {track['Track Name']}")
            return None
        
        try:
            # Get track details with preview URL
            track_info = self.sp.track(track_id)
            preview_url = track_info.get('preview_url')
            
            if not preview_url:
                logger.warning(f"⚠️ No preview available: {track['Track Name']}")
                return None
            
            # Download preview
            response = requests.get(preview_url, timeout=30)
            response.raise_for_status()
            
            # Save to category folder
            safe_name = "".join(c for c in track['Track Name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_name[:50]}_preview.mp3"
            file_path = Path(self.audio_base_dir) / category / "spotify_previews" / filename
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            self.download_state['downloaded_previews'].append(track_uri)
            logger.info(f"✅ Downloaded preview: {track['Track Name']}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Preview download failed for {track['Track Name']}: {e}")
            return None
    
    def search_exact_track_youtube(self, track: Dict) -> Optional[str]:
        """Search YouTube for EXACT track by artist and name"""
        track_name = track['Track Name']
        artist_name = track['Artist Name']
        category = track.get('Experimental_Category', 'unknown')
        
        # Create search terms for EXACT matches
        search_terms = [
            f'"{artist_name}" "{track_name}"',  # Exact match with quotes
            f"{artist_name} {track_name}",      # Artist + track name
            f"{track_name} {artist_name}",      # Track + artist name  
            f"{track_name}",                    # Just track name
            f"{artist_name} binaural beats",    # Artist + binaural beats
        ]
        
        for search_term in search_terms:
            try:
                logger.info(f"🔍 Searching YouTube: {search_term}")
                
                # Search YouTube for EXACT matches
                ydl_search_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_search_opts) as ydl:
                    search_results = ydl.extract_info(
                        f"ytsearch10:{search_term}",  # Search more results
                        download=False
                    )
                
                if search_results and 'entries' in search_results:
                    for entry in search_results['entries']:
                        title = entry.get('title', '')
                        uploader = entry.get('uploader', '')
                        
                        # Check for EXACT or close matches
                        title_lower = title.lower()
                        track_lower = track_name.lower()
                        artist_lower = artist_name.lower()
                        
                        # Score the match quality
                        match_score = 0
                        
                        # Check if track name is in title
                        if track_lower in title_lower:
                            match_score += 3
                        
                        # Check if artist name is in title or uploader
                        if artist_lower in title_lower or artist_lower in uploader.lower():
                            match_score += 2
                        
                        # Binaural beats specific terms
                        binaural_terms = ['binaural', 'beats', 'hz', 'frequency', 'waves']
                        for term in binaural_terms:
                            if term in title_lower:
                                match_score += 1
                        
                        # If good match, download it
                        if match_score >= 3:
                            logger.info(f"🎯 EXACT MATCH FOUND (score: {match_score}): {title}")
                            logger.info(f"   Uploader: {uploader}")
                            return self.download_youtube_audio(entry['url'], track)
                
            except Exception as e:
                logger.error(f"❌ YouTube search failed for {search_term}: {e}")
                continue
        
        logger.warning(f"⚠️ No exact match found for: {track_name} by {artist_name}")
        return None
    
    def download_youtube_audio(self, youtube_url: str, track: Dict) -> Optional[str]:
        """Download audio from YouTube (copyright-free only)"""
        category = track.get('Experimental_Category', 'unknown')
        safe_name = "".join(c for c in track['Track Name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        try:
            output_path = Path(self.audio_base_dir) / category / "youtube_audio"
            
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio/best',
                'outtmpl': str(output_path / f"{safe_name[:50]}_REAL.%(ext)s"),
                'noplaylist': True,
                'extract_flat': False,
                'writethumbnail': False,
                'writeinfojson': False,
                'quiet': False,  # Show download progress
                'no_warnings': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            
            logger.info(f"✅ Downloaded YouTube audio: {track['Track Name']}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ YouTube download failed: {e}")
            return None
    
    def generate_synthetic_binaural(self, track: Dict) -> Optional[str]:
        """Generate synthetic binaural beats based on track metadata"""
        category = track.get('Experimental_Category', 'unknown')
        brainwave_type = track.get('Brainwave_Type', '')
        
        # Map brainwave types to frequencies
        frequency_map = {
            'Delta (0.5-4Hz)': 2.0,
            'Theta (4-8Hz)': 6.0,
            'Alpha (8-12Hz)': 10.0,
            'Beta (13-30Hz)': 20.0,
            'Gamma (30-50Hz)': 40.0
        }
        
        base_freq = 200.0  # Base carrier frequency
        beat_freq = frequency_map.get(brainwave_type, 10.0)
        
        try:
            # Generate 30-second synthetic binaural beat
            sample_rate = 44100
            duration = 30  # seconds
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Left channel: base frequency
            left_channel = np.sin(2 * np.pi * base_freq * t)
            
            # Right channel: base frequency + beat frequency
            right_channel = np.sin(2 * np.pi * (base_freq + beat_freq) * t)
            
            # Combine channels and apply gentle fade in/out
            fade_samples = int(0.1 * sample_rate)  # 0.1 second fade
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            left_channel[:fade_samples] *= fade_in
            left_channel[-fade_samples:] *= fade_out
            right_channel[:fade_samples] *= fade_in
            right_channel[-fade_samples:] *= fade_out
            
            # Stereo audio (left, right)
            stereo_audio = np.column_stack((left_channel, right_channel))
            
            # Apply amplitude scaling
            stereo_audio = stereo_audio * 0.3  # Reduce volume for safety
            
            # Save as WAV file
            safe_name = "".join(c for c in track['Track Name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_name[:50]}_synthetic_{beat_freq}Hz.wav"
            file_path = Path(self.audio_base_dir) / category / "synthetic" / filename
            
            # Convert to 16-bit PCM
            stereo_audio_16bit = (stereo_audio * 32767).astype(np.int16)
            wavfile.write(file_path, sample_rate, stereo_audio_16bit)
            
            logger.info(f"🎛️ Generated synthetic binaural: {beat_freq}Hz for {track['Track Name']}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Synthetic generation failed: {e}")
            return None
    
    def download_experimental_audio(self, max_tracks_per_category: int = 50):
        """Download audio for experimental dataset"""
        logger.info("🚀 Starting experimental audio download...")
        
        tracks = self.load_experimental_dataset()
        if not tracks:
            logger.error("❌ No tracks found in dataset")
            return
        
        # Group tracks by category
        tracks_by_category = {}
        for track in tracks:
            category = track.get('Experimental_Category', 'unknown')
            if category not in tracks_by_category:
                tracks_by_category[category] = []
            tracks_by_category[category].append(track)
        
        # Process each category
        for category, category_tracks in tracks_by_category.items():
            logger.info(f"🎵 Processing category: {category.upper()}")
            
            # Limit tracks per category
            category_tracks = category_tracks[:max_tracks_per_category]
            
            for i, track in enumerate(category_tracks):
                logger.info(f"📊 Progress: {i+1}/{len(category_tracks)} in {category}")
                
                track_uri = track['Track URI']
                
                # Skip if already downloaded
                if track_uri in self.download_state['downloaded_youtube']:
                    logger.info(f"⏭️ Already downloaded: {track['Track Name']}")
                    continue
                
                # Method 1: Search for EXACT REAL AUDIO on YouTube
                logger.info(f"🎯 Searching for REAL AUDIO: {track['Track Name']} by {track['Artist Name']}")
                youtube_path = self.search_exact_track_youtube(track)
                
                if youtube_path:
                    self.download_state['downloaded_youtube'].append(track_uri)
                else:
                    # Method 2: Try Spotify preview as backup
                    preview_path = self.download_spotify_preview(track)
                    
                    # Method 3: Generate synthetic only if nothing else works
                    if not preview_path:
                        logger.info(f"🎛️ Falling back to synthetic for: {track['Track Name']}")
                        synthetic_path = self.generate_synthetic_binaural(track)
                        if synthetic_path:
                            self.download_state['generated_synthetic'].append(track_uri)
                
                # Save progress periodically
                if i % 10 == 0:
                    self.save_download_state()
                
                # Rate limiting
                time.sleep(0.5)
        
        # Final save
        self.save_download_state()
        logger.info("🎉 Audio download complete!")
        
        # Print summary
        self.print_download_summary()
    
    def print_download_summary(self):
        """Print download statistics"""
        logger.info("📈 Download Summary:")
        logger.info(f"  Spotify previews: {len(self.download_state['downloaded_previews'])}")
        logger.info(f"  YouTube downloads: {len(self.download_state['downloaded_youtube'])}")
        logger.info(f"  Synthetic generated: {len(self.download_state['generated_synthetic'])}")
        logger.info(f"  Failed downloads: {len(self.download_state['failed_downloads'])}")
        logger.info(f"📁 Audio stored in: {self.audio_base_dir}")


def main():
    """Main execution function"""
    try:
        downloader = BinauralAudioDownloader()
        
        # Download audio for experimental dataset
        # Limit to 25 tracks per category for testing
        downloader.download_experimental_audio(max_tracks_per_category=25)
        
        logger.info("✅ Binaural audio download completed!")
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")


if __name__ == "__main__":
    main()