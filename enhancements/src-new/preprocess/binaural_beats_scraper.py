"""
🧠 Y.M.I.R Binaural Beats & Brainwave Entrainment Scraper
========================================================
Specialized scraper for experimental audio content:
✅ Binaural beats (Focus, Sleep, Study, Meditation)
✅ Isochronic tones and brainwave entrainment
✅ Therapeutic frequencies (432Hz, 528Hz, Solfeggio)
✅ Ambient and healing audio content
✅ Categorized by purpose/mood for experimental modes
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import spotipy.exceptions
import csv
import json
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('binaural_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Fix Windows console encoding
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
logger = logging.getLogger(__name__)

class BinauralBeatsSpotifyScraper:
    """Specialized scraper for binaural beats and brainwave entrainment content"""
    
    def __init__(self, client_id: str = None, client_secret: str = None, redirect_uri: str = None):
        # Load credentials from environment variables if not provided
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        self.redirect_uri = redirect_uri or os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8080')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify credentials not found. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env file")
        
        # Try Client Credentials first
        try:
            logger.info("🔑 Initializing Spotify Client Credentials...")
            self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            ))
            logger.info("✅ Client Credentials authentication successful")
        except Exception as e:
            logger.warning(f"❌ Client Credentials failed: {e}")
            logger.info("🔑 Falling back to OAuth authentication...")
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret, 
                redirect_uri=self.redirect_uri,
                scope=[]
            ))
        
        # File paths for binaural beats dataset
        self.csv_file = "datasets/binaural_beats_experimental.csv"
        self.state_file = "binaural_scraper_state.json"
        self.processed_uris_file = "binaural_processed_uris.json"
        
        # Load existing state
        self.existing_uris = self.load_existing_uris()
        self.scraper_state = self.load_scraper_state()
        
        # Specialized search queries for experimental audio content
        self.experimental_queries = {
            'focus': [
                "binaural beats focus",
                "binaural beats concentration", 
                "alpha waves focus",
                "beta waves concentration",
                "40hz gamma waves",
                "focus music binaural",
                "study beats focus",
                "concentration enhancement audio"
            ],
            'sleep': [
                "binaural beats sleep",
                "binaural beats deep sleep",
                "delta waves sleep",
                "theta waves sleep",
                "sleep meditation binaural",
                "deep sleep frequencies",
                "1hz delta waves",
                "4hz theta sleep"
            ],
            'meditation': [
                "binaural beats meditation",
                "theta waves meditation",
                "meditation frequencies",
                "7.83hz schumann resonance",
                "om meditation frequency",
                "chakra frequencies",
                "tibetan singing bowls",
                "meditation ambient"
            ],
            'relaxation': [
                "binaural beats relaxation",
                "binaural beats stress relief",
                "alpha waves relaxation",
                "calm frequencies",
                "anxiety relief binaural",
                "stress reduction audio",
                "peaceful frequencies",
                "tranquil binaural"
            ],
            'creativity': [
                "binaural beats creativity",
                "theta waves creativity",
                "creative flow frequencies",
                "inspiration frequencies",
                "artistic enhancement audio",
                "creative meditation",
                "flow state binaural",
                "6hz theta creativity"
            ],
            'healing': [
                "432hz healing frequency",
                "528hz love frequency", 
                "solfeggio frequencies",
                "healing frequencies",
                "528hz dna repair",
                "741hz detox frequency",
                "852hz awakening",
                "963hz pineal gland"
            ],
            'isochronic': [
                "isochronic tones",
                "isochronic tones focus",
                "isochronic tones sleep",
                "isochronic meditation",
                "brainwave entrainment",
                "isochronic beats",
                "pulsed tones",
                "monaural beats"
            ],
            'ambient': [
                "nature sounds meditation",
                "rain sounds sleep",
                "ocean waves relaxation", 
                "forest sounds healing",
                "white noise",
                "brown noise sleep",
                "pink noise",
                "ambient healing music"
            ]
        }
        
        # API rate limiting
        self.requests_count = 0
        self.max_requests_per_hour = 1000
        self.start_time = time.time()
        
        logger.info("🧠 Binaural Beats Scraper initialized")
        logger.info(f"📊 Existing tracks: {len(self.existing_uris)}")
    
    def load_existing_uris(self) -> Set[str]:
        """Load existing binaural beats track URIs"""
        existing_uris = set()
        
        # Load from binaural beats dataset
        if os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        uri = row.get('Track URI', '').strip()
                        if uri:
                            existing_uris.add(uri)
                logger.info(f"📁 Loaded {len(existing_uris)} existing binaural URIs")
            except Exception as e:
                logger.error(f"❌ Error loading binaural dataset: {e}")
        
        # Load from processed URIs cache
        if os.path.exists(self.processed_uris_file):
            try:
                with open(self.processed_uris_file, 'r') as file:
                    cached_uris = json.load(file)
                    existing_uris.update(cached_uris)
                logger.info(f"💾 Total URIs (including cache): {len(existing_uris)}")
            except Exception as e:
                logger.error(f"❌ Error loading URI cache: {e}")
        
        return existing_uris
    
    def save_processed_uris(self):
        """Save processed URIs to cache"""
        try:
            with open(self.processed_uris_file, 'w') as file:
                json.dump(list(self.existing_uris), file)
            logger.info(f"💾 Saved {len(self.existing_uris)} processed URIs")
        except Exception as e:
            logger.error(f"❌ Error saving URI cache: {e}")
    
    def load_scraper_state(self) -> Dict:
        """Load scraper state for resuming"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as file:
                    state = json.load(file)
                logger.info(f"🔄 Loaded scraper state: Category {state.get('current_category', 'focus')}")
                return state
            except Exception as e:
                logger.error(f"❌ Error loading state: {e}")
        
        return {
            'current_category': 'focus',
            'current_query_index': 0,
            'current_offset': 0,
            'last_run': None,
            'total_added': 0,
            'categories_completed': []
        }
    
    def save_scraper_state(self):
        """Save current scraper state"""
        self.scraper_state['last_run'] = datetime.now().isoformat()
        try:
            with open(self.state_file, 'w') as file:
                json.dump(self.scraper_state, file, indent=2)
            logger.info("💾 Scraper state saved")
        except Exception as e:
            logger.error(f"❌ Error saving state: {e}")
    
    def check_rate_limit(self):
        """Smart rate limiting"""
        self.requests_count += 1
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            requests_per_second = self.requests_count / elapsed_time
            if requests_per_second > 2:
                sleep_time = 0.5
                logger.info(f"⏱️ Rate limiting: sleeping {sleep_time}s")
                time.sleep(sleep_time)
    
    def get_track_details_batch(self, track_uris: List[str], category: str) -> List[Dict]:
        """Get track details with experimental audio categorization"""
        track_details = []
        
        batch_size = 20
        for i in range(0, len(track_uris), batch_size):
            batch_uris = track_uris[i:i + batch_size]
            
            try:
                self.check_rate_limit()
                
                # Get track info
                tracks_batch = self.sp.tracks(batch_uris)
                
                # Try to get audio features (may fail due to deprecation/permissions)
                audio_features_batch = None
                try:
                    audio_features_batch = self.sp.audio_features(batch_uris)
                    logger.debug("✅ Audio features successful for batch")
                except spotipy.exceptions.SpotifyException as e:
                    if e.http_status == 403:
                        logger.warning(f"⚠️ Audio features forbidden (403) - using defaults")
                    else:
                        logger.warning(f"⚠️ Audio features error ({e.http_status}) - using defaults")
                    audio_features_batch = [None] * len(batch_uris)
                except Exception as e:
                    logger.warning(f"⚠️ Audio features failed: {e} - using defaults")
                    audio_features_batch = [None] * len(batch_uris)
                
                # Process each track
                for j, track in enumerate(tracks_batch['tracks']):
                    if track:
                        try:
                            # Get artist details
                            self.check_rate_limit()
                            artist = self.sp.artist(track['artists'][0]['uri'])
                            
                            # Get audio features or use defaults
                            if audio_features_batch and j < len(audio_features_batch) and audio_features_batch[j]:
                                af = audio_features_batch[j]
                                audio_data = {
                                    'Danceability': af['danceability'],
                                    'Energy': af['energy'],
                                    'Key': af['key'],
                                    'Loudness': af['loudness'],
                                    'Mode': af['mode'],
                                    'Speechiness': af['speechiness'],
                                    'Acousticness': af['acousticness'],
                                    'Instrumentalness': af['instrumentalness'],
                                    'Liveness': af['liveness'],
                                    'Valence': af['valence'],
                                    'Tempo': af['tempo'],
                                }
                            else:
                                # Defaults optimized for binaural/ambient content
                                audio_data = {
                                    'Danceability': 0.1,  # Low for ambient
                                    'Energy': 0.2,        # Low for relaxation
                                    'Key': 5,
                                    'Loudness': -12.0,    # Quieter for meditation
                                    'Mode': 1,
                                    'Speechiness': 0.02,  # Minimal speech
                                    'Acousticness': 0.8,  # High for natural sounds
                                    'Instrumentalness': 0.9,  # Mostly instrumental
                                    'Liveness': 0.05,     # Studio recordings
                                    'Valence': 0.4,       # Neutral to calming
                                    'Tempo': 60.0,        # Slow for relaxation
                                }
                            
                            track_detail = {
                                'Track URI': track['uri'],
                                'Track Name': track['name'],
                                'Artist URI': artist['uri'],
                                'Artist Name': artist['name'],
                                'Artist Popularity': artist['popularity'],
                                'Artist Genres': str(artist['genres']),
                                'Album': track['album']['name'],
                                'Track Popularity': track['popularity'],
                                'Duration (ms)': track['duration_ms'],
                                'Experimental_Category': category,  # NEW: Category for experimental use
                                'Brainwave_Type': self.classify_brainwave_type(track['name'], category),  # NEW
                                'Purpose': category,  # NEW: Purpose/mood classification
                                **audio_data
                            }
                            track_details.append(track_detail)
                            
                        except Exception as e:
                            logger.warning(f"⚠️ Error processing track {track['uri']}: {e}")
                
                logger.info(f"✅ Processed batch: {len(track_details)} tracks for {category}")
                
            except spotipy.exceptions.SpotifyException as e:
                if e.http_status == 429:
                    retry_after = int(e.headers.get('Retry-After', 60))
                    logger.warning(f"⏱️ Rate limit hit, waiting {retry_after}s")
                    time.sleep(retry_after)
                else:
                    logger.error(f"❌ Spotify API error: {e}")
            except Exception as e:
                logger.error(f"❌ Batch processing error: {e}")
        
        return track_details
    
    def classify_brainwave_type(self, track_name: str, category: str) -> str:
        """Classify brainwave type based on track name and category"""
        track_lower = track_name.lower()
        
        # Frequency-based classification
        if any(freq in track_lower for freq in ['delta', '1hz', '2hz', '3hz', '4hz']):
            return 'Delta (0.5-4Hz)'
        elif any(freq in track_lower for freq in ['theta', '5hz', '6hz', '7hz', '8hz']):
            return 'Theta (4-8Hz)'
        elif any(freq in track_lower for freq in ['alpha', '9hz', '10hz', '11hz', '12hz']):
            return 'Alpha (8-12Hz)'
        elif any(freq in track_lower for freq in ['beta', '13hz', '14hz', '15hz', '20hz', '30hz']):
            return 'Beta (13-30Hz)'
        elif any(freq in track_lower for freq in ['gamma', '40hz', '50hz']):
            return 'Gamma (30-50Hz)'
        
        # Category-based classification
        category_mapping = {
            'sleep': 'Delta (0.5-4Hz)',
            'meditation': 'Theta (4-8Hz)',
            'relaxation': 'Alpha (8-12Hz)',
            'focus': 'Beta (13-30Hz)',
            'creativity': 'Theta (4-8Hz)'
        }
        
        return category_mapping.get(category, 'Mixed/Unknown')
    
    def search_experimental_tracks(self, target_per_category: int = 50) -> List[Dict]:
        """Search for experimental audio tracks by category"""
        all_new_tracks = []
        
        categories = list(self.experimental_queries.keys())
        current_category = self.scraper_state['current_category']
        
        # Start from where we left off
        if current_category in categories:
            start_index = categories.index(current_category)
        else:
            start_index = 0
        
        logger.info(f"🎯 Starting experimental search from category: {current_category}")
        
        for category_index in range(start_index, len(categories)):
            category = categories[category_index]
            queries = self.experimental_queries[category]
            
            if category in self.scraper_state.get('categories_completed', []):
                logger.info(f"⏭️ Skipping completed category: {category}")
                continue
            
            logger.info(f"🎵 Processing category: {category.upper()}")
            category_tracks = []
            new_uris_batch = []
            
            query_index = self.scraper_state.get('current_query_index', 0) if category == current_category else 0
            offset = self.scraper_state.get('current_offset', 0) if category == current_category else 0
            
            while len(category_tracks) < target_per_category and query_index < len(queries):
                current_query = queries[query_index]
                
                try:
                    logger.info(f"🔍 Searching: '{current_query}' (offset: {offset})")
                    self.check_rate_limit()
                    
                    results = self.sp.search(
                        q=current_query,
                        type='track',
                        limit=50,
                        offset=offset
                    )
                    
                    if not results['tracks']['items']:
                        logger.info(f"⏭️ No more results for: {current_query}")
                        query_index += 1
                        offset = 0
                        continue
                    
                    # Collect new URIs
                    batch_new_uris = []
                    for track in results['tracks']['items']:
                        if track and track['uri'] not in self.existing_uris:
                            batch_new_uris.append(track['uri'])
                            self.existing_uris.add(track['uri'])
                    
                    logger.info(f"🆕 Found {len(batch_new_uris)} new URIs for {category}")
                    
                    if batch_new_uris:
                        new_uris_batch.extend(batch_new_uris)
                        
                        # Process in batches
                        if len(new_uris_batch) >= 50:
                            logger.info(f"⚙️ Processing batch of {len(new_uris_batch)} URIs...")
                            batch_details = self.get_track_details_batch(new_uris_batch, category)
                            category_tracks.extend(batch_details)
                            new_uris_batch = []
                            
                            logger.info(f"📊 Category progress: {len(category_tracks)}/{target_per_category}")
                    
                    offset += 50
                    
                    # Save state periodically
                    if len(category_tracks) % 25 == 0:
                        self.scraper_state['current_category'] = category
                        self.scraper_state['current_query_index'] = query_index
                        self.scraper_state['current_offset'] = offset
                        self.save_scraper_state()
                        self.save_processed_uris()
                
                except spotipy.exceptions.SpotifyException as e:
                    if e.http_status == 429:
                        retry_after = int(e.headers.get('Retry-After', 60))
                        logger.warning(f"⏱️ Rate limit hit, waiting {retry_after}s")
                        time.sleep(retry_after)
                    else:
                        logger.error(f"❌ Search error: {e}")
                        query_index += 1
                        offset = 0
                except Exception as e:
                    logger.error(f"❌ Unexpected error: {e}")
                    query_index += 1
                    offset = 0
            
            # Process remaining URIs for this category
            if new_uris_batch:
                logger.info(f"⚙️ Processing final batch of {len(new_uris_batch)} URIs for {category}...")
                batch_details = self.get_track_details_batch(new_uris_batch, category)
                category_tracks.extend(batch_details)
            
            all_new_tracks.extend(category_tracks)
            
            # Mark category as completed
            if category not in self.scraper_state.get('categories_completed', []):
                self.scraper_state.setdefault('categories_completed', []).append(category)
            
            logger.info(f"✅ Completed {category}: {len(category_tracks)} tracks")
            
            # Move to next category
            if category_index + 1 < len(categories):
                self.scraper_state['current_category'] = categories[category_index + 1]
                self.scraper_state['current_query_index'] = 0
                self.scraper_state['current_offset'] = 0
        
        self.scraper_state['total_added'] += len(all_new_tracks)
        logger.info(f"🎉 Search complete! Found {len(all_new_tracks)} experimental tracks")
        return all_new_tracks
    
    def save_to_csv(self, new_tracks: List[Dict]):
        """Save experimental tracks to specialized dataset"""
        if not new_tracks:
            logger.info("ℹ️ No new tracks to save")
            return
        
        try:
            # Define columns for experimental dataset
            experimental_columns = [
                'Track URI', 'Track Name', 'Artist URI', 'Artist Name', 
                'Artist Popularity', 'Artist Genres', 'Album', 'Track Popularity',
                'Duration (ms)', 'Experimental_Category', 'Brainwave_Type', 'Purpose',
                'Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness',
                'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo'
            ]
            
            # Check if file exists
            file_exists = os.path.exists(self.csv_file)
            
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=experimental_columns)
                
                # Write header if new file
                if not file_exists:
                    writer.writeheader()
                    logger.info("📄 Created new binaural beats dataset with headers")
                
                # Write tracks
                for track in new_tracks:
                    filtered_track = {col: track.get(col) for col in experimental_columns}
                    writer.writerow(filtered_track)
                
                logger.info(f"💾 Saved {len(new_tracks)} experimental tracks to dataset")
                logger.info(f"📁 Dataset location: {self.csv_file}")
                
        except Exception as e:
            logger.error(f"❌ Error saving experimental dataset: {e}")
    
    def enhance_experimental_dataset(self, target_per_category: int = 50):
        """Main method to build experimental audio dataset"""
        logger.info("🚀 Starting experimental dataset enhancement...")
        
        # Show current stats
        logger.info(f"📊 Current experimental tracks: {len(self.existing_uris)}")
        
        # Search for experimental tracks
        new_tracks = self.search_experimental_tracks(target_per_category)
        
        if new_tracks:
            # Save to specialized experimental dataset
            self.save_to_csv(new_tracks)
            
            # Update caches
            self.save_processed_uris()
            self.save_scraper_state()
            
            # Show category breakdown
            category_counts = {}
            for track in new_tracks:
                category = track.get('Experimental_Category', 'unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
            
            logger.info("📈 Category breakdown:")
            for category, count in category_counts.items():
                logger.info(f"  {category}: {count} tracks")
            
            logger.info(f"🎉 Successfully added {len(new_tracks)} experimental tracks!")
        else:
            logger.info("ℹ️ No new experimental tracks found")
        
        logger.info(f"✅ Experimental dataset ready: {self.csv_file}")


def main():
    """Main execution function for binaural beats scraping"""
    try:
        # Initialize specialized scraper
        scraper = BinauralBeatsSpotifyScraper()
        
        # Build experimental dataset (50 tracks per category)
        scraper.enhance_experimental_dataset(target_per_category=50)
        
        logger.info("🎯 Binaural beats scraping completed!")
        
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")


if __name__ == "__main__":
    main()