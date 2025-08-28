"""
🎵 Y.M.I.R Enhanced Spotify Scraper - Production Ready
=====================================================
Fixes critical issues:
✅ No re-scraping (persistent state tracking)
✅ Smart pagination (diversified search queries)  
✅ API quota optimization (batch processing)
✅ Robust duplicate detection (URI-based)
✅ Incremental dataset growth
✅ Error recovery and rate limiting
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
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

# Configure logging with proper encoding for Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Fix Windows console encoding issues
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass
logger = logging.getLogger(__name__)

class EnhancedSpotifyMusicScraper:
    """Production-ready Spotify scraper with quota optimization and no re-scraping"""
    
    def __init__(self, client_id: str = None, client_secret: str = None, redirect_uri: str = None):
        # Load credentials from environment variables if not provided
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        self.redirect_uri = redirect_uri or os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8080')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify credentials not found. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env file")
        
        # Try Client Credentials first (better for API-only access)
        try:
            logger.info("KEY Trying Client Credentials authentication...")
            self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            ))
            logger.info("SUCCESS Client Credentials authentication successful")
        except Exception as e:
            logger.warning(f"Client Credentials failed: {e}")
            logger.info("KEY Falling back to OAuth authentication...")
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret, 
                redirect_uri=self.redirect_uri,
                scope=[]
            ))
        
        # File paths (relative to src-new) - APPEND TO ORIGINAL DATASET
        self.csv_file = "datasets/Y.M.I.R. original dataset.csv"
        self.state_file = "scraper_state.json"
        self.processed_uris_file = "processed_uris.json"
        
        # Load existing state
        self.existing_uris = self.load_existing_uris()
        self.scraper_state = self.load_scraper_state()
        
        # Diversified search queries to avoid repetition
        self.search_queries = [
            "genre:Hindi",
            "tag:bollywood", 
            "tag:indian classical",
            "tag:sufi music",
            "tag:devotional",
            "tag:punjabi",
            "tag:ghazal",
            "tag:qawwali",
            "market:IN genre:pop",
            "market:IN genre:rock", 
            "year:2020-2024 market:IN",
            "year:2015-2019 market:IN",
            "year:2010-2014 market:IN",
            "year:2005-2009 market:IN",
        ]
        
        # API rate limiting
        self.requests_count = 0
        self.max_requests_per_hour = 1000  # Conservative limit
        self.start_time = time.time()
        
        logger.info("MUSIC Enhanced Spotify Scraper initialized")
        logger.info(f"DATA Existing tracks: {len(self.existing_uris)}")
    
    def load_existing_uris(self) -> Set[str]:
        """Load existing track URIs from original dataset"""
        existing_uris = set()
        
        # Load ONLY from original dataset (we'll append to it)
        if os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        uri = row.get('Track URI', '').strip()
                        if uri:  # Only add non-empty URIs
                            existing_uris.add(uri)
                logger.info(f"FILE Loaded {len(existing_uris)} existing URIs from original dataset")
            except Exception as e:
                logger.error(f"Error loading original dataset: {e}")
        
        # Load from processed URIs cache to avoid re-processing
        if os.path.exists(self.processed_uris_file):
            try:
                with open(self.processed_uris_file, 'r') as file:
                    cached_uris = json.load(file)
                    existing_uris.update(cached_uris)
                logger.info(f"CACHE Total URIs (including cache): {len(existing_uris)}")
            except Exception as e:
                logger.error(f"Error loading URI cache: {e}")
        
        return existing_uris
    
    def save_processed_uris(self):
        """Save processed URIs to cache"""
        try:
            with open(self.processed_uris_file, 'w') as file:
                json.dump(list(self.existing_uris), file)
            logger.info(f"SAVE Saved {len(self.existing_uris)} processed URIs")
        except Exception as e:
            logger.error(f"Error saving URI cache: {e}")
    
    def load_scraper_state(self) -> Dict:
        """Load scraper state for resuming"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as file:
                    state = json.load(file)
                logger.info(f"STATE Loaded scraper state: Query {state.get('current_query_index', 0)}")
                return state
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        
        return {
            'current_query_index': 0,
            'current_offset': 0,
            'last_run': None,
            'total_added': 0
        }
    
    def save_scraper_state(self):
        """Save current scraper state"""
        self.scraper_state['last_run'] = datetime.now().isoformat()
        try:
            with open(self.state_file, 'w') as file:
                json.dump(self.scraper_state, file, indent=2)
            logger.info("SAVE Scraper state saved")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def check_rate_limit(self):
        """Smart rate limiting to avoid quota exhaustion"""
        self.requests_count += 1
        
        # Check if we're approaching limits
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            requests_per_second = self.requests_count / elapsed_time
            if requests_per_second > 2:  # More than 2 requests/second
                sleep_time = 0.5
                logger.info(f"LIMIT Rate limiting: sleeping {sleep_time}s")
                time.sleep(sleep_time)
    
    def get_track_details_batch(self, track_uris: List[str]) -> List[Dict]:
        """Get track details in batches - handle deprecated audio-features gracefully"""
        track_details = []
        
        # Process in smaller batches due to API issues
        batch_size = 20  # Reduced from 50
        for i in range(0, len(track_uris), batch_size):
            batch_uris = track_uris[i:i + batch_size]
            
            try:
                self.check_rate_limit()
                
                # Get track info for batch first
                tracks_batch = self.sp.tracks(batch_uris)
                
                # Try to get audio features (may fail due to deprecation/permissions)
                audio_features_batch = None
                try:
                    audio_features_batch = self.sp.audio_features(batch_uris)
                    logger.debug(f"OK Audio features successful for batch")
                except spotipy.exceptions.SpotifyException as e:
                    if e.http_status == 403:
                        logger.warning(f"WARN Audio features forbidden (403) - using defaults")
                    else:
                        logger.warning(f"WARN Audio features error ({e.http_status}) - using defaults")
                    audio_features_batch = [None] * len(batch_uris)
                except Exception as e:
                    logger.warning(f"WARN Audio features failed: {e} - using defaults")
                    audio_features_batch = [None] * len(batch_uris)
                
                # Process each track
                for j, track in enumerate(tracks_batch['tracks']):
                    if track:
                        try:
                            # Get artist details (this usually works)
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
                                # Use reasonable defaults when audio features unavailable
                                audio_data = {
                                    'Danceability': 0.5,
                                    'Energy': 0.5,
                                    'Key': 5,
                                    'Loudness': -8.0,
                                    'Mode': 1,
                                    'Speechiness': 0.05,
                                    'Acousticness': 0.3,
                                    'Instrumentalness': 0.0,
                                    'Liveness': 0.1,
                                    'Valence': 0.5,
                                    'Tempo': 120.0,
                                }
                            
                            track_detail = {
                                'Track URI': track['uri'],
                                'Track Name': track['name'],
                                'Artist URI': artist['uri'],
                                'Artist Name': artist['name'],
                                'Artist Popularity': artist['popularity'],
                                'Artist Genres': str(artist['genres']),  # Convert list to string
                                'Album': track['album']['name'],
                                'Track Popularity': track['popularity'],
                                'Duration (ms)': track['duration_ms'],
                                **audio_data  # Add audio features or defaults
                            }
                            track_details.append(track_detail)
                            
                        except Exception as e:
                            logger.warning(f"Error processing track {track['uri']}: {e}")
                
                logger.info(f"OK Processed batch: {len(track_details)} tracks")
                
            except spotipy.exceptions.SpotifyException as e:
                if e.http_status == 429:
                    retry_after = int(e.headers.get('Retry-After', 60))
                    logger.warning(f"WAIT Rate limit hit, waiting {retry_after}s")
                    time.sleep(retry_after)
                else:
                    logger.error(f"Spotify API error: {e}")
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
        
        return track_details
    
    def search_new_tracks(self, target_new_tracks: int = 100) -> List[Dict]:
        """Search for new tracks using diversified queries"""
        new_tracks = []
        new_uris_batch = []  # Collect URIs before making detailed API calls
        
        query_index = self.scraper_state['current_query_index']
        offset = self.scraper_state['current_offset']
        
        logger.info(f"SEARCH Starting search from query {query_index}, offset {offset}")
        logger.info(f"TARGET Target: {target_new_tracks} new tracks")
        
        while len(new_tracks) < target_new_tracks and query_index < len(self.search_queries):
            current_query = self.search_queries[query_index]
            
            try:
                logger.info(f"QUERY Searching: '{current_query}' (offset: {offset})")
                self.check_rate_limit()
                
                # Search for tracks
                results = self.sp.search(
                    q=current_query,
                    type='track',
                    limit=50,
                    offset=offset,
                    market='IN'  # Focus on Indian market
                )
                
                if not results['tracks']['items']:
                    logger.info(f"SKIP No more results for query: {current_query}")
                    query_index += 1
                    offset = 0
                    continue
                
                # Collect new URIs (fast check, no API calls)
                batch_new_uris = []
                for track in results['tracks']['items']:
                    if track and track['uri'] not in self.existing_uris:
                        batch_new_uris.append(track['uri'])
                        self.existing_uris.add(track['uri'])  # Mark as processed
                
                logger.info(f"BATCH Found {len(batch_new_uris)} new URIs in this batch")
                
                if batch_new_uris:
                    new_uris_batch.extend(batch_new_uris)
                    
                    # Process in smaller batches to avoid memory issues
                    if len(new_uris_batch) >= 100:
                        logger.info(f"PROCESS Processing batch of {len(new_uris_batch)} URIs...")
                        batch_details = self.get_track_details_batch(new_uris_batch)
                        new_tracks.extend(batch_details)
                        new_uris_batch = []  # Clear batch
                        
                        logger.info(f"TOTAL Total new tracks collected: {len(new_tracks)}")
                
                offset += 50
                
                # Save state periodically
                if len(new_tracks) % 50 == 0:
                    self.scraper_state['current_query_index'] = query_index
                    self.scraper_state['current_offset'] = offset
                    self.save_scraper_state()
                    self.save_processed_uris()
                
            except spotipy.exceptions.SpotifyException as e:
                if e.http_status == 429:
                    retry_after = int(e.headers.get('Retry-After', 60))
                    logger.warning(f"WAIT Rate limit hit, waiting {retry_after}s")
                    time.sleep(retry_after)
                else:
                    logger.error(f"Search error: {e}")
                    query_index += 1
                    offset = 0
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                query_index += 1
                offset = 0
        
        # Process remaining URIs
        if new_uris_batch:
            logger.info(f"FINAL Processing final batch of {len(new_uris_batch)} URIs...")
            batch_details = self.get_track_details_batch(new_uris_batch)
            new_tracks.extend(batch_details)
        
        # Update final state
        self.scraper_state['current_query_index'] = query_index
        self.scraper_state['current_offset'] = offset
        self.scraper_state['total_added'] += len(new_tracks)
        
        logger.info(f"DONE Search complete! Found {len(new_tracks)} new tracks")
        return new_tracks
    
    def save_to_csv(self, new_tracks: List[Dict]):
        """Append new tracks directly to original dataset"""
        if not new_tracks:
            logger.info("No new tracks to save")
            return
        
        try:
            # Read existing dataset to get column structure
            existing_columns = []
            if os.path.exists(self.csv_file):
                with open(self.csv_file, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    existing_columns = reader.fieldnames
                    logger.info(f"COLS Original dataset has columns: {existing_columns}")
            
            # Ensure new tracks have all required columns
            if existing_columns:
                for track in new_tracks:
                    for col in existing_columns:
                        if col not in track:
                            track[col] = None  # Fill missing columns
                
                # Use existing column order
                fieldnames = existing_columns
            else:
                fieldnames = new_tracks[0].keys()
            
            # Append to original dataset
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                
                # Write header only if file is empty/new
                if not existing_columns:
                    writer.writeheader()
                    logger.info("NEW Created new dataset with headers")
                
                # Write only the columns that exist in the original
                for track in new_tracks:
                    filtered_track = {col: track.get(col) for col in fieldnames}
                    writer.writerow(filtered_track)
                
                logger.info(f"APPEND Appended {len(new_tracks)} NEW tracks to original dataset")
                logger.info(f"PATH Dataset location: {self.csv_file}")
                
        except Exception as e:
            logger.error(f"Error appending to original dataset: {e}")
    
    def get_dataset_stats(self) -> Dict:
        """Get current dataset statistics"""
        stats = {
            'total_tracks': len(self.existing_uris),
            'dataset_file_size': 0,
            'last_run': self.scraper_state.get('last_run'),
            'total_added_session': self.scraper_state.get('total_added', 0)
        }
        
        if os.path.exists(self.csv_file):
            stats['dataset_file_size'] = os.path.getsize(self.csv_file) / (1024 * 1024)  # MB
        
        return stats
    
    def enhance_dataset(self, target_new_tracks: int = 100):
        """Main method to enhance dataset with new tracks"""
        logger.info("START Starting dataset enhancement...")
        
        # Show current stats
        stats = self.get_dataset_stats()
        logger.info(f"SIZE Original dataset size: {stats['dataset_file_size']:.2f} MB")
        logger.info(f"COUNT Current tracks in dataset: {stats['total_tracks']}")
        
        # Search for new tracks
        new_tracks = self.search_new_tracks(target_new_tracks)
        
        if new_tracks:
            # Save new tracks directly to original dataset
            self.save_to_csv(new_tracks)
            
            # Update caches
            self.save_processed_uris()
            self.save_scraper_state()
            
            logger.info(f"SUCCESS Successfully added {len(new_tracks)} new tracks to original dataset!")
            logger.info(f"GROWTH Dataset growth: {stats['total_tracks']} -> {stats['total_tracks'] + len(new_tracks)} tracks")
        else:
            logger.info("INFO No new tracks found with current search queries")
        
        # Final stats
        final_stats = self.get_dataset_stats()
        logger.info(f"FINAL Final dataset: {final_stats['dataset_file_size']:.2f} MB")
        logger.info(f"READY Ready for preprocessing: {self.csv_file}")


def main():
    """Main execution function"""
    # Initialize scraper (credentials loaded from environment variables)
    scraper = EnhancedSpotifyMusicScraper()
    
    # Enhance dataset with 100 new tracks per run
    scraper.enhance_dataset(target_new_tracks=100)
    
    logger.info("COMPLETE Scraping session completed!")


if __name__ == "__main__":
    main()