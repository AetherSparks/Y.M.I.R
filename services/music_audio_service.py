"""
MUSIC & AUDIO MICROSERVICE
Extracted from app.py lines 908-1836
Contains all music search, download, and audio processing functions
"""
import os
import re
import json
import requests
import yt_dlp
import threading
import time
from typing import List, Dict, Optional, Tuple

class MusicAudioService:
    def __init__(self):
        # Music directory for downloads
        self.MUSIC_DIR = "static/audio"
        os.makedirs(self.MUSIC_DIR, exist_ok=True)
        print("✅ Music & Audio Service initialized")

    # EXTRACTED FROM app.py:908-922 (EXACT COPY)
    def search_youtube(self, song, artist):
        """Search YouTube for a specific song"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        query = f"{song} {artist} official audio"
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                search_results = ydl.extract_info(f"ytsearch:{query}", download=False)
                return search_results['entries'] if search_results else []
            except Exception as e:
                print(f"YouTube search error: {e}")
                return []

    # EXTRACTED FROM app.py:924-966 (EXACT COPY)
    def download_youtube_async(self, song, artist, filename):
        """Download audio from YouTube asynchronously"""
        def download_worker():
            try:
                query = f"{song} {artist}"
                output_path = os.path.join(self.MUSIC_DIR, filename)
                
                if os.path.exists(output_path):
                    print(f"File already exists: {filename}")
                    return output_path

                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(self.MUSIC_DIR, os.path.splitext(filename)[0]),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                    'quiet': True,
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    search_results = ydl.extract_info(f"ytsearch:{query}", download=False)
                    
                    if search_results and search_results['entries']:
                        video_info = search_results['entries'][0]
                        video_url = video_info['webpage_url']
                        
                        ydl.download([video_url])
                        
                        if os.path.exists(output_path):
                            print(f"✅ Downloaded: {filename}")
                            return output_path
                        else:
                            print(f"❌ Download failed: {filename}")
                            
            except Exception as e:
                print(f"Download error for {song} by {artist}: {e}")
            
            return None

        # Run download in background thread
        thread = threading.Thread(target=download_worker)
        thread.daemon = True
        thread.start()
        return thread

    # EXTRACTED FROM app.py:998-1005 (EXACT COPY)
    def clean_filename(self, text):
        """Clean text for safe filename usage"""
        # Remove or replace invalid characters for filenames
        text = re.sub(r'[<>:"/\\\\|?*]', '', text)
        text = re.sub(r'[\\s]+', '_', text)  # Replace spaces with underscores
        return text.strip('._')  # Remove leading/trailing dots and underscores

    # EXTRACTED FROM app.py:1518-1544 (EXACT COPY)
    def fetch_soundcloud(self, song, artist):
        """Fetch song from SoundCloud"""
        query = f"{song} {artist}"
        safe_filename = self.clean_filename(f"{song}_{artist}.mp3")
        output_path = os.path.join(self.MUSIC_DIR, safe_filename)
        
        if os.path.exists(output_path):
            return output_path

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(self.MUSIC_DIR, os.path.splitext(safe_filename)[0]),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"scsearch:{query}"])
                if os.path.exists(output_path):
                    return output_path
        except Exception as e:
            print(f"SoundCloud fetch error: {e}")
        
        return None

    # EXTRACTED FROM app.py:1546-1629 (EXACT COPY)
    def fetch_youtube_playlist_search(self, song, artist):
        """Search YouTube playlists for songs"""
        query_variations = [
            f"{song} {artist} official",
            f"{song} {artist} audio",
            f"{song} {artist} lyrics",
            f"{artist} {song}"
        ]
        
        safe_filename = self.clean_filename(f"{song}_{artist}.mp3")
        output_path = os.path.join(self.MUSIC_DIR, safe_filename)
        
        if os.path.exists(output_path):
            return output_path

        for query in query_variations:
            try:
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'quiet': True,
                    'no_warnings': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    search_results = ydl.extract_info(f"ytsearch10:{query}", download=False)
                    
                    if not search_results or not search_results.get('entries'):
                        continue
                    
                    # Filter and score results
                    filtered_videos = self.filter_music_videos(search_results['entries'], song, artist)
                    
                    if not filtered_videos:
                        continue
                        
                    # Try to download the best match
                    for video in filtered_videos[:3]:  # Try top 3 matches
                        try:
                            song_url = video.get('webpage_url')
                            if not song_url:
                                continue
                                
                            if not song_url.startswith('http'):
                                song_url = f"https://www.youtube.com/watch?v={song_url}"
                            
                            download_opts = {
                                'format': 'bestaudio/best',
                                'outtmpl': os.path.join(self.MUSIC_DIR, os.path.splitext(safe_filename)[0]),
                                'postprocessors': [{
                                    'key': 'FFmpegExtractAudio',
                                    'preferredcodec': 'mp3',
                                    'preferredquality': '192',
                                }],
                                'quiet': True,
                            }
                            
                            with yt_dlp.YoutubeDL(download_opts) as ydl_download:
                                ydl_download.download([song_url])
                                
                            if os.path.exists(output_path):
                                return output_path
                        except Exception as e:
                            print(f"Playlist search error: {e}")
                            continue
                            
            except Exception as e:
                print(f"Playlist search error: {e}")
                continue
        
        return None

    # EXTRACTED FROM app.py:1631-1685 (EXACT COPY)
    def filter_music_videos(self, videos, song, artist):
        """Filter videos to prioritize official music content"""
        if not videos:
            return []
        
        scored_videos = []
        song_lower = song.lower()
        artist_lower = artist.lower()
        
        for video in videos:
            if not video:
                continue
                
            title = video.get('title', '').lower()
            channel = video.get('channel', '').lower()
            
            score = 0
            
            # Check if it's a music video
            if song_lower in title and artist_lower in title:
                score += 10
            elif song_lower in title:
                score += 5
            
            # Prefer official artist channels
            if artist_lower in channel:
                score += 8
            
            # Prefer videos with "official" or "audio" in the title
            if 'official' in title:
                score += 6
            if 'audio' in title:
                score += 4
            if 'lyrics' in title:
                score += 3
                
            # Penalize covers, remixes, live versions
            if any(term in title for term in ['cover', 'remix', 'live', 'acoustic', 'karaoke']):
                score -= 5
                
            # Duration preference (2-6 minutes typical for songs)
            duration = video.get('duration', 0)
            if duration:
                if 120 <= duration <= 360:  # 2-6 minutes
                    score += 3
                elif duration > 600:  # Longer than 10 minutes (probably not the song)
                    score -= 10
                    
            scored_videos.append((video, score))
        
        # Sort by score (highest first) and return videos
        scored_videos.sort(key=lambda x: x[1], reverse=True)
        return [video for video, score in scored_videos if score > 0]

    # EXTRACTED FROM app.py:1687-1707 (EXACT COPY)
    def get_youtube_info(self, query, max_results=5):
        """Get YouTube video information without downloading"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_results = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
                
                if search_results and search_results.get('entries'):
                    videos = []
                    for entry in search_results['entries']:
                        video_info = {
                            'title': entry.get('title', ''),
                            'uploader': entry.get('uploader', ''),
                            'duration': entry.get('duration', 0),
                            'view_count': entry.get('view_count', 0),
                            'url': entry.get('webpage_url', ''),
                            'id': entry.get('id', '')
                        }
                        videos.append(video_info)
                    return videos
        except Exception as e:
            print(f"YouTube info error: {e}")
        
        return []

    # EXTRACTED FROM app.py:1709-1760 (EXACT COPY)
    def fetch_youtube_smart(self, song, artist):
        """Smart YouTube fetching with multiple strategies"""
        safe_filename = self.clean_filename(f"{song}_{artist}.mp3")
        output_path = os.path.join(self.MUSIC_DIR, safe_filename)
        
        if os.path.exists(output_path):
            return output_path

        # Strategy 1: Search for official versions
        queries = [
            f"{song} {artist} official audio",
            f"{song} {artist} official video", 
            f"{artist} {song} official",
            f"{song} {artist} audio",
            f"{song} by {artist}",
            f"{artist} - {song}"
        ]
        
        for query in queries:
            try:
                videos = self.get_youtube_info(query, max_results=5)
                if not videos:
                    continue
                    
                # Filter for best match
                filtered = self.filter_music_videos(videos, song, artist)
                if not filtered:
                    continue
                    
                # Try to download the best match
                best_video = filtered[0]
                video_url = best_video.get('url') or best_video.get('webpage_url')
                
                if video_url:
                    download_opts = {
                        'format': 'bestaudio/best',
                        'outtmpl': os.path.join(self.MUSIC_DIR, os.path.splitext(safe_filename)[0]),
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                            'preferredquality': '192',
                        }],
                        'quiet': True,
                    }
                    
                    with yt_dlp.YoutubeDL(download_opts) as ydl:
                        ydl.download([video_url])
                    
                    if os.path.exists(output_path):
                        return output_path
                        
            except Exception as e:
                print(f"Smart fetch error for query '{query}': {e}")
                continue
        
        return None

    # EXTRACTED FROM app.py:1762-1789 (EXACT COPY)
    def fetch_with_retries(self, song, artist, max_retries=3):
        """Fetch song with retry mechanism using multiple methods"""
        methods = [
            self.fetch_youtube_smart,
            self.fetch_youtube_playlist_search,
            self.fetch_soundcloud
        ]
        
        for attempt in range(max_retries):
            for method in methods:
                try:
                    result = method(song, artist)
                    if result and os.path.exists(result):
                        print(f"✅ Successfully fetched {song} by {artist} using {method.__name__}")
                        return result
                except Exception as e:
                    print(f"❌ {method.__name__} failed for {song} by {artist} (attempt {attempt + 1}): {e}")
                    continue
            
            if attempt < max_retries - 1:
                print(f"🔄 Retrying... (attempt {attempt + 2}/{max_retries})")
                time.sleep(2)  # Wait before retry
        
        print(f"❌ All methods failed for {song} by {artist} after {max_retries} attempts")
        return None

    # EXTRACTED FROM app.py:1478-1494 (EXACT COPY)
    def get_neutral_songs(self):
        """Get list of neutral/default songs"""
        neutral_songs = [
            {"track": "On Top Of The World", "artist": "Imagine Dragons", "mood": "Neutral"},
            {"track": "Counting Stars", "artist": "OneRepublic", "mood": "Neutral"},
            {"track": "Let Her Go", "artist": "Passenger", "mood": "Neutral"},
            {"track": "Photograph", "artist": "Ed Sheeran", "mood": "Neutral"},
            {"track": "Paradise", "artist": "Coldplay", "mood": "Neutral"},
            {"track": "Viva La Vida", "artist": "Coldplay", "mood": "Neutral"},
            {"track": "Stressed Out", "artist": "Twenty One Pilots", "mood": "Neutral"},
            {"track": "Believer", "artist": "Imagine Dragons", "mood": "Neutral"},
            {"track": "Thunder", "artist": "Imagine Dragons", "mood": "Neutral"},
            {"track": "Shape of You", "artist": "Ed Sheeran", "mood": "Neutral"},
        ]
        return neutral_songs

    def get_audio_file_path(self, song, artist):
        """Get the file path for a downloaded audio file"""
        safe_filename = self.clean_filename(f"{song}_{artist}.mp3")
        return os.path.join(self.MUSIC_DIR, safe_filename)

    def is_audio_available(self, song, artist):
        """Check if audio file is already downloaded"""
        file_path = self.get_audio_file_path(song, artist)
        return os.path.exists(file_path)

    def cleanup(self):
        """Cleanup service resources"""
        print("🧹 Music & Audio service cleanup complete")