# import numpy as np
# import torch
# import librosa
# import soundfile as sf
# import pygame
# import time
# import os
# import pyttsx3  # Simpler TTS library
# from pydub import AudioSegment
# from pydub.silence import split_on_silence

# import numpy as np
# import librosa
# import soundfile as sf
# import pygame
# import time
# import os
# import pyttsx3
# from pydub import AudioSegment
# from pydub.playback import play
# from pydub.effects import low_pass_filter, high_pass_filter
# import scipy.signal as signal
# import random

# class AISingerGenerator:
#     def __init__(self):
#         # Initialize the pyttsx3 engine
#         self.engine = pyttsx3.init()
        
#         # Configure voice properties - slower for singing
#         self.engine.setProperty('rate', 120)  # Even slower for singing
#         voices = self.engine.getProperty('voices')
#         # Choose a voice that sounds good for singing
#         if len(voices) > 1:
#             self.engine.setProperty('voice', voices[1].id)  # Usually the second voice is female
        
#         # Parameters for singing style
#         self.pitch_factor = 1.0
#         self.duration_factor = 1.2
#         self.sample_rate = 22050
#         self.vibrato_rate = 5.5  # Hz - vibrato oscillation rate
#         self.vibrato_depth = 0.3  # Depth of vibrato effect
        
#     def analyze_backing_track(self, backing_track_path):
#         print("Analyzing backing track...")
#         y, sr = librosa.load(backing_track_path, sr=self.sample_rate)
        
#         # Extract tempo and beat positions
#         tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
#         # Convert tempo to a scalar if it's a numpy array
#         if isinstance(tempo, np.ndarray):
#             tempo = float(tempo[0])  # Take the first value if it's an array
        
#         beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
#         # Extract musical key with more robust analysis
#         chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
#         chroma_norm = np.mean(chroma, axis=1)
#         key_index = np.argmax(chroma_norm)
#         keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
#         estimated_key = keys[key_index]
        
#         # Determine if major or minor based on relative strength of major/minor thirds
#         major_minor = "major" if chroma_norm[(key_index + 4) % 12] > chroma_norm[(key_index + 3) % 12] else "minor"
        
#         print(f"Track analysis complete: Tempo: {tempo:.1f} BPM, Key: {estimated_key} {major_minor}")
        
#         return {
#             'tempo': tempo,
#             'beat_times': beat_times,
#             'key': estimated_key,
#             'mode': major_minor,
#             'duration': len(y) / sr,
#             'key_index': key_index
#         }
        
#     def align_lyrics_to_beats(self, lyrics, beat_times, tempo):
#         """
#         More sophisticated alignment of lyrics to beat times
#         """
#         lines = lyrics.strip().split('\n')
#         words = []
        
#         # Process special markers like [Chorus], [Verse], etc.
#         current_section = "Intro"
#         for line in lines:
#             line = line.strip()
#             if not line:
#                 continue
                
#             if line.startswith('[') and line.endswith(']'):
#                 current_section = line[1:-1]
#                 continue
                
#             for word in line.split():
#                 words.append((word, current_section))
        
#         # Calculate words per beat based on tempo
#         # Faster tempos need more words per beat
#         tempo_factor = tempo / 100.0  # Normalize around 100 BPM
#         words_per_beat = max(1, round(len(words) / len(beat_times) * tempo_factor))
        
#         aligned_lyrics = []
#         for i in range(0, len(words), words_per_beat):
#             word_chunk = words[i:i+words_per_beat]
#             text = ' '.join([w[0] for w in word_chunk])
#             section = word_chunk[0][1] if word_chunk else "Unknown"
            
#             beat_index = min(i // words_per_beat, len(beat_times) - 1)
#             timing = beat_times[beat_index]
            
#             aligned_lyrics.append((text, timing, section))
        
#         return aligned_lyrics
    
#     def apply_vibrato(self, audio_segment, rate=5.5, depth=0.3):
#         """
#         Apply vibrato effect to make it sound more like singing
#         """
#         # Convert AudioSegment to numpy array
#         audio_array = np.array(audio_segment.get_array_of_samples())
#         sample_rate = audio_segment.frame_rate
        
#         # Create time array
#         t = np.arange(0, len(audio_array)) / sample_rate
        
#         # Generate vibrato using sine wave
#         vibrato = depth * np.sin(2.0 * np.pi * rate * t)
        
#         # Apply vibrato by phase modulation
#         # This is a simplified approach - real vibrato is more complex
#         modulated = np.zeros_like(audio_array, dtype=float)
#         for i in range(len(audio_array)):
#             # Use vibrato to slightly modify the sampling position
#             src_pos = i + vibrato[i] * sample_rate / 50
#             src_pos_int = int(src_pos)
#             if 0 <= src_pos_int < len(audio_array) - 1:
#                 # Linear interpolation
#                 frac = src_pos - src_pos_int
#                 modulated[i] = (1 - frac) * audio_array[src_pos_int] + frac * audio_array[src_pos_int + 1]
#             elif 0 <= src_pos_int < len(audio_array):
#                 modulated[i] = audio_array[src_pos_int]
        
#         # Convert back to same format as input
#         modulated = np.clip(modulated, np.iinfo(audio_array.dtype).min, np.iinfo(audio_array.dtype).max)
#         modulated = modulated.astype(audio_array.dtype)
        
#         # Create new AudioSegment
#         return audio_segment._spawn(modulated.tobytes())
    
#     def apply_musical_effects(self, audio_segment, section_type):
#         """
#         Apply different effects based on section type (verse, chorus, etc.)
#         """
#         # Apply base effects
#         # Add reverb (simple approximation with decay)
#         reverb_ms = 100
#         reverb = audio_segment.overlay(
#             audio_segment - 12,  # Quieter copy
#             position=reverb_ms  # Delayed by reverb_ms
#         )
        
#         # Section-specific effects
#         if "chorus" in section_type.lower():
#             # More reverb and brighter sound for chorus
#             reverb = reverb.overlay(
#                 audio_segment - 16,
#                 position=reverb_ms * 2
#             )
#             # Boost higher frequencies slightly
#             reverb = high_pass_filter(reverb, 300)
            
#         elif "verse" in section_type.lower():
#             # Clearer sound for verses
#             reverb = low_pass_filter(reverb, 8000)
        
#         elif "bridge" in section_type.lower():
#             # More dramatic effect for bridge
#             reverb = reverb.overlay(
#                 audio_segment - 14,
#                 position=reverb_ms * 1.5
#             )
        
#         return reverb
    
#     def adjust_pitch_for_key(self, audio_segment, key_index, section_type):
#         """
#         Adjust pitch according to the musical key and section type
#         """
#         # Define pitch shift based on key and section
#         # This is a simplified approach - real melodic mapping would be more complex
        
#         # Base pitch shift (semitones)
#         base_pitch = 0
        
#         # Section-specific pitch adjustments
#         if "chorus" in section_type.lower():
#             # Choruses often go higher
#             base_pitch += 2
#         elif "bridge" in section_type.lower():
#             # Bridges might go even higher
#             base_pitch += 3
            
#         # Add slight randomness for a more natural singing feel
#         # But keep it musical by quantizing to semitones
#         random_pitch = random.choice([0, 1, 2, -1, -2])
        
#         # Total pitch shift
#         pitch_shift = base_pitch + random_pitch
        
#         # Apply the pitch shift
#         if pitch_shift != 0:
#             # For pydub we need to modify playback speed
#             # This is a simplification - real pitch shifting would keep duration
#             speed_factor = 2 ** (pitch_shift / 12.0)  # 12 semitones in an octave
#             modified = audio_segment._spawn(audio_segment.raw_data, overrides={
#                 "frame_rate": int(audio_segment.frame_rate * speed_factor)
#             }).set_frame_rate(audio_segment.frame_rate)
            
#             return modified
            
#         return audio_segment
    
#     def generate_singing_voice(self, lyrics, backing_track_path, output_path="ai_singing_output.wav"):
#         """
#         Generate a singing voice from lyrics that aligns with the backing track
#         """
#         # Analyze the backing track
#         track_info = self.analyze_backing_track(backing_track_path)
        
#         # Align lyrics to beats with improved algorithm
#         aligned_lyrics = self.align_lyrics_to_beats(lyrics, track_info['beat_times'], track_info['tempo'])
        
#         print("Generating singing voice...")
        
#         # Generate each segment
#         temp_files = []
#         for i, (text, timing, section) in enumerate(aligned_lyrics):
#             if text.strip():
#                 temp_file = f"temp_vocal_{i}.wav"
                
#                 # Adjust TTS parameters based on section
#                 if "chorus" in section.lower():
#                     self.engine.setProperty('rate', 130)  # Slightly slower for chorus
#                 elif "verse" in section.lower():
#                     self.engine.setProperty('rate', 140)  # Normal for verse
#                 elif "bridge" in section.lower():
#                     self.engine.setProperty('rate', 120)  # Slowest for bridge
                
#                 # Save speech to file
#                 self.engine.save_to_file(text, temp_file)
#                 self.engine.runAndWait()
#                 temp_files.append((temp_file, section))
        
#         # Process and combine all segments
#         if temp_files:
#             # Combine audio segments with effects
#             combined = AudioSegment.empty()
            
#             for temp_file, section in temp_files:
#                 if os.path.exists(temp_file):
#                     # Load segment
#                     segment = AudioSegment.from_file(temp_file)
                    
#                     # Apply vocal effects to make it sound like singing
#                     # 1. Slow down for singing-like effect
#                     segment = segment._spawn(segment.raw_data, overrides={
#                        "frame_rate": int(segment.frame_rate * 0.85)
#                     }).set_frame_rate(segment.frame_rate)
                    
#                     # 2. Apply pitch adjustments based on musical key
#                     segment = self.adjust_pitch_for_key(segment, track_info['key_index'], section)
                    
#                     # 3. Add vibrato effect (characteristic of singing)
#                     segment = self.apply_vibrato(segment, rate=self.vibrato_rate, depth=self.vibrato_depth)
                    
#                     # 4. Apply section-specific effects
#                     segment = self.apply_musical_effects(segment, section)
                    
#                     # Add to combined track
#                     combined += segment
                    
#             # Save the vocals
#             combined.export("temp_vocals.wav", format="wav")
            
#             # Mix with backing track with advanced mixing
#             self.mix_vocals_with_backing(
#                 vocals_path="temp_vocals.wav",
#                 backing_path=backing_track_path,
#                 output_path=output_path,
#                 track_info=track_info
#             )
            
#             # Clean up temporary files
#             for temp_file, _ in temp_files:
#                 if os.path.exists(temp_file):
#                     os.remove(temp_file)
#             if os.path.exists("temp_vocals.wav"):
#                 os.remove("temp_vocals.wav")
                
#             print(f"AI singing generated and saved to {output_path}")
#             return output_path
#         else:
#             print("No valid segments to generate audio")
#             return None
    
#     def mix_vocals_with_backing(self, vocals_path, backing_path, output_path, track_info=None):
#         """
#         Mix the generated vocals with the backing track with enhanced mixing
#         """
#         print("Mixing vocals with advanced techniques...")
        
#         # Load audio files
#         vocals = AudioSegment.from_file(vocals_path)
#         backing = AudioSegment.from_file(backing_path)
        
#         # Apply EQ to vocals for better clarity
#         # High-pass to remove low rumble
#         vocals = high_pass_filter(vocals, 120)
#         # Boost presence range for clarity (simplified EQ)
        
#         # Match durations
#         if len(vocals) < len(backing):
#             silence = AudioSegment.silent(duration=len(backing) - len(vocals))
#             vocals = vocals + silence
#         else:
#             vocals = vocals[:len(backing)]
        
#         # Apply compression to vocals (simulated with volume adjustments)
#         # Normalize to consistent volume
#         vocals = vocals.normalize()
        
#         # Adjust volume balance
#         vocals = vocals - 2  # Reduce vocals volume 
#         backing = backing - 12  # Reduce backing track more
        
#         # Mix tracks with improved balance
#         mixed = backing.overlay(vocals)
        
#         # Final master processing
#         mixed = mixed.normalize()
        
#         # Export final mix
#         mixed.export(output_path, format="wav")
#         print(f"Mixed audio saved to {output_path}")

#     def play_audio(self, audio_path):
#         """
#         Play the generated audio file
#         """
#         pygame.mixer.init()
#         pygame.mixer.music.load(audio_path)
#         print(f"Playing: {audio_path}")
#         pygame.mixer.music.play()
        
#         # Wait for playback to finish
#         while pygame.mixer.music.get_busy():
#             time.sleep(1)

# # Example usage
# if __name__ == "__main__":
#     # Create an instance of the singer generator
#     singer = AISingerGenerator()
    
#     # Example lyrics and backing track (you would provide your own)
#     lyrics = '''Walking through the city lights
#     Holding hands on Friday night
#     Your smile cuts through all the noise
#     In this moment, pure joy
    
#     [Chorus]
#     We're shining bright like the stars above
#     Filled with hope, filled with love
#     Every heartbeat sings your name
#     This feeling I can't explain
    
#     [Verse 2]
#     Memories we're making now
#     Promise me we'll work it out
#     Through the good and stormy days
#     I'll be yours in every way
    
#     [Chorus]
#     We're shining bright like the stars above
#     Filled with hope, filled with love
#     Every heartbeat sings your name
#     This feeling I can't explain
    
#     [Bridge]
#     Time stands still when you're with me
#     Together is where we're meant to be
    
#     [Chorus]
#     We're shining bright like the stars above
#     Filled with hope, filled with love
#     Every heartbeat sings your name
#     This feeling I can't explain
    
#     '''
    
#     backing_track_path = "melodic_integrated_song.wav"  # Replace with your path
    
#     # Generate singing
#     output_file = singer.generate_singing_voice(lyrics, backing_track_path)
    
#     # Play the result if generation was successful
#     if output_file:
#         singer.play_audio(output_file)


































































































































































































































































































# import random
# import sys
# import time
# import os
# import traceback

# # Debug system info
# print(f"Python version: {sys.version}")
# print(f"System platform: {sys.platform}")

# # Import with error handling
# def safe_import(module_name):
#     try:
#         if module_name == 'numpy':
#             import numpy as np
#             print(f"✓ Successfully imported {module_name} {np.__version__}")
#             return np
#         elif module_name == 'torch':
#             import torch
#             print(f"✓ Successfully imported {module_name} {torch.__version__}")
#             return torch
#         elif module_name == 'librosa':
#             import librosa
#             print(f"✓ Successfully imported {module_name} {librosa.__version__}")
#             return librosa
#         elif module_name == 'soundfile':
#             import soundfile as sf
#             print(f"✓ Successfully imported {module_name} {sf.__version__}")
#             return sf
#         elif module_name == 'pygame':
#             import pygame
#             print(f"✓ Successfully imported {module_name} {pygame.__version__}")
#             return pygame
#         elif module_name == 'pyttsx3':
#             import pyttsx3
#             print(f"✓ Successfully imported {module_name}")
#             return pyttsx3
#         elif module_name == 'pydub':
#             from pydub import AudioSegment, playback, effects
#             print(f"✓ Successfully imported {module_name}")
#             return (AudioSegment, playback, effects)
#         elif module_name == 'scipy':
#             import scipy
#             print(f"✓ Successfully imported {module_name} {scipy.__version__}")
#             return scipy
#         elif module_name == 'pyrubberband':
#             import pyrubberband as pyrb
#             print(f"✓ Successfully imported {module_name}")
#             return pyrb
#         elif module_name == 'sounddevice':
#             import sounddevice as sd
#             print(f"✓ Successfully imported {module_name} {sd.__version__}")
#             return sd
#         elif module_name == 'pyworld':
#             import pyworld as pw
#             print(f"✓ Successfully imported {module_name}")
#             return pw
#         elif module_name == 'parselmouth':
#             import parselmouth
#             print(f"✓ Successfully imported {module_name} {parselmouth.__version__}")
#             return parselmouth
#         else:
#             raise ImportError(f"Unknown module: {module_name}")
#     except ImportError as e:
#         print(f"✗ Failed to import {module_name}: {e}")
#         return None

# # Try to import all required modules
# np = safe_import('numpy')
# torch = safe_import('torch')
# librosa = safe_import('librosa')
# sf = safe_import('soundfile')
# pygame = safe_import('pygame')
# pyttsx3 = safe_import('pyttsx3')
# AudioSegment, play, effects = safe_import('pydub')
# scipy = safe_import('scipy')
# pyrb = safe_import('pyrubberband')
# sd = safe_import('sounddevice')
# pw = safe_import('pyworld')
# parselmouth = safe_import('parselmouth')

# # Check if any critical module is missing
# critical_modules = [np, librosa, pyttsx3, AudioSegment]
# if None in critical_modules:
#     print("ERROR: Critical modules missing. Cannot continue.")
#     sys.exit(1)

# # Now import the specific components needed
# from scipy.interpolate import interp1d
# from pydub.effects import low_pass_filter, high_pass_filter


# class AISingerGenerator:
#     def __init__(self):
#         print("Initializing AISingerGenerator...")
        
#         # Initialize the pyttsx3 engine
#         try:
#             self.engine = pyttsx3.init()
#             print("✓ TTS engine initialized")
            
#             # Configure voice properties - slower for singing
#             self.engine.setProperty('rate', 120)
            
#             # Check available voices
#             voices = self.engine.getProperty('voices')
#             print(f"Available voices: {len(voices)}")
            
#             if len(voices) > 1:
#                 self.engine.setProperty('voice', voices[1].id)  # Usually female voice
#                 print(f"Selected voice ID: {voices[1].id}")
#             else:
#                 print("Only one voice available, using default")
                
#         except Exception as e:
#             print(f"ERROR initializing TTS engine: {e}")
#             raise
        
#         # Advanced parameters for singing style
#         self.pitch_factor = 1.0
#         self.duration_factor = 1.2
#         self.sample_rate = 44100  # Higher sample rate for better quality
#         self.vibrato_rate = 5.5  # Hz - vibrato oscillation rate
#         self.vibrato_depth = 0.3  # Depth of vibrato effect
        
#         # Vocal characteristics
#         self.formant_shift = 1.0
#         self.breathiness = 0.2
#         self.vocal_tension = 0.5
#         self.jitter = 0.005
#         self.shimmer = 0.01
        
#         print("AISingerGenerator initialized successfully")
    
#     # 1. Fix librosa.feature.sync issue in analyze_backing_track method
#     def analyze_backing_track(self, backing_track_path):
#         """
#         Comprehensive musical analysis of the backing track
#         """
#         print(f"Analyzing backing track: {backing_track_path}")
        
#         # Check if file exists
#         if not os.path.exists(backing_track_path):
#             print(f"ERROR: Backing track file not found: {backing_track_path}")
#             raise FileNotFoundError(f"File not found: {backing_track_path}")
        
#         try:
#             y, sr = librosa.load(backing_track_path, sr=self.sample_rate, mono=True)
#             print(f"✓ Loaded audio file. Duration: {len(y)/sr:.2f}s, Sample rate: {sr}Hz")
            
#             # Check if audio is valid
#             if len(y) < sr:  # Less than 1 second
#                 print("WARNING: Audio file too short, might give inaccurate analysis")
            
#             # Extract tempo with advanced beat tracking
#             print("Extracting tempo...")
#             onset_env = librosa.onset.onset_strength(y=y, sr=sr)
#             tempo_distribution = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, 
#                                                 aggregate=None)
#             # Get the most likely tempo
#             tempo = float(tempo_distribution[np.argmax(tempo_distribution)])
#             print(f"✓ Detected tempo: {tempo:.1f} BPM")
            
#             # Use dynamic beat tracking for better accuracy
#             print("Tracking beats...")
#             beat_frames = librosa.beat.beat_track(
#                 onset_envelope=onset_env,
#                 sr=sr,
#                 trim=False,
#                 start_bpm=tempo,
#                 tightness=100
#             )[1]
            
#             beat_times = librosa.frames_to_time(beat_frames, sr=sr)
#             print(f"✓ Found {len(beat_times)} beats")
            
#             # Enhanced harmonic analysis
#             print("Performing harmonic analysis...")
#             y_harmonic = librosa.effects.harmonic(y)
            
#             # Use multiple chroma features for more accurate key detection
#             chroma_cqt = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
#             chroma_stft = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
            
#             # Weighted combination of chroma features
#             chroma_combined = 0.6 * chroma_cqt + 0.4 * chroma_stft
#             chroma_norm = np.mean(chroma_combined, axis=1)
            
#             # Krumhansl-Schmuckler key finding algorithm
#             major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
#             minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
#             # Normalize profiles
#             major_profile = major_profile / major_profile.sum()
#             minor_profile = minor_profile / minor_profile.sum()
            
#             # Calculate correlation with profiles
#             major_corr = []
#             minor_corr = []
#             for i in range(12):
#                 rolled_chroma = np.roll(chroma_norm, i)
#                 major_corr.append(np.corrcoef(rolled_chroma, major_profile)[0, 1])
#                 minor_corr.append(np.corrcoef(rolled_chroma, minor_profile)[0, 1])
            
#             major_key_index = np.argmax(major_corr)
#             minor_key_index = np.argmax(minor_corr)
            
#             if major_corr[major_key_index] > minor_corr[minor_key_index]:
#                 key_index = major_key_index
#                 mode = "major"
#             else:
#                 key_index = minor_key_index
#                 mode = "minor"
            
#             keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
#             estimated_key = keys[key_index]
#             print(f"✓ Detected key: {estimated_key} {mode}")
            
#             # Detect chord progressions
#             print("Analyzing chord progression...")
#             # Fix: Use librosa.util.sync instead of librosa.feature.sync
#             chroma_frames = librosa.util.sync(chroma_combined, beat_frames)
            
#             # Extract chord qualities based on triad patterns
#             chord_progression = []
#             for frame in range(min(chroma_frames.shape[1], 10)):  # Debug: just check first 10 frames
#                 chord_vector = chroma_frames[:, frame]
#                 root = np.argmax(chord_vector)
                
#                 # Check for major/minor/diminished triads
#                 major_third = (root + 4) % 12
#                 minor_third = (root + 3) % 12
#                 fifth = (root + 7) % 12
                
#                 if chord_vector[major_third] > 0.5 and chord_vector[fifth] > 0.5:
#                     chord_progression.append(f"{keys[root]}maj")
#                 elif chord_vector[minor_third] > 0.5 and chord_vector[fifth] > 0.5:
#                     chord_progression.append(f"{keys[root]}min")
#                 else:
#                     chord_progression.append(f"{keys[root]}")
            
#             print(f"✓ Analyzed {len(chord_progression)} chords")
            
#             # Rest of function remains the same...
#             # Extract dynamic contour
#             print("Extracting dynamics...")
#             dynamics = librosa.feature.rms(y=y).flatten()
#             dynamics_smooth = scipy.ndimage.gaussian_filter1d(dynamics, sigma=4)
            
#             # Detect song sections (verse, chorus) based on spectral contrast
#             print("Identifying song sections...")
#             spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
#             contrast_avg = np.mean(spectral_contrast, axis=0)
#             contrast_smooth = scipy.ndimage.gaussian_filter1d(contrast_avg, sigma=4)
            
#             # Identify high energy sections (likely choruses)
#             threshold = np.percentile(contrast_smooth, 70)
#             high_energy_frames = contrast_smooth > threshold
            
#             segment_boundaries = librosa.segment.agglomerative(spectral_contrast, 4)
#             segment_times = librosa.frames_to_time(segment_boundaries, sr=sr)
            
#             print(f"✓ Identified {len(segment_times)} major sections in the track")
            
#             analysis_result = {
#                 'tempo': tempo,
#                 'beat_times': beat_times,
#                 'key': estimated_key,
#                 'mode': mode,
#                 'duration': len(y) / sr,
#                 'key_index': key_index,
#                 'chord_progression': chord_progression,
#                 'dynamics': dynamics_smooth,
#                 'segment_boundaries': segment_times,
#                 'high_energy_frames': high_energy_frames
#             }
            
#             print("Track analysis complete!")
#             return analysis_result
            
#         except Exception as e:
#             print(f"ERROR during backing track analysis: {e}")
#             # Provide a minimal fallback for testing
#             print("Using fallback track analysis values")
#             return {
#                 'tempo': 120.0,
#                 'beat_times': np.arange(0, 60, 0.5),  # 2 beats per second at 120 BPM
#                 'key': 'C',
#                 'mode': 'major',
#                 'duration': 60.0,
#                 'key_index': 0,
#                 'chord_progression': ['Cmaj', 'Gmaj', 'Amin', 'Fmaj'] * 15,  # Simple progression
#                 'dynamics': np.ones(100),
#                 'segment_boundaries': [0, 15, 30, 45],
#                 'high_energy_frames': np.zeros(100, dtype=bool)
#             }

    
#     def align_lyrics_to_beats(self, lyrics, beat_times, tempo, chord_progression=None, track_dynamics=None):
#         """
#         Advanced musical-linguistic alignment algorithm
#         """
#         print(f"Aligning lyrics to {len(beat_times)} beats at {tempo} BPM")
        
#         # Debug: Print first few lines of lyrics
#         first_few_lines = '\n'.join(lyrics.strip().split('\n')[:5])
#         print(f"First few lines of lyrics:\n{first_few_lines}...")
        
#         # Validate inputs
#         if not lyrics.strip():
#             print("ERROR: No lyrics provided")
#             return []
            
#         if len(beat_times) < 2:
#             print("ERROR: Not enough beat times for alignment")
#             return []
        
#         lines = lyrics.strip().split('\n')
#         words = []
        
#         # Process special markers like [Chorus], [Verse], etc.
#         current_section = "Intro"
#         section_importance = {"Intro": 0.6, "Verse": 0.7, "Chorus": 1.0, "Bridge": 0.9, "Outro": 0.5}
        
#         # Map words with their syllable counts for better rhythmic alignment
#         word_to_syllables = {}
        
#         print("Processing lyrics...")
#         for line in lines:
#             line = line.strip()
#             if not line:
#                 continue
                
#             if line.startswith('[') and line.endswith(']'):
#                 current_section = line[1:-1]
#                 print(f"Found section marker: {current_section}")
#                 continue
            
#             # Analyze the line's natural rhythm
#             line_words = line.split()
#             for word in line_words:
#                 # Count syllables in each word (simple approximation)
#                 clean_word = ''.join(c for c in word.lower() if c.isalpha())
#                 syllable_count = self._count_syllables(clean_word)
#                 word_to_syllables[word] = syllable_count
#                 words.append((word, current_section, syllable_count))
        
#         print(f"✓ Processed {len(words)} words across {len(set(w[1] for w in words))} sections")
        
#         # Calculate dynamic syllable-to-beat mapping based on tempo
#         # Faster tempos need more syllables per beat
#         tempo_factor = tempo / 100.0  # Normalize around 100 BPM
        
#         # Get total syllables
#         total_syllables = sum(w[2] for w in words)
#         avg_syllables_per_beat = total_syllables / len(beat_times) if len(beat_times) > 0 else 1.0
        
#         print(f"Total syllables: {total_syllables}, Avg syllables per beat: {avg_syllables_per_beat:.2f}")
        
#         # Create adaptive distribution based on section importance
#         aligned_lyrics = []
#         current_beat = 0
#         current_word_index = 0
#         current_syllable_count = 0
#         current_text_chunk = []
        
#         print("Aligning words to beats...")
#         while current_word_index < len(words) and current_beat < len(beat_times):
#             word, section, syllables = words[current_word_index]
            
#             # Determine weight for this section
#             weight = section_importance.get(section, 0.7)
            
#             # Calculate target syllables for this beat based on section
#             target_syllables = max(1, avg_syllables_per_beat * weight * tempo_factor)
            
#             current_text_chunk.append(word)
#             current_syllable_count += syllables
#             current_word_index += 1
            
#             # Check if we've reached our target syllable count for this beat
#             if current_syllable_count >= target_syllables or current_word_index >= len(words):
#                 text = ' '.join(current_text_chunk)
                
#                 # Add musical expression markers based on section
#                 if section == "Chorus":
#                     text = self._add_expression_markers(text, "emphatic")
#                 elif section == "Bridge":
#                     text = self._add_expression_markers(text, "dramatic")
#                 elif section == "Verse":
#                     text = self._add_expression_markers(text, "narrative")
                
#                 timing = beat_times[current_beat]
                
#                 # Get chord at this point if available
#                 current_chord = None
#                 if chord_progression and current_beat < len(chord_progression):
#                     current_chord = chord_progression[current_beat]
                
#                 # Get dynamics at this point
#                 current_dynamic = None
#                 if track_dynamics is not None:
#                     beat_frame = min(int(timing * self.sample_rate), len(track_dynamics)-1)
#                     if 0 <= beat_frame < len(track_dynamics):
#                         current_dynamic = track_dynamics[beat_frame]
#                     else:
#                         print(f"WARNING: Beat frame {beat_frame} out of range for dynamics array of length {len(track_dynamics)}")
                
#                 aligned_lyrics.append((text, timing, section, current_chord, current_dynamic))
                
#                 # Reset for next beat
#                 current_text_chunk = []
#                 current_syllable_count = 0
#                 current_beat += 1
                
#                 # Debug: print some alignments
#                 if len(aligned_lyrics) <= 3 or len(aligned_lyrics) % 20 == 0:
#                     print(f"Aligned: '{text}' at {timing:.2f}s ({section})")
        
#         print(f"✓ Created {len(aligned_lyrics)} aligned lyric segments")
#         return aligned_lyrics
    
#     def _count_syllables(self, word):
#         """
#         Count syllables in a word with advanced linguistic rules
#         """
#         # More sophisticated syllable counting
#         vowels = "aeiouy"
#         word = word.lower()
#         count = 0
        
#         # Special case for empty strings
#         if len(word) == 0:
#             return 0
            
#         # Count vowel groups
#         if word[0] in vowels:
#             count += 1
        
#         for i in range(1, len(word)):
#             if word[i] in vowels and word[i-1] not in vowels:
#                 count += 1
                
#         # Adjust for common patterns
#         if word.endswith("e") and len(word) > 2 and word[-2] not in vowels:
#             count -= 1
#         if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
#             count += 1
#         if word.endswith("es") or word.endswith("ed") and len(word) > 2:
#             if word[-3] not in vowels:
#                 count -= 1
        
#         # Ensure at least one syllable per word
#         return max(1, count)
    
#     def _add_expression_markers(self, text, style):
#         """
#         Add vocal expression markers to guide the TTS engine
#         """
#         words = text.split()
#         if not words:
#             return text
            
#         if style == "emphatic":
#             # Add emphasis to key words
#             for i in range(len(words)):
#                 if len(words[i]) > 3 and random.random() > 0.6:
#                     words[i] = words[i].upper()
        
#         elif style == "dramatic":
#             # Add pauses and stretching
#             for i in range(len(words)):
#                 if random.random() > 0.7:
#                     vowels = "aeiouAEIOU"
#                     new_word = ""
#                     for c in words[i]:
#                         new_word += c
#                         if c.lower() in vowels and random.random() > 0.5:
#                             new_word += c * random.randint(1, 2)
#                     words[i] = new_word
        
#         elif style == "narrative":
#             # More natural speech patterns
#             pass
        
#         return " ".join(words)
    
#     def extract_formants(self, audio_array, sample_rate):
#         """
#         Extract vocal formants using standard signal processing instead of Praat
#         """
#         print(f"Extracting formants from audio ({len(audio_array)} samples)")
        
#         try:
#             # Convert to float for processing
#             audio_float = audio_array.astype(float) / 32768.0
            
#             # Use simple LPC (Linear Predictive Coding) for formant estimation
#             # This is a simpler alternative to Praat's formant extraction
#             frame_length = int(0.025 * sample_rate)  # 25ms frames
#             hop_length = int(0.01 * sample_rate)     # 10ms hop
            
#             num_frames = 1 + (len(audio_float) - frame_length) // hop_length
#             formants = np.zeros((num_frames, 5))
#             times = np.zeros(num_frames)
            
#             print(f"Processing {num_frames} frames for formant extraction")
            
#             # Process a subset of frames for debugging
#             max_frames_to_process = min(num_frames, 20)  # Process at most 20 frames for debugging
            
#             for i in range(max_frames_to_process):
#                 start = i * hop_length
#                 end = start + frame_length
#                 if end <= len(audio_float):
#                     frame = audio_float[start:end] * np.hamming(frame_length)
                    
#                     # Basic LPC analysis
#                     try:
#                         A = librosa.lpc(frame, order=10)
                        
#                         # Find roots of the LPC polynomial
#                         roots = np.roots(A)
#                         roots = roots[np.imag(roots) > 0]
                        
#                         # Convert roots to frequencies
#                         angles = np.arctan2(np.imag(roots), np.real(roots))
#                         frequencies = angles * sample_rate / (2 * np.pi)
                        
#                         # Sort by frequency
#                         frequencies = np.sort(frequencies)
                        
#                         # Store the first 5 formants (or fewer if not enough found)
#                         for j in range(min(5, len(frequencies))):
#                             if j < len(frequencies):
#                                 formants[i, j] = frequencies[j]
                        
#                         times[i] = (start + frame_length/2) / sample_rate
#                     except Exception as e:
#                         print(f"WARNING: Error in LPC analysis for frame {i}: {e}")
            
#             print(f"✓ Extracted formants from {max_frames_to_process} frames")
#             return formants, times
            
#         except Exception as e:
#             print(f"ERROR during formant extraction: {e}")
#             # Return empty arrays as fallback
#             return np.zeros((1, 5)), np.zeros(1)
    
#     # 2. Fix the fix_length() error in modify_formants method
#     def modify_formants(self, audio_array, sample_rate, shift_factor=1.0, maintain_energy=True):
#         """
#         Modify vocal formants using standard DSP techniques instead of Praat
#         """
#         print(f"Modifying formants with shift factor {shift_factor}")
        
#         try:
#             # Convert to float
#             audio_float = audio_array.astype(float) / 32768.0
            
#             # For formant shifting without Praat, we'll use a simpler approach
#             # with spectral envelope manipulation
            
#             # STFT - Short-time Fourier transform
#             n_fft = 2048
#             hop_length = int(0.01 * sample_rate)  # 10ms hop
            
#             print("Computing STFT...")
#             # Get magnitude and phase
#             stft = librosa.stft(audio_float, n_fft=n_fft, hop_length=hop_length)
#             magnitude, phase = librosa.magphase(stft)
            
#             print(f"STFT shape: {stft.shape}")
            
#             # Apply formant shift through simple frequency warping
#             # This is a simplified approach compared to full formant manipulation
#             freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
            
#             # Create warping function - different for each formant region
#             warp = np.ones_like(freqs)
            
#             # First formant region (approx 500Hz)
#             mask_f1 = (freqs > 300) & (freqs < 800)
#             warp[mask_f1] = shift_factor * 1.1
            
#             # Second formant region (approx 1500Hz)
#             mask_f2 = (freqs > 800) & (freqs < 2500)
#             warp[mask_f2] = shift_factor * 0.9
            
#             # Higher formants
#             mask_higher = freqs >= 2500
#             warp[mask_higher] = shift_factor
            
#             print("Applying frequency warping...")
#             # Apply the warping through interpolation
#             # First, create an interpolator for each column of the STFT
#             warped_magnitude = np.zeros_like(magnitude)
            
#             # Process a subset of frames for debugging
#             max_cols_to_process = min(magnitude.shape[1], 20)  # Process at most 20 columns for debugging
            
#             for i in range(max_cols_to_process):
#                 try:
#                     # Create interpolation function
#                     interpolator = interp1d(
#                         freqs * warp, 
#                         magnitude[:, i], 
#                         bounds_error=False, 
#                         fill_value=0
#                     )
                    
#                     # Apply to original frequency bins
#                     warped_magnitude[:, i] = interpolator(freqs)
#                 except Exception as e:
#                     print(f"WARNING: Error in warping column {i}: {e}")
#                     warped_magnitude[:, i] = magnitude[:, i]  # Use original as fallback
            
#             # For the rest of the columns, just copy the original
#             if magnitude.shape[1] > max_cols_to_process:
#                 warped_magnitude[:, max_cols_to_process:] = magnitude[:, max_cols_to_process:]
#                 print(f"Skipped warping {magnitude.shape[1] - max_cols_to_process} columns")
            
#             print("Reconstructing audio signal...")
#             # Reconstruct signal with original phase
#             modified_stft = warped_magnitude * phase
#             y_modified = librosa.istft(modified_stft, hop_length=hop_length)
            
#             # Ensure the output has the same length as input
#             # Fix: Change to use size parameter instead of positional
#             y_modified = librosa.util.fix_length(y_modified, size=len(audio_float))
            
#             # Normalize energy if requested
#             if maintain_energy:
#                 energy_original = np.sum(audio_float**2)
#                 energy_modified = np.sum(y_modified**2)
#                 if energy_modified > 0:
#                     energy_ratio = np.sqrt(energy_original / energy_modified)
#                     y_modified *= energy_ratio
#                     print(f"Applied energy normalization factor: {energy_ratio:.3f}")
            
#             # Convert back to int16
#             result = np.clip(y_modified * 32768, -32768, 32767).astype(np.int16)
#             print(f"✓ Formant modification complete, output shape: {result.shape}")
#             return result
            
#         except Exception as e:
#             print(f"ERROR during formant modification: {e}")
#             # Return original audio as fallback
#             print("Returning original audio")
#             return audio_array
        
#     def apply_vibrato(self, audio_segment, rate=5.5, depth=0.3, style="lyrical"):
#         """
#         Apply professional-grade vibrato effect with musical expressiveness
#         """
#         print(f"Applying {style} vibrato (rate={rate}Hz, depth={depth})")
        
        
#         # Convert AudioSegment to numpy array
#         audio_array = np.array(audio_segment.get_array_of_samples())
#         sample_rate = audio_segment.frame_rate
        
#         print(f"Audio array shape: {audio_array.shape}, Sample rate: {sample_rate}Hz")
        
#         # Create time array
#         t = np.arange(0, len(audio_array)) / sample_rate
        
#         # Create a natural vibrato envelope
#         # Real singers don't apply vibrato immediately or constantly
#         envelope = np.ones_like(t)
#         attack_time = 0.2  # seconds before vibrato kicks in
#         release_time = 0.15  # seconds to fade out vibrato
        
#         attack_samples = int(attack_time * sample_rate)
#         release_samples = int(release_time * sample_rate)
        
#         # Apply envelope for natural vibrato
#         if attack_samples < len(envelope):
#             envelope[:attack_samples] = np.linspace(0, 1, attack_samples)**2  # Quadratic fade-in
        
#         if release_samples < len(envelope):
#             envelope[-release_samples:] = np.linspace(1, 0, release_samples)**0.5  # Square root fade-out
        
#         # Apply different vibrato styles
#         if style == "lyrical":
#             # Smoothly varying vibrato rate (common in classical singing)
#             rate_mod = rate + 0.5 * np.sin(2.0 * np.pi * 0.2 * t)  # Slight fluctuation in rate
#             depth_mod = depth * (1.0 + 0.2 * np.sin(2.0 * np.pi * 0.1 * t))  # Fluctuation in depth
#             vibrato = depth_mod * np.sin(2.0 * np.pi * np.cumsum(rate_mod / sample_rate))
#         elif style == "pop":
#             # Faster, more consistent vibrato common in pop music
#             vibrato = depth * np.sin(2.0 * np.pi * rate * 1.2 * t)
#         elif style == "dramatic":
#             # Wider, more pronounced vibrato for dramatic passages
#             vibrato = depth * 1.5 * np.sin(2.0 * np.pi * rate * 0.8 * t)
#         else:
#             # Default musical vibrato
#             vibrato = depth * np.sin(2.0 * np.pi * rate * t)
    
#         # Apply envelope to make vibrato natural
#         vibrato = vibrato * envelope
        
#         # Apply phase modulation with high-quality interpolation
#         modulated = np.zeros_like(audio_array, dtype=float)
        
#         # Create interpolation function for entire audio
#         x = np.arange(len(audio_array))
#         interp_func = interp1d(x, audio_array, kind='cubic', bounds_error=False, fill_value=0)
        
#         # Apply vibrato with cubic interpolation
#         for i in range(len(audio_array)):
#             src_pos = i + vibrato[i] * sample_rate / 15  # Scaled modulation depth
#             modulated[i] = interp_func(src_pos)
        
#         # Convert back to same format as input
#         modulated = np.clip(modulated, np.iinfo(audio_array.dtype).min, np.iinfo(audio_array.dtype).max)
#         modulated = modulated.astype(audio_array.dtype)
        
#         # Create new AudioSegment
#         return audio_segment._spawn(modulated.tobytes())
    
#     def apply_musical_effects(self, audio_segment, section_type, dynamic_level=0.7):
#         """
#         Apply musical effects using only core pydub operations
#         """
#         try:
#             audio = audio_segment
            
#             # 1. Apply EQ using basic filters
#             if "chorus" in section_type.lower():
#                 audio = self._apply_eq(audio, low_cut=120, high_boost=4.0, low_boost=2.0)
#             elif "verse" in section_type.lower():
#                 audio = self._apply_eq(audio, low_cut=150, high_boost=2.0)
#             elif "bridge" in section_type.lower():
#                 audio = self._apply_eq(audio, low_cut=100, high_boost=5.0, low_boost=3.0)
            
#             # 2. Apply reverb using delay effects
#             audio = self._apply_reverb(audio, section_type)
            
#             # 3. Apply compression
#             audio = self._apply_compression(audio, dynamic_level)
            
#             return audio
            
#         except Exception as e:
#             print(f"ERROR applying effects: {e}")
#             return audio_segment  # Return original if processing fails

#     def _apply_eq(self, audio, low_cut=0, high_boost=0, low_boost=0):
#         """Basic EQ using available pydub filters"""
#         # Apply high pass filter
#         if low_cut > 0:
#             audio = high_pass_filter(audio, low_cut)
        
#         # Simulate high shelf boost (presence)
#         if high_boost > 0:
#             high_freq = audio.high_pass_filter(3500)
#             audio = audio.overlay(high_freq + high_boost)
        
#         # Simulate low shelf boost (warmth)
#         if low_boost > 0:
#             low_freq = audio.low_pass_filter(250)
#             audio = audio.overlay(low_freq + low_boost)
        
#         return audio

#     def _apply_reverb(self, audio, section_type):
#         """Create reverb effect using delays"""
#         # Early reflection
#         early_reflection = audio - 14
#         reverb = audio.overlay(early_reflection, position=30)
        
#         # Add delays based on section type
#         delay_times = []
#         if "chorus" in section_type.lower():
#             delay_times = [80, 150, 220]
#         elif "verse" in section_type.lower():
#             delay_times = [60, 120]
#         elif "bridge" in section_type.lower():
#             delay_times = [100, 180, 260]
#         else:
#             delay_times = [80, 160]
        
#         for delay in delay_times:
#             delayed = audio - (16 + delay_times.index(delay) * 4)
#             reverb = reverb.overlay(delayed, position=delay)
        
#         return reverb

#     def _apply_compression(self, audio, dynamic_level):
#         """Simple compression effect"""
#         # Convert to numpy array for processing
#         samples = np.array(audio.get_array_of_samples())
        
#         # Calculate RMS level
#         rms = np.sqrt(np.mean(samples**2))
#         threshold = rms * (1.5 - dynamic_level)
        
#         if threshold > 0:
#             # Simple soft-knee compression
#             ratio = 4.0
#             compressed = np.where(
#                 np.abs(samples) > threshold,
#                 np.sign(samples) * (threshold + (np.abs(samples) - threshold)/ratio),
#                 samples
#             )
            
#             # Apply makeup gain
#             makeup_gain = 1.0 + (1.0 - dynamic_level) * 0.5
#             compressed = compressed * makeup_gain
            
#             # Convert back to audio segment
#             compressed = np.clip(compressed, -32768, 32767).astype(np.int16)
#             return audio._spawn(compressed.tobytes())
        
#         return audio
    
#     def adjust_pitch_for_key(self, audio_segment, key_index, section_type, chord=None):
#         """
#         Sophisticated pitch adjustment with multiple fallback options when rubberband fails
#         """
#         # Define pitch mappings
#         major_scale = [0, 2, 4, 5, 7, 9, 11]
#         minor_scale = [0, 2, 3, 5, 7, 8, 10]
        
#         # Select appropriate pitch based on section and chord
#         possible_pitches = self._select_musical_pitches(section_type, chord)
        
#         # Select target pitch shift with voice leading
#         pitch_shift = self._select_pitch_with_voice_leading(possible_pitches)
        
#         # Convert to numpy array
#         y = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
#         sr = audio_segment.frame_rate
        
#         # Attempt 1: First try pyrubberband (best quality)
#         try:
#             y_shifted = pyrb.pitch_shift(y, sr, pitch_shift)
#             print("✓ Pitch adjustment using pyrubberband succeeded")
#             return self._finalize_pitch_shift(audio_segment, y_shifted)
#         except Exception as e:
#             print(f"Rubberband pitch shift failed: {e}. Trying fallback methods...")

#         # Attempt 2: Librosa's phase vocoder (medium quality)
#         try:
#             y_shifted = librosa.effects.pitch_shift(
#                 y, 
#                 sr=sr, 
#                 n_steps=pitch_shift,
#                 bins_per_octave=24,  # Higher precision
#                 res_type='kaiser_fast'
#             )
#             print("✓ Pitch adjustment using librosa succeeded")
#             return self._finalize_pitch_shift(audio_segment, y_shifted)
#         except Exception as e:
#             print(f"Librosa pitch shift failed: {e}. Trying simple resampling...")

#         # Attempt 3: Simple resampling (basic quality)
#         try:
#             # Calculate new sample rate
#             new_rate = int(sr * (2 ** (pitch_shift / 12)))
#             shifted = audio_segment.set_frame_rate(new_rate)
            
#             # Convert back to original sample rate while preserving duration
#             shifted = shifted.set_frame_rate(sr)
#             print("✓ Pitch adjustment using resampling succeeded")
#             return shifted
#         except Exception as e:
#             print(f"All pitch adjustment methods failed: {e}. Returning original audio.")
#             return audio_segment

#     def _select_musical_pitches(self, section_type, chord):
#         """Select musically appropriate pitches based on context"""
#         if "chorus" in section_type.lower():
#             return [4, 5, 7, 9, 11, 12, 14]
#         elif "verse" in section_type.lower():
#             return [0, 2, 4, 5, 7, 9]
#         elif "bridge" in section_type.lower():
#             return [2, 5, 7, 9, 12]
#         else:
#             return [0, 2, 4, 5, 7, 9]

#     def _select_pitch_with_voice_leading(self, possible_pitches):
#         """Select pitch with smooth voice leading"""
#         if hasattr(self, 'last_pitch_shift'):
#             distances = [abs((p - self.last_pitch_shift) % 12) for p in possible_pitches]
#             pitch_shift = possible_pitches[np.argmin(distances)]
#         else:
#             pitch_shift = random.choice(possible_pitches)
#         self.last_pitch_shift = pitch_shift
#         return pitch_shift

#     def _finalize_pitch_shift(self, original_segment, shifted_samples):
#         """Convert numpy array back to AudioSegment with natural pitch variations"""
#         # Add subtle natural pitch variations
#         t = np.arange(len(shifted_samples)) / original_segment.frame_rate
#         mod_env = np.sin(2 * np.pi * 0.5 * t)  # Slow modulation
#         resampling_factors = 1.0 + mod_env * 0.003  # Very subtle
        
#         # Apply using high-quality interpolation
#         x_indices = np.arange(len(shifted_samples))
#         warped_indices = np.cumsum(resampling_factors)
#         warped_indices = warped_indices / warped_indices[-1] * len(shifted_samples)
        
#         interp_func = interp1d(
#             warped_indices, 
#             shifted_samples, 
#             kind='cubic', 
#             bounds_error=False, 
#             fill_value=0
#         )
#         y_final = interp_func(x_indices)
        
#         # Convert back to audio segment format
#         shifted_audio = np.clip(y_final * 32768, -32768, 32767).astype(np.int16)
#         return original_segment._spawn(shifted_audio.tobytes())
    
#     def generate_vocal_phrases(self, aligned_lyrics, backing_track_info):
#         """
#         More robust version with better error handling and progress tracking
#         """
#         print("Generating vocal phrases with musical expression...")
#         output_segments = []
#         total_phrases = len(aligned_lyrics)
        
#         # Extract key musical information
#         key = backing_track_info['key']
#         mode = backing_track_info['mode']
#         key_index = backing_track_info['key_index']
#         tempo = backing_track_info['tempo']
        
#         # Adjust vocal characteristics
#         self._set_vocal_characteristics(mode, tempo)
        
#         # Generate each vocal phrase with progress tracking
#         for i, (text, timing, section, chord, dynamic) in enumerate(aligned_lyrics):
#             try:
#                 print(f"\nProcessing phrase {i+1}/{total_phrases}: '{text}'")
                
#                 # Skip empty phrases
#                 if not text.strip():
#                     continue
                
#                 # 1. Generate base speech
#                 temp_file = f"temp_phrase_{i}.wav"
#                 if not self._generate_speech(text, temp_file):
#                     continue
                
#                 # 2. Load and process speech
#                 speech = self._load_and_process_audio(temp_file, i, timing, aligned_lyrics)
#                 if speech is None:
#                     continue
                
#                 # 3. Apply vocal transformations
#                 processed = self._apply_vocal_transformations(
#                     speech, 
#                     key_index, 
#                     section, 
#                     chord, 
#                     dynamic
#                 )
                
#                 if processed:
#                     # Add to output with proper timing
#                     self._add_to_output(output_segments, processed, timing, i)
                
#                 # Clean up temp file
#                 self._cleanup_temp_file(temp_file)
                
#             except Exception as e:
#                 print(f"ERROR processing phrase {i+1}: {e}")
#                 traceback.print_exc()
#                 continue
        
#         # Combine all segments
#         return self._finalize_output(output_segments)

#     def _set_vocal_characteristics(self, mode, tempo):
#         """Configure vocal settings based on song characteristics"""
#         if mode == "major":
#             self.formant_shift = 1.05
#             self.breathiness = 0.15
#         else:
#             self.formant_shift = 0.95
#             self.breathiness = 0.25
        
#         self.duration_factor = 1.0 if tempo > 120 else 1.3

#     def _generate_speech(self, text, temp_file):
#         """Generate base speech with pyttsx3"""
#         try:
#             self.engine.save_to_file(text, temp_file)
#             self.engine.runAndWait()
#             return True
#         except Exception as e:
#             print(f"ERROR generating speech: {e}")
#             return False

#     def _load_and_process_audio(self, temp_file, i, timing, aligned_lyrics):
#         """Load and time-stretch audio"""
#         try:
#             speech = AudioSegment.from_file(temp_file)
            
#             # Adjust duration to match musical timing
#             if i < len(aligned_lyrics) - 1:
#                 next_timing = aligned_lyrics[i+1][1]
#                 target_duration = (next_timing - timing) * 1000
#                 current_duration = len(speech)
                
#                 if current_duration > 0 and 0.5 <= target_duration / current_duration <= 2.0:
#                     return self._time_stretch(speech, current_duration, target_duration)
            
#             return speech
#         except Exception as e:
#             print(f"ERROR loading/processing audio: {e}")
#             return None

#     def _apply_vocal_transformations(self, speech, key_index, section, chord, dynamic):
#         """Apply all vocal effects"""
#         try:
#             # Convert to numpy array for processing
#             y = np.array(speech.get_array_of_samples())
#             sr = speech.frame_rate
            
#             # Apply formant modification
#             y_formant = self.modify_formants(y, sr, self.formant_shift)
#             speech = speech._spawn(y_formant.tobytes())
            
#             # Apply pitch adjustment
#             speech = self.adjust_pitch_for_key(speech, key_index, section, chord)
            
#             # Apply vibrato based on section
#             vibrato_style = "pop"
#             if "chorus" in section.lower():
#                 vibrato_style = "dramatic"
#             elif "verse" in section.lower():
#                 vibrato_style = "lyrical"
                
#             speech = self.apply_vibrato(
#                 speech, 
#                 rate=5.5, 
#                 depth=0.3, 
#                 style=vibrato_style
#             )
            
#             # Apply musical effects
#             dynamic_level = 0.7 if dynamic is None else min(1.0, dynamic / 0.1)
#             return self.apply_musical_effects(speech, section, dynamic_level)
            
#         except Exception as e:
#             print(f"ERROR applying vocal transformations: {e}")
#             return None

#     def _add_to_output(self, output_segments, processed, timing, i):
#         """Add processed audio to output with proper timing"""
#         if i == 0:
#             silence_duration = int(timing * 1000)
#             if silence_duration > 0:
#                 output_segments.append(AudioSegment.silent(duration=silence_duration))
        
#         output_segments.append(processed)
#         print(f"✓ Successfully processed and added phrase {i+1}")

#     def _cleanup_temp_file(self, temp_file):
#         """Clean up temporary files"""
#         try:
#             os.remove(temp_file)
#         except:
#             pass

#     def _finalize_output(self, output_segments):
#         """Combine all segments into final output"""
#         if not output_segments:
#             return AudioSegment.silent(duration=1000)
        
#         final_output = output_segments[0]
#         for segment in output_segments[1:]:
#             final_output = final_output.append(segment, crossfade=0)
        
#         return final_output
    
#     def _time_stretch(self, audio_segment, current_duration, target_duration):
#         """
#         Musically-aware time stretching with vowel preservation
#         """
#         # Convert to numpy array
#         y = np.array(audio_segment.get_array_of_samples())
#         sr = audio_segment.frame_rate
        
#         # Calculate stretch factor
#         stretch_factor = target_duration / current_duration
        
#         try:
#             # Use pyrubberband for high-quality time stretching
#             y_stretched = pyrb.time_stretch(y.astype(np.float32), sr, stretch_factor)
#             stretched_audio = np.clip(y_stretched * 32768, -32768, 32767).astype(np.int16)
#             return audio_segment._spawn(stretched_audio.tobytes())
#         except Exception as e:
#             print(f"Advanced time stretching failed: {e}. Using simple method.")
#             # Fallback to simple resampling
#             return audio_segment.set_frame_rate(int(audio_segment.frame_rate / stretch_factor))
    
#     def mix_with_backing_track(self, vocals, backing_track_path):
#         """Windows-compatible memory-efficient mixing"""
#         print("Performing final mix (Windows-optimized version)...")
        
#         try:
#             # First verify the backing track exists
#             if not os.path.exists(backing_track_path):
#                 raise FileNotFoundError(f"Backing track not found: {backing_track_path}")

#             # Load the entire backing track (pydub handles this efficiently)
#             try:
#                 backing_track = AudioSegment.from_file(backing_track_path)
#             except Exception as e:
#                 raise Exception(f"Couldn't load backing track: {e}")

#             # Ensure lengths match (pad or trim vocals if needed)
#             if len(vocals) > len(backing_track):
#                 print(f"Warning: Vocals ({len(vocals)/1000}s) longer than backing track ({len(backing_track)/1000}s) - trimming")
#                 vocals = vocals[:len(backing_track)]
#             elif len(vocals) < len(backing_track):
#                 print(f"Note: Vocals ({len(vocals)/1000}s) shorter than backing track ({len(backing_track)/1000}s) - padding with silence")
#                 silence = AudioSegment.silent(duration=len(backing_track) - len(vocals))
#                 vocals = vocals + silence

#             # Set relative volumes
#             backing_track = backing_track - 6  # Reduce backing track volume
#             vocals = vocals - 3  # Slightly reduce vocal volume

#             # Simple overlay mix (most reliable method)
#             print("Mixing vocals with backing track...")
#             mixed = backing_track.overlay(vocals)

#             # Clean up explicitly
#             del backing_track
#             del vocals

#             return mixed

#         except Exception as e:
#             print(f"ERROR during mixing: {str(e)}")
#             traceback.print_exc()
            
#             # Try to return at least the vocals if mixing failed
#             return vocals if 'vocals' in locals() else AudioSegment.silent(duration=1000)

#     def _process_mix_chunk(self, vocal_chunk, backing_chunk):
#         """Process individual mix chunks"""
#         try:
#             # Apply vocal processing
#             vocals = vocal_chunk.set_channels(1) - 3  # Slightly quieter
            
#             # Process backing track
#             backing = backing_chunk - 6  # Lower volume
            
#             # Simple mix without advanced processing
#             return backing.overlay(vocals)
            
#         except Exception as e:
#             print(f"ERROR processing chunk: {e}")
#             return vocal_chunk  # Fallback to just vocals

#         # Compress each band differently
#     def compress_band(band, threshold=-15, ratio=4):
#         array = np.array(band.get_array_of_samples()).astype(np.float32)
#         # Simple RMS-based compression
#         rms = np.sqrt(np.mean(array**2))
#         threshold_linear = 10**(threshold/20) * 32768
#         if rms > threshold_linear:
#             gain_reduction = (threshold_linear / rms) ** (1 - 1/ratio)
#             array = array * gain_reduction
#         return band._spawn((array.astype(np.int16)).tobytes())
    
    
    
#     def generate_ai_singing(self, lyrics, backing_track_path, output_path):
#         """
#         Main function to generate AI singing with musical awareness
#         """
#         print("Starting AI Singer process (with progress tracking)...")
        
#         try:
        
#             # 1. Analyze backing track
#             track_info = self.analyze_backing_track(backing_track_path)
            
#             # 2. Align lyrics to beats
#             aligned_lyrics = self.align_lyrics_to_beats(
#                 lyrics, 
#                 track_info['beat_times'], 
#                 track_info['tempo'],
#                 track_info.get('chord_progression'),
#                 track_info.get('dynamics')
#             )
            
#             # 3. Generate vocal phrases
#             vocals = self.generate_vocal_phrases(aligned_lyrics, track_info)
            
#             print("\nStarting final mix...")
#             start_time = time.time()
#             final_mix = self.mix_with_backing_track(vocals, backing_track_path)
            
#             print(f"Mixing completed in {time.time()-start_time:.2f} seconds")
#             print(f"Final audio length: {len(final_mix)/1000:.2f} seconds")
            
#             print("Exporting final mix...")
#             final_mix.export(output_path, format="wav")
#             print(f"Successfully exported to {output_path}")
            
#             return output_path
            
#         except Exception as e:
#             print(f"Fatal error in generation: {str(e)}")
#             traceback.print_exc()
#             return None
            



# # Example usage of the AISingerGenerator
# if __name__ == "__main__":
#     # Create the generator
#     singer = AISingerGenerator()
    
#     # Example lyrics with section markers
#     example_lyrics = """
#     [Intro]
    
#     [Verse 1]
#     Walking through the city lights
#     Holding hands on Friday night
#     Your smile cuts through all the noise
#     In this moment, pure joy
    
#     [Chorus]
#     We're shining bright like the stars above
#     Filled with hope, filled with love
#     Every heartbeat sings your name
#     This feeling I can't explain
    
#     [Verse 2]
#     Memories we're making now
#     Promise me we'll work it out
#     Through the good and stormy days
#     I'll be yours in every way
    
#     [Chorus]
#     We're shining bright like the stars above
#     Filled with hope, filled with love
#     Every heartbeat sings your name
#     This feeling I can't explain
    
#     [Bridge]
#     Time stands still when you're with me
#     Together is where we're meant to be
    
#     [Chorus]
#     We're shining bright like the stars above
#     Filled with hope, filled with love
#     Every heartbeat sings your name
#     This feeling I can't explain
    
#     [Outro]
#     """
    
#     # Generate the singing
#     singer.generate_ai_singing(
#         lyrics=example_lyrics,
#         backing_track_path="melodic_integrated_song.wav",
#         output_path="ai_singing_output.wav"
#     )




































































import os
import sys
import requests
import zipfile
import torch
from pathlib import Path

# --- 1. Setup Environment ---
RVC_DIR = "RVC"
MODEL_URL = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/pretrained_v2/f0G32k.pth"  # Free pretrained model

# --- 2. Download and Setup RVC ---
if not os.path.exists(RVC_DIR):
    print("Setting up RVC...")
    os.makedirs(RVC_DIR, exist_ok=True)
    
    # Download RVC files
    rvc_zip_url = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/archive/refs/heads/main.zip"
    zip_path = "rvc.zip"
    
    print("Downloading RVC...")
    with requests.get(rvc_zip_url, stream=True) as r:
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print("Extracting RVC...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    os.rename("Retrieval-based-Voice-Conversion-WebUI-main", RVC_DIR)
    os.remove(zip_path)
    
    # Install requirements
    print("Installing dependencies...")
    os.system(f"pip install -r {RVC_DIR}/requirements.txt")

# --- 3. Download Model ---
model_path = f"{RVC_DIR}/weights/model.pth"
if not os.path.exists(model_path):
    print("Downloading voice model...")
    os.makedirs(f"{RVC_DIR}/weights", exist_ok=True)
    with requests.get(MODEL_URL, stream=True) as r:
        with open(model_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# --- 4. Setup Paths ---
sys.path.append(str(Path(RVC_DIR).absolute()))

# --- 5. Import RVC Function ---
from infer import GradioInfer

# --- 6. Process Your Files ---
print("Generating AI singing...")
GradioInfer(
    input_path="melodic_integrated_song.wav",  # Your melody file
    model_path=model_path,                     # Pretrained model
    output_path="ai_singing.wav",              # Output file
    index_ratio=0.5,                           # Voice quality (0-1)
    f0_up_key=0                                # Pitch adjustment (0 for none)
)

print("✅ AI singing generated at ai_singing.wav")