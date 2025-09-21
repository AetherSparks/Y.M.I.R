import numpy as np
import soundfile as sf
import simpleaudio as sa
import os


BRAINWAVE_BANDS = {
    'delta': 3,
    'theta': 6,
    'alpha': 10,
    'beta': 20,
    'gamma': 40,
    'epsilon': 100
}

def generate_variation(base_freq, band, duration=10, fs=44100, version=1):
    beat_freq = BRAINWAVE_BANDS[band]
    
    # Small random variations
    carrier = base_freq + np.random.uniform(-20, 20)      # carrier freq variation ±20 Hz
    mod_depth = np.random.uniform(0.4, 0.6)              # amplitude modulation depth
    beat_var = beat_freq + np.random.uniform(-0.2, 0.2)  # beat freq variation ±0.2 Hz

    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    carrier_wave = np.sin(2.0 * np.pi * carrier * t)
    
    # Smooth sine modulation with depth variation
    modulation = 0.5 + mod_depth * 0.5 * np.sin(2.0 * np.pi * beat_var * t)
    signal = carrier_wave * modulation
    
    # Fade-in/out
    ramp = int(0.02 * fs)
    env = np.ones_like(signal)
    env[:ramp] = np.linspace(0, 1.0, ramp)
    env[-ramp:] = np.linspace(1.0, 0.0, ramp)
    signal *= env
    
    # Stereo
    stereo = np.vstack([signal, signal]).T
    stereo /= np.max(np.abs(stereo)) + 1e-12
    
    # Save to WAV
    output_folder = 'experimental/isochronic_tone'
    os.makedirs(output_folder, exist_ok=True)
    # Create the full path for the WAV file
    filename = os.path.join(output_folder, f'isochronic_{band}_v{version}_{int(carrier)}Hz.wav')
    # Write the file
    sf.write(filename, stereo, fs)
    print(f"WAV written to {filename}")
    
    audio_data = np.ascontiguousarray((stereo * 32767).astype(np.int16))
    sa.play_buffer(audio_data, num_channels=2, bytes_per_sample=2, sample_rate=fs)

if __name__ == "__main__":
    # Generate 3 variations of alpha, beta, and gamma
    for band in ['alpha', 'beta', 'gamma', 'delta', 'theta', 'epsilon']:
        for v in range(1, 4):
            generate_variation(base_freq=200, band=band, duration=1000, version=v)
