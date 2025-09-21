import numpy as np
import soundfile as sf
import simpleaudio as sa

BRAINWAVE_BANDS = {
    'delta': 3,    # Hz beat frequency
    'theta': 6,
    'alpha': 10,
    'beta': 20,
    'gamma': 40,
    'epsilon': 100
}

def generate_binaural_variation(base_freq=200.0, band='alpha', duration=10.0, fs=44100, version=1):
    """
    Generate one variation of a binaural beat for a given brainwave band.
    """
    if band not in BRAINWAVE_BANDS:
        raise ValueError(f"Invalid band '{band}', choose from {list(BRAINWAVE_BANDS.keys())}")

    # Base binaural beat frequency for this band
    beat_freq = BRAINWAVE_BANDS[band]

    # Introduce small random variations for different versions
    left_freq = base_freq + np.random.uniform(-10, 10)          # vary left frequency ±10Hz
    right_freq = left_freq + beat_freq + np.random.uniform(-0.5, 0.5)  # vary beat ±0.5Hz

    t = np.linspace(0, duration, int(fs*duration), endpoint=False)

    # Optional smooth amplitude modulation to avoid harshness
    amp_mod = 0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t)  # slow 0.5Hz gentle tremolo
    left = np.sin(2.0 * np.pi * left_freq * t) * amp_mod
    right = np.sin(2.0 * np.pi * right_freq * t) * amp_mod

    # Fade-in/out envelope
    ramp = int(0.02 * fs)
    env = np.ones_like(t)
    env[:ramp] = np.linspace(0, 1.0, ramp)
    env[-ramp:] = np.linspace(1.0, 0.0, ramp)
    left *= env
    right *= env

    stereo = np.vstack([left, right]).T
    stereo /= np.max(np.abs(stereo)) + 1e-12

    filename = f'binaural_{band}_v{version}_{int(left_freq)}Hz.wav'
    sf.write(filename, stereo, fs)
    print(f"WAV written to {filename}")

    # Play (C-contiguous)
    audio_data = np.ascontiguousarray((stereo * 32767).astype(np.int16))
    sa.play_buffer(audio_data, num_channels=2, bytes_per_sample=2, sample_rate=fs)


if __name__ == "__main__":
    # Generate 3 variations for alpha, beta, and gamma
    for band in ['alpha', 'beta', 'gamma', 'delta', 'theta', 'epsilon']:
        for v in range(1, 4):
            generate_binaural_variation(base_freq=250, band=band, duration=1000, version=v)
