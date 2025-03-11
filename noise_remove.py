import librosa
import librosa.display
import numpy as np
import scipy.signal as sig  # Use an alias for scipy.signal
import noisereduce as nr
import soundfile as sf

# Load the audio file
input_file = "output/source_0.wav"
y, sr = librosa.load(input_file, sr=None)

# Apply a high-pass filter to remove DC noise
def high_pass_filter(audio_data, cutoff_freq, sr, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = sig.butter(order, normal_cutoff, btype='high', analog=False)
    return sig.filtfilt(b, a, audio_data)

# Apply a low-pass filter to remove high-frequency noise
def low_pass_filter(audio_data, cutoff_freq, sr, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = sig.butter(order, normal_cutoff, btype='low', analog=False)
    return sig.filtfilt(b, a, audio_data)

# Step 1: Remove DC noise (high-pass filter with 20 Hz cutoff)
dc_removed_signal = high_pass_filter(y, cutoff_freq=20, sr=sr)

# Step 2: Apply a low-pass filter to remove high-frequency noise (above 3kHz)
filtered_signal = low_pass_filter(dc_removed_signal, cutoff_freq=3000, sr=sr)

# Step 3: Apply noise reduction using spectral gating
reduced_noise = nr.reduce_noise(y=filtered_signal, sr=sr, prop_decrease=0.8)

# Save the cleaned audio
output_file = "cleaned_audio.wav"
sf.write(output_file, reduced_noise, sr)

print(f"Filtered audio saved as {output_file}")