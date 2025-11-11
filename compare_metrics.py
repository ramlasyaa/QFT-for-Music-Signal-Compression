import os
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- CONFIG ----------------
folder = "/Users/ramlasya/Documents/Quantum computing/Project"  # folder where all audios are saved
original_file = os.path.join(folder, "sample_1.wav")
classical_files = {
    "Top-K": os.path.join(folder, "reconstructed_topk.wav"),
    "Low-pass": os.path.join(folder, "reconstructed_lowpass.wav"),
    "Threshold-dB": os.path.join(folder, "reconstructed_thresholddb.wav")
}
quantum_file = os.path.join(folder, "qft_outputs_final/qft_recon_m4_fl16.wav")

# Compression ratios (adjust based on your code)
comp_ratios = {
    "Top-K": 32/1024,
    "Low-pass": 4000/8000,
    "Threshold-D": 0.25,
    "Quantum-QFT": 4/16
}

# ----------------------------------------

# ---------- Helpers ----------
def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

def load_audio_mono(path):
    check_file(path)
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)  # convert stereo to mono
    return y, sr

def align_lengths(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n]

def snr_db(orig, recon, eps=1e-12):
    o, r = align_lengths(orig, recon)
    noise = o - r
    return 10 * np.log10((np.sum(o**2)+eps)/(np.sum(noise**2)+eps))

def mse(orig, recon):
    o, r = align_lengths(orig, recon)
    return np.mean((o - r)**2)

def stft_mag_db(y, n_fft=1024, hop_length=256, win_length=1024, window="hann"):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    D = librosa.amplitude_to_db(np.abs(S)+1e-12, ref=np.max)
    return D

def plot_wave_and_spec(y_dict, sr, filename):
    plt.figure(figsize=(12, 10))
    methods = list(y_dict.keys())
    for i, method in enumerate(methods):
        y = y_dict[method]
        plt.subplot(len(methods), 2, i*2+1)
        librosa.display.waveshow(y, sr=sr)
        plt.title(f"Waveform: {method}")
        plt.subplot(len(methods), 2, i*2+2)
        D = stft_mag_db(y)
        librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", cmap="magma")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Spectrogram: {method}")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_error_wave_and_spec(orig, y_dict, filename):
    plt.figure(figsize=(12, 10))
    methods = list(y_dict.keys())
    for i, method in enumerate(methods):
        y = y_dict[method]
        o, r = align_lengths(orig, y)
        err = o - r
        plt.subplot(len(methods), 2, i*2+1)
        plt.plot(err)
        plt.title(f"Error waveform: {method}")
        plt.subplot(len(methods), 2, i*2+2)
        D_err = stft_mag_db(err)
        librosa.display.specshow(D_err, sr=sr, x_axis="time", y_axis="hz", cmap="inferno")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Error spectrogram: {method}")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# ---------- Load audios ----------
orig, sr = load_audio_mono(original_file)

y_dict = {}
for name, path in classical_files.items():
    y_dict[name], _ = load_audio_mono(path)

y_dict["Quantum-QFT"], _ = load_audio_mono(quantum_file)

# ---------- Compute metrics ----------
metrics = {"Method": [], "SNR(dB)": [], "MSE": [], "Compression Ratio": []}
for method, y in y_dict.items():
    snr_val = snr_db(orig, y)
    mse_val = mse(orig, y)
    comp_val = comp_ratios.get(method, 0)
    metrics["Method"].append(method)
    metrics["SNR(dB)"].append(round(snr_val,2))
    metrics["MSE"].append(round(mse_val,6))
    metrics["Compression Ratio"].append(round(comp_val,4))

df_metrics = pd.DataFrame(metrics)
print("\n=== Comparison Metrics ===")
print(df_metrics)
df_metrics.to_csv("metrics_comparison.csv", index=False)

# ---------- Plot waveforms + spectrograms ----------
plot_wave_and_spec({"Original": orig, **y_dict}, sr, "wave_spec_comparison.png")

# ---------- Plot error waveforms + error spectrograms ----------
plot_error_wave_and_spec(orig, y_dict, "error_wave_spec_comparison.png")

# ---------- Optional: SNR vs Compression bar chart ----------
plt.figure(figsize=(8,5))
x_pos = np.arange(len(df_metrics))
plt.bar(x_pos-0.15, df_metrics["SNR(dB)"], width=0.3, label="SNR(dB)")
plt.bar(x_pos+0.15, df_metrics["Compression Ratio"], width=0.3, label="Compression Ratio")
plt.xticks(x_pos, df_metrics["Method"])
plt.ylabel("Value")
plt.title("SNR vs Compression Ratio")
plt.legend()
plt.tight_layout()
plt.savefig("snr_vs_comp.png", dpi=150)
plt.close()
