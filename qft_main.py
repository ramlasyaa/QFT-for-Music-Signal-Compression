import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# =============== CONFIG ===============
input_file = "./sample_1.wav"  
target_sr = 16000
k = 32                  # Top-K per frame
cutoff_hz = 4000        # Low-pass cutoff
threshold_db = -40      # Threshold in dB
n_fft = 1024
hop_length = 256
win_length = 1024
window = "hann"
eps = 1e-12
# ======================================

# ---------- Helpers ----------
def ensure_mono(y):
    if y.ndim == 2:
        return np.mean(y, axis=1)
    return y

def align_lengths(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n]

def snr_db(original, reconstructed, eps=1e-12):
    o, r = align_lengths(original, reconstructed)
    noise = o - r
    num = np.sum(o**2) + eps
    den = np.sum(noise**2) + eps
    return 10.0 * np.log10(num / den)

def stft_mag_db(y, sr):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    D = librosa.amplitude_to_db(np.abs(S) + eps, ref=np.max)
    return D

def plot_wave_and_spec(y, sr, title, filename):
    plt.figure(figsize=(12, 6))
    # Waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform: {title}")
    # Spectrogram (magnitude dB)
    plt.subplot(2, 1, 2)
    D = stft_mag_db(y, sr)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", cmap="magma", hop_length=hop_length)
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram: {title}")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def save_wav(filename, y, sr):
    sf.write(filename, y, sr)

def plot_three_waves(err_topk, err_low, err_thresh, filename):
    plt.figure(figsize=(12, 6))
    # Top-K error
    plt.subplot(3, 1, 1)
    plt.plot(err_topk, linewidth=0.8)
    plt.title("Error waveform: Top-K (Original - Reconstructed)")
    # Low-pass error
    plt.subplot(3, 1, 2)
    plt.plot(err_low, linewidth=0.8)
    plt.title("Error waveform: Low-pass (Original - Reconstructed)")
    # Threshold error
    plt.subplot(3, 1, 3)
    plt.plot(err_thresh, linewidth=0.8)
    plt.title("Error waveform: Threshold-dB (Original - Reconstructed)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_three_specs(spec_topk, spec_low, spec_thresh, sr, filename, main_titles):
    plt.figure(figsize=(12, 6))
    # Top-K
    plt.subplot(3, 1, 1)
    librosa.display.specshow(spec_topk, sr=sr, x_axis="time", y_axis="hz", cmap="inferno", hop_length=hop_length)
    plt.colorbar(format="%+2.0f dB")
    plt.title(main_titles[0])
    # Low-pass
    plt.subplot(3, 1, 2)
    librosa.display.specshow(spec_low, sr=sr, x_axis="time", y_axis="hz", cmap="inferno", hop_length=hop_length)
    plt.colorbar(format="%+2.0f dB")
    plt.title(main_titles[1])
    # Threshold
    plt.subplot(3, 1, 3)
    librosa.display.specshow(spec_thresh, sr=sr, x_axis="time", y_axis="hz", cmap="inferno", hop_length=hop_length)
    plt.colorbar(format="%+2.0f dB")
    plt.title(main_titles[2])
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_masks_side_by_side(mask_topk, mask_low, mask_thresh, filename):
    plt.figure(figsize=(12, 6))
    # Top-K mask
    plt.subplot(3, 1, 1)
    plt.imshow(mask_topk.astype(float), aspect="auto", origin="lower", cmap="gray_r", interpolation="nearest")
    plt.title("Kept bins mask: Top-K (white=kept, black=removed)")
    plt.xlabel("Frames")
    plt.ylabel("Frequency bins (low→high)")
    # Low-pass mask
    plt.subplot(3, 1, 2)
    plt.imshow(mask_low.astype(float), aspect="auto", origin="lower", cmap="gray_r", interpolation="nearest")
    plt.title("Kept bins mask: Low-pass")
    plt.xlabel("Frames")
    plt.ylabel("Frequency bins (low→high)")
    # Threshold mask
    plt.subplot(3, 1, 3)
    plt.imshow(mask_thresh.astype(float), aspect="auto", origin="lower", cmap="gray_r", interpolation="nearest")
    plt.title("Kept bins mask: Threshold-dB")
    plt.xlabel("Frames")
    plt.ylabel("Frequency bins (low→high)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_methods_side_by_side(y_topk, y_low, y_thresh, sr, filename):
    # 3 rows x 2 cols: waveform + spectrogram per method
    plt.figure(figsize=(12, 10))

    # Top-K row
    plt.subplot(3, 2, 1)
    librosa.display.waveshow(y_topk, sr=sr)
    plt.title("Waveform: Top-K")
    plt.subplot(3, 2, 2)
    D_topk = stft_mag_db(y_topk, sr)
    librosa.display.specshow(D_topk, sr=sr, x_axis="time", y_axis="hz", cmap="magma", hop_length=hop_length)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram: Top-K")

    # Low-pass row
    plt.subplot(3, 2, 3)
    librosa.display.waveshow(y_low, sr=sr)
    plt.title("Waveform: Low-pass")
    plt.subplot(3, 2, 4)
    D_low = stft_mag_db(y_low, sr)
    librosa.display.specshow(D_low, sr=sr, x_axis="time", y_axis="hz", cmap="magma", hop_length=hop_length)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram: Low-pass")

    # Threshold row
    plt.subplot(3, 2, 5)
    librosa.display.waveshow(y_thresh, sr=sr)
    plt.title("Waveform: Threshold-dB")
    plt.subplot(3, 2, 6)
    D_thresh = stft_mag_db(y_thresh, sr)
    librosa.display.specshow(D_thresh, sr=sr, x_axis="time", y_axis="hz", cmap="magma", hop_length=hop_length)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram: Threshold-dB")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# ---------- Load ----------
print("Loading audio...")
x, sr = librosa.load(input_file, sr=target_sr, mono=True)
x = ensure_mono(x)
print(f"Audio loaded: {len(x)/sr:.2f} sec, sr={sr}")

# Baseline original plot only
plot_wave_and_spec(x, sr, "Original", "plot_original.png")

# ---------- STFT ----------
X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)

# ---------- Methods (no single-method plots) ----------
results = {}

# A) Top-K per frame (by magnitude)
print("Running Top-K...")
X_topk = np.zeros_like(X, dtype=X.dtype)
for i in range(X.shape[1]):
    mag = np.abs(X[:, i])
    if k >= len(mag):
        idx = np.arange(len(mag))
    else:
        idx = np.argpartition(mag, -k)[-k:]  # faster than full sort
    X_topk[idx, i] = X[idx, i]
rec_topk = librosa.istft(X_topk, hop_length=hop_length, win_length=win_length, window=window, length=len(x))
save_wav("reconstructed_topk.wav", rec_topk, sr)
results["Top-K"] = {"y": rec_topk, "snr": snr_db(x, rec_topk)}

# B) Low-pass (hard cutoff on frequency bins)
print("Running Low-pass...")
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
lp_mask = (freqs[:, None] <= cutoff_hz)
X_low = X * lp_mask
rec_low = librosa.istft(X_low, hop_length=hop_length, win_length=win_length, window=window, length=len(x))
save_wav("reconstructed_lowpass.wav", rec_low, sr)
results["Low-pass"] = {"y": rec_low, "snr": snr_db(x, rec_low)}

# C) Threshold in dB (relative to global max)
print("Running Threshold-dB...")
X_mag = np.abs(X) + eps
X_db = librosa.amplitude_to_db(X_mag, ref=np.max)
th_mask = (X_db >= threshold_db)
X_thresh = X * th_mask
rec_thresh = librosa.istft(X_thresh, hop_length=hop_length, win_length=win_length, window=window, length=len(x))
save_wav("reconstructed_thresholddb.wav", rec_thresh, sr)
results["Threshold-D"] = {"y": rec_thresh, "snr": snr_db(x, rec_thresh)}

# ---------- Derived data for grouped panels ----------
# Errors
o_tk, r_tk = align_lengths(x, rec_topk)
o_lp, r_lp = align_lengths(x, rec_low)
o_th, r_th = align_lengths(x, rec_thresh)
err_topk = o_tk - r_tk
err_low = o_lp - r_lp
err_thresh = o_th - r_th

# Error spectrograms (in dB)
E_topk_db = stft_mag_db(err_topk, sr)
E_low_db = stft_mag_db(err_low, sr)
E_thresh_db = stft_mag_db(err_thresh, sr)

# Grouped comparative panels
print("Creating grouped comparative panels...")
plot_methods_side_by_side(rec_topk, rec_low, rec_thresh, sr, "plot_methods.png")
plot_three_waves(err_topk, err_low, err_thresh, "plot_err_waves.png")
plot_three_specs(E_topk_db, E_low_db, E_thresh_db, sr, "plot_err_specs.png",
                 ["Error spectrogram: Top-K", "Error spectrogram: Low-pass", "Error spectrogram: Threshold-dB"])
plot_masks_side_by_side(np.abs(X_topk) > 0, lp_mask, th_mask, "plot_masks.png")

# ---------- Report ----------
print("\n=== Results ===")
if os.path.exists(input_file):
    print("Original size: {:.2f} KB".format(os.path.getsize(input_file)/1024))
else:
    print("Original file not found!")

file_map = {
    "Top-K": "reconstructed_topk.wav",
    "Low-pass": "reconstructed_lowpass.wav",
    "Threshold-D": "reconstructed_thresholddb.wav"
}

best_name = None
best_snr = -np.inf
for name, data in results.items():
    snr_val = data["snr"]
    filename = file_map.get(name)
    size_kb = os.path.getsize(filename)/1024 if filename and os.path.exists(filename) else float("nan")
    print(f"{name} -> SNR={snr_val:.2f} dB, Size={size_kb:.2f} KB")
    if snr_val > best_snr:
        best_snr = snr_val
        best_name = name

print(f"\nRecommended (by SNR): {best_name} ({best_snr:.2f} dB)")

print("\nGenerated visuals:")
print("- plot_original.png")
print("- plot_methods.png             # Top-K, Low-pass, Threshold side-by-side")
print("- plot_err_waves.png           # Error waveforms side-by-side")
print("- plot_err_specs.png           # Error spectrograms side-by-side")
print("- plot_masks.png               # Kept-bins masks side-by-side")
