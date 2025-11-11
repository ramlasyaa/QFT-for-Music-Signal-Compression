"""
Quantum-assisted music compression (simulation).
- FRAME_LEN must be power-of-two (e.g., 8,16,32). Keep small for simulation.
- Uses Qiskit statevector simulator to obtain QFT probabilities per frame.
- Uses those probabilities to select top-M FFT bins (classical inverse FFT used to reconstruct).
"""

import os, math, sys
import numpy as np
import soundfile as sf
import librosa, librosa.display
import matplotlib.pyplot as plt

# New Qiskit import style
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
# create simulator instance
backend_statevector = AerSimulator(method="statevector")


# ----------------- CONFIG -----------------
INPUT_FILE = "./sample_1.wav"   
SR = 16000
FRAME_LEN = 16        # power of two: 8,16,32. keep small for simulation
HOP = FRAME_LEN       # non-overlap (change to overlap if you want)
TOP_M = 4           # number of frequency bins to keep per frame
OUT_DIR = "qft_outputs_final"
os.makedirs(OUT_DIR, exist_ok=True)
EPS = 1e-12
# ------------------------------------------

def ensure_power_of_two(n):
    return (n & (n - 1) == 0) and n > 0

if not ensure_power_of_two(FRAME_LEN):
    raise ValueError("FRAME_LEN must be a power of two (8,16,32...)")

def load_audio(path, sr=SR):
    y, s = librosa.load(path, sr=sr, mono=True)
    return y, s

def qft_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits)
    for j in range(n_qubits):
        qc.h(j)
        for k in range(2, n_qubits - j + 1):
            angle = np.pi / (2 ** (k - 1))
            qc.cp(angle, j + k - 1, j)
    for i in range(n_qubits // 2):
        qc.swap(i, n_qubits - i - 1)
    return qc

def amplitude_encode_statevector(frame):
    vec = np.array(frame, dtype=float)
    # optional remove DC: vec = vec - np.mean(vec)
    norm = np.linalg.norm(vec)
    if norm < EPS:
        return np.ones_like(vec) / math.sqrt(len(vec))
    return vec / norm

def simulate_qft_probs(frame):
    n = len(frame)
    n_qubits = int(math.log2(n))
    
    # QFT circuit
    qc = qft_circuit(n_qubits)
    
    # Prepare amplitude-encoded state
    prep = QuantumCircuit(n_qubits)
    state = amplitude_encode_statevector(frame)
    prep.initialize(state, list(range(n_qubits)))
    
    # Combine prep + QFT
    full = prep.compose(qc)
    
    # Get statevector directly
    sv = Statevector.from_instruction(full)
    
    # Probabilities
    probs = np.abs(sv.data) ** 2
    
    # Match length
    if len(probs) != n:
        probs = probs[:n]
    
    return probs


def process_audio(y):
    N = len(y)
    pad = (FRAME_LEN - (N % FRAME_LEN)) % FRAME_LEN
    if pad:
        y = np.concatenate([y, np.zeros(pad)])
    reconstructed = np.zeros_like(y)
    total_kept_coeffs = 0
    total_coeffs = 0
    selected_bins_history = []
    for i in range(0, len(y) - FRAME_LEN + 1, HOP):
        frame = y[i:i+FRAME_LEN]
        bins = np.fft.fft(frame)
        probs = simulate_qft_probs(frame)
        m = min(TOP_M, len(probs))
        top_idx = np.argpartition(probs, -m)[-m:]
        top_idx = np.unique(top_idx)  # unique, sorted-like
        selected_bins_history.append(top_idx)
        # keep these bins + their conjugate mirrors (for real signal)
        bins_comp = np.zeros_like(bins, dtype=complex)
        for k in top_idx:
            bins_comp[k] = bins[k]
            # keep mirror for real-signal symmetry
            mirror = (-k) % FRAME_LEN
            bins_comp[mirror] = bins[mirror]
        reconstructed_frame = np.real(np.fft.ifft(bins_comp))
        reconstructed[i:i+FRAME_LEN] = reconstructed_frame
        total_kept_coeffs += len(top_idx) * 2  # approximate (with mirrors)
        total_coeffs += FRAME_LEN
    compression_ratio = (total_kept_coeffs + EPS) / (total_coeffs + EPS)
    return reconstructed[:N], compression_ratio, selected_bins_history

def snr_db(original, reconstructed):
    L = min(len(original), len(reconstructed))
    o = original[:L]; r = reconstructed[:L]
    noise = o - r
    p_sig = np.mean(o**2) + EPS
    p_noise = np.mean(noise**2) + EPS
    return 10 * np.log10(p_sig / p_noise)

def plot_spec(y, sr, filename, title):
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    plt.figure(figsize=(8,3.2))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max),
                            sr=sr, hop_length=256, x_axis='time', y_axis='hz')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def main():
    if not os.path.exists(INPUT_FILE):
        print("Input audio not found:", INPUT_FILE)
        sys.exit(1)
    y, sr = load_audio(INPUT_FILE, SR)
    print(f"Loaded {len(y)/sr:.2f}s audio, sr={sr}")
    recon, comp_ratio, bins_hist = process_audio(y)
    recon = recon / np.max(np.abs(recon)) * np.max(np.abs(y))
    out_wav = os.path.join(OUT_DIR, f"qft_recon_m{TOP_M}_fl{FRAME_LEN}.wav")
    sf.write(out_wav, recon, sr)
    plot_spec(y, sr, os.path.join(OUT_DIR, "orig_spec.png"), "Original")
    plot_spec(recon, sr, os.path.join(OUT_DIR, "qft_recon_spec.png"), "QFT-assisted Recon")
    snr_val = snr_db(y, recon)
    mse_val = np.mean((y - recon)**2)
    print(f"Saved: {out_wav}")
    print(f"SNR (original vs reconstructed, normalized): {snr_val:.2f} dB")
    print(f"MSE: {mse_val:.6f}")
    print(f"Estimated compression ratio (kept coeffs / total coeffs): {comp_ratio:.4f}")
    
    plt.figure(figsize=(10,3))
    plt.plot(y, label="Original", alpha=0.7)
    plt.plot(recon, label="Reconstructed", alpha=0.7)
    plt.title("Original vs Reconstructed Audio")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # frequent bins
    if len(bins_hist):
        all_sel = np.concatenate(bins_hist)
        uniq, cnt = np.unique(all_sel, return_counts=True)
        order = np.argsort(-cnt)
        top_frequent = uniq[order][:10]
        print("Most frequently selected bins (top 10):", top_frequent.tolist())
    print("All outputs in", OUT_DIR)

if __name__ == "__main__":
    main()
