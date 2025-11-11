Quantum Fourier Transform for Music Signal Compression
📌 Overview

This project explores how Quantum Fourier Transform (QFT) can be applied to music signal compression, leveraging quantum principles to efficiently represent and compress audio waveforms.
By simulating QFT using quantum computing libraries, the project compares quantum-based frequency domain transformations with classical Fourier approaches (FFT).

⚙️ Project Objectives

Implement Quantum Fourier Transform (QFT) using Python (Qiskit or similar library).

Compress and reconstruct music/audio signals using QFT circuits.

Compare quantum vs. classical signal compression in terms of fidelity and efficiency.

Demonstrate potential benefits of quantum-based data compression.

🧩 Project Structure
📁 Quantum Computing Project
│
├── qft_music_compression.py      # Main implementation code
├── audio_processing.py           # Preprocessing, normalization, FFT comparison
├── qft_simulation.ipynb          # Jupyter notebook with visualizations
├── data/
│   ├── sample_music.wav          # Input test audio file
│   └── compressed_output.wav     # Reconstructed/compressed output
├── results/
│   ├── plots/
│   │   ├── qft_circuit.png
│   │   └── waveform_comparison.png
│   └── metrics.txt
└── README.md                     # Project documentation

🧠 Technologies Used

Python 3.10+

Qiskit – for quantum circuit simulation

NumPy / SciPy – for signal processing

Matplotlib / Librosa – for visualization and waveform analysis

Jupyter Notebook – for testing and results presentation

📊 Comparison Metrics
Metric	Classical FFT	Quantum QFT
Computation Complexity	O(N log N)	O((log N)²) theoretically
Output Fidelity	High	Comparable
Compression Ratio	Moderate	Potentially higher (for quantum states)
🚀 How to Run

Clone the repository

git clone https://github.com/ramlasyaa/QFT-for-Music-Signal-Compression.git
cd QFT-for-Music-Signal-Compression


Install dependencies

pip install -r requirements.txt


Run the main script

python qft_music_compression.py


(Optional) Open the notebook

jupyter notebook qft_simulation.ipynb

📈 Results

The QFT successfully compresses music signals into smaller quantum states.

The reconstructed signals maintain significant similarity to the original waveform.

Demonstrates quantum advantage potential for audio data representation.

📚 References

Nielsen, M. & Chuang, I. Quantum Computation and Quantum Information

Qiskit Documentation: https://qiskit.org/documentation/

Librosa Audio Analysis Toolkit

👩‍💻 Author

Ram Lasya
GitHub: @ramlasyaa

Project: Quantum Fourier Transform for Music Signal Compression
