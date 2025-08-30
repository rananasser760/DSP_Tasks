# Digital Signal Processing (DSP) Package 📡

This Python package provides a comprehensive suite of **Digital Signal Processing (DSP)** tasks implemented using **Tkinter GUI**. It is designed for educational and practical purposes to help users analyze, manipulate, and visualize signals interactively.

## Features 🚀

- ### The DSP package contains **7 main tasks**:
- Fully interactive Tkinter GUI
- Supports file input/output for signals and coefficients
- Visualize time domain & frequency domain signals
- Handles multiple DSP tasks in one application

---

### **Task 1: Signal Generation and Display 🎵**
- Read signal samples from a `.txt` file and display them in:
  - Continuous representation
  - Discrete representation
- Generate sinusoidal or cosinusoidal signals by letting the user choose:
  - Signal type: sine or cosine
  - Amplitude (A)
  - Phase shift (θ)
  - Analog frequency
  - Sampling frequency

---

### **Task 2: Signal Operations ➕➖✖️**
- **Addition:** Add multiple input signals and display the resulting signal.
- **Subtraction:** Subtract input signals and display the resulting signal. 
- **Multiplication:** Multiply a signal by a constant to amplify or reduce amplitude. Multiplying by `-1` inverts the signal.
- **Squaring:** Compute the square of the input signals.
- **Normalization:** Normalize signal values to:
  - Range [-1, 1]
  - Range [0, 1] (user choice)
- **Accumulation:** Compute the accumulated sum of the input signal.

---

### **Task 3: Signal Quantization 🎚️**
- Quantize an input signal's samples.
- Users can specify:
  - Number of quantization levels
  - Or number of bits, which will be converted to the corresponding levels.

---

### **Task 4: Fourier Transform DFT & IDFT 🔄**
- Apply **Fourier Transform (DFT/FFT)** on any input signal.
- Display:
  - Frequency vs. Amplitude
  - Frequency vs. Phase
- Users provide the sampling frequency in Hz.
- Reconstruct the signal using **Inverse DFT (IDFT)**.

---

### **Task 5: Discrete Cosine Transform (DCT) & Time-Domain Operations ⏱️**
- Compute DCT of an input signal.
- Display the result and allow users to save the first `m` coefficients to a `.txt` file.
- **Time-Domain Features:**
  - **Sharpening:** First and second derivatives of the signal
    - First derivative: `y(n) = x(n) - x(n-1)`
    - Second derivative: `y(n) = x(n+1) - 2x(n) + x(n-1)`
  - **Signal Delaying/Advancing:** Shift signals by `k` steps.
  - **Folding:** Fold signals and apply delaying/advancing on folded signals.

---

### **Task 6: Smoothing, DC Removal, Convolution, and Correlation 🧼**
- **Smoothing:** Moving average with user-defined window size.
- **DC Component Removal:**
  - In time domain
  - In frequency domain
- **Convolution:** Convolve two signals.
- **Correlation:** Compute normalized cross-correlation of two signals.

---

### **Task 7: Filtering & Resampling 🛠️**
- **FIR Filtering:**
  - Apply FIR filters (Low, High, Band Pass, Band Stop) on input signals.
  - Users specify filter type and design parameters.
  - Compute filter coefficients and convolve with the input signal.
  - Display and save the resulting signal and coefficients.
- **Resampling:**
  - Users provide input signal, low-pass filter specifications, and decimation/interpolation factors `M` & `L`.
  - Handles all cases:
    1. `M = 0 & L ≠ 0`: Upsample then apply low-pass filter
    2. `M ≠ 0 & L = 0`: Apply filter then downsample
    3. `M ≠ 0 & L ≠ 0`: Change sample rate by fraction (upsample → filter → downsample)
    4. `M = 0 & L = 0`: Return error message

---

## Requirements 🛠️
- Python 3.x
- Tkinter (usually included with Python)
- NumPy
- Math
- Matplotlib
- SciPy (for advanced DSP operations)

---

## Usage 💡
1. Clone the repository:
   ```bash
   git clone <https://github.com/rananasser760/DSP_Tasks.git>

2. Run the main GUI application:
   ```bash
    python package.py

---
## Notes 💡

All tasks are interactive and designed for learning and experimenting with signals.
Ideal for students or engineers learning Digital Signal Processing.

---
## Team Members:

### Rana Nasser
### Gihad Mahmoud
