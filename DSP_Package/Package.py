import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import Button, Menu, Tk,simpledialog, Toplevel, messagebox, filedialog, Frame, StringVar
from tkinter import *


class SignalProcessor:
    def __init__(self, master):
        self.master = master
        self.master.title("Signal Processing Framework")
        self.master.geometry("400x600")  # Set default window size
        self.master.config(bg="black")  # Black background color
        self.signals = []  # Store multiple signals
        self.quantization_choice = IntVar(value=1)  # Default to levels
        self.levels_bits_entry = StringVar()
        self.create_widgets()

    def create_widgets(self):
        
        # Create and style menu
        menubar = Menu(self.master, bg="black", fg="white", activebackground="#2980b9", activeforeground="white")
        signal_menu = Menu(menubar, tearoff=0, bg="black", fg="white", activebackground="#2980b9", activeforeground="white")
        signal_menu.add_command(label="Sine Wave", command=self.generate_sine_wave)
        signal_menu.add_command(label="Cosine Wave", command=self.generate_cosine_wave)
        menubar.add_cascade(label="Signal Generation", menu=signal_menu)

        arithmetic_menu = Menu(menubar, tearoff=0, bg="black", fg="white", activebackground="#2980b9", activeforeground="white")
        arithmetic_menu.add_command(label="Addition", command=self.add_signals)
        arithmetic_menu.add_command(label="Subtraction", command=self.subtract_signals)
        arithmetic_menu.add_command(label="Multiplication", command=self.multiply_signal)
        arithmetic_menu.add_command(label="Squaring", command=self.square_signal)
        arithmetic_menu.add_command(label="Normalization", command=self.normalize_signal)
        arithmetic_menu.add_command(label="Accumulation", command=self.accumulate_signal)
        menubar.add_cascade(label="Arithmetic Operations", menu=arithmetic_menu)

        freq_menu = Menu(menubar, tearoff=0, bg="black", fg="white", activebackground="#2980b9", activeforeground="white")
        freq_menu.add_command(label="Apply Fourier Transform", command=self.apply_fourier_transform)
        freq_menu.add_command(label="Reconstruct Signal (IDFT)", command=self.reconstruct_signal)
        freq_menu.add_command(label="Compare DFT", command=self.compare_dft)
        freq_menu.add_command(label="Compare IDFT", command=self.compare_idft)
        freq_menu.add_command(label="Compute DCT", command=self.compute_dct)
        freq_menu.add_command(label="Compare DCT", command=self.compare_dct_results)

        menubar.add_cascade(label="Frequency Domain", menu=freq_menu)
        # Time Domain Menu for time-domain operations
        time_domain_menu = Menu(menubar, tearoff=0)
        time_domain_menu.add_command(label="First Derivative", command=self.first_derivative)
        time_domain_menu.add_command(label="Second Derivative", command=self.second_derivative)
        time_domain_menu.add_command(label="Shift Signal", command=self.shift)
        time_domain_menu.add_command(label="Fold Signal (Reverse)", command=self.fold_signal)
        time_domain_menu.add_command(label="Shift and Fold", command=self.shift_and_fold_signal)
        time_domain_menu.add_command(label="FIR Resampling", command=self.create_fir_resampling_gui)  # Add new menu item

        lf = Label(self.master, text="Constant")
        lf.pack()

        txt = Text(self.master, width=50, height=2)
        txt.pack()


        # Shifting Buttons
        frame_shift = Frame(self.master, bg="black")
        frame_shift.pack(pady=5)

        shifting = Button(frame_shift, text="Shift", command=self.shift, bg="#2980b9", fg="white", font=("Arial", 10), width=20)  # Replace 'self.shift_function' with your actual function
        shifting.pack(side=LEFT, padx=5)

        shiftingFold = Button(frame_shift, text="Fold Shift", command=self.shift_and_fold_signal, bg="#2980b9", fg="white", font=("Arial", 10), width=20)  # Replace 'self.fold_shift_function' with your actual function
        shiftingFold.pack(side=LEFT, padx=5)

      

        menubar.add_cascade(label="Time Domain", menu=time_domain_menu)

        
        self.master.config(menu=menubar)

        # Smooth Buttons
        frame_smooth = Frame(self.master, bg="black")
        frame_smooth.pack(pady=5)

        smooth_button = Button(frame_smooth, text="Smooth Signal", command=self.smooth_signal, bg="#2980b9", fg="white", font=("Arial", 10), width=20)  # Replace 'self.smooth_signal_function' with your actual function
        smooth_button.pack(side=LEFT, padx=5)

        smoothCompare_button = Button(frame_smooth, text="Compare Smooth Signal", command=self.compare_signal, bg="#2980b9", fg="white", font=("Arial", 10), width=20)  # Replace 'self.compare_smooth_signal_function' with your actual function
        smoothCompare_button.pack(side=LEFT, padx=5)

        # Remove DC Buttons
        frame_remove_dc = Frame(self.master, bg="black")
        frame_remove_dc.pack(pady=5)

        remove_dc_time_button = Button(frame_remove_dc, text="Remove DC Time", command=self.remove_dc_time, bg="#2980b9", fg="white", font=("Arial", 10), width=20)  # Replace 'self.remove_dc_time_function' with your actual function
        remove_dc_time_button.pack(side=LEFT, padx=5)

        remove_dc_freq_button = Button(frame_remove_dc, text="Remove DC Freq", command=self.remove_dc_freq, bg="#2980b9", fg="white", font=("Arial", 10), width=20)  # Replace 'self.remove_dc_freq_function' with your actual function
        remove_dc_freq_button.pack(side=LEFT, padx=5)

        # Correlation Buttons
        frame_correlation = Frame(self.master, bg="black")
        frame_correlation.pack(pady=5)

        correlation_button = Button(frame_correlation, text="Correlation", command=self.compute_cross_correlation, bg="#2980b9", fg="white", font=("Arial", 10), width=20)  # Replace 'self.correlation_function' with your actual function
        correlation_button.pack(side=LEFT, padx=5)

        correlationCompare_button = Button(frame_correlation, text="Compare Correlation", command=self.compare_signalss, bg="#2980b9", fg="white", font=("Arial", 10), width=20)  # Replace 'self.compare_correlation_function' with your actual function
        correlationCompare_button.pack(side=LEFT, padx=5)


          # Convolution Button
        convolve_button = Button(self.master, text="Convolve Signals", command=self.convolve_signals,
                                  bg="#2980b9", fg="white", font=("Arial", 12), padx=10, pady=5)
        convolve_button.pack(pady=10)
        # Load Signal Button
        self.load_button = Button(self.master, text="Load Signal", command=self.load_signal,
                                  bg="#2980b9", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.load_button.pack(pady=(2, 10))

        # Compare Signals Button
        self.compare_button = Button(self.master, text="Compare Signals", command=self.compare_signals,
                                     bg="#2980b9", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.compare_button.pack(pady=(2, 20))

        self.window_button = Button(self.master, text="window Signals", command=self.create_fir_resampling_gui,
                                     bg="#2980b9", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.window_button.pack(pady=(2, 20))

        # Quantization Section Frame
        quantization_frame = Frame(self.master, bg="black", bd=2, relief=SUNKEN)
        quantization_frame.pack(pady=10, padx=10, fill="x")

        # Quantization label
        self.label = Label(quantization_frame, text="Enter the number of bits or levels:", bg="black",
                           fg="white", font=("Arial", 10))
        self.label.pack(pady=5)

        # Input Entry for levels/bits
        self.input_entry = Entry(quantization_frame, textvariable=self.levels_bits_entry,
                                 font=("Arial", 10), width=20, bg="#333", fg="white")
        self.input_entry.pack(pady=5)

        # Radio Buttons for Quantization Type
        quantization_type_label = Label(quantization_frame, text="Choose quantization input type:", bg="black",
                                        fg="white", font=("Arial", 10, "bold"))
        quantization_type_label.pack(pady=(10, 5))

        self.levels_radio = Radiobutton(quantization_frame, text="Number of Levels", variable=self.quantization_choice,
                                        value=1, bg="black", fg="white", selectcolor="black", font=("Arial", 10))
        self.levels_radio.pack(pady=2)

        self.bits_radio = Radiobutton(quantization_frame, text="Number of Bits", variable=self.quantization_choice,
                                      value=2, bg="black", fg="white", selectcolor="black", font=("Arial", 10))
        self.bits_radio.pack(pady=2)

        # Quantize Button
        self.quantize_button = Button(quantization_frame, text="Quantize", command=self.perform_quantization,
                                      bg="#2980b9", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.quantize_button.pack(pady=10)
    '''
    def open_fir_window(self):
        fir_window = Toplevel(self.master)
        fir_window.title("FIR Filter Resampling")
        fir_window.geometry("500x600")
        self.fir_frame = Frame(fir_window)
        self.fir_frame.pack(pady=20)
        self.create_fir_resampling_gui()  # Call the FIR GUI function here
    '''

    def create_fir_resampling_gui(self):
        task7_window = Toplevel(root)
        task7_window.geometry("500x400")
        task7_window.title("Task 7")
        task7_window.configure(bg="black")  # Set window background color to black

        self.fir_frame = Frame(task7_window, bg="black")  # Set frame background color to black
        self.fir_frame.pack(pady=20)

        # FIR Filter Section
        tk.Label(self.fir_frame, text="FIR Filter Design", font=("Arial", 15, "bold"), bg="black", fg="white").grid(row=0, column=0, pady=10)

        tk.Label(self.fir_frame, text="Filter Type", font=("Arial", 10, "bold"), bg="black", fg="white").grid(row=1, column=0)
        self.filter_type_var = StringVar(value="LowPass")
        tk.OptionMenu(self.fir_frame, self.filter_type_var, "LowPass LP", "HighPass HP", "BandPass BP", "BandStop BS").grid(row=1, column=1)

        tk.Label(self.fir_frame, text="Sampling Frequency (FS)", font=("Arial", 10, "bold"), bg="black", fg="white").grid(row=2, column=0)
        self.entry_fs = tk.Entry(self.fir_frame)
        self.entry_fs.grid(row=2, column=1)

        tk.Label(self.fir_frame, text="StopBandAttenuation (DB)", font=("Arial", 10, "bold"), bg="black", fg="white").grid(row=3, column=0)
        self.entry_stop_att = tk.Entry(self.fir_frame)
        self.entry_stop_att.grid(row=3, column=1)

        tk.Label(self.fir_frame, text="Cutoff Frequencies 'comma separated' (FC)", font=("Arial", 10, "bold"), bg="black", fg="white").grid(row=4, column=0)
        self.entry_fc = tk.Entry(self.fir_frame)
        self.entry_fc.grid(row=4, column=1)

        tk.Label(self.fir_frame, text="TransitionBand (TB) ", font=("Arial", 10, "bold"), bg="black", fg="white").grid(row=5, column=0)
        self.entry_trans_band = tk.Entry(self.fir_frame)
        self.entry_trans_band.grid(row=5, column=1)

        # Styled Buttons
        button_style = {"bg": "#2980b9", "fg": "white", "font": ("Arial", 12), "padx": 10, "pady": 5}
        tk.Button(self.fir_frame, text="Apply FIR Filter", command=self.FIR_filter, **button_style).grid(row=6, column=0, columnspan=2, pady=10)
        tk.Button(self.fir_frame, text="Convolve Signal", command=self.convolve_signal, **button_style).grid(row=7, column=0, columnspan=2, pady=10)





        # Resampling Section
        tk.Label(self.fir_frame, text="Sampling", font=("Arial", 15, "bold"), bg="black", fg="white").grid(row=8, column=0, pady=20)

        tk.Label(self.fir_frame, text="Interpolation Factor (L)", font=("Arial", 10, "bold"), bg="black", fg="white").grid(row=9, column=0)
        self.entry_L = tk.Entry(self.fir_frame)
        self.entry_L.grid(row=9, column=1)

        tk.Label(self.fir_frame, text="Decimation Factor (M)", font=("Arial", 10, "bold"), bg="black", fg="white").grid(row=10, column=0)
        self.entry_M = tk.Entry(self.fir_frame)
        self.entry_M.grid(row=10, column=1)

        tk.Label(self.fir_frame, text="Sampling Sampling Frequency (FS)", font=("Arial", 10, "bold"), bg="black", fg="white").grid(row=11, column=0)
        self.entry_resampling_fs = tk.Entry(self.fir_frame)
        self.entry_resampling_fs.grid(row=11, column=1)

        tk.Label(self.fir_frame, text="Sampling StopBandAttenuation (DB)", font=("Arial", 10, "bold"), bg="black", fg="white").grid(row=12, column=0)
        self.entry_stop_att_resampling = tk.Entry(self.fir_frame)
        self.entry_stop_att_resampling.grid(row=12, column=1)

        tk.Label(self.fir_frame, text="Sampling Cutoff Frequency (FC)", font=("Arial", 10, "bold"), bg="black", fg="white").grid(row=13, column=0)
        self.entry_resampling_cutoff = tk.Entry(self.fir_frame)
        self.entry_resampling_cutoff.grid(row=13, column=1)

        tk.Label(self.fir_frame, text="Sampling TransitionBand (TB)", font=("Arial", 10, "bold"), bg="black", fg="white").grid(row=14, column=0)
        self.entry_trans_band_resampling = tk.Entry(self.fir_frame)
        self.entry_trans_band_resampling.grid(row=14, column=1)

        tk.Button(self.fir_frame, text="Apply Sampling", command=self.perform_resampling, **button_style).grid(row=15, column=0, columnspan=2, pady=20)



 ########################################################################################################################################################################################################################

    '''
    def load_signal(self):
        file_path = filedialog.askopenfilename(filetypes=[("Signal Files", "*.txt")])
        if file_path:
            signal = self.read_signal_from_file(file_path)
            if signal is not None:
                t = np.arange(len(signal))  # Generate time samples for the length of the signal
                self.signals.append((t, signal))
                self.plot_signal((t, signal), label=f"Loaded Signal {len(self.signals)}")

    '''
    '''
    def read_signal_from_file(self, file_path):
        with open(file_path, 'r') as f:
            signal_type = int(f.readline().strip())  # 0 for Time, 1 for Frequency
            f.readline()  # Skip IsPeriodic
            n_samples = int(f.readline().strip())
            samples = []
            if signal_type == 0:  # Time Domain
                for _ in range(n_samples):
                    line = f.readline().strip().split()
                    samples.append(float(line[1]))  # Assuming [Index SampleAmp] format
            else:
                messagebox.showerror("Error", "This program only supports time-domain signals for now.")
                return None
        return np.array(samples)
    '''
    '''
    def read_signal_from_file(self, file_path):
        with open(file_path, 'r') as f:
            signal_type = int(f.readline().strip())  # 0 for Time, 1 for Frequency
            f.readline()  # Read the empty line
            
            n_samples = int(f.readline().strip())
            samples = []
            
            # Read each line for the number of samples expected
            for _ in range(n_samples):
                line = f.readline().strip()
                if line:  # Ensure the line is not empty
                    # Split by whitespace
                    values = line.split()
                    
                    # If we have two columns, take the second one
                    if len(values) > 1:
                        try:
                            sample = float(values[1].rstrip('f'))  # Remove trailing 'f' if present
                            samples.append(sample)
                        except ValueError:
                            print(f"Invalid sample value: {line}")  # Print error for invalid floats
                    else:
                        print(f"Expected two values, but got: {line}")  # If there's only one value

            return np.array(samples) if samples else None
    '''


    def load_signal(self):
        # Allow the user to load a signal from a file
        file_path = filedialog.askopenfilename(title="Select Signal File",
                                               filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")))

        if not file_path:
            print("No file selected.")
            return

        # Read the signal data from the file
        try:
            signal_data = self.read_signal_from_file(file_path)
            self.signals.append(signal_data)
            self.plot_signal(signal_data, label="Loaded Signal")
        except Exception as e:
            print(f"Error loading signal: {e}")

    def read_signal_from_file(self, file_path):
        """Reads a signal file with the given format."""
        with open(file_path, 'r') as f:
            # Read header information
            signal_type = int(f.readline().strip())  # First line (0 for time-domain signals)
            f.readline()  # Second line (unused)
            n_samples = int(f.readline().strip())  # Number of samples
            
            # Skip any unused line
            line = f.readline()

            # Read indices and values
            indices = []
            samples = []
            while line:
                parts = line.strip().split()
                if len(parts) == 2:
                    indices.append(int(parts[0]))  # Index
                    samples.append(float(parts[1]))  # Value
                line = f.readline()
        
        if signal_type != 0:
            messagebox.showerror("Error", "Only time-domain signals are supported.")
            return None
        
        return np.array(indices), np.array(samples)  # Return indices and samples as NumPy arrays


    def generate_sine_wave(self):
        self.signal, t = self.create_sinusoidal_signal(type="sin")
        self.signals.append((t, self.signal))
        self.plot_signal((t, self.signal), label=f"Generated Sine Wave {len(self.signals)}")

    def generate_cosine_wave(self):
        self.signal, t = self.create_sinusoidal_signal(type="cos")
        self.signals.append((t, self.signal))
        self.plot_signal((t, self.signal), label=f"Generated Cosine Wave {len(self.signals)}")

    def create_sinusoidal_signal(self, type):
        A = float(simpledialog.askstring("Input", "Enter amplitude (A):"))
        analog_frequency = float(simpledialog.askstring("Input", "Enter analog frequency (Hz):"))
        sampling_frequency = float(simpledialog.askstring("Input", "Enter sampling frequency (Hz):"))
        phase_shift = float(simpledialog.askstring("Input", "Enter phase shift (radians):"))

        T = 1 / sampling_frequency  # Sampling period
        t = np.arange(0, 1, T)  # Default to 1 second duration

        if type == "sin":
            signal = A * np.sin(2 * np.pi * analog_frequency * t + phase_shift)
        else:
            signal = A * np.cos(2 * np.pi * analog_frequency * t + phase_shift)

        return signal, t

    def plot_signal(self, signal_data, label):
        t, signal = signal_data
        plt.figure()

        plt.subplot(2, 1, 1)
        plt.plot(t, signal, label=f"{label} (Continuous)", color='b')
        plt.title(f"{label} - Continuous Representation")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.stem(t, signal, linefmt='r-', markerfmt='ro', basefmt='k', label=f"{label} (Discrete)")
        plt.title(f"{label} - Discrete Representation")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

    def plot_result(signal, title):
        plt.figure()
        plt.plot(signal)
        plt.title(title)
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.show()


    def are_signals_equal(self, signal1, signal2):
        # Compare the signals within a tolerance
        if len(signal1) != len(signal2):
            return False
        for s1, s2 in zip(signal1, signal2):
            if abs(s1 - s2) > 0.01:  # Allow for a tolerance of 0.01
                return False
        return True

    def compare_signals(self):
        if len(self.signals) < 2:
            messagebox.showerror("Error", "Not enough signals to compare.")
            return

        # Ask the user to select two signals to compare
        signal1_idx = simpledialog.askinteger("Input", f"Enter first signal number (1-{len(self.signals)}):")
        signal2_idx = simpledialog.askinteger("Input", f"Enter second signal number (1-{len(self.signals)}):")

        # Make sure the indices are valid
        if signal1_idx is None or signal2_idx is None or signal1_idx < 1 or signal2_idx < 1 or signal1_idx > len(self.signals) or signal2_idx > len(self.signals):
            messagebox.showerror("Error", "Invalid signal numbers selected.")
            return

        # Retrieve the selected signals
        signal1 = self.signals[signal1_idx - 1][1]  # Get the signal amplitudes
        signal2 = self.signals[signal2_idx - 1][1]  # Get the signal amplitudes

        # Compare signals
        if self.are_signals_equal(signal1, signal2):
            print("Test case passed successfully.")
        else:
            print("Test case failed.")

        # Plotting for comparison
        self.plot_two_signals(self.signals[signal1_idx - 1], self.signals[signal2_idx - 1])

    def plot_two_signals(self, signal1, signal2):
        t1, sig1 = signal1
        t2, sig2 = signal2
        
        plt.figure()
        
        # Plot first signal
        plt.subplot(2, 1, 1)
        plt.plot(t1, sig1, label="Signal 1 (Continuous)", color='b')
        plt.stem(t1, sig1, label='Signal 1 (Discrete)', linefmt='r-', markerfmt='ro', basefmt='k')
        plt.title("Signal 1")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        # Plot second signal
        plt.subplot(2, 1, 2)
        plt.plot(t2, sig2, label="Signal 2 (Continuous)", color='g')
        plt.stem(t2, sig2, label='Signal 2 (Discrete)', linefmt='c-', markerfmt='co', basefmt='k')
        plt.title("Signal 2")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        plt.tight_layout()
        plt.show()


 #####################################################################################################################################################################################################################


    def add_signals(self):
        if len(self.signals) < 2:
            messagebox.showerror("Error", "At least two signals are required for addition.")
            return

        # Ask the user to select signals for addition
        signal_indices = simpledialog.askstring("Input", "Enter signal numbers to add (comma-separated):")
        indices = list(map(int, signal_indices.split(",")))

        # Validate indices
        if any(idx < 1 or idx > len(self.signals) for idx in indices):
            messagebox.showerror("Error", "Invalid signal numbers selected.")
            return

        # Add signals
        result_signal = sum(self.signals[idx - 1][1] for idx in indices)  # Adding selected signals
        #result_index = len(self.signals) + 1  # Generate a new index for the result
        self.signals.append((self.signals[0][0], result_signal))  # Store result with time index
        self.plot_signal((self.signals[0][0], result_signal), label="Result of Addition")

    def subtract_signals(self):
        if len(self.signals) < 2:
            messagebox.showerror("Error", "At least two signals are required for subtraction.")
            return

        # Ask the user to select two signals
        signal1_idx = simpledialog.askinteger("Input", "Enter first signal number to subtract from:")
        signal2_idx = simpledialog.askinteger("Input", "Enter second signal number to subtract:")

        # Validate indices
        if signal1_idx < 1 or signal2_idx < 1 or signal1_idx > len(self.signals) or signal2_idx > len(self.signals):
            messagebox.showerror("Error", "Invalid signal numbers selected.")
            return

        # Check if the signals have the same length for subtraction
        signal1 = self.signals[signal1_idx - 1][1]
        signal2 = self.signals[signal2_idx - 1][1]

        if len(signal1) != len(signal2):
            messagebox.showerror("Error", "Signals must be of the same length for subtraction.")
            return

        # Subtract signals using absolute difference
        result_signal = np.abs(signal1 - signal2)
        result_index = len(self.signals) + 1  # Generate a new index for the result
        self.signals.append((self.signals[0][0], result_signal))  # Store result with time index
        self.plot_signal((self.signals[signal1_idx - 1][0], result_signal), label="Result of Absolute Subtraction")

    def multiply_signal(self):
        if len(self.signals) < 1:
            messagebox.showerror("Error", "At least one signal is required for multiplication.")
            return

        # Ask the user to select a signal and a constant to multiply
        signal_idx = simpledialog.askinteger("Input", "Enter signal number to multiply:")
        constant = float(simpledialog.askstring("Input", "Enter constant to multiply with:"))

        # Validate index
        if signal_idx < 1 or signal_idx > len(self.signals):
            messagebox.showerror("Error", "Invalid signal number selected.")
            return

        # Multiply signal by constant
        result_signal = self.signals[signal_idx - 1][1] * constant
        #result_index = len(self.signals) + 1  # Generate a new index for the result
        self.signals.append((self.signals[0][0], result_signal))  # Store result with time index
        self.plot_signal((self.signals[signal_idx - 1][0], result_signal), label="Result of Multiplication")

    def square_signal(self):
        if len(self.signals) < 1:
            messagebox.showerror("Error", "At least one signal is required for squaring.")
            return

        # Ask the user to select a signal to square
        signal_idx = simpledialog.askinteger("Input", "Enter signal number to square:")

        # Validate index
        if signal_idx < 1 or signal_idx > len(self.signals):
            messagebox.showerror("Error", "Invalid signal number selected.")
            return

        # Square the signal
        result_signal = self.signals[signal_idx - 1][1] ** 2
        #result_index = len(self.signals) + 1  # Generate a new index for the result
        self.signals.append((self.signals[0][0], result_signal))  # Store result with time index
        self.plot_signal((self.signals[signal_idx - 1][0], result_signal), label="Result of Squaring")

    def normalize_signal(self):
        if len(self.signals) < 1:
            messagebox.showerror("Error", "At least one signal is required for normalization.")
            return

        # Ask the user to select a signal for normalization
        signal_idx = simpledialog.askinteger("Input", "Enter signal number to normalize:")

        # Validate index
        if signal_idx < 1 or signal_idx > len(self.signals):
            messagebox.showerror("Error", "Invalid signal number selected.")
            return

        # Ask the user for the normalization range
        range_choice = simpledialog.askstring("Input", "Normalize to (0 to 1) or (-1 to 1)? Enter '0 to 1' or '-1 to 1':")
        
        # Normalize the signal
        min_value = np.min(self.signals[signal_idx - 1][1])
        max_value = np.max(self.signals[signal_idx - 1][1])
        
        # Check if min_value equals max_value to avoid division by zero
        if max_value == min_value:
            messagebox.showerror("Error", "Signal values are all the same. Cannot normalize.")
            return

        original_signal = self.signals[signal_idx - 1][1]
        
        if range_choice == "0 to 1":
            # Normalize to range [0, 1]
            normalized_signal = (original_signal - min_value) / (max_value - min_value)
            self.signals.append((self.signals[0][0], normalized_signal))  # Store result with time index


        elif range_choice == "-1 to 1":
            # Normalize to range [-1, 1]
            normalized_signal = 2 * (original_signal - min_value) / (max_value - min_value) - 1
            self.signals.append((self.signals[0][0], normalized_signal))  # Store result with time index

        else:
            messagebox.showerror("Error", "Invalid range choice. Please enter '0-1' or '-1 to 1'.")
            return

        self.plot_signal((self.signals[signal_idx - 1][0], normalized_signal), label="Normalized Signal")

    def accumulate_signal(self):
        if len(self.signals) < 1:
            messagebox.showerror("Error", "At least one signal is required for accumulation.")
            return

        # Ask the user to select a signal for accumulation
        signal_indx = simpledialog.askinteger("Input", "Enter hhe signal number to accumulate:")

        # Validate index
        if signal_indx < 1 or signal_indx > len(self.signals):
            messagebox.showerror("Error", "Invalid signal number selected")
            return

        # Get the selected signal (we assume it's a 1D list or array)
        signal = self.signals[signal_indx - 1][1]

        # Initialize an empty list for the accumulated signal
        accumulated_signal = []

        # Variable to keep track of the cumulative sum
        cumulate_sum = 0

        # Manually calculate the cumulative sum
        for i in signal:
            cumulate_sum += i  # Add the current value to the cumulative sum
            accumulated_signal.append(cumulate_sum)  # Append the result to the accumulated_signal list

        # Store the accumulated signal with the time index (same as the original signal)
        self.signals.append((self.signals[0][0], accumulated_signal))  # Store result with time index

        # Plot the accumulated signal
        self.plot_signal((self.signals[signal_indx - 1][0], accumulated_signal), label="Accumulated Signal")



    def quantize_signal(self, signal, levels):
        min_val, max_val = np.min(signal), np.max(signal)
        delta = (max_val - min_val) / levels

        quantized_signal = np.zeros_like(signal)
        quantization_error = np.zeros_like(signal)
        encoded_signal = []
        interval_indices = []

        for i, sample in enumerate(signal):
            zone_index = int(np.floor((sample - min_val) / delta))
            zone_index = np.clip(zone_index, 0, levels - 1)
            midpoint = min_val + (zone_index + 0.5) * delta
            quantized_signal[i] = midpoint
            quantization_error[i] = midpoint - sample
            encoded_signal.append(f"{zone_index:0{int(np.log2(levels))}b}")
            interval_indices.append(int(zone_index) + 1)
        return quantized_signal, quantization_error, encoded_signal, interval_indices

    def perform_quantization(self):
        file = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt')])
        if file:
            levels_or_bits_str = self.levels_bits_entry.get()  # Get the input as a string
            if not levels_or_bits_str.isdigit():  # Check if it's a valid integer
                messagebox.showerror("Error", "Please enter a valid number of bits or levels.")
                return

            levels_or_bits = int(levels_or_bits_str)  # Convert to integer if valid
            if self.quantization_choice.get() == 1:  # Levels
                levels = levels_or_bits
            else:  # Bits
                levels = 2 ** levels_or_bits

            signal = self.read_signal_from_file(file)
            if signal is not None:
                quantized_signal, quantization_error, encoded_signal, interval_indices = self.quantize_signal(signal, levels)
               
            else:
                 print("Please select a signal file.")
            if self.quantization_choice.get() == 1:  # Levels
                  QuantizationTest2("Quan2_Out.txt",interval_indices,encoded_signal,quantized_signal,quantization_error)
            else:
                 QuantizationTest1("Quan1_Out.txt",encoded_signal,quantized_signal)




 ##############################################################################################################################################################################################################


    def compute_dct(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal available. Generate or load a signal first.")
            return
        
        signal = self.signals[-1][1]  # Get the latest signal
        N = len(signal)
        self.dct_coefficients = np.zeros(N)
        
        # DCT computation using the formula provided
        for k in range(N):
            sum_term = 0
            for n in range(N):
                sum_term += signal[n] * math.cos((np.pi / (4 * N)) * (2 * n - 1) * (2 * k - 1))
            self.dct_coefficients[k] = np.sqrt(2 / N) * sum_term
        
        self.plot_dct(self.dct_coefficients)
        self.select_and_save_dct_coefficients(self.dct_coefficients)

    def plot_dct(self, dct_coefficients):
        plt.figure(figsize=(10, 6))
        plt.plot(dct_coefficients, label="DCT Coefficients")
        plt.title("DCT of the Signal")
        plt.xlabel("Coefficient Index")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        plt.show()

    def select_and_save_dct_coefficients(self, dct_coefficients):
        # Ask the user for the number of coefficients to save
        m = simpledialog.askinteger("Select Coefficients", "Enter the number of coefficients to save (m):")
        if m is None or m <= 0 or m > len(dct_coefficients):
            messagebox.showerror("Error", "Invalid number of coefficients selected.")
            return
        
        # Get first 'm' coefficients
        selected_coefficients = dct_coefficients[:m]
        
        # Save the selected coefficients to a file
        self.save_dct_coefficients(selected_coefficients)

    def save_dct_coefficients(self, dct_coefficients):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    # Write the header as in your actual format
                    f.write("0\n1\n6\n")  # Add header "0\n1\n6\n" for actual format
                    for coeff in dct_coefficients:
                        f.write(f"0 {coeff:.8f}\n")  #
                messagebox.showinfo("Success", "DCT coefficients saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while saving the file: {e}")
    def compare_dct_results(self):
        # Ensure DCT coefficients are computed first
        if self.dct_coefficients is None:
            messagebox.showerror("Error", "Please compute the DCT first.")
            return

        # Path to your expected DCT coefficients file
        file_name = "DCT_output.txt"
        
        result = self.SignalSamplesAreEqual(file_name, self.dct_coefficients)
        
        # Display message based on the result
        if "Error" in result:
            messagebox.showerror("Error", result)
        else:
            messagebox.showinfo("Success", result)

    @staticmethod
    def SignalSamplesAreEqual(file_name, samples, precision=2):
        expected_indices = []
        expected_samples = []

        # Read the expected output file
        try:
            with open(file_name, 'r') as f:
                # Skip first 4 lines as they're headers or irrelevant
                for _ in range(4):
                    next(f)

                # Read the actual expected values
                line = f.readline()
                while line:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        idx = int(parts[0])
                        coeff = float(parts[1])
                        expected_indices.append(idx)
                        expected_samples.append(round(coeff, precision))  # Round expected coefficients
                    line = f.readline()
        except FileNotFoundError:
            messagebox.showerror("Error", "File not found. Please check the file path.")
            return "Error: File not found. Please check the file path."

        # Skip the first 4 elements in the samples list (assuming you want to ignore the first 4 coefficients)
        samples = samples[1:]

        # Debug: Print lengths of expected and computed samples
        print(f"Expected samples length: {len(expected_samples)}")
        print(f"Computed samples length: {len(samples)}")

        # Ensure both lists are of the same length or handle accordingly
        if len(expected_samples) != len(samples):
            if len(samples) > len(expected_samples):
                samples = samples[:len(expected_samples)]  # Trim extra coefficients from computed samples
            else:
                messagebox.showerror("Error", f"Test case failed: Expected samples length is {len(expected_samples)}, but computed samples length is {len(samples)}.")
                return "Error: Sample lengths do not match."

        # Compare each coefficient with rounding and tolerance
        for i in range(len(expected_samples)):
            if round(samples[i], precision) != expected_samples[i]:
                messagebox.showerror("Error", f"Test case failed: Coefficient mismatch at index {i} (expected {expected_samples[i]}, got {round(samples[i], precision)}).")
                return f"Error: Coefficient mismatch at index {i} (expected {expected_samples[i]}, got {round(samples[i], precision)})."

        return "Success: Test case passed successfully."






    def calculate_dft(self, signal):
        N = len(signal)
        freq_components = np.zeros((N, 2))  
        for k in range(N):  
            for n in range(N):  
                freq_components[k][0] += signal[n] * np.cos(-2 * np.pi * k * n / N)  
                freq_components[k][1] += signal[n] * np.sin(-2 * np.pi * k * n / N)  

        amplitudes = np.linalg.norm(freq_components, axis=1)  
        phases = np.arctan2(freq_components[:, 1], freq_components[:, 0])  

        return amplitudes, phases, N

    def calculate_idft(self, amplitudes, phases):
        N = len(amplitudes)
        reconstructed_signal = np.zeros(N)

        for n in range(N):  
            for k in range(N):  
                real_part = amplitudes[k] * np.cos(phases[k])
                imag_part = amplitudes[k] * np.sin(phases[k])
                
                reconstructed_signal[n] += (real_part * np.cos(2 * np.pi * k * n / N) -
                                             imag_part * np.sin(2 * np.pi * k * n / N))

        # Normalize by the number of samples
        return reconstructed_signal / N  # Return the reconstructed signal scaled by N


    def apply_fourier_transform(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal available. Generate or load a signal first.")
            return

        signal = self.signals[-1][1]
        sampling_frequency = float(simpledialog.askstring("Input", "Enter sampling frequency (Hz):"))
        
        amplitudes, phases, N = self.calculate_dft(signal)
        frequencies = np.arange(N) / (N / sampling_frequency)  # Frequency bins

        # Plot Amplitude Spectrum
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.stem(frequencies[:N // 2], amplitudes[:N // 2])
        plt.title("Frequency vs Amplitude")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        # Plot Phase Spectrum
        plt.subplot(2, 1, 2)
        plt.stem(frequencies[:N // 2], phases[:N // 2])
        plt.title("Frequency vs Phase")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (radians)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

        # Store Fourier data for IDFT
        self.amplitudes = amplitudes
        self.phases = phases

    def reconstruct_signal(self):
        if not hasattr(self, 'amplitudes') or not hasattr(self, 'phases'):
            messagebox.showerror("Error", "No frequency components available. Apply Fourier Transform first.")
            return
        
        self.reconstructed_signal = self.calculate_idft(self.amplitudes, self.phases)  # Assign to self

        # Plot reconstructed signal
        t = self.t if hasattr(self, 't') else np.arange(len(self.reconstructed_signal))
        plt.figure()
        plt.plot(t, self.reconstructed_signal, color='purple')
        plt.title("Reconstructed Signal (IDFT)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()


    def load_dft(self, file_path):
        amplitudes = []
        phases = []
        
        with open(file_path, 'r') as f:
            # Skip the first three lines
            for _ in range(3):
                next(f)
            
            # Read the remaining lines for amplitude and phase data
            for line in f:
                if line.strip():  # Ensure line is not empty
                    parts = line.split()
                    if len(parts) == 2:
                        amplitude = float(parts[0].rstrip('f'))
                        phase = float(parts[1].rstrip('f'))
                        amplitudes.append(amplitude)
                        phases.append(phase)
                    else:
                        print(f"Unexpected line format: {line}")
        
        return amplitudes, phases

    def compare_dft(self):
        dft_file = filedialog.askopenfilename(filetypes=[("DFT Output Files", "*.txt")])
        if dft_file:
            dft_data = self.load_dft(dft_file)
            if dft_data is not None:
                amplitude_match = SignalCompareAmplitude(self.amplitudes, dft_data[0])
                phase_match = SignalComparePhaseShift(self.phases, dft_data[1])
                if amplitude_match and phase_match:
                    messagebox.showinfo("Comparison Result", "DFT data matches the loaded DFT output!")
                else:
                    messagebox.showwarning("Comparison Result", "DFT data does not match.")
    def load_idft(self, file_path):
        idft_samples = []
        
        with open(file_path, 'r') as f:
            # Skip the first three lines
            for _ in range(3):
                next(f)
            
            # Read each line and convert it to float
            for line in f:
                if line.strip():  # Ensure the line is not empty
                    try:
                        # Split line into components and take the second value
                        values = line.split()
                        if len(values) > 1:  # Check if we have at least two values
                            sample = float(values[1].rstrip('f'))  # Remove trailing 'f' if present
                            idft_samples.append(sample)
                    except ValueError:
                        print(f"Invalid sample value: {line}")  # Print error for invalid floats
                
        return np.array(idft_samples) if idft_samples else None



    def compare_idft(self):
        idft_file = filedialog.askopenfilename(filetypes=[("IDFT Output Files", "*.txt")])
        if idft_file:
            idft_data = self.load_idft(idft_file)
            if idft_data is not None:
                if not hasattr(self, 'reconstructed_signal'):
                    messagebox.showerror("Error", "No reconstructed signal available. Run IDFT first.")
                    return

                idft_match = SignalCompareAmplitude(self.reconstructed_signal, idft_data)
                if idft_match:
                    messagebox.showinfo("Comparison Result", "IDFT data matches the loaded IDFT output!")
                else:
                    messagebox.showwarning("Comparison Result", "IDFT data does not match.")


###########################################################################################################################################################################################################################################


    def first_derivative(self):
            InputSignal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                        19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0,
                        36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0,
                        53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0,
                        70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0,
                        87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0]
            expectedOutput_first = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1]
            

            """
            Write your Code here:
            Start
            """
            # Compute the first and second derivatives
            FirstDrev = []

            # First Derivative: y(n) = x(n) - x(n-1)
            for i in range(1, len(InputSignal)):
                FirstDrev.append(InputSignal[i] - InputSignal[i - 1])

            # Second Derivative: y(n) = x(n+1) - 2*x(n) + x(n-1)
            """
            End
            """

            """
            Testing your Code
            """
            # Plotting the results
            plt.figure(figsize=(10, 6))

            # Plot first derivative
            plt.subplot(2, 1, 1)
            plt.plot(range(1, len(InputSignal)), FirstDrev, label="First Derivative", color='blue')
            plt.title('First Derivative')
            plt.xlabel('n')
            plt.ylabel('First Derivative Value')
            plt.grid(True)
            plt.legend()

            if ((len(FirstDrev) != len(expectedOutput_first))):
                print("mismatch in length")
                return
            first = True
            for i in range(len(expectedOutput_first)):
                if abs(FirstDrev[i] - expectedOutput_first[i]) < 0.01:
                    continue
                else:
                    first = False
                    print("1st derivative wrong")
                    return

            if (first ):
                print("Derivative first Test case passed successfully")
            else:
                print("Derivative first Test case failed")
            plt.show()
            return

            

    def second_derivative (self):
        InputSignal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                       18.0,
                       19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0,
                       35.0,
                       36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0,
                       52.0,
                       53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0,
                       69.0,
                       70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0,
                       86.0,
                       87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0]
      
        expectedOutput_second = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0,
                                 0, 0, 0, 0, 0, 0, 0, 0]

        """
        Write your Code here:
        Start
        """
        # Compute the first and second derivatives
        SecondDrev = []

        # First Derivative: y(n) = x(n) - x(n-1)
        # Second Derivative: y(n) = x(n+1) - 2*x(n) + x(n-1)
        for i in range(1, len(InputSignal) - 1):
            SecondDrev.append(InputSignal[i + 1] - 2 * InputSignal[i] + InputSignal[i - 1])
        """
        End
        """

        """
        Testing your Code
        """
         # Plot second derivative
        plt.subplot(2, 1, 2)
        plt.plot(range(1, len(InputSignal) - 1), SecondDrev, label="Second Derivative", color='red')
        plt.title('Second Derivative')
        plt.xlabel('n')
        plt.ylabel('Second Derivative Value')
        plt.grid(True)
        plt.legend()

       
        
        if ((len(SecondDrev) != len(expectedOutput_second))):
            print("mismatch in length")
            return
        second = True

        for i in range(len(expectedOutput_second)):
            if abs(SecondDrev[i] - expectedOutput_second[i]) < 0.01:
                continue
            else:
                second = False
                print("2nd derivative wrong")
                return
        if (second):
            print("Derivative second Test case passed successfully")
        else:
            print("Derivative second Test case failed")
        plt.show()
        return







    def shift_signal(self):
        if len(self.signals) == 0:
            print("No signal available to shift.")
            return

        # Load the most recent signal
        signal_data = self.signals[-1]
        t, signal = signal_data

        # Ask the user for the shift amount (k)
        k = int(simpledialog.askstring("Input", "Enter shift value (k):"))
        print(f"Shift (k={k}) completed.")

        shifted_t, shifted_signal = self.shift(t, signal, k)

        #self.plot_signal((shifted_t, shifted_signal), label="Shifted Signal")
        


    def shift(self, t, signal, k):
        # Shifting the signal by k steps
        shifted_t = t + k * (t[1] - t[0])  # Adjust time indices for shifting
        shifted_signal = np.roll(signal, k)  # Roll the signal by k steps
        return shifted_t, shifted_signal

    def fold_signal(self):
        if len(self.signals) == 0:
            print("No signal available to fold.")
            return

        # Load the most recent signal
        t, signal = self.signals[-1]

        # Generate folded indices and samples
        folded_indices = [-i for i in t[::-1]]  # Reverse and negate the indices
        folded_samples = np.flip(signal)  # Use np.flip to reverse the signal

        # Plot the folded signal
        self.plot_signal((t, folded_samples), label="Folded Signal (Reversed)")
        #
        #print("Test case passed successfully")

        # Assuming Output_fold.txt is the file you want to use to save or load
        output_file = "Output_fold.txt"

        # Call ShiftFoldSignal function with the required arguments
        Shift_Fold_Signal(output_file, folded_indices, folded_samples)


    def shift_and_fold_signals(self):
        if len(self.signals) == 0:
            print("No signal available to shift and fold.")
            return

        # Load the most recent signal
        t, signal = self.signals[-1]
        k = int(simpledialog.askstring("Input", "Enter shift value (k):"))

        folded_indices = [-i for i in t[::-1]]
        folded_samples = list(signal[::-1])

        if k > 0:
            new_indices = [i + k for i in folded_indices]
            new_samples = folded_samples[k:] + [0] * k
        elif k < 0:
            new_indices = [i - abs(k) for i in folded_indices]
            new_samples = folded_samples + [0] * abs(k)
        else:
            new_indices = folded_indices
            new_samples = folded_samples

        print(f"Shift and Fold (k={k}) completed.")
        if k == 500:
             output_file = "Output_ShifFoldedby500.txt"
             Shift_Fold_Signal(output_file, new_indices, new_samples)
        elif k == -500:
             output_file = "Output_ShifFoldedby-500.txt"
             Shift_Fold_Signal(output_file, new_indices, new_samples)



    
    def load_signal(self):
        # Allow the user to load a signal from a file
        file_path = filedialog.askopenfilename(title="Select Signal File",
                                               filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")))

        if not file_path:
            print("No file selected.")
            return

        # Read the signal data from the file
        try:
            signal_data = self.read_signal_from_file(file_path)
            self.signals.append(signal_data)
            self.plot_signal(signal_data, label="Loaded Signal")
        except Exception as e:
            print(f"Error loading signal: {e}")

    def read_signal_from_file(self, file_path):
        # Read signal from a text file (expecting a format with two columns: time and amplitude)
        indices = []
        samples = []

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    indices.append(float(parts[0]))
                    samples.append(float(parts[1]))

        # Return the loaded signal as a tuple (time, amplitude)
        t = np.array(indices)
        signal = np.array(samples)
        return t, signal
   
    
##########################################################################################################################################################################################################################
# TASK 6
#########
    def smooth_signal(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded to smooth.")
            return
        
        signal_index = simpledialog.askinteger("Input", f"Enter signal index (1 to {len(self.signals)}):") - 1
        if signal_index < 0 or signal_index >= len(self.signals):
            messagebox.showerror("Error", "Invalid signal index.")
            return
        
        indices, signal = self.signals[signal_index]
        self.window_size = simpledialog.askinteger("Input", "Enter the window size for smoothing:")

        # Validate window size
        if self.window_size is None:
            messagebox.showerror("Error", "Window size input canceled.")
            return
        if self.window_size <= 0:
            messagebox.showerror("Error", "Window size must be a positive integer.")
            return

        # Smooth the signal
        self.smoothed_signal = self.compute_moving_average(signal, self.window_size)
        self.smoothed_indices = indices[:len(self.smoothed_signal)]  # Adjust indices length to match smoothed signal
        #self.plot_signal((self.smoothed_indices, self.smoothed_signal), label=f"Smoothed Signal {signal_index + 1}")


    def compute_moving_average(self, signal, window_size):
        return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')

    def compare_signal(self):
        if self.smoothed_signal is None or self.smoothed_indices is None:
            messagebox.showerror("Error", "No smoothed signal available. Please smooth a signal first.")
            return
        
        file_path = filedialog.askopenfilename(filetypes=[("Expected Signal Files", "*.txt")])
        if not file_path:
            return

        SignalSamplesAreEqual(file_path, self.smoothed_indices, self.smoothed_signal)



    def remove_dc_time(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded.")
            return

        signal_index = simpledialog.askinteger("Input", f"Enter signal index (1 to {len(self.signals)}):") - 1
        if signal_index < 0 or signal_index >= len(self.signals):
            messagebox.showerror("Error", "Invalid signal index.")
            return

        indices, signal = self.signals[signal_index]

        # Calculate mean manually
        sum_signal = 0
        for value in signal:
            sum_signal += value
        mean_value = sum_signal / len(signal)

        # Remove DC component
        dc_removed_signal = [value - mean_value for value in signal]

        # Option to compare with expected file
        file_path = filedialog.askopenfilename(title="Select Expected Result File",
                                            filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")))
        if file_path:
            SignalSamplesAreEqual(file_path, indices, dc_removed_signal)

        # Plot the DC removed signal
        self.plot_signal((indices, dc_removed_signal), label="DC Removed (Time Domain)")
        print(f"DC component (Time Domain) removed successfully. Mean value: {mean_value}")
        return dc_removed_signal
    

    def remove_dc_freq(self):
        if not self.signals:
            messagebox.showerror("Error", "No signal loaded.")
            return

        signal_index = simpledialog.askinteger("Input", f"Enter signal index (1 to {len(self.signals)}):") - 1
        if signal_index < 0 or signal_index >= len(self.signals):
            messagebox.showerror("Error", "Invalid signal index.")
            return

        indices, signal = self.signals[signal_index]

        # Perform DFT
        amplitudes, phases, N = self.calculate_dft(signal)

        # Remove the DC component (zero frequency component)
        amplitudes[0] = 0

        # Perform IDFT
        dc_removed_signal = self.calculate_idft(amplitudes, phases)

        # Compare with expected file
        file_path = filedialog.askopenfilename(title="Select Expected Result File",
                                            filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")))
        if not file_path:
            print("No comparison file selected.")
            return

        # Read and compare signals
        SignalSamplesAreEqual(file_path, indices, dc_removed_signal)

        # Plot the signal with DC component removed
        self.plot_signal((indices, dc_removed_signal), label="DC Removed (Frequency Domain)")


    def convolve_signals(self):
        if len(self.signals) < 2:
            messagebox.showerror("Error", "At least two signals are required for convolution.")
            return

        # Select two signals
        signal1_index = simpledialog.askinteger("Input", f"Enter the first signal index (1 to {len(self.signals)}):") - 1
        signal2_index = simpledialog.askinteger("Input", f"Enter the second signal index (1 to {len(self.signals)}):") - 1

        if signal1_index < 0 or signal1_index >= len(self.signals) or \
        signal2_index < 0 or signal2_index >= len(self.signals):
            messagebox.showerror("Error", "Invalid signal index.")
            return

        indices1, samples1 = self.signals[signal1_index]
        indices2, samples2 = self.signals[signal2_index]

        # Perform convolution manually
        convolved_samples = []
        convolved_indices = []

        # Compute the range of indices for the convolved signal
        start_index = indices1[0] + indices2[0]
        end_index = indices1[-1] + indices2[-1]
        convolved_indices = list(range(start_index, end_index + 1))

        for k in range(len(convolved_indices)):
            conv_sum = 0
            for n in range(len(samples1)):
                j = k - n
                if 0 <= j < len(samples2):
                    conv_sum += samples1[n] * samples2[j]
            convolved_samples.append(conv_sum)

        # Validate with ConvTest
        ConvTest(convolved_indices, convolved_samples)

        # Plot the convolved signal
        self.plot_signal((convolved_indices, convolved_samples), label="Convolved Signal")



    def compute_cross_correlation(self):
        if len(self.signals) < 2:
            messagebox.showerror("Error", "At least two signals are required for cross-correlation.")
            return

        # Select the last two signals for comparison
        _, signal1 = self.signals[-2]
        _, signal2 = self.signals[-1]

        # Ensure the signals are of the same length
        if len(signal1) != len(signal2):
            messagebox.showerror("Error", "Signals must have the same length for cross-correlation.")
            return

        # Compute normalized cross-correlation
        indices, correlation = self.normalized_cross_correlation(signal1, signal2)

        # Print the cross-correlation values
        print("Normalized Cross-Correlation Results:")
        for i, corr_value in zip(indices, correlation):
            print(f"{i} {corr_value}")

        # Plot the cross-correlation
        plt.figure()
        plt.stem(indices, correlation, linefmt="g-", markerfmt="go", basefmt="r-", label="Cross-Correlation")
        plt.title("Normalized Cross-Correlation")
        plt.xlabel("Lag")
        plt.ylabel("Correlation Coefficient")
        plt.legend()
        plt.grid()
        plt.show()

        # Save indices and correlation for later comparison
        self.last_correlation_indices = indices
        self.last_correlation_values = correlation

    def normalized_cross_correlation(self, signal1, signal2):
        N = len(signal1)
        correlation = []
        indices = []

        # Compute normalized cross-correlation for each lag
        for lag in range(N):
            # Shift signal2 by 'lag'
            shifted_signal2 = np.roll(signal2, lag)
            numerator = np.sum(signal1 * shifted_signal2)
            denominator = np.sqrt(np.sum(signal1 ** 2) * np.sum(signal2 ** 2))
            correlation_value = numerator / denominator

            indices.append(lag)
            correlation.append(correlation_value)

        return indices, correlation
    def compare_signalss(self):
        if not hasattr(self, 'last_correlation_indices') or not hasattr(self, 'last_correlation_values'):
            messagebox.showerror("Error", "No correlation computed to compare.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Expected Correlation File", "*.txt")])
        if not file_path:
            return

        Compare_Signals(file_path, self.last_correlation_indices, self.last_correlation_values)




    signals = []
    def shift(txt):
        input_file = filedialog.askopenfilename(title="Select Input File", filetypes=[("Text Files", "*.txt")])
        xShFo=[]
        x_indices = []
        ySignal = []
        with open(input_file, "r") as f:
            for i in range(3):
                next(f)
            for line in f:
                parts = line.strip().split()
                x_indices.append(float(parts[0]))
                ySignal.append(float(parts[1]))

        const = int(txt)
        for i in range(len(x_indices)):
            xShFo.append(x_indices[i] + const)
        figure, shift = plt.subplots(2, 1, figsize=(6, 8))
        shift[0].plot(x_indices, ySignal)
        shift[0].set_title("Original signal")
        shift[1].plot(xShFo, ySignal)
        shift[1].set_title("Shifted signal")
        plt.show()

    def shift_and_fold_signal(txt):
        input_file = filedialog.askopenfilename(title="Select Input File", filetypes=[("Text Files", "*.txt")])
        xShFo=[]
        x_indices = []
        ySignal = []
        yFolded = []
        with open(input_file, "r") as f:
            for i in range(3):
                next(f)
            for line in f:
                parts = line.strip().split()
                x_indices.append(float(parts[0]))
                ySignal.append(float(parts[1]))

        for i in range(len(ySignal) - 1, -1, -1):
            yFolded.append(ySignal[i])
        const = int(txt)
        for i in range(len(x_indices)):
            xShFo.append(x_indices[i] + const)
        figure, shift = plt.subplots(2, 1, figsize=(6, 8))
        shift[0].plot(x_indices, ySignal)
        shift[0].set_title("Original signal")
        shift[1].plot(xShFo, yFolded)
        shift[1].set_title("Shifted signal")
        plt.show()
        if const == 500:
            Shift_Fold_Signal("Output_ShifFoldedby500.txt",xShFo,yFolded)
        else:
            Shift_Fold_Signal("Output_ShiftFoldedby-500.txt", xShFo, yFolded)



###########################################################################################################################################################################################################
#TASK 7
##########
    def FIR_filter(self):
            try:
                # Read specifications
                filter_type = self.filter_type_var.get()
                fs = float(self.entry_fs.get())
                fc = list(map(float, self.entry_fc.get().split(",")))
                stop_att = float(self.entry_stop_att.get())
                trans_band = float(self.entry_trans_band.get())

                # Calculate filter coefficients
                h, n, N = calculate_fir_coefficients(filter_type, fc, fs, trans_band, stop_att)

                # Save coefficients with their corresponding n values
                with open("fir_coefficients.txt", "w") as f:
                    for n_val, coef in zip(n, h):
                        f.write(f"{n_val} {coef}\n")

                plt.figure()
                plt.title("Filter Coefficients")
                plt.stem(n, h)  
                plt.xlabel("n")
                plt.ylabel("h[n]")
                plt.grid()
                plt.show()

            except Exception as e:
                messagebox.showerror("Error", str(e))
    def convolve_signal(self):
        try:
            signal_file = filedialog.askopenfilename(title="Select Signal File")
            if not signal_file:
                messagebox.showwarning("No File Selected", "Please select a signal file.")
                return

            try:
                with open(signal_file, "r") as f:
                    f.readline()  
                    f.readline()  
                    N = int(f.readline().strip())  

                    expected_indices = []
                    expected_samples = []
                    line = f.readline()
                    while line:
                        L = line.strip()
                        if len(L.split()) == 2:
                            L = L.split()
                            V1 = int(L[0]) 
                            V2 = float(L[1])  
                            expected_indices.append(V1)
                            expected_samples.append(V2)
                        line = f.readline()

                    x_values = np.array(expected_indices)
                    y_values = np.array(expected_samples)

                    if len(x_values) != N:
                        raise ValueError(f"Expected {N} samples, but found {len(x_values)}.")
            except Exception as e:
                messagebox.showerror("File Error", f"Error reading signal file: {e}")
                return

            try:
                filter_type = self.filter_type_var.get()
                fs = float(self.entry_fs.get())
                fc = list(map(float, self.entry_fc.get().split(",")))
                stop_att = float(self.entry_stop_att.get())
                trans_band = float(self.entry_trans_band.get())

                h, n, _ = calculate_fir_coefficients(filter_type, fc, fs, trans_band, stop_att)
            except Exception as e:
                messagebox.showerror("Filter Error", f"Error calculating filter coefficients: {e}")
                return

            filtered_x, filtered_y = apply_filter(x_values, y_values, n, h)
            compare_FIR(filtered_x,filtered_y)

            output_file = "filtered_signal.txt"
            with open(output_file, "w") as f:
                for x, y_f in zip(filtered_x, filtered_y):
                    f.write(f"{x} {y_f}\n")

            plt.figure(figsize=(10, 6))
            plt.plot(x_values, y_values, label="Original Signal", color='blue')
            plt.plot(filtered_x, filtered_y, label="Filtered Signal", linestyle="--", color='red')
            plt.xlabel("x")
            plt.ylabel("Signal Amplitude")
            plt.title("Signal Filtering")
            plt.legend()
            plt.grid()
            plt.show()


        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    def perform_resampling(self):
        try:
            # Load input signal
            signal_file = filedialog.askopenfilename(title="Select Signal File")
            with open(signal_file, "r") as f:
                f.readline()  
                f.readline() 
                N = int(f.readline().strip())  

                expected_indices = []
                expected_samples = []
                line = f.readline()
                while line:
                    L = line.strip()
                    if len(L.split()) == 2:
                        L = L.split()
                        V1 = int(L[0])  
                        V2 = float(L[1])  
                        expected_indices.append(V1)
                        expected_samples.append(V2)
                    line = f.readline()

                x_values = np.array(expected_indices)
                y_values = np.array(expected_samples)
            M = int(self.entry_M.get())
            L = int(self.entry_L.get())
            fs = float(self.entry_resampling_fs.get())
            cutoff_freq = float(self.entry_resampling_cutoff.get())
            stop_att = float(self.entry_stop_att_resampling.get())
            trans_band = float(self.entry_trans_band_resampling.get())
            # Perform resampling
            x_axis, resampled_signal = resample_signal( x_values ,y_values, M, L, fs, cutoff_freq, stop_att, trans_band)
            print("resampled_signal:")
            print(resampled_signal)
            print("x_axis:")
            print(x_axis)

            compare_sampling(x_axis, resampled_signal)
            output_file = "resampled_signal.txt"
            with open(output_file, "w") as f:
                for x, y_f in zip(x_axis,  resampled_signal):
                    f.write(f"{x} {y_f}\n")
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.title("Original Signal")
            plt.plot(y_values)
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.title("Sampled Signal")
            plt.plot(resampled_signal)
            plt.grid()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", str(e))
   

###########################################################################################################################################################################################################
#TASK 7 
#########

def compare_FIR(n,h):
    case_number = int(input("Enter a number between 1 and 8 to select a case: "))

    if case_number == 1:
        Compare_Signals7("lab 7/FIR test cases/Testcase 1/LPFCoefficients.txt",n, h)

    elif case_number == 2:
        Compare_Signals7("lab 7/FIR test cases/Testcase 2/ecg_low_pass_filtered.txt",n, h)

    elif case_number == 3:
        Compare_Signals7("lab 7/FIR test cases/Testcase 3/HPFCoefficients.txt",n,h)

    elif case_number == 4:
        Compare_Signals7("lab 7/FIR test cases/Testcase 4/ecg_high_pass_filtered.txt",n,h)

    elif case_number == 5:
        Compare_Signals7("lab 7/FIR test cases/Testcase 5/BPFCoefficients.txt",n,h)

    elif case_number == 6:
        Compare_Signals7("lab 7/FIR test cases/Testcase 6/ecg_band_pass_filtered.txt",n,h)

    elif case_number == 7:

        Compare_Signals7("lab 7/FIR test cases/Testcase 7/BSFCoefficients.txt",n,h)
    elif case_number == 8:

        Compare_Signals7("lab 7/FIR test cases/Testcase 8/ecg_band_stop_filtered.txt",n,h)


def compare_sampling(n,h):
    case_number = int(input("Enter a number between 1 and 3 to select a case: "))

    if case_number == 1:
        Compare_Signals7("lab 7/Sampling test cases/Testcase 1/Sampling_Down.txt",n, h)
    elif case_number == 2:
        Compare_Signals7("lab 7/Sampling test cases/Testcase 2/Sampling_Up.txt",n, h)
    elif case_number == 3:
        Compare_Signals7("lab 7/Sampling test cases/Testcase 3/Sampling_Up_Down.txt",n, h)



def hamming_window(N,n):
    return [0.54 + 0.46 * math.cos(2 * math.pi * n_i / N) for n_i in n]
def hanning_window(N,n):
    return [0.5 + (0.5* math.cos(2 * math.pi * n_i / N)) for n_i in n]
def rectangular_window(N):
    return [1] * (N + 1)
def blackman_window(N,n):
    return [0.42 + 0.5 * math.cos(2 * math.pi * n_i / (N-1)) + 0.08 * math.cos(4 * math.pi * n_i / (N-1)) for n_i in n]
def choose_window(stop_att):
    if stop_att <= 21:
        return "rectangular"
    elif stop_att <= 44:
        return "hanning"
    elif stop_att <= 53:
        return "hamming"
    elif stop_att <= 74:
        return "blackman"
    else:
        raise ValueError("Not Allow")

def calculate_fir_coefficients(filter_type, fc, fs, trans_band, stop_att):
    # Adjust cutoff frequencies and normalize
    if filter_type == "LowPass LP":
        fc = [fc[0] + trans_band / 2]
    elif filter_type == "HighPass HP":
        fc = [fc[0] - trans_band / 2]
    elif filter_type == "BandPass BP":
        fc = [fc[0] - trans_band / 2, fc[1] + trans_band / 2]
    elif filter_type == "BandStop BS":
        fc = [fc[0] + trans_band / 2, fc[1] - trans_band / 2]
    else:
        raise ValueError("Not Allow")

    fc = [f / fs for f in fc]  # Normalize frequencies
    trans_width = trans_band / fs
    if stop_att <= 21:
        N = int(np.ceil(0.9 / trans_width))
    elif stop_att <= 44:
        N = int(np.ceil(3.1 / trans_width))
    elif stop_att <= 53:
        N = int(np.ceil(3.3 / trans_width))
    elif stop_att <= 74:
        N = int(np.ceil(5.5 / trans_width))
    else:
        raise ValueError("Not Allow")
    if N % 2 == 0:
        N += 1

    # Calculate ideal impulse response
    n = np.arange(-(N // 2), N // 2 + 1)

    h_d = np.zeros_like(n, dtype=float)

    if filter_type == "LowPass LP":
        h_d[n == 0] = 2 * fc[0]

        h_d[n != 0] = 2 * fc[0] * np.sin(2 * np.pi * fc[0] * n[n != 0]) / (2 * np.pi * fc[0] * n[n != 0])

    elif filter_type == "HighPass HP":
        h_d[n == 0] = 1 - 2 * fc[0]

        h_d[n != 0] = -2 * fc[0] * np.sin(2 * np.pi * fc[0] * n[n != 0]) / (2 * np.pi * fc[0] * n[n != 0])

    elif filter_type == "BandPass BP":
        h_d[n == 0] = 2 * (fc[1] - fc[0])

        h_d[n != 0] = (
                2 * fc[1] * np.sin(2 * np.pi * fc[1] * n[n != 0]) / (2 * np.pi * fc[1] * n[n != 0])
                - 2 * fc[0] * np.sin(2 * np.pi * fc[0] * n[n != 0]) / (2 * np.pi * fc[0] * n[n != 0])
        )

    elif filter_type == "BandStop BS":
        h_d[n == 0] = 1 - 2 * (fc[1] - fc[0])

        h_d[n != 0] = (
                2 * fc[0] * np.sin(2 * np.pi * fc[0] * n[n != 0]) / (2 * np.pi * fc[0] * n[n != 0])
                - 2 * fc[1] * np.sin(2 * np.pi * fc[1] * n[n != 0]) / (2 * np.pi * fc[1] * n[n != 0])
        )

    else:
        raise ValueError("Not Allow")

    # Apply window
    window = choose_window(stop_att)
    if window == "rectangular":
        #w = np.ones(N + 1)
        w=rectangular_window(N)
    elif window == "hanning":
        #w = np.hanning(N + 1)
        w=hanning_window(N,n)
    elif window == "hamming":
        #w = np.hamming(N + 1)
        w=hamming_window(N,n)
    elif window == "blackman":
        #w = np.blackman(N + 1)
        w=blackman_window(N,n)

    if len(w) != len(h_d):
        w = np.resize(w, h_d.shape)

    h = h_d * w

    compare_FIR(n, h)
    return h,n, N
def apply_filter(x_values1, y_values1, x_values2, y_values2):
    len1 = len(y_values1)
    len2 = len(y_values2)
    result=[]
    start_index = int(min(x_values1) + min(x_values2))
    end_index = int(max(x_values1) + max(x_values2))

    x_values = list(range(start_index, end_index + 1))
    for n in range(len1 + len2 - 1):
        sum = 0
        for m in range(min(n, len1 - 1) + 1):
            if 0 <= n - m < len2:
                sum += y_values1[m] * y_values2[n - m]
        result.append(sum)
    return x_values, result

def resample_signal(input_x,input_signal, M, L, fs, cutoff_freq, stop_att, trans_band):
    if not isinstance(cutoff_freq, list):
        cutoff_freq = [cutoff_freq]

    h, n, _ = calculate_fir_coefficients("LowPass LP", cutoff_freq, fs, trans_band, stop_att)

    if M == 0 and L != 0:
        upsampled_signal = np.zeros(len(input_signal) * L)
        upsampled_signal[::L] = input_signal
        upsampled_x=np.arange(0, len(input_x)*L-2 )
        filtered_x,filtered_signal = apply_filter(upsampled_x,upsampled_signal,n,h)

        return filtered_x, filtered_signal

    elif M != 0 and L == 0:
        # Filter and downsample
        filtered_x,filtered_signal = apply_filter(input_x,input_signal,n,h)
        downsampled_signal = filtered_signal[::M]
        downsampled_x = filtered_x[:len(downsampled_signal)]

        return downsampled_x,downsampled_signal

    elif M != 0 and L != 0:
        # Upsample, filter, and downsample
        upsampled_signal = np.zeros(len(input_signal) * L)
        upsampled_signal[::L] = input_signal
        upsampled_x = np.arange(0, len(input_x)*L-2)
        filtered_x, filtered_signal = apply_filter(upsampled_x,upsampled_signal,n,h)
        resampled_signal = filtered_signal[::M]
        resampled_signal=resampled_signal[:(len(resampled_signal)-1)]
        downsampled_x = filtered_x[:len(resampled_signal)]
        print(" len downsampled_x")
        print(len(downsampled_x))
        print(" len resampled_signal")
        print(len(resampled_signal))
        return downsampled_x, resampled_signal

    else:
        raise ValueError("L and M can't be zero.")


def Compare_Signals7(file_name,Your_indices,Your_samples):      
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one") 
            return
    print("Test case passed successfully")


###########################################################################################################################################################################################################
def SignalSamplesAreEqual(file_name,indices,samples):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
                
    if len(expected_samples)!=len(samples):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one") 
            return
    print("Test case passed successfully")


def ConvTest(Your_indices,Your_samples): 
    """
    Test inputs
    InputIndicesSignal1 =[-2, -1, 0, 1]
    InputSamplesSignal1 = [1, 2, 1, 1 ]
    
    InputIndicesSignal2=[0, 1, 2, 3, 4, 5 ]
    InputSamplesSignal2 = [ 1, -1, 0, 0, 1, 1 ]
    """
    
    expected_indices=[-2, -1, 0, 1, 2, 3, 4, 5, 6]
    expected_samples = [1, 1, -1, 0, 0, 3, 3, 2, 1 ]

    
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Conv Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Conv Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Conv Test case failed, your signal have different values from the expected one") 
            return
    print("Conv Test case passed successfully")


def Compare_Signals(file_name,Your_indices,Your_samples):      
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Correlation Test case failed, your signal have different values from the expected one") 
            return
    print("Correlation Test case passed successfully")

###############################################################################################################################################################################################




def Shift_Fold_Signal(file_name,Your_indices,Your_samples):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Shift_Fold_Signal Test case failed, your signal have different values from the expected one")
            return
    print("Shift_Fold_Signal Test case passed successfully")
###########################################################################################################################################################################################################################################
# Use to test the Amplitude of DFT and IDFT
def SignalCompareAmplitude(SignalInput=[], SignalOutput=[]):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            # Check if the amplitude is approximately equal
            if not np.isclose(SignalInput[i], SignalOutput[i], rtol=1e-2):  # 2% tolerance
                print("Amplitude Passed failed")  
                return False
    print("Amplitude Passed successfully")  
    return True
def RoundPhaseShift(P):
    # Normalize phase to be in the range [0, 2*pi)
    while P < 0:
        P += 2 * math.pi
    return float(P % (2 * math.pi))

def SignalComparePhaseShift(SignalInput=[], SignalOutput=[]):
    if len(SignalInput) != len(SignalOutput):
        return False
    else:
        for i in range(len(SignalInput)):
            # Normalize both phases
            normalized_input = RoundPhaseShift(SignalInput[i])
            normalized_output = RoundPhaseShift(SignalOutput[i])
            # Check if the normalized phase is approximately equal
            if not np.isclose(normalized_input, normalized_output, rtol=1e-2):  # 2% tolerance
                print("Phase Passed failed")
                return False
    print("Phase Passed successfully")
    return True


 ##################################################################################################################################################################################################################




def QuantizationTest1(file_name,Your_EncodedValues,Your_QuantizedValues):
    expectedEncodedValues=[]
    expectedQuantizedValues=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V2=str(L[0])
                V3=float(L[1])
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                line = f.readline()
            else:
                break
    if( (len(Your_EncodedValues)!=len(expectedEncodedValues)) or (len(Your_QuantizedValues)!=len(expectedQuantizedValues))):
        print("QuantizationTest1 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            print("QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one")
            return
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print("QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one")
            return
    print("QuantizationTest1 Test case passed successfully")


def QuantizationTest2(file_name, Your_IntervalIndices, Your_EncodedValues, Your_QuantizedValues, Your_SampledError):
    expectedIntervalIndices = []
    expectedEncodedValues = []
    expectedQuantizedValues = []
    expectedSampledError = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 4:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = str(L[1])
                V3 = float(L[2])
                V4 = float(L[3])
                expectedIntervalIndices.append(V1)
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                expectedSampledError.append(V4)
                line = f.readline()
            else:
                break
    if (len(Your_IntervalIndices) != len(expectedIntervalIndices)
            or len(Your_EncodedValues) != len(expectedEncodedValues)
            or len(Your_QuantizedValues) != len(expectedQuantizedValues)
            or len(Your_SampledError) != len(expectedSampledError)):
        print("QuantizationTest2 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_IntervalIndices)):
        if (Your_IntervalIndices[i] != expectedIntervalIndices[i]):
            print("QuantizationTest2 Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(Your_EncodedValues)):
        if (Your_EncodedValues[i] != expectedEncodedValues[i]):
            print(
                "QuantizationTest2 Test case failed, your EncodedValues have different EncodedValues from the expected one")
            return

    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print(
                "QuantizationTest2 Test case failed, your QuantizedValues have different values from the expected one")
            return
    for i in range(len(expectedSampledError)):
        if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
            continue
        else:
            print("QuantizationTest2 Test case failed, your SampledError have different values from the expected one")
            return
    print("QuantizationTest2 Test case passed successfully")






########################################################################################################################################################################################################################################################################################

if __name__ == "__main__":
    root = Tk()
    app = SignalProcessor(root)
    root.mainloop()