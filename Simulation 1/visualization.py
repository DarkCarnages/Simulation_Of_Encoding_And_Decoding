import matplotlib.pyplot as plt
import numpy as np

def plot_signals(bitstream, encoded_signal, encoding_type="Manchester"):
    """
    Plots the original bitstream and the encoded signal.
    encoding_type: "Manchester" or "MLT-3"
    """
    plt.figure(figsize=(12, 5))

    # --- Plot original bitstream ---
    plt.subplot(2, 1, 1)
    t_bits = np.arange(len(bitstream) + 1)
    # Prepend the first value to the bitstream so the step plot transitions at each bit index
    bits_plot = np.array([bitstream[0]] + bitstream)
    plt.step(t_bits, bits_plot, where='pre', linewidth=2)
    plt.scatter(np.arange(len(bitstream)), bitstream, color='red', zorder=5, label='Bit Value')
    plt.title("Original Bitstream")
    plt.ylim(-0.5, 1.5)
    plt.xlabel("Bit Index")
    plt.ylabel("Bit Value")
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')

    # --- Plot encoded signal ---
    plt.subplot(2, 1, 2)
    if encoding_type == "Manchester":
        t_encoded = np.arange(len(encoded_signal) + 1) / 2
        signal_plot = np.array([encoded_signal[0]] + encoded_signal)
        plt.step(t_encoded, signal_plot, where='pre', linewidth=2)
        plt.title("Manchester Encoded Signal")
        plt.ylim(-0.5, 1.5)
        plt.xlabel("Bit Index (Manchester half-steps)")
        plt.ylabel("Signal Level")
    elif encoding_type == "MLT-3":
        t_encoded = np.arange(len(encoded_signal) + 1)
        signal_plot = np.array([encoded_signal[0]] + encoded_signal)
        plt.step(t_encoded, signal_plot, where='pre', linewidth=2)
        plt.title("MLT-3 Encoded Signal")
        plt.ylim(-1.5, 1.5)
        plt.xlabel("Bit Index")
        plt.ylabel("Signal Level")
    else:
        raise ValueError("Invalid encoding_type: choose 'Manchester' or 'MLT-3'")

    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()