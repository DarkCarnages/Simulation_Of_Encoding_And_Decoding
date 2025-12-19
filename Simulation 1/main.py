from encoding import manchester_encode, manchester_decode, mlt3_encode, mlt3_decode
from visualization import plot_signals

def get_and_save_bitstream():
    """
    Prompts the user for a bitstream, validates it, prints it, and saves it to a file.
    Returns the bitstream as a list of integers.
    """
    while True:
        bitstream_str = input("Enter a bitstream (sequence of 0s and 1s, e.g., 1011001): ").strip()
        if all(c in '01' for c in bitstream_str) and bitstream_str:
            break
        print("Invalid input! Please enter only 0s and 1s.")

    bitstream = [int(c) for c in bitstream_str]

    print(f"\nBitstream entered: {bitstream}")

    with open("output_bitstream.txt", "w") as f:
        f.write("".join(str(bit) for bit in bitstream))
        f.write("\n")
        f.write(str(bitstream))  # also save as list for reference

    print("Bitstream has been saved to output_bitstream.txt")
    return bitstream

def main():
    bitstream = get_and_save_bitstream()
    print("\nSelect encoding scheme:")
    print("1. Manchester Encoding")
    print("2. MLT-3 Encoding")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == '1':
        encoded = manchester_encode(bitstream)
        print(f"\nManchester Encoded Signal: {encoded}")
        decoded = manchester_decode(encoded)
        print(f"Manchester Decoded Bitstream: {decoded}")
        plot_signals(bitstream, encoded, encoding_type="Manchester")  # <--- ADD THIS!
    elif choice == '2':
        encoded = mlt3_encode(bitstream)
        print(f"\nMLT-3 Encoded Signal: {encoded}")
        decoded = mlt3_decode(encoded)
        print(f"MLT-3 Decoded Bitstream: {decoded}")
        plot_signals(bitstream, encoded, encoding_type="MLT-3")  # <--- AND THIS!
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()