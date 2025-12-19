import numpy as np
def manchester_encode(bitstream):
    encoded = []
    for bit in bitstream:
        if bit == 1:
            encoded.extend([1, 0])
        elif bit == 0:
            encoded.extend([0, 1])
        else:
            raise ValueError("Invalid bit in bitstream")
    return encoded

def manchester_decode(encoded):
    encoded = list(encoded)
    if len(encoded) % 2 != 0:
        return ['?'] * (len(encoded) // 2)
    decoded = []
    for i in range(0, len(encoded), 2):
        pair = [int(encoded[i]), int(encoded[i+1])]
        if pair == [1, 0]:
            decoded.append(1)
        elif pair == [0, 1]:
            decoded.append(0)
        else:
            decoded.append('?')
    return decoded

def mlt3_encode(bitstream):
    levels = [0, 1, 0, -1]
    state_idx = 0
    signal = []
    for bit in bitstream:
        if bit == 1:
            state_idx = (state_idx + 1) % 4
            signal.append(levels[state_idx])
        else:
            signal.append(levels[state_idx])
    return signal

def mlt3_decode(signal):
    signal = list(signal)
    if not signal:
        return []
    prev = signal[0]
    decoded = []
    for s in signal:
        if s != prev:
            decoded.append(1)
        else:
            decoded.append(0)
        prev = s
    return decoded

def nrz_encode(bitstream):
    return list(bitstream)

def nrz_decode(encoded):
    try:
        return [int(b) if b in [0, 1] else '?' for b in encoded]
    except Exception:
        return ['?' for _ in encoded]

def nrzi_encode(bitstream):
    output = []
    last = 0
    for bit in bitstream:
        if bit == 1:
            last = 1 - last
        output.append(last)
    return output

def nrzi_decode(encoded):
    encoded = list(encoded)
    try:
        decoded = []
        prev = 0
        for s in encoded:
            if s == prev:
                decoded.append(0)
            else:
                decoded.append(1)
            prev = s
        return decoded
    except Exception:
        return ['?' for _ in encoded]

def ami_encode(bitstream):
    output = []
    last_mark = -1
    for bit in bitstream:
        if bit == 1:
            last_mark *= -1
            output.append(last_mark)
        else:
            output.append(0)
    return output

def ami_decode(encoded):
    try:
        return [1 if abs(int(s)) == 1 else 0 if int(s) == 0 else '?' for s in encoded]
    except Exception:
        return ['?' for _ in encoded]

FOURB_FIVEB_TABLE = {
    '0000': '11110', '0001': '01001', '0010': '10100', '0011': '10101',
    '0100': '01010', '0101': '01011', '0110': '01110', '0111': '01111',
    '1000': '10010', '1001': '10011', '1010': '10110', '1011': '10111',
    '1100': '11010', '1101': '11011', '1110': '11100', '1111': '11101',
}
FIVEB_FOURB_TABLE = {v: k for k, v in FOURB_FIVEB_TABLE.items()}

def _bits_to_str(bits):
    return ''.join(str(int(b)) for b in bits)

def _str_to_bits(s):
    return [int(b) for b in s]

def fourb_fiveb_encode(bitstream):
    bits = list(bitstream)
    while len(bits) % 4 != 0:
        bits.append(0)
    encoded = []
    for i in range(0, len(bits), 4):
        chunk = bits[i:i+4]
        code = FOURB_FIVEB_TABLE.get(_bits_to_str(chunk), None)
        if code is not None:
            encoded.extend(_str_to_bits(code))
        else:
            encoded.extend(['?']*5)
    return encoded

def fourb_fiveb_decode(encoded):
    bits = list(encoded)
    if len(bits) % 5 != 0:
        return ['?'] * (len(bits) // 5 * 4)
    decoded = []
    for i in range(0, len(bits), 5):
        code = _bits_to_str(bits[i:i+5])
        if code in FIVEB_FOURB_TABLE:
            decoded.extend(_str_to_bits(FIVEB_FOURB_TABLE[code]))
        else:
            decoded.extend(['?','?','?','?'])
    return decoded




def bipolar_rz_encode(bitstream):
    """+1, -1 alternating for 1s, zero for 0, return-to-zero in middle."""
    signal = []
    last = -1
    for bit in bitstream:
        if bit == 1:
            last = -last  # alternate polarity
            signal.extend([last, 0])  # RZ: return to zero
        else:
            signal.extend([0, 0])
    return signal

def bipolar_rz_decode(encoded):
    """Decode RZ bipolar: first half determines bit."""
    decoded = []
    if len(encoded) % 2 != 0:
        return ['?'] * (len(encoded)//2)

    for i in range(0, len(encoded), 2):
        if encoded[i] in [1, -1]:
            decoded.append(1)
        elif encoded[i] == 0:
            decoded.append(0)
        else:
            decoded.append('?')
    return decoded


def diff_manchester_encode(bitstream):
    signal = []
    last = 1  # starting polarity
    for bit in bitstream:
        if bit == 1:
            # Transition at the middle, not at start
            signal.extend([last, -last])
            last = -last
        else:
            # Transition at start AND middle
            last = -last
            signal.extend([last, -last])
    return signal

def diff_manchester_decode(encoded):
    if len(encoded) % 2 != 0:
        return ['?'] * (len(encoded)//2)
        
    decoded = []
    for i in range(0, len(encoded), 2):
        first, second = encoded[i], encoded[i+1]
        if first != second:
            decoded.append(0)
        else:
            decoded.append('?') # invalid but safety
    return decoded

def ask_encode(bitstream, amplitude_high=1, amplitude_low=0):
    return [amplitude_high if b == 1 else amplitude_low for b in bitstream]

def ask_decode(encoded):
    return [1 if s > 0.5 else 0 for s in encoded]


def fsk_encode(bitstream, f0=1, f1=2, samples=20):
    signal = []
    t = np.linspace(0, np.pi*2, samples)
    for b in bitstream:
        freq = f1 if b == 1 else f0
        signal.extend(np.sin(freq*t))
    return list(signal)

def fsk_decode(encoded):
    # basic detection: compare slope patterns
    decoded = []
    chunk = len(encoded)//len(encoded)  # placeholder
    # Advanced decoding requires FFT; for now, placeholder:
    return ['?'] * (len(encoded)//20)


def psk_encode(bitstream, samples=20):
    signal = []
    t = np.linspace(0, np.pi*2, samples)
    for b in bitstream:
        if b == 1:
            signal.extend(np.sin(t))
        else:
            signal.extend(np.sin(t + np.pi)) # 180° phase shift
    return list(signal)

def psk_decode(encoded):
    # basic method: detect sign
    chunk = len(encoded)//len(encoded)
    return ['?'] * (len(encoded)//20)



ENCODINGS = {
    "Manchester": (manchester_encode, manchester_decode),
    "MLT-3": (mlt3_encode, mlt3_decode),
    "NRZ": (nrz_encode, nrz_decode),
    "NRZI": (nrzi_encode, nrzi_decode),
    "AMI": (ami_encode, ami_decode),
    "4B/5B": (fourb_fiveb_encode, fourb_fiveb_decode),
    "RZ Bipolar": (bipolar_rz_encode, bipolar_rz_decode),
    "Differential Manchester": (diff_manchester_encode, diff_manchester_decode),
    "ASK": (ask_encode, ask_decode),
    "FSK": (fsk_encode, fsk_decode),
    "PSK": (psk_encode, psk_decode),

}