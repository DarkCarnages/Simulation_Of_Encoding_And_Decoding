import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests, time, random, base64
from io import BytesIO



st.set_page_config(
    page_title="Digital Communication Simulator",
    layout="wide"
)



def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None


wave_animation = load_lottie("https://assets6.lottiefiles.com/packages/lf20_kyu7xb1v.json")


st.markdown("""
<style>
@keyframes fadeIn {
    from {opacity:0;}
    to {opacity:1;}
}

.block-container {
    animation: fadeIn 1.2s ease-in-out;
    background: #0e1117 !important;
    padding: 20px;
    border-radius: 14px;
}

[data-testid="stSidebar"] {
    background: #141621;
    border-right: 2px solid #00eaff50;
}

.sidebar-title {
    color: cyan;
    font-size: 22px;
    font-weight: bold;
    text-align:center;
}

h1 {
    color: cyan;
    text-align: center;
    font-weight: 800;
    animation: glow 2.5s infinite alternate;
}
@keyframes glow {
    from { text-shadow: 0 0 10px cyan; }
    to { text-shadow: 0 0 25px #00fff2; }
}

hr {
    border: 1px solid #00eaff50;
}

.stButton>button {
    background: #005bff;
    color: white;
    padding: 12px;
    border-radius: 8px;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.07);
    background: #0039b3;
    box-shadow: 0px 0px 12px cyan;
}
</style>
""", unsafe_allow_html=True)



col1, col2, col3 = st.columns([1,2,1])
with col2:
    st_lottie(wave_animation, height=160)

st.markdown("<h1>📡 Digital Communication Simulator</h1><hr>", unsafe_allow_html=True)



st.sidebar.markdown("<div class='sidebar-title'>🧭 Navigation</div>", unsafe_allow_html=True)
menu = st.sidebar.radio(
    "Choose Category:",
    [
        "Encoding & Modulation",
        "Channel Processing",
        "Analysis Tools",
        "Transmission Systems"
    ]
)



if "bits" not in st.session_state: st.session_state.bits=None
if "encoding_name" not in st.session_state: st.session_state.encoding_name=None
if "encoded_clean" not in st.session_state: st.session_state.encoded_clean=None
if "encoded_noisy" not in st.session_state: st.session_state.encoded_noisy=None



def awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    
    if len(signal) == 0:
        return signal
    snr_linear = 10 ** (snr_db / 10.0)
    power_signal = np.mean(np.abs(signal) ** 2)
    noise_power = power_signal / snr_linear if snr_linear > 0 else power_signal
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise


def calc_ber(original_bits, decoded_bits):
  
    n = min(len(original_bits), len(decoded_bits))
    if n == 0:
        return 0, 0.0
    errors = sum(1 for i in range(n) if original_bits[i] != decoded_bits[i])
    ber = errors / n
    return errors, ber


def bits_to_string(bits):
    return ''.join(str(b) for b in bits)


def bits_to_spaced_string(bits):
    return ' '.join(str(b) for b in bits)


def download_button_binary(data: bytes, filename: str, label: str):
    
    b64 = base64.b64encode(data).decode()
    href = f'<a download="{filename}" href="data:file/txt;base64,{b64}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)



def encode_nrz(bits):
    return np.array(bits, dtype=float)


def decode_nrz(signal, bit_len):
    # Threshold at 0.5
    sig = np.array(signal[:bit_len])
    return [1 if s >= 0.5 else 0 for s in sig]


def encode_nrzi(bits):
    out = []
    level = 0
    for b in bits:
        if b == 1:
            level = 1 - level
        out.append(level)
    return np.array(out, dtype=float)


def decode_nrzi(signal, bit_len):
    # First threshold to 0/1
    sig = np.array(signal[:bit_len])
    sig = np.where(sig >= 0.5, 1, 0)
    decoded = []
    prev = 0
    for s in sig:
        decoded.append(0 if s == prev else 1)
        prev = s
    return decoded


def encode_manchester(bits):
    encoded = []
    for b in bits:
        if b == 1:
            encoded.extend([1, 0])
        else:
            encoded.extend([0, 1])
    return np.array(encoded, dtype=float)


def decode_manchester(signal, bit_len):
    sig = np.array(signal[:2 * bit_len])
    sig = np.where(sig >= 0.5, 1, 0)
    decoded = []
    for i in range(0, len(sig), 2):
        pair = sig[i:i + 2]
        if len(pair) < 2:
            break
        if (pair == np.array([1, 0])).all():
            decoded.append(1)
        elif (pair == np.array([0, 1])).all():
            decoded.append(0)
        else:
           
            decoded.append(int(pair[0]))
    return decoded[:bit_len]


def encode_diff_manchester(bits):
    out = []
    last = 1
    for b in bits:
        if b == 1:
        
            out.extend([last, -last])
            last = -last
        else:
           
            last = -last
            out.extend([last, -last])
    return np.array(out, dtype=float)


def decode_diff_manchester(signal, bit_len):
   
    sig = np.array(signal[:2 * bit_len])
    sig = np.where(sig >= 0, 1, -1)  
    decoded = []
    num_pairs = len(sig) // 2
    if num_pairs == 0:
        return []
    decoded.append(0) 
    for i in range(1, num_pairs):
        prev_second = sig[2 * i - 1]
        curr_first = sig[2 * i]
        if prev_second != curr_first:
            decoded.append(0)
        else:
            decoded.append(1)
    return decoded[:bit_len]


def encode_mlt3(bits):
    levels = [0, 1, 0, -1]
    idx = 0
    result = []
    for b in bits:
        if b == 1:
            idx = (idx + 1) % 4
        result.append(levels[idx])
    return np.array(result, dtype=float)


def decode_mlt3(signal, bit_len):
    sig = np.array(signal[:bit_len])
 
    levels = np.array([-1, 0, 1], dtype=float)
    quantized = []
    for s in sig:
        idx = np.argmin(np.abs(levels - s))
        quantized.append(levels[idx])
    decoded = []
    prev = quantized[0] if len(quantized) > 0 else 0
    # First bit: assume 0
    decoded.append(0)
    for lvl in quantized[1:]:
        if lvl == prev:
            decoded.append(0)
        else:
            decoded.append(1)
        prev = lvl
    return decoded[:bit_len]


FOURB5B_TABLE = {
    '0000': '11110', '0001': '01001', '0010': '10100', '0011': '10101',
    '0100': '01010', '0101': '01011', '0110': '01110', '0111': '01111',
    '1000': '10010', '1001': '10011', '1010': '10110', '1011': '10111',
    '1100': '11010', '1101': '11011', '1110': '11100', '1111': '11101',
}
FIVEB4B_TABLE = {v: k for k, v in FOURB5B_TABLE.items()}


def encode_4b5b(bits):
    b_str = ''.join(str(x) for x in bits)
    while len(b_str) % 4 != 0:
        b_str += '0'
    encoded = ""
    for i in range(0, len(b_str), 4):
        chunk = b_str[i:i + 4]
        encoded += FOURB5B_TABLE.get(chunk, "00000")
    return np.array([int(x) for x in encoded], dtype=float)


def decode_4b5b(signal, bit_len):
    
    sig = np.where(np.array(signal) >= 0.5, 1, 0)
    
    n5 = len(sig) // 5
    sig = sig[:n5 * 5]
    decoded_bits = []
    for i in range(0, len(sig), 5):
        code = ''.join(str(int(x)) for x in sig[i:i + 5])
        nibble = FIVEB4B_TABLE.get(code, "0000")
        decoded_bits.extend(int(b) for b in nibble)
    return decoded_bits[:bit_len]


def encode_bipolar_rz(bits):
    out = []
    last = -1
    for b in bits:
        if b == 1:
            last *= -1
            out.extend([last, 0])
        else:
            out.extend([0, 0])
    return np.array(out, dtype=float)


def decode_bipolar_rz(signal, bit_len):
    sig = np.array(signal[:2 * bit_len])
    decoded = []
    for i in range(0, len(sig), 2):
        first_half = sig[i]
        decoded.append(1 if np.abs(first_half) > 0.3 else 0)
    return decoded[:bit_len]




def encode_ask(bits, samples_per_bit=50):
    t = np.linspace(0, 1, samples_per_bit)
    waveform = []
    for b in bits:
        if b == 1:
            waveform.extend(np.sin(2 * np.pi * 5 * t))  
        else:
            waveform.extend(np.zeros_like(t))
    return np.array(waveform, dtype=float)


def decode_ask(signal, bit_len):
    sig = np.array(signal)
    if bit_len == 0:
        return []
    spb = len(sig) // bit_len
    decoded = []
    for i in range(bit_len):
        chunk = sig[i * spb:(i + 1) * spb]
        avg_amp = np.mean(np.abs(chunk))
        decoded.append(1 if avg_amp > 0.3 else 0)
    return decoded


def encode_fsk(bits, samples_per_bit=50, f0=3, f1=8):
    t = np.linspace(0, 1, samples_per_bit)
    waveform = []
    for b in bits:
        f = f1 if b == 1 else f0
        waveform.extend(np.sin(2 * np.pi * f * t))
    return np.array(waveform, dtype=float)


def decode_fsk(signal, bit_len, f0=3, f1=8):
    sig = np.array(signal)
    if bit_len == 0:
        return []
    spb = len(sig) // bit_len
    t = np.linspace(0, 1, spb)
    ref0 = np.sin(2 * np.pi * f0 * t)
    ref1 = np.sin(2 * np.pi * f1 * t)
    decoded = []
    for i in range(bit_len):
        chunk = sig[i * spb:(i + 1) * spb]
        c0 = np.dot(chunk, ref0)
        c1 = np.dot(chunk, ref1)
        decoded.append(1 if c1 >= c0 else 0)
    return decoded


def encode_bpsk(bits, samples_per_bit=50):
    t = np.linspace(0, 2 * np.pi, samples_per_bit)
    waveform = []
    for b in bits:
        if b == 1:
            waveform.extend(np.sin(t))
        else:
            waveform.extend(np.sin(t + np.pi))
    return np.array(waveform, dtype=float)


def decode_bpsk(signal, bit_len):
    sig = np.array(signal)
    if bit_len == 0:
        return []
    spb = len(sig) // bit_len
    t = np.linspace(0, 2 * np.pi, spb)
    ref = np.sin(t)
    decoded = []
    for i in range(bit_len):
        chunk = sig[i * spb:(i + 1) * spb]
        corr = np.dot(chunk, ref)
        decoded.append(1 if corr >= 0 else 0)
    return decoded


def encode_qpsk(bits, samples_per_bit=50):
    t = np.linspace(0, 2 * np.pi, samples_per_bit)
    phase_map = {"00": 0, "01": np.pi / 2, "11": np.pi, "10": 3 * np.pi / 2}
    local_bits = bits[:]
    if len(local_bits) % 2 != 0:
        local_bits.append(0)
    waveform = []
    for i in range(0, len(local_bits), 2):
        symbol = f"{local_bits[i]}{local_bits[i+1]}"
        phase = phase_map[symbol]
        waveform.extend(np.sin(t + phase))
    return np.array(waveform, dtype=float)


def decode_qpsk(signal, bit_len, samples_per_bit=50):
    sig = np.array(signal)
    if bit_len == 0:
        return []
    n_symbols = (bit_len + 1) // 2
    spb = len(sig) // n_symbols
    t = np.linspace(0, 2 * np.pi, spb)
    phase_map = {"00": 0, "01": np.pi / 2, "11": np.pi, "10": 3 * np.pi / 2}
    refs = {sym: np.sin(t + ph) for sym, ph in phase_map.items()}
    decoded_bits = []
    for i in range(n_symbols):
        chunk = sig[i * spb:(i + 1) * spb]
        best_sym = None
        best_corr = -1e9
        for sym, ref in refs.items():
            corr = np.dot(chunk, ref)
            if corr > best_corr:
                best_corr = corr
                best_sym = sym
        decoded_bits.extend(int(b) for b in best_sym)
    return decoded_bits[:bit_len]



ENCODING_SCHEMES = {
    "NRZ": {
        "category": "line",
        "encode": encode_nrz,
        "decode": decode_nrz
    },
    "NRZI": {
        "category": "line",
        "encode": encode_nrzi,
        "decode": decode_nrzi
    },
    "Manchester": {
        "category": "line",
        "encode": encode_manchester,
        "decode": decode_manchester
    },
    "Differential Manchester": {
        "category": "line",
        "encode": encode_diff_manchester,
        "decode": decode_diff_manchester
    },
    "MLT-3": {
        "category": "line",
        "encode": encode_mlt3,
        "decode": decode_mlt3
    },
    "4B/5B": {
        "category": "line",
        "encode": encode_4b5b,
        "decode": decode_4b5b
    },
    "Bipolar RZ": {
        "category": "line",
        "encode": encode_bipolar_rz,
        "decode": decode_bipolar_rz
    },
    "ASK": {
        "category": "modulation",
        "encode": encode_ask,
        "decode": decode_ask
    },
    "FSK": {
        "category": "modulation",
        "encode": encode_fsk,
        "decode": decode_fsk
    },
    "BPSK": {
        "category": "modulation",
        "encode": encode_bpsk,
        "decode": decode_bpsk
    },
    "QPSK": {
        "category": "modulation",
        "encode": encode_qpsk,
        "decode": decode_qpsk
    }
}


def decode_dispatch(scheme_name, signal, original_bits):
    
    scheme = ENCODING_SCHEMES[scheme_name]
    bit_len = len(original_bits)
    if scheme_name == "FSK":
        return scheme["decode"](signal, bit_len, f0=3, f1=8)
    elif scheme_name == "QPSK":
        return scheme["decode"](signal, bit_len, samples_per_bit=50)
    else:
        return scheme["decode"](signal, bit_len)
    

def xor_bits(a, b):
    
    return ''.join('0' if a[i] == b[i] else '1' for i in range(len(a)))

def crc_generate(payload_bits, generator_bits):
   
    payload = ''.join(str(b) for b in payload_bits)
    generator = ''.join(str(b) for b in generator_bits)

   
    appended = payload + '0' * (len(generator) - 1)
    working = appended

    for i in range(len(payload)):
        if working[i] == '1':
            
            working = (
                working[:i] +
                xor_bits(working[i:i+len(generator)], generator) +
                working[i+len(generator):]
            )
    
    
    remainder = working[-(len(generator)-1):]
    return remainder, appended, working  

def crc_check(received_bits, generator_bits):
    
    generator = ''.join(str(b) for b in generator_bits)
    working = ''.join(str(b) for b in received_bits)

    for i in range(len(received_bits) - (len(generator) - 1)):
        if working[i] == '1':
            working = (
                working[:i] +
                xor_bits(working[i:i+len(generator)], generator) +
                working[i+len(generator):]
            )
    
    remainder = working[-(len(generator)-1):]
    return remainder == '0' * (len(generator)-1), remainder




if menu == "Encoding & Modulation":
    tab1, tab2 = st.tabs(["Line Coding", "Digital Modulation"])

    # ---- LINE CODING TAB ----
    with tab1:
        st.subheader("Line Coding Schemes")
        st.write("NRZ, NRZI, Manchester, Differential Manchester, 4B/5B, Bipolar RZ, MLT-3")

        bitstream = st.text_input("Enter Bitstream:", "1011001101")

        encoding_choice = st.selectbox("Choose Encoding Scheme:", ["NRZ","NRZI","Manchester","Differential Manchester","4B/5B","MLT-3","Bipolar RZ"])

        if st.button("Encode (Line Coding)"):
            bits=[int(b) for b in bitstream]
            st.session_state.bits=bits
            st.session_state.encoding_name=encoding_choice
            encoded = ENCODING_SCHEMES[encoding_choice]["encode"](bits)
            st.session_state.encoded_clean=encoded

            fig = plt.figure(figsize=(9,3))
            plt.step(range(len(encoded)), encoded, where='post')
            plt.title(f"{encoding_choice} Waveform")
            plt.grid(True)
            st.pyplot(fig)

    
    st.write("### 🎞 Live Encoding Simulation")
    
    speed_encode = st.slider("Animation Speed (ms per step)", 100, 2000, 600)
    
    # UI Buttons
    colA, colB, colC = st.columns(3)
    play_btn = colA.button("Play")
    pause_btn = colB.button("Pause")
    restart_btn = colC.button("Restart")
    

    if "encode_index" not in st.session_state:
        st.session_state.encode_index = 0
    
    if "animation_running" not in st.session_state:
        st.session_state.animation_running = False
    
    if "live_encoded" not in st.session_state:
        st.session_state.live_encoded = []
    
    # Button Logic
    if play_btn:
        st.session_state.animation_running = True
    
    if pause_btn:
        st.session_state.animation_running = False
    
    if restart_btn:
        st.session_state.encode_index = 0
        st.session_state.live_encoded = []
        st.session_state.animation_running = False
    
    
    waveform_box = st.empty()
    status_box = st.empty()
    result_box = st.empty()
    
    bits = [int(b) for b in bitstream if b in "01"]
    encode_fn = ENCODING_SCHEMES[encoding_choice]["encode"]
    
    
    if st.session_state.animation_running and st.session_state.encode_index < len(bits):
    
        bit = bits[st.session_state.encode_index]
    
       
        encoded_step = encode_fn([bit])
    
        
        st.session_state.live_encoded.extend(list(encoded_step))
    
     
        status_box.write(f"Encoding bit {st.session_state.encode_index+1}/{len(bits)} → `{bit}`")
        result_box.code("".join(str(b) for b in bits[:st.session_state.encode_index+1]))
    
      
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=st.session_state.live_encoded, mode="lines"))
        fig.update_layout(title=f"Live Encoding: {encoding_choice}", height=240)
        waveform_box.plotly_chart(fig, use_container_width=True)
    
       
        st.session_state.encode_index += 1
    
   
        time.sleep(speed_encode/1000)
    
    elif st.session_state.encode_index >= len(bits):
        st.success("Encoding Complete!")
    
       
        st.session_state.bits = bits
        st.session_state.encoding_name = encoding_choice
        st.session_state.encoded_clean = np.array(st.session_state.live_encoded, dtype=float)
    


    with tab2:
        st.subheader("Modulation Techniques")
        mod_choice = st.selectbox("Select Modulation Type:", ["ASK","FSK","BPSK","QPSK"])
        
        if st.button("Encode Signal (Modulation)"):
            bits = st.session_state.bits or [1,0,1,1,0]
            st.session_state.encoding_name = mod_choice
            encoded = ENCODING_SCHEMES[mod_choice]["encode"](bits)
            st.session_state.encoded_clean = encoded
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=encoded, mode="lines"))
            fig.update_layout(title=f"{mod_choice} Signal", height=350)
            st.plotly_chart(fig, use_container_width=True)




elif menu == "Channel Processing":
    st.subheader("Noise Injection and Receiver Decoding")

    if st.session_state.encoded_clean is None:
        st.warning("Encode a signal first.")
    else:
        snr=st.slider("SNR (dB)", -5,40,10)
        if st.button("Apply Noise + Decode"):
            noisy=awgn(st.session_state.encoded_clean,snr)
            st.session_state.encoded_noisy=noisy

            decoded = decode_dispatch(
                st.session_state.encoding_name,
                noisy,
                st.session_state.bits
            )

            st.write("### Comparison")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=st.session_state.encoded_clean,name="Clean"))
            fig.add_trace(go.Scatter(y=noisy,name="Noisy",opacity=0.5))
            fig.update_layout(height=300)
            st.plotly_chart(fig,use_container_width=True)

            st.write("Decoded Output: ", decoded)



elif menu == "Analysis Tools":
    tab1, tab2 = st.tabs(["BER Comparison", "📈 Spectrum FFT"])

    with tab1:
        st.subheader("Bit Error Rate")
        if st.session_state.encoded_noisy is None:
            st.warning("Run noisy decode first.")
        else:
            errors,ber = calc_ber(st.session_state.bits, decode_dispatch(st.session_state.encoding_name, st.session_state.encoded_noisy, st.session_state.bits))
            st.metric("Errors", errors)
            st.metric("BER", f"{ber:.6f}")

    with tab2:
        st.subheader("Frequency Spectrum via FFT")
        if st.session_state.encoded_clean is None:
            st.warning("Encode signal first.")
        else:
            signal=st.session_state.encoded_clean
            fft=np.abs(np.fft.rfft(signal))
            fig=go.Figure()
            fig.add_trace(go.Scatter(y=fft,mode="lines"))
            fig.update_layout(height=300)
            st.plotly_chart(fig,use_container_width=True)




elif menu == "Transmission Systems":
    tabA, tabB, tabC = st.tabs([
        "Live Packet Transfer",
        "CRC Frame Check",
        "Sliding Window Protocol"
    ])


    with tabA:
        st.subheader("Real-Time Animated Transmission")
        st.subheader("Step 6: Real-Time Packet-Based Transmission Simulation with Live Waveform")

        if st.session_state["encoded_clean"] is None or st.session_state["bits"] is None:
            st.warning("⚠ Please encode a bitstream first in Tab 1.")
        else:
        
            bits = st.session_state["bits"]
            scheme_name = st.session_state["encoding_name"]
            encoded_full = st.session_state["encoded_clean"]
    
            st.write("###Packet Structure")
    
            st.code("""
[Preamble: 10101010] 
[Start Flag: 01111110] 
[Header: 0001] 
[Payload: <User Bits>] 
[Parity Bit: XOR of payload]
[End Flag: 01111110]
""")
    
            sim_snr = st.slider("Transmission Noise Level (SNR dB)", -5, 40, 12)
            speed = st.slider("Transmission Speed (ms per step)", 200, 2000, 1200)
    
            if st.button("Begin Packet Transmission"):
            
                preamble   = [1,0,1,0,1,0,1,0]
                start_flag = [0,1,1,1,1,1,1,0]
                header     = [0,0,0,1]
                parity     = [sum(bits) % 2]
                end_flag   = [0,1,1,1,1,1,1,0]
    
                complete_packet = preamble + start_flag + header + bits + parity + end_flag
    
                st.write(" Final Packet to Transmit")
                st.code(bits_to_string(complete_packet))
    
                receiver_bits = []
                errors = 0
    
                
                progress_bar = st.progress(0)
                status_msg   = st.empty()
                waveform_global = st.empty()
                waveform_bit = st.empty()
                receiver_msg = st.empty()
                uncertainty_msg = st.empty()
    
                full_clean_buffer = []
                full_noisy_buffer = []
    
                for i, bit in enumerate(complete_packet):
                
                    
                    if i < len(encoded_full):
                        encoded_bit = encoded_full[i:i+1]
                    else:
                        encoded_bit = np.array([bit], dtype=float)
    
                   
                    noisy_bit = awgn(encoded_bit, sim_snr)
    
                    full_clean_buffer.extend(encoded_bit)
                    full_noisy_buffer.extend(noisy_bit)
    
                   
                    decoded = decode_dispatch(scheme_name, noisy_bit, [bit])
                    decoded = list(decoded) if decoded is not None else []
    
                
                    if not decoded:
                        rx_bit = 1 if np.mean(noisy_bit) > 0 else 0
                        uncertainty_msg.write("⚠ Receiver uncertainty detected — applying best guess...")
                    elif len(decoded) > 1:
                        rx_bit = 1 if decoded.count(1) > decoded.count(0) else 0
                    else:
                        rx_bit = decoded[0]
    
                    receiver_bits.append(rx_bit)
    
                    bit_status = "OK..." if rx_bit == bit else "ERROR ???"
                    if rx_bit != bit:
                        errors += 1
    
                
                    status_msg.write(
                        f"Transmitting bit {i+1}/{len(complete_packet)} → `{bit}` → Result: {bit_status}"
                    )
    
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(y=encoded_bit, mode="lines", name="Clean"))
                    fig1.add_trace(go.Scatter(y=noisy_bit, mode="lines", name="Noisy", opacity=0.5))
                    fig1.update_layout(
                        title=f"Bit {i+1} Waveform View",
                        height=230,
                        showlegend=False
                    )
                    waveform_bit.plotly_chart(fig1, use_container_width=True)
    
                   
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(y=full_clean_buffer, mode="lines", name="Clean"))
                    fig2.add_trace(go.Scatter(y=full_noisy_buffer, mode="lines", name="Noisy", opacity=0.5))
                    fig2.update_layout(
                        title="Live Scrolling Waveform (Full Transmission)",
                        height=260,
                        xaxis_title="Time →",
                        yaxis_title="Amplitude"
                    )
                    waveform_global.plotly_chart(fig2, use_container_width=True)
    
                    
                    receiver_msg.code(f"Receiver Output:  {''.join(str(x) for x in receiver_bits)}")
    
                    
                    progress_bar.progress((i+1)/len(complete_packet))
    
                   
                    time.sleep(speed/1000)
    
               
                st.success("Transmission Completed!")
                ber = errors / len(complete_packet)
    
                st.write("### Final Transmission Report")
                st.metric("Total Bit Errors", errors)
                st.metric("Final BER", f"{ber:.6f}")
    
                st.code(f"TX Packet: {bits_to_string(complete_packet)}\nRX Packet: {bits_to_string(receiver_bits)}")
    
                if errors == 0:
                    st.success("Packet Delivered Successfully — No retransmission required.")
                else:
                    st.error("Errors detected — Receiver would request retransmission (ARQ).")



    
    with tabB:
        st.subheader("CRC Error Detection")
        
        if st.session_state["bits"] is None:
            st.warning("Please enter and encode a bitstream first in Tab 1.")
        else:
    
            payload = st.session_state["bits"]
            generator_str = st.text_input("Enter Generator Polynomial Bits:", "1101")
    
            try:
                generator_bits = [int(b) for b in generator_str.strip() if b in "01"]
                if len(generator_bits) < 2:
                    raise ValueError
            except:
                st.error("Invalid polynomial. Only 0 and 1 allowed.")
                st.stop()
    
            st.write("###Step 1 — Append Zeros")
            zeros_to_append = len(generator_bits) - 1
            appended = ''.join(str(b) for b in payload) + ('0' * zeros_to_append)
            st.code(f"{''.join(str(b) for b in payload)} → {appended}")
    
            st.write("### Step 2 — XOR Division (Animated)")
    
            remainder, appended_full, xor_trace = crc_generate(payload, generator_bits)
    
           
            speed = st.slider("Animation Speed (ms/step)", 200, 1500, 900)
    
            output_box = st.empty()
    
            working = appended
            for i in range(len(payload)):
                if working[i] == '1':
                    step_str = working[:i] + xor_bits(working[i:i+len(generator_bits)], generator_str) + working[i+len(generator_bits):]
                else:
                    step_str = working  
    
                output_box.code(
                    f"Step {i+1}:\n"
                    f"Working:  {working}\n"
                    f"Divider:   {' ' * i}{generator_str if working[i] == '1' else ''}\n"
                    f"Result:    {step_str}",
                    language="text"
                )
                working = step_str
                time.sleep(speed / 1000)
    
            st.success(f"Remainder: {remainder}")
    
            final_frame = ''.join(str(b) for b in payload) + remainder
    
            st.write("### Final CRC Frame:")
            st.code(final_frame)
    
            st.write("### Simulated Transmission + Validation")
    
            snr_crc = st.slider("Channel SNR (dB)", -5, 40, 12)
    
            noisy_signal = awgn([int(b) for b in final_frame], snr_crc)
            rx_bits = [1 if v > 0 else 0 for v in noisy_signal]
    
            valid, rx_remainder = crc_check(rx_bits, generator_bits)
    
            st.code(
                f"Received: {''.join(str(b) for b in rx_bits)}\n"
                f"Remainder: {rx_remainder}"
            )
    
            if valid:
                st.success("CRC VALID: Frame accepted!")
            else:
                st.error("CRC FAILED: Receiver detects corruption.")
    
   
    with tabC:
        st.subheader("Sliding Window ARQ Simulation")
        if st.session_state["bits"] is None:
            st.warning("⚠ Please enter a bitstream in Tab 1.")
        else:
            bits = st.session_state["bits"]

            st.write("### Frame Creation from Bitstream")

            frame_size = st.slider("Bits per Frame", 2, 8, 4)
            window_size = st.slider("Sliding Window Size", 2, 5, 3)
            loss_prob = st.slider("Packet Loss Probability", 0.0, 0.7, 0.3, 0.1)
            step_delay = st.slider("Animation Speed (ms per step)", 200, 2000, 800)
            
            frames = []
            for i in range(0, len(bits), frame_size):
                frames.append(bits[i:i+frame_size])
            num_frames = len(frames)
            st.code("Frames:\n" + "\n".join(
                [f"Frame {i}: {bits_to_string(f)}" for i, f in enumerate(frames)]
            ))
            if st.button("Start Sliding Window Simulation"):
                st.write("---")
                status_box = st.empty()
                chart_box = st.empty()
                log_box = st.empty()


                statuses = ["pending"] * num_frames
                base = 0
                next_seq = 0
                step = 0
                max_steps = 100 
                log_lines = []

                def color_for_status(s):
                    return {
                        "pending": "lightgray",
                        "sent": "deepskyblue",
                        "acked": "lightgreen",
                        "retransmit": "red"
                    }.get(s, "lightgray")

                while base < num_frames and step < max_steps:
                    step += 1
                    while next_seq < num_frames and next_seq < base + window_size:
                        if statuses[next_seq] == "pending" or statuses[next_seq] == "retransmit":
                            lost = random.random() < loss_prob
                            if lost:
                                statuses[next_seq] = "retransmit"
                                log_lines.append(f"Step {step}: Frame {next_seq} LOST on channel.")
                            else:
                                statuses[next_seq] = "sent"
                                log_lines.append(f"Step {step}: Frame {next_seq} SENT successfully.")
                        next_seq += 1
                    moved = False
                    for f_idx in range(base, next_seq):
                        if statuses[f_idx] == "sent":
                            statuses[f_idx] = "acked"
                            log_lines.append(f"Step {step}: ACK received for Frame {f_idx}.")
                            moved = True
                    while base < num_frames and statuses[base] == "acked":
                        base += 1
                        moved = True
                    for f_idx in range(num_frames):
                        if statuses[f_idx] == "retransmit" and f_idx >= base:
                            next_seq = f_idx
                            log_lines.append(f"Step {step}: Retransmission scheduled from Frame {f_idx}.")
                            break
                    
                    x_vals = list(range(num_frames))
                    y_vals = [1] * num_frames
                    colors = [color_for_status(s) for s in statuses]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=x_vals,
                        y=y_vals,
                        marker_color=colors,
                        hovertext=[f"Frame {i}: {statuses[i]}" for i in range(num_frames)],
                        showlegend=False
                    ))
                    fig.add_vrect(
                        x0=base - 0.5,
                        x1=min(base + window_size - 0.5, num_frames - 0.5),
                        fillcolor="rgba(255,255,0,0.2)",
                        line_width=1,
                        annotation_text="Window",
                        annotation_position="top left"
                    )
                    fig.update_layout(
                        title=f"Sliding Window State (Step {step})",
                        xaxis_title="Frame Index",
                        yaxis_title="Status",
                        yaxis=dict(showticklabels=False),
                        height=320
                    )
                    chart_box.plotly_chart(fig, use_container_width=True)
                    status_box.write(
                        f"Base: {base}, Next Seq: {next_seq}, Window Size: {window_size}"
                    )
                    log_box.code("\n".join(log_lines[-8:]))  
                    time.sleep(step_delay/1000)
                if base >= num_frames:
                    st.success("All frames successfully transmitted and acknowledged.")
                else:
                    st.error("Simulation stopped before completing all transfers (step limit reached).")