# 📡 Digital Communication Encoding & Error Control Simulator

## 🚀 Project Overview

This project is an **interactive simulation platform for digital communication systems** designed to visually demonstrate how binary data is encoded, transmitted, checked for errors, and reliably delivered.

Instead of limiting learning to theory, this simulator lets users **see signals evolve**, **observe encoding behavior**, and **understand reliability mechanisms** through real-time interaction.

The project is structured in **two simulation levels**, allowing a gradual and intuitive learning experience.

---

## 🎯 What This Simulator Offers

✅ Interactive waveform visualization
✅ Multiple digital line encoding techniques
✅ Error detection using CRC
✅ Reliable transmission using Sliding Window ARQ
✅ Clear separation between basic and advanced concepts

This makes the simulator suitable for **learning, experimentation, and demonstrations**.

---

## 🧩 Encoding Techniques Implemented

### 🔹 Line Encoding (Simulation 1)

The following **line encoding techniques** are implemented and visualized:

* 🔸 NRZ
* 🔸 NRZ-L
* 🔸 NRZ-I
* 🔸 Manchester
* 🔸 4B/5B

These techniques highlight:

* Signal transitions
* Synchronization properties
* Encoding efficiency

---

## 🛡️ Error Detection & Reliability (Simulation 2)

Simulation 2 extends the system into a **complete communication pipeline**.

### 🔁 CRC – Cyclic Redundancy Check

* Detects corrupted frames using polynomial division
* Identifies valid vs invalid transmissions

### 📦 ARQ – Automatic Repeat reQuest

* Implements **Sliding Window protocol**
* Demonstrates acknowledgments, retransmissions, and timeouts
* Shows how reliability is achieved over noisy channels

---

## 🗂️ Project Structure

The project is organized into two clearly separated simulations:

```
├── Simulation 1
│   └── Line Encoding Simulation
│       ├── NRZ
│       ├── NRZ-L
│       ├── NRZ-I
│       ├── Manchester
│       └── 4B/5B
│
├── Simulation 2
│   └── Complete Communication Simulation
│       ├── Line Encoding
│       ├── CRC Error Detection
│       └── Sliding Window ARQ
```

---

## 🧠 Design Philosophy

* **Simulation 1** focuses purely on **line encoding**

  * No noise
  * No protocols
  * Clean waveform understanding

* **Simulation 2** represents **real-world communication**

  * Error detection
  * Retransmission
  * End-to-end reliability

This modular design keeps concepts **clear, scalable, and easy to extend**.

---

## ⚙️ How to Run the Project

### 🟢 Step 1: Create a Virtual Environment

```bash
python -m venv venv
```

### 🟢 Step 2: Activate the Environment

```bash
venv/Scripts/activate
```

### 🟢 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### 🟢 Step 4: Choose a Simulation

For encoding-only simulation:

```bash
cd Simulation\ 1
```

For full communication simulation:

```bash
cd Simulation\ 2
```

### 🟢 Step 5: Run the Application

```bash
streamlit run app.py
```

---

## 🧪 How to Use the Simulator

* Select or enter a binary input sequence
* Choose an encoding technique
* Observe the generated waveform
* In Simulation 2:

  * Monitor CRC validation
  * Watch ARQ retransmissions in action

---

## 🧰 Technologies Used

* 🐍 Python
* 🌐 Streamlit
* 📊 NumPy
* 📈 Matplotlib / Plotly
