import sys
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from encoding import ENCODINGS
import numpy as np
import random
import os

class EncoderAnimator(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bitstream Encoder/Decoder - PyQtGraph Step-by-Step Animation")
        self.resize(1200, 860)
        layout = QtWidgets.QVBoxLayout(self)

        # Input and controls
        input_layout = QtWidgets.QHBoxLayout()
        self.input_edit = QtWidgets.QLineEdit()
        self.input_edit.setPlaceholderText("Enter bitstream (0s and 1s)")
        input_layout.addWidget(self.input_edit)

        self.schemes = list(ENCODINGS.keys())
        self.scheme_combo = QtWidgets.QComboBox()
        self.scheme_combo.addItems(self.schemes)
        input_layout.addWidget(self.scheme_combo)

        self.animate_btn = QtWidgets.QPushButton("Animate")
        self.animate_btn.clicked.connect(self.start_animation)
        input_layout.addWidget(self.animate_btn)

        self.animate_noisy_btn = QtWidgets.QPushButton("Animate Noisy")
        self.animate_noisy_btn.clicked.connect(self.start_animation_noisy)
        self.animate_noisy_btn.setEnabled(False)
        input_layout.addWidget(self.animate_noisy_btn)

        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)
        input_layout.addWidget(self.pause_btn)

        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(16)
        self.speed_slider.setToolTip("Speed (lower = faster)")
        self.speed_slider.valueChanged.connect(self.adjust_speed)
        input_layout.addWidget(QtWidgets.QLabel("Speed:"))
        input_layout.addWidget(self.speed_slider)

        layout.addLayout(input_layout)

        # --- File Import/Export controls ---
        file_layout = QtWidgets.QHBoxLayout()
        self.load_bitstream_btn = QtWidgets.QPushButton("Load Bitstream")
        self.load_bitstream_btn.clicked.connect(self.load_bitstream)
        file_layout.addWidget(self.load_bitstream_btn)
        self.save_bitstream_btn = QtWidgets.QPushButton("Save Bitstream")
        self.save_bitstream_btn.clicked.connect(self.save_bitstream)
        file_layout.addWidget(self.save_bitstream_btn)
        self.save_results_btn = QtWidgets.QPushButton("Save Results")
        self.save_results_btn.clicked.connect(self.save_results)
        file_layout.addWidget(self.save_results_btn)
        layout.addLayout(file_layout)

        # --- Noise controls ---
        noise_layout = QtWidgets.QHBoxLayout()
        self.noise_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.noise_slider.setMinimum(0)
        self.noise_slider.setMaximum(100)
        self.noise_slider.setValue(0)
        self.noise_slider.setToolTip("Noise/Error Rate (%)")
        self.noise_label = QtWidgets.QLabel("Noise: 0%")
        self.noise_slider.valueChanged.connect(lambda v: self.noise_label.setText(f"Noise: {v}%"))
        noise_layout.addWidget(self.noise_label)
        noise_layout.addWidget(self.noise_slider)
        self.noise_btn = QtWidgets.QPushButton("Add Noise")
        self.noise_btn.clicked.connect(self.add_noise)
        noise_layout.addWidget(self.noise_btn)
        layout.addLayout(noise_layout)

        # --- Step-by-step controls ---
        step_layout = QtWidgets.QHBoxLayout()
        self.step_mode_label = QtWidgets.QLabel("Step-by-Step: (works for both clean and noisy)")
        step_layout.addWidget(self.step_mode_label)
        self.step_back_btn = QtWidgets.QPushButton("Step Back")
        self.step_back_btn.clicked.connect(self.step_back)
        self.step_back_btn.setEnabled(False)
        step_layout.addWidget(self.step_back_btn)
        self.step_forward_btn = QtWidgets.QPushButton("Step Forward")
        self.step_forward_btn.clicked.connect(self.step_forward)
        self.step_forward_btn.setEnabled(False)
        step_layout.addWidget(self.step_forward_btn)
        self.reset_step_btn = QtWidgets.QPushButton("Reset Animation")
        self.reset_step_btn.clicked.connect(self.reset_step)
        self.reset_step_btn.setEnabled(False)
        step_layout.addWidget(self.reset_step_btn)
        # Mode selection (clean/noisy)
        self.step_mode_combo = QtWidgets.QComboBox()
        self.step_mode_combo.addItems(["Clean", "Noisy"])
        self.step_mode_combo.setToolTip("Choose which signal to step through")
        step_layout.addWidget(QtWidgets.QLabel("Step Mode:"))
        step_layout.addWidget(self.step_mode_combo)
        layout.addLayout(step_layout)

        # Animated decoded bits display
        self.decoded_bits_label = QtWidgets.QLabel("Decoded Bits: ")
        font = QtGui.QFont("Consolas", 14)
        self.decoded_bits_label.setFont(font)
        layout.addWidget(self.decoded_bits_label)

        # PyQtGraph plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('k')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.animate_step)
        self.animation_speed = 16  # ms (default ~60fps)

        # State
        self.encoded_signal = np.array([])
        self.encoded_signal_noisy = np.array([])
        self.anim_curve = None
        self.anim_playhead = None
        self.anim_step = 0
        self.anim_frame = 0
        self.scheme = None
        self.x = np.array([])
        self.bg_curve = None
        self.decoded_bits = []
        self.decoded_curve = None
        self.animating = False
        self.paused = False
        self.use_noisy = False
        self.reference_decoded_bits = []
        self.step_mode = False
        self.step_position = 0
        self.step_decoded_bits = []
        self.step_scatter_x = []
        self.step_scatter_y = []
        self.step_signal = None
        self.step_x = None
        self.step_reference_decoded_bits = []

    # --- File I/O ---
    def load_bitstream(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Bitstream File", "", "Text Files (*.txt);;All Files (*)")
        if fn and os.path.exists(fn):
            with open(fn, "r") as f:
                data = f.read().strip().replace("\n", "").replace(" ", "")
                data = ''.join(c for c in data if c in '01')
                self.input_edit.setText(data)

    def save_bitstream(self):
        bitstream = self.input_edit.text().strip()
        if not bitstream:
            QtWidgets.QMessageBox.warning(self, "No Bitstream", "Nothing to save.")
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Bitstream", "", "Text Files (*.txt);;All Files (*)")
        if fn:
            with open(fn, "w") as f:
                f.write(bitstream)

    def save_results(self):
        bitstream = self.input_edit.text().strip()
        scheme = self.scheme_combo.currentText()
        encoded = ' '.join(str(x) for x in self.encoded_signal.tolist()) if self.encoded_signal.size > 0 else ""
        noisy = ' '.join(str(x) for x in self.encoded_signal_noisy.tolist()) if self.encoded_signal_noisy.size > 0 else ""
        decoded = ''.join(str(x) for x in self.decoded_bits) if self.decoded_bits else ""
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results", "", "Text Files (*.txt);;All Files (*)")
        if fn:
            with open(fn, "w") as f:
                f.write(f"Bitstream: {bitstream}\n")
                f.write(f"Scheme: {scheme}\n")
                f.write(f"Encoded Signal: {encoded}\n")
                if noisy:
                    f.write(f"Noisy Encoded Signal: {noisy}\n")
                f.write(f"Decoded Bits: {decoded}\n")

    # --- Animation logic, noise, etc. ---
    def adjust_speed(self, value):
        self.animation_speed = max(1, value)
        if self.timer.isActive():
            self.timer.setInterval(self.animation_speed)

    def start_animation(self):
        self.step_mode = False
        self.use_noisy = False
        self._animate_common()
        self.enable_step_controls(False)

    def start_animation_noisy(self):
        self.step_mode = False
        self.use_noisy = True
        self._animate_common()
        self.enable_step_controls(False)

    def _animate_common(self):
        bitstream_str = self.input_edit.text().strip()
        if not all(c in '01' for c in bitstream_str) or not bitstream_str:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Enter a bitstream of 0s and 1s.")
            return

        bitstream = [int(b) for b in bitstream_str]
        self.scheme = self.scheme_combo.currentText()
        encode_func = ENCODINGS[self.scheme][0]
        decode_func = ENCODINGS[self.scheme][1]
        self.encoded_signal = np.array(encode_func(bitstream))

        if self.scheme == "Manchester":
            self.x = np.array([i/2 for i in range(len(self.encoded_signal))])
        else:
            self.x = np.arange(len(self.encoded_signal))

        # Always compute reference decoded bits for error highlight
        self.reference_decoded_bits = decode_func(self.encoded_signal.tolist())

        # Choose which signal to animate: clean or noisy
        if self.use_noisy and len(self.encoded_signal_noisy) == len(self.encoded_signal):
            signal = self.encoded_signal_noisy
        else:
            signal = self.encoded_signal

        self.decoded_bits = decode_func(list(signal))
        self.decoded_curve_x = np.arange(len(self.decoded_bits))
        self.decoded_curve_y = [None] * len(self.decoded_bits)

        # Plot faint background signal ONCE
        self.plot_widget.clear()
        self.bg_curve = pg.PlotCurveItem(self.x, signal, pen=pg.mkPen((100, 100, 255, 60), width=2, style=QtCore.Qt.DashLine))
        self.plot_widget.addItem(self.bg_curve)

        # Main animated curve (just the trail)
        self.anim_curve = pg.PlotCurveItem([], [], pen=pg.mkPen('c', width=3))
        self.plot_widget.addItem(self.anim_curve)

        # Decoded bits curve
        self.decoded_curve = pg.ScatterPlotItem([], [], symbol='o', size=16, brush=pg.mkBrush(0, 255, 0, 180))
        self.plot_widget.addItem(self.decoded_curve)

        # Animated playhead
        self.anim_playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=4, style=QtCore.Qt.DashLine))
        self.plot_widget.addItem(self.anim_playhead)

        # Set axis ranges
        if self.scheme == "Manchester":
            self.plot_widget.setYRange(-0.5, 1.5)
            self.plot_widget.setXRange(0, max(self.x))
        elif self.scheme == "MLT-3" or self.scheme == "AMI":
            self.plot_widget.setYRange(-1.5, 1.5)
            self.plot_widget.setXRange(0, max(self.x))
        else:
            self.plot_widget.setYRange(-0.5, 1.5)
            self.plot_widget.setXRange(0, max(self.x))

        self.anim_frame = 0
        self.animating = True
        self.paused = False
        self.pause_btn.setChecked(False)
        self.pause_btn.setEnabled(True)
        self.decoded_bits_label.setText("Decoded Bits: ")
        self.timer.start(self.animation_speed)

    def add_noise(self):
        if self.encoded_signal.size == 0:
            QtWidgets.QMessageBox.warning(self, "No Encoded Signal", "Encode a bitstream first (click Animate).")
            return

        # Use current encoding scheme to decide levels to flip
        noise_percent = self.noise_slider.value()
        n = len(self.encoded_signal)
        n_errors = int((noise_percent/100.0) * n)
        error_indices = random.sample(range(n), n_errors) if n_errors > 0 else []
        signal_noisy = self.encoded_signal.copy()

        scheme = self.scheme_combo.currentText()
        for idx in error_indices:
            if scheme == "Manchester" or scheme in ["NRZ", "NRZI", "4B/5B"]:
                # Flip 0 <-> 1
                if signal_noisy[idx] in [0, 1]:
                    signal_noisy[idx] = 1 - signal_noisy[idx]
            elif scheme == "MLT-3":
                levels = [-1, 0, 1]
                current = signal_noisy[idx]
                choices = [l for l in levels if l != current]
                signal_noisy[idx] = random.choice(choices)
            elif scheme == "AMI":
                # Flip sign if not zero
                if signal_noisy[idx] != 0:
                    signal_noisy[idx] = -signal_noisy[idx]
        self.encoded_signal_noisy = signal_noisy
        self.animate_noisy_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "Noise Added", f"Applied noise to {n_errors} out of {n} encoded samples ({noise_percent}%).\nClick 'Animate Noisy' to view.")
        self.enable_step_controls(True)

    # --- Step-by-step mode ---
    def enable_step_controls(self, enable):
        self.step_forward_btn.setEnabled(enable)
        self.step_back_btn.setEnabled(enable)
        self.reset_step_btn.setEnabled(enable)
        self.step_mode_combo.setEnabled(enable)

    def reset_step(self):
        self.step_mode = True
        self.step_position = 0
        self.setup_step_mode()
        self.draw_step_frame()

    def step_forward(self):
        if not self.step_mode:
            self.reset_step()
            return
        if self.step_position < len(self.step_decoded_bits):
            self.step_position += 1
            self.draw_step_frame()

    def step_back(self):
        if not self.step_mode:
            self.reset_step()
            return
        if self.step_position > 0:
            self.step_position -= 1
            self.draw_step_frame()

    def setup_step_mode(self):
        # Choose which signal to step through
        mode = self.step_mode_combo.currentText()
        if mode == "Noisy" and self.encoded_signal_noisy.size == self.encoded_signal.size and self.encoded_signal.size > 0:
            signal = self.encoded_signal_noisy
            decode_func = ENCODINGS[self.scheme][1]
            self.step_decoded_bits = decode_func(list(signal))
            self.step_reference_decoded_bits = ENCODINGS[self.scheme][1](self.encoded_signal.tolist())
            self.step_signal = signal
            self.step_x = self.x
        else:
            signal = self.encoded_signal
            decode_func = ENCODINGS[self.scheme][1]
            self.step_decoded_bits = decode_func(list(signal))
            self.step_reference_decoded_bits = self.step_decoded_bits
            self.step_signal = signal
            self.step_x = self.x

        self.plot_widget.clear()
        self.bg_curve = pg.PlotCurveItem(self.step_x, self.step_signal, pen=pg.mkPen((100, 100, 255, 60), width=2, style=QtCore.Qt.DashLine))
        self.plot_widget.addItem(self.bg_curve)
        self.anim_curve = pg.PlotCurveItem([], [], pen=pg.mkPen('c', width=3))
        self.plot_widget.addItem(self.anim_curve)
        self.decoded_curve = pg.ScatterPlotItem([], [], symbol='o', size=16, brush=pg.mkBrush(0, 255, 0, 180))
        self.plot_widget.addItem(self.decoded_curve)
        self.anim_playhead = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=4, style=QtCore.Qt.DashLine))
        self.plot_widget.addItem(self.anim_playhead)
        if self.scheme == "Manchester":
            self.plot_widget.setYRange(-0.5, 1.5)
            self.plot_widget.setXRange(0, max(self.step_x))
        elif self.scheme == "MLT-3" or self.scheme == "AMI":
            self.plot_widget.setYRange(-1.5, 1.5)
            self.plot_widget.setXRange(0, max(self.step_x))
        else:
            self.plot_widget.setYRange(-0.5, 1.5)
            self.plot_widget.setXRange(0, max(self.step_x))

    def draw_step_frame(self):
        idx = max(0, self.step_position)
        trail_start = max(0, idx-50+1)
        if idx >= len(self.step_x):
            x_playhead = self.step_x[-1]
            y_playhead = self.step_signal[-1]
        else:
            x_playhead = self.step_x[idx]
            y_playhead = self.step_signal[idx]
        trail_x = np.append(self.step_x[trail_start:idx+1], x_playhead)
        trail_y = np.append(self.step_signal[trail_start:idx+1], y_playhead)
        self.anim_curve.setData(trail_x, trail_y)
        self.anim_playhead.setPos(x_playhead)
        self.plot_widget.plot([x_playhead], [y_playhead],
                              pen=None,
                              symbol='o',
                              symbolBrush=pg.mkBrush(255, 255, 0, 255),
                              symbolPen=None,
                              symbolSize=28,
                              clear=False)
        # Only numeric decoded bits, colored by error/correct; '?' shown as red at y=1
        scatter_x = []
        scatter_y = []
        brushes = []
        for i, (xi, yi) in enumerate(zip(range(idx), self.step_decoded_bits[:idx])):
            if yi == '?' or (self.step_decoded_bits[i] != self.step_reference_decoded_bits[i]):
                scatter_x.append(xi)
                scatter_y.append(1)
                brushes.append(pg.mkBrush(255, 0, 0, 220))  # red for error/'?'
            else:
                try:
                    yval = float(yi)
                    scatter_x.append(xi)
                    scatter_y.append(yval)
                    brushes.append(pg.mkBrush(0, 255, 0, 180))  # green for correct
                except Exception:
                    pass
        # Current bit as yellow
        if idx < len(self.step_decoded_bits):
            if self.step_decoded_bits[idx] == '?':
                scatter_x.append(idx)
                scatter_y.append(1)
                brushes.append(pg.mkBrush(255, 255, 0, 220))  # yellow for current unknown bit
            else:
                try:
                    yval = float(self.step_decoded_bits[idx])
                    scatter_x.append(idx)
                    scatter_y.append(yval)
                    brushes.append(pg.mkBrush(255, 255, 0, 220))
                except Exception:
                    pass
        self.decoded_curve.setData(scatter_x, scatter_y, brush=brushes)
        # Highlight current decoded bit in label
        label_str = "Decoded Bits: "
        mismatches = 0
        for i in range(idx):
            if self.step_decoded_bits[i] != self.step_reference_decoded_bits[i]:
                label_str += f'<span style="color:red">{self.step_decoded_bits[i]}</span>'
                mismatches += 1
            else:
                label_str += str(self.step_decoded_bits[i])
        if idx < len(self.step_decoded_bits):
            if self.step_decoded_bits[idx] != self.step_reference_decoded_bits[idx]:
                label_str += f'<span style="background-color:yellow;color:red">{self.step_decoded_bits[idx]}</span>'
                mismatches += 1
            else:
                label_str += f'<span style="background-color:yellow">{self.step_decoded_bits[idx]}</span>'
        if idx > 0:
            label_str += f'   <span style="color:red">(Errors: {mismatches})</span>'
        self.decoded_bits_label.setText(label_str)
        self.decoded_bits_label.setTextFormat(QtCore.Qt.RichText)

    def animate_step(self):
        if self.paused or not self.animating:
            return

        frames_per_segment = 8  # Higher = slower but smoother
        if self.use_noisy and len(self.encoded_signal_noisy) == len(self.encoded_signal):
            signal = self.encoded_signal_noisy
            decoded_bits = self.decoded_bits
        else:
            signal = self.encoded_signal
            decoded_bits = self.reference_decoded_bits

        total_points = len(self.x)
        total_frames = (total_points-1) * frames_per_segment

        if self.anim_frame > total_frames:
            self.timer.stop()
            self.animating = False
            self.pause_btn.setEnabled(False)
            self.enable_step_controls(True)
            return

        # Interpolate playhead position (for smooth movement)
        playhead_pos = self.anim_frame / frames_per_segment
        idx = int(playhead_pos)
        if idx >= total_points - 1:
            x_playhead = self.x[-1]
            y_playhead = signal[-1]
        else:
            x0, x1 = self.x[idx], self.x[idx+1]
            y0, y1 = signal[idx], signal[idx+1]
            frac = playhead_pos - idx
            x_playhead = x0 + frac * (x1 - x0)
            y_playhead = y0 + frac * (y1 - y0)

        # Plot only a moving window as trail
        window = 50  # Trail length in points
        trail_start = max(0, idx-window+1)
        trail_x = np.append(self.x[trail_start:idx+1], x_playhead)
        trail_y = np.append(signal[trail_start:idx+1], y_playhead)
        self.anim_curve.setData(trail_x, trail_y)

        # Move playhead
        self.anim_playhead.setPos(x_playhead)
        self.plot_widget.plot([x_playhead], [y_playhead],
                              pen=None,
                              symbol='o',
                              symbolBrush=pg.mkBrush(255, 255, 0, 255),
                              symbolPen=None,
                              symbolSize=28,
                              clear=False)

        # Decoded bits as playhead passes (show errors in red, correct in green, '?' as red at y=1, current in yellow)
        bits_revealed = int(np.clip(playhead_pos, 0, len(decoded_bits)))
        scatter_x = []
        scatter_y = []
        brushes = []
        for i, (xi, yi) in enumerate(zip(self.decoded_curve_x[:bits_revealed], decoded_bits[:bits_revealed])):
            if yi == '?' or (self.use_noisy and len(self.reference_decoded_bits) == len(decoded_bits) and yi != self.reference_decoded_bits[i]):
                scatter_x.append(xi)
                scatter_y.append(1)
                brushes.append(pg.mkBrush(255, 0, 0, 220))  # red for error/'?'
            else:
                try:
                    yval = float(yi)
                    scatter_x.append(xi)
                    scatter_y.append(yval)
                    brushes.append(pg.mkBrush(0, 255, 0, 180))  # green for correct
                except Exception:
                    pass
        # Show current bit (playhead) as yellow, if exists
        if bits_revealed < len(decoded_bits):
            if decoded_bits[bits_revealed] == '?':
                scatter_x.append(self.decoded_curve_x[bits_revealed])
                scatter_y.append(1)
                brushes.append(pg.mkBrush(255, 255, 0, 220))
            else:
                try:
                    yval = float(decoded_bits[bits_revealed])
                    scatter_x.append(self.decoded_curve_x[bits_revealed])
                    scatter_y.append(yval)
                    brushes.append(pg.mkBrush(255, 255, 0, 220))  # yellow for current bit
                except Exception:
                    pass
        self.decoded_curve.setData(scatter_x, scatter_y, brush=brushes)

        # Error highlighting for noisy decoding:
        label_str = "Decoded Bits: "
        if self.use_noisy and len(self.reference_decoded_bits) == len(decoded_bits):
            mismatches = 0
            for i in range(bits_revealed):
                if decoded_bits[i] != self.reference_decoded_bits[i]:
                    label_str += f'<span style="color:red">{decoded_bits[i]}</span>'
                    mismatches += 1
                else:
                    label_str += str(decoded_bits[i])
            # Mark current bit being decoded
            if bits_revealed < len(decoded_bits):
                if decoded_bits[bits_revealed] != self.reference_decoded_bits[bits_revealed]:
                    label_str += f'<span style="background-color:yellow;color:red">{decoded_bits[bits_revealed]}</span>'
                    mismatches += 1
                else:
                    label_str += f'<span style="background-color:yellow">{decoded_bits[bits_revealed]}</span>'
            if bits_revealed > 0:
                label_str += f'   <span style="color:red">(Errors: {mismatches})</span>'
            self.decoded_bits_label.setText(label_str)
            self.decoded_bits_label.setTextFormat(QtCore.Qt.RichText)
        else:
            for i in range(bits_revealed):
                label_str += str(decoded_bits[i])
            if bits_revealed < len(decoded_bits):
                label_str += f'<span style="background-color:yellow">{decoded_bits[bits_revealed]}</span>'
            self.decoded_bits_label.setText(label_str)
            self.decoded_bits_label.setTextFormat(QtCore.Qt.RichText)

        self.anim_frame += 1

    def toggle_pause(self):
        if self.paused:
            self.paused = False
            self.pause_btn.setText("Pause")
            if self.animating:
                self.timer.start(self.animation_speed)
        else:
            self.paused = True
            self.pause_btn.setText("Resume")
            self.timer.stop()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = EncoderAnimator()
    win.show()
    sys.exit(app.exec_())