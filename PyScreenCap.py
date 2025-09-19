# ==================================================
# PyScreenCap - Python Screen Recorder with GUI
#
# Features:
#  - Multi-monitor selection
#  - Audio recording (microphone or system/loopback)
#  - Audio level preview (even without recording)
#  - FPS slider (default 30) for video recording
#  - GUI with PySide6
#  - Final output is MP4 (temporary AVI/WAV are auto-deleted)
#
# On startup, required packages are auto-checked and
# installed in the background if missing:
#   PySide6, mss, opencv-python, numpy,
#   sounddevice, soundcard, moviepy, imageio-ffmpeg
# ==================================================

# ---------- auto-installer ----------
import importlib, subprocess, sys

# Map pip package -> importable module name
required = {
    "PySide6": "PySide6",
    "mss": "mss",
    "opencv-python": "cv2",        # cv2 is the module name
    "numpy": "numpy",
    "sounddevice": "sounddevice",
    "soundcard": "soundcard",
    "moviepy": "moviepy",
    "imageio-ffmpeg": "imageio_ffmpeg",  # provides ffmpeg binary for moviepy
}

for pip_name, module_name in required.items():
    try:
        importlib.import_module(module_name)
    except ImportError:
        print(f"[SETUP] Installing missing package: {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "--quiet"])

# ---------- regular imports ----------
import sys, os, time, datetime, threading, queue, wave
import numpy as np
import mss, cv2
import sounddevice as sd
import soundcard as sc
from moviepy.editor import VideoFileClip, AudioFileClip

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QProgressBar, QHBoxLayout, QVBoxLayout, QMessageBox, QCheckBox, QSlider
)

# ---------- utilities ----------
def ts() -> str:
    """Timestamp for filenames."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def desktop_path() -> str:
    """User desktop path (Windows/macOS/Linux)."""
    return os.path.join(os.path.expanduser("~"), "Desktop")

def np_to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Convert BGR numpy array (OpenCV) to QImage for preview."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)

def rms_dbfs(x: np.ndarray) -> float:
    """RMS level in dBFS for a float32 [-1..1] block."""
    if x.size == 0:
        return -120.0
    rms = np.sqrt(np.mean(np.square(x.astype(np.float64))))
    if rms <= 1e-12:
        return -120.0
    return 20.0 * np.log10(rms + 1e-12)

# ---------- screen preview worker (runs when idle) ----------
class PreviewWorker(QObject):
    """Grabs low-FPS screenshots of the selected monitor for live preview."""
    frameReady = Signal(QImage)
    info = Signal(str)

    def __init__(self, monitor_index=1, fps=5):
        super().__init__()
        self._idx = int(monitor_index)
        self._fps = max(1, int(fps))
        self._running = False
        self._th = None

    def set_monitor(self, idx: int):
        self._idx = int(idx)

    def start(self):
        if self._running:
            return
        self._running = True
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        """Stop and wait briefly so the device is fully released."""
        self._running = False
        th = self._th
        if th is not None and th.is_alive():
            th.join(timeout=1.0)
        self._th = None

    def _grab_frame(self, sct, idx: int):
        mons = sct.monitors
        if len(mons) <= 1:
            idx = 1
        idx = max(1, min(idx, len(mons) - 1))
        mon = mons[idx]
        img = np.array(sct.grab(mon))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _run(self):
        try:
            sct = mss.mss()
            tick = 1.0 / self._fps
            last = 0.0
            while self._running:
                t = time.time()
                if t - last >= tick:
                    last = t
                    frame = self._grab_frame(sct, self._idx)
                    self.frameReady.emit(np_to_qimage(frame))
                else:
                    time.sleep(0.005)
        except Exception as e:
            self.info.emit(f"Preview error: {e}")

# ---------- screen recorder ----------
class ScreenRecorder(QObject):
    """High-FPS screen capture to AVI (temp) and live frame emission for preview."""
    frameReady = Signal(QImage)
    stopped = Signal(str)
    info = Signal(str)

    def __init__(self, monitor_index=1, fps=30):
        super().__init__()
        self._idx = int(monitor_index)
        self._fps = int(fps)
        self._running = False
        self._th = None
        self._avi_path = None

    def set_monitor(self, idx: int):
        self._idx = int(idx)

    def set_fps(self, fps: int):
        self._fps = max(5, int(fps))

    def start(self, avi_path: str):
        if self._running:
            return
        self._avi_path = avi_path
        self._running = True
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        self._running = False

    def _run(self):
        out = None
        try:
            sct = mss.mss()
            mons = sct.monitors
            idx = max(1, min(self._idx, len(mons) - 1)) if len(mons) > 1 else 1
            mon = mons[idx]
            w, h = mon["width"], mon["height"]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(self._avi_path, fourcc, self._fps, (w, h))
            self.info.emit(f"ScreenRecorder: monitor {idx} {w}x{h}@{self._fps}")
            frame_interval = 1.0 / self._fps
            last = time.time()
            frames = 0
            while self._running:
                img = np.array(sct.grab(mon))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                out.write(frame)
                frames += 1
                self.frameReady.emit(np_to_qimage(frame))
                sleep_t = frame_interval - (time.time() - last)
                if sleep_t > 0:
                    time.sleep(sleep_t)
                last = time.time()
                if frames % (self._fps * 3) == 0:
                    self.info.emit(f"[Screen] {frames} frames")
        except Exception as e:
            self.info.emit(f"ScreenRecorder error: {e}")
        finally:
            try:
                if out is not None:
                    out.release()
            except Exception:
                pass
            self.info.emit("ScreenRecorder stopped")
            self.stopped.emit(self._avi_path if self._avi_path else "")

# ---------- audio level monitor (idle-only, not recording) ----------
class AudioLevelMonitor(QObject):
    """Lightweight audio grabber for level meter while not recording."""
    levelChanged = Signal(float)
    info = Signal(str)

    def __init__(self, samplerate=48000, channels=2, device_id=None, device_name=None, use_loopback=True, blocksize=1024):
        super().__init__()
        self._sr = int(samplerate)
        self._ch = int(channels)
        self._id = device_id
        self._name = device_name
        self._loop = bool(use_loopback)
        self._bs = int(blocksize)
        self._running = False
        self._th = None
        self._sd_stream = None
        self._sc_rec = None

    def configure(self, samplerate, channels, device_id, device_name, use_loopback):
        """Update configuration; caller decides when to (re)start."""
        self._sr = int(samplerate)
        self._ch = int(channels)
        self._id = device_id
        self._name = device_name
        self._loop = bool(use_loopback)

    def start(self):
        if self._running:
            return
        self._running = True
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        """Stop and join to fully release the device (prevents WASAPI glitches)."""
        self._running = False
        th = self._th
        if th is not None and th.is_alive():
            th.join(timeout=1.5)
        self._th = None

    def _run(self):
        try:
            if self._loop:
                self._run_loopback()
            else:
                self._run_mic()
        except Exception as e:
            self.info.emit(f"[Mon] error: {e}")

    def _run_loopback(self):
        self.info.emit(f"[SC][Mon] sr={self._sr} ch={self._ch} name={self._name}")
        # Reopen on transient WASAPI errors with a small backoff
        reopen_backoff = 0.2
        while self._running:
            try:
                mic = None
                if self._name:
                    try:
                        mic = sc.get_microphone(self._name, include_loopback=True)
                    except Exception:
                        mic = None
                if mic is None:
                    spk = sc.default_speaker()
                    mic = sc.get_microphone(spk.name, include_loopback=True)
                    self.info.emit(f"[SC][Mon] default: {spk.name}")

                ch = self._ch
                try:
                    rec = mic.recorder(samplerate=self._sr, channels=ch, blocksize=self._bs)
                except Exception:
                    ch = 1
                    rec = mic.recorder(samplerate=self._sr, channels=ch, blocksize=self._bs)
                self._sc_rec = rec
                with rec:
                    chunk = max(1024, self._sr // 10)  # ~100ms blocks
                    while self._running:
                        data = rec.record(chunk)
                        self.levelChanged.emit(rms_dbfs(data))
                reopen_backoff = 0.2  # reset after clean run
            except Exception as e:
                self.info.emit(f"[SC][Mon] reopen due to: {e}")
                time.sleep(reopen_backoff)
                reopen_backoff = min(1.0, reopen_backoff * 2)

    def _run_mic(self):
        self.info.emit(f"[SD][Mon] id={self._id} sr={self._sr} ch={self._ch}")
        while self._running:
            try:
                def cb(indata, frames, time_info, status):
                    if status:
                        self.info.emit(f"[SD][Mon] {status}")
                    self.levelChanged.emit(rms_dbfs(indata))
                self._sd_stream = sd.InputStream(
                    device=self._id, samplerate=self._sr, channels=self._ch,
                    dtype="float32", callback=cb, blocksize=self._bs
                )
                with self._sd_stream:
                    while self._running:
                        time.sleep(0.05)
            except Exception as e:
                self.info.emit(f"[SD][Mon] reopen due to: {e}")
                time.sleep(0.2)

# ---------- audio recorder (loopback via soundcard, mic via sounddevice) ----------
class AudioRecorder(QObject):
    """Records audio to WAV (temp) while emitting level for UI."""
    levelChanged = Signal(float)
    stopped = Signal(str)
    info = Signal(str)

    def __init__(self, samplerate=48000, channels=2, device_id=None, device_name=None,
                 use_loopback=True, blocksize=1024):
        super().__init__()
        self._sr = int(samplerate)
        self._ch = int(channels)
        self._dev_id = device_id
        self._dev_name = device_name
        self._loop = bool(use_loopback)
        self._bs = int(blocksize)
        self._running = False
        self._th = None
        self._wav_path = None
        self._sd_stream = None
        self._sc_recorder = None

    def start(self, wav_path: str):
        if self._running:
            return
        self._wav_path = wav_path
        self._running = True
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        self._running = False

    def _run(self):
        if self._loop:
            self._run_loopback()
        else:
            self._run_mic()

    def _run_loopback(self):
        self.info.emit(f"[SC] Loopback REC sr={self._sr} ch={self._ch} name={self._dev_name}")
        try:
            mic = None
            if self._dev_name:
                try:
                    mic = sc.get_microphone(self._dev_name, include_loopback=True)
                except Exception:
                    mic = None
            if mic is None:
                spk = sc.default_speaker()
                mic = sc.get_microphone(spk.name, include_loopback=True)
                self.info.emit(f"[SC] default: {spk.name}")

            # Open recorder, fallback to mono if stereo fails
            ch_try = self._ch
            try:
                rec = mic.recorder(samplerate=self._sr, channels=ch_try, blocksize=self._bs)
            except Exception as e:
                self.info.emit(f"[SC] recorder(ch={ch_try}) failed: {e} -> using ch=1")
                ch_try = 1
                rec = mic.recorder(samplerate=self._sr, channels=ch_try, blocksize=self._bs)
            self._ch = ch_try

            # Collect frames and write once at the end
            with rec:
                chunk = max(1024, self._sr // 10)
                frames = []
                while self._running:
                    data = rec.record(chunk)
                    self.levelChanged.emit(rms_dbfs(data))
                    frames.append(data.copy())

            samples = np.concatenate(frames, axis=0) if frames else np.zeros((0, self._ch), np.float32)
            self._write_wav(samples)

        except Exception as e:
            self.info.emit(f"[SC] loopback error: {e}")
            self._write_wav(np.zeros((0, self._ch), np.float32))
        finally:
            self.stopped.emit(self._wav_path)
            self.info.emit("AudioRecorder stopped")

    def _run_mic(self):
        self.info.emit(f"[SD] Mic REC id={self._dev_id} sr={self._sr} ch={self._ch}")
        try:
            frames = []

            def cb(indata, frames_cnt, time_info, status):
                if status:
                    self.info.emit(f"[SD] status: {status}")
                self.levelChanged.emit(rms_dbfs(indata))
                frames.append(indata.copy())

            self._sd_stream = sd.InputStream(
                device=self._dev_id, samplerate=self._sr, channels=self._ch,
                dtype="float32", callback=cb, blocksize=self._bs
            )
            with self._sd_stream:
                while self._running:
                    time.sleep(0.05)

            samples = np.concatenate(frames, axis=0) if frames else np.zeros((0, self._ch), np.float32)
            self._write_wav(samples)

        except Exception as e:
            self.info.emit(f"[SD] error: {e}")
            self._write_wav(np.zeros((0, self._ch), np.float32))
        finally:
            self.stopped.emit(self._wav_path)
            self.info.emit("AudioRecorder stopped")

    def _write_wav(self, samples: np.ndarray):
        """Write collected float32 samples to WAV (int16), or log if empty."""
        try:
            if samples.size > 0:
                pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
                with wave.open(self._wav_path, "wb") as wf:
                    wf.setnchannels(self._ch)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self._sr)
                    wf.writeframes(pcm.tobytes())
                self.info.emit(f"WAV saved: {self._wav_path}")
            else:
                self.info.emit("Audio empty; no WAV written")
        except Exception as e:
            self.info.emit(f"WAV write error: {e}")

# ---------- main window ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyScreenCap")
        self.resize(1120, 640)

        # state
        self.is_recording = False
        self._screen_done = False
        self._audio_done = False
        self.avi_path = ""
        self.wav_path = ""
        self.output_mp4 = ""

        # audio config
        self.use_loopback = True
        self.selected_device_id = None       # mic (sounddevice)
        self.selected_device_name = None     # speaker name (soundcard)
        self.selected_channels = 2
        self.device_samplerate = 48000

        # video config
        self.fps = 30                        # <- default FPS
        self.monitor_index = 1

        # ---------- UI ----------
        central = QWidget(self)
        self.setCentralWidget(central)

        self.preview = QLabel("Preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        self.preview.setStyleSheet("background:#111; color:#ccc;")

        self.loopbackChk = QCheckBox("System sound (loopback)")
        self.loopbackChk.setChecked(self.use_loopback)
        self.loopbackChk.stateChanged.connect(self.on_loopback_toggle)

        self.monitorBox = QComboBox()
        self.monitorBox.setToolTip("Select monitor to capture")

        self.deviceBox = QComboBox()
        self.deviceBox.setToolTip("Select audio device (speaker for loopback, microphone otherwise)")

        self.chanBox = QComboBox()
        self.chanBox.setToolTip("Channels (1 = mono, 2 = stereo)")

        self.levelBar = QProgressBar()
        self.levelBar.setRange(0, 100)
        self.levelBar.setFormat("Level")

        # --- FPS slider ---
        self.fpsLabel = QLabel(f"FPS: {self.fps}")
        self.fpsSlider = QSlider(Qt.Horizontal)
        self.fpsSlider.setMinimum(1)
        self.fpsSlider.setMaximum(60)
        self.fpsSlider.setSingleStep(1)
        self.fpsSlider.setPageStep(1)
        self.fpsSlider.setValue(self.fps)
        self.fpsSlider.setToolTip("Recording FPS (1–60)")
        self.fpsSlider.valueChanged.connect(self.on_fps_changed)

        self.btnRecord = QPushButton("Record")
        self.btnStop = QPushButton("Stop")
        self.btnStop.setEnabled(False)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Monitor:"))
        row1.addWidget(self.monitorBox)
        row1.addSpacing(12)
        row1.addWidget(self.loopbackChk)
        row1.addWidget(QLabel("Device:"))
        row1.addWidget(self.deviceBox, 1)
        row1.addWidget(QLabel("Channels:"))
        row1.addWidget(self.chanBox)
        row1.addWidget(QLabel("Level:"))
        row1.addWidget(self.levelBar, 1)

        row2 = QHBoxLayout()
        row2.addWidget(self.btnRecord)
        row2.addWidget(self.btnStop)
        row2.addSpacing(24)
        row2.addWidget(self.fpsLabel)
        row2.addWidget(self.fpsSlider, 1)
        row2.addStretch(1)

        layout = QVBoxLayout(central)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(self.preview, 1)

        # actions
        self.btnRecord.clicked.connect(self.on_record)
        self.btnStop.clicked.connect(self.on_stop)

        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        self.menuBar().addAction(act_quit)

        # populate devices/monitors
        self.populate_monitors()
        self.populate_devices()

        # workers
        self.previewWorker = PreviewWorker(monitor_index=self.monitor_index, fps=5)  # keep lightweight
        self.previewWorker.frameReady.connect(self.on_frame)
        self.previewWorker.info.connect(self.on_info)
        self.previewWorker.start()

        self.audioMon = AudioLevelMonitor(
            samplerate=self.device_samplerate,
            channels=self.selected_channels,
            device_id=self.selected_device_id,
            device_name=self.selected_device_name,
            use_loopback=self.use_loopback,
            blocksize=1024
        )
        self.audioMon.levelChanged.connect(self.on_level)
        self.audioMon.info.connect(self.on_info)
        self.audioMon.start()

        self.screen = ScreenRecorder(monitor_index=self.monitor_index, fps=self.fps)
        self.audio = None  # built at record time

        # small UI keepalive
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(500)
        self.ui_timer.timeout.connect(lambda: None)
        self.ui_timer.start()

    # ---------- monitors ----------
    def populate_monitors(self):
        self.monitorBox.clear()
        try:
            sct = mss.mss()
            mons = sct.monitors
            for i in range(1, len(mons)):
                m = mons[i]
                self.monitorBox.addItem(f"{i}: {m['width']}x{m['height']} @ ({m['left']},{m['top']})", userData=i)
            self.monitorBox.setCurrentIndex(0)
            self.monitor_index = self.monitorBox.currentData()
            self.monitorBox.currentIndexChanged.connect(self.on_monitor_changed)
        except Exception as e:
            QMessageBox.warning(self, "Monitor", f"Could not enumerate monitors: {e}")
            self.monitor_index = 1

    def on_monitor_changed(self, idx):
        self.monitor_index = self.monitorBox.itemData(idx)
        # update preview and recorder target
        self.previewWorker.set_monitor(self.monitor_index)
        self.screen.set_monitor(self.monitor_index)

    # ---------- audio devices ----------
    def populate_devices(self):
        self.deviceBox.clear()
        self.selected_device_id = None
        the_name = None
        self.selected_device_name = None

        if self.use_loopback:
            # speakers for loopback
            speakers = sc.all_speakers()
            for spk in speakers:
                self.deviceBox.addItem(spk.name, userData=spk.name)
            if speakers:
                self.deviceBox.setCurrentIndex(0)
                the_name = self.deviceBox.currentData()
                self.selected_device_name = the_name
            # typical for loopback: 48k / stereo
            self.chanBox.clear()
            self.chanBox.addItem("1 (Mono)", 1)
            self.chanBox.addItem("2 (Stereo)", 2)
            self.chanBox.setCurrentIndex(1)
            self.selected_channels = 2
            self.device_samplerate = 48000
        else:
            # microphones
            devices = sd.query_devices()
            mics = [(i, d) for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]
            for i, d in mics:
                self.deviceBox.addItem(f"{i}: {d['name']} (in={d['max_input_channels']})", userData=i)
            if mics:
                self.deviceBox.setCurrentIndex(0)
                self.selected_device_id = self.deviceBox.currentData()
                d0 = sd.query_devices(self.selected_device_id)
                self.device_samplerate = int(d0["default_samplerate"]) if d0["default_samplerate"] > 0 else 44100
                max_in = int(d0["max_input_channels"]) or 1
            else:
                self.device_samplerate = 44100
                max_in = 1
            self.chanBox.clear()
            self.chanBox.addItem("1 (Mono)", 1)
            if mics and max_in >= 2:
                self.chanBox.addItem("2 (Stereo)", 2)
            self.chanBox.setCurrentIndex(0)
            self.selected_channels = 1

        self.deviceBox.currentIndexChanged.connect(self.on_device_changed)
        self.chanBox.currentIndexChanged.connect(self.on_channels_changed)

    def on_loopback_toggle(self, state):
        self.use_loopback = (state == Qt.Checked)
        # reconnect lists cleanly
        try:
            self.deviceBox.currentIndexChanged.disconnect()
        except Exception:
            pass
        try:
            self.chanBox.currentIndexChanged.disconnect()
        except Exception:
            pass
        self.populate_devices()

        # (re)start audio level monitor only when NOT recording
        if not self.is_recording:
            self.audioMon.stop()
            self.audioMon.configure(self.device_samplerate, self.selected_channels,
                                    self.selected_device_id, self.selected_device_name, self.use_loopback)
            self.audioMon.start()

    def on_device_changed(self, idx):
        if self.use_loopback:
            self.selected_device_name = self.deviceBox.itemData(idx)
        else:
            self.selected_device_id = self.deviceBox.itemData(idx)
            dev = sd.query_devices(self.selected_device_id)
            self.device_samplerate = int(dev["default_samplerate"]) if dev["default_samplerate"] > 0 else 44100

        if not self.is_recording:
            self.audioMon.stop()
            self.audioMon.configure(self.device_samplerate, self.selected_channels,
                                    self.selected_device_id, self.selected_device_name, self.use_loopback)
            self.audioMon.start()

    def on_channels_changed(self, idx):
        self.selected_channels = self.chanBox.itemData(idx)
        if not self.is_recording:
            self.audioMon.stop()
            self.audioMon.configure(self.device_samplerate, self.selected_channels,
                                    self.selected_device_id, self.selected_device_name, self.use_loopback)
            self.audioMon.start()

    # ---------- FPS handling ----------
    def on_fps_changed(self, value: int):
        self.fps = int(value)
        self.fpsLabel.setText(f"FPS: {self.fps}")
        # Apply to recorder instance for the next run
        self.screen.set_fps(self.fps)
        self.on_info(f"Recording FPS set to {self.fps} (applies on next recording)")

    # ---------- recording ----------
    def on_record(self):
        if self.is_recording:
            return

        # temp files (cleaned up after export)
        out_dir = desktop_path()
        t = ts()
        self.avi_path = os.path.join(out_dir, f"video_{t}.avi")
        self.wav_path = os.path.join(out_dir, f"audio_{t}.wav")
        self.output_mp4 = os.path.join(out_dir, f"screen_record_{t}.mp4")

        # effective audio params
        sr = self.device_samplerate if self.device_samplerate > 0 else (48000 if self.use_loopback else 44100)
        ch = self.selected_channels if self.selected_channels in (1, 2) else 1
        if not self.use_loopback and self.selected_device_id is not None:
            dev = sd.query_devices(self.selected_device_id)
            max_in = int(dev["max_input_channels"]) or 1
            if ch > max_in:
                ch = max_in

        # stop the audio level monitor to free the device
        self.audioMon.stop()

        # build fresh workers with current configuration
        self.screen = ScreenRecorder(monitor_index=self.monitor_index, fps=self.fps)
        self.audio = AudioRecorder(
            samplerate=sr, channels=ch,
            device_id=self.selected_device_id, device_name=self.selected_device_name,
            use_loopback=self.use_loopback, blocksize=1024
        )

        # connect signals
        self.screen.frameReady.connect(self.on_frame)
        self.screen.stopped.connect(self.on_screen_stopped)
        self.screen.info.connect(self.on_info)
        self.audio.levelChanged.connect(self.on_level)
        self.audio.stopped.connect(self.on_audio_stopped)
        self.audio.info.connect(self.on_info)

        # reset completion flags
        self._screen_done = False
        self._audio_done = False

        # pause preview (we will show frames from the recorder instead)
        self.previewWorker.stop()

        # start recording
        self.screen.start(self.avi_path)
        self.audio.start(self.wav_path)
        self.is_recording = True
        self.btnRecord.setEnabled(False)
        self.btnStop.setEnabled(True)
        self.on_info(f"Recording started -> {self.output_mp4} | monitor={self.monitor_index}, fps={self.fps}, sr={sr}, ch={ch}, loopback={self.use_loopback}")

    def on_stop(self):
        if not self.is_recording:
            return
        self.on_info("Stopping requested …")
        self.screen.stop()
        self.audio.stop()
        self.btnStop.setEnabled(False)

    def on_screen_stopped(self, path):
        self._screen_done = True
        self.on_info(f"Screen stopped: {path}")
        self._maybe_mux()

    def on_audio_stopped(self, path):
        self._audio_done = True
        self.on_info(f"Audio stopped: {path if path else 'empty'}")
        self._maybe_mux()

    def _maybe_mux(self):
        """Mux once both recorders are done. Delete temp files; keep only MP4."""
        if not self.is_recording:
            return
        if not self._screen_done or not self._audio_done:
            return
        if not (self.avi_path and os.path.exists(self.avi_path)):
            return

        have_audio = bool(self.wav_path and os.path.exists(self.wav_path) and os.path.getsize(self.wav_path) > 44)
        try:
            self.on_info("Exporting MP4…" if not have_audio else "Merging video + audio to MP4…")
            v = VideoFileClip(self.avi_path)
            if have_audio:
                a = AudioFileClip(self.wav_path)
                final = v.set_audio(a)
                final.write_videofile(self.output_mp4, codec="libx264", audio_codec="aac")
                a.close()
            else:
                v.write_videofile(self.output_mp4, codec="libx264")
            v.close()
            self.on_info(f"Done: {self.output_mp4}")
        except Exception as e:
            QMessageBox.critical(self, "Export", f"Export error: {e}")
            self.on_info(f"Mux error: {e}")
        finally:
            # remove temp files no matter what
            for p in (self.avi_path, self.wav_path):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            self.is_recording = False
            self.btnRecord.setEnabled(True)
            self.btnStop.setEnabled(False)
            # resume preview and audio monitor
            self.previewWorker.set_monitor(self.monitor_index)
            self.previewWorker.start()
            self.audioMon.configure(self.device_samplerate, self.selected_channels,
                                    self.selected_device_id, self.selected_device_name, self.use_loopback)
            self.audioMon.start()

    # ---------- UI slots ----------
    def on_frame(self, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        self.preview.setPixmap(pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_level(self, db: float):
        # Map -60..0 dBFS to 0..100
        val = int(np.interp(db, [-60.0, 0.0], [0, 100]))
        val = max(0, min(100, val))
        self.levelBar.setValue(val)

    def on_info(self, msg: str):
        print(msg)
        self.statusBar().showMessage(msg, 4000)

# ---------- main ----------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
