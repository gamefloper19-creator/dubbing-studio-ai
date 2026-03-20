"""
Dubbing Studio - Native PyQt6 GUI
Professional AI Documentary Dubbing Platform.
"""

import sys
import os
import logging
import traceback
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QCheckBox,
    QProgressBar, QTextEdit, QFileDialog, QFrame, QScrollArea,
    QSizePolicy, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QIcon, QColor, QPalette, QLinearGradient, QGradient

from dubbing_studio import __app_name__, __version__
from dubbing_studio.config import AppConfig, SUPPORTED_LANGUAGES
from dubbing_studio.pipeline import DubbingPipeline, PIPELINE_STAGES
from dubbing_studio.hardware.optimizer import HardwareOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -- Styling Constants (Black & Gold Elegance) --
COLORS = {
    "bg": "#0B0B0B",
    "bg_light": "#14213D",
    "panel": "#121826",
    "accent": "#FCA311",
    "text": "#F5F5F5",
    "muted": "#E5E5E5",
    "border": "rgba(255, 255, 255, 0.08)",
}

QSS = f"""
QMainWindow {{
    background-color: {COLORS['bg']};
}}

QWidget {{
    color: {COLORS['text']};
    font-family: 'Segoe UI', sans-serif;
    font-size: 14px;
}}

QFrame#MainPanel {{
    background-color: {COLORS['panel']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
}}

QLabel#Title {{
    font-size: 28px;
    font-weight: bold;
    color: {COLORS['text']};
    margin-bottom: 5px;
}}

QLabel#Subtitle {{
    font-size: 12px;
    color: {COLORS['accent']};
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 20px;
}}

QPushButton {{
    background-color: {COLORS['accent']};
    color: #000000;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
}}

QPushButton:hover {{
    background-color: #e59400;
}}

QPushButton#SecondaryBtn {{
    background-color: transparent;
    border: 1px solid {COLORS['accent']};
    color: {COLORS['accent']};
}}

QPushButton#SecondaryBtn:hover {{
    background-color: rgba(252, 163, 17, 0.1);
}}

QLineEdit, QComboBox {{
    background-color: #0F172A;
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 8px;
    color: {COLORS['text']};
}}

QLineEdit:focus, QComboBox:focus {{
    border: 1px solid {COLORS['accent']};
}}

QProgressBar {{
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    text-align: center;
    background-color: #0F172A;
}}

QProgressBar::chunk {{
    background-color: {COLORS['accent']};
    border-radius: 4px;
}}

QTextEdit {{
    background-color: #0C1120;
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    font-family: 'Consolas', monospace;
    font-size: 12px;
    color: {COLORS['muted']};
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 1px solid {COLORS['border']};
    background: #0F172A;
}}

QCheckBox::indicator:checked {{
    background: {COLORS['accent']};
}}
"""

class WorkerThread(QThread):
    progress_signal = pyqtSignal(str, float)
    finished_signal = pyqtSignal(object, str) # result, error
    log_signal = pyqtSignal(str)

    def __init__(self, pipeline, params):
        super().__init__()
        self.pipeline = pipeline
        self.params = params

    def run(self):
        try:
            self.log_signal.emit("Starting pipeline...")
            
            def progress_hook(stage, prog):
                self.progress_signal.emit(stage, prog)
                self.log_signal.emit(f"[{int(prog*100)}%] {stage}")

            result = self.pipeline.process_video(
                video_path=self.params['video_path'],
                target_language=self.params['target_language'],
                narrator_style=self.params['narrator_style'],
                progress_callback=progress_hook
            )
            self.finished_signal.emit(result, "")
        except Exception as e:
            err = traceback.format_exc()
            self.log_signal.emit(f"ERROR: {str(e)}")
            self.finished_signal.emit(None, err)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = AppConfig.from_env()
        self.pipeline = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"{__app_name__} v{__version__}")
        self.setMinimumSize(900, 700)
        self.setStyleSheet(QSS)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # Header
        header_layout = QVBoxLayout()
        title_label = QLabel(__app_name__)
        title_label.setObjectName("Title")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label = QLabel("Professional AI Documentary Dubbing Platform")
        subtitle_label.setObjectName("Subtitle")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        main_layout.addLayout(header_layout)

        # Content Panel
        panel = QFrame()
        panel.setObjectName("MainPanel")
        panel_layout = QGridLayout(panel)
        panel_layout.setContentsMargins(20, 20, 20, 20)
        panel_layout.setSpacing(15)

        # Input File
        panel_layout.addWidget(QLabel("Video Input:"), 0, 0)
        self.video_input = QLineEdit()
        self.video_input.setPlaceholderText("Select video or paste YouTube URL...")
        panel_layout.addWidget(self.video_input, 0, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("SecondaryBtn")
        browse_btn.setFixedWidth(100)
        browse_btn.clicked.connect(self.browse_video)
        panel_layout.addWidget(browse_btn, 0, 2)

        # Language Selection
        panel_layout.addWidget(QLabel("Target Language:"), 1, 0)
        self.lang_combo = QComboBox()
        for code, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1]):
            self.lang_combo.addItem(name, code)
        self.lang_combo.setCurrentText("Hindi")
        panel_layout.addWidget(self.lang_combo, 1, 1, 1, 2)

        # Narrator Style
        panel_layout.addWidget(QLabel("Narration Style:"), 2, 0)
        self.style_combo = QComboBox()
        self.style_combo.addItems(["documentary", "cinematic", "calm", "storytelling"])
        panel_layout.addWidget(self.style_combo, 2, 1, 1, 2)

        # Whisper Model
        panel_layout.addWidget(QLabel("Whisper Model:"), 3, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["auto", "tiny", "base", "medium", "large-v3"])
        panel_layout.addWidget(self.model_combo, 3, 1, 1, 2)

        # Toggles
        toggles_layout = QHBoxLayout()
        self.embed_sub_cb = QCheckBox("Embed Subtitles")
        self.clone_voice_cb = QCheckBox("Voice Cloning")
        self.diarization_cb = QCheckBox("Multi-Speaker")
        toggles_layout.addWidget(self.embed_sub_cb)
        toggles_layout.addWidget(self.clone_voice_cb)
        toggles_layout.addWidget(self.diarization_cb)
        panel_layout.addLayout(toggles_layout, 4, 1, 1, 2)

        # Gemini API Key
        panel_layout.addWidget(QLabel("Gemini API Key:"), 5, 0)
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setText(os.environ.get("GEMINI_API_KEY", ""))
        panel_layout.addWidget(self.api_key_input, 5, 1, 1, 2)

        main_layout.addWidget(panel)

        # Action Button
        self.start_btn = QPushButton("START DUBBING")
        self.start_btn.setFixedHeight(50)
        self.start_btn.clicked.connect(self.start_processing)
        main_layout.addWidget(self.start_btn)

        # Progress Section
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% - Ready")
        main_layout.addWidget(self.progress_bar)

        # Console Output
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("Console logs will appear here...")
        main_layout.addWidget(self.console)

        # Footer
        footer_layout = QHBoxLayout()
        self.hw_label = QLabel("Detecting hardware...")
        self.hw_label.setStyleSheet(f"color: {COLORS['accent']}; font-size: 11px;")
        footer_layout.addWidget(self.hw_label)
        footer_layout.addStretch()
        copy_err_btn = QPushButton("Copy Last Error")
        copy_err_btn.setObjectName("SecondaryBtn")
        copy_err_btn.setFixedWidth(120)
        copy_err_btn.clicked.connect(self.copy_error)
        footer_layout.addWidget(copy_err_btn)
        main_layout.addLayout(footer_layout)

        self.detect_hardware()

    def detect_hardware(self):
        try:
            opt = HardwareOptimizer()
            info = opt.detect_hardware()
            self.hw_label.setText(f"System: {info.total_cores} Cores | {info.total_ram_gb}GB RAM | GPU: {'Detected' if info.gpu_available else 'None'}")
        except:
            self.hw_label.setText("Hardware detection failed")

    def browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.mkv *.avi *.mov)")
        if path:
            self.video_input.setText(path)

    def start_processing(self):
        video_path = self.video_input.text().strip()
        if not video_path:
            self.log("Please select a video file or enter a YouTube URL.")
            return

        api_key = self.api_key_input.text().strip()
        if not api_key:
            self.log("Gemini API key is required.")
            return

        # Update config
        self.config.translation.api_key = api_key
        self.config.whisper.model_size = self.model_combo.currentText()
        self.config.subtitle.embed_in_video = self.embed_sub_cb.isChecked()
        
        self.pipeline = DubbingPipeline(self.config)
        
        params = {
            'video_path': video_path,
            'target_language': self.lang_combo.currentData(),
            'narrator_style': self.style_combo.currentText()
        }

        self.start_btn.setEnabled(False)
        self.console.clear()
        self.progress_bar.setValue(0)
        
        self.worker = WorkerThread(self.pipeline, params)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def update_progress(self, stage, prog):
        self.progress_bar.setValue(int(prog * 100))
        self.progress_bar.setFormat(f"%p% - {stage}")

    def log(self, message):
        self.console.append(message)
        scroll = self.console.verticalScrollBar()
        scroll.setValue(scroll.maximum())

    def on_finished(self, result, error):
        self.start_btn.setEnabled(True)
        if error:
            self.last_error = error
            self.log("\n!!! PROCESSING FAILED !!!")
            self.log(error)
            self.progress_bar.setFormat("Error - See Logs")
        else:
            self.log("\n" + "="*40)
            self.log("DUBBING COMPLETE!")
            self.log(f"Output: {result.output_video_path}")
            self.log("="*40)
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Success!")

    def copy_error(self):
        if hasattr(self, 'last_error'):
            QApplication.clipboard().setText(self.last_error)
            self.log("Last error copied to clipboard.")
        else:
            self.log("No error to copy.")

def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()
