import sys
import os
import asyncio
from typing import Optional
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QFileDialog, QProgressBar, 
    QTextEdit, QCheckBox, QSlider, QTabWidget, QGroupBox, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon

# Import dubbing studio components
from dubbing_studio.config import AppConfig, SUPPORTED_LANGUAGES
from dubbing_studio.pipeline import DubbingPipeline
from dubbing_studio import __app_name__, __version__
import logging

LANGUAGE_CHOICES = [(name, code) for code, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1])]
NARRATOR_STYLES = ["documentary", "cinematic", "calm", "storytelling"]
WHISPER_MODELS = ["tiny", "base", "medium", "large-v3"]
SUBTITLE_FORMATS = ["srt", "vtt", "ass"]

class WorkerThread(QThread):
    progress_update = pyqtSignal(str, float)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, config, video_path, target_language, narrator_style):
        super().__init__()
        self.config = config
        self.video_path = video_path
        self.target_language = target_language
        self.narrator_style = narrator_style
        
    def run(self):
        try:
            pipeline = DubbingPipeline(self.config)
            
            def progress_callback(stage: str, progress: float):
                self.progress_update.emit(stage, progress)
                
            result = pipeline.process_video(
                video_path=self.video_path,
                target_language=self.target_language,
                narrator_style=self.narrator_style,
                progress_callback=progress_callback
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class DubbingStudioGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{__app_name__} v{__version__} - Native GUI")
        self.resize(800, 600)
        self.setup_theme()
        self.init_ui()
        self.config = AppConfig.from_env()
        self.config.setup_dirs()
        self.worker = None

    def setup_theme(self):
        # Dark Neon Theme
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(18, 18, 24))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.Base, QColor(28, 28, 38))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(38, 38, 50))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Text, QColor(230, 230, 230))
        palette.setColor(QPalette.ColorRole.Button, QColor(36, 36, 50))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(250, 250, 250))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 153, 255)) # Neon Blue
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #333333;
                color: #888888;
            }
            QPushButton#primaryBtn {
                background-color: #d6006e; /* Neon Pink */
                font-size: 14px;
                padding: 10px 20px;
            }
            QPushButton#primaryBtn:hover {
                background-color: #a30054;
            }
            QGroupBox {
                border: 1px solid #3c3c50;
                border-radius: 6px;
                margin-top: 10px;
                font-weight: bold;
                color: #00e5ff; /* Cyan */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QComboBox, QLineEdit {
                background-color: #1e1e28;
                border: 1px solid #3c3c50;
                color: white;
                padding: 4px;
                border-radius: 2px;
            }
            QProgressBar {
                border: 1px solid #3c3c50;
                border-radius: 4px;
                text-align: center;
                background-color: #1e1e28;
            }
            QProgressBar::chunk {
                background-color: #00e5ff; /* Cyan progress */
                width: 10px;
                margin: 0.5px;
            }
            QTextEdit {
                background-color: #1a1a24;
                border: 1px solid #3c3c50;
                color: #00ffcc; /* Mint log text */
                font-family: Consolas, monospace;
            }
        """)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Header
        header = QLabel(f"<b>{__app_name__}</b> - AI Dubbing Studio")
        header.setFont(QFont("Arial", 16))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #00e5ff; margin-bottom: 10px;")
        layout.addWidget(header)
        
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Single Video Tab
        single_tab = QWidget()
        self.setup_single_tab(single_tab)
        tabs.addTab(single_tab, "Single Video")
        
        # Logs Panel
        log_group = QGroupBox("Processing Output & Logs")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        
        # Progress
        progress_layout = QHBoxLayout()
        self.status_label = QLabel("Idle")
        self.status_label.setFixedWidth(200)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        log_layout.addLayout(progress_layout)
        
        layout.addWidget(log_group)

    def setup_single_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Video Input
        upload_group = QGroupBox("Input Video")
        upload_layout = QHBoxLayout(upload_group)
        self.video_path_lbl = QLabel("No video selected")
        self.video_path_lbl.setStyleSheet("color: #aaaaaa;")
        btn_browse = QPushButton("Browse Video...")
        btn_browse.clicked.connect(self.browse_video)
        upload_layout.addWidget(btn_browse)
        upload_layout.addWidget(self.video_path_lbl, 1)
        layout.addWidget(upload_group)
        
        # Settings
        settings_group = QGroupBox("Dubbing Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Target Language:"))
        self.lang_cb = QComboBox()
        for name, code in LANGUAGE_CHOICES:
            self.lang_cb.addItem(name, code)
        # Default to Hindi
        hi_idx = self.lang_cb.findData("hi")
        if hi_idx >= 0: self.lang_cb.setCurrentIndex(hi_idx)
        row1.addWidget(self.lang_cb, 1)
        
        row1.addWidget(QLabel("Narrator Style:"))
        self.style_cb = QComboBox()
        self.style_cb.addItems(NARRATOR_STYLES)
        row1.addWidget(self.style_cb, 1)
        settings_layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Gemini API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Required for translation or use GEMINI_API_KEY env var")
        self.api_key_input.setText(os.environ.get("GEMINI_API_KEY", ""))
        row2.addWidget(self.api_key_input, 1)
        settings_layout.addLayout(row2)
        
        adv_group = QGroupBox("Advanced Options")
        adv_layout = QHBoxLayout(adv_group)
        
        adv_layout.addWidget(QLabel("Whisper Model:"))
        self.whisper_cb = QComboBox()
        self.whisper_cb.addItems(WHISPER_MODELS)
        self.whisper_cb.setCurrentText("base")
        adv_layout.addWidget(self.whisper_cb)
        
        self.embed_chk = QCheckBox("Embed Subtitles")
        adv_layout.addWidget(self.embed_chk)
        
        settings_layout.addWidget(adv_group)
        layout.addWidget(settings_group)
        
        layout.addStretch()
        
        self.start_btn = QPushButton("Start Dubbing")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.clicked.connect(self.start_dubbing)
        self.start_btn.setMinimumHeight(45)
        layout.addWidget(self.start_btn)
        
        self.video_file = None

    def browse_video(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.mkv *.avi *.mov *.webm)"
        )
        if file:
            self.video_file = file
            self.video_path_lbl.setText(file)

    def log(self, msg):
        self.log_output.append(msg)
        # Scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def start_dubbing(self):
        if not self.video_file:
            self.log("ERROR: Please select a video file first.")
            return
            
        api_key = self.api_key_input.text().strip()
        if not api_key and not os.environ.get("GEMINI_API_KEY"):
            self.log("ERROR: Gemini API Key is missing. Please provide it.")
            return

        # Configure
        if api_key:
            self.config.translation.api_key = api_key
            os.environ["GEMINI_API_KEY"] = api_key
            
        self.config.whisper.model_size = self.whisper_cb.currentText()
        self.config.subtitle.embed_in_video = self.embed_chk.isChecked()
        
        target_lang = self.lang_cb.currentData()
        narrator = self.style_cb.currentText()
        
        self.log(f"Starting dubbing to {self.lang_cb.currentText()} - Style: {narrator}")
        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing...")
        
        self.worker = WorkerThread(self.config, self.video_file, target_lang, narrator)
        self.worker.progress_update.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()
        
    def on_progress(self, stage, progress):
        self.status_label.setText(stage)
        self.progress_bar.setValue(int(progress * 100))
        pct = int(progress * 100)
        self.log(f"[{pct:3d}%] {stage}")
        
    def on_finished(self, result):
        self.start_btn.setEnabled(True)
        self.status_label.setText("Completed")
        self.progress_bar.setValue(100)
        self.log("="*40)
        self.log("Dubbing Complete!")
        self.log(f"Source Language: {result.source_language}")
        self.log(f"Target Language: {result.target_language}")
        self.log(f"Total Segments: {result.total_segments}")
        self.log(f"Processing Time: {result.processing_time:.1f}s")
        self.log(f"Output Video: {result.output_video_path}")
        self.log("="*40)
        
    def on_error(self, err_msg):
        self.start_btn.setEnabled(True)
        self.status_label.setText("Failed")
        self.log(f"ERROR: {err_msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Set app wide dark style base
    app.setStyle("Fusion")
    window = DubbingStudioGUI()
    window.show()
    sys.exit(app.exec())
