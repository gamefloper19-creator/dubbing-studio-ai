"""
Dubbing Studio - Windows Desktop Application (PySide6)

A standalone PySide6-based GUI for the Dubbing Studio AI pipeline,
designed to be packaged as DubbingStudio.exe via PyInstaller.

Tabs:
    1. Single Video Dubbing
    2. Batch Processing
    3. Voice Library
    4. Voice Cloning
    5. YouTube Dubbing
    6. Model Manager
    7. System Information
"""

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, QObject, QMimeData, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont, QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from dubbing_studio import __app_name__, __version__
from dubbing_studio.config import AppConfig, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

# Constants
NARRATOR_STYLES = ["documentary", "cinematic", "calm", "storytelling"]
WHISPER_MODELS = ["tiny", "base", "medium", "large-v3"]
SUBTITLE_FORMATS = ["srt", "vtt", "ass"]
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v")
VIDEO_FILTER = "Video Files (*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv *.m4v);;All Files (*)"
AUDIO_FILTER = "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg);;All Files (*)"


# ── Signal bridge for thread-safe UI updates ──

class WorkerSignals(QObject):
    """Signals for background worker threads."""
    progress = Signal(float, str)
    finished = Signal(str)
    error = Signal(str)
    log = Signal(str)


# ── Drag-and-drop line edit ──

class DropLineEdit(QLineEdit):
    """A QLineEdit that accepts drag-and-drop file paths."""

    def __init__(self, file_filter: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._filter_exts: tuple[str, ...] = ()
        if file_filter:
            self._filter_exts = tuple(
                ext.strip() for ext in file_filter.replace("*", "").split()
                if ext.startswith(".")
            )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if not self._filter_exts or path.lower().endswith(self._filter_exts):
                self.setText(path)
                event.acceptProposedAction()
            else:
                event.ignore()


# ── Log handler that emits signals ──

class SignalLogHandler(logging.Handler):
    """Logging handler that emits log records as Qt signals."""

    def __init__(self, signals: WorkerSignals):
        super().__init__()
        self._signals = signals

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self._signals.log.emit(msg)


# ── Main Application Window ──

class DubbingStudioApp(QMainWindow):
    """Main desktop application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{__app_name__} v{__version__}")
        self.setMinimumSize(960, 720)
        self.resize(1080, 780)

        self._processing = False
        self._config: Optional[AppConfig] = None
        self._signals = WorkerSignals()

        # Connect signals
        self._signals.progress.connect(self._on_progress)
        self._signals.finished.connect(self._on_finished)
        self._signals.error.connect(self._on_error)
        self._signals.log.connect(self._append_log)

        # Install log handler
        handler = SignalLogHandler(self._signals)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(handler)

        self._build_menu()
        self._build_ui()

        self._append_log(f"{__app_name__} v{__version__} ready.")
        self._append_log("Select a video file and configure settings to begin dubbing.")

    # ── Config ──

    def _get_config(self) -> AppConfig:
        if self._config is None:
            self._config = AppConfig.from_env()
            self._config.setup_dirs()
        return self._config

    # ── Menu bar ──

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        open_action = QAction("&Open Video...", self)
        open_action.triggered.connect(self._browse_video)
        file_menu.addAction(open_action)
        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    # ── Central widget ──

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        header = QLabel(f"{__app_name__} v{__version__}")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        subtitle = QLabel("Professional AI Documentary Dubbing Platform")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Splitter: tabs on top, log on bottom
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter, stretch=1)

        # Tab widget
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)

        self._build_single_tab()
        self._build_batch_tab()
        self._build_voice_library_tab()
        self._build_voice_cloning_tab()
        self._build_youtube_tab()
        self._build_model_manager_tab()
        self._build_system_tab()

        # Log panel
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMaximumBlockCount(2000)
        log_layout.addWidget(self.log_text)
        splitter.addWidget(log_group)

        splitter.setSizes([500, 200])

        # Progress bar
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar, stretch=1)
        self.progress_label = QLabel("Idle")
        progress_layout.addWidget(self.progress_label)
        layout.addLayout(progress_layout)

    # ──────────────────────────────────────────────
    #  Tab 1: Single Video Dubbing
    # ──────────────────────────────────────────────

    def _build_single_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "Single Video")
        layout = QVBoxLayout(tab)

        # Video file (drag-and-drop)
        file_group = QGroupBox("Video File (drag && drop supported)")
        file_layout = QHBoxLayout(file_group)
        self.video_path = DropLineEdit(VIDEO_FILTER)
        self.video_path.setPlaceholderText("Drag a video file here or click Browse...")
        file_layout.addWidget(self.video_path, stretch=1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_video)
        file_layout.addWidget(browse_btn)
        layout.addWidget(file_group)

        # Settings
        settings_group = QGroupBox("Dubbing Settings")
        form = QFormLayout(settings_group)

        # Language
        self.lang_combo = QComboBox()
        lang_items = sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1])
        for code, name in lang_items:
            self.lang_combo.addItem(f"{name} ({code})", code)
        # Default to Hindi
        for i in range(self.lang_combo.count()):
            if self.lang_combo.itemData(i) == "hi":
                self.lang_combo.setCurrentIndex(i)
                break
        form.addRow("Target Language:", self.lang_combo)

        # Narrator style
        self.style_combo = QComboBox()
        self.style_combo.addItems(NARRATOR_STYLES)
        form.addRow("Narrator Style:", self.style_combo)

        # Whisper model
        self.whisper_combo = QComboBox()
        self.whisper_combo.addItems(WHISPER_MODELS)
        self.whisper_combo.setCurrentText("base")
        form.addRow("Whisper Model:", self.whisper_combo)

        # Subtitle format
        self.subtitle_combo = QComboBox()
        self.subtitle_combo.addItems(SUBTITLE_FORMATS)
        form.addRow("Subtitle Format:", self.subtitle_combo)

        # Embed subtitles
        self.embed_subs_check = QCheckBox("Embed subtitles into video")
        form.addRow("", self.embed_subs_check)

        # Background volume
        vol_layout = QHBoxLayout()
        self.bg_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.bg_volume_slider.setRange(0, 100)
        self.bg_volume_slider.setValue(15)
        self.bg_volume_label = QLabel("15%")
        self.bg_volume_slider.valueChanged.connect(
            lambda v: self.bg_volume_label.setText(f"{v}%")
        )
        vol_layout.addWidget(self.bg_volume_slider, stretch=1)
        vol_layout.addWidget(self.bg_volume_label)
        form.addRow("Background Volume:", vol_layout)

        # API key
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Enter Gemini API key or set GEMINI_API_KEY env var")
        self.api_key_input.setText(os.environ.get("GEMINI_API_KEY", ""))
        form.addRow("Gemini API Key:", self.api_key_input)

        layout.addWidget(settings_group)

        # Output directory
        out_group = QGroupBox("Output Directory")
        out_layout = QHBoxLayout(out_group)
        self.output_dir_input = QLineEdit("output")
        out_layout.addWidget(self.output_dir_input, stretch=1)
        out_browse = QPushButton("Browse...")
        out_browse.clicked.connect(self._browse_output_dir)
        out_layout.addWidget(out_browse)
        layout.addWidget(out_group)

        # Action buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Dubbing")
        self.start_btn.setStyleSheet("font-weight: bold; padding: 8px 20px;")
        self.start_btn.clicked.connect(self._start_single_dubbing)
        btn_layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_processing)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    # ──────────────────────────────────────────────
    #  Tab 2: Batch Processing
    # ──────────────────────────────────────────────

    def _build_batch_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "Batch Processing")
        layout = QVBoxLayout(tab)

        # Directory
        dir_group = QGroupBox("Video Directory")
        dir_layout = QHBoxLayout(dir_group)
        self.batch_dir_input = QLineEdit()
        self.batch_dir_input.setPlaceholderText("Select a directory containing video files...")
        dir_layout.addWidget(self.batch_dir_input, stretch=1)
        dir_browse = QPushButton("Browse...")
        dir_browse.clicked.connect(self._browse_batch_dir)
        dir_layout.addWidget(dir_browse)
        layout.addWidget(dir_group)

        # Batch settings
        settings_group = QGroupBox("Batch Settings")
        form = QFormLayout(settings_group)

        self.batch_lang_combo = QComboBox()
        for code, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1]):
            self.batch_lang_combo.addItem(f"{name} ({code})", code)
        for i in range(self.batch_lang_combo.count()):
            if self.batch_lang_combo.itemData(i) == "hi":
                self.batch_lang_combo.setCurrentIndex(i)
                break
        form.addRow("Target Language:", self.batch_lang_combo)

        self.batch_style_combo = QComboBox()
        self.batch_style_combo.addItems(NARRATOR_STYLES)
        form.addRow("Narrator Style:", self.batch_style_combo)

        self.max_concurrent_spin = QSpinBox()
        self.max_concurrent_spin.setRange(1, 16)
        self.max_concurrent_spin.setValue(4)
        form.addRow("Max Concurrent Jobs:", self.max_concurrent_spin)

        layout.addWidget(settings_group)

        # Batch queue table
        queue_group = QGroupBox("Processing Queue")
        queue_layout = QVBoxLayout(queue_group)
        self.batch_table = QTableWidget(0, 4)
        self.batch_table.setHorizontalHeaderLabels(["File", "Status", "Progress", "Output"])
        self.batch_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.batch_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        queue_layout.addWidget(self.batch_table)
        layout.addWidget(queue_group)

        # Actions
        btn_layout = QHBoxLayout()
        self.batch_start_btn = QPushButton("Start Batch Processing")
        self.batch_start_btn.setStyleSheet("font-weight: bold; padding: 8px 20px;")
        self.batch_start_btn.clicked.connect(self._start_batch_processing)
        btn_layout.addWidget(self.batch_start_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    # ──────────────────────────────────────────────
    #  Tab 3: Voice Library
    # ──────────────────────────────────────────────

    def _build_voice_library_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "Voice Library")
        layout = QVBoxLayout(tab)

        # Filter
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by Language:"))
        self.voice_lang_filter = QComboBox()
        self.voice_lang_filter.addItem("All Languages", "")
        for code, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1]):
            self.voice_lang_filter.addItem(f"{name} ({code})", code)
        self.voice_lang_filter.currentIndexChanged.connect(self._refresh_voice_table)
        filter_layout.addWidget(self.voice_lang_filter)
        filter_layout.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_voice_table)
        filter_layout.addWidget(refresh_btn)
        layout.addLayout(filter_layout)

        # Voice table
        self.voice_table = QTableWidget(0, 5)
        self.voice_table.setHorizontalHeaderLabels(["Name", "Language", "Gender", "Style", "Engine"])
        self.voice_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.voice_table)

        # Summary
        self.voice_summary_label = QLabel("")
        layout.addWidget(self.voice_summary_label)

        # Load data
        self._refresh_voice_table()

    def _refresh_voice_table(self) -> None:
        """Refresh the voice library table."""
        try:
            from dubbing_studio.tts.voice_library import VoiceLibrary
            library = VoiceLibrary()
            lang = self.voice_lang_filter.currentData()
            rows = library.format_for_display(language=lang if lang else None)
            summary = library.get_library_summary()

            self.voice_table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                for j, val in enumerate(row):
                    self.voice_table.setItem(i, j, QTableWidgetItem(str(val)))

            self.voice_summary_label.setText(
                f"Total voices: {summary['total_voices']} | "
                f"Languages: {summary['languages']} | "
                f"Styles: {', '.join(summary['styles'])} | "
                f"Engines: {', '.join(summary['engines'])}"
            )
        except Exception as e:
            self.voice_summary_label.setText(f"Error loading voice library: {e}")

    # ──────────────────────────────────────────────
    #  Tab 4: Voice Cloning
    # ──────────────────────────────────────────────

    def _build_voice_cloning_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "Voice Cloning")
        layout = QVBoxLayout(tab)

        # Create profile
        create_group = QGroupBox("Create Voice Profile")
        create_form = QFormLayout(create_group)

        sample_layout = QHBoxLayout()
        self.clone_audio_input = DropLineEdit(AUDIO_FILTER)
        self.clone_audio_input.setPlaceholderText("Select voice sample audio (10-60 seconds)...")
        sample_layout.addWidget(self.clone_audio_input, stretch=1)
        sample_browse = QPushButton("Browse...")
        sample_browse.clicked.connect(self._browse_clone_audio)
        sample_layout.addWidget(sample_browse)
        create_form.addRow("Voice Sample:", sample_layout)

        self.clone_name_input = QLineEdit()
        self.clone_name_input.setPlaceholderText("e.g., David Attenborough Style")
        create_form.addRow("Profile Name:", self.clone_name_input)

        self.clone_gender_combo = QComboBox()
        self.clone_gender_combo.addItems(["male", "female"])
        create_form.addRow("Gender:", self.clone_gender_combo)

        self.clone_lang_combo = QComboBox()
        for code, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1]):
            self.clone_lang_combo.addItem(f"{name} ({code})", code)
        create_form.addRow("Language:", self.clone_lang_combo)

        self.clone_desc_input = QLineEdit()
        self.clone_desc_input.setPlaceholderText("Optional description...")
        create_form.addRow("Description:", self.clone_desc_input)

        create_btn = QPushButton("Create Voice Profile")
        create_btn.clicked.connect(self._create_voice_profile)
        create_form.addRow("", create_btn)

        layout.addWidget(create_group)

        # Saved profiles
        profiles_group = QGroupBox("Saved Voice Profiles")
        profiles_layout = QVBoxLayout(profiles_group)

        self.profiles_text = QPlainTextEdit()
        self.profiles_text.setReadOnly(True)
        self.profiles_text.setMaximumHeight(150)
        profiles_layout.addWidget(self.profiles_text)

        profiles_btn_layout = QHBoxLayout()
        refresh_profiles_btn = QPushButton("Refresh Profiles")
        refresh_profiles_btn.clicked.connect(self._refresh_profiles)
        profiles_btn_layout.addWidget(refresh_profiles_btn)

        profiles_btn_layout.addWidget(QLabel("Delete ID:"))
        self.delete_profile_id_input = QLineEdit()
        self.delete_profile_id_input.setMaximumWidth(200)
        profiles_btn_layout.addWidget(self.delete_profile_id_input)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_voice_profile)
        profiles_btn_layout.addWidget(delete_btn)

        profiles_btn_layout.addStretch()
        profiles_layout.addLayout(profiles_btn_layout)
        layout.addWidget(profiles_group)

        layout.addStretch()

    # ──────────────────────────────────────────────
    #  Tab 5: YouTube Dubbing
    # ──────────────────────────────────────────────

    def _build_youtube_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "YouTube Dubbing")
        layout = QVBoxLayout(tab)

        # URL input
        url_group = QGroupBox("YouTube Video URL")
        url_layout = QHBoxLayout(url_group)
        self.youtube_url_input = QLineEdit()
        self.youtube_url_input.setPlaceholderText("https://www.youtube.com/watch?v=...")
        url_layout.addWidget(self.youtube_url_input, stretch=1)
        layout.addWidget(url_group)

        # Settings
        yt_settings = QGroupBox("YouTube Dubbing Settings")
        yt_form = QFormLayout(yt_settings)

        self.yt_lang_combo = QComboBox()
        for code, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1]):
            self.yt_lang_combo.addItem(f"{name} ({code})", code)
        for i in range(self.yt_lang_combo.count()):
            if self.yt_lang_combo.itemData(i) == "hi":
                self.yt_lang_combo.setCurrentIndex(i)
                break
        yt_form.addRow("Target Language:", self.yt_lang_combo)

        self.yt_style_combo = QComboBox()
        self.yt_style_combo.addItems(NARRATOR_STYLES)
        yt_form.addRow("Narrator Style:", self.yt_style_combo)

        self.yt_whisper_combo = QComboBox()
        self.yt_whisper_combo.addItems(WHISPER_MODELS)
        self.yt_whisper_combo.setCurrentText("base")
        yt_form.addRow("Whisper Model:", self.yt_whisper_combo)

        self.yt_api_key_input = QLineEdit()
        self.yt_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.yt_api_key_input.setPlaceholderText("Gemini API key (or set GEMINI_API_KEY env var)")
        self.yt_api_key_input.setText(os.environ.get("GEMINI_API_KEY", ""))
        yt_form.addRow("Gemini API Key:", self.yt_api_key_input)

        layout.addWidget(yt_settings)

        # Status
        self.yt_status_text = QPlainTextEdit()
        self.yt_status_text.setReadOnly(True)
        self.yt_status_text.setMaximumHeight(150)
        layout.addWidget(self.yt_status_text)

        # Start button
        btn_layout = QHBoxLayout()
        self.yt_start_btn = QPushButton("Download && Dub")
        self.yt_start_btn.setStyleSheet("font-weight: bold; padding: 8px 20px;")
        self.yt_start_btn.clicked.connect(self._start_youtube_dubbing)
        btn_layout.addWidget(self.yt_start_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()

    # ──────────────────────────────────────────────
    #  Tab 6: Model Manager
    # ──────────────────────────────────────────────

    def _build_model_manager_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "Model Manager")
        layout = QVBoxLayout(tab)

        self.model_status_text = QPlainTextEdit()
        self.model_status_text.setReadOnly(True)
        self.model_status_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.model_status_text)

        btn_layout = QHBoxLayout()
        scan_btn = QPushButton("Scan Models")
        scan_btn.clicked.connect(self._scan_models)
        btn_layout.addWidget(scan_btn)

        download_btn = QPushButton("Download Missing Models")
        download_btn.clicked.connect(self._download_missing_models)
        btn_layout.addWidget(download_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Auto-scan on first view
        self._scan_models()

    # ──────────────────────────────────────────────
    #  Tab 7: System Information
    # ──────────────────────────────────────────────

    def _build_system_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "System Info")
        layout = QVBoxLayout(tab)

        # Hardware info
        hw_group = QGroupBox("Hardware Detection")
        hw_layout = QVBoxLayout(hw_group)
        self.hw_info_text = QPlainTextEdit()
        self.hw_info_text.setReadOnly(True)
        self.hw_info_text.setFont(QFont("Consolas", 10))
        hw_layout.addWidget(self.hw_info_text)

        refresh_hw_btn = QPushButton("Refresh Hardware Info")
        refresh_hw_btn.clicked.connect(self._refresh_hardware_info)
        hw_layout.addWidget(refresh_hw_btn)
        layout.addWidget(hw_group)

        # Advanced settings
        adv_group = QGroupBox("Advanced Pipeline Settings")
        adv_form = QFormLayout(adv_group)

        self.emotion_check = QCheckBox("Enable emotion-aware narration")
        self.emotion_check.setChecked(True)
        adv_form.addRow(self.emotion_check)

        self.cinematic_check = QCheckBox("Enable cinematic narration optimization")
        self.cinematic_check.setChecked(True)
        adv_form.addRow(self.cinematic_check)

        self.ducking_check = QCheckBox("Enable audio ducking")
        self.ducking_check.setChecked(True)
        adv_form.addRow(self.ducking_check)

        layout.addWidget(adv_group)

        # About
        about_group = QGroupBox("About")
        about_layout = QVBoxLayout(about_group)
        about_text = QLabel(
            f"<b>{__app_name__} v{__version__}</b><br>"
            "Professional AI Documentary Dubbing Platform<br><br>"
            "An automated pipeline for converting videos into another language "
            "with natural narration and accurate timing.<br><br>"
            "<b>Pipeline:</b> Video Input → Audio Extraction → Audio Cleaning → "
            "Silence Detection → Segmentation → Speech Recognition → "
            "Speaker Detection → Translation → Emotion Detection → "
            "Voice Selection → TTS → Timing Alignment → Background Mixing → "
            "Subtitle Generation → Video Rendering<br><br>"
            "<b>Powered by:</b> Whisper, Gemini, Qwen3-TTS, Chatterbox, LuxTTS, Edge-TTS"
        )
        about_text.setWordWrap(True)
        about_layout.addWidget(about_text)
        layout.addWidget(about_group)

        # Auto-load hardware info
        self._refresh_hardware_info()

    # ──────────────────────────────────────────────
    #  File browsing helpers
    # ──────────────────────────────────────────────

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", VIDEO_FILTER)
        if path:
            self.video_path.setText(path)
            self._append_log(f"Selected video: {path}")

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_input.setText(path)

    def _browse_batch_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Video Directory")
        if path:
            self.batch_dir_input.setText(path)
            count = sum(
                1 for f in Path(path).iterdir()
                if f.suffix.lower() in VIDEO_EXTENSIONS
            )
            self._append_log(f"Batch directory: {path} ({count} video files found)")

    def _browse_clone_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Voice Sample", "", AUDIO_FILTER)
        if path:
            self.clone_audio_input.setText(path)

    # ──────────────────────────────────────────────
    #  Processing state management
    # ──────────────────────────────────────────────

    def _set_processing(self, active: bool) -> None:
        self._processing = active
        self.start_btn.setEnabled(not active)
        self.batch_start_btn.setEnabled(not active)
        self.yt_start_btn.setEnabled(not active)
        self.cancel_btn.setEnabled(active)
        if not active:
            self.progress_bar.setValue(0)
            self.progress_label.setText("Idle")

    def _cancel_processing(self) -> None:
        self._append_log("Cancellation requested. Will stop after the current stage.")
        self._set_processing(False)

    # ──────────────────────────────────────────────
    #  Signal handlers
    # ──────────────────────────────────────────────

    def _on_progress(self, pct: float, stage: str) -> None:
        self.progress_bar.setValue(int(pct))
        self.progress_label.setText(f"{pct:.0f}% - {stage}")

    def _on_finished(self, summary: str) -> None:
        self._set_processing(False)
        self._append_log(summary)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Complete")
        QMessageBox.information(self, "Dubbing Complete", summary)

    def _on_error(self, error: str) -> None:
        self._set_processing(False)
        self._append_log(f"ERROR: {error}")
        QMessageBox.critical(self, "Error", f"Processing failed:\n\n{error}")

    def _append_log(self, message: str) -> None:
        self.log_text.appendPlainText(message)

    # ──────────────────────────────────────────────
    #  Single video dubbing
    # ──────────────────────────────────────────────

    def _start_single_dubbing(self) -> None:
        video_path = self.video_path.text().strip()
        if not video_path:
            QMessageBox.warning(self, "Input Required", "Please select a video file.")
            return
        if not Path(video_path).exists():
            QMessageBox.critical(self, "File Not Found", f"Video file not found:\n{video_path}")
            return

        api_key = self.api_key_input.text().strip()
        if not api_key and not os.environ.get("GEMINI_API_KEY"):
            QMessageBox.warning(
                self, "API Key Required",
                "Please enter a Gemini API key for translation,\n"
                "or set the GEMINI_API_KEY environment variable.",
            )
            return

        self._set_processing(True)
        self._append_log("Starting dubbing pipeline...")

        lang_code = self.lang_combo.currentData()
        style = self.style_combo.currentText()
        whisper_model = self.whisper_combo.currentText()
        subtitle_fmt = self.subtitle_combo.currentText()
        embed_subs = self.embed_subs_check.isChecked()
        bg_vol = self.bg_volume_slider.value()
        output_dir = self.output_dir_input.text().strip() or "output"
        emotion_enabled = self.emotion_check.isChecked()
        cinematic_enabled = self.cinematic_check.isChecked()
        ducking_enabled = self.ducking_check.isChecked()

        def _run() -> None:
            try:
                from dubbing_studio.pipeline import DubbingPipeline

                config = self._get_config()
                if api_key:
                    config.translation.api_key = api_key
                config.whisper.model_size = whisper_model
                config.subtitle.embed_in_video = embed_subs
                config.subtitle.format = subtitle_fmt
                config.mixing.background_volume = bg_vol / 100.0
                config.voice.narrator_style = style
                config.emotion.enabled = emotion_enabled
                config.cinematic.enabled = cinematic_enabled
                config.mixing.ducking_enabled = ducking_enabled
                config.output_dir = output_dir

                pipeline = DubbingPipeline(config)

                def on_progress(stage: str, progress: float) -> None:
                    self._signals.progress.emit(progress * 100, stage)

                result = pipeline.process_video(
                    video_path=video_path,
                    target_language=lang_code,
                    narrator_style=style,
                    progress_callback=on_progress,
                    output_dir=output_dir,
                )

                summary = (
                    f"Dubbing Complete!\n"
                    f"Source: {result.source_language}\n"
                    f"Target: {result.target_language}\n"
                    f"Segments: {result.total_segments}\n"
                    f"Duration: {result.total_duration:.1f}s\n"
                    f"Processing Time: {result.processing_time:.1f}s\n"
                    f"Output Video: {result.output_video_path}\n"
                    f"Output Audio: {result.output_audio_path}"
                )
                self._signals.finished.emit(summary)

            except Exception as e:
                logger.error("Dubbing failed: %s", e, exc_info=True)
                self._signals.error.emit(str(e))

        threading.Thread(target=_run, daemon=True).start()

    # ──────────────────────────────────────────────
    #  Batch processing
    # ──────────────────────────────────────────────

    def _start_batch_processing(self) -> None:
        batch_dir = self.batch_dir_input.text().strip()
        if not batch_dir:
            QMessageBox.warning(self, "Input Required", "Please select a video directory.")
            return
        if not Path(batch_dir).is_dir():
            QMessageBox.critical(self, "Not Found", f"Directory not found:\n{batch_dir}")
            return

        api_key = self.api_key_input.text().strip()
        if not api_key and not os.environ.get("GEMINI_API_KEY"):
            QMessageBox.warning(self, "API Key Required", "Please enter a Gemini API key on the Single Video tab.")
            return

        self._set_processing(True)
        self._append_log("Starting batch processing...")

        lang_code = self.batch_lang_combo.currentData()
        style = self.batch_style_combo.currentText()
        max_concurrent = self.max_concurrent_spin.value()

        def _run() -> None:
            try:
                from dubbing_studio.batch.processor import BatchProcessor
                from dubbing_studio.pipeline import DubbingPipeline

                config = self._get_config()
                if api_key:
                    config.translation.api_key = api_key
                config.batch.max_concurrent = max_concurrent
                config.emotion.enabled = self.emotion_check.isChecked()
                config.cinematic.enabled = self.cinematic_check.isChecked()

                pipeline = DubbingPipeline(config)
                batch = BatchProcessor(config.batch)

                video_files = sorted([
                    str(f) for f in Path(batch_dir).iterdir()
                    if f.suffix.lower() in VIDEO_EXTENSIONS
                ])

                if not video_files:
                    self._signals.error.emit(f"No video files found in {batch_dir}")
                    return

                # Populate table
                self._signals.log.emit(f"Found {len(video_files)} video files")

                batch.add_videos(video_files, lang_code, style)

                def on_batch_progress(prog):
                    pct = prog.overall_progress * 100
                    desc = f"Batch: {prog.completed_jobs}/{prog.total_jobs} done"
                    self._signals.progress.emit(pct, desc)

                jobs = batch.process_all(
                    dubbing_function=pipeline.process_video_for_batch,
                    progress_callback=on_batch_progress,
                )

                lines = [f"Batch complete: {len(video_files)} videos"]
                for job in jobs:
                    name = Path(job.video_path).name
                    if job.status.value == "completed":
                        lines.append(f"  [DONE] {name} -> {job.output_path}")
                    else:
                        msg = job.error_message[:80] if job.error_message else "unknown"
                        lines.append(f"  [FAIL] {name}: {msg}")

                self._signals.finished.emit("\n".join(lines))

            except Exception as e:
                logger.error("Batch processing failed: %s", e, exc_info=True)
                self._signals.error.emit(str(e))

        threading.Thread(target=_run, daemon=True).start()

    # ──────────────────────────────────────────────
    #  Voice cloning
    # ──────────────────────────────────────────────

    def _create_voice_profile(self) -> None:
        audio_path = self.clone_audio_input.text().strip()
        profile_name = self.clone_name_input.text().strip()

        if not audio_path:
            QMessageBox.warning(self, "Input Required", "Please select a voice sample audio file.")
            return
        if not profile_name:
            QMessageBox.warning(self, "Input Required", "Please provide a name for the voice profile.")
            return

        def _run() -> None:
            try:
                from dubbing_studio.cloning.voice_cloner import VoiceCloner
                config = self._get_config()
                cloner = VoiceCloner(config.cloning)

                profile = cloner.create_voice_profile(
                    sample_path=audio_path,
                    name=profile_name,
                    gender=self.clone_gender_combo.currentText(),
                    language=self.clone_lang_combo.currentData(),
                    description=self.clone_desc_input.text().strip(),
                )

                summary = (
                    f"Voice profile created!\n"
                    f"ID: {profile.profile_id}\n"
                    f"Name: {profile.name}\n"
                    f"Duration: {profile.duration:.1f}s\n"
                    f"Gender: {profile.gender}\n"
                    f"Language: {profile.language}"
                )
                self._signals.finished.emit(summary)
            except Exception as e:
                self._signals.error.emit(f"Error creating voice profile: {e}")

        threading.Thread(target=_run, daemon=True).start()

    def _refresh_profiles(self) -> None:
        try:
            from dubbing_studio.cloning.voice_cloner import VoiceCloner
            config = self._get_config()
            cloner = VoiceCloner(config.cloning)
            profiles = cloner.profile_manager.list_profiles()

            if not profiles:
                self.profiles_text.setPlainText("No voice profiles saved yet.")
                return

            lines = ["Saved Voice Profiles:", "-" * 40]
            for p in profiles:
                lines.append(
                    f"  [{p.profile_id}] {p.name} ({p.gender}, {p.language}, {p.duration:.1f}s)"
                )
            self.profiles_text.setPlainText("\n".join(lines))
        except Exception as e:
            self.profiles_text.setPlainText(f"Error listing profiles: {e}")

    def _delete_voice_profile(self) -> None:
        profile_id = self.delete_profile_id_input.text().strip()
        if not profile_id:
            QMessageBox.warning(self, "Input Required", "Please enter a profile ID to delete.")
            return
        try:
            from dubbing_studio.cloning.voice_cloner import VoiceCloner
            config = self._get_config()
            cloner = VoiceCloner(config.cloning)
            success = cloner.profile_manager.delete_profile(profile_id)
            if success:
                self._append_log(f"Profile {profile_id} deleted.")
                self._refresh_profiles()
            else:
                QMessageBox.warning(self, "Not Found", f"Profile {profile_id} not found.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error deleting profile: {e}")

    # ──────────────────────────────────────────────
    #  YouTube dubbing
    # ──────────────────────────────────────────────

    def _start_youtube_dubbing(self) -> None:
        url = self.youtube_url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Input Required", "Please enter a YouTube URL.")
            return

        api_key = self.yt_api_key_input.text().strip()
        if not api_key and not os.environ.get("GEMINI_API_KEY"):
            QMessageBox.warning(self, "API Key Required", "Please provide a Gemini API key.")
            return

        self._set_processing(True)
        self.yt_status_text.setPlainText("Starting YouTube dubbing pipeline...")

        lang_code = self.yt_lang_combo.currentData()
        style = self.yt_style_combo.currentText()
        whisper_model = self.yt_whisper_combo.currentText()

        def _run() -> None:
            try:
                from dubbing_studio.pipeline import DubbingPipeline
                from dubbing_studio.youtube.pipeline import YouTubeDubbingPipeline

                config = self._get_config()
                if api_key:
                    config.translation.api_key = api_key
                config.whisper.model_size = whisper_model
                config.voice.narrator_style = style

                pipeline = DubbingPipeline(config)
                yt = YouTubeDubbingPipeline(config.youtube)

                self._signals.progress.emit(10, "Downloading video...")
                download_dir = str(Path(config.temp_dir) / "youtube")
                video_path, metadata = yt.download_video(url, download_dir)
                self._signals.log.emit(f"Downloaded: {metadata.title}")

                self._signals.progress.emit(20, "Processing dubbing pipeline...")

                def on_progress(stage: str, prog: float) -> None:
                    adjusted = 20 + prog * 70
                    self._signals.progress.emit(adjusted, stage)

                result = pipeline.process_video(
                    video_path=video_path,
                    target_language=lang_code,
                    narrator_style=style,
                    progress_callback=on_progress,
                )

                summary = (
                    f"YouTube Dubbing Complete!\n"
                    f"Title: {metadata.title}\n"
                    f"Source: {result.source_language}\n"
                    f"Target: {result.target_language}\n"
                    f"Segments: {result.total_segments}\n"
                    f"Processing Time: {result.processing_time:.1f}s\n"
                    f"Output: {result.output_video_path}"
                )
                self._signals.finished.emit(summary)

            except Exception as e:
                logger.error("YouTube dubbing failed: %s", e, exc_info=True)
                self._signals.error.emit(str(e))

        threading.Thread(target=_run, daemon=True).start()

    # ──────────────────────────────────────────────
    #  Model manager
    # ──────────────────────────────────────────────

    def _scan_models(self) -> None:
        def _run() -> None:
            try:
                from dubbing_studio.models.manager import ModelManager
                config = self._get_config()
                mgr = ModelManager(config.model_management)
                models = mgr.scan_models()

                lines = ["Model Status:", "=" * 50]
                for m in models:
                    status = "INSTALLED" if m.installed else "MISSING"
                    req = " (required)" if m.required else ""
                    size = f" [{m.size_mb}MB]" if m.size_mb > 0 else ""
                    lines.append(f"  [{status}] {m.name}{req}{size}")
                    lines.append(f"             {m.description}")

                cache_size = mgr.get_cache_size()
                lines.extend(["=" * 50, f"Cache size: {cache_size:.1f} MB"])

                self._signals.log.emit("\n".join(lines))
                # Update the text in the model tab directly
                self.model_status_text.setPlainText("\n".join(lines))
            except Exception as e:
                self.model_status_text.setPlainText(f"Error scanning models: {e}")

        threading.Thread(target=_run, daemon=True).start()

    def _download_missing_models(self) -> None:
        def _run() -> None:
            try:
                from dubbing_studio.models.manager import ModelManager
                config = self._get_config()
                mgr = ModelManager(config.model_management)
                missing = mgr.get_missing_required()

                if not missing:
                    self._signals.log.emit("All required models are installed!")
                    self.model_status_text.setPlainText("All required models are installed!")
                    return

                lines = [f"Downloading {len(missing)} missing models..."]
                for i, model in enumerate(missing):
                    self._signals.progress.emit(
                        (i + 1) / len(missing) * 100,
                        f"Installing {model.name}...",
                    )
                    success = mgr.download_model(model.name)
                    status = "OK" if success else "FAILED"
                    lines.append(f"  [{status}] {model.name}")

                self._signals.log.emit("\n".join(lines))
                self._scan_models()
            except Exception as e:
                self._signals.error.emit(f"Error downloading models: {e}")

        threading.Thread(target=_run, daemon=True).start()

    # ──────────────────────────────────────────────
    #  Hardware info
    # ──────────────────────────────────────────────

    def _refresh_hardware_info(self) -> None:
        def _detect() -> None:
            try:
                from dubbing_studio.hardware.optimizer import HardwareOptimizer
                optimizer = HardwareOptimizer()
                info = optimizer.detect_hardware()
                lines = [
                    f"Platform:             {info.platform}",
                    f"GPU Available:        {'Yes' if info.has_gpu else 'No'} ({info.gpu_name})",
                ]
                if info.has_gpu:
                    lines.append(f"GPU Memory:           {info.gpu_memory_mb} MB")
                lines.extend([
                    f"CPU Cores:            {info.cpu_count}",
                    f"RAM:                  {info.ram_gb:.1f} GB",
                    f"Recommended Whisper:  {info.recommended_whisper_model}",
                    f"Recommended Batch:    {info.recommended_batch_size}",
                ])
                self.hw_info_text.setPlainText("\n".join(lines))
            except Exception as e:
                self.hw_info_text.setPlainText(f"Hardware detection error: {e}")

        threading.Thread(target=_detect, daemon=True).start()

    # ──────────────────────────────────────────────
    #  About dialog
    # ──────────────────────────────────────────────

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About",
            f"<b>{__app_name__} v{__version__}</b><br><br>"
            "Professional AI Documentary Dubbing Platform<br><br>"
            "An automated pipeline for converting videos into another language "
            "with natural narration and accurate timing.<br><br>"
            "Powered by Whisper, Gemini, Qwen3-TTS, Chatterbox, LuxTTS, and Edge-TTS.",
        )


def main() -> None:
    """Launch the desktop application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = QApplication(sys.argv)
    app.setApplicationName(__app_name__)
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")

    window = DubbingStudioApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
