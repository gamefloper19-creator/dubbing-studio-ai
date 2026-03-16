"""
PyInstaller Build Script for Dubbing Studio.

Creates a standalone Windows executable (DubbingStudio.exe).

Usage:
    python build.py          # Build the executable
    python build.py --clean  # Clean build artifacts first
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


APP_NAME = "DubbingStudio"
MAIN_SCRIPT = "main.py"
ICON_PATH = None  # Set to .ico path if available


def clean_build():
    """Remove previous build artifacts."""
    dirs_to_clean = ["build", "dist", f"{APP_NAME}.spec"]
    for d in dirs_to_clean:
        path = Path(d)
        if path.is_dir():
            shutil.rmtree(path)
            print(f"Cleaned: {d}")
        elif path.is_file():
            path.unlink()
            print(f"Cleaned: {d}")


def build_executable():
    """Build the standalone executable using PyInstaller."""
    print(f"Building {APP_NAME}...")
    print("=" * 50)

    # Ensure PyInstaller is available
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Build command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--onedir",  # Create a directory with all dependencies
        "--windowed",  # No console window (GUI app)
        "--noconfirm",

        # Include all dubbing_studio packages
        "--collect-all", "dubbing_studio",

        # Include data files
        "--add-data", f"dubbing_studio{os.pathsep}dubbing_studio",

        # Hidden imports for dynamic loading
        "--hidden-import", "gradio",
        "--hidden-import", "edge_tts",
        "--hidden-import", "whisper",
        "--hidden-import", "google.generativeai",
        "--hidden-import", "psutil",
        "--hidden-import", "dubbing_studio.emotion.analyzer",
        "--hidden-import", "dubbing_studio.cloning.voice_cloner",
        "--hidden-import", "dubbing_studio.diarization.speaker_detector",
        "--hidden-import", "dubbing_studio.narration.engine",
        "--hidden-import", "dubbing_studio.timing.advanced_aligner",
        "--hidden-import", "dubbing_studio.youtube.pipeline",
        "--hidden-import", "dubbing_studio.models.manager",
        "--hidden-import", "dubbing_studio.tts.voice_library",

        # Exclude unnecessary modules to reduce size
        "--exclude-module", "matplotlib",
        "--exclude-module", "tkinter",
        "--exclude-module", "unittest",
        "--exclude-module", "test",
    ]

    # Add icon if available
    if ICON_PATH and Path(ICON_PATH).exists():
        cmd.extend(["--icon", ICON_PATH])

    # Main script
    cmd.append(MAIN_SCRIPT)

    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    result = subprocess.run(cmd)

    if result.returncode == 0:
        dist_path = Path("dist") / APP_NAME
        print("\n" + "=" * 50)
        print(f"Build successful!")
        print(f"Output: {dist_path.resolve()}")
        print(f"Executable: {dist_path / f'{APP_NAME}.exe'}")

        # Calculate total size
        total_size = sum(
            f.stat().st_size for f in dist_path.rglob("*") if f.is_file()
        ) if dist_path.exists() else 0
        print(f"Total size: {total_size / (1024 * 1024):.1f} MB")
        print("=" * 50)
    else:
        print("\nBuild failed!")
        sys.exit(1)


def main():
    if "--clean" in sys.argv:
        clean_build()

    if "--clean-only" in sys.argv:
        return

    build_executable()


if __name__ == "__main__":
    main()
