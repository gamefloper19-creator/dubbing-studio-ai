"""
Build script for Dubbing Studio.

Creates the DubbingStudio executable using PyInstaller.

Usage:
    python build.py

Requirements:
    - PyInstaller >= 6.0
    - All project dependencies installed
    - FFmpeg must be available on the target system

Output:
    dist/DubbingStudio/DubbingStudio.exe (Windows)
    dist/DubbingStudio/DubbingStudio     (Linux/macOS)
"""

import os
import platform
import shutil
import subprocess
import sys


def check_prerequisites():
    """Check that all build prerequisites are met."""
    print("Checking prerequisites...")

    # Check Python version
    if sys.version_info < (3, 10):
        print(f"ERROR: Python 3.10+ required, found {sys.version}")
        return False

    # Check PyInstaller
    try:
        import PyInstaller
        print(f"  PyInstaller: {PyInstaller.__version__}")
    except ImportError:
        print("  PyInstaller: NOT FOUND - install with: pip install pyinstaller")
        return False

    # Check FFmpeg
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            print(f"  FFmpeg: {version_line}")
        else:
            print("  FFmpeg: NOT FOUND (required for audio/video processing)")
            return False
    except FileNotFoundError:
        print("  FFmpeg: NOT FOUND (required for audio/video processing)")
        return False

    # Check core dependencies
    deps = {
        "gradio": "gradio",
        "whisper": "openai-whisper",
        "google.generativeai": "google-generativeai",
        "edge_tts": "edge-tts",
        "psutil": "psutil",
    }

    for module, package in deps.items():
        try:
            __import__(module)
            print(f"  {package}: OK")
        except ImportError:
            print(f"  {package}: NOT FOUND - install with: pip install {package}")
            return False

    print("All prerequisites met.")
    return True


def clean_build():
    """Clean previous build artifacts."""
    print("Cleaning previous build artifacts...")
    for path in ["build", "dist"]:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"  Removed {path}/")


def build_executable():
    """Build the executable using PyInstaller."""
    print("Building DubbingStudio executable...")
    print(f"  Platform: {platform.system()} {platform.machine()}")

    spec_file = os.path.join(os.path.dirname(__file__), "DubbingStudio.spec")

    if not os.path.exists(spec_file):
        print(f"ERROR: Spec file not found: {spec_file}")
        return False

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        spec_file,
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))

    if result.returncode != 0:
        print("ERROR: PyInstaller build failed!")
        return False

    # Check output
    exe_name = "DubbingStudio.exe" if platform.system() == "Windows" else "DubbingStudio"
    exe_path = os.path.join("dist", "DubbingStudio", exe_name)

    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"  Build successful: {exe_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"ERROR: Expected output not found: {exe_path}")
        return False


def main():
    """Main build entry point."""
    print("=" * 50)
    print("Dubbing Studio - Build Script")
    print("=" * 50)

    if not check_prerequisites():
        print("\nBuild aborted: prerequisites not met.")
        sys.exit(1)

    clean_build()

    if build_executable():
        print("\n" + "=" * 50)
        print("BUILD SUCCESSFUL")
        print("=" * 50)
        exe_name = "DubbingStudio.exe" if platform.system() == "Windows" else "DubbingStudio"
        print(f"Executable: dist/DubbingStudio/{exe_name}")
        print("\nNote: FFmpeg must be installed on the target system.")
        print("      Download from: https://ffmpeg.org/download.html")
    else:
        print("\nBUILD FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
