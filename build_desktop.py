#!/usr/bin/env python3
"""
Build script for DubbingStudio.exe

Builds the standalone Windows desktop application using PyInstaller.

Usage:
    python build_desktop.py                # Build with spec file
    python build_desktop.py --onedir       # Build as one-directory bundle
    python build_desktop.py --clean        # Clean previous builds first

Requirements:
    pip install pyinstaller
    (or: pip install dubbing-studio[build])
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def check_pyinstaller() -> bool:
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller  # noqa: F401
        return True
    except ImportError:
        return False


def check_pyside6() -> bool:
    """Check if PySide6 is installed."""
    try:
        import PySide6  # noqa: F401
        return True
    except ImportError:
        return False


def clean_build(project_dir: Path) -> None:
    """Remove previous build artifacts."""
    for d in ("build", "dist"):
        target = project_dir / d
        if target.exists():
            print(f"  Removing {target}")
            shutil.rmtree(target)
    spec_build = project_dir / "__pycache__"
    if spec_build.exists():
        shutil.rmtree(spec_build)


def build_with_spec(project_dir: Path) -> int:
    """Build using the spec file (one-file bundle)."""
    spec = project_dir / "DubbingStudio.spec"
    if not spec.exists():
        print(f"ERROR: Spec file not found: {spec}")
        return 1

    cmd = [sys.executable, "-m", "PyInstaller", str(spec)]
    print(f"  Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(project_dir))


def build_onedir(project_dir: Path) -> int:
    """Build as a one-directory bundle."""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "DubbingStudio",
        "--windowed",
        "--noconfirm",
        "--add-data", "dubbing_studio:dubbing_studio",
        "--hidden-import", "PySide6.QtCore",
        "--hidden-import", "PySide6.QtGui",
        "--hidden-import", "PySide6.QtWidgets",
        "--hidden-import", "dubbing_studio.pipeline",
        "--hidden-import", "dubbing_studio.config",
        "desktop_app.py",
    ]
    print(f"  Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(project_dir))


def main() -> None:
    """Build the desktop application."""
    parser = argparse.ArgumentParser(description="Build DubbingStudio.exe")
    parser.add_argument("--onedir", action="store_true", help="Build as one-directory bundle")
    parser.add_argument("--clean", action="store_true", help="Clean previous builds first")
    args = parser.parse_args()

    project_dir = Path(__file__).parent

    print("DubbingStudio Build")
    print("=" * 40)

    # Dependency checks
    if not check_pyinstaller():
        print("ERROR: PyInstaller is not installed.")
        print("  Install with: pip install pyinstaller")
        sys.exit(1)

    if not check_pyside6():
        print("ERROR: PySide6 is not installed.")
        print("  Install with: pip install PySide6")
        sys.exit(1)

    if args.clean:
        print("\nCleaning previous builds...")
        clean_build(project_dir)

    print("\nBuilding DubbingStudio.exe...")
    if args.onedir:
        rc = build_onedir(project_dir)
    else:
        rc = build_with_spec(project_dir)

    if rc == 0:
        dist = project_dir / "dist"
        print(f"\nBuild complete! Output: {dist}")
        if (dist / "DubbingStudio.exe").exists():
            size = (dist / "DubbingStudio.exe").stat().st_size / (1024 * 1024)
            print(f"  DubbingStudio.exe ({size:.1f} MB)")
        elif (dist / "DubbingStudio").exists():
            print("  DubbingStudio/ (one-directory bundle)")
    else:
        print(f"\nBuild failed with exit code {rc}")
        sys.exit(rc)


if __name__ == "__main__":
    main()
