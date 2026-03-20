"""Setup configuration for Dubbing Studio."""

from setuptools import setup, find_packages

from dubbing_studio import __version__

setup(
    name="dubbing-studio",
    version=__version__,
    description="Professional AI Documentary Dubbing Platform",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Dubbing Studio Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "PyQt6",
        "gradio>=4.0.0",
        "openai-whisper>=20231117",
        "google-generativeai>=0.5.0",
        "edge-tts>=6.1.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "gpu": [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
        ],
        "qwen": [
            "transformers>=4.35.0",
            "torch>=2.0.0",
        ],
        "chatterbox": [
            "chatterbox-tts",
        ],
        "luxtts": [
            "TTS>=0.22.0",
        ],
        "all": [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "transformers>=4.35.0",
            "chatterbox-tts",
            "TTS>=0.22.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dubbing-studio=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Multimedia :: Video",
    ],
)
