"""
Auto YouTube Dubbing Pipeline.

Provides end-to-end workflow for downloading YouTube videos,
processing through the dubbing pipeline, and optionally uploading
the dubbed version back to YouTube.
"""

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from dubbing_studio.config import YouTubeConfig

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """YouTube video metadata."""
    title: str = ""
    description: str = ""
    channel: str = ""
    duration: float = 0.0
    upload_date: str = ""
    language: str = ""
    tags: list[str] = field(default_factory=list)
    thumbnail_url: str = ""
    video_id: str = ""


@dataclass
class YouTubeDubbingResult:
    """Result of a YouTube dubbing operation."""
    input_url: str
    downloaded_path: str = ""
    dubbed_path: str = ""
    subtitle_paths: list[str] = field(default_factory=list)
    metadata: Optional[VideoMetadata] = None
    translated_metadata: Optional[dict] = None
    upload_url: str = ""
    success: bool = False
    error: str = ""


class YouTubeDubbingPipeline:
    """
    End-to-end YouTube dubbing workflow.

    1. Download video from URL
    2. Extract metadata
    3. Process through dubbing pipeline
    4. Export dubbed version
    5. Optionally upload to YouTube
    """

    def __init__(self, config: Optional[YouTubeConfig] = None):
        self.config = config or YouTubeConfig()

    def download_video(
        self,
        url: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> tuple[str, VideoMetadata]:
        """
        Download a YouTube video.

        Args:
            url: YouTube video URL.
            output_dir: Directory to save downloaded video.
            progress_callback: Optional callback for progress updates.

        Returns:
            Tuple of (downloaded file path, video metadata).

        Raises:
            RuntimeError: If download fails.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Extract video metadata first
        metadata = self._extract_metadata(url)

        if progress_callback:
            progress_callback(0.1, "Downloading video...")

        # Determine quality format
        format_spec = self._get_format_spec()

        # Safe filename from title
        safe_title = re.sub(r'[^\w\s-]', '', metadata.title or "video")
        safe_title = re.sub(r'\s+', '_', safe_title)[:100]

        output_path = str(Path(output_dir) / f"{safe_title}.mp4")

        # Download using yt-dlp
        cmd = [
            "yt-dlp",
            "--format", format_spec,
            "--merge-output-format", "mp4",
            "--output", output_path,
            "--no-playlist",
            url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Video download failed: {result.stderr}")

        # Find the actual output file (yt-dlp may modify the name)
        if not Path(output_path).exists():
            # Look for files matching the pattern
            possible_files = list(Path(output_dir).glob(f"{safe_title}*"))
            if possible_files:
                output_path = str(possible_files[0])
            else:
                raise RuntimeError("Downloaded file not found")

        if progress_callback:
            progress_callback(1.0, "Download complete")

        logger.info("Downloaded video: %s -> %s", url, output_path)

        return output_path, metadata

    def _extract_metadata(self, url: str) -> VideoMetadata:
        """Extract video metadata using yt-dlp."""
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning("Metadata extraction failed: %s", result.stderr[:200])
            return VideoMetadata(video_id=self._extract_video_id(url))

        try:
            info = json.loads(result.stdout)

            return VideoMetadata(
                title=info.get("title", ""),
                description=info.get("description", ""),
                channel=info.get("channel", info.get("uploader", "")),
                duration=float(info.get("duration", 0)),
                upload_date=info.get("upload_date", ""),
                language=info.get("language", ""),
                tags=info.get("tags", []) or [],
                thumbnail_url=info.get("thumbnail", ""),
                video_id=info.get("id", self._extract_video_id(url)),
            )
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to parse metadata: %s", e)
            return VideoMetadata(video_id=self._extract_video_id(url))

    def translate_metadata(
        self,
        metadata: VideoMetadata,
        translator: Any,
        target_language: str,
    ) -> dict:
        """
        Translate video metadata for the dubbed version.

        Args:
            metadata: Original video metadata.
            translator: Translator instance for translation.
            target_language: Target language code.

        Returns:
            Dict with translated title, description, and tags.
        """
        translated = {}

        try:
            if metadata.title:
                translated["title"] = translator.translate_text(
                    metadata.title, target_language
                )

            if metadata.description:
                # Translate first 500 chars of description
                desc_text = metadata.description[:500]
                translated["description"] = translator.translate_text(
                    desc_text, target_language
                )

            if metadata.tags:
                translated["tags"] = [
                    translator.translate_text(tag, target_language)
                    for tag in metadata.tags[:10]
                ]

        except Exception as e:
            logger.warning("Metadata translation failed: %s", e)
            translated.setdefault("title", metadata.title)
            translated.setdefault("description", metadata.description)
            translated.setdefault("tags", metadata.tags)

        return translated

    def upload_to_youtube(
        self,
        video_path: str,
        metadata: dict,
        subtitle_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """
        Upload dubbed video to YouTube.

        Args:
            video_path: Path to dubbed video file.
            metadata: Video metadata (title, description, tags).
            subtitle_path: Optional path to subtitle file.
            progress_callback: Optional progress callback.

        Returns:
            URL of uploaded video.

        Raises:
            RuntimeError: If upload fails or API is not configured.
        """
        if not self.config.upload_enabled:
            raise RuntimeError(
                "YouTube upload is not enabled. "
                "Set upload_enabled=True and provide API credentials."
            )

        if not self.config.client_secrets_file:
            raise RuntimeError(
                "YouTube API client secrets file not configured. "
                "Set client_secrets_file in YouTubeConfig."
            )

        if progress_callback:
            progress_callback(0.1, "Preparing upload...")

        try:
            return self._upload_with_api(video_path, metadata, subtitle_path)
        except ImportError:
            raise RuntimeError(
                "google-api-python-client is required for YouTube upload. "
                "Install with: pip install google-api-python-client google-auth-oauthlib"
            )

    def _upload_with_api(
        self,
        video_path: str,
        metadata: dict,
        subtitle_path: Optional[str],
    ) -> str:
        """Upload video using YouTube Data API v3."""
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google_auth_oauthlib.flow import InstalledAppFlow

        scopes = ["https://www.googleapis.com/auth/youtube.upload"]

        flow = InstalledAppFlow.from_client_secrets_file(
            self.config.client_secrets_file, scopes
        )
        credentials = flow.run_local_server(port=0)

        youtube = build("youtube", "v3", credentials=credentials)

        body = {
            "snippet": {
                "title": metadata.get("title", "Dubbed Video"),
                "description": metadata.get("description", ""),
                "tags": metadata.get("tags", []),
                "categoryId": "22",  # People & Blogs
            },
            "status": {
                "privacyStatus": "private",  # Upload as private by default
            },
        }

        media = MediaFileUpload(
            video_path, mimetype="video/mp4", resumable=True
        )

        request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media,
        )

        response = request.execute()
        video_id = response["id"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # Upload subtitles if provided
        if subtitle_path and self.config.upload_subtitles and Path(subtitle_path).exists():
            self._upload_subtitles(youtube, video_id, subtitle_path)

        logger.info("Uploaded video to YouTube: %s", video_url)
        return video_url

    def _upload_subtitles(
        self,
        youtube: Any,
        video_id: str,
        subtitle_path: str,
    ) -> None:
        """Upload subtitles to a YouTube video."""
        try:
            from googleapiclient.http import MediaFileUpload

            body = {
                "snippet": {
                    "videoId": video_id,
                    "language": "en",
                    "name": "Dubbed Subtitles",
                },
            }

            media = MediaFileUpload(subtitle_path, mimetype="application/x-subrip")

            youtube.captions().insert(
                part="snippet",
                body=body,
                media_body=media,
            ).execute()

            logger.info("Uploaded subtitles for video %s", video_id)

        except Exception as e:
            logger.warning("Subtitle upload failed: %s", e)

    def _get_format_spec(self) -> str:
        """Get yt-dlp format specification based on quality setting."""
        quality_map = {
            "best": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "1080p": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best",
            "720p": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best",
            "480p": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]/best",
        }
        return quality_map.get(self.config.download_quality, quality_map["best"])

    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from various YouTube URL formats."""
        patterns = [
            r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'(?:embed/)([a-zA-Z0-9_-]{11})',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return ""
