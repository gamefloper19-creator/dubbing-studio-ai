"""
Smart translation engine using Google Gemini or compatible providers.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dubbing_studio.config import TranslationConfig, SUPPORTED_LANGUAGES
from dubbing_studio.speech.recognizer import TranscriptionSegment

logger = logging.getLogger(__name__)


@dataclass
class TranslatedSegment:
    """A translated segment with timing information."""
    segment_id: str
    start_time: float
    end_time: float
    original_text: str
    translated_text: str
    source_language: str
    target_language: str


class Translator:
    """
    Smart translation engine focused on natural narration style.

    Uses semantic translation rather than word-for-word translation
    to maintain natural storytelling tone.
    """

    def __init__(self, config: Optional[TranslationConfig] = None):
        self.config = config or TranslationConfig()
        self._client = None
        self._cache: dict[str, str] = {}
        self._cache_dir = Path("cache/translations")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_cache()

    def _cache_key(self, text: str, target_language: str, source_language: str = "") -> str:
        """Generate a deterministic cache key for a translation request."""
        raw = f"{source_language}|{target_language}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        """Load the translation cache from disk."""
        cache_file = self._cache_dir / "cache.json"
        if cache_file.exists():
            try:
                self._cache = json.loads(cache_file.read_text(encoding="utf-8"))
                logger.debug("Loaded %d cached translations", len(self._cache))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load translation cache: %s", e)
                self._cache = {}

    def _save_cache(self) -> None:
        """Persist the translation cache to disk."""
        cache_file = self._cache_dir / "cache.json"
        try:
            cache_file.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Failed to save translation cache: %s", e)

    def _init_client(self) -> None:
        """Initialize the translation API client."""
        if self.config.provider == "gemini":
            self._init_gemini()
        else:
            raise ValueError(f"Unsupported translation provider: {self.config.provider}")

    def _init_gemini(self) -> None:
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Google Generative AI SDK not installed. "
                "Install with: pip install google-generativeai"
            )

        if not self.config.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable "
                "or pass api_key in TranslationConfig."
            )

        genai.configure(api_key=self.config.api_key)
        self._client = genai.GenerativeModel(
            model_name=self.config.model,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": 2048,
            },
        )
        logger.info("Gemini translation client initialized (model=%s)", self.config.model)

    def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: str = "",
        context: str = "",
    ) -> str:
        """
        Translate text with natural narration style.

        Args:
            text: Text to translate.
            target_language: Target language code (e.g., 'hi', 'es').
            source_language: Source language code (auto-detected if empty).
            context: Additional context for better translation.

        Returns:
            Translated text.
        """
        if self._client is None:
            self._init_client()

        target_lang_name = SUPPORTED_LANGUAGES.get(target_language, target_language)
        source_lang_name = SUPPORTED_LANGUAGES.get(source_language, source_language)

        prompt = self._build_translation_prompt(
            text, target_lang_name, source_lang_name, context
        )

        # Check cache first
        key = self._cache_key(text, target_language, source_language)
        if key in self._cache:
            logger.debug("Translation cache hit for: '%s'", text[:50])
            return self._cache[key]

        for attempt in range(self.config.max_retries):
            try:
                response = self._client.generate_content(prompt)
                translated = response.text.strip()

                # Clean up any quotation marks or metadata the model may add
                translated = self._clean_translation(translated)

                # Store in cache
                self._cache[key] = translated
                self._save_cache()

                logger.debug(
                    "Translated: '%s' -> '%s'",
                    text[:50], translated[:50],
                )
                return translated

            except Exception as e:
                logger.warning(
                    "Translation attempt %d failed: %s",
                    attempt + 1, str(e),
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
                else:
                    raise RuntimeError(
                        f"Translation failed after {self.config.max_retries} attempts: {e}"
                    ) from e

        return text  # fallback to original

    def translate_segments(
        self,
        segments: list[TranscriptionSegment],
        target_language: str,
    ) -> list[TranslatedSegment]:
        """
        Translate a list of transcription segments.

        Maintains context between segments for coherent translation.

        Args:
            segments: List of TranscriptionSegment objects.
            target_language: Target language code.

        Returns:
            List of TranslatedSegment objects.
        """
        if not segments:
            return []

        translated_segments = []
        context_window = []  # Keep recent segments for context

        for i, seg in enumerate(segments):
            # Build context from surrounding segments
            context = self._build_context(segments, i, context_window)

            translated_text = self.translate_text(
                text=seg.text,
                target_language=target_language,
                source_language=seg.language,
                context=context,
            )

            translated_segments.append(TranslatedSegment(
                segment_id=seg.segment_id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                original_text=seg.text,
                translated_text=translated_text,
                source_language=seg.language,
                target_language=target_language,
            ))

            # Update context window (keep last 3 translations)
            context_window.append(translated_text)
            if len(context_window) > 3:
                context_window.pop(0)

            # Rate limiting for API calls
            if i < len(segments) - 1:
                time.sleep(0.5)

        logger.info(
            "Translated %d segments to %s",
            len(translated_segments), target_language,
        )
        return translated_segments

    def _build_translation_prompt(
        self,
        text: str,
        target_language: str,
        source_language: str,
        context: str,
    ) -> str:
        """Build the translation prompt with narration style instructions."""
        parts = [self.config.system_prompt, ""]

        if source_language:
            parts.append(f"Source language: {source_language}")
        parts.append(f"Target language: {target_language}")

        if context:
            parts.append(f"\nContext from surrounding narration:\n{context}")

        parts.append(f"\nText to translate:\n{text}")
        parts.append(
            "\nProvide ONLY the translated text, nothing else. "
            "No explanations, no quotation marks, no labels."
        )

        return "\n".join(parts)

    def _build_context(
        self,
        segments: list[TranscriptionSegment],
        current_index: int,
        previous_translations: list[str],
    ) -> str:
        """Build context string from surrounding segments."""
        context_parts = []

        # Add previous translations for continuity
        if previous_translations:
            recent = previous_translations[-2:]
            context_parts.append(
                "Previous translated lines: "
                + " | ".join(recent)
            )

        # Add next original segment for forward context
        if current_index + 1 < len(segments):
            next_text = segments[current_index + 1].text
            context_parts.append(f"Next line (original): {next_text}")

        return "\n".join(context_parts)

    def _clean_translation(self, text: str) -> str:
        """Clean up translated text, removing unwanted artifacts."""
        # Remove surrounding quotes
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1]

        # Remove common LLM prefixes
        prefixes_to_remove = [
            "Translation:",
            "Translated text:",
            "Here is the translation:",
            "Here's the translation:",
        ]
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        return text.strip()

    def batch_translate(
        self,
        texts: list[str],
        target_language: str,
        source_language: str = "",
    ) -> list[str]:
        """
        Translate multiple texts in batch, sending them as a single request
        for efficiency.

        Args:
            texts: List of texts to translate.
            target_language: Target language code.
            source_language: Source language code.

        Returns:
            List of translated texts.
        """
        if not texts:
            return []

        if self._client is None:
            self._init_client()

        target_lang_name = SUPPORTED_LANGUAGES.get(target_language, target_language)

        # Build batch prompt
        numbered_texts = "\n".join(
            f"[{i+1}] {text}" for i, text in enumerate(texts)
        )

        prompt = (
            f"{self.config.system_prompt}\n\n"
            f"Target language: {target_lang_name}\n\n"
            f"Translate each numbered line below. "
            f"Return ONLY the translations, one per line, with the same numbering.\n\n"
            f"{numbered_texts}"
        )

        try:
            response = self._client.generate_content(prompt)
            result_text = response.text.strip()

            # Parse numbered results
            translations = []
            for line in result_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Remove numbering prefix like [1], 1., 1), etc.
                for prefix_format in [f"[{len(translations)+1}]", f"{len(translations)+1}.", f"{len(translations)+1})"]:
                    if line.startswith(prefix_format):
                        line = line[len(prefix_format):].strip()
                        break
                translations.append(self._clean_translation(line))

            # Ensure we have the right number of translations
            while len(translations) < len(texts):
                # Fallback: translate missing ones individually
                idx = len(translations)
                translated = self.translate_text(
                    texts[idx], target_language, source_language
                )
                translations.append(translated)

            return translations[:len(texts)]

        except Exception as e:
            logger.warning("Batch translation failed, falling back to individual: %s", e)
            return [
                self.translate_text(text, target_language, source_language)
                for text in texts
            ]
