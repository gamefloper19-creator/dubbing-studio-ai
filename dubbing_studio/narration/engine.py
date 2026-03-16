"""
Cinematic Narration Engine.

Optimizes narration by merging short segments into natural paragraphs,
smoothing sentence transitions, and adjusting pacing for documentary storytelling.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from dubbing_studio.config import CinematicNarrationConfig
from dubbing_studio.translation.translator import TranslatedSegment

logger = logging.getLogger(__name__)


@dataclass
class NarrationBlock:
    """A merged narration block combining multiple segments."""
    block_id: str
    segments: list[TranslatedSegment]
    merged_text: str
    start_time: float
    end_time: float
    duration: float
    word_count: int
    estimated_wpm: float
    pause_after: float  # recommended pause after this block


class CinematicNarrationEngine:
    """
    Optimize narration for professional documentary storytelling.

    Features:
    - Merge short segments into natural paragraphs
    - Smooth sentence transitions
    - Adjust pacing to match documentary style
    - Avoid unnatural pauses or abrupt transitions
    """

    def __init__(self, config: Optional[CinematicNarrationConfig] = None):
        self.config = config or CinematicNarrationConfig()

    def optimize_narration(
        self,
        segments: list[TranslatedSegment],
    ) -> list[TranslatedSegment]:
        """
        Optimize translated segments for cinematic narration.

        Args:
            segments: List of translated segments.

        Returns:
            Optimized list of translated segments.
        """
        if not segments:
            return segments

        optimized = list(segments)

        # Step 1: Merge short segments into natural paragraphs
        if self.config.merge_short_segments:
            optimized = self._merge_short_segments(optimized)

        # Step 2: Smooth sentence transitions
        if self.config.smooth_transitions:
            optimized = self._smooth_transitions(optimized)

        # Step 3: Optimize pacing
        if self.config.optimize_pacing:
            optimized = self._optimize_pacing(optimized)

        logger.info(
            "Narration optimized: %d segments -> %d segments",
            len(segments), len(optimized),
        )

        return optimized

    def _merge_short_segments(
        self,
        segments: list[TranslatedSegment],
    ) -> list[TranslatedSegment]:
        """
        Merge very short segments into longer natural paragraphs.

        Short segments (< min_paragraph_duration) are combined with
        adjacent segments to create flowing narration blocks.
        """
        if len(segments) <= 1:
            return segments

        min_dur = self.config.min_paragraph_duration
        merged = []
        current_group: list[TranslatedSegment] = [segments[0]]

        for seg in segments[1:]:
            current_duration = current_group[-1].end_time - current_group[0].start_time
            seg_duration = seg.end_time - seg.start_time

            # Check gap between segments
            gap = seg.start_time - current_group[-1].end_time

            # Merge if current group is short and gap is small
            if (current_duration < min_dur and gap < 2.0) or (seg_duration < 1.5 and gap < 1.0):
                current_group.append(seg)
            else:
                # Finalize current group
                merged.append(self._create_merged_segment(current_group))
                current_group = [seg]

        # Finalize last group
        if current_group:
            merged.append(self._create_merged_segment(current_group))

        return merged

    def _create_merged_segment(
        self,
        group: list[TranslatedSegment],
    ) -> TranslatedSegment:
        """Create a single merged segment from a group of segments."""
        if len(group) == 1:
            return group[0]

        # Combine texts with proper spacing
        merged_original = " ".join(seg.original_text for seg in group)
        merged_translated = self._join_sentences(
            [seg.translated_text for seg in group]
        )

        return TranslatedSegment(
            segment_id=group[0].segment_id,
            start_time=group[0].start_time,
            end_time=group[-1].end_time,
            original_text=merged_original,
            translated_text=merged_translated,
            source_language=group[0].source_language,
            target_language=group[0].target_language,
        )

    def _join_sentences(self, texts: list[str]) -> str:
        """
        Join multiple text segments into a flowing paragraph.

        Handles proper punctuation and spacing between sentences.
        """
        if not texts:
            return ""
        if len(texts) == 1:
            return texts[0]

        result_parts = []
        for text in texts:
            text = text.strip()
            if not text:
                continue

            # Add proper sentence ending if missing
            if result_parts and not result_parts[-1][-1] in ".!?;:,":
                result_parts[-1] += "."

            result_parts.append(text)

        return " ".join(result_parts)

    def _smooth_transitions(
        self,
        segments: list[TranslatedSegment],
    ) -> list[TranslatedSegment]:
        """
        Smooth transitions between segments.

        Removes abrupt text transitions and ensures natural flow
        between narration segments.
        """
        if len(segments) <= 1:
            return segments

        smoothed = []

        for i, seg in enumerate(segments):
            text = seg.translated_text.strip()

            # Remove redundant sentence starters that break flow
            if i > 0:
                text = self._remove_redundant_connectors(text)

            # Ensure proper sentence endings
            if text and text[-1] not in ".!?":
                text += "."

            smoothed.append(TranslatedSegment(
                segment_id=seg.segment_id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                original_text=seg.original_text,
                translated_text=text,
                source_language=seg.source_language,
                target_language=seg.target_language,
            ))

        return smoothed

    def _remove_redundant_connectors(self, text: str) -> str:
        """Remove redundant transition words at the start of segments."""
        redundant_patterns = [
            r"^(And then,?\s+)",
            r"^(So,?\s+)",
            r"^(Well,?\s+)",
            r"^(Now,?\s+)",
            r"^(You see,?\s+)",
            r"^(As I was saying,?\s+)",
        ]

        for pattern in redundant_patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                # Only remove if the remaining text is a complete sentence
                remaining = text[match.end():]
                if len(remaining) > 10 and remaining[0].isalpha():
                    text = remaining[0].upper() + remaining[1:]
                break

        return text

    def _optimize_pacing(
        self,
        segments: list[TranslatedSegment],
    ) -> list[TranslatedSegment]:
        """
        Optimize pacing for documentary narration.

        Adjusts segment timing to target a natural WPM rate
        suitable for documentary storytelling.
        """
        target_wpm = self.config.target_wpm
        optimized = []

        for seg in segments:
            text = seg.translated_text
            word_count = len(text.split())
            duration = seg.end_time - seg.start_time

            if duration <= 0 or word_count == 0:
                optimized.append(seg)
                continue

            current_wpm = (word_count / duration) * 60
            transition_pause = self.config.transition_pause

            # If speaking too fast, the timing alignment will handle speed
            # But we can adjust timing hints
            if current_wpm > target_wpm * 1.3:
                # Suggest slightly longer duration by adjusting end time
                ideal_duration = (word_count / target_wpm) * 60
                # Don't exceed a reasonable stretch
                new_end = min(
                    seg.start_time + ideal_duration,
                    seg.end_time + transition_pause,
                )
                optimized.append(TranslatedSegment(
                    segment_id=seg.segment_id,
                    start_time=seg.start_time,
                    end_time=new_end,
                    original_text=seg.original_text,
                    translated_text=text,
                    source_language=seg.source_language,
                    target_language=seg.target_language,
                ))
            else:
                optimized.append(seg)

        return optimized

    def analyze_narration_blocks(
        self,
        segments: list[TranslatedSegment],
    ) -> list[NarrationBlock]:
        """
        Analyze segments and create narration blocks with metadata.

        Useful for UI display and detailed narration planning.
        """
        blocks = []

        for i, seg in enumerate(segments):
            word_count = len(seg.translated_text.split())
            duration = seg.end_time - seg.start_time
            wpm = (word_count / duration * 60) if duration > 0 else 0

            # Calculate recommended pause after block
            if i < len(segments) - 1:
                gap = segments[i + 1].start_time - seg.end_time
                pause_after = max(self.config.transition_pause, gap)
            else:
                pause_after = 0.0

            blocks.append(NarrationBlock(
                block_id=seg.segment_id,
                segments=[seg],
                merged_text=seg.translated_text,
                start_time=seg.start_time,
                end_time=seg.end_time,
                duration=duration,
                word_count=word_count,
                estimated_wpm=round(wpm, 1),
                pause_after=round(pause_after, 3),
            ))

        return blocks
