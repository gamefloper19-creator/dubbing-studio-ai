from pathlib import Path

from dubbing_studio.subtitle.generator import SubtitleGenerator
from dubbing_studio.translation.translator import TranslatedSegment


def test_subtitle_generation_all_formats(tmp_path: Path) -> None:
    gen = SubtitleGenerator()
    segments = [
        TranslatedSegment(
            segment_id="001",
            start_time=0.0,
            end_time=1.23,
            original_text="Hello",
            translated_text="Hola",
            source_language="en",
            target_language="es",
        )
    ]

    out_dir = tmp_path / "subs"
    paths = gen.generate_all_formats(segments, str(out_dir), "test")

    srt_path = Path(paths["srt"])
    vtt_path = Path(paths["vtt"])
    ass_path = Path(paths["ass"])

    assert srt_path.exists()
    assert vtt_path.exists()
    assert ass_path.exists()

    srt_text = srt_path.read_text(encoding="utf-8")
    vtt_text = vtt_path.read_text(encoding="utf-8")
    ass_text = ass_path.read_text(encoding="utf-8")

    assert "00:00:00,000" in srt_text
    assert "Hola" in srt_text
    assert vtt_text.startswith("WEBVTT")
    assert "Dialogue:" in ass_text
