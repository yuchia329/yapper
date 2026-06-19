"""s11 generates a valid ASS on the voiceover timeline (cues abut, CJK font set)."""

from pathlib import Path

from jieshuoforge.schemas import Edl, EdlSegment
from jieshuoforge.stages import s11_subs


def _edl() -> Edl:
    return Edl(
        fps=30, width=1920, height=1080,
        segments=[
            EdlSegment(segment_id="seg_000", line_id="l0", src_in=0, src_out=4, vo_file="/v/0.wav", vo_duration=4.0, subtitle_text="第一句"),
            EdlSegment(segment_id="seg_001", line_id="l1", src_in=4, src_out=10, vo_file="/v/1.wav", vo_duration=6.0, subtitle_text="第二句\n换行"),
        ],
    )


def test_ass_cues_are_sequential_and_cover_durations(tmp_path: Path):
    out = s11_subs.build_ass(_edl(), tmp_path / "subs.ass", width=1920, height=1080, font_name="Noto Sans CJK SC")
    body = out.read_text(encoding="utf-8")
    dialogues = [ln for ln in body.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogues) == 2
    # first cue 0 -> 4.0s, second 4.0 -> 10.0s (abutting; audio-driven timeline)
    assert "0:00:00.00,0:00:04.00" in dialogues[0]
    assert "0:00:04.00,0:00:10.00" in dialogues[1]
    assert "第一句" in dialogues[0]
    assert "\\N" in dialogues[1]  # newline converted to ASS line break
    assert "Noto Sans CJK SC" in body
    assert "BorderStyle" in body  # styled box header present
