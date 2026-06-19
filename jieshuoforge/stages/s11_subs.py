"""Stage 11 — subtitle generation (styled .ass on the voiceover timeline).

For the MVP, subtitles are line-level: each narration line gets one cue spanning
its EDL segment's screen time (which equals its measured voiceover duration). This
keeps the subtitle timeline aligned to the voiceover without a separate forced
aligner — word-level karaoke is a later polish. Generating the .ass needs no
libass; only burning it in (s12) does.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..schemas import Edl

log = logging.getLogger("jieshuoforge.s11")


def _ts(t: float) -> str:
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def build_ass(
    edl: Edl,
    out_path: str | Path,
    *,
    width: int,
    height: int,
    font_name: str = "Noto Sans CJK SC",
    font_size: int = 48,
    margin_v: int = 60,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # BorderStyle=3 → opaque box behind text; Alignment=2 → bottom-center
        f"Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,3,2,0,2,40,40,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    t = 0.0
    for seg in edl.segments:
        start, end = t, t + seg.screen_duration
        text = seg.subtitle_text.replace("\n", "\\N").strip()
        lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Default,,0,0,0,,{text}\n")
        t = end

    out_path.write_text("".join(lines), encoding="utf-8")
    log.info("[subs] %d cues -> %s", len(edl.segments), out_path.name)
    return out_path
