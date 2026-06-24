"""Stage 11 — subtitle generation (styled .ass on the voiceover timeline).

For the MVP, subtitles are line-level: each narration line gets one cue spanning
its EDL segment's screen time (which equals its measured voiceover duration). This
keeps the subtitle timeline aligned to the voiceover without a separate forced
aligner — word-level karaoke is a later polish. Generating the .ass needs no
libass; only burning it in (s12) does.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ..schemas import Edl

log = logging.getLogger("yapper.s11")

_SENT_END = "。！？!?…"          # strong breaks: end of a sentence
_CLAUSE = "，、；：,;:"            # soft breaks: within a sentence


def _max_chars(width: int, font_size: int) -> int:
    """Roughly how many CJK glyphs fit on one line at this frame width/font (full-width glyphs
    ≈ font_size wide; leave ~8% for L/R margins). Used to size the timed pieces — narrower
    (portrait) frames get more, shorter pieces. WrapStyle still wraps any residual overflow."""
    return max(6, int(width * 0.92 / max(1, font_size)))


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split a caption into display pieces: explicit newlines are author breaks (hard), then split
    on sentence enders, then clause marks, and hard-split anything still longer than ``max_chars``
    (so a long unpunctuated line can't overflow a narrow frame)."""
    chunks: list[str] = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        if not line:
            continue
        for sent in (s.strip() for s in re.split(rf"(?<=[{_SENT_END}])", line) if s.strip()):
            if len(sent) <= max_chars:
                chunks.append(sent)
                continue
            buf = ""
            for piece in (p.strip() for p in re.split(rf"(?<=[{_CLAUSE}])", sent) if p.strip()):
                if len(buf) + len(piece) <= max_chars:
                    buf += piece
                else:
                    if buf:
                        chunks.append(buf)
                    while len(piece) > max_chars:    # still too long -> hard split
                        chunks.append(piece[:max_chars])
                        piece = piece[max_chars:]
                    buf = piece
            if buf:
                chunks.append(buf)
    return chunks or [" ".join(text.split())]


def _segment_cues(text: str, duration: float, max_chars: int) -> list[tuple[float, float, str]]:
    """Local-timeline (0..duration) cues for one caption, split into timed pieces. Each piece's
    span is proportional to its character count (≈ constant speaking rate), so pieces appear
    roughly when that text is being spoken. NOTE: this is an estimate, not true word-level sync
    (the TTS returns no timestamps). Returns [(start, end, piece), ...]."""
    chunks = _chunk_text(text, max_chars)
    if not chunks:
        return []
    if len(chunks) == 1 or duration <= 0:
        return [(0.0, duration, chunks[0])] if len(chunks) == 1 else \
               [(0.0, duration, " ".join(chunks))]
    total = sum(len(c) for c in chunks)
    cues: list[tuple[float, float, str]] = []
    t = 0.0
    for i, c in enumerate(chunks):
        end = duration if i == len(chunks) - 1 else round(t + duration * len(c) / total, 3)
        cues.append((t, end, c))
        t = end
    return cues


def _ts(t: float) -> str:
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_header(width: int, height: int, font_name: str, font_size: int, margin_v: int) -> str:
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\n"
        # WrapStyle 2 = smart wrap: a chunk that's still too wide for a narrow (portrait) frame
        # wraps onto multiple lines instead of overflowing past the edges.
        "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
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


def _dialogue(start: float, end: float, text: str) -> str:
    return f"Dialogue: 0,{_ts(start)},{_ts(end)},Default,,0,0,0,,{text.replace(chr(10), chr(92) + 'N').strip()}\n"


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

    max_chars = _max_chars(width, font_size)
    lines = [_ass_header(width, height, font_name, font_size, margin_v)]
    t = 0.0
    n_cues = 0
    for seg in edl.segments:
        for cs, ce, piece in _segment_cues(seg.subtitle_text, seg.screen_duration, max_chars):
            lines.append(_dialogue(t + cs, t + ce, piece))
            n_cues += 1
        t += seg.screen_duration

    out_path.write_text("".join(lines), encoding="utf-8")
    log.info("[subs] %d cues (%d segments) -> %s", n_cues, len(edl.segments), out_path.name)
    return out_path


def build_segment_ass(
    text: str,
    duration: float,
    out_path: str | Path,
    *,
    width: int,
    height: int,
    font_name: str = "Noto Sans CJK SC",
    font_size: int = 48,
    margin_v: int = 60,
) -> Path | None:
    """Write a .ass for one segment on its LOCAL timeline ``[0, duration]`` (the segment video is
    reset to PTS 0), for burning in during that segment's encode.

    The caption is split into timed pieces (sentence/clause-aware, sized to the frame width) so
    long lines don't overflow a narrow/portrait frame and pieces appear roughly in step with the
    speech — an estimate by text proportion, not true word-level sync. Returns ``None`` for empty
    text (nothing to burn)."""
    if not text or not text.strip():
        return None
    cues = _segment_cues(text, duration, _max_chars(width, font_size))
    if not cues:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = _ass_header(width, height, font_name, font_size, margin_v) + "".join(
        _dialogue(s, e, piece) for s, e, piece in cues
    )
    out_path.write_text(body, encoding="utf-8")
    return out_path
