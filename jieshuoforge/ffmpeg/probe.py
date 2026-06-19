"""ffprobe wrappers: container/stream metadata, fps, and VFR detection."""

from __future__ import annotations

import json

from .run import FFPROBE, run

_TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}


def probe_json(path: str) -> dict:
    out = run(
        [
            FFPROBE,
            "-v", "error",
            "-show_format",
            "-show_streams",
            "-of", "json",
            path,
        ],
        capture=True,
    )
    return json.loads(out)


def duration_sec(path: str) -> float:
    """Media duration in seconds via ffprobe (used to measure TTS output)."""
    out = run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture=True,
    ).strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def _parse_rate(rate: str | None) -> float:
    """Parse an ffprobe rational like '24000/1001' into fps."""
    if not rate or rate in ("0/0", "N/A"):
        return 0.0
    if "/" in rate:
        num, _, den = rate.partition("/")
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(rate)


def detect_vfr(video_stream: dict) -> tuple[bool, float]:
    """Heuristic VFR detection + the fps to use.

    Compares the codec frame rate (r_frame_rate) with the average frame rate
    (avg_frame_rate). A meaningful divergence flags variable frame rate, where
    frame-number-based cuts would drift — so downstream we always cut on seconds.
    Returns (is_vfr, fps).
    """
    r_fps = _parse_rate(video_stream.get("r_frame_rate"))
    avg_fps = _parse_rate(video_stream.get("avg_frame_rate"))
    fps = avg_fps or r_fps or 24.0
    if r_fps and avg_fps:
        is_vfr = abs(r_fps - avg_fps) / max(r_fps, avg_fps) > 0.01
    else:
        is_vfr = False
    return is_vfr, fps


def is_text_subtitle(codec_name: str | None) -> bool:
    return (codec_name or "").lower() in _TEXT_SUB_CODECS
