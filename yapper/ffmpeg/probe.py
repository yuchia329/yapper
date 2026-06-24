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


def video_rotation(video_stream: dict) -> int:
    """Display rotation in degrees, normalized to {0, 90, 180, 270}.

    Phone videos record in one sensor orientation and store a rotation as a display-matrix
    side-data entry (or a legacy ``rotate`` tag), so the *encoded* width/height are not what's
    displayed. ffmpeg auto-applies this on decode; we read it to pick the right output shape."""
    rot = 0.0
    tag = (video_stream.get("tags", {}) or {}).get("rotate")
    if tag is not None:
        try:
            rot = float(tag)
        except (TypeError, ValueError):
            rot = 0.0
    for sd in video_stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rot = float(sd["rotation"])
            except (TypeError, ValueError):
                pass
            break
    return int(round(rot)) % 360


def _even(x: float) -> int:
    n = int(round(x))
    return max(2, n - (n % 2))


def fit_output_dims(src_w: int, src_h: int, *, long_cap: int, short_cap: int) -> tuple[int, int]:
    """Output (w, h) preserving the SOURCE aspect ratio, fitted into an orientation-aware cap box
    (the longer side ≤ ``long_cap``, the shorter ≤ ``short_cap``) and never upscaled. Both rounded
    to even (yuv420p needs even dims). So a 1080×1920 portrait stays 1080×1920; a 4K landscape
    becomes 1920×1080; a 4K portrait becomes 1080×1920. Unknown source → the full cap box."""
    if src_w <= 0 or src_h <= 0:
        return (_even(long_cap), _even(short_cap))
    long_side, short_side = max(src_w, src_h), min(src_w, src_h)
    scale = min(long_cap / long_side, short_cap / short_side, 1.0)
    return (_even(src_w * scale), _even(src_h * scale))
