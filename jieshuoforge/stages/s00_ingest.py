"""Stage 0 — ingest & probe.

Reads container/stream metadata, detects VFR (so we cut on seconds, not frames),
and lists subtitle tracks so a clean text track can short-circuit ASR.
"""

from __future__ import annotations

from pathlib import Path

from ..ffmpeg.probe import detect_vfr, is_text_subtitle, probe_json
from ..schemas import AudioStream, ProbeManifest, SubtitleStream


def run(movie_path: str | Path) -> ProbeManifest:
    movie_path = str(movie_path)
    info = probe_json(movie_path)
    streams = info.get("streams", [])
    fmt = info.get("format", {})

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"no video stream found in {movie_path}")
    is_vfr, fps = detect_vfr(video)

    audio_streams = [
        AudioStream(
            index=s["index"],
            codec=s.get("codec_name", "?"),
            channels=int(s.get("channels", 0) or 0),
            language=(s.get("tags", {}) or {}).get("language"),
        )
        for s in streams
        if s.get("codec_type") == "audio"
    ]

    subtitle_streams = [
        SubtitleStream(
            index=s["index"],
            codec=s.get("codec_name", "?"),
            language=(s.get("tags", {}) or {}).get("language"),
            is_text=is_text_subtitle(s.get("codec_name")),
        )
        for s in streams
        if s.get("codec_type") == "subtitle"
    ]

    duration = float(fmt.get("duration", 0.0) or video.get("duration", 0.0) or 0.0)

    return ProbeManifest(
        source_path=str(Path(movie_path).resolve()),
        container=fmt.get("format_name", "?"),
        duration_sec=duration,
        width=int(video.get("width", 0) or 0),
        height=int(video.get("height", 0) or 0),
        fps=round(fps, 4),
        is_vfr=is_vfr,
        video_codec=video.get("codec_name", "?"),
        audio_streams=audio_streams,
        subtitle_streams=subtitle_streams,
    )
