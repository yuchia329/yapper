"""Stage 2 — dialogue transcript (ground truth).

Tiered, cheapest-first:
  1. If the source carries a clean TEXT subtitle track, copy it out with ffmpeg
     (free, frame-accurate, 0% WER) and parse it — no ASR compute at all.
  2. Otherwise fall back to WhisperX on the GPU server (handled by the ASR client).

For foreign/Hollywood sources this short-circuit hits often, so the whole front
half can run locally with no server dependency.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ..ffmpeg.run import FFMPEG, run
from ..schemas import ProbeManifest, Transcript, TranscriptSegment

log = logging.getLogger("yapper.s02")

_TS_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt(text: str) -> list[TranscriptSegment]:
    """Parse SRT/VTT timing blocks into transcript segments (tags stripped)."""
    segments: list[TranscriptSegment] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip())
    for block in blocks:
        m = _TS_RE.search(block)
        if not m:
            continue
        start = _to_seconds(*m.group(1, 2, 3, 4))
        end = _to_seconds(*m.group(5, 6, 7, 8))
        # text = everything after the timecode line
        lines = block.split("\n")
        ts_idx = next((i for i, ln in enumerate(lines) if _TS_RE.search(ln)), 0)
        body = " ".join(lines[ts_idx + 1 :]).strip()
        body = re.sub(r"<[^>]+>", "", body)  # strip styling tags
        if body:
            segments.append(TranscriptSegment(start=start, end=end, text=body))
    return segments


def extract_embedded_subtitle(movie_path: str, stream_index: int, out_srt: Path) -> Path:
    out_srt.parent.mkdir(parents=True, exist_ok=True)
    run([FFMPEG, "-y", "-i", str(movie_path), "-map", f"0:{stream_index}", "-c:s", "srt", str(out_srt)])
    return out_srt


def run_stage(
    probe: ProbeManifest,
    movie_path: str | Path,
    work_dir: str | Path,
    *,
    prefer_embedded_subs: bool = True,
) -> Transcript | None:
    """Return a Transcript from embedded subs, or None if ASR (server) is required."""
    work_dir = Path(work_dir)
    sub = probe.best_text_subtitle
    if prefer_embedded_subs and sub is not None:
        log.info("extracting embedded text subtitle (stream %d, %s)", sub.index, sub.language)
        srt = extract_embedded_subtitle(str(movie_path), sub.index, work_dir / "embedded.srt")
        segments = parse_srt(srt.read_text(encoding="utf-8", errors="replace"))
        if segments:
            return Transcript(language=sub.language or "und", source="embedded_sub", segments=segments)
        log.warning("embedded subtitle parsed to 0 segments; will need ASR")
    return None  # caller routes to the GPU-server WhisperX client
