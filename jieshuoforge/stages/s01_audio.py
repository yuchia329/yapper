"""Stage 1 — audio extraction.

Pull a 16 kHz mono PCM WAV (what ASR wants) so the upload to the GPU server is
small. The original audio is left untouched for later clip extraction/ducking.
"""

from __future__ import annotations

from pathlib import Path

from ..ffmpeg.run import FFMPEG, run


def run_stage(movie_path: str | Path, out_wav: str | Path) -> Path:
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            FFMPEG, "-y",
            "-i", str(movie_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(out_wav),
        ]
    )
    return out_wav
