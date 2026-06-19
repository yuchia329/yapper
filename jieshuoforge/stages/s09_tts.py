"""Stage 9 — voiceover synthesis.

Synthesize each script line separately (parallelizable, regenerable per line) with
the pinned cloned voice and a fixed seed so timbre never drifts. Each clip is
loudness-normalized to a consistent target so it sits predictably over the ducked
movie audio downstream, then its real duration is measured (the audio-driven EDL
depends on MEASURED, not estimated, durations).

The ``synthesizer`` argument is any object exposing
``synthesize(text, out_path, seed=..., speed=...) -> Path`` (the TTSClient, or a
mock in tests).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..ffmpeg.probe import duration_sec
from ..ffmpeg.run import FFMPEG, run
from ..schemas import Script, VOLine, VOManifest

log = logging.getLogger("jieshuoforge.s09")


def _loudnorm(src: Path, dst: Path, *, target_lufs: float, audio_rate: int) -> None:
    run(
        [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
            "-ar", str(audio_rate), "-ac", "1",
            str(dst),
        ]
    )


def run_stage(
    script: Script,
    synthesizer,
    vo_dir: str | Path,
    *,
    voice_id: str,
    seed: int = 42,
    target_lufs: float = -16.0,
    audio_rate: int = 48000,
    ref_wav: str | None = None,
    ref_text: str | None = None,
    workers: int = 1,
) -> VOManifest:
    vo_dir = Path(vo_dir)
    raw_dir = vo_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Only narration lines get a voiceover; playback lines play original audio (no TTS).
    ref_kw: dict = {}
    if ref_wav:
        ref_kw["ref_wav"] = ref_wav
    if ref_text:
        ref_kw["ref_text"] = ref_text

    narration = [line for line in script.lines if line.kind != "playback"]

    def _synth(line) -> VOLine:
        # Each line is independent (own files, fixed seed), so lines parallelize.
        raw = raw_dir / f"{line.line_id}.wav"
        synthesizer.synthesize(line.text, raw, seed=seed, **ref_kw)
        final = vo_dir / f"{line.line_id}.wav"
        _loudnorm(raw, final, target_lufs=target_lufs, audio_rate=audio_rate)
        dur = duration_sec(str(final))
        log.info("[tts] %s -> %.2fs", line.line_id, dur)
        return VOLine(
            line_id=line.line_id, text=line.text, file=str(final), measured_duration_sec=round(dur, 3)
        )

    # Concurrency is bounded by the single TTS GPU server, so keep `workers` low;
    # ex.map preserves line order (the voiceover timeline) regardless of finish order.
    n = max(1, int(workers))
    if n > 1 and len(narration) > 1:
        log.info("[tts] synthesizing %d lines with %d workers", len(narration), min(n, len(narration)))
        with ThreadPoolExecutor(max_workers=n) as ex:
            lines: list[VOLine] = list(ex.map(_synth, narration))
    else:
        lines = [_synth(line) for line in narration]

    total = sum(l.measured_duration_sec for l in lines)
    log.info("[tts] %d lines, %.1fs total voiceover", len(lines), total)
    return VOManifest(voice_id=voice_id, seed=seed, lines=lines)
