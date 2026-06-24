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
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..ffmpeg.probe import duration_sec
from ..ffmpeg.run import FFMPEG, run
from ..schemas import Script, VOLine, VOManifest

log = logging.getLogger("yapper.s09")


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
    synthesizer=None,
    vo_dir: str | Path = "vo",
    *,
    voice_id: str,
    seed: int = 42,
    target_lufs: float = -16.0,
    audio_rate: int = 48000,
    ref_wav: str | None = None,
    ref_text: str | None = None,
    workers: int = 1,
    synthesizers: list | None = None,
) -> VOManifest:
    """Synthesize each narration line, parallelizing across one or more TTS instances.

    Pass ``synthesizers`` (a pool of independent gRPC clients, each a separately-leased
    gpud instance) to fan out: lines are sharded across the pool so synthesis runs truly
    in parallel — N instances ≈ N× throughput. Each TTS server serializes its own work via
    a process-wide GPU lock, so parallelism comes from the NUMBER OF INSTANCES, not from
    threads hitting one instance. Falls back to the single ``synthesizer`` (the CLI / legacy
    path), where ``workers`` threads share that one instance. Output order always matches
    script order (the voiceover timeline) regardless of completion order.
    """
    if synthesizers is not None and len(synthesizers) == 0:
        raise ValueError("empty synthesizer pool — lease at least one TTS instance")
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

    def _synth(line, syn) -> VOLine:
        # Each line is independent (own files, fixed seed), so lines parallelize.
        raw = raw_dir / f"{line.line_id}.wav"
        syn.synthesize(line.text, raw, seed=seed, **ref_kw)
        final = vo_dir / f"{line.line_id}.wav"
        _loudnorm(raw, final, target_lufs=target_lufs, audio_rate=audio_rate)
        dur = duration_sec(str(final))
        log.info("[tts] %s -> %.2fs", line.line_id, dur)
        return VOLine(
            line_id=line.line_id, text=line.text, file=str(final), measured_duration_sec=round(dur, 3)
        )

    pool = list(synthesizers) if synthesizers else None
    if pool and len(pool) > 1 and len(narration) > 1:
        # Fan out across instances: a queue hands each worker a distinct instance, so at most
        # `len(pool)` lines synthesize at once and each instance serves one line at a time.
        log.info("[tts] synthesizing %d lines across %d TTS instances", len(narration), len(pool))
        avail: queue.Queue = queue.Queue()
        for s in pool:
            avail.put(s)

        def _via_pool(line) -> VOLine:
            syn = avail.get()
            try:
                return _synth(line, syn)
            finally:
                avail.put(syn)

        with ThreadPoolExecutor(max_workers=len(pool)) as ex:
            lines: list[VOLine] = list(ex.map(_via_pool, narration))
    else:
        # Single instance: legacy path. `workers` threads share it (bounded by the server's
        # GPU lock, so usually leave at 1); ex.map still preserves line order.
        syn = (pool[0] if pool else synthesizer)
        n = max(1, int(workers))
        if n > 1 and len(narration) > 1:
            log.info("[tts] synthesizing %d lines with %d workers", len(narration), min(n, len(narration)))
            with ThreadPoolExecutor(max_workers=n) as ex:
                lines = list(ex.map(lambda ln: _synth(ln, syn), narration))
        else:
            lines = [_synth(line, syn) for line in narration]

    total = sum(vl.measured_duration_sec for vl in lines)
    log.info("[tts] %d lines, %.1fs total voiceover", len(lines), total)
    return VOManifest(voice_id=voice_id, seed=seed, lines=lines)
