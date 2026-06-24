"""Build the score bed: source-separate a movie's audio into a no-dialogue track.

Under narration the commentary plays the film's OWN score/SFX (not its dialogue) beneath
the voiceover, so the background isn't dead air and the dialogue doesn't clash with
the narration. We get that track by running Demucs on the movie's audio and keeping
the `no_vocals` stem (everything except speech/singing).

This runs once per movie and is cached at artifacts/<movie>/audio_novocals.flac; the
renderer (s12 / graph.py) picks it up automatically when [audio] score_bed = true.

Quality matters for a music bed, so we separate the full-quality stereo audio (the
movie's own opus stream, copied losslessly — NOT the 16 kHz mono ASR wav). Demucs is
heavy, so it runs on the GPU server: copy the audio stream -> scp up -> demucs -> scp
the flac stem back. Transfers retry on the connection resets the link sometimes throws
(macOS ships openrsync, which lacks GNU rsync's resume flags, so we use scp + retries).

Usage (server reachable via ssh):
    uv run python scripts/separate_score.py data/RushHour1.webm
    uv run python scripts/separate_score.py data/RushHour1.webm --gpu 5 --demucs "~/jieshuo/demucs/.venv/bin/demucs"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from yapper.config import load_config  # noqa: E402
from yapper.ffmpeg.run import FFMPEG  # noqa: E402

_SCP_KEEPALIVE = ["-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def copy(src: str, dst: str, tries: int = 4) -> None:
    """scp with retries — the link occasionally resets mid-transfer; just try again."""
    for i in range(1, tries + 1):
        if subprocess.run(["scp", *_SCP_KEEPALIVE, src, dst]).returncode == 0:
            return
        print(f"  transfer attempt {i}/{tries} failed; retrying…")
    raise SystemExit(f"transfer failed after {tries} attempts: {src} -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("movie", help="movie path (its audio is separated into the score bed)")
    ap.add_argument("--server", default="nlp", help="ssh host")
    ap.add_argument("--remote-dir", default="~/jieshuo/separate", help="remote working dir")
    ap.add_argument("--gpu", default="5", help="CUDA_VISIBLE_DEVICES on the server")
    ap.add_argument("--model", default="htdemucs", help="demucs model")
    ap.add_argument("--demucs", default="~/jieshuo/demucs/.venv/bin/demucs", help="demucs invocation on the server")
    args = ap.parse_args()

    cfg = load_config()
    movie = Path(args.movie)
    mdir = cfg.movie_dir(str(movie))
    slug = mdir.name
    out_local = mdir / "audio_novocals.flac"

    # 1. copy the movie's audio stream losslessly (small upload, full quality for separation)
    src_local = mdir / "audio_src.mka"
    print(f"[1/4] extracting movie audio (stream copy) -> {src_local.name}")
    sh([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(movie), "-vn", "-c:a", "copy", str(src_local)])

    remote = args.remote_dir
    remote_src = f"{remote}/{slug}.mka"
    remote_out = f"{remote}/out"
    remote_novocals = f"{remote_out}/{args.model}/{slug}/no_vocals.flac"

    print(f"[2/4] uploading -> {args.server}:{remote_src}")
    sh(["ssh", args.server, f"mkdir -p {remote}"])
    copy(str(src_local), f"{args.server}:{remote_src}")

    print(f"[3/4] separating on {args.server} (gpu {args.gpu}, {args.model}) — takes a while…")
    # The demucs venv has no torchaudio audio backend and can't decode opus, so transcode
    # the uploaded .mka to FLAC server-side with the CLI ffmpeg (in ~/.local/bin), which the
    # torchaudio `soundfile` backend reads natively. One-time venv setup:
    #   uv pip install --python ~/jieshuo/demucs/.venv/bin/python soundfile
    sh(["ssh", args.server,
        f"cd {remote} && export PATH=$HOME/.local/bin:$PATH && "
        f"ffmpeg -y -hide_banner -loglevel error -i {slug}.mka -ac 2 -ar 44100 {slug}.flac && "
        f"CUDA_VISIBLE_DEVICES={args.gpu} {args.demucs} "
        f"--two-stems=vocals --flac -n {args.model} -o {remote_out} {slug}.flac"])

    print(f"[4/4] downloading no_vocals stem -> {out_local}")
    copy(f"{args.server}:{remote_novocals}", str(out_local))
    print(f"done -> {out_local}")
    print("Re-run the back half ([audio] score_bed = true) to use it.")


if __name__ == "__main__":
    main()
