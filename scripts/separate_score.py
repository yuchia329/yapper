"""Build the score bed: source-separate a movie's audio into a no-dialogue track.

Under narration the recap plays the film's OWN score/SFX (not its dialogue) beneath
the voiceover, so the background isn't dead air and the dialogue doesn't clash with
the Mandarin narration. We get that track by running Demucs on the extracted audio
and keeping the `no_vocals` stem (everything except speech/singing).

This runs once per movie and is cached at artifacts/<movie>/audio_novocals.wav; the
renderer (s12 / graph.py) picks it up automatically when [audio] score_bed = true.

Demucs is heavy, so it runs on the GPU server: upload audio.wav -> demucs --two-stems
-> download the no_vocals stem. Mirrors scripts/build_narrator_ref.py's orchestration.

Usage (SSH tunnel/host reachable):
    uv run python scripts/separate_score.py data/RushHour1.webm
    uv run python scripts/separate_score.py data/RushHour1.webm --gpu 3 --demucs "uv run --project ~/jieshuo/demucs demucs"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jieshuoforge.config import load_config  # noqa: E402


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("movie", help="movie path (used to locate artifacts/<slug>/audio.wav)")
    ap.add_argument("--server", default="nlp", help="ssh host")
    ap.add_argument("--remote-dir", default="~/jieshuo/separate", help="remote working dir")
    ap.add_argument("--gpu", default="3", help="CUDA_VISIBLE_DEVICES on the server")
    ap.add_argument("--model", default="htdemucs", help="demucs model")
    ap.add_argument("--demucs", default="demucs", help="how to invoke demucs on the server")
    args = ap.parse_args()

    cfg = load_config()
    mdir = cfg.movie_dir(args.movie)
    slug = mdir.name
    local_wav = mdir / "audio.wav"
    if not local_wav.exists():
        raise SystemExit(f"{local_wav} missing — run the front half first (it extracts audio.wav)")
    out_local = mdir / "audio_novocals.wav"

    remote = args.remote_dir
    remote_wav = f"{remote}/{slug}.wav"
    # demucs --two-stems=vocals writes <out>/<model>/<stem-name>/no_vocals.wav
    remote_out = f"{remote}/out"
    remote_novocals = f"{remote_out}/{args.model}/{slug}/no_vocals.wav"

    print(f"[1/3] uploading {local_wav.name} -> {args.server}:{remote_wav}")
    sh(["ssh", args.server, f"mkdir -p {remote}"])
    sh(["scp", "-q", str(local_wav), f"{args.server}:{remote_wav}"])

    print(f"[2/3] separating on {args.server} (gpu {args.gpu}, {args.model}) — this takes a while…")
    demucs_cmd = (
        f"cd {remote} && CUDA_VISIBLE_DEVICES={args.gpu} "
        f"{args.demucs} --two-stems=vocals -n {args.model} -o {remote_out} {slug}.wav"
    )
    sh(["ssh", args.server, demucs_cmd])

    print(f"[3/3] downloading no_vocals stem -> {out_local}")
    sh(["scp", "-q", f"{args.server}:{remote_novocals}", str(out_local)])
    print(f"done -> {out_local}")
    print("Set [audio] score_bed = true (default) and re-run the back half to use it.")


if __name__ == "__main__":
    main()
