"""Thin, logged wrapper around ffmpeg/ffprobe subprocess calls.

Every media operation goes through here so the exact command is logged and
failures raise with stderr attached — renders stay reproducible and debuggable.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("yapper.ffmpeg")

# Prefer (1) an explicit FFMPEG_BIN/FFPROBE_BIN env override, (2) the vendored
# static build under vendor/ffmpeg (built with libass for CJK subtitle burn-in,
# which the system Homebrew ffmpeg lacks), (3) whatever is on PATH.
_VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "ffmpeg"


def _resolve(name: str, envvar: str) -> str:
    override = os.environ.get(envvar)
    if override:
        return override
    vendored = _VENDOR / name
    if vendored.exists():
        return str(vendored)
    return shutil.which(name) or name


FFMPEG = _resolve("ffmpeg", "FFMPEG_BIN")
FFPROBE = _resolve("ffprobe", "FFPROBE_BIN")


class FFmpegError(RuntimeError):
    pass


def run(cmd: list[str], *, capture: bool = False) -> str:
    """Run a command, raising FFmpegError on non-zero exit.

    Returns stdout when ``capture`` is True, else "".
    """
    log.debug("exec: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-2000:]
        raise FFmpegError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc.stdout or ""


def has_filter(name: str) -> bool:
    """Whether the installed ffmpeg exposes a given filter (e.g. 'subtitles')."""
    try:
        out = run([FFMPEG, "-hide_banner", "-filters"], capture=True)
    except FFmpegError:
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines() if len(line.split()) > 1)
