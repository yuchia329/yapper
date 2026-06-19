"""Spoken-duration estimation for Mandarin 解说 narration.

Used to budget runtime in code (structured outputs can't enforce length) before
spending any TTS compute. Mandarin recap narration is fast — count CJK characters
individually and Latin runs as ~words.
"""

from __future__ import annotations

import re

CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+")

DEFAULT_CPS = 4.0  # CJK chars/sec — empirically matched to CosyVoice2 output (measured ~3.8)
DEFAULT_WPS = 2.0  # Latin words per second


def estimate_spoken_seconds(text: str, cps: float = DEFAULT_CPS, wps: float = DEFAULT_WPS) -> float:
    cjk = len(CJK_RE.findall(text))
    latin = len(LATIN_WORD_RE.findall(text))
    secs = cjk / cps + latin / wps
    return round(max(0.4, secs), 3)
