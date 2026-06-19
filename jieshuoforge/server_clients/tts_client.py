"""HTTP client for the GPU-server TTS service (CosyVoice2 / IndexTTS2).

The service holds ONE pinned, cloned narrator voice (configured at startup from a
reference clip). We send text + a fixed seed per line and get back a WAV. Word
timestamps are NOT requested here — subtitles are produced later by force-aligning
the generated voiceover (s11), which keeps the two timelines cleanly separate.
"""

from __future__ import annotations

from pathlib import Path

import httpx


class TTSClient:
    def __init__(self, base_url: str, timeout: float = 600.0, client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)

    def health(self) -> dict:
        r = self._client.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def synthesize(
        self,
        text: str,
        out_path: str | Path,
        *,
        seed: int = 42,
        speed: float = 1.0,
        ref_wav: str | None = None,
        ref_text: str | None = None,
    ) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {"text": text, "seed": seed, "speed": speed}
        if ref_wav:  # server-side path to the cloned-narrator reference clip
            payload["ref_wav"] = ref_wav
        if ref_text:
            payload["ref_text"] = ref_text
        r = self._client.post(f"{self.base_url}/synthesize", json=payload)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return out_path
