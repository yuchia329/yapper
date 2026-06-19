"""HTTP client for the GPU-server WhisperX ASR service.

Used as the fallback when a movie has no embedded text subtitles. Uploads the
16 kHz mono WAV, gets back word-level segments (+ optional speaker labels), and
maps them into the pipeline's Transcript artifact.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from ..schemas import Transcript, TranscriptSegment, TranscriptWord


class ASRClient:
    def __init__(self, base_url: str, timeout: float = 1800.0, client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)

    def health(self) -> dict:
        r = self._client.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def transcribe(
        self,
        wav_path: str | Path,
        *,
        language: str | None = None,
        diarize: bool = False,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> Transcript:
        data = {"diarize": str(diarize).lower()}
        if language:
            data["language"] = language
        if min_speakers is not None:
            data["min_speakers"] = str(min_speakers)
        if max_speakers is not None:
            data["max_speakers"] = str(max_speakers)
        with open(wav_path, "rb") as fh:
            files = {"file": (Path(wav_path).name, fh, "audio/wav")}
            r = self._client.post(f"{self.base_url}/transcribe", data=data, files=files)
        r.raise_for_status()
        return self._parse(r.json())

    @staticmethod
    def _parse(payload: dict) -> Transcript:
        segments = []
        for seg in payload.get("segments", []):
            words = [
                TranscriptWord(
                    text=w.get("word", w.get("text", "")),
                    start=float(w.get("start", seg["start"])),
                    end=float(w.get("end", seg["end"])),
                    score=w.get("score"),
                )
                for w in seg.get("words", [])
                if w.get("start") is not None
            ]
            segments.append(
                TranscriptSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=seg.get("text", "").strip(),
                    speaker=seg.get("speaker"),
                    words=words,
                )
            )
        return Transcript(language=payload.get("language", "und"), source="asr", segments=segments)
