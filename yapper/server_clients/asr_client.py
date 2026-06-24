"""gRPC client for the GPU-server WhisperX ASR service.

Used as the fallback when a movie has no embedded text subtitles. Streams the 16 kHz
mono WAV up in chunks (client-streaming — the file is far larger than gRPC's default
message limit), gets back word-level segments (+ optional speaker labels), and maps
them into the pipeline's Transcript artifact.

Same public surface as the former REST client (``health`` / ``transcribe``) so callers
are unchanged; the constructor now takes a gRPC ``target`` (host:port) instead of a URL.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import grpc

from yapper_rpc import asr_pb2, asr_pb2_grpc

from ..schemas import Transcript, TranscriptSegment, TranscriptWord

# WhisperX replies are small; the upload is large but streamed, so a modest recv cap
# is fine. Keepalive guards long transcriptions against idle-connection resets.
_CHUNK = 1 << 20  # 1 MiB audio frames
_CHANNEL_OPTS = [
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 30_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
]


class ASRClient:
    def __init__(self, target: str, timeout: float = 1800.0, channel: grpc.Channel | None = None):
        # ``target`` is host:port (e.g. "asr:50051"); accept a stray scheme for safety.
        self.target = target.replace("grpc://", "").replace("http://", "").rstrip("/")
        self.timeout = timeout
        self._channel = channel or grpc.insecure_channel(self.target, options=_CHANNEL_OPTS)
        self._stub = asr_pb2_grpc.AsrStub(self._channel)

    def health(self) -> dict:
        """Standard grpc.health.v1 check; shape kept dict-like for the old callers."""
        from grpc_health.v1 import health_pb2, health_pb2_grpc

        h = health_pb2_grpc.HealthStub(self._channel)
        resp = h.Check(health_pb2.HealthCheckRequest(service="yapper_rpc.Asr"), timeout=10)
        ok = resp.status == health_pb2.HealthCheckResponse.SERVING
        return {"status": "ok" if ok else "not_serving"}

    def transcribe(
        self,
        wav_path: str | Path,
        *,
        language: str | None = None,
        diarize: bool = False,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> Transcript:
        cfg = asr_pb2.TranscribeConfig(
            language=language or "",
            diarize=diarize,
            min_speakers=min_speakers or 0,
            max_speakers=max_speakers or 0,
        )

        def _requests() -> Iterator[asr_pb2.TranscribeRequest]:
            yield asr_pb2.TranscribeRequest(config=cfg)  # config first
            with open(wav_path, "rb") as fh:
                while chunk := fh.read(_CHUNK):
                    yield asr_pb2.TranscribeRequest(audio_chunk=chunk)

        reply = self._stub.Transcribe(_requests(), timeout=self.timeout)
        return self._to_transcript(reply)

    @staticmethod
    def _to_transcript(reply: asr_pb2.TranscriptReply) -> Transcript:
        segments = []
        for seg in reply.segments:
            words = [
                TranscriptWord(
                    text=w.text,
                    start=w.start,
                    end=w.end,
                    score=(w.score if w.has_score else None),
                )
                for w in seg.words
            ]
            segments.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    speaker=seg.speaker or None,
                    words=words,
                )
            )
        return Transcript(language=reply.language or "und", source="asr", segments=segments)
