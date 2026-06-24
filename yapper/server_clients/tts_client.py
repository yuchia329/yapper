"""gRPC client for the GPU-server TTS service (CosyVoice2 / IndexTTS2).

The service holds ONE pinned, cloned narrator voice (configured at startup from a
reference clip). We send text + a fixed seed per line and stream back the WAV in
chunks (server-streaming), concatenating them into the output file. Word timestamps
are NOT requested here — subtitles are produced later by force-aligning the generated
voiceover (s11), keeping the two timelines cleanly separate.

Same public surface as the former REST client (``health`` / ``synthesize``) so callers
are unchanged; the constructor now takes a gRPC ``target`` (host:port) instead of a URL.
"""

from __future__ import annotations

from pathlib import Path

import grpc

from yapper_rpc import tts_pb2, tts_pb2_grpc

_CHANNEL_OPTS = [
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 30_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
]


class TTSClient:
    def __init__(self, target: str, timeout: float = 600.0, channel: grpc.Channel | None = None):
        self.target = target.replace("grpc://", "").replace("http://", "").rstrip("/")
        self.timeout = timeout
        # Own (and therefore close) the channel only when we created it; never close one the
        # caller injected (e.g. a shared test channel).
        self._owns_channel = channel is None
        self._channel = channel or grpc.insecure_channel(self.target, options=_CHANNEL_OPTS)
        self._stub = tts_pb2_grpc.TtsStub(self._channel)

    def close(self) -> None:
        """Close the gRPC channel (frees its background threads/fds). Idempotent; a no-op for
        an injected channel. Important when many clients are created per run (TTS fan-out)."""
        if self._owns_channel and self._channel is not None:
            self._channel.close()
            self._channel = None

    def __enter__(self) -> "TTSClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def health(self) -> dict:
        from grpc_health.v1 import health_pb2, health_pb2_grpc

        h = health_pb2_grpc.HealthStub(self._channel)
        resp = h.Check(health_pb2.HealthCheckRequest(service="yapper_rpc.Tts"), timeout=10)
        ok = resp.status == health_pb2.HealthCheckResponse.SERVING
        return {"status": "ok" if ok else "not_serving"}

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
        req = tts_pb2.SynthesizeRequest(
            text=text, seed=seed, speed=speed,
            ref_wav=ref_wav or "", ref_text=ref_text or "",
        )
        with open(out_path, "wb") as fh:
            for chunk in self._stub.Synthesize(req, timeout=self.timeout):
                fh.write(chunk.data)
        return out_path
