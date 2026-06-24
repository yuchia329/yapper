"""CosyVoice2 TTS service — gRPC (deploy on the GPU server, one RTX 3090).

Holds ONE pinned, cloned narrator voice (zero-shot from a reference clip + transcript,
set via env at startup). Implements yapper_rpc.Tts/Synthesize (unary request ->
server-streaming WAV byte chunks).

Run:
  CUDA_VISIBLE_DEVICES=1 TTS_MODEL_DIR=pretrained_models/CosyVoice2-0.5B \
  TTS_REF_WAV=/voices/narrator.wav TTS_REF_TEXT="参考音频的文字内容" \
  TTS_GRPC_PORT=50052 TTS_METRICS_PORT=9102 python tts_service.py

Needs (in the TTS venv): grpcio, grpcio-health-checking, protobuf, prometheus-client,
CosyVoice2 (source install) — plus the `yapper_rpc` package on PYTHONPATH.
See README_deploy.md for the CosyVoice2 install + model download.
"""

from __future__ import annotations

import io
import os
import sys
import threading
from concurrent import futures

import grpc
import torch
import torchaudio
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

# CosyVoice2 is a SOURCE checkout (not pip-installed) and expects its bundled Matcha-TTS
# submodule on sys.path too. Resolve the repo (default <repo>/CosyVoice, two levels up from
# this file) and prepend both, so `import cosyvoice` / `import matcha` work regardless of how
# PYTHONPATH is set — gpud only guarantees yapper_rpc is importable. Override via COSYVOICE_ROOT.
_COSYVOICE = os.environ.get(
    "COSYVOICE_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CosyVoice"),
)
for _p in (_COSYVOICE, os.path.join(_COSYVOICE, "third_party", "Matcha-TTS")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from cosyvoice.cli.cosyvoice import CosyVoice2 as _CosyVoice  # noqa: E402
from cosyvoice.utils.common import set_all_random_seed  # noqa: E402

from yapper_rpc import tts_pb2, tts_pb2_grpc

from _metrics import TTS_AUDIO_SECONDS, TTS_LINES, MetricsInterceptor, start_metrics_server

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "pretrained_models/CosyVoice2-0.5B")
# No clip yet? Default to CosyVoice2's bundled sample prompt so the service runs
# out-of-the-box for the first end-to-end test. Swap TTS_REF_WAV/TTS_REF_TEXT later.
REF_WAV = os.environ.get("TTS_REF_WAV", "CosyVoice/asset/zero_shot_prompt.wav")
REF_TEXT = os.environ.get("TTS_REF_TEXT", "希望你以后能够做的比我还好呦。")
PORT = int(os.environ.get("TTS_GRPC_PORT", "50052"))
METRICS_PORT = int(os.environ.get("TTS_METRICS_PORT", "9102"))
_CHUNK = 256 * 1024  # WAV byte frames streamed to the client

_model = None
_gpu_lock = threading.Lock()  # one model on one GPU: serialize synthesis


def get_model():
    global _model
    if _model is None:
        _model = _CosyVoice(model_dir=MODEL_DIR, fp16=True)
    return _model


class TtsServicer(tts_pb2_grpc.TtsServicer):
    def Synthesize(self, request, context):
        cv = get_model()
        ref_wav = request.ref_wav or REF_WAV
        ref_text = request.ref_text or REF_TEXT
        speed = request.speed or 1.0
        with _gpu_lock:
            set_all_random_seed(request.seed or 42)  # pin timbre/prosody so the voice never drifts
            chunks = [
                out["tts_speech"]
                for out in cv.inference_zero_shot(request.text, ref_text, ref_wav, stream=False, speed=speed)
            ]
            audio = torch.concat(chunks, dim=1)
            buf = io.BytesIO()
            torchaudio.save(buf, audio, cv.sample_rate, format="wav")
        data = buf.getvalue()
        TTS_LINES.labels("tts").inc()
        TTS_AUDIO_SECONDS.labels("tts").inc(audio.shape[1] / cv.sample_rate)
        for i in range(0, len(data), _CHUNK):
            yield tts_pb2.AudioChunk(data=data[i:i + _CHUNK])


def serve() -> None:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        interceptors=[MetricsInterceptor("tts")],
    )
    tts_pb2_grpc.add_TtsServicer_to_server(TtsServicer(), server)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("yapper_rpc.Tts", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)

    start_metrics_server("tts", METRICS_PORT)
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"TTS gRPC on :{PORT}  (metrics :{METRICS_PORT}, model_dir={MODEL_DIR})")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
