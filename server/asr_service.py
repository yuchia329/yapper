"""WhisperX ASR service — gRPC (deploy on the GPU server, one RTX 3090).

Implements yapper_rpc.Asr/Transcribe (client-streaming: config message then WAV byte
chunks) -> TranscriptReply with word-level segments (+ optional speaker labels).

Run:
  CUDA_VISIBLE_DEVICES=0 HF_TOKEN=... ASR_GRPC_PORT=50051 ASR_METRICS_PORT=9101 \
    python asr_service.py

Needs (in the ASR venv): grpcio, grpcio-health-checking, protobuf, prometheus-client,
whisperx — plus the `yapper_rpc` package on PYTHONPATH (copy it next to server/).
See README_deploy.md for the CUDA 12 / cuDNN 9 setup.
"""

from __future__ import annotations

import os
import tempfile
import threading
from concurrent import futures

import grpc
import torch
import whisperx
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from yapper_rpc import asr_pb2, asr_pb2_grpc

from _metrics import ASR_SEGMENTS, MetricsInterceptor, start_metrics_server

MODEL = os.environ.get("ASR_MODEL", "large-v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE = os.environ.get("ASR_COMPUTE", "float16" if DEVICE == "cuda" else "float32")
HF_TOKEN = os.environ.get("HF_TOKEN")
PORT = int(os.environ.get("ASR_GRPC_PORT", "50051"))
METRICS_PORT = int(os.environ.get("ASR_METRICS_PORT", "9101"))

_model = None
_gpu_lock = threading.Lock()  # one model on one GPU: never run two transcriptions at once


def get_model():
    global _model
    if _model is None:
        _model = whisperx.load_model(MODEL, device=DEVICE, compute_type=COMPUTE)
    return _model


class AsrServicer(asr_pb2_grpc.AsrServicer):
    def Transcribe(self, request_iterator, context) -> asr_pb2.TranscriptReply:
        cfg = asr_pb2.TranscribeConfig()
        buf = bytearray()
        for req in request_iterator:
            kind = req.WhichOneof("payload")
            if kind == "config":
                cfg = req.config
            elif kind == "audio_chunk":
                buf.extend(req.audio_chunk)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(bytes(buf))
            path = tmp.name
        try:
            with _gpu_lock:
                audio = whisperx.load_audio(path)
                language = cfg.language or None
                result = get_model().transcribe(audio, batch_size=16, language=language)
                lang = result["language"]

                # word-level timestamps via forced alignment (essential for clip cuts/subs)
                try:
                    model_a, metadata = whisperx.load_align_model(language_code=lang, device=DEVICE)
                    result = whisperx.align(
                        result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False
                    )
                except Exception:  # noqa: BLE001 — some languages lack an alignment model
                    pass

                if cfg.diarize and HF_TOKEN:
                    dia = whisperx.diarize.DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
                    diarize_segments = dia(
                        audio,
                        min_speakers=cfg.min_speakers or None,
                        max_speakers=cfg.max_speakers or None,
                    )
                    result = whisperx.assign_word_speakers(diarize_segments, result)

            segments = []
            for s in result["segments"]:
                words = []
                for w in s.get("words", []):
                    if w.get("start") is None:
                        continue
                    score = w.get("score")
                    words.append(asr_pb2.Word(
                        text=w.get("word", w.get("text", "")),
                        start=float(w.get("start", s["start"])),
                        end=float(w.get("end", s["end"])),
                        score=float(score) if score is not None else 0.0,
                        has_score=score is not None,
                    ))
                segments.append(asr_pb2.Segment(
                    start=float(s["start"]), end=float(s["end"]),
                    text=s.get("text", "").strip(), speaker=s.get("speaker") or "", words=words,
                ))
            ASR_SEGMENTS.labels("asr").inc(len(segments))
            return asr_pb2.TranscriptReply(language=lang, segments=segments)
        finally:
            os.unlink(path)


def serve() -> None:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        interceptors=[MetricsInterceptor("asr")],
        options=[("grpc.max_receive_message_length", 64 * 1024 * 1024)],
    )
    asr_pb2_grpc.add_AsrServicer_to_server(AsrServicer(), server)
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("yapper_rpc.Asr", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)

    start_metrics_server("asr", METRICS_PORT)
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"ASR gRPC on :{PORT}  (metrics :{METRICS_PORT}, model={MODEL}, device={DEVICE})")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
