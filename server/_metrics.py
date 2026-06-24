"""Prometheus instrumentation for the gRPC GPU services (ASR + TTS).

A gRPC ServerInterceptor times every RPC (request seconds, in-flight, errors), and a
side HTTP server exposes /metrics on a plain port for Prometheus to scrape (gRPC and
Prometheus's pull model don't mix, so metrics ride a separate HTTP port). Pair with
dcgm-exporter on the box for GPU utilization/memory.
"""

from __future__ import annotations

import time

import grpc
from prometheus_client import Counter, Gauge, Histogram, start_http_server

_REQ = Histogram(
    "yapper_gpu_request_seconds", "GPU RPC latency", ["service", "method"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600),
)
_INFLIGHT = Gauge("yapper_gpu_inflight_requests", "In-flight RPCs", ["service", "method"])
_UP = Gauge("yapper_gpu_service_up", "Service liveness (1=up)", ["service"])
ERRORS = Counter("yapper_gpu_errors_total", "RPC errors", ["service", "method"])

# service-specific
TTS_LINES = Counter("yapper_tts_lines_synthesized_total", "Voiceover lines synthesized", ["service"])
TTS_AUDIO_SECONDS = Counter("yapper_tts_audio_seconds_total", "Seconds of audio produced", ["service"])
ASR_SEGMENTS = Counter("yapper_asr_segments_total", "Transcript segments produced", ["service"])


def _short(method: str) -> str:
    return method.rsplit("/", 1)[-1] or method


class MetricsInterceptor(grpc.ServerInterceptor):
    """Wraps each RPC behavior with timing/inflight/error metrics, for all four
    streaming combinations (unary/stream × unary/stream)."""

    def __init__(self, service: str):
        self.service = service

    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        if handler is None:
            return None
        svc, method = self.service, _short(handler_call_details.method)

        def timed_unary(behavior):
            def wrapper(request, context):
                _INFLIGHT.labels(svc, method).inc()
                t = time.perf_counter()
                try:
                    return behavior(request, context)
                except Exception:
                    ERRORS.labels(svc, method).inc()
                    raise
                finally:
                    _REQ.labels(svc, method).observe(time.perf_counter() - t)
                    _INFLIGHT.labels(svc, method).dec()
            return wrapper

        def timed_stream_response(behavior):
            def wrapper(request, context):
                _INFLIGHT.labels(svc, method).inc()
                t = time.perf_counter()
                try:
                    yield from behavior(request, context)
                except Exception:
                    ERRORS.labels(svc, method).inc()
                    raise
                finally:
                    _REQ.labels(svc, method).observe(time.perf_counter() - t)
                    _INFLIGHT.labels(svc, method).dec()
            return wrapper

        rs, ss = handler.request_streaming, handler.response_streaming
        ser, deser = handler.response_serializer, handler.request_deserializer
        if not rs and not ss:
            return grpc.unary_unary_rpc_method_handler(
                timed_unary(handler.unary_unary), request_deserializer=deser, response_serializer=ser)
        if rs and not ss:  # client-streaming (ASR Transcribe)
            return grpc.stream_unary_rpc_method_handler(
                timed_unary(handler.stream_unary), request_deserializer=deser, response_serializer=ser)
        if not rs and ss:  # server-streaming (TTS Synthesize)
            return grpc.unary_stream_rpc_method_handler(
                timed_stream_response(handler.unary_stream), request_deserializer=deser, response_serializer=ser)
        return grpc.stream_stream_rpc_method_handler(
            timed_stream_response(handler.stream_stream), request_deserializer=deser, response_serializer=ser)


def start_metrics_server(service: str, port: int) -> None:
    """Expose /metrics on ``port`` and mark the service up."""
    start_http_server(port)
    _UP.labels(service).set(1)
