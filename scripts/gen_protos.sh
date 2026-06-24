#!/usr/bin/env bash
# Regenerate the gRPC python stubs from yapper_rpc/*.proto into yapper_rpc/.
# The generated *_pb2.py / *_pb2_grpc.py are committed so neither the web image nor
# the GPU box needs protoc at build/deploy time.
#
#   bash scripts/gen_protos.sh
#
# Runs grpcio-tools via an ephemeral uv env — no global install needed.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/yapper_rpc"

# Pin the generator to protobuf-4.25 gencode ON PURPOSE. The TTS env (CosyVoice) hard-pins
# protobuf==4.25, whose runtime has no `google.protobuf.runtime_version`. Stubs from a newer
# protoc (5.27+) emit `from google.protobuf import runtime_version` + a min-runtime assert and
# fail to import there. Protobuf 4.25 gencode has no such line and STILL imports cleanly under
# the ASR/web env's protobuf 6.x (protobuf runtimes are backward-compatible with older gencode).
# grpcio-tools 1.62.3 bundles protoc 25.x (protobuf 4.25.3) and predates the grpc-version check
# in _pb2_grpc.py, so the stubs also run under the TTS env's older grpcio (1.57). Lowest common
# denominator across all consumers — DON'T bump without re-checking the TTS env.
GRPCIO_TOOLS_VERSION="${GRPCIO_TOOLS_VERSION:-1.62.3}"

# Generate with package-qualified imports so `from yapper_rpc import asr_pb2` works.
uv run --no-project --with "grpcio-tools==${GRPCIO_TOOLS_VERSION}" \
  python -m grpc_tools.protoc \
  -I"$REPO" \
  --python_out="$REPO" \
  --grpc_python_out="$REPO" \
  yapper_rpc/asr.proto yapper_rpc/tts.proto yapper_rpc/gpud.proto

echo "generated stubs in $OUT:"
ls -1 "$OUT"/*_pb2*.py
