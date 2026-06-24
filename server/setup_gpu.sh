#!/usr/bin/env bash
# One-time GPU-box setup for the yapper inference services (uv-native).
#
# Run ON the box, after copying server/ + yapper_rpc/ to $JIESHUO_ROOT (default ~/jieshuo):
#   # from your machine:
#   scp -r server yapper_rpc nlp:~/jieshuo/
#   # on the box:
#   ssh nlp 'bash ~/jieshuo/server/setup_gpu.sh'
#
# What it does (idempotent — re-running skips finished steps):
#   1. install uv (if absent)
#   2. uv sync the ASR/gpud env (server/asr) and the TTS env (server/tts)
#   3. clone CosyVoice2 (--recursive) and install its source requirements into the TTS env
#   4. download the CosyVoice2 weights via modelscope
#
# Override anything via env: JIESHUO_ROOT, PYTHON_VERSION, COSYVOICE_REPO, MODEL_ID, MODEL_DIR.
set -euo pipefail

ROOT="${JIESHUO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # parent of server/ = ~/jieshuo
# Independent interpreters per env: ASR needs >=3.11 (onnxruntime dropped 3.10 wheels);
# TTS stays on 3.10 for CosyVoice2 / Matcha-TTS.
ASR_PYTHON="${ASR_PYTHON:-3.12}"
TTS_PYTHON="${TTS_PYTHON:-3.10}"
COSYVOICE_REPO="${COSYVOICE_REPO:-https://github.com/FunAudioLLM/CosyVoice.git}"
MODEL_ID="${MODEL_ID:-iic/CosyVoice2-0.5B}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/CosyVoice2-0.5B}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# 0. preconditions
[ -d "$ROOT/server/asr" ] && [ -d "$ROOT/server/tts" ] || {
  echo "ERROR: $ROOT/server/{asr,tts} not found — copy server/ to $ROOT first" >&2; exit 1; }
[ -d "$ROOT/yapper_rpc" ] || echo "WARN: $ROOT/yapper_rpc missing — copy the gRPC stubs too (needed at runtime)"
log "jieshuo root: $ROOT   asr-python: $ASR_PYTHON   tts-python: $TTS_PYTHON"
mkdir -p "$ROOT/models" "$ROOT/voices"

# 1. uv
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH after install" >&2; exit 1; }
uv python install "$ASR_PYTHON" "$TTS_PYTHON" 2>/dev/null || true   # ensure interpreters available

# 2. ASR + gpud env
log "syncing ASR/gpud env (server/asr, python $ASR_PYTHON)"
uv sync --python "$ASR_PYTHON" --project "$ROOT/server/asr"

# 3. CosyVoice2 (source install) + TTS env
if [ ! -d "$ROOT/CosyVoice" ]; then
  log "cloning CosyVoice2 (--recursive)"
  git clone --recursive --depth 1 "$COSYVOICE_REPO" "$ROOT/CosyVoice"
else
  log "CosyVoice2 already present — skipping clone"
fi
log "syncing TTS env (server/tts, python $TTS_PYTHON)"
uv sync --python "$TTS_PYTHON" --project "$ROOT/server/tts"
log "installing CosyVoice2 requirements into the TTS env"
# CosyVoice's requirements.txt declares extra indexes (pytorch cu121 + the onnxruntime-cuda
# Azure feed) AND pins protobuf==4.25, which is published only on PyPI. uv's default
# first-index-wins (anti dependency-confusion) finds protobuf on the Azure feed at the wrong
# version and refuses to fall through to PyPI -> "no solution". Allow best-match across the
# (trusted) indexes this requirements file itself declares.
uv pip install --python "$ROOT/server/tts/.venv/bin/python" \
  --index-strategy unsafe-best-match -r "$ROOT/CosyVoice/requirements.txt"
# CosyVoice's deps transitively pull the latest setuptools (>=81), which DROPPED pkg_resources —
# but its lightning==2.2.4 imports pkg_resources at load time. Re-pin <81 LAST so it sticks.
log "re-pinning setuptools<81 (pkg_resources needed by CosyVoice's lightning)"
uv pip install --python "$ROOT/server/tts/.venv/bin/python" "setuptools<81"

# 4. model weights — snapshot_download is resumable + idempotent. Guard on a SENTINEL file
# (llm.pt, the big required weight that lands late), NOT "dir non-empty": an interrupted
# download leaves a partial non-empty dir, so a non-empty check would skip it forever and
# ship a model that can't load. Re-running just fetches the missing files.
if [ -f "$MODEL_DIR/llm.pt" ]; then
  log "weights already complete at $MODEL_DIR (llm.pt present) — skipping download"
else
  log "downloading $MODEL_ID -> $MODEL_DIR (resumes if partial)"
  MODEL_ID="$MODEL_ID" MODEL_DIR="$MODEL_DIR" \
    uv run --no-sync --project "$ROOT/server/tts" python - <<'PY'
import os
from modelscope import snapshot_download
snapshot_download(os.environ["MODEL_ID"], local_dir=os.environ["MODEL_DIR"])
PY
fi

log "setup complete — envs: server/asr, server/tts | weights: $MODEL_DIR"
cat <<EOF

Next steps:
  • (optional) put a cloned-narrator reference clip at $ROOT/voices/narrator.wav
  • if WhisperX hits a cuDNN error at runtime:
      uv pip install --python $ROOT/server/asr/.venv/bin/python nvidia-cudnn-cu12
  • start gpud (on-demand pools); see server/README_deploy.md for full options:
      cd $ROOT && PYTHONPATH=$ROOT HF_TOKEN=hf_xxx \\
        GPUD_PORT_RANGE=50060-50099 GPUD_ASR_MAX=3 GPUD_TTS_MAX=3 \\
        uv run --no-sync --project server/asr python server/gpud.py
EOF
