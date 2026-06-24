#!/usr/bin/env bash
# Verify the GPU-box setup END-TO-END, WITHOUT the platform (no Redis / Postgres / Celery /
# web). Everything the platform leans on lives on the box, so the box is independently
# checkable. Run it on the box:
#
#   ssh nlp 'bash ~/jieshuo/server/verify_gpu.sh'            # stages 1-4
#   ssh nlp 'RUN_GPUD=1 bash ~/jieshuo/server/verify_gpu.sh' # + the gpud orchestration e2e
#
# Exit 0 iff every check passes. Stages escalate cheap -> expensive:
#   1. inventory  — venvs, CosyVoice source + Matcha-TTS submodule, stubs, weight files
#   2. ASR env    — torch+CUDA, whisperx, NVML, yapper_rpc + grpc_health imports
#   3. TTS env    — load CosyVoice2 + synthesize a clip  (the real proof; writes a WAV)
#   4. ASR engine — faster-whisper/ctranslate2 on GPU, transcribe that clip (round-trip)
#   5. gpud e2e   — Acquire -> gpud launches the TTS service -> Synthesize over the lease
#                   target -> Release   (exactly what a Celery worker does; opt-in RUN_GPUD=1)
#
# Knobs: JIESHUO_ROOT, VERIFY_GPU (CUDA index; default = GPU with most free vRAM),
#        VERIFY_ASR_MODEL (default "tiny"; set "" to skip stage 4), RUN_GPUD=1,
#        TTS_REF_WAV / TTS_REF_TEXT (default: CosyVoice's bundled sample prompt).
set -uo pipefail   # deliberately NOT -e: run every check and tally, don't abort on first fail

ROOT="${JIESHUO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
ASR_PY="$ROOT/server/asr/.venv/bin/python"
TTS_PY="$ROOT/server/tts/.venv/bin/python"
MODEL_DIR="${TTS_MODEL_DIR:-$ROOT/models/CosyVoice2-0.5B}"
export PYTHONPATH="$ROOT:$ROOT/server"      # yapper_rpc (ROOT) + _metrics (server/); cosyvoice self-bootstraps
VERIFY_ASR_MODEL="${VERIFY_ASR_MODEL-tiny}"

PASS=0; FAIL=0
ok()   { printf '  \033[32m\xe2\x9c\x93\033[0m %s\n' "$*"; PASS=$((PASS + 1)); }
bad()  { printf '  \033[31m\xe2\x9c\x97\033[0m %s\n' "$*"; FAIL=$((FAIL + 1)); }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }   # advisory, not a failure
hdr()  { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

command -v nvidia-smi >/dev/null 2>&1 || { echo "no nvidia-smi on PATH — is this the GPU box?"; exit 1; }
if [ -z "${VERIFY_GPU:-}" ]; then
  VERIFY_GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
fi
export CUDA_VISIBLE_DEVICES="$VERIFY_GPU"

# ---- 1. inventory -----------------------------------------------------------
hdr "1. Inventory (files only — instant)"
[ -x "$ASR_PY" ] && ok "ASR venv"                       || bad "ASR venv missing — run setup_gpu.sh"
[ -x "$TTS_PY" ] && ok "TTS venv"                       || bad "TTS venv missing — run setup_gpu.sh"
[ -d "$ROOT/CosyVoice/cosyvoice" ] && ok "CosyVoice source checkout" || bad "CosyVoice/ missing"
[ -d "$ROOT/CosyVoice/third_party/Matcha-TTS" ] && ok "Matcha-TTS submodule" || bad "Matcha-TTS missing — git clone --recursive"
[ -f "$ROOT/yapper_rpc/tts_pb2.py" ] && ok "yapper_rpc stubs" || bad "yapper_rpc/ stubs missing — scp to box"
# Per-file presence is ADVISORY — the authoritative completeness check is whether CosyVoice2
# actually loads (stage 3). A given snapshot may legitimately omit some files (e.g. spk2info.pt,
# used only for pre-registered speakers; zero-shot derives the embedding from the ref clip).
for f in llm.pt hift.pt flow.pt speech_tokenizer_v2.onnx campplus.onnx cosyvoice2.yaml spk2info.pt; do
  [ -e "$MODEL_DIR/$f" ] && ok "weight: $f" || warn "weight not present: $f (stage 3 confirms if it matters)"
done
printf '  GPU under test: %s\n' "$CUDA_VISIBLE_DEVICES"

# ---- 2. ASR env imports -----------------------------------------------------
hdr "2. ASR env — imports, CUDA, NVML, stubs"
"$ASR_PY" - <<'PY'
import sys
try:
    import torch, whisperx, pynvml                                   # noqa: F401
    import yapper_rpc.asr_pb2, yapper_rpc.asr_pb2_grpc               # noqa: F401
    import yapper_rpc.gpud_pb2_grpc                                  # noqa: F401
    from grpc_health.v1 import health_pb2                            # noqa: F401
    pynvml.nvmlInit()
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    print(f"  torch {torch.__version__} cuda={torch.version.cuda} "
          f"gpus_visible={torch.cuda.device_count()}; whisperx + NVML + stubs + health import ok")
except Exception:
    import traceback; traceback.print_exc(); sys.exit(1)
PY
[ $? = 0 ] && ok "ASR env imports + CUDA + NVML" || bad "ASR env check failed (above)"

# ---- 3. TTS env: load CosyVoice2 + synth ------------------------------------
hdr "3. TTS env — load CosyVoice2 + synthesize a clip"
if [ -f "$MODEL_DIR/llm.pt" ]; then     # llm.pt is the sentinel for "weights downloaded"
  TTS_MODEL_DIR="$MODEL_DIR" \
  TTS_REF_WAV="${TTS_REF_WAV:-$ROOT/CosyVoice/asset/zero_shot_prompt.wav}" \
  TTS_REF_TEXT="${TTS_REF_TEXT:-希望你以后能够做的比我还好呦。}" \
  "$TTS_PY" - <<'PY'
import sys, time
try:
    import torch, torchaudio
    import tts_service as T                 # self-bootstraps cosyvoice + matcha onto sys.path
    t0 = time.time(); cv = T.get_model()
    chunks = [o["tts_speech"] for o in
              cv.inference_zero_shot("设置验证，合成一句测试语音。", T.REF_TEXT, T.REF_WAV, stream=False)]
    audio = torch.concat(chunks, dim=1); dur = audio.shape[1] / cv.sample_rate
    assert dur > 0.3, f"audio too short ({dur:.2f}s)"
    torchaudio.save("/tmp/yapper_tts_smoke.wav", audio.cpu(), cv.sample_rate, format="wav")
    print(f"  CosyVoice2 loaded in {time.time()-t0:.1f}s; synthesized {dur:.2f}s @ {cv.sample_rate}Hz "
          f"-> /tmp/yapper_tts_smoke.wav")
except Exception:
    import traceback; traceback.print_exc(); sys.exit(1)
PY
  [ $? = 0 ] && ok "TTS load + synthesis" || bad "TTS load/synth failed (above)"
else
  bad "TTS load skipped — $MODEL_DIR/llm.pt missing; run setup_gpu.sh to (re)download weights"
fi

# ---- 4. ASR engine on GPU ---------------------------------------------------
if [ -n "$VERIFY_ASR_MODEL" ]; then
  hdr "4. ASR engine — faster-whisper/ctranslate2 on GPU (transcribe the clip)"
  VAM="$VERIFY_ASR_MODEL" "$ASR_PY" - <<'PY'
import sys, os
try:
    import torch                                  # noqa: F401 — preloads CUDA/cuDNN into the process
    from faster_whisper import WhisperModel
    wav = "/tmp/yapper_tts_smoke.wav"
    if not os.path.exists(wav):
        print("  no TTS clip from stage 3 — skipping"); sys.exit(2)
    m = WhisperModel(os.environ["VAM"], device="cuda", compute_type="float16")
    segs, info = m.transcribe(wav, language="zh", beam_size=1)   # PyAV decodes the WAV — no ffmpeg binary
    txt = "".join(s.text for s in segs).strip()
    print(f"  ctranslate2 GPU ok | model={os.environ['VAM']} | transcript: {txt[:80] or '(short clip)'}")
except Exception:
    import traceback; traceback.print_exc(); sys.exit(1)
PY
  rc=$?
  if   [ $rc = 0 ]; then ok "ASR engine load + transcription (TTS->ASR round-trip)"
  elif [ $rc = 2 ]; then bad "ASR stage skipped (no clip from stage 3)"
  else bad "ASR engine failed (above) — if it's a cuDNN load error, the env needs cuDNN on LD_LIBRARY_PATH"; fi
fi

# ---- 5. gpud orchestration end-to-end (opt-in) ------------------------------
if [ "${RUN_GPUD:-}" = 1 ]; then
  hdr "5. gpud e2e — Acquire -> launch -> Synthesize over lease -> Release"
  cd "$ROOT"
  env -u CUDA_VISIBLE_DEVICES \
    TTS_MODEL_DIR="$MODEL_DIR" \
    TTS_REF_WAV="${TTS_REF_WAV:-$ROOT/CosyVoice/asset/zero_shot_prompt.wav}" \
    TTS_REF_TEXT="${TTS_REF_TEXT:-希望你以后能够做的比我还好呦。}" \
    GPUD_PORT_RANGE="${GPUD_PORT_RANGE:-50060-50099}" GPUD_ASR_MAX=1 GPUD_TTS_MAX=1 \
    "$ASR_PY" "$ROOT/server/gpud.py" >/tmp/yapper_gpud_verify.log 2>&1 &
  GPUD_PID=$!
  "$ASR_PY" - <<'PY'
import grpc, time, sys
from yapper_rpc import gpud_pb2, gpud_pb2_grpc, tts_pb2, tts_pb2_grpc
g = gpud_pb2_grpc.GpudStub(grpc.insecure_channel("localhost:50050"))
up = False
for _ in range(30):
    try: g.Status(gpud_pb2.StatusRequest(), timeout=3); up = True; break
    except Exception: time.sleep(1)
if not up:
    print("  gpud never came up (see /tmp/yapper_gpud_verify.log)"); sys.exit(1)
try:
    lease = g.Acquire(gpud_pb2.AcquireRequest(service="tts"), timeout=300)   # launches + waits SERVING
    print(f"  Acquire -> target={lease.target} ready={lease.ready}")
    t = tts_pb2_grpc.TtsStub(grpc.insecure_channel(lease.target))
    data = b"".join(c.data for c in t.Synthesize(tts_pb2.SynthesizeRequest(text="网关验证完成。"), timeout=180))
    assert len(data) > 1000, f"too few WAV bytes: {len(data)}"
    open("/tmp/yapper_gpud_smoke.wav", "wb").write(data)
    print(f"  Synthesize over lease -> {len(data)} WAV bytes -> /tmp/yapper_gpud_smoke.wav")
    g.Release(gpud_pb2.LeaseRef(lease_id=lease.lease_id))
    print("  Release ok")
except Exception:
    import traceback; traceback.print_exc(); sys.exit(1)
PY
  rc=$?
  kill "$GPUD_PID" 2>/dev/null; wait "$GPUD_PID" 2>/dev/null
  [ $rc = 0 ] && ok "gpud lease -> launch -> synth -> release" || bad "gpud e2e failed (see /tmp/yapper_gpud_verify.log)"
fi

# ---- summary ----------------------------------------------------------------
hdr "Summary"
printf '  PASS=%d  FAIL=%d\n' "$PASS" "$FAIL"
if [ "$FAIL" = 0 ]; then printf '  \033[1;32mGPU box VERIFIED\033[0m\n'; exit 0
else printf '  \033[1;31mSETUP INCOMPLETE\033[0m — see failures above\n'; exit 1; fi
