# GPU-server deployment (`ssh nlp` — nlp-gpu-01, 6× RTX 3090)

Two **gRPC** inference services the orchestrator/workers call (over an SSH tunnel):
**CosyVoice2** (TTS, port 50052) and **WhisperX** (ASR, port 50051). Each also exposes a
plain-HTTP `/metrics` side port (9102 / 9101) for Prometheus. Each runs in its own
**uv project** under `~/jieshuo/server/`, pinned to a free GPU. No conda, no sudo, no
hand-rolled venvs — `uv sync` builds the env, `uv run` executes.

Layout created on the server:
```
~/jieshuo/
  CosyVoice/            # cloned repo (--recursive) + asset/zero_shot_prompt.wav
  server/               # scp'd from this repo
    asr/  pyproject.toml # uv sync -> server/asr/.venv   (WhisperX + gpud + NVML)
    tts/  pyproject.toml # uv sync -> server/tts/.venv   (+ uv pip install CosyVoice reqs)
    asr_service.py tts_service.py gpud.py _metrics.py
  yapper_rpc/           # generated gRPC stubs (scp'd from this repo; on PYTHONPATH)
  models/CosyVoice2-0.5B# downloaded weights
  voices/               # your cloned-narrator reference clip goes here
```
> Copy `server/` and `yapper_rpc/` from this repo to `~/jieshuo/`, and run from `~/jieshuo`
> with `PYTHONPATH=~/jieshuo` so `import yapper_rpc` and `import _metrics` resolve (the
> uv envs manage third-party deps; the code + stubs ride PYTHONPATH).

## One-time setup (uv)

**One command** (idempotent — installs uv, syncs both envs, clones CosyVoice2 + installs
its reqs, downloads weights):
```bash
# from your machine: push code + stubs, then run the setup script on the box
scp -r server yapper_rpc nlp:~/jieshuo/
ssh nlp 'bash ~/jieshuo/server/setup_gpu.sh'
```

Or the equivalent steps by hand:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # uv
scp -r server yapper_rpc nlp:~/jieshuo/                  # (from your machine)

# ASR + gpud env — Python 3.12 (whisperx's onnxruntime has no 3.10 wheels)
uv sync --python 3.12 --project ~/jieshuo/server/asr

# TTS env: CosyVoice2 is a SOURCE install — keep Python 3.10 for its deps
cd ~/jieshuo && git clone --recursive --depth 1 https://github.com/FunAudioLLM/CosyVoice.git
uv sync --python 3.10 --project ~/jieshuo/server/tts
# --index-strategy unsafe-best-match: CosyVoice's reqs declare extra indexes (pytorch cu121 +
# onnxruntime Azure) but pin protobuf==4.25, which lives only on PyPI — let uv span all of them.
uv pip install --python ~/jieshuo/server/tts/.venv/bin/python \
  --index-strategy unsafe-best-match -r ~/jieshuo/CosyVoice/requirements.txt
# weights -> ~/jieshuo/models/CosyVoice2-0.5B (modelscope comes from CosyVoice's reqs)
uv run --no-sync --project ~/jieshuo/server/tts python -c \
  "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='$HOME/jieshuo/models/CosyVoice2-0.5B')"
```
> The server has **no ffmpeg/sox**; the TTS service only handles WAV via `soundfile`,
> and loudness normalization happens on the orchestrator. WhisperX bundles its own audio
> loading. If WhisperX hits a cuDNN error at runtime, add cuDNN 9 to its env:
> `uv pip install --python ~/jieshuo/server/asr/.venv/bin/python nvidia-cudnn-cu12`.

## Run — option A: on-demand pools via gpud (recommended)

`gpud` is a tiny always-on supervisor (no model loaded). It launches ASR/TTS instances on
GPUs that have free vRAM **only when a worker leases one**, and reaps them after an idle
grace — so vRAM is held only while there's work, up to `GPUD_ASR_MAX` / `GPUD_TTS_MAX`
instances. Install `nvidia-ml-py` in the gpud env (NVML), plus grpcio/health/prometheus.

gpud runs in the ASR uv env (NVML lives there) and launches each service via `uv run`
(its `GPUD_ASR_CMD` / `GPUD_TTS_CMD` default to `uv run --no-sync --project server/{asr,tts}
python server/{asr,tts}_service.py`). It passes its own env (HF_TOKEN, model dirs, ref wav)
to children plus CUDA_VISIBLE_DEVICES + the assigned port.
```bash
cd ~/jieshuo
HF_TOKEN=hf_xxx ASR_MODEL=large-v3 \
TTS_REF_WAV=~/jieshuo/voices/narrator.wav TTS_REF_TEXT="参考音频的文字" \
GPUD_PORT_RANGE=50060-50099 GPUD_ASR_MAX=3 GPUD_TTS_MAX=3 \
PYTHONPATH=~/jieshuo uv run --no-sync --project server/asr python server/gpud.py   # gpud -> :50050 (metrics :9050)
```
The platform sets `GPU_SUPERVISOR_TARGET=…:50050`; workers lease per task. Raising
`GPUD_ASR_MAX`/`GPUD_TTS_MAX` (within the port range) needs no tunnel change.

### Idempotent start + shared gpud (one supervisor for prod *and* local dev)
gpud is a shared pool — there should only ever be ONE. Put the launch env in
`~/jieshuo/gpud.env` on the box (gitignored there; same vars as the inline command above)
and use the idempotent launcher, which no-ops if gpud is already up:
```bash
# ~/jieshuo/gpud.env  (on the box)
HF_TOKEN=hf_xxx
ASR_MODEL=large-v3
TTS_REF_WAV=/home/you/jieshuo/voices/narrator.wav
TTS_REF_TEXT=参考音频的文字
# optional: GPUD_ASR_MAX / GPUD_TTS_MAX / GPUD_PORT_RANGE overrides

scp server/start_gpud.sh nlp:~/jieshuo/server/      # one-time
ssh nlp 'bash ~/jieshuo/server/start_gpud.sh'        # starts gpud only if not already running
```
`scripts/gpu_tunnel.sh` calls this automatically before opening the tunnel (see below), so a
local `docker compose` dev session shares the box's gpud rather than launching a second one.

## Run — option B: always-on servers (manual)
```bash
# each in its own shell / tmux; run from ~/jieshuo, PYTHONPATH so yapper_rpc + _metrics resolve:
cd ~/jieshuo
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=~/jieshuo TTS_GRPC_PORT=50052 TTS_METRICS_PORT=9102 \
    uv run --no-sync --project server/tts python server/tts_service.py     # CosyVoice2 -> :50052
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=~/jieshuo ASR_MODEL=large-v3 HF_TOKEN=hf_xxx \
    ASR_GRPC_PORT=50051 ASR_METRICS_PORT=9101 \
    uv run --no-sync --project server/asr python server/asr_service.py     # WhisperX -> :50051
```
The TTS service defaults to the bundled `asset/zero_shot_prompt.wav` placeholder voice.
Swap in your cloned narrator later:
```bash
TTS_REF_WAV=~/jieshuo/voices/narrator.wav TTS_REF_TEXT="参考音频的文字" \
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=~/jieshuo \
  uv run --no-sync --project server/tts python server/tts_service.py
```
> ⚠️ Cloning a real person's voice carries legal/ToS risk independent of the model
> license. Prefer your own or a licensed recording.

## Reaching the services (SSH tunnel)
The box is reached via SSH port-forwarding. With gpud, forward gpud + the instance range
once (a helper generates the flags); the pool can grow within the range with no re-tunnel:
```bash
bash scripts/gpu_tunnel.sh nlp   # ensure gpud (idempotent) + open a BACKGROUND tunnel (50050 + pool 50060-50099 + metrics 9050), then return
# then: export GPU_SUPERVISOR_TARGET=localhost:50050   (BIND=0.0.0.0 -> host.docker.internal:50050 for compose-local)
bash scripts/gpu_tunnel.sh down  # tear that background tunnel back down
# Re-running is safe (no-op if the tunnel is already up). Knobs: ENSURE_GPUD=0 skips the gpud
# auto-start; `bash scripts/gpu_tunnel.sh fg` runs the tunnel in the FOREGROUND instead (blocks).
```
Prod runs the same forwards under an autossh sidecar in the worker-asr/worker-tts pods
(see deploy/k8s/overlays/prod/patch-gpu-sidecar.yaml). Always-on (option B) dev tunnel:
```bash
ssh -N -L 50051:localhost:50051 -L 50052:localhost:50052 nlp
# then ASR_GRPC_TARGET=localhost:50051  TTS_GRPC_TARGET=localhost:50052
```
Verify with grpcurl (or the client's `health()`):
```bash
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check
grpcurl -plaintext localhost:50052 list
curl -s localhost:9101/metrics | head   # ASR Prometheus side port
```

## Scaling across the 6 GPUs (later)
Run multiple TTS servers on different `CUDA_VISIBLE_DEVICES`/ports behind a gRPC-aware
round-robin (headless Service + client-side LB), and a second ASR server for batch
throughput. The clients are stateless — point them at the LB target.
