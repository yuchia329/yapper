# GPU-server deployment (`ssh nlp` — nlp-gpu-01, 6× RTX 3090)

Two FastAPI inference services the Mac orchestrator calls over an SSH tunnel:
**CosyVoice2** (TTS, port 8901) and **WhisperX** (ASR, port 8900). Each runs in its
own `uv` venv under `~/jieshuo` and is pinned to a free GPU. No conda, no sudo.

Layout created on the server:
```
~/jieshuo/
  CosyVoice/            # cloned repo (--recursive) + .venv (py3.10) + asset/zero_shot_prompt.wav
  asr/.venv             # WhisperX venv (py3.10)
  server/               # tts_service.py, asr_service.py (scp'd from this repo)
  models/CosyVoice2-0.5B# downloaded weights
  voices/               # your cloned-narrator reference clip goes here
  run_tts.sh run_asr.sh # launchers (set PYTHONPATH/env/GPU, exec uvicorn)
```

## One-time setup (already scripted above, recorded here for reproducibility)
```bash
# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# TTS: CosyVoice2
cd ~/jieshuo && git clone --recursive --depth 1 https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice && uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt soundfile "uvicorn[standard]"
# weights -> ~/jieshuo/models/CosyVoice2-0.5B
.venv/bin/python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='$HOME/jieshuo/models/CosyVoice2-0.5B')"

# ASR: WhisperX
cd ~/jieshuo/asr && uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python whisperx fastapi "uvicorn[standard]" python-multipart
```
> The server has **no ffmpeg/sox**; the TTS service only handles WAV via `soundfile`,
> and loudness normalization happens on the Mac. WhisperX bundles its own audio
> loading. If WhisperX hits a cuDNN error at runtime, install matching cuDNN 9 into
> its venv (`uv pip install nvidia-cudnn-cu12`).

## Run
```bash
# on the server (each in its own shell / tmux):
CUDA_VISIBLE_DEVICES=2 ~/jieshuo/run_tts.sh     # CosyVoice2 -> :8901
CUDA_VISIBLE_DEVICES=3 ASR_MODEL=large-v3 HF_TOKEN=hf_xxx ~/jieshuo/run_asr.sh  # WhisperX -> :8900
```
The TTS launcher defaults to the bundled `asset/zero_shot_prompt.wav` placeholder
voice. Swap in your cloned narrator later:
```bash
TTS_REF_WAV=~/jieshuo/voices/narrator.wav TTS_REF_TEXT="参考音频的文字" CUDA_VISIBLE_DEVICES=2 ~/jieshuo/run_tts.sh
```
> ⚠️ Cloning a real person's voice carries legal/ToS risk independent of the model
> license. Prefer your own or a licensed recording.

## Tunnel to the Mac
```bash
ssh -N -L 8900:localhost:8900 -L 8901:localhost:8901 nlp
```
Then the Mac `.env` already points at `http://localhost:8900` / `:8901`. Verify:
```bash
bash scripts/setup_check.sh   # checks /health on both
```

## Scaling across the 6 GPUs (later)
Run multiple TTS workers on different `CUDA_VISIBLE_DEVICES` behind a round-robin to
parallelize per-line synthesis; add a second ASR worker for batch throughput. The
Mac clients are stateless — point them at a load balancer.
