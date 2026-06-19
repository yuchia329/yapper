"""CosyVoice2 TTS service (deploy on the GPU server, one RTX 3090).

Holds ONE pinned, cloned narrator voice (zero-shot from a reference clip + its
transcript, set via env at startup). POST text -> WAV bytes.

POST /synthesize  {"text": str, "seed": int, "speed": float}  -> audio/wav

Run:
  CUDA_VISIBLE_DEVICES=1 \
  TTS_MODEL_DIR=pretrained_models/CosyVoice2-0.5B \
  TTS_REF_WAV=/voices/narrator.wav TTS_REF_TEXT="参考音频的文字内容" \
  uvicorn tts_service:app --host 0.0.0.0 --port 8901

See README_deploy.md for CosyVoice2 install + model download (it is a source install
with a Matcha-TTS submodule, not a plain pip package).
"""

from __future__ import annotations

import io
import os

import torch
import torchaudio
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from cosyvoice.cli.cosyvoice import CosyVoice2 as _CosyVoice
from cosyvoice.utils.common import set_all_random_seed

MODEL_DIR = os.environ.get("TTS_MODEL_DIR", "pretrained_models/CosyVoice2-0.5B")
# No clip yet? Default to CosyVoice2's bundled sample prompt so the service runs
# out-of-the-box for the first end-to-end test. Swap TTS_REF_WAV/TTS_REF_TEXT for
# the real cloned narrator later.
REF_WAV = os.environ.get("TTS_REF_WAV", "CosyVoice/asset/zero_shot_prompt.wav")
REF_TEXT = os.environ.get("TTS_REF_TEXT", "希望你以后能够做的比我还好呦。")
VOICE_ID = os.environ.get("TTS_VOICE_ID", "placeholder" if "TTS_REF_WAV" not in os.environ else "narrator")

app = FastAPI(title="jieshuoforge-tts")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = _CosyVoice(model_dir=MODEL_DIR, fp16=True)
    return _model


class SynthReq(BaseModel):
    text: str
    seed: int = 42
    speed: float = 1.0
    ref_wav: str | None = None   # server-side path to a reference clip; falls back to REF_WAV
    ref_text: str | None = None  # transcript of ref_wav; falls back to REF_TEXT


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "voice": VOICE_ID, "sample_rate": get_model().sample_rate}


@app.post("/synthesize")
def synthesize(req: SynthReq) -> Response:
    cv = get_model()
    set_all_random_seed(req.seed)  # pin timbre/prosody so the voice never drifts
    ref_wav = req.ref_wav or REF_WAV
    ref_text = req.ref_text or REF_TEXT
    chunks = [
        out["tts_speech"]
        for out in cv.inference_zero_shot(req.text, ref_text, ref_wav, stream=False, speed=req.speed)
    ]
    audio = torch.concat(chunks, dim=1)
    buf = io.BytesIO()
    torchaudio.save(buf, audio, cv.sample_rate, format="wav")
    return Response(content=buf.getvalue(), media_type="audio/wav")
