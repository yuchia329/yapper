"""WhisperX ASR service (deploy on the GPU server, one RTX 3090).

POST /transcribe  (multipart: file=<wav>, form: language, diarize, min/max_speakers)
  -> {"language": str, "segments": [{start,end,text,speaker?,words:[{word,start,end,score}]}]}

Run:
  CUDA_VISIBLE_DEVICES=0 HF_TOKEN=... uvicorn asr_service:app --host 0.0.0.0 --port 8900

See README_deploy.md for the CUDA 12 / cuDNN 9 setup (the #1 source of install pain).
"""

from __future__ import annotations

import os
import tempfile

import torch
import whisperx
from fastapi import FastAPI, File, Form, UploadFile

MODEL = os.environ.get("ASR_MODEL", "large-v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE = os.environ.get("ASR_COMPUTE", "float16" if DEVICE == "cuda" else "float32")
HF_TOKEN = os.environ.get("HF_TOKEN")

app = FastAPI(title="jieshuoforge-asr")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = whisperx.load_model(MODEL, device=DEVICE, compute_type=COMPUTE)
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL, "device": DEVICE}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    diarize: bool = Form(False),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        audio = whisperx.load_audio(path)
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

        if diarize and HF_TOKEN:
            dia = whisperx.diarize.DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
            diarize_segments = dia(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        segments = [
            {
                "start": s["start"],
                "end": s["end"],
                "text": s.get("text", "").strip(),
                "speaker": s.get("speaker"),
                "words": [
                    {"word": w.get("word", ""), "start": w.get("start"), "end": w.get("end"), "score": w.get("score")}
                    for w in s.get("words", [])
                ],
            }
            for s in result["segments"]
        ]
        return {"language": lang, "segments": segments}
    finally:
        os.unlink(path)
