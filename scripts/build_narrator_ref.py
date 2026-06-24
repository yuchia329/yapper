"""Auto-extract a cloned-narrator reference clip (+ few-shot style text) from a
reference 解說 video.

Why: the commentary's voice sounded "too AI" because TTS used CosyVoice2's bundled
placeholder clip. This builds a real narrator reference by transcribing a reference
video (Mandarin) via the GPU WhisperX service, finding the cleanest continuous
narration span, and cutting it out as the zero-shot clone prompt. The transcript
also seeds data/refs/fewshot.txt, which prompts.py injects to fix the script tone.

The reference audio has music/clip bleed under the narration (user accepted this);
for a cleaner clone, optionally isolate vocals with Demucs on the GPU box first
(see --vocals) and point this at the isolated stem.

Works for any narration language via --language: pass an English reference video to
build a natural English narrator voice (CosyVoice2 zero-shot clones the timbre, so a
native-English reference makes the commentary sound natural rather than accented).

Usage (needs the SSH tunnel up + ASR service reachable at ASR_SERVER_URL):
    uv run python scripts/build_narrator_ref.py data/RushHour1_final.webm           # Mandarin
    uv run python scripts/build_narrator_ref.py some_english_narration.mp4 --language en

Outputs under data/refs/ (suffixed by language; zh keeps legacy names):
    <name>.<lang>.json     full WhisperX transcript
    <name>.<lang>.txt      full joined narration (inspect / hand-edit few-shots from this)
    fewshot[.<lang>].txt   style few-shots injected into the system prompt (first ~800 chars)
    narrator_ref[.<lang>].wav  the clone prompt clip (16 kHz mono)
    narrator_ref[.<lang>].txt  its transcript (REF_TEXT)

Then upload the clip to the GPU box and set reference_clip / reference_text in
config/pipeline.toml — under [tts] for zh, or [tts.<lang>] (e.g. [tts.en]) for other
languages (reference_clip is the path ON the server).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

from yapper.ffmpeg.run import FFMPEG, run  # noqa: E402
from yapper.schemas import Transcript  # noqa: E402
from yapper.server_clients.asr_client import ASRClient  # noqa: E402

REFS = REPO / "data" / "refs"


def extract_wav(src: Path, dst: Path) -> None:
    run(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dst),
        ]
    )


def cut_span(src_wav: Path, t0: float, dur: float, dst: Path) -> None:
    run(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{t0:.3f}",
            "-t",
            f"{dur:.3f}",
            "-i",
            str(src_wav),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dst),
        ]
    )


# Traditional-only Chinese glyphs: a 解說 narrator writes Simplified; Traditional text
# is almost always an embedded ORIGINAL movie clip (hardsub dialogue), i.e. the wrong
# voice to clone. Used to skip those segments when picking the narrator reference.
_TRAD = set("們這還麼學員當來時個認譯說讀寫聽愛點擊關註開頭結還沒沒")


def _looks_like_clip(text: str) -> bool:
    return any(c in _TRAD for c in text)


def best_span(
    tr: Transcript, target_len: float, min_len: float, lang: str = "zh"
) -> tuple[float, float, str]:
    """Pick a clean narrator reference: a single WHOLE narration segment whose duration
    fits the clone budget (so its transcript matches the audio exactly), preferring the
    longest such. For zh, skips Traditional-text segments (embedded movie clips). Falls
    back to the longest segment, trimmed to target_len, if none fit."""
    if lang == "zh":
        segs = [
            s for s in tr.segments if s.text.strip() and not _looks_like_clip(s.text)
        ]
    else:
        segs = [s for s in tr.segments if s.text.strip()]
    if not segs:
        raise SystemExit("no clean narration segments in transcript")
    hi = target_len * 1.25
    in_range = [s for s in segs if min_len <= (s.end - s.start) <= hi]
    if in_range:
        s = max(in_range, key=lambda s: s.end - s.start)
        return s.start, s.end, s.text.strip()
    # nothing in range -> take the longest segment and trim to target_len
    s = max(segs, key=lambda s: s.end - s.start)
    return s.start, min(s.end, s.start + target_len), s.text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "video", help="reference 解說 video, e.g. data/RushHour1_final.webm"
    )
    ap.add_argument("--asr-url", default=None, help="override ASR_SERVER_URL")
    ap.add_argument(
        "--target-len", type=float, default=14.0, help="clone clip length (s)"
    )
    ap.add_argument("--min-len", type=float, default=8.0)
    ap.add_argument("--language", default="zh")
    args = ap.parse_args()

    load_dotenv(REPO / ".env")
    lang = args.language.lower()
    asr_url = args.asr_url or os.environ.get("ASR_SERVER_URL")
    if not asr_url:
        raise SystemExit(
            "set ASR_SERVER_URL (start the tunnel + ASR service) or pass --asr-url"
        )

    REFS.mkdir(parents=True, exist_ok=True)
    src = Path(args.video).resolve()
    name = src.stem
    full_wav = REFS / f"{name}.16k.wav"
    tr_json = REFS / f"{name}.{lang}.json"

    print(f"[1/4] extracting audio -> {full_wav.name}")
    extract_wav(src, full_wav)

    if tr_json.exists():
        print(f"[2/4] using cached transcript {tr_json.name}")
        tr = Transcript.load(tr_json)
    else:
        print(f"[2/4] transcribing ({lang}) via {asr_url} …")
        tr = ASRClient(asr_url).transcribe(full_wav, language=lang)
        tr.save(tr_json)
    print(f"      {len(tr.segments)} segments")

    joined = " ".join(s.text.strip() for s in tr.segments if s.text.strip())
    (REFS / f"{name}.{lang}.txt").write_text(joined, encoding="utf-8")
    # prompts.py reads fewshot.<lang>.txt (zh also falls back to the legacy fewshot.txt).
    fewshot = REFS / ("fewshot.txt" if lang == "zh" else f"fewshot.{lang}.txt")
    if not fewshot.exists():
        fewshot.write_text(joined[:800].strip(), encoding="utf-8")
        print(
            f"[3/4] wrote {fewshot.name} ({min(len(joined),800)} chars) — review/trim by hand"
        )
    else:
        print(f"[3/4] {fewshot.name} already exists — left as-is")

    t0, t1, text = best_span(tr, args.target_len, args.min_len, lang=lang)
    ref_stem = "narrator_ref" if lang == "zh" else f"narrator_ref.{lang}"
    ref_wav = REFS / f"{ref_stem}.wav"
    cut_span(full_wav, t0, t1 - t0, ref_wav)
    (REFS / f"{ref_stem}.txt").write_text(text, encoding="utf-8")
    print(f"[4/4] clone clip {ref_wav.name}: {t0:.1f}-{t1:.1f}s ({t1-t0:.1f}s)")
    print(f"      REF_TEXT: {text}")
    print()
    section = "[tts]" if lang == "zh" else f"[tts.{lang}]"
    print(
        f"Next: upload {ref_wav.name} to the GPU box, then set in config/pipeline.toml {section}:"
    )
    print(f'  reference_clip = "/abs/server/path/{ref_wav.name}"')
    print(f'  reference_text = "{text}"')


if __name__ == "__main__":
    main()
