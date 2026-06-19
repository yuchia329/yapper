"""Auto-extract a cloned-narrator reference clip (+ few-shot style text) from a
reference 解说 video.

Why: the recap's voice sounded "too AI" because TTS used CosyVoice2's bundled
placeholder clip. This builds a real narrator reference by transcribing a reference
video (Mandarin) via the GPU WhisperX service, finding the cleanest continuous
narration span, and cutting it out as the zero-shot clone prompt. The transcript
also seeds data/refs/fewshot.txt, which prompts.py injects to fix the script tone.

The reference audio has music/clip bleed under the narration (user accepted this);
for a cleaner clone, optionally isolate vocals with Demucs on the GPU box first
(see --vocals) and point this at the isolated stem.

Usage (needs the SSH tunnel up + ASR service reachable at ASR_SERVER_URL):
    uv run python scripts/build_narrator_ref.py data/RushHour1_final.webm
    uv run python scripts/build_narrator_ref.py data/RushHour1_final.webm --target-len 14

Outputs under data/refs/:
    <name>.zh.json   full WhisperX transcript (Mandarin)
    <name>.zh.txt    full joined narration (inspect / hand-edit few-shots from this)
    fewshot.txt      style few-shots injected into the system prompt (first ~800 chars)
    narrator_ref.wav the clone prompt clip (16 kHz mono)
    narrator_ref.txt its transcript (REF_TEXT)

Then upload narrator_ref.wav to the GPU box and set [tts] reference_clip /
reference_text in config/pipeline.toml (reference_clip is the path ON the server).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

from jieshuoforge.ffmpeg.run import FFMPEG, run  # noqa: E402
from jieshuoforge.schemas import Transcript  # noqa: E402
from jieshuoforge.server_clients.asr_client import ASRClient  # noqa: E402

REFS = REPO / "data" / "refs"


def extract_wav(src: Path, dst: Path) -> None:
    run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(dst)])


def cut_span(src_wav: Path, t0: float, dur: float, dst: Path) -> None:
    run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
         "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", str(src_wav),
         "-ac", "1", "-ar", "16000", str(dst)])


# Traditional-only Chinese glyphs: a 解说 narrator writes Simplified; Traditional text
# is almost always an embedded ORIGINAL movie clip (hardsub dialogue), i.e. the wrong
# voice to clone. Used to skip those segments when picking the narrator reference.
_TRAD = set("們這還麼學員當來時個認譯說讀寫聽愛點擊關註開頭結還沒沒")


def _looks_like_clip(text: str) -> bool:
    return any(c in _TRAD for c in text)


def best_span(tr: Transcript, target_len: float, min_len: float) -> tuple[float, float, str]:
    """Pick a clean narrator reference: a single WHOLE narration segment whose duration
    fits the clone budget (so its transcript matches the audio exactly), preferring the
    longest such. Skips Traditional-text segments (embedded movie clips). Falls back to
    the longest non-clip segment, trimmed to target_len, if none fit."""
    segs = [s for s in tr.segments if s.text.strip() and not _looks_like_clip(s.text)]
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
    ap.add_argument("video", help="reference 解说 video, e.g. data/RushHour1_final.webm")
    ap.add_argument("--asr-url", default=None, help="override ASR_SERVER_URL")
    ap.add_argument("--target-len", type=float, default=14.0, help="clone clip length (s)")
    ap.add_argument("--min-len", type=float, default=8.0)
    ap.add_argument("--language", default="zh")
    args = ap.parse_args()

    load_dotenv(REPO / ".env")
    asr_url = args.asr_url or os.environ.get("ASR_SERVER_URL")
    if not asr_url:
        raise SystemExit("set ASR_SERVER_URL (start the tunnel + ASR service) or pass --asr-url")

    REFS.mkdir(parents=True, exist_ok=True)
    src = Path(args.video).resolve()
    name = src.stem
    full_wav = REFS / f"{name}.16k.wav"
    tr_json = REFS / f"{name}.zh.json"

    print(f"[1/4] extracting audio -> {full_wav.name}")
    extract_wav(src, full_wav)

    if tr_json.exists():
        print(f"[2/4] using cached transcript {tr_json.name}")
        tr = Transcript.load(tr_json)
    else:
        print(f"[2/4] transcribing ({args.language}) via {asr_url} …")
        tr = ASRClient(asr_url).transcribe(full_wav, language=args.language)
        tr.save(tr_json)
    print(f"      {len(tr.segments)} segments")

    joined = " ".join(s.text.strip() for s in tr.segments if s.text.strip())
    (REFS / f"{name}.zh.txt").write_text(joined, encoding="utf-8")
    fewshot = REFS / "fewshot.txt"
    if not fewshot.exists():
        fewshot.write_text(joined[:800].strip(), encoding="utf-8")
        print(f"[3/4] wrote {fewshot.name} ({min(len(joined),800)} chars) — review/trim by hand")
    else:
        print(f"[3/4] {fewshot.name} already exists — left as-is")

    t0, t1, text = best_span(tr, args.target_len, args.min_len)
    ref_wav = REFS / "narrator_ref.wav"
    cut_span(full_wav, t0, t1 - t0, ref_wav)
    (REFS / "narrator_ref.txt").write_text(text, encoding="utf-8")
    print(f"[4/4] clone clip {ref_wav.name}: {t0:.1f}-{t1:.1f}s ({t1-t0:.1f}s)")
    print(f"      REF_TEXT: {text}")
    print()
    print("Next: upload narrator_ref.wav to the GPU box, then set in config/pipeline.toml [tts]:")
    print('  reference_clip = "/abs/server/path/narrator_ref.wav"')
    print(f'  reference_text = "{text}"')


if __name__ == "__main__":
    main()
