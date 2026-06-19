"""Mac-side server clients + s09 validated without a real GPU server:
the HTTP contract is exercised with httpx.MockTransport, and s09 runs end-to-end
with a synthesizer that emits real WAVs via ffmpeg."""

from pathlib import Path

import httpx

from jieshuoforge.ffmpeg.run import FFMPEG, run
from jieshuoforge.schemas import Script, ScriptLine
from jieshuoforge.server_clients.asr_client import ASRClient
from jieshuoforge.server_clients.tts_client import TTSClient
from jieshuoforge.stages import s09_tts


def test_asr_client_parses_words_and_speakers():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/transcribe"
        return httpx.Response(
            200,
            json={
                "language": "en",
                "segments": [
                    {
                        "start": 1.0, "end": 4.0, "text": "Hello there.", "speaker": "SPEAKER_00",
                        "words": [
                            {"word": "Hello", "start": 1.0, "end": 1.5, "score": 0.9},
                            {"word": "there.", "start": 1.6, "end": 2.0, "score": 0.8},
                        ],
                    }
                ],
            },
        )

    client = ASRClient("http://gpu:8900", client=httpx.Client(transport=httpx.MockTransport(handler)))
    # any existing file works as the upload body
    tr = client.transcribe(__file__, language="en")
    assert tr.source == "asr" and tr.language == "en"
    assert tr.segments[0].text == "Hello there."
    assert tr.segments[0].speaker == "SPEAKER_00"
    assert tr.segments[0].words[0].text == "Hello"  # 'word' key normalized to .text


def test_tts_client_writes_returned_audio(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/synthesize"
        return httpx.Response(200, content=b"RIFFfakewavbytes")

    client = TTSClient("http://gpu:8901", client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = client.synthesize("你好", tmp_path / "l0.wav", seed=7)
    assert out.read_bytes() == b"RIFFfakewavbytes"


class _SineSynth:
    """Stand-in TTS: emits a real WAV whose length scales with text length."""

    def synthesize(self, text, out_path, *, seed=42, speed=1.0):
        dur = max(0.5, len(text) * 0.15)
        run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", f"sine=frequency=300:duration={dur}:sample_rate=48000", "-ac", "1", str(out_path)])
        return Path(out_path)


def test_s09_builds_voiceover_manifest_with_measured_durations(tmp_path: Path):
    script = Script(lines=[
        ScriptLine(line_id="line_000", text="第一句旁白", clip_refs=["clip_0000"], est_spoken_seconds=2),
        ScriptLine(line_id="line_001", text="这是更长一些的第二句旁白内容", clip_refs=["clip_0001"], est_spoken_seconds=4),
    ])
    vo = s09_tts.run_stage(script, _SineSynth(), tmp_path / "vo", voice_id="narrator", seed=1)
    assert vo.voice_id == "narrator" and vo.seed == 1
    assert len(vo.lines) == 2
    assert all(Path(l.file).exists() and l.measured_duration_sec > 0 for l in vo.lines)
    # longer text -> longer measured voiceover
    assert vo.lines[1].measured_duration_sec > vo.lines[0].measured_duration_sec
