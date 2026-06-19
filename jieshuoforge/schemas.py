"""Typed artifacts exchanged between pipeline stages.

Every stage reads one artifact and writes another. These Pydantic models are the
contracts; they validate on load/save so a malformed hand-off fails loudly and
early. The two artifacts Claude produces (Screenplay, Script) double as JSON
schemas for the Anthropic structured-output API via ``model_json_schema()``.

Timecodes are ALWAYS seconds (float), never frame numbers — VFR films drift on
frame-based cuts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------


class Artifact(BaseModel):
    """Base for any artifact serialized to disk as JSON."""

    model_config = ConfigDict(extra="forbid")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls: type[_A], path: str | Path) -> _A:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


_A = TypeVar("_A", bound=Artifact)


# ---------------------------------------------------------------------------
# s00 — ingest / probe
# ---------------------------------------------------------------------------


class SubtitleStream(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    codec: str
    language: str | None = None
    is_text: bool  # text (srt/ass/subrip) can be copied losslessly; image (pgs/dvd) needs OCR


class AudioStream(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    codec: str
    channels: int
    language: str | None = None


class ProbeManifest(Artifact):
    source_path: str
    container: str
    duration_sec: float
    width: int
    height: int
    fps: float
    is_vfr: bool
    video_codec: str
    audio_streams: list[AudioStream] = Field(default_factory=list)
    subtitle_streams: list[SubtitleStream] = Field(default_factory=list)

    @property
    def best_text_subtitle(self) -> SubtitleStream | None:
        """A copyable text subtitle track, if any (lets us skip ASR)."""
        return next((s for s in self.subtitle_streams if s.is_text), None)


# ---------------------------------------------------------------------------
# s02 — transcript (ground truth for dialogue)
# ---------------------------------------------------------------------------


class TranscriptWord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    start: float
    end: float
    score: float | None = None


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[TranscriptWord] = Field(default_factory=list)


class Transcript(Artifact):
    language: str
    source: Literal["embedded_sub", "asr"]
    segments: list[TranscriptSegment] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# s03 — shots
# ---------------------------------------------------------------------------


class Shot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class ShotList(Artifact):
    shots: list[Shot] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# s04 — scenes + keyframes -> clip index (also persisted to SQLite via db.py)
# ---------------------------------------------------------------------------


class Keyframe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    t_sec: float
    score: float


class Clip(BaseModel):
    """A narrative scene: the unit the LLM grounds narration against.

    ``clip_id`` is the stable handle the whole pipeline references (e.g. ``clip_0042``).
    """

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    scene_index: int
    t_start: float
    t_end: float
    shot_indices: list[int] = Field(default_factory=list)
    speaker: str | None = None
    dialogue_text: str = ""
    keyframes: list[Keyframe] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


# ---------------------------------------------------------------------------
# s06 — plot understanding (Claude MAP pass)
# ---------------------------------------------------------------------------


class Character(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Character name as spoken in dialogue (anchors identity)")
    description: str = Field(description="Who they are and their role in the plot")


class Beat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beat_id: str = Field(description="Stable id, e.g. beat_01")
    summary: str = Field(description="What happens in this story beat (plot, not narration)")
    clip_refs: list[str] = Field(description="clip_ids whose footage covers this beat")
    est_spoken_seconds: float = Field(description="Rough narration time this beat needs")
    importance: int = Field(description="1=droppable subplot ... 5=main throughline", ge=1, le=5)


class Screenplay(Artifact):
    """Output of the MAP pass: the condensed plot skeleton."""

    logline: str = Field(description="One-sentence hook for the whole recap")
    characters: list[Character] = Field(description="Main characters anchored to spoken names")
    beats: list[Beat] = Field(description="Ordered story beats on the 主线 throughline")


# ---------------------------------------------------------------------------
# s07 — 解说 narration (Claude REDUCE pass)
# ---------------------------------------------------------------------------


class ScriptLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_id: str = Field(description="Stable id, e.g. line_001")
    kind: Literal["narration", "playback"] = Field(
        default="narration",
        description="narration=念解说词，配音盖在画面上；playback=放一小段原片原声，不配解说",
    )
    text: str = Field(
        description="narration: 要念出来的中文解说词；playback: 这段原片要显示的中文字幕（通常是原台词的翻译）"
    )
    clip_refs: list[str] = Field(
        description="clip_ids whose footage plays under this line（playback 只填一个）"
    )
    quote: str = Field(
        default="",
        description="仅 playback 用：要原声播放的那句原片台词原文（从该场景对白里照抄，用于定位精确片段）；narration 留空",
    )
    importance: int = Field(
        default=3,
        ge=1,
        le=5,
        description="1=可快速略过的次要铺垫 … 5=主线高潮/名场面（决定该句给多少篇幅与画面时间）",
    )
    est_spoken_seconds: float = Field(
        default=0.0, description="narration 的大致口播时长；playback 填 0"
    )


class Script(Artifact):
    platform: str = "douyin"
    lines: list[ScriptLine] = Field(default_factory=list)

    @property
    def total_est_seconds(self) -> float:
        return sum(line.est_spoken_seconds for line in self.lines)


# ---------------------------------------------------------------------------
# s09 — TTS voiceover manifest
# ---------------------------------------------------------------------------


class VOLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_id: str
    text: str
    file: str
    measured_duration_sec: float
    words: list[TranscriptWord] = Field(default_factory=list)  # for subtitle alignment


class VOManifest(Artifact):
    voice_id: str
    seed: int
    lines: list[VOLine] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# s10 — edit decision list (audio-driven)
# ---------------------------------------------------------------------------


class EdlSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str
    line_id: str
    kind: Literal["narration", "playback"] = "narration"
    src_in: float  # source movie in-point (seconds)
    src_out: float  # source movie out-point (seconds)
    vo_file: str | None = None  # None for playback (original audio plays, no voiceover)
    vo_in: float = 0.0  # offset into vo_file (for montage sub-segments sharing one line's voiceover)
    vo_duration: float  # this segment's screen time; for playback == the source window length
    subtitle_text: str

    @property
    def screen_duration(self) -> float:
        return self.vo_duration


class Edl(Artifact):
    fps: float
    width: int
    height: int
    segments: list[EdlSegment] = Field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return sum(s.screen_duration for s in self.segments)


# ---------------------------------------------------------------------------
# Claude structured-output schemas (strict JSON for output_config.format)
# ---------------------------------------------------------------------------


def screenplay_json_schema() -> dict:
    return Screenplay.model_json_schema()


def script_json_schema() -> dict:
    return Script.model_json_schema()


def _pretty(obj: dict) -> str:  # tiny helper for debugging / prompt embedding
    return json.dumps(obj, ensure_ascii=False, indent=2)
