"""Stage 10 — audio-driven EDL build (with raw-playback breathing segments).

Two segment kinds:

  - narration: footage conformed to the MEASURED voiceover duration (the line's
    voiceover plays once, ducked under it downstream). A line referencing several
    scenes becomes a montage — its voiceover sliced evenly across them.
  - playback: a window of original footage plays at full original-audio volume with
    NO voiceover (key dialogue / fight / reveal / song / joke). The window is
    grounded to the featured line's transcript span (ASR word timecodes, never an
    LLM timestamp), clamped to [min_window, max_window] — so a long scene shows a
    continuous portion and narration resumes after.

Source video is never time-stretched: narration screen time == measured VO; playback
screen time == the chosen source window.
"""

from __future__ import annotations

import logging
import sqlite3
from difflib import SequenceMatcher

from .. import db
from ..schemas import Edl, EdlSegment, Script, Transcript, VOManifest

log = logging.getLogger("yapper.s10")


def run_stage(
    script: Script,
    vo: VOManifest,
    conn: sqlite3.Connection,
    *,
    fps: float,
    width: int,
    height: int,
    transcript: Transcript | None = None,
    min_window: float = 3.0,
    max_window: float = 12.0,
) -> Edl:
    vo_by_line = {v.line_id: v for v in vo.lines}
    segments: list[EdlSegment] = []
    used_out: dict[str, float] = {}  # clip_id -> last src_out emitted (advance window on reuse)

    for line in script.lines:
        clips = db.resolve_clip_refs(conn, line.clip_refs)

        if line.kind == "playback":
            clip = clips[0]
            src_in, src_out = _playback_window(
                clip, line.quote, transcript, min_w=min_window, max_w=max_window
            )
            segments.append(
                EdlSegment(
                    segment_id=f"seg_{len(segments):03d}",
                    line_id=line.line_id,
                    kind="playback",
                    src_in=round(src_in, 3),
                    src_out=round(src_out, 3),
                    vo_file=None,
                    vo_in=0.0,
                    vo_duration=round(src_out - src_in, 3),
                    subtitle_text=line.text,
                )
            )
            continue

        vo_line = vo_by_line.get(line.line_id)
        if vo_line is None:
            raise KeyError(f"no voiceover for {line.line_id}")
        total = vo_line.measured_duration_sec
        n = len(clips)
        slice_dur = total / n  # split the line's voiceover evenly across its scenes (montage)

        for j, clip in enumerate(clips):
            # last slice absorbs float remainder so the sub-segments sum to `total`
            seg_dur = total - slice_dur * (n - 1) if j == n - 1 else slice_dur
            src_in, src_out = _narration_window(clip, seg_dur, start_after=used_out.get(clip.clip_id))
            used_out[clip.clip_id] = src_out
            segments.append(
                EdlSegment(
                    segment_id=f"seg_{len(segments):03d}",
                    line_id=line.line_id,
                    kind="narration",
                    src_in=round(src_in, 3),
                    src_out=round(src_out, 3),
                    vo_file=vo_line.file,
                    vo_in=round(slice_dur * j, 3),   # offset into this line's voiceover
                    vo_duration=round(seg_dur, 3),
                    subtitle_text=line.text,          # same caption across the line's sub-segments
                )
            )

    edl = Edl(fps=fps, width=width, height=height, segments=segments)
    n_play = sum(1 for s in segments if s.kind == "playback")
    log.info(
        "EDL: %d segments (%d playback) across %d lines, %.1fs total",
        len(segments), n_play, len(script.lines), edl.total_duration,
    )
    return edl


def _narration_window(clip, dur: float, start_after: float | None = None) -> tuple[float, float]:
    """A `dur`-long window inside the scene. Normally centered on the scene's
    representative keyframe (the frame the model saw), so footage shows the narrated
    moment, not the scene's opening. If this clip was already used (reuse across
    lines), advance past the previously-shown window so we don't replay the same
    footage."""
    latest_start = max(clip.t_start, clip.t_end - dur)
    if start_after is not None and start_after < latest_start - 0.05:
        src_in = min(max(start_after, clip.t_start), latest_start)
    elif clip.keyframes:
        kf_t = clip.keyframes[0].t_sec
        src_in = min(max(kf_t - dur / 2, clip.t_start), latest_start)
    else:
        src_in = clip.t_start
    return src_in, min(clip.t_end, src_in + dur)


def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _playback_window(clip, quote: str, transcript: Transcript | None, *, min_w: float, max_w: float) -> tuple[float, float]:
    """Locate the window to play raw. Prefer the transcript segment matching `quote`
    (exact dialogue span); else the scene's whole dialogue span; else a keyframe
    window. Always clamped to the clip and to [min_w, max_w]."""
    segs = []
    if transcript is not None:
        for s in transcript.segments:
            mid = (s.start + s.end) / 2
            if clip.t_start <= mid < clip.t_end and s.text.strip():
                segs.append(s)

    start = end = None
    if segs:
        q = _norm(quote)
        if q:
            best = max(segs, key=lambda s: SequenceMatcher(None, q, _norm(s.text)).ratio())
            if SequenceMatcher(None, q, _norm(best.text)).ratio() >= 0.3:
                start, end = best.start, best.end
        if start is None:  # no good quote match -> whole dialogue span of the scene
            start, end = segs[0].start, segs[-1].end

    if start is None:  # dialogue-free scene -> keyframe window
        return _narration_window(clip, min(max_w, max(min_w, clip.duration)))

    start = max(start, clip.t_start)
    end = min(end, clip.t_end)
    dur = end - start
    if dur < min_w:  # pad symmetrically within the clip up to min_w
        want = min(min_w, clip.duration)
        pad = (want - dur) / 2
        start = max(clip.t_start, start - pad)
        end = min(clip.t_end, start + want)
        start = max(clip.t_start, end - want)
    if end - start > max_w:
        end = start + max_w
    return start, end
