"""EDL -> ffmpeg command translator (the ONLY place filter_complex is built).

Strategy (per the plan): normalize every segment to one canonical profile, THEN
concat — frame-accurate cuts at arbitrary points require a re-encode, and concat
is only safe across identical codec params. Each segment is rendered with:
  - video: source window scaled/padded to the canonical frame, freeze-padded to
    the voiceover duration (footage conformed to audio, never time-stretched)
  - audio: original movie audio ducked under the voiceover (sidechaincompress),
    then mixed with the voiceover and limited.
Subtitle burn-in (when enabled) is folded into each segment's encode via the optional
``ass_path`` on ``normalize_segment_cmd`` — one cue per segment, no separate full-length
pass; needs an ffmpeg built with libass. The stream-copy concat then carries it through.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..schemas import Edl, EdlSegment
from .run import FFMPEG, run

log = logging.getLogger("yapper.ffmpeg.graph")


def _f(x: float) -> str:
    return f"{x:.3f}"


def normalize_segment_cmd(
    movie_path: str,
    seg: EdlSegment,
    out_path: Path,
    *,
    width: int,
    height: int,
    fps: float,
    vcodec: str,
    pix_fmt: str,
    audio_rate: int,
    ducking: dict,
    target_lufs: float = -16.0,
    score_stem: str | None = None,
    bed_gain_db: float = -14.0,
    ass_path: str | Path | None = None,
    fonts_dir: str | Path | None = None,
    preset: str | None = None,
    crf: int | None = None,
) -> list[str]:
    span = max(0.1, seg.src_out - seg.src_in)
    vd = seg.vo_duration
    # libx264 speed/quality knobs (the big render-time lever). Only meaningful for x264; other
    # encoders (nvenc/videotoolbox) reject -preset/-crf, so gate on the codec.
    venc = ["-c:v", vcodec]
    if vcodec in ("libx264", "libx265") and preset:
        venc += ["-preset", str(preset)]
    if vcodec in ("libx264", "libx265") and crf is not None:
        venc += ["-crf", str(crf)]

    # Optional subtitle burn-in DURING the segment encode (one cue spanning [0, vd] in the
    # segment's local timeline). Doing it here avoids a second full-length re-encode pass —
    # the stream-copy concat then carries the burned-in subs through to the final video.
    sub = ""
    if ass_path is not None:
        s = f"subtitles={_escape(str(ass_path))}"
        if fonts_dir is not None:
            s += f":fontsdir={_escape(str(fonts_dir))}"
        sub = f",{s}"

    # shared video conform: scale/pad to the canonical frame, freeze-pad to the
    # segment duration (footage conformed to audio, never time-stretched).
    vfc = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
        f"tpad=stop_mode=clone:stop_duration={_f(vd)},trim=duration={_f(vd)},"
        f"setpts=PTS-STARTPTS{sub}[v];"
    )

    if seg.kind == "playback":
        # raw playback: original audio at full volume, no voiceover, no ducking.
        fc = (
            vfc
            + f"[0:a]aresample=async=1,apad,atrim=duration={_f(vd)},asetpts=PTS-STARTPTS,"
            f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11,alimiter=limit=0.95,aresample={audio_rate}[a]"
        )
        return [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", _f(seg.src_in), "-t", _f(span), "-i", movie_path,
            "-filter_complex", fc,
            "-map", "[v]", "-map", "[a]",
            "-t", _f(vd), "-r", str(fps),
            *venc, "-pix_fmt", pix_fmt,
            "-c:a", "aac", "-ar", str(audio_rate), "-ac", "2",
            str(out_path),
        ]

    if score_stem:
        # narration with a SCORE BED: the film's own score/SFX (dialogue removed by
        # Demucs) plays under the voiceover. No sidechaincompress — dialogue is gone so
        # nothing competes with the VO; the bed just sits at a fixed subordinate level.
        # Inputs: 0=movie (video), 1=score stem (same window, bed audio), 2=voiceover.
        bed_vol = 10 ** (bed_gain_db / 20.0)
        fc = (
            vfc
            + f"[1:a]aresample=async=1,apad,atrim=duration={_f(vd)},asetpts=PTS-STARTPTS,volume={bed_vol:.4f}[bed];"
            f"[2:a]asetpts=PTS-STARTPTS[vo];"
            f"[bed][vo]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
            f"alimiter=limit=0.95,aresample={audio_rate}[a]"
        )
        return [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", _f(seg.src_in), "-t", _f(span), "-i", movie_path,
            "-ss", _f(seg.src_in), "-t", _f(span), "-i", score_stem,   # score bed, same window
            "-ss", _f(seg.vo_in), "-t", _f(vd), "-i", seg.vo_file,     # this segment's slice of the line's VO
            "-filter_complex", fc,
            "-map", "[v]", "-map", "[a]",
            "-t", _f(vd), "-r", str(fps),
            *venc, "-pix_fmt", pix_fmt,
            "-c:a", "aac", "-ar", str(audio_rate), "-ac", "2",
            str(out_path),
        ]

    # fallback narration (no score stem): duck the RAW movie audio under the VO.
    th = ducking.get("threshold", 0.03)
    ratio = ducking.get("ratio", 8)
    attack = ducking.get("attack", 5)
    release = ducking.get("release", 300)

    fc = (
        vfc
        + f"[1:a]asplit=2[scvo][mixvo];"
        f"[0:a]aresample=async=1,apad,atrim=duration={_f(vd)},asetpts=PTS-STARTPTS[orig];"
        f"[orig][scvo]sidechaincompress=threshold={th}:ratio={ratio}:attack={attack}:release={release}[duck];"
        f"[duck][mixvo]amix=inputs=2:duration=longest:dropout_transition=0,"
        f"alimiter=limit=0.95,aresample={audio_rate}[a]"
    )
    return [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", _f(seg.src_in), "-t", _f(span), "-i", movie_path,
        "-ss", _f(seg.vo_in), "-t", _f(vd), "-i", seg.vo_file,  # this segment's slice of the line's voiceover
        "-filter_complex", fc,
        "-map", "[v]", "-map", "[a]",
        "-t", _f(vd), "-r", str(fps),
        *venc, "-pix_fmt", pix_fmt,
        "-c:a", "aac", "-ar", str(audio_rate), "-ac", "2",
        str(out_path),
    ]


def concat_cmd(list_file: Path, out_path: Path, *, audio_rate: int = 48000) -> list[str]:
    # Video is stream-copied (every segment already shares the canonical profile, with subs
    # burned in) — fast and lossless. Audio is RE-ENCODED into ONE continuous stream:
    # copying independently-encoded per-segment AAC leaves each segment's encoder
    # priming/padding embedded at the joins, audible as a click/pop at every segment
    # boundary. Decoding+re-encoding across the concat yields a single continuous AAC stream
    # with no per-segment priming, removing those boundary artifacts (video untouched, so
    # the extra cost is just the cheap audio re-encode).
    return [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c:v", "copy", "-c:a", "aac", "-ar", str(audio_rate), "-ac", "2",
        str(out_path),
    ]


def _escape(path: str) -> str:
    # ffmpeg filter arg escaping for paths (colons, etc.)
    return path.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")


def write_concat_list(segment_paths: list[Path], list_file: Path) -> Path:
    lines = [f"file '{p.resolve()}'" for p in segment_paths]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_file


def render_segments(
    movie_path: str,
    edl: Edl,
    scratch_dir: Path,
    *,
    render_cfg: dict,
    ducking_cfg: dict,
    score_stem: str | None = None,
    bed_gain_db: float = -14.0,
    workers: int = 1,
    seg_ass: dict[str, Path] | None = None,
    fonts_dir: str | Path | None = None,
) -> list[Path]:
    """Normalize every EDL segment to one canonical profile.

    Each segment is an independent ffmpeg re-encode, so when ``workers`` > 1 they
    run on a thread pool (ffmpeg releases the GIL in the subprocess). Output order
    matches ``edl.segments`` regardless of completion order, so the downstream
    stream-copy concat stays correct. A failing segment re-raises and aborts.

    ``seg_ass`` maps ``segment_id`` -> a one-cue .ass to burn into that segment (subtitles
    rendered in the per-segment pass instead of a separate full-length burn pass).
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    segs = list(edl.segments)
    seg_ass = seg_ass or {}

    def _encode(seg: EdlSegment) -> Path:
        out = scratch_dir / f"{seg.segment_id}.mp4"
        cmd = normalize_segment_cmd(
            movie_path, seg, out,
            width=render_cfg["width"], height=render_cfg["height"], fps=render_cfg["fps"],
            vcodec=render_cfg["vcodec"], pix_fmt=render_cfg["pix_fmt"],
            audio_rate=render_cfg["audio_rate"], ducking=ducking_cfg,
            target_lufs=float(render_cfg.get("target_lufs", -16.0)),
            score_stem=score_stem, bed_gain_db=bed_gain_db,
            ass_path=seg_ass.get(seg.segment_id), fonts_dir=fonts_dir,
            preset=render_cfg.get("preset"), crf=render_cfg.get("crf"),
        )
        run(cmd)
        return out

    n = max(1, int(workers))
    if n > 1 and len(segs) > 1:
        log.info("rendering %d segments with %d workers", len(segs), min(n, len(segs)))
        with ThreadPoolExecutor(max_workers=n) as ex:
            # ex.map preserves input order and re-raises the first segment error.
            return list(ex.map(_encode, segs))
    return [_encode(seg) for seg in segs]
