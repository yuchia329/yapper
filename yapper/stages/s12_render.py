"""Stage 12 — assemble & render.

Normalize each EDL segment to the canonical profile — burning that segment's one
subtitle cue in DURING this same pass — then stream-copy concat to the final video.
Folding subtitles into the per-segment encode avoids a second full-length re-encode
(roughly halving render CPU). Burn-in needs an ffmpeg compiled with libass; if absent
we render without subs and warn rather than fail, so we still produce a watchable video.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ..ffmpeg import graph
from ..ffmpeg.run import has_filter, run
from ..schemas import Edl
from . import s11_subs

log = logging.getLogger("yapper.s12")


def _resolve_workers(render_cfg: dict) -> int:
    """How many segments to encode in parallel. ``[render].workers`` in config;
    0 (the default) means auto. libx264 is already multi-threaded, so we cap the
    auto value low to avoid oversubscribing cores (each ffmpeg spawns its own
    threads); benchmark and raise it explicitly if your box has cores to spare."""
    configured = int(render_cfg.get("workers", 0))
    if configured > 0:
        return configured
    return max(1, min(4, (os.cpu_count() or 4)))


def run_stage(
    movie_path: str | Path,
    edl: Edl,
    *,
    scratch_dir: str | Path,
    out_path: str | Path,
    render_cfg: dict,
    ducking_cfg: dict,
    subtitle_style: dict | None = None,
    fonts_dir: str | Path | None = None,
    score_stem: str | None = None,
    bed_gain_db: float = -14.0,
) -> Path:
    movie_path = str(movie_path)
    scratch_dir = Path(scratch_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build one .ass per segment (text spanning its local [0, screen_duration]) to burn in
    # during the per-segment encode — no separate full-length burn pass. ``subtitle_style``
    # is the [subtitles] style dict (font_name/font_size/margin_v); None = no subtitles.
    seg_ass: dict[str, Path] = {}
    want_subs = subtitle_style is not None
    if want_subs and not has_filter("subtitles"):
        log.warning("ffmpeg has no 'subtitles' filter (libass missing) — rendering WITHOUT subtitles")
        want_subs = False
    if want_subs:
        subs_dir = scratch_dir / "subs"
        for seg in edl.segments:
            p = s11_subs.build_segment_ass(
                seg.subtitle_text, seg.screen_duration, subs_dir / f"{seg.segment_id}.ass",
                width=render_cfg["width"], height=render_cfg["height"], **subtitle_style,
            )
            if p is not None:
                seg_ass[seg.segment_id] = p

    workers = _resolve_workers(render_cfg)
    log.info(
        "rendering %d segments to %s%s (workers=%d, subs=%s)",
        len(edl.segments), scratch_dir, " (score bed)" if score_stem else "", workers, bool(seg_ass),
    )
    seg_paths = graph.render_segments(
        movie_path, edl, scratch_dir, render_cfg=render_cfg, ducking_cfg=ducking_cfg,
        score_stem=score_stem, bed_gain_db=bed_gain_db, workers=workers,
        seg_ass=seg_ass, fonts_dir=Path(fonts_dir) if fonts_dir else None,
    )

    # Subtitles are already burned into each segment, so the concat IS the final. Video is
    # stream-copied; audio is re-encoded continuously to avoid per-segment AAC priming clicks
    # at the boundaries.
    list_file = graph.write_concat_list(seg_paths, scratch_dir / "concat.txt")
    run(graph.concat_cmd(list_file, out_path, audio_rate=int(render_cfg["audio_rate"])))

    log.info("wrote %s", out_path)
    return out_path
