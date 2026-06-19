"""Stage 12 — assemble & render.

Normalize each EDL segment to the canonical profile, concat them, then (if an ASS
subtitle file is supplied and the ffmpeg build supports it) burn in subtitles.
Subtitle burn-in needs an ffmpeg compiled with libass; if absent we render without
subs and warn rather than fail, so the MVP still produces a watchable video.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from ..ffmpeg import graph
from ..ffmpeg.run import has_filter, run
from ..schemas import Edl

log = logging.getLogger("jieshuoforge.s12")


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
    ass_path: str | Path | None = None,
    fonts_dir: str | Path | None = None,
    score_stem: str | None = None,
    bed_gain_db: float = -14.0,
) -> Path:
    movie_path = str(movie_path)
    scratch_dir = Path(scratch_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workers = _resolve_workers(render_cfg)
    log.info(
        "rendering %d segments to %s%s (workers=%d)",
        len(edl.segments), scratch_dir, " (score bed)" if score_stem else "", workers,
    )
    seg_paths = graph.render_segments(
        movie_path, edl, scratch_dir, render_cfg=render_cfg, ducking_cfg=ducking_cfg,
        score_stem=score_stem, bed_gain_db=bed_gain_db, workers=workers,
    )

    list_file = graph.write_concat_list(seg_paths, scratch_dir / "concat.txt")
    concat_out = scratch_dir / "concat.mp4"
    run(graph.concat_cmd(list_file, concat_out))

    if ass_path is not None and has_filter("subtitles"):
        log.info("burning subtitles from %s", ass_path)
        run(
            graph.burn_subs_cmd(
                concat_out, Path(ass_path), out_path,
                fonts_dir=Path(fonts_dir) if fonts_dir else None,
                vcodec=render_cfg["vcodec"], pix_fmt=render_cfg["pix_fmt"],
            )
        )
    else:
        if ass_path is not None:
            log.warning("ffmpeg has no 'subtitles' filter (libass missing) — rendering WITHOUT subtitles")
        shutil.move(str(concat_out), str(out_path))

    log.info("wrote %s", out_path)
    return out_path
