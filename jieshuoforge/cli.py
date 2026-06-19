"""Command-line interface.

    jieshuo probe MOVIE              # inspect a file (codecs, fps, VFR, subtitle tracks)
    jieshuo run MOVIE [--until S]    # run the pipeline (front half implemented today)
    jieshuo stages                   # list stage names usable with --until
"""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.table import Table

from . import pipeline
from .config import load_config
from .stages import s00_ingest
from .timing import RunTimer

app = typer.Typer(add_completion=False, help="电影解说 — movie -> Mandarin narrated recap")
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(name)s: %(message)s",
    )


@app.command()
def probe(movie: str) -> None:
    """Probe a movie file and print its key properties."""
    pm = s00_ingest.run(movie)
    t = Table(title=f"probe: {movie}")
    t.add_column("field")
    t.add_column("value")
    t.add_row("duration", f"{pm.duration_sec:.1f}s")
    t.add_row("resolution", f"{pm.width}x{pm.height}")
    t.add_row("fps", str(pm.fps))
    t.add_row("VFR", str(pm.is_vfr))
    t.add_row("video codec", pm.video_codec)
    t.add_row("audio tracks", str(len(pm.audio_streams)))
    t.add_row("subtitle tracks", ", ".join(f"{s.codec}({'text' if s.is_text else 'image'})" for s in pm.subtitle_streams) or "none")
    t.add_row("best text sub", pm.best_text_subtitle.codec if pm.best_text_subtitle else "none (ASR needed)")
    console.print(t)


@app.command()
def run(
    movie: str,
    until: str = typer.Option(None, help="stop after this stage (see `jieshuo stages`)"),
    force: bool = typer.Option(False, help="re-run stages even if artifacts exist"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the recap pipeline on a movie."""
    _setup_logging(verbose)
    if until is not None and until not in pipeline.ALL_STAGES:
        raise typer.BadParameter(f"unknown stage '{until}'. Valid: {', '.join(pipeline.ALL_STAGES)}")
    cfg = load_config()
    # One timer for the whole run, so timings.json covers front + back half together.
    timings_path = cfg.movie_dir(movie) / "timings.json"
    timer = RunTimer(timings_path)
    mdir = pipeline.run_front_half(movie, cfg, until=until, force=force, timer=timer)
    console.print(f"[green]front half complete[/green] — artifacts in {mdir}")

    if until is not None and until in pipeline.FRONT_HALF:
        console.print(f"per-stage timings written to {timings_path}")
        return  # stopped before the back half by request

    try:
        out = pipeline.run_back_half(movie, cfg, force=force, timer=timer)
        console.print(f"[green]done[/green] — recap at {out}")
    except RuntimeError as e:  # missing LLM_API_KEY / TTS_SERVER_URL, etc.
        console.print(f"[yellow]back half skipped[/yellow]: {e}")
        console.print("set LLM_API_KEY (script) and TTS_SERVER_URL (voiceover), then re-run.")
    console.print(f"per-stage timings written to {timings_path}")


@app.command()
def stages() -> None:
    """List pipeline stage names."""
    console.print("front half (local):", ", ".join(pipeline.FRONT_HALF))
    console.print("back half (API+GPU server):", ", ".join(pipeline.BACK_HALF))


if __name__ == "__main__":
    app()
