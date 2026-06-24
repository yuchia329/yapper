"""Command-line interface.

yap probe MOVIE              # inspect a file (codecs, fps, VFR, subtitle tracks)
yap run MOVIE [--until S]    # run the pipeline (front half implemented today)
yap stages                   # list stage names usable with --until
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

app = typer.Typer(
    add_completion=False, help="短片解說 — funny Mandarin commentary for movies & clips"
)
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
    t.add_row(
        "subtitle tracks",
        ", ".join(
            f"{s.codec}({'text' if s.is_text else 'image'})"
            for s in pm.subtitle_streams
        )
        or "none",
    )
    t.add_row(
        "best text sub",
        pm.best_text_subtitle.codec if pm.best_text_subtitle else "none (ASR needed)",
    )
    console.print(t)


@app.command()
def run(
    movie: str,
    until: str = typer.Option(None, help="stop after this stage (see `yap stages`)"),
    force: bool = typer.Option(False, help="re-run stages even if artifacts exist"),
    lang: str = typer.Option(
        None,
        "--lang",
        help="narration language: zh (Mandarin 解說) | en (English commentary); default from config",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the commentary pipeline on a movie or clip."""
    _setup_logging(verbose)
    if until is not None and until not in pipeline.ALL_STAGES:
        raise typer.BadParameter(
            f"unknown stage '{until}'. Valid: {', '.join(pipeline.ALL_STAGES)}"
        )
    if lang is not None and lang.lower() not in ("zh", "en"):
        raise typer.BadParameter(f"unknown language '{lang}'. Valid: zh, en")
    cfg = load_config()
    # One timer for the whole run, written into the per-language back-half dir so each
    # narration language keeps its own timings (front + back half of this run).
    timings_path = pipeline.back_half_dir(cfg, movie, lang) / "timings.json"
    timings_path.parent.mkdir(parents=True, exist_ok=True)
    timer = RunTimer(timings_path)
    mdir = pipeline.run_front_half(movie, cfg, until=until, force=force, timer=timer)
    console.print(f"[green]front half complete[/green] — artifacts in {mdir}")

    if until is not None and until in pipeline.FRONT_HALF:
        console.print(f"per-stage timings written to {timings_path}")
        return  # stopped before the back half by request

    try:
        out = pipeline.run_back_half(movie, cfg, force=force, timer=timer, lang=lang)
        console.print(f"[green]done[/green] — commentary at {out}")
    except RuntimeError as e:  # missing LLM_API_KEY / TTS_GRPC_TARGET, etc.
        console.print(f"[yellow]back half skipped[/yellow]: {e}")
        console.print(
            "set LLM_API_KEY (script) and TTS_GRPC_TARGET (voiceover gRPC), then re-run."
        )
    console.print(f"per-stage timings written to {timings_path}")


@app.command()
def submit(
    movie: str,
    remote: str = typer.Option(
        ..., "--remote", help="platform base URL, e.g. https://yapper.example.com"
    ),
    lang: str = typer.Option("zh", "--lang", help="narration language: zh | en"),
    poll: bool = typer.Option(True, help="stream run progress until done"),
) -> None:
    """Upload a movie to a running platform and generate commentary remotely.

    Convenience wrapper over the HTTP API (presigned upload -> front half -> run). The
    local `yap run` is unaffected; this just drives the same pipeline on the server.
    """
    import time
    from pathlib import Path

    import httpx

    base = remote.rstrip("/")
    path = Path(movie)
    if not path.exists():
        raise typer.BadParameter(f"file not found: {movie}")
    with httpx.Client(timeout=60, follow_redirects=True) as c:
        # cookie jar persists the minted session across calls
        reg = (
            c.post(f"{base}/api/movies", json={"filename": path.name})
            .raise_for_status()
            .json()
        )
        console.print(f"uploading [bold]{path.name}[/bold] → storage…")
        with path.open("rb") as f:
            up = c.request(
                reg["method"],
                reg["upload_url"],
                content=f.read(),
                headers={"Content-Type": "video/mp4"},
                timeout=None,
            )
            up.raise_for_status()
        c.post(f"{base}/api/movies/{reg['movie_id']}/complete").raise_for_status()
        console.print("front half queued; waiting for it to finish…")
        # wait for movie ready, then start the run
        mid = reg["movie_id"]
        while True:
            movies = c.get(f"{base}/api/movies").json()["movies"]
            m = next((x for x in movies if x["id"] == mid), None)
            if m and m["status"] == "ready":
                break
            if m and m["status"] == "error":
                console.print(f"[red]front half failed[/red]: {m.get('error')}")
                raise typer.Exit(1)
            time.sleep(5)
        run = (
            c.post(f"{base}/api/runs", json={"movie_id": mid, "lang": lang})
            .raise_for_status()
            .json()
        )
        console.print(f"run [bold]{run['id']}[/bold] ({lang}) started")
        if not poll:
            console.print(f"track at {base}/api/runs/{run['id']}")
            return
        while run["status"] not in ("done", "error"):
            time.sleep(5)
            run = c.get(f"{base}/api/runs/{run['id']}").json()
            done = [s["stage"] for s in run.get("stages", [])]
            console.print(f"  {run['status']} · stages: {', '.join(done) or '…'}")
        if run["status"] == "error":
            console.print(f"[red]run failed[/red]: {run.get('error')}")
            raise typer.Exit(1)
        res = c.get(f"{base}/api/runs/{run['id']}/result").json()
        console.print(f"[green]done[/green] — commentary: {res['url']}")


@app.command()
def stages() -> None:
    """List pipeline stage names."""
    console.print("front half (local):", ", ".join(pipeline.FRONT_HALF))
    console.print("back half (API+GPU server):", ", ".join(pipeline.BACK_HALF))


if __name__ == "__main__":
    app()
