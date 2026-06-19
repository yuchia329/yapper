"""Pipeline orchestration: run stages in order, caching each artifact so re-runs
are resumable. A stage is skipped when its artifact already exists (unless
``force``). The "front half" (ingest -> clip_index) runs fully locally; the back
half (script -> TTS -> render) requires the Claude API + GPU server and is wired
in a later phase.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import Config
from .schemas import ProbeManifest, ShotList, Transcript
from .stages import s00_ingest, s01_audio, s02_asr, s03_shots, s04_scenes_keyframes
from .timing import RunTimer

log = logging.getLogger("jieshuoforge.pipeline")


def _asr_via_server(wav: Path, cfg: Config) -> Transcript | None:
    """Fallback ASR through the GPU-server WhisperX service, if configured/reachable."""
    url = cfg.env("ASR_SERVER_URL")
    if not url:
        return None
    from .server_clients.asr_client import ASRClient

    asr = cfg.section("asr")
    try:
        client = ASRClient(url)
        client.health()
        log.info("[asr] transcribing via server %s", url)
        return client.transcribe(
            wav, language=asr.get("language"), diarize=asr.get("diarize", False)
        )
    except Exception as e:  # noqa: BLE001 — server optional; degrade gracefully
        log.warning("[asr] server ASR failed (%s): %s", url, e)
        return None

# ordered stage names usable with --until
FRONT_HALF = ["ingest", "audio", "asr", "shots", "scenes"]
BACK_HALF = ["understand", "script", "budget", "tts", "edl", "subs", "render"]
ALL_STAGES = FRONT_HALF + BACK_HALF


def run_front_half(
    movie_path: str | Path,
    cfg: Config,
    *,
    until: str | None = None,
    force: bool = False,
    timer: RunTimer | None = None,
) -> Path:
    """Run ingest -> clip_index. Returns the per-movie working directory."""
    movie_path = str(Path(movie_path).resolve())
    mdir = cfg.movie_dir(movie_path)
    log.info("working dir: %s", mdir)
    if timer is None:
        timer = RunTimer(mdir / "timings.json")

    def done(stage: str) -> bool:
        return until is not None and ALL_STAGES.index(until) < ALL_STAGES.index(stage)

    # s00 ingest -----------------------------------------------------------
    probe_path = mdir / "probe.json"
    if force or not probe_path.exists():
        with timer.stage("ingest"):
            probe = s00_ingest.run(movie_path)
            probe.save(probe_path)
        log.info("[ingest] %.0fs %dx%d fps=%s vfr=%s subs=%d", probe.duration_sec, probe.width,
                 probe.height, probe.fps, probe.is_vfr, len(probe.subtitle_streams))
    else:
        timer.skipped("ingest")
        probe = ProbeManifest.load(probe_path)
    if done("audio"):
        return mdir

    # s01 audio ------------------------------------------------------------
    wav = mdir / "audio.wav"
    if force or not wav.exists():
        with timer.stage("audio"):
            s01_audio.run_stage(movie_path, wav)
        log.info("[audio] extracted %s", wav.name)
    else:
        timer.skipped("audio")
    if done("asr"):
        return mdir

    # s02 transcript (embedded subs first; else needs server ASR) ----------
    transcript_path = mdir / "transcript.json"
    if force or not transcript_path.exists():
        with timer.stage("asr"):
            transcript = s02_asr.run_stage(
                probe, movie_path, mdir,
                prefer_embedded_subs=cfg.section("asr").get("prefer_embedded_subs", True),
            )
            if transcript is None:
                transcript = _asr_via_server(wav, cfg)
            if transcript is not None:
                transcript.save(transcript_path)
                log.info("[asr] %d segments from %s", len(transcript.segments), transcript.source)
            else:
                log.warning(
                    "[asr] no embedded subtitles and no reachable ASR server — continuing WITHOUT dialogue; "
                    "scenes will have empty dialogue_text. Set ASR_SERVER_URL to enable WhisperX."
                )
    else:
        timer.skipped("asr")
    transcript = Transcript.load(transcript_path) if transcript_path.exists() else None
    if done("shots"):
        return mdir

    # s03 shots ------------------------------------------------------------
    shots_path = mdir / "shots.json"
    if force or not shots_path.exists():
        with timer.stage("shots"):
            sd = cfg.section("scene_detect")
            shots = s03_shots.run_stage(
                movie_path,
                adaptive_threshold=sd.get("adaptive_threshold", 1.2),
                min_shot_len_sec=sd.get("min_shot_len_sec", 0.6),
                downscale=sd.get("downscale", 0),
            )
            shots.save(shots_path)
        log.info("[shots] %d shots", len(shots.shots))
    else:
        timer.skipped("shots")
        shots = ShotList.load(shots_path)
    if done("scenes"):
        return mdir

    # s04 scenes + keyframes + clip index ----------------------------------
    db_path = mdir / "clip_index.sqlite"
    if force or not db_path.exists():
        with timer.stage("scenes"):
            clips = s04_scenes_keyframes.run_stage(
                movie_path, shots, transcript,
                keyframes_dir=mdir / "keyframes", db_path=db_path,
                scene_group_cfg=cfg.section("scene_group"), keyframe_cfg=cfg.section("keyframe"),
            )
        log.info("[scenes] %d clips indexed -> %s", len(clips), db_path.name)
    else:
        timer.skipped("scenes")
        log.info("[scenes] using cached clip_index (%s)", db_path.name)
    return mdir


def run_back_half(
    movie_path: str | Path, cfg: Config, *, force: bool = False, timer: RunTimer | None = None
) -> Path:
    """Run script generation -> TTS -> EDL -> subtitles -> render.

    Requires LLM_API_KEY (OpenAI-compatible script brain, e.g. MiniMax) and
    TTS_SERVER_URL (voiceover). Assumes the front half has produced the clip index
    for this movie.
    """
    from . import db
    from .llm.client import LLMClient
    from .schemas import Edl, Screenplay, Script, Transcript, VOManifest
    from .server_clients.tts_client import TTSClient
    from .stages import s06_understand, s07_script, s08_budget, s09_tts, s10_edl, s11_subs, s12_render

    movie_path = str(Path(movie_path).resolve())
    mdir = cfg.movie_dir(movie_path)
    if timer is None:
        timer = RunTimer(mdir / "timings.json")
    db_path = mdir / "clip_index.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"clip index missing ({db_path}); run the front half first")
    conn = db.connect(db_path)

    platform = "douyin"
    pcfg = cfg.raw.get("platform", {}).get(platform, {})
    target_sec = int((pcfg.get("target_min_sec", 600) + pcfg.get("target_max_sec", 720)) / 2)
    lcfg = cfg.section("llm")

    screenplay_path = mdir / "screenplay.json"
    script_path = mdir / "script.json"
    vo_path = mdir / "vo_manifest.json"

    # Build the LLM client only if a script stage actually has to run; re-renders
    # from a cached script/voiceover then need no API key (and no tunnel for TTS).
    client = None
    if force or not screenplay_path.exists() or not script_path.exists():
        client = LLMClient(
            api_key=cfg.require_env("LLM_API_KEY"),
            base_url=cfg.env("LLM_BASE_URL") or lcfg.get("base_url", "https://api.minimax.io/v1"),
            model=cfg.env("LLM_MODEL") or lcfg.get("model", "MiniMax-M3"),
            temperature=float(lcfg.get("temperature", 0.7)),
            max_output_tokens=int(lcfg.get("max_output_tokens", 32000)),
            vision=bool(lcfg.get("vision", True)),
            max_images=int(lcfg.get("max_images", 180)),
            timeout=float(lcfg.get("request_timeout_sec", 1800)),
        )

    # s06 understand (MAP) -------------------------------------------------
    if force or not screenplay_path.exists():
        with timer.stage("understand"):
            screenplay = s06_understand.run_stage(
                conn, client, target_sec=target_sec, thinking=lcfg.get("thinking_map", "adaptive")
            )
            screenplay.save(screenplay_path)
    else:
        timer.skipped("understand")
        screenplay = Screenplay.load(screenplay_path)

    # s07 script (REDUCE) --------------------------------------------------
    if force or not script_path.exists():
        with timer.stage("script"):
            script = s07_script.run_stage(
                conn, client, screenplay,
                platform=platform, structure=pcfg.get("structure", ""), target_sec=target_sec,
                thinking=lcfg.get("thinking_reduce", "disabled"),
            )
            script.save(script_path)
    else:
        timer.skipped("script")
        script = Script.load(script_path)

    # s08 budget + grounding (deterministic) -------------------------------
    with timer.stage("budget"):
        script_final = s08_budget.run_stage(
            script, conn,
            target_min_sec=pcfg.get("target_min_sec", 600),
            target_max_sec=pcfg.get("target_max_sec", 720),
        )
        script_final.save(mdir / "script_final.json")

    # s09 TTS --------------------------------------------------------------
    tcfg = cfg.section("tts")
    if force or not vo_path.exists():
        with timer.stage("tts"):
            tts = TTSClient(cfg.require_env("TTS_SERVER_URL"))
            vo = s09_tts.run_stage(
                script_final, tts, mdir / "vo",
                voice_id=tcfg.get("provider", "narrator"), seed=int(tcfg.get("seed", 42)),
                target_lufs=float(tcfg.get("target_lufs", -16.0)),
                audio_rate=int(cfg.section("render").get("audio_rate", 48000)),
                ref_wav=tcfg.get("reference_clip") or None,
                ref_text=tcfg.get("reference_text") or None,
                workers=int(tcfg.get("workers", 1)),
            )
            vo.save(vo_path)
    else:
        timer.skipped("tts")
        vo = VOManifest.load(vo_path)
        log.info("[tts] using cached voiceover (%d lines)", len(vo.lines))

    # s10 EDL --------------------------------------------------------------
    rcfg = cfg.section("render")
    transcript_path = mdir / "transcript.json"
    transcript = Transcript.load(transcript_path) if transcript_path.exists() else None
    pbcfg = cfg.section("playback")
    with timer.stage("edl"):
        edl = s10_edl.run_stage(
            script_final, vo, conn,
            fps=rcfg["fps"], width=rcfg["width"], height=rcfg["height"],
            transcript=transcript,
            min_window=float(pbcfg.get("min_window_sec", 3.0)),
            max_window=float(pbcfg.get("max_window_sec", 12.0)),
        )
        edl.save(mdir / "edl.json")

    # s11 subtitles --------------------------------------------------------
    scfg = cfg.section("subtitles")
    with timer.stage("subs"):
        ass = s11_subs.build_ass(
            edl, mdir / "subs.ass",
            width=rcfg["width"], height=rcfg["height"],
            font_name=scfg.get("font_name", "Noto Sans CJK SC"),
            font_size=int(scfg.get("font_size", 48)), margin_v=int(scfg.get("margin_v", 60)),
        )

    # s12 render -----------------------------------------------------------
    acfg = cfg.section("audio")
    stem_path = mdir / "audio_novocals.wav"
    score_stem = str(stem_path) if acfg.get("score_bed", False) and stem_path.exists() else None
    if acfg.get("score_bed", False) and score_stem is None:
        log.warning(
            "[render] score_bed enabled but %s missing — run scripts/separate_score.py; "
            "falling back to ducked original audio under narration", stem_path.name,
        )
    with timer.stage("render"):
        out = s12_render.run_stage(
            movie_path, edl,
            scratch_dir=cfg.scratch_dir / mdir.name, out_path=mdir / "recap_final.mp4",
            render_cfg=rcfg, ducking_cfg=cfg.section("ducking"),
            ass_path=ass, fonts_dir=cfg.repo_root / "config" / "fonts",
            score_stem=score_stem, bed_gain_db=float(acfg.get("bed_gain_db", -14.0)),
        )
    log.info("[done] %s", out)
    return out
