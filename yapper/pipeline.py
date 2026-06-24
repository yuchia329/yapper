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

log = logging.getLogger("yapper.pipeline")


def _asr_via_server(
    wav: Path, cfg: Config, target: str | None = None
) -> Transcript | None:
    """Fallback ASR through the GPU-server WhisperX service (gRPC), if configured/reachable.
    ``target`` overrides ASR_GRPC_TARGET (the platform passes a gpud-leased instance).
    """
    target = target or cfg.env("ASR_GRPC_TARGET")
    if not target:
        return None
    from .server_clients.asr_client import ASRClient

    asr = cfg.section("asr")
    try:
        client = ASRClient(target)
        client.health()
        log.info("[asr] transcribing via gRPC %s", target)
        return client.transcribe(
            wav, language=asr.get("language"), diarize=asr.get("diarize", False)
        )
    except Exception as e:  # noqa: BLE001 — server optional; degrade gracefully
        log.warning("[asr] server ASR failed (%s): %s", target, e)
        return None


# ordered stage names usable with --until
FRONT_HALF = ["ingest", "audio", "asr", "shots", "scenes"]
BACK_HALF = ["understand", "script", "budget", "tts", "edl", "subs", "render"]
ALL_STAGES = FRONT_HALF + BACK_HALF

# Which compute each stage needs (drives the per-stage styling in the platform UI):
#   "llm" = MiniMax script brain · "gpu" = GPU box (WhisperX ASR / CosyVoice TTS) ·
#   "cpu" = local CPU / ffmpeg. Mirrors the Celery queue routing in yapper_web.
STAGE_COMPUTE = {
    "ingest": "cpu",
    "audio": "cpu",
    "asr": "gpu",
    "shots": "cpu",
    "scenes": "cpu",
    "understand": "llm",
    "script": "llm",
    "budget": "cpu",
    "tts": "gpu",
    "edl": "cpu",
    "subs": "cpu",
    "render": "cpu",
}

SUPPORTED_LANGS = ("zh", "en")


def resolve_lang(cfg: Config, lang: str | None = None) -> str:
    """Effective narration language: the explicit override, else the config default."""
    lang = (lang or cfg.section("narration").get("language", "zh")).lower()
    if lang not in SUPPORTED_LANGS:
        raise RuntimeError(
            f"unsupported narration language {lang!r} (expected one of {SUPPORTED_LANGS})"
        )
    return lang


def back_half_dir(cfg: Config, movie_path: str | Path, lang: str | None = None) -> Path:
    """Per-language back-half output dir: artifacts/<slug>/<lang>/."""
    return cfg.movie_dir(movie_path) / resolve_lang(cfg, lang)


def run_front_half(
    movie_path: str | Path,
    cfg: Config,
    *,
    until: str | None = None,
    force: bool = False,
    timer: RunTimer | None = None,
    asr_target: str | None = None,
) -> Path:
    """Run ingest -> clip_index. Returns the per-movie working directory.

    ``asr_target`` overrides ASR_GRPC_TARGET — the platform passes the gpud-leased ASR
    instance's host:port; ``None`` falls back to the env (CLI / always-on path).
    """
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
        log.info(
            "[ingest] %.0fs %dx%d fps=%s vfr=%s subs=%d",
            probe.duration_sec,
            probe.width,
            probe.height,
            probe.fps,
            probe.is_vfr,
            len(probe.subtitle_streams),
        )
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
                probe,
                movie_path,
                mdir,
                prefer_embedded_subs=cfg.section("asr").get(
                    "prefer_embedded_subs", True
                ),
            )
            if transcript is None:
                transcript = _asr_via_server(wav, cfg, target=asr_target)
            if transcript is not None:
                transcript.save(transcript_path)
                log.info(
                    "[asr] %d segments from %s",
                    len(transcript.segments),
                    transcript.source,
                )
            else:
                log.warning(
                    "[asr] no embedded subtitles and no reachable ASR server — continuing WITHOUT dialogue; "
                    "scenes will have empty dialogue_text. Set ASR_GRPC_TARGET to enable WhisperX."
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
                movie_path,
                shots,
                transcript,
                keyframes_dir=mdir / "keyframes",
                db_path=db_path,
                scene_group_cfg=cfg.section("scene_group"),
                keyframe_cfg=cfg.section("keyframe"),
            )
        log.info("[scenes] %d clips indexed -> %s", len(clips), db_path.name)
    else:
        timer.skipped("scenes")
        log.info("[scenes] using cached clip_index (%s)", db_path.name)
    return mdir


def run_back_half(
    movie_path: str | Path,
    cfg: Config,
    *,
    force: bool = False,
    timer: RunTimer | None = None,
    lang: str | None = None,
    llm_client=None,
    until: str | None = None,
    tts_target: str | None = None,
    tts_targets: list[str] | None = None,
) -> Path:
    """Run script generation -> TTS -> EDL -> subtitles -> render.

    Requires LLM_API_KEY (OpenAI-compatible script brain, e.g. MiniMax) and
    TTS_GRPC_TARGET (voiceover gRPC service). Assumes the front half has produced the
    clip index for this movie.

    ``llm_client`` lets a caller (the platform worker) inject a pre-built ``LLMClient``
    with cost-control hooks attached; when ``None`` the client is built here as the CLI
    does, only if a script stage actually needs to run.

    ``until`` stops after the named back-half stage (one of ``BACK_HALF``) and returns
    the per-language dir, so the platform can chain the back half as resource-routed
    tasks (llm -> gpu -> render) that resume from cached artifacts. ``None`` runs through
    render and returns the final mp4 path (CLI behaviour, unchanged).
    """

    def _stop(stage: str) -> bool:
        return (
            until is not None
            and until in BACK_HALF
            and BACK_HALF.index(stage) >= BACK_HALF.index(until)
        )

    from . import db
    from .llm.client import LLMClient
    from .schemas import Screenplay, Script, Transcript, VOManifest
    from .server_clients.tts_client import TTSClient
    from .stages import (
        s06_understand,
        s07_script,
        s08_budget,
        s09_tts,
        s10_edl,
        s11_subs,
        s12_render,
    )

    movie_path = str(Path(movie_path).resolve())
    mdir = cfg.movie_dir(movie_path)
    db_path = mdir / "clip_index.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(
            f"clip index missing ({db_path}); run the front half first"
        )
    conn = db.connect(db_path)

    narr = cfg.section("narration")
    lang = resolve_lang(cfg, lang)
    # Each language maps to a platform profile (Douyin 解說 vs YouTube recap conventions).
    platform = "douyin" if lang == "zh" else "youtube_recap"
    pcfg = cfg.raw.get("platform", {}).get(platform, {})
    target_sec = int(
        (pcfg.get("target_min_sec", 600) + pcfg.get("target_max_sec", 720)) / 2
    )
    # Spoken-rate calibration for the runtime budget: cps for CJK, wps for English words.
    cps = float(narr.get("zh_cps", 4.0))
    wps = float(narr.get("en_wps", 2.5))
    log.info(
        "[back] narration language=%s platform=%s target=%ds",
        lang,
        platform,
        target_sec,
    )
    lcfg = cfg.section("llm")

    # Shared front-half artifacts (transcript, clip index, score bed) live at the movie
    # root; the language-specific back-half outputs go in a per-language subdir, so
    # multiple narration languages coexist without overwriting each other.
    bdir = mdir / lang
    bdir.mkdir(parents=True, exist_ok=True)
    if timer is None:
        timer = RunTimer(bdir / "timings.json")
    screenplay_path = bdir / "screenplay.json"
    script_path = bdir / "script.json"
    vo_path = bdir / "vo_manifest.json"

    # Build the LLM client only if a script stage actually has to run; re-renders
    # from a cached script/voiceover then need no API key (and no tunnel for TTS).
    # A caller-supplied client (with cost-control hooks) takes precedence.
    client = llm_client
    if client is None and (
        force or not screenplay_path.exists() or not script_path.exists()
    ):
        client = LLMClient(
            api_key=cfg.require_env("LLM_API_KEY"),
            base_url=cfg.env("LLM_BASE_URL")
            or lcfg.get("base_url", "https://api.minimax.io/v1"),
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
                conn,
                client,
                target_sec=target_sec,
                thinking=lcfg.get("thinking_map", "adaptive"),
                lang=lang,
            )
            screenplay.save(screenplay_path)
    else:
        timer.skipped("understand")
        screenplay = Screenplay.load(screenplay_path)
    if _stop("understand"):
        return bdir

    # s07 script (REDUCE) --------------------------------------------------
    if force or not script_path.exists():
        with timer.stage("script"):
            script = s07_script.run_stage(
                conn,
                client,
                screenplay,
                platform=platform,
                structure=pcfg.get("structure", ""),
                target_sec=target_sec,
                thinking=lcfg.get("thinking_reduce", "disabled"),
                lang=lang,
            )
            script.save(script_path)
    else:
        timer.skipped("script")
        script = Script.load(script_path)
    if _stop("script"):
        return bdir

    # s08 budget + grounding (deterministic) -------------------------------
    with timer.stage("budget"):
        script_final = s08_budget.run_stage(
            script,
            conn,
            target_min_sec=pcfg.get("target_min_sec", 600),
            target_max_sec=pcfg.get("target_max_sec", 720),
            cps=cps,
            wps=wps,
        )
        script_final.save(bdir / "script_final.json")
    if _stop("budget"):
        return bdir

    # s09 TTS --------------------------------------------------------------
    tcfg = cfg.section("tts")
    # Per-language cloned reference voice. zh uses the top-level reference_clip/text;
    # other languages use [tts.<lang>] (e.g. [tts.en]); if unset, the server falls back
    # to its own default voice (which may carry an accent for non-native text).
    if lang == "zh":
        ref_wav = tcfg.get("reference_clip") or None
        ref_text = tcfg.get("reference_text") or None
    else:
        voice = tcfg.get(lang, {})
        ref_wav = voice.get("reference_clip") or None
        ref_text = voice.get("reference_text") or None
        if not ref_wav:
            log.warning(
                "[tts] no [tts.%s] reference voice configured — the TTS server will use its default voice; "
                "set [tts.%s].reference_clip/reference_text for a %s narrator",
                lang,
                lang,
                lang,
            )
    if force or not vo_path.exists():
        with timer.stage("tts"):
            # One client per leased instance (the platform passes gpud-leased targets); lines
            # are sharded across them in parallel. CLI/single-target -> a pool of one.
            targets = [
                t for t in (tts_targets or ([tts_target] if tts_target else [])) if t
            ]
            if not targets:
                targets = [cfg.require_env("TTS_GRPC_TARGET")]
            synths = [TTSClient(t) for t in targets]
            try:
                vo = s09_tts.run_stage(
                    script_final,
                    synths[0],
                    bdir / "vo",
                    voice_id=tcfg.get("provider", "narrator"),
                    seed=int(tcfg.get("seed", 42)),
                    target_lufs=float(tcfg.get("target_lufs", -16.0)),
                    audio_rate=int(cfg.section("render").get("audio_rate", 48000)),
                    ref_wav=ref_wav,
                    ref_text=ref_text,
                    workers=int(tcfg.get("workers", 1)),
                    synthesizers=synths,
                )
                vo.save(vo_path)
            finally:
                for (
                    s
                ) in synths:  # release the K gRPC channels (the worker is long-lived)
                    try:
                        s.close()
                    except Exception:  # noqa: BLE001
                        pass
    else:
        timer.skipped("tts")
        vo = VOManifest.load(vo_path)
        log.info("[tts] using cached voiceover (%d lines)", len(vo.lines))
    if _stop("tts"):
        return bdir

    # s10 EDL --------------------------------------------------------------
    # Match the OUTPUT aspect ratio to the SOURCE so a portrait phone video stays portrait
    # instead of being letterboxed into 1920x1080. Derive dims from the probe (orientation-aware,
    # capped to the configured box); the EDL/subtitles/render all read these. Copy the section so
    # we don't mutate the cached config. Missing probe -> the configured dims unchanged.
    rcfg = dict(cfg.section("render"))
    probe_path = mdir / "probe.json"
    if probe_path.exists():
        from .ffmpeg.probe import fit_output_dims

        pm = ProbeManifest.load(probe_path)
        long_cap, short_cap = max(rcfg["width"], rcfg["height"]), min(
            rcfg["width"], rcfg["height"]
        )
        rcfg["width"], rcfg["height"] = fit_output_dims(
            pm.display_width, pm.display_height, long_cap=long_cap, short_cap=short_cap
        )
        log.info(
            "[render] output %dx%d (source %dx%d rot=%d)",
            rcfg["width"],
            rcfg["height"],
            pm.display_width,
            pm.display_height,
            pm.rotation,
        )
    transcript_path = mdir / "transcript.json"
    transcript = Transcript.load(transcript_path) if transcript_path.exists() else None
    pbcfg = cfg.section("playback")
    with timer.stage("edl"):
        edl = s10_edl.run_stage(
            script_final,
            vo,
            conn,
            fps=rcfg["fps"],
            width=rcfg["width"],
            height=rcfg["height"],
            transcript=transcript,
            min_window=float(pbcfg.get("min_window_sec", 3.0)),
            max_window=float(pbcfg.get("max_window_sec", 12.0)),
        )
        edl.save(bdir / "edl.json")
    if _stop("edl"):
        return bdir

    # s11 subtitles --------------------------------------------------------
    scfg = cfg.section("subtitles")
    with timer.stage("subs"):
        # Global .ass written as a downloadable sidecar; the burn-in itself is now per-segment
        # in s12 (built from the same [subtitles] style), so the return value isn't needed here.
        s11_subs.build_ass(
            edl,
            bdir / "subs.ass",
            width=rcfg["width"],
            height=rcfg["height"],
            font_name=scfg.get("font_name", "Noto Sans CJK SC"),
            font_size=int(scfg.get("font_size", 48)),
            margin_v=int(scfg.get("margin_v", 60)),
        )
    if _stop("subs"):
        return bdir

    # s12 render -----------------------------------------------------------
    acfg = cfg.section("audio")
    # The Demucs score bed (dialogue removed) lives at the movie root, language-independent.
    # Accept any container demucs/ffmpeg produced (flac preferred for size+quality).
    score_stem = None
    if acfg.get("score_bed", False):
        stem = next(
            (
                mdir / f"audio_novocals.{ext}"
                for ext in ("flac", "wav", "opus")
                if (mdir / f"audio_novocals.{ext}").exists()
            ),
            None,
        )
        if stem is not None:
            score_stem = str(stem)
        else:
            log.warning(
                "[render] score_bed enabled but no audio_novocals.* at %s — run scripts/separate_score.py; "
                "falling back to ducked original audio under narration",
                mdir,
            )
    with timer.stage("render"):
        out = s12_render.run_stage(
            movie_path,
            edl,
            scratch_dir=cfg.scratch_dir / mdir.name / lang,
            out_path=bdir / "recap_final.mp4",
            render_cfg=rcfg,
            ducking_cfg=cfg.section("ducking"),
            # subtitles burned per-segment from this style (the global subs.ass above stays as a
            # sidecar artifact); fonts dir carries the bundled CJK font.
            subtitle_style={
                "font_name": scfg.get("font_name", "Noto Sans CJK SC"),
                "font_size": int(scfg.get("font_size", 48)),
                "margin_v": int(scfg.get("margin_v", 60)),
            },
            fonts_dir=cfg.repo_root / "config" / "fonts",
            score_stem=score_stem,
            bed_gain_db=float(acfg.get("bed_gain_db", -14.0)),
        )
    log.info("[done] %s", out)
    return out
