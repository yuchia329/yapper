"""Stage 4 — scene grouping + keyframe selection + clip index (the spine).

Two-tier reduction for token economics: ~1-2.5k raw shots collapse into ~40-120
narrative *scenes*. Each scene gets 1-3 representative keyframes chosen by
sharpness+brightness scoring (avoids black/blur/transition frames), downscaled to
cap Claude's image-token cost. The stable ``clip_id`` is assigned HERE — it is the
handle every later stage grounds against. Results are written to the SQLite index.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .. import db
from ..ffmpeg.run import FFMPEG, run
from ..schemas import Clip, Keyframe, ShotList, Transcript

log = logging.getLogger("yapper.s04")


def _resolve_workers(keyframe_cfg: dict) -> int:
    """Parallel scene-extraction workers. ``[keyframe].workers`` in config; 0 (the
    default) means auto. Frame seeks are light, so the auto cap is higher than the
    render cap."""
    configured = int(keyframe_cfg.get("workers", 0))
    if configured > 0:
        return configured
    return max(1, min(8, (os.cpu_count() or 4)))


# --- scene grouping --------------------------------------------------------


def group_shots_into_scenes(
    shots: ShotList, *, target_min: int, target_max: int, target_scenes: int | None = None
) -> list[tuple[float, float, list[int]]]:
    """Merge consecutive shots into ~target scenes of roughly equal duration.

    ``target_scenes`` sets the count directly (clamped to [target_min, target_max, n]);
    if unset, falls back to the old ~shots/8 heuristic. Finer scenes give the LLM more
    distinct footage to ground each narration line against.

    Returns a list of (t_start, t_end, [shot_indices]).
    """
    if not shots.shots:
        return []
    n = len(shots.shots)
    if n <= target_min:
        return [(s.start, s.end, [s.index]) for s in shots.shots]

    desired = int(target_scenes) if target_scenes else n // 8
    target_count = min(max(desired, target_min), target_max, n)
    total = shots.shots[-1].end - shots.shots[0].start
    target_dur = total / target_count if target_count else total

    scenes: list[tuple[float, float, list[int]]] = []
    cur_start = shots.shots[0].start
    cur_idx: list[int] = []
    for shot in shots.shots:
        cur_idx.append(shot.index)
        if shot.end - cur_start >= target_dur and len(scenes) < target_count - 1:
            scenes.append((cur_start, shot.end, cur_idx))
            cur_start = shot.end
            cur_idx = []
    if cur_idx:
        scenes.append((cur_start, shots.shots[-1].end, cur_idx))
    return scenes


def _dialogue_for(
    transcript: Transcript | None, t0: float, t1: float
) -> tuple[str, str | None]:
    if transcript is None:
        return "", None
    texts: list[str] = []
    speakers: list[str] = []
    for seg in transcript.segments:
        mid = (seg.start + seg.end) / 2
        if t0 <= mid < t1:
            if seg.text.strip():
                texts.append(seg.text.strip())
            if seg.speaker:
                speakers.append(seg.speaker)
    speaker = Counter(speakers).most_common(1)[0][0] if speakers else None
    return " ".join(texts), speaker


# --- keyframe scoring ------------------------------------------------------


def _extract_frame(movie_path: str, t: float, out_jpg: Path, long_edge: int, quality: int) -> bool:
    scale = (
        f"scale='if(gt(iw,ih),{long_edge},-2)':'if(gt(iw,ih),-2,{long_edge})'"
    )
    try:
        run(
            [
                FFMPEG, "-y",
                "-ss", f"{t:.3f}",
                "-i", movie_path,
                "-frames:v", "1",
                "-vf", scale,
                "-q:v", str(max(2, 31 - int(quality / 100 * 29))),
                str(out_jpg),
            ]
        )
        return out_jpg.exists()
    except Exception as e:  # noqa: BLE001 — a single bad seek shouldn't kill the run
        log.warning("frame extract failed at %.2fs: %s", t, e)
        return False


def _score(jpg: Path) -> tuple[float, float]:
    """Return (sharpness, brightness) for a frame; (-inf, 0) if unreadable."""
    img = cv2.imread(str(jpg))
    if img is None:
        return float("-inf"), 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    bright = float(lab[:, :, 0].mean())
    return sharp, bright


def _zscore(xs: list[float]) -> list[float]:
    arr = np.asarray(xs, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return [0.0] * len(xs)
    mean = arr[finite].mean()
    std = arr[finite].std()
    if std == 0:
        return [0.0 if np.isfinite(x) else -1e9 for x in xs]
    return [float((x - mean) / std) if np.isfinite(x) else -1e9 for x in xs]


def select_keyframes(
    movie_path: str,
    clip_id: str,
    t0: float,
    t1: float,
    tmp_dir: Path,
    out_dir: Path,
    *,
    candidates: int,
    keep: int,
    skip_edge_frac: float,
    sharpness_weight: float,
    brightness_weight: float,
    long_edge: int,
    quality: int,
) -> list[Keyframe]:
    span = t1 - t0
    lo = t0 + span * skip_edge_frac
    hi = t1 - span * skip_edge_frac
    if hi <= lo:
        lo, hi = t0, t1
    times = list(np.linspace(lo, hi, max(1, candidates)))

    cand_files: list[tuple[float, Path]] = []
    for j, t in enumerate(times):
        cf = tmp_dir / f"{clip_id}_cand{j}.jpg"
        if _extract_frame(movie_path, float(t), cf, long_edge, quality):
            cand_files.append((float(t), cf))
    if not cand_files:
        return []

    sharps, brights = [], []
    for _, cf in cand_files:
        s, b = _score(cf)
        sharps.append(s)
        brights.append(b)
    zs, zb = _zscore(sharps), _zscore(brights)
    scored = [
        (sharpness_weight * zs[i] + brightness_weight * zb[i], cand_files[i][0], cand_files[i][1])
        for i in range(len(cand_files))
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    keyframes: list[Keyframe] = []
    for rank, (score, t, src) in enumerate(scored[:keep]):
        dest = out_dir / f"{clip_id}_kf{rank}.jpg"
        src.replace(dest)
        keyframes.append(Keyframe(path=str(dest), t_sec=t, score=round(score, 4)))
    # clean up unused candidates
    for _, _, src in scored[keep:]:
        src.unlink(missing_ok=True)
    return keyframes


# --- stage entry -----------------------------------------------------------


def run_stage(
    movie_path: str | Path,
    shots: ShotList,
    transcript: Transcript | None,
    *,
    keyframes_dir: str | Path,
    db_path: str | Path,
    scene_group_cfg: dict,
    keyframe_cfg: dict,
) -> list[Clip]:
    movie_path = str(movie_path)
    keyframes_dir = Path(keyframes_dir)
    tmp_dir = keyframes_dir / "_cand"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    scenes = group_shots_into_scenes(
        shots,
        target_min=int(scene_group_cfg.get("target_scenes_min", 40)),
        target_max=int(scene_group_cfg.get("target_scenes_max", 360)),
        target_scenes=scene_group_cfg.get("target_scenes"),
    )
    log.info("grouped %d shots into %d scenes", len(shots.shots), len(scenes))

    def _build_clip(item: tuple[int, tuple[float, float, list[int]]]) -> Clip:
        scene_index, (t0, t1, shot_idx) = item
        clip_id = f"clip_{scene_index:04d}"
        dialogue, speaker = _dialogue_for(transcript, t0, t1)
        keyframes = select_keyframes(
            movie_path,
            clip_id,
            t0,
            t1,
            tmp_dir,
            keyframes_dir,
            candidates=int(keyframe_cfg.get("candidates", 5)),
            keep=int(scene_group_cfg.get("keyframes_per_scene", 2)),
            skip_edge_frac=float(keyframe_cfg.get("skip_edge_frac", 0.10)),
            sharpness_weight=float(keyframe_cfg.get("sharpness_weight", 0.7)),
            brightness_weight=float(keyframe_cfg.get("brightness_weight", 0.3)),
            long_edge=int(keyframe_cfg.get("downscale_long_edge_px", 1024)),
            quality=int(keyframe_cfg.get("jpeg_quality", 85)),
        )
        return Clip(
            clip_id=clip_id,
            scene_index=scene_index,
            t_start=t0,
            t_end=t1,
            shot_indices=shot_idx,
            speaker=speaker,
            dialogue_text=dialogue,
            keyframes=keyframes,
        )

    # Each scene's keyframes are extracted/scored independently and written to
    # clip_id-prefixed files (no cross-scene collisions), so scenes parallelize
    # across a thread pool (ffmpeg seeks + cv2 release the GIL). ex.map preserves
    # scene order, which is the clip index ordering.
    workers = _resolve_workers(keyframe_cfg)
    items = list(enumerate(scenes))
    if workers > 1 and len(items) > 1:
        log.info("extracting keyframes with %d workers", min(workers, len(items)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            clips: list[Clip] = list(ex.map(_build_clip, items))
    else:
        clips = [_build_clip(it) for it in items]

    tmp_dir.rmdir() if tmp_dir.exists() and not any(tmp_dir.iterdir()) else None
    conn = db.connect(db_path)
    db.write_clips(conn, clips)
    conn.close()
    log.info("wrote %d clips to %s", len(clips), db_path)
    return clips
