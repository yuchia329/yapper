"""Stage 3 — shot detection (PySceneDetect, CPU).

Splits the film into shots as second-precise boundaries. Over-segmentation is
harmless (scenes are re-grouped in s04); under-segmentation is not. Uses
AdaptiveDetector (robust to camera motion) with auto-downscale (decode is the
bottleneck, and boundaries don't need full resolution).
"""

from __future__ import annotations

from pathlib import Path

from scenedetect import AdaptiveDetector, SceneManager, open_video

from ..schemas import Shot, ShotList


def run_stage(
    movie_path: str | Path,
    *,
    adaptive_threshold: float = 1.2,
    min_shot_len_sec: float = 0.6,
    downscale: int = 0,
) -> ShotList:
    video = open_video(str(movie_path))
    fps = video.frame_rate or 24.0
    min_scene_len = max(1, int(round(min_shot_len_sec * fps)))

    sm = SceneManager()
    if downscale and downscale > 0:
        sm.downscale = downscale
    else:
        sm.auto_downscale = True
    sm.add_detector(
        AdaptiveDetector(adaptive_threshold=adaptive_threshold, min_scene_len=min_scene_len)
    )
    sm.detect_scenes(video, show_progress=False)
    scene_list = sm.get_scene_list()

    if not scene_list:
        # single continuous shot spanning the whole video
        return ShotList(shots=[Shot(index=0, start=0.0, end=float(video.duration.get_seconds()))])

    shots = [
        Shot(index=i, start=start.get_seconds(), end=end.get_seconds())
        for i, (start, end) in enumerate(scene_list)
    ]
    return ShotList(shots=shots)
