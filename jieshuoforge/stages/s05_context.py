"""Stage 5 — indexed-context assembly.

Turn the clip index into the prompt prefix the model reads: for each scene, a text
block tagged with its clip_id (+ time range, speaker, dialogue), followed by its
keyframe images (OpenAI-style image_url blocks). The clip_id labels are how the
model grounds its output. The MAP and REDUCE passes reuse this same prefix.
"""

from __future__ import annotations

import sqlite3

from .. import db
from ..llm.client import image_block, text_block


def _fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _action_tag(clip) -> str:
    """A cheap action-vs-dialogue signal from data already on the clip: action scenes
    cut fast and talk little; talky scenes are the inverse. Lets the model expand
    physical action and compress conversation-heavy exposition."""
    dur = clip.duration
    if dur <= 0:
        return ""
    cuts_per_sec = len(clip.shot_indices) / dur
    dialogue_cps = len(clip.dialogue_text) / dur
    score = 5.0 * cuts_per_sec - 0.3 * dialogue_cps
    if score > 2.0:
        return " [动作场面]"
    if score < -2.0:
        return " [对话为主]"
    return ""


def build_source_blocks(
    conn: sqlite3.Connection, *, include_images: bool = True, max_images: int = 180
) -> list[dict]:
    """Assemble the prompt prefix. Every scene contributes a clip_id-tagged text
    block (dialogue is the ground truth). Keyframe images are capped at
    ``max_images`` to stay under provider limits (MiniMax allows ≤200 images/request):
    one best keyframe per scene, and if there are more scenes than the budget, an
    evenly-spaced subset across the timeline. Text is never dropped.
    """
    clips = db.clips_in_order(conn)

    selected: set[str] = set()
    if include_images and max_images > 0 and clips:
        n = len(clips)
        if n <= max_images:
            selected = {c.clip_id for c in clips}
        else:  # evenly spaced subset across the film
            step = n / max_images
            selected = {clips[min(n - 1, int(i * step))].clip_id for i in range(max_images)}

    blocks: list[dict] = []
    for clip in clips:
        header = f"[{clip.clip_id}] {_fmt(clip.t_start)}–{_fmt(clip.t_end)}"
        if clip.speaker:
            header += f" · {clip.speaker}"
        header += _action_tag(clip)
        body = clip.dialogue_text.strip() or "（无对白）"
        blocks.append(text_block(f"{header}\n{body}"))
        if clip.clip_id in selected and clip.keyframes:
            blocks.append(image_block(clip.keyframes[0].path))  # single best keyframe
    return blocks
