"""Stage 8 — runtime budget enforcement + clip_ref resolution (deterministic).

The safety net between the LLM and TTS spend. It (1) recomputes each narration
line's spoken duration in code (never trusts the model's estimate; playback lines
carry no voiceover so they cost 0 budget), (2) validates every clip_ref against the
index — a hallucinated or empty ref fails here, before any money is spent — and
(3) trims to the target runtime if it runs long by dropping the LOWEST-importance
lines first, never the climax (importance>=4), the final CTA line, or a playback
moment. (The old prefix-trim amputated the ending; importance-trim preserves the
setup -> climax -> resolution arc.) No LLM involved.
"""

from __future__ import annotations

import logging
import sqlite3

from .. import db
from ..schemas import Script
from ..textutil import estimate_spoken_seconds

log = logging.getLogger("yapper.s08")


class GroundingError(ValueError):
    pass


def run_stage(
    script: Script,
    conn: sqlite3.Connection,
    *,
    target_min_sec: float,
    target_max_sec: float,
    cps: float = 4.0,
    wps: float = 2.0,
) -> Script:
    # 1. recompute durations from text (don't trust the model); playback lines cost 0.
    #    cps governs CJK (Mandarin); wps governs Latin words (English) — both apply to
    #    whatever the line actually contains, so mixed-language text is handled too.
    for line in script.lines:
        if line.kind == "playback":
            line.est_spoken_seconds = 0.0
            line.clip_refs = line.clip_refs[:1]  # playback plays exactly one scene
            if not line.quote:
                log.warning("%s is playback but has no quote — will use the scene's dialogue span", line.line_id)
        else:
            line.est_spoken_seconds = estimate_spoken_seconds(line.text, cps=cps, wps=wps)

    # 2. validate grounding — every line must reference at least one real clip
    for line in script.lines:
        if not line.clip_refs:
            raise GroundingError(f"{line.line_id} has no clip_refs")
        db.resolve_clip_refs(conn, line.clip_refs)  # raises ClipRefError on unknown

    # Budget is narration voiceover time only (playback windows add to runtime separately).
    narration_total = sum(l.est_spoken_seconds for l in script.lines)
    n_play = sum(1 for l in script.lines if l.kind == "playback")
    log.info(
        "script: %.1fs narration across %d lines (%d playback), target %.0f-%.0fs",
        narration_total, len(script.lines), n_play, target_min_sec, target_max_sec,
    )

    # 3. trim if over budget — drop lowest-importance droppable lines first
    if narration_total <= target_max_sec or len(script.lines) <= 1:
        if narration_total < target_min_sec:
            log.warning(
                "script is %.0fs UNDER the %.0fs floor — consider richer narration",
                narration_total, target_min_sec,
            )
        return script

    last_idx = len(script.lines) - 1
    # Protect: the final CTA line, the climax (importance>=4), and every playback moment.
    droppable = [
        i for i, l in enumerate(script.lines)
        if l.kind == "narration" and l.importance <= 3 and i != last_idx
    ]
    # remove low-importance, long-winded lines first
    droppable.sort(key=lambda i: (script.lines[i].importance, -script.lines[i].est_spoken_seconds))

    drop: set[int] = set()
    acc = narration_total
    for i in droppable:
        if acc <= target_max_sec:
            break
        drop.add(i)
        acc -= script.lines[i].est_spoken_seconds

    kept = [l for i, l in enumerate(script.lines) if i not in drop]
    if acc > target_max_sec:
        log.warning(
            "still %.0fs after dropping %d low-importance lines (only high-importance/climax left) — keeping arc intact",
            acc, len(drop),
        )
    else:
        log.warning(
            "trimmed %d low-importance lines to fit %.0fs budget (was %.0fs, now %.0fs)",
            len(drop), target_max_sec, narration_total, acc,
        )
    return Script(platform=script.platform, lines=kept)
