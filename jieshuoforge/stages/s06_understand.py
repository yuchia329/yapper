"""Stage 6 — plot understanding (Claude MAP pass).

Claude reads the whole indexed film (transcript + keyframes) in one context and
returns a beat sheet: the 主线 throughline with subplots dropped, characters
anchored to spoken names, each beat carrying clip_refs + an estimated narration
length. Refusals (graphic content) propagate as RefusalError for the caller to
isolate/retry/flag.
"""

from __future__ import annotations

import logging
import sqlite3

from ..llm import prompts
from ..llm.client import LLMClient
from ..schemas import Screenplay, screenplay_json_schema
from . import s05_context

log = logging.getLogger("jieshuoforge.s06")


def run_stage(
    conn: sqlite3.Connection, client: LLMClient, *, target_sec: int, thinking: str | None = "adaptive"
) -> Screenplay:
    source_blocks = s05_context.build_source_blocks(conn, include_images=client.vision, max_images=client.max_images)
    data, _usage = client.complete_structured(
        system_text=prompts.build_system_text(),
        source_blocks=source_blocks,
        instruction=prompts.map_instruction(target_sec),
        schema=screenplay_json_schema(),
        thinking=thinking,
    )
    screenplay = Screenplay.model_validate(data)
    log.info("[understand] %d beats, %d characters", len(screenplay.beats), len(screenplay.characters))
    return screenplay
