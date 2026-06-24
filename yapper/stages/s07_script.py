"""Stage 7 — 解說 narration (Claude REDUCE pass).

Reuses the cached indexed-source prefix from the MAP pass (back-to-back, within
the cache TTL) and, grounded in the beat sheet, writes the Mandarin narration in
the target platform's structure. The strict schema forces a clip_refs list on
every line; the runtime budget (s08) enforces length deterministically afterward.
"""

from __future__ import annotations

import logging
import sqlite3

from ..llm import prompts
from ..llm.client import LLMClient
from ..schemas import Screenplay, Script, script_json_schema
from . import s05_context

log = logging.getLogger("yapper.s07")


def run_stage(
    conn: sqlite3.Connection,
    client: LLMClient,
    screenplay: Screenplay,
    *,
    platform: str,
    structure: str,
    target_sec: int,
    thinking: str | None = "disabled",
    lang: str = "zh",
) -> Script:
    # Text-only source for REDUCE: the MAP pass already saw the keyframes and encoded
    # what it learned in the beat sheet, and clip_refs are grounded via the beat sheet +
    # per-clip dialogue text. Re-sending ~180 images here just bloats the prompt and makes
    # the (reasoning-model) call time out, so we drop them.
    source_blocks = s05_context.build_source_blocks(conn, include_images=False)
    beat_sheet = (
        prompts.beat_sheet_header(lang) + "\n" + screenplay.model_dump_json(indent=2)
    )
    instruction = (
        beat_sheet
        + "\n\n"
        + prompts.reduce_instruction(platform, structure, target_sec, lang)
    )

    data, _usage = client.complete_structured(
        system_text=prompts.build_system_text(lang),
        source_blocks=source_blocks,
        instruction=instruction,
        schema=script_json_schema(),
        thinking=thinking,
    )
    script = Script.model_validate(data)
    script.platform = platform
    log.info(
        "[script] %d lines, ~%.0fs narration",
        len(script.lines),
        script.total_est_seconds,
    )
    return script
