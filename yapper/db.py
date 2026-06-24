"""SQLite ``clip_ref`` index — the single source of truth for timecodes.

The pipeline's load-bearing contract: the LLM emits ``clip_id`` references, never
raw timestamps. This module owns the mapping ``clip_id -> {t_start, t_end, ...}``
and is the ONLY place real timecodes are resolved. ``resolve_clip_refs`` rejects
unknown ids, which is what stops a hallucinated reference from reaching the editor.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .schemas import Clip, Keyframe


class ClipRefError(KeyError):
    """Raised when a clip_ref does not exist in the index (grounding violation)."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    clip_id       TEXT PRIMARY KEY,
    scene_index   INTEGER NOT NULL,
    t_start       REAL NOT NULL,
    t_end         REAL NOT NULL,
    shot_indices  TEXT NOT NULL,   -- json list[int]
    speaker       TEXT,
    dialogue_text TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS keyframes (
    clip_id  TEXT NOT NULL REFERENCES clips(clip_id),
    path     TEXT NOT NULL,
    t_sec    REAL NOT NULL,
    score    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_keyframes_clip ON keyframes(clip_id);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def write_clips(conn: sqlite3.Connection, clips: list[Clip]) -> None:
    """Replace the index contents with ``clips`` (idempotent re-runs)."""
    with conn:
        conn.execute("DELETE FROM keyframes")
        conn.execute("DELETE FROM clips")
        for c in clips:
            conn.execute(
                "INSERT INTO clips (clip_id, scene_index, t_start, t_end, shot_indices, speaker, dialogue_text)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    c.clip_id,
                    c.scene_index,
                    c.t_start,
                    c.t_end,
                    json.dumps(c.shot_indices),
                    c.speaker,
                    c.dialogue_text,
                ),
            )
            for kf in c.keyframes:
                conn.execute(
                    "INSERT INTO keyframes (clip_id, path, t_sec, score) VALUES (?, ?, ?, ?)",
                    (c.clip_id, kf.path, kf.t_sec, kf.score),
                )


def _row_to_clip(conn: sqlite3.Connection, row: sqlite3.Row) -> Clip:
    kf_rows = conn.execute(
        "SELECT path, t_sec, score FROM keyframes WHERE clip_id = ? ORDER BY t_sec",
        (row["clip_id"],),
    ).fetchall()
    return Clip(
        clip_id=row["clip_id"],
        scene_index=row["scene_index"],
        t_start=row["t_start"],
        t_end=row["t_end"],
        shot_indices=json.loads(row["shot_indices"]),
        speaker=row["speaker"],
        dialogue_text=row["dialogue_text"],
        keyframes=[Keyframe(path=k["path"], t_sec=k["t_sec"], score=k["score"]) for k in kf_rows],
    )


def get_clip(conn: sqlite3.Connection, clip_id: str) -> Clip:
    row = conn.execute("SELECT * FROM clips WHERE clip_id = ?", (clip_id,)).fetchone()
    if row is None:
        raise ClipRefError(clip_id)
    return _row_to_clip(conn, row)


def all_clip_ids(conn: sqlite3.Connection) -> set[str]:
    return {r["clip_id"] for r in conn.execute("SELECT clip_id FROM clips")}


def clips_in_order(conn: sqlite3.Connection) -> list[Clip]:
    rows = conn.execute("SELECT * FROM clips ORDER BY scene_index").fetchall()
    return [_row_to_clip(conn, r) for r in rows]


def resolve_clip_refs(conn: sqlite3.Connection, refs: list[str]) -> list[Clip]:
    """Resolve clip_ids to Clips, preserving order. Raises ClipRefError on any miss.

    This is the choke point that enforces grounding: nothing downstream gets a
    timecode that didn't originate from the index.
    """
    known = all_clip_ids(conn)
    missing = [r for r in refs if r not in known]
    if missing:
        raise ClipRefError(f"unknown clip_refs: {missing}")
    return [get_clip(conn, r) for r in refs]
