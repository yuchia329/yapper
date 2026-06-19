"""The grounding contract: the LLM references clip_ids; unknown/empty refs are
rejected before they can reach the editor."""

import pytest

from jieshuoforge import db
from jieshuoforge.schemas import Clip, Script, ScriptLine
from jieshuoforge.stages import s08_budget


def make_conn():
    conn = db.connect(":memory:")
    db.write_clips(
        conn,
        [
            Clip(clip_id="clip_0000", scene_index=0, t_start=0.0, t_end=5.0, shot_indices=[0]),
            Clip(clip_id="clip_0001", scene_index=1, t_start=5.0, t_end=10.0, shot_indices=[1]),
        ],
    )
    return conn


def test_resolve_known_preserves_order():
    conn = make_conn()
    clips = db.resolve_clip_refs(conn, ["clip_0001", "clip_0000"])
    assert [c.clip_id for c in clips] == ["clip_0001", "clip_0000"]


def test_resolve_unknown_raises():
    conn = make_conn()
    with pytest.raises(db.ClipRefError):
        db.resolve_clip_refs(conn, ["clip_0000", "clip_9999"])


def test_budget_stage_rejects_hallucinated_ref():
    conn = make_conn()
    script = Script(lines=[ScriptLine(line_id="l0", text="测试", clip_refs=["clip_4242"], est_spoken_seconds=0)])
    with pytest.raises(db.ClipRefError):
        s08_budget.run_stage(script, conn, target_min_sec=1, target_max_sec=999)


def test_budget_stage_rejects_empty_refs():
    conn = make_conn()
    script = Script(lines=[ScriptLine(line_id="l0", text="测试", clip_refs=[], est_spoken_seconds=0)])
    with pytest.raises(s08_budget.GroundingError):
        s08_budget.run_stage(script, conn, target_min_sec=1, target_max_sec=999)
