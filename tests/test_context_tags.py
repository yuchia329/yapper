"""The s05 context tags scenes action-vs-dialogue so the LLM can expand physical
action and compress talky exposition. The signal is derived on the fly from data
already on each clip (shot cut-rate + dialogue density) — no schema/db change."""

from jieshuoforge import db
from jieshuoforge.schemas import Clip
from jieshuoforge.stages import s05_context


def _conn_with(clips):
    conn = db.connect(":memory:")
    db.write_clips(conn, clips)
    return conn


def test_action_scene_tagged_and_talky_scene_tagged():
    clips = [
        # fast cutting, no dialogue -> action
        Clip(clip_id="clip_0000", scene_index=0, t_start=0.0, t_end=10.0,
             shot_indices=list(range(15)), dialogue_text=""),
        # one long take, dense dialogue -> talky
        Clip(clip_id="clip_0001", scene_index=1, t_start=10.0, t_end=30.0,
             shot_indices=[15], dialogue_text="说" * 400),
    ]
    blocks = s05_context.build_source_blocks(_conn_with(clips), include_images=False)
    texts = [b["text"] for b in blocks]
    action_block = next(t for t in texts if t.startswith("[clip_0000]"))
    talky_block = next(t for t in texts if t.startswith("[clip_0001]"))
    assert "[动作场面]" in action_block
    assert "[对话为主]" in talky_block


def test_neutral_scene_gets_no_tag():
    clips = [
        Clip(clip_id="clip_0000", scene_index=0, t_start=0.0, t_end=20.0,
             shot_indices=[0, 1, 2], dialogue_text="说" * 20),  # moderate on both axes
    ]
    blocks = s05_context.build_source_blocks(_conn_with(clips), include_images=False)
    header = blocks[0]["text"].splitlines()[0]
    assert "[动作场面]" not in header and "[对话为主]" not in header
