"""s05 builds the Claude prompt prefix: clip_id-tagged text + keyframe images,
with exactly one cache breakpoint on the last block."""

from pathlib import Path

from PIL import Image

from jieshuoforge import db
from jieshuoforge.schemas import Clip, Keyframe
from jieshuoforge.stages import s05_context


def _tiny_jpg(path: Path) -> str:
    Image.new("RGB", (8, 8), (123, 50, 200)).save(path, "JPEG")
    return str(path)


def test_source_blocks_tag_clip_ids_and_interleave_images(tmp_path: Path):
    conn = db.connect(":memory:")
    kf = _tiny_jpg(tmp_path / "clip_0000_kf0.jpg")
    db.write_clips(
        conn,
        [
            Clip(clip_id="clip_0000", scene_index=0, t_start=0.0, t_end=5.0, shot_indices=[0],
                 speaker="SPEAKER_00", dialogue_text="A man walks in.",
                 keyframes=[Keyframe(path=kf, t_sec=2.5, score=1.0)]),
            Clip(clip_id="clip_0001", scene_index=1, t_start=5.0, t_end=9.0, shot_indices=[1],
                 dialogue_text=""),
        ],
    )
    blocks = s05_context.build_source_blocks(conn)

    text_blocks = [b for b in blocks if b["type"] == "text"]
    image_blocks = [b for b in blocks if b["type"] == "image_url"]
    assert any("clip_0000" in b["text"] for b in text_blocks)
    assert any("clip_0001" in b["text"] for b in text_blocks)
    assert any("（无对白）" in b["text"] for b in text_blocks)  # empty dialogue placeholder
    assert len(image_blocks) == 1
    assert image_blocks[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
