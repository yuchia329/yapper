"""Runtime budget is enforced in code (structured outputs can't constrain length)."""

from jieshuoforge import db
from jieshuoforge.schemas import Clip, Script, ScriptLine
from jieshuoforge.stages import s08_budget
from jieshuoforge.textutil import estimate_spoken_seconds


def _conn():
    conn = db.connect(":memory:")
    db.write_clips(conn, [Clip(clip_id="clip_0000", scene_index=0, t_start=0.0, t_end=600.0, shot_indices=[0])])
    return conn


def _script(n: int) -> Script:
    return Script(
        lines=[
            ScriptLine(line_id=f"l{i:03d}", text="这是一句中文旁白用来测试时长预算的功能" * 3, clip_refs=["clip_0000"], est_spoken_seconds=0)
            for i in range(n)
        ]
    )


def test_estimate_grows_with_length():
    short = estimate_spoken_seconds("你好世界")
    long = estimate_spoken_seconds("你好世界" * 20)
    assert long > short > 0


def test_estimate_handles_mixed_cjk_and_latin():
    assert estimate_spoken_seconds("主角John在2024年回到了纽约") > 0


def test_budget_recomputes_durations():
    conn = _conn()
    s = Script(lines=[ScriptLine(line_id="l0", text="你好世界你好世界", clip_refs=["clip_0000"], est_spoken_seconds=999)])
    out = s08_budget.run_stage(s, conn, target_min_sec=0, target_max_sec=999)
    assert out.lines[0].est_spoken_seconds < 10  # 999 was overwritten by the real estimate


def test_budget_trims_when_over_and_keeps_last_line():
    conn = _conn()
    s = _script(40)
    last_id = s.lines[-1].line_id
    out = s08_budget.run_stage(s, conn, target_min_sec=10, target_max_sec=30)
    assert out.total_est_seconds <= 30
    assert out.lines[-1].line_id == last_id  # CTA/ending always retained
    assert len(out.lines) < 40


def test_budget_keeps_script_when_under():
    conn = _conn()
    s = _script(3)
    out = s08_budget.run_stage(s, conn, target_min_sec=0, target_max_sec=999)
    assert len(out.lines) == 3


def test_budget_importance_trim_protects_climax_cta_and_playback():
    conn = _conn()
    filler = [
        ScriptLine(line_id=f"l{i:03d}", text="这是一句中文旁白用来测试时长预算的功能" * 3,
                   clip_refs=["clip_0000"], importance=2)
        for i in range(20)
    ]
    climax = ScriptLine(line_id="climax", text="高潮场面" * 20, clip_refs=["clip_0000"], importance=5)
    play = ScriptLine(line_id="play", kind="playback", text="原声字幕",
                      clip_refs=["clip_0000"], quote="some line", importance=3)
    cta = ScriptLine(line_id="cta", text="点赞加关注" * 3, clip_refs=["clip_0000"], importance=3)
    s = Script(lines=[*filler, climax, play, cta])

    out = s08_budget.run_stage(s, conn, target_min_sec=10, target_max_sec=40)
    ids = [l.line_id for l in out.lines]

    assert "climax" in ids            # importance>=4 never dropped
    assert "cta" in ids               # final CTA line never dropped
    assert "play" in ids              # playback moment never dropped (and costs 0 budget)
    assert ids[-1] == "cta"           # ending stays last (no positional amputation)
    assert len(out.lines) < 23        # most low-importance filler dropped
    # playback carries no voiceover cost
    assert next(l for l in out.lines if l.line_id == "play").est_spoken_seconds == 0.0
    assert out.total_est_seconds <= 40
